defmodule Mix.Tasks.BenchmarkOuroPipeline do
  @moduledoc """
  Benchmarks each stage of the Ouro-backed response pipeline in isolation,
  then end-to-end, collecting empirical latency for timeout characterization.

  ## Stages

    1. Raw Ouro.Model.generate — cold vs warm, varied max_new_tokens
    2. Production-shaped RealizationPacket → Ouro.Model.generate
    3. RefinementLoop.single_pass on a frozen InternalModel
    4. Full RefinementLoop.generate (up to 3 iterations)
    5. End-to-end Brain.evaluate (with :infinity timeout)
    6. Queued Brain.evaluate (two back-to-back calls)

  ## Usage

      mix benchmark_ouro_pipeline              # Run all stages
      mix benchmark_ouro_pipeline --stage 1    # Run only stage 1
      mix benchmark_ouro_pipeline --runs 5     # Repeat each measurement N times (default 3)
  """

  use Mix.Task
  require Logger

  @shortdoc "Benchmark Ouro pipeline stages for latency characterization"

  @default_runs 3

  @test_inputs [
    "Hello! I'm Austin. It is nice to finally meet you my friend.",
    "What have you been up to?",
    "Tell me about the weather in New York City."
  ]

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [stage: :integer, runs: :integer]
      )

    stage_filter = opts[:stage]
    runs = opts[:runs] || @default_runs

    Application.put_env(:brain, :skip_ml_init, false)
    Mix.Task.run("app.start")

    info("")
    info("=" |> String.duplicate(72))
    info("  Ouro Pipeline Latency Benchmark")
    info("=" |> String.duplicate(72))
    info("  Runs per measurement: #{runs}")
    info("  Test inputs: #{length(@test_inputs)}")
    info("")

    wait_for_ouro_ready(300)

    results = %{}

    results =
      if stage_filter == nil or stage_filter == 0 do
        run_stage_0_diagnostics(results)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 1 do
        run_stage_1(results, runs)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 2 do
        run_stage_2(results, runs)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 3 do
        run_stage_3(results, runs)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 4 do
        run_stage_4(results, runs)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 5 do
        run_stage_5(results, runs)
      else
        results
      end

    results =
      if stage_filter == nil or stage_filter == 6 do
        run_stage_6(results)
      else
        results
      end

    info("")
    info("=" |> String.duplicate(72))
    info("  Summary")
    info("=" |> String.duplicate(72))
    print_summary(results)
  end

  # ---------------------------------------------------------------------------
  # Stage 0: Prompt diagnostics — what does Ouro actually see?
  # ---------------------------------------------------------------------------
  defp run_stage_0_diagnostics(results) do
    header("Stage 0: Prompt Diagnostics")

    analysis = build_test_chunk_analysis()
    primitives = build_test_primitives()
    messages = Brain.Response.RealizationPacket.build(primitives, analysis, [])

    info("  --- Chat template as sent to format_chatml ---")
    Enum.each(messages, fn %{role: role, content: content} ->
      content_preview = String.slice(content, 0, 300)
      info("  [#{role}] (#{String.length(content)} chars):")
      info("    #{content_preview}")
      if String.length(content) > 300, do: info("    ... (truncated)")
    end)

    chatml_prompt = format_chatml_diagnostic(messages)
    prompt_bytes = byte_size(chatml_prompt)
    info("")
    info("  --- Full ChatML prompt (#{prompt_bytes} bytes) ---")
    info("  #{String.slice(chatml_prompt, 0, 600)}")
    if prompt_bytes > 600, do: info("  ... (truncated, #{prompt_bytes} total bytes)")

    ml_config = Application.get_env(:brain, :ml, [])
    seq_len = ml_config[:ouro_sequence_length] || 4096
    max_tokens = ml_config[:ouro_max_new_tokens] || 256

    info("")
    info("  --- Compile constraints ---")
    info("  sequence_length: #{seq_len} tokens (prompt + generated output must fit)")
    info("  max_new_tokens: #{max_tokens} (baked into Nx.Serving at load time)")

    info("")
    info("  --- EOS token analysis ---")
    info("  tokenizer_config.json eos_token: <|endoftext|>")
    info("  Ouro.Tokenizer fallback eos: <|im_end|>")
    info("  format_chatml turn delimiter: <|im_end|>")
    info("  Ouro.Spec default eos_token_id: 2")
    info("  NOTE: If Bumblebee's generation looks for <|endoftext|> as EOS")
    info("        but the model emits <|im_end|> to end a turn, generation")
    info("        may overshoot until hitting sequence_length or max_new_tokens.")

    info("")
    info("  --- sequence_length budget ---")
    estimated_prompt_tokens = div(prompt_bytes, 4)
    remaining = seq_len - estimated_prompt_tokens
    info("  Estimated prompt tokens (bytes/4 heuristic): ~#{estimated_prompt_tokens}")
    info("  Remaining token budget in #{seq_len}-token window: ~#{remaining}")
    if remaining < 50 do
      warn("  !! Very little room for generation — prompt may consume nearly all of the #{seq_len}-token window")
    end

    info("")

    simple_messages = [
      %{role: "system", content: "You are a helpful assistant. Respond concisely."},
      %{role: "user", content: "Hello, how are you today?"}
    ]

    simple_prompt = format_chatml_diagnostic(simple_messages)
    info("  --- Comparison: simple prompt (#{byte_size(simple_prompt)} bytes) ---")
    info("  #{simple_prompt}")
    info("")

    Map.put(results, :stage_0, %{
      production_prompt_bytes: prompt_bytes,
      simple_prompt_bytes: byte_size(simple_prompt),
      estimated_prompt_tokens: estimated_prompt_tokens
    })
  end

  defp format_chatml_diagnostic(messages) do
    body =
      Enum.map_join(messages, fn %{role: role, content: content} ->
        "<|im_start|>#{role}\n#{content}<|im_end|>\n"
      end)

    body <> "<|im_start|>assistant\n"
  end

  # ---------------------------------------------------------------------------
  # Stage 1: Raw Ouro.Model.generate
  # ---------------------------------------------------------------------------
  defp run_stage_1(results, runs) do
    header("Stage 1: Raw Ouro.Model.generate (isolation)")

    simple_messages = [
      %{role: "system", content: "You are a helpful assistant. Respond concisely."},
      %{role: "user", content: "Hello, how are you today?"}
    ]

    info("  1a. Default opts (no per-call overrides)")
    default_timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} = timed(fn -> Brain.ML.Ouro.Model.generate(simple_messages) end)
        status = format_gen_result(result)
        info("      Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("  1b. With max_new_tokens: 50 (testing if opts are applied)")
    short_timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} =
          timed(fn -> Brain.ML.Ouro.Model.generate(simple_messages, max_new_tokens: 50) end)

        status = format_gen_result(result)
        info("      Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("  1c. With max_new_tokens: 200")
    long_timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} =
          timed(fn -> Brain.ML.Ouro.Model.generate(simple_messages, max_new_tokens: 200) end)

        status = format_gen_result(result)
        info("      Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("")
    info("  Summary Stage 1:")
    info("    Default opts:        #{format_stats(default_timings)}")
    info("    max_new_tokens=50:   #{format_stats(short_timings)}")
    info("    max_new_tokens=200:  #{format_stats(long_timings)}")
    opts_applied = stats(short_timings).median != stats(default_timings).median
    info("    Opts appear to affect latency: #{opts_applied}")
    info("")

    Map.put(results, :stage_1, %{
      default: stats(default_timings),
      short: stats(short_timings),
      long: stats(long_timings)
    })
  end

  # ---------------------------------------------------------------------------
  # Stage 2: Production-shaped RealizationPacket → Ouro.Model.generate
  # ---------------------------------------------------------------------------
  defp run_stage_2(results, runs) do
    header("Stage 2: Production-shaped RealizationPacket → Ouro.Model.generate")

    analysis = build_test_chunk_analysis()
    primitives = build_test_primitives()

    messages =
      Brain.Response.RealizationPacket.build(primitives, analysis, [])

    info("  Packet size: #{Jason.encode!(messages) |> byte_size()} bytes")
    info("")

    timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} = timed(fn -> Brain.ML.Ouro.Model.generate(messages) end)
        status = format_gen_result(result)
        info("    Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("")
    info("  Summary Stage 2: #{format_stats(timings)}")
    info("")

    Map.put(results, :stage_2, stats(timings))
  end

  # ---------------------------------------------------------------------------
  # Stage 3: RefinementLoop.single_pass
  # ---------------------------------------------------------------------------
  defp run_stage_3(results, runs) do
    header("Stage 3: RefinementLoop.single_pass (1 Ouro call)")

    model = build_test_internal_model()

    timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} =
          timed(fn ->
            try do
              Brain.Response.RefinementLoop.single_pass(model, [])
            rescue
              e -> {:error, Exception.message(e)}
            end
          end)

        status =
          case result do
            {:ok, response, meta} when is_binary(response) ->
              "ok (#{String.length(response)} chars, #{meta[:iterations] || 1} iter)"

            {:error, reason} ->
              "error: #{inspect(reason)}"

            other ->
              "unexpected: #{inspect(other) |> String.slice(0, 100)}"
          end

        info("    Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("")
    info("  Summary Stage 3: #{format_stats(timings)}")
    info("")

    Map.put(results, :stage_3, stats(timings))
  end

  # ---------------------------------------------------------------------------
  # Stage 4: Full RefinementLoop.generate (up to 3 iterations)
  # ---------------------------------------------------------------------------
  defp run_stage_4(results, runs) do
    header("Stage 4: Full RefinementLoop.generate (up to @max_iterations)")

    model = build_test_internal_model()

    timings =
      Enum.map(1..runs, fn i ->
        {elapsed, result} =
          timed(fn ->
            try do
              Brain.Response.RefinementLoop.generate(model, [])
            rescue
              e -> {:error, Exception.message(e)}
            end
          end)

        status =
          case result do
            {:ok, response, meta} when is_binary(response) ->
              iters = Map.get(meta, :iterations, "?")
              "ok (#{String.length(response)} chars, #{iters} iterations)"

            {:ok, nil, meta} ->
              "silence preferred (#{Map.get(meta, :method)})"

            {:error, reason} ->
              "error: #{inspect(reason)}"

            other ->
              "unexpected: #{inspect(other) |> String.slice(0, 100)}"
          end

        info("    Run #{i}: #{elapsed}ms — #{status}")
        elapsed
      end)

    info("")
    info("  Summary Stage 4: #{format_stats(timings)}")
    info("")

    Map.put(results, :stage_4, stats(timings))
  end

  # ---------------------------------------------------------------------------
  # Stage 5: End-to-end Brain.evaluate (with :infinity timeout)
  # ---------------------------------------------------------------------------
  defp run_stage_5(results, runs) do
    header("Stage 5: End-to-end Brain.evaluate")

    test_input = List.first(@test_inputs)

    timings =
      Enum.map(1..runs, fn i ->
        {:ok, conv} = ensure_conversation()

        {elapsed, result} =
          timed(fn ->
            try do
              Brain.evaluate(conv, test_input, timeout: :infinity)
            rescue
              e -> {:error, Exception.message(e)}
            catch
              :exit, reason -> {:error, {:exit, reason}}
            end
          end)

        info("    Run #{i}: #{elapsed}ms — #{format_eval_result(result)}")
        elapsed
      end)

    info("")
    info("  Summary Stage 5: #{format_stats(timings)}")
    info("")

    Map.put(results, :stage_5, stats(timings))
  end

  # ---------------------------------------------------------------------------
  # Stage 6: Queued Brain.evaluate (two back-to-back)
  # ---------------------------------------------------------------------------
  defp run_stage_6(results) do
    header("Stage 6: Concurrent Brain.evaluate (queueing measurement)")

    {:ok, conv} = ensure_conversation()
    [input_a, input_b | _] = @test_inputs

    info("    Sending two evaluate calls concurrently...")

    parent = self()

    task_a =
      Task.async(fn ->
        {elapsed, result} =
          timed(fn ->
            try do
              Brain.evaluate(conv, input_a, timeout: :infinity)
            rescue
              e -> {:error, Exception.message(e)}
            catch
              :exit, reason -> {:error, {:exit, reason}}
            end
          end)

        send(parent, {:task_a_done, elapsed})
        {elapsed, result}
      end)

    Process.sleep(50)

    task_b =
      Task.async(fn ->
        {elapsed, result} =
          timed(fn ->
            try do
              Brain.evaluate(conv, input_b, timeout: :infinity)
            rescue
              e -> {:error, Exception.message(e)}
            catch
              :exit, reason -> {:error, {:exit, reason}}
            end
          end)

        send(parent, {:task_b_done, elapsed})
        {elapsed, result}
      end)

    {elapsed_a, result_a} = Task.await(task_a, :infinity)
    {elapsed_b, result_b} = Task.await(task_b, :infinity)

    info("    Call A: #{elapsed_a}ms — #{format_eval_result(result_a)}")
    info("    Call B: #{elapsed_b}ms — #{format_eval_result(result_b)}")
    info("    Queueing overhead (B - A): #{elapsed_b - elapsed_a}ms")
    info("")

    Map.put(results, :stage_6, %{
      call_a: elapsed_a,
      call_b: elapsed_b,
      queueing_overhead: elapsed_b - elapsed_a
    })
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp wait_for_ouro_ready(0) do
    warn("Ouro model did not become ready within timeout — proceeding anyway")
  end

  defp wait_for_ouro_ready(retries) do
    if Brain.ML.Ouro.Model.ready?() do
      info("  Ouro model: READY")
    else
      if rem(retries, 30) == 0, do: info("  Waiting for Ouro model to load...")
      Process.sleep(1_000)
      wait_for_ouro_ready(retries - 1)
    end
  end

  defp ensure_conversation do
    case Brain.create_conversation() do
      {:ok, id} when is_binary(id) -> {:ok, id}
      other -> {:error, "Could not create conversation: #{inspect(other)}"}
    end
  end

  defp build_test_chunk_analysis do
    %Brain.Analysis.ChunkAnalysis{
      chunk_index: 0,
      text: "Hello! I'm Austin. It is nice to finally meet you my friend.",
      intent: "greeting.social",
      confidence: 0.85,
      response_strategy: :can_respond,
      discourse: %Brain.Analysis.DiscourseResult{
        addressee: :bot,
        confidence: 0.9,
        direct_address_detected: false
      },
      speech_act: %Brain.Analysis.SpeechActResult{
        category: :expressive,
        sub_type: :greeting,
        confidence: 0.9,
        is_question: false,
        is_imperative: false
      },
      sentiment: %{label: :positive, confidence: 0.8},
      entities: [
        %{"type" => "person", "name" => "Austin", "confidence" => 0.95}
      ],
      epistemic_status: :unchecked
    }
  end

  defp build_test_primitives do
    [
      %Brain.Response.Primitive{
        type: :acknowledgment,
        variant: :social,
        content: %{speech_act_sub_type: :greeting, entity_name: "Austin"},
        confidence: 0.9
      },
      %Brain.Response.Primitive{
        type: :content,
        variant: :factual,
        content: %{fact: "I am a chatbot built with Elixir and classical NLP."},
        confidence: 0.8
      },
      %Brain.Response.Primitive{
        type: :follow_up,
        variant: :elaboration,
        content: %{prompt: "What would you like to talk about?"},
        confidence: 0.7
      }
    ]
  end

  defp build_test_internal_model do
    analysis = build_test_chunk_analysis()

    %Brain.Analysis.InternalModel{
      raw_input: "Hello! I'm Austin. It is nice to finally meet you my friend.",
      chunks: [
        %Brain.Analysis.Chunk{
          text: "Hello! I'm Austin. It is nice to finally meet you my friend.",
          index: 0,
          start_pos: 0,
          end_pos: 60
        }
      ],
      analyses: [analysis],
      overall_strategy: :can_respond,
      suggested_prompts: [],
      metadata: %{},
      created_at: System.system_time(:millisecond)
    }
  end

  defp timed(fun) do
    start = System.monotonic_time(:millisecond)
    result = fun.()
    elapsed = System.monotonic_time(:millisecond) - start
    {elapsed, result}
  end

  defp stats(timings) do
    sorted = Enum.sort(timings)
    count = length(sorted)

    %{
      min: List.first(sorted),
      max: List.last(sorted),
      median: Enum.at(sorted, div(count, 2)),
      mean: Enum.sum(sorted) / count |> Float.round(0),
      count: count,
      all: sorted
    }
  end

  defp format_stats(timings) do
    s = stats(timings)
    "min=#{s.min}ms  median=#{s.median}ms  max=#{s.max}ms  mean=#{s.mean}ms  (n=#{s.count})"
  end

  defp format_gen_result({:ok, text}) when is_binary(text) do
    word_count = text |> String.split() |> length()
    preview = String.slice(text, 0, 80) |> String.replace("\n", " ")
    "ok (#{word_count} words, #{String.length(text)} chars): \"#{preview}...\""
  end

  defp format_gen_result(:fallback), do: "FALLBACK (model not ready)"
  defp format_gen_result({:error, reason}), do: "error: #{inspect(reason)}"
  defp format_gen_result(other), do: "unexpected: #{inspect(other) |> String.slice(0, 100)}"

  defp format_eval_result({:ok, %{response: response, processing_method: method}})
       when is_binary(response) do
    "ok (#{String.length(response)} chars, method=#{method})"
  end

  defp format_eval_result({:ok, %{response: response}}) when is_binary(response) do
    "ok (#{String.length(response)} chars)"
  end

  defp format_eval_result({:error, reason}) do
    "error: #{inspect(reason) |> String.slice(0, 100)}"
  end

  defp format_eval_result(other) do
    "result: #{inspect(other) |> String.slice(0, 100)}"
  end

  defp print_summary(results) do
    info("")

    if Map.has_key?(results, :stage_1) do
      s1 = results.stage_1
      info("  Stage 1 — Raw Ouro.Model.generate:")
      info("    Default:            #{fmt(s1.default)}")
      info("    max_new_tokens=50:  #{fmt(s1.short)}")
      info("    max_new_tokens=200: #{fmt(s1.long)}")
      same_median = s1.short.median == s1.default.median and s1.long.median == s1.default.median

      if same_median do
        warn("    !! All variants have same median — opts likely IGNORED by Nx.Serving")
      end
    end

    if Map.has_key?(results, :stage_2) do
      info("  Stage 2 — Production RealizationPacket → Ouro: #{fmt(results.stage_2)}")
    end

    if Map.has_key?(results, :stage_3) do
      info("  Stage 3 — RefinementLoop.single_pass:          #{fmt(results.stage_3)}")
    end

    if Map.has_key?(results, :stage_4) do
      info("  Stage 4 — RefinementLoop.generate (full):      #{fmt(results.stage_4)}")
    end

    if Map.has_key?(results, :stage_5) do
      info("  Stage 5 — Brain.evaluate E2E:                  #{fmt(results.stage_5)}")
    end

    if Map.has_key?(results, :stage_6) do
      s6 = results.stage_6
      info("  Stage 6 — Concurrent Brain.evaluate:")
      info("    Call A: #{s6.call_a}ms")
      info("    Call B: #{s6.call_b}ms")
      info("    Queueing overhead: #{s6.queueing_overhead}ms")
    end

    info("")
    info("=" |> String.duplicate(72))
    info("  Benchmark complete. Use these numbers to inform timeout/UX decisions.")
    info("=" |> String.duplicate(72))
  end

  defp fmt(%{min: min, median: median, max: max, mean: mean, count: count}) do
    "min=#{min}ms  med=#{median}ms  max=#{max}ms  avg=#{mean}ms  (n=#{count})"
  end

  defp fmt(other), do: inspect(other)

  defp header(title) do
    info("")
    info("-" |> String.duplicate(72))
    info("  #{title}")
    info("-" |> String.duplicate(72))
  end

  defp info(msg), do: Mix.shell().info(msg)
  defp warn(msg), do: Mix.shell().info([:yellow, msg])
end
