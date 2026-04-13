defmodule Mix.Tasks.Ouro.Verify do
  @moduledoc """
  Verifies the Ouro LoopLM integration end-to-end via Bumblebee.

  Runs progressive checks:
    1. Model files present
    2. Bumblebee loads model spec and parameters
    3. Bumblebee loads tokenizer
    4. Full generation pipeline (prompt → serving → text)

  ## Usage

      mix ouro.verify           # Run all checks
      mix ouro.verify --step 2  # Run up to step 2 only
  """

  use Mix.Task

  alias Brain.ML.Ouro.ModelDownloader

  @shortdoc "Verifies Ouro LoopLM integration end-to-end"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [step: :integer])
    max_step = Keyword.get(opts, :step, 4)

    Mix.Task.run("app.start")

    model_dir = ouro_dir()

    info("Ouro LoopLM Verification (Bumblebee)")
    info("======================================")
    info("Model directory: #{model_dir}")
    info("")

    steps = [
      {1, "Model files present", fn -> verify_files(model_dir) end},
      {2, "Bumblebee loads model", fn -> verify_model(model_dir) end},
      {3, "Bumblebee loads tokenizer", fn -> verify_tokenizer(model_dir) end},
      {4, "Full generation", fn -> verify_generation() end}
    ]

    results =
      steps
      |> Enum.filter(fn {n, _, _} -> n <= max_step end)
      |> Enum.reduce_while([], fn {n, name, check_fn}, acc ->
        info("Step #{n}: #{name}")

        case run_check(check_fn) do
          {:ok, detail} ->
            success("  PASS: #{detail}")
            {:cont, [{n, :pass, ""} | acc]}

          {:error, reason} ->
            fail("  FAIL: #{reason}")
            {:halt, [{n, :fail, reason} | acc]}
        end
      end)

    info("")
    passed = Enum.count(results, fn {_, status, _} -> status == :pass end)
    total = length(results)
    info("Results: #{passed}/#{total} passed")

    if Enum.all?(results, fn {_, status, _} -> status == :pass end) do
      success("All checks passed!")
    end
  end

  defp run_check(fun) do
    try do
      fun.()
    rescue
      e ->
        {:error,
         "#{Exception.message(e)}\n#{Exception.format_stacktrace(__STACKTRACE__) |> String.slice(0, 500)}"}
    catch
      kind, reason ->
        {:error, "#{kind}: #{inspect(reason)}"}
    end
  end

  defp verify_files(model_dir) do
    required = ModelDownloader.required_files()

    missing =
      Enum.filter(required, fn f ->
        not File.exists?(Path.join(model_dir, f))
      end)

    if missing == [] do
      weights_path = Path.join(model_dir, "model.safetensors")

      case ModelDownloader.validate_model_safetensors(weights_path) do
        :ok ->
          safetensors_size =
            weights_path
            |> File.stat!()
            |> Map.get(:size)
            |> format_size()

          {:ok, "All #{length(required)} files present (model: #{safetensors_size})"}

        {:error, reason} ->
          {:error,
           "model.safetensors failed sanity check: #{reason}. " <>
             "Bumblebee would fail later with :eof when reading tensors."}
      end
    else
      {:error, "Missing files: #{Enum.join(missing, ", ")}"}
    end
  end

  defp verify_model(model_dir) do
    repo = {:local, model_dir}

    case Bumblebee.load_model(repo,
           module: Brain.ML.Ouro.Spec,
           architecture: :for_causal_language_modeling,
           type: :bf16,
           backend: {EXLA.Backend, client: :host}
         ) do
      {:ok, %{spec: spec, params: params}} ->
        param_count =
          params.data
          |> count_tensors()

        {:ok,
         "Model loaded: #{spec.num_blocks} blocks × #{spec.max_recurrence} recurrence, #{param_count} param tensors"}

      {:error, reason} ->
        {:error, "Bumblebee.load_model failed: #{inspect(reason)}"}
    end
  end

  defp verify_tokenizer(model_dir) do
    repo = {:local, model_dir}

    case Brain.ML.Ouro.Tokenizer.load(repo) do
      {:ok, tokenizer} ->
        input = Bumblebee.apply_tokenizer(tokenizer, "Hello, how are you?")
        num_tokens = Nx.axis_size(input["input_ids"], 1)
        {:ok, "Tokenizer loaded. \"Hello, how are you?\" → #{num_tokens} tokens"}

      {:error, reason} ->
        {:error, "Bumblebee.load_tokenizer failed: #{inspect(reason)}"}
    end
  end

  defp verify_generation do
    info("  Waiting for Ouro.Model GenServer to be ready...")

    case wait_for_ready(300) do
      :ok ->
        messages = [
          %{
            role: "system",
            content: "You are a response realization engine. Output only the response text."
          },
          %{
            role: "user",
            content:
              Jason.encode!(%{
                mode: "plan_realization",
                tone: "neutral",
                plan: [
                  %{
                    type: "acknowledgment",
                    variant: "social",
                    payload: %{speech_act_sub_type: "greeting"}
                  },
                  %{type: "follow_up", variant: "elaboration", payload: %{}}
                ]
              })
          }
        ]

        info("  Generating response from test plan...")

        case Brain.ML.Ouro.Model.generate(messages, max_new_tokens: 50) do
          {:ok, text} ->
            {:ok, "Generated: \"#{String.slice(text, 0, 200)}\""}

          :fallback ->
            {:error, "Model returned :fallback despite being ready"}

          {:error, reason} ->
            {:error, "Generation failed: #{inspect(reason)}"}
        end

      :timeout ->
        {:error, "Ouro.Model GenServer not ready after 300s"}
    end
  end

  defp wait_for_ready(0), do: :timeout

  defp wait_for_ready(retries) do
    if Brain.ML.Ouro.Model.ready?() do
      :ok
    else
      Process.sleep(1_000)
      wait_for_ready(retries - 1)
    end
  end

  defp count_tensors(data) when is_map(data) do
    Enum.reduce(data, 0, fn
      {_k, %Nx.Tensor{}}, acc -> acc + 1
      {_k, v}, acc when is_map(v) -> acc + count_tensors(v)
      _, acc -> acc
    end)
  end

  defp ouro_dir do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/ml_models/ouro"
      priv -> Path.join(to_string(priv), "ml_models/ouro")
    end
  end

  defp format_size(bytes) when bytes < 1_048_576, do: "#{Float.round(bytes / 1_024, 1)} KB"

  defp format_size(bytes) when bytes < 1_073_741_824,
    do: "#{Float.round(bytes / 1_048_576, 1)} MB"

  defp format_size(bytes), do: "#{Float.round(bytes / 1_073_741_824, 2)} GB"

  defp info(msg), do: Mix.shell().info(msg)
  defp success(msg), do: Mix.shell().info([:green, msg])
  defp fail(msg), do: Mix.shell().info([:red, msg])
end
