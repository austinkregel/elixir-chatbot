defmodule Brain.Response.CompressedDialogIntegrationTest do
  @moduledoc """
  Integration tests for the compressed dialog pipeline.

  These tests exercise the full path from raw text through:

      Analysis Pipeline → DiscoursePlanner → ContentSpecifier
        → SurfaceRealizer → ResponseEvaluator → RefinementLoop

  The generated response is then re-analyzed through the same pipeline
  to verify it reads as coherent language — not concatenated template
  fragments or garbage output.

  Each test follows the pattern:
  1. Analyze the user input
  2. Generate a response via the synthesis pipeline
  3. Validate structural integrity (primitives, composition)
  4. Re-analyze the response to verify it reads as natural language
  5. Assert discourse-appropriate properties (speech act, sentiment, etc.)
  """
  use ExUnit.Case, async: false

  alias Brain.Analysis.Pipeline
  alias Brain.Response.{RefinementLoop, Primitive, DiscoursePlanner, ContentSpecifier}
  alias Brain.Response.ResponseEvaluator.Score

  @moduletag :integration

  @min_response_words 3

  # ──────────────────────────────────────────────────────────────────
  # Helpers
  # ──────────────────────────────────────────────────────────────────

  defp analyze(text, opts \\ []) do
    Pipeline.process(text, opts)
  end

  defp generate(model, opts \\ []) do
    RefinementLoop.generate(model, opts)
  end

  defp primitive_types(metadata) do
    metadata.primitives |> Enum.map(& &1.type)
  end

  defp find_primitive(metadata, type) do
    Enum.find(metadata.primitives, &(&1.type == type))
  end

  defp find_primitive(metadata, type, variant) do
    Enum.find(metadata.primitives, &(&1.type == type and &1.variant == variant))
  end

  defp word_count(text), do: text |> String.split() |> length()

  defp sentences(text) do
    text
    |> String.split(~r/[.!?]+/, trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
  end

  # ── Response quality assertions ──

  defp assert_coherent_response(input, response, metadata) do
    assert is_binary(response),
      "Response must be a string, got: #{inspect(response)}"

    assert word_count(response) >= @min_response_words,
      "Response too short (#{word_count(response)} words): #{inspect(response)}"

    refute String.contains?(response, "$"),
      "Response contains unresolved template placeholder: #{inspect(response)}"

    refute String.contains?(response, "%{"),
      "Response contains unresolved interpolation: #{inspect(response)}"

    sents = sentences(response)
    dupes = sents -- Enum.uniq(sents)
    assert dupes == [],
      """
      Response contains duplicate sentences (sign of primitive concatenation):
      Duplicated: #{inspect(dupes)}
      Full response: #{inspect(response)}
      """

    response_model = analyze(response)
    assert response_model.analyses != [],
      """
      Generated response could not be analyzed as language.
      Input:    #{inspect(input)}
      Response: #{inspect(response)}
      """

    response_analysis = List.first(response_model.analyses)

    response_sentiment = response_analysis.sentiment || %{}
    response_sentiment_label = Map.get(response_sentiment, :label, :neutral)
    response_sentiment_conf = Map.get(response_sentiment, :confidence, 0.0)

    input_model = analyze(input)
    input_analysis = List.first(input_model.analyses) || %{}
    input_sentiment = Map.get(input_analysis, :sentiment) || %{}
    input_sentiment_label = Map.get(input_sentiment, :label, :neutral)
    input_is_negative = input_sentiment_label == :negative

    refute response_sentiment_label == :negative and response_sentiment_conf > 0.3 and not input_is_negative,
      """
      Response to user has negative sentiment — the bot should not sound hostile.
      Input:    #{inspect(input)}
      Response: #{inspect(response)}
      Sentiment: #{inspect(response_sentiment)}
      """

    assert_no_echo(input, response)

    rendered_texts =
      metadata.primitives
      |> Enum.map(& &1.rendered)
      |> Enum.reject(&(is_nil(&1) or &1 == ""))

    assert rendered_texts != [],
      "No primitives were rendered: #{inspect(Enum.map(metadata.primitives, &{&1.type, &1.variant}))}"

    for rendered <- rendered_texts do
      assert String.contains?(response, rendered),
        """
        Response is not composed from its primitives.
        Missing: #{inspect(rendered)}
        Response: #{inspect(response)}
        """
    end

    {input, response, response_model}
  end

  defp assert_no_echo(input, response) do
    input_words = input |> String.downcase() |> String.split()

    if length(input_words) < 4 do
      :ok
    else
      response_words = response |> String.downcase() |> String.split()
      echoed = find_longest_echo(input_words, response_words)

      assert length(echoed) < 4,
        """
        Response echoes the user's input verbatim (#{length(echoed)} consecutive words).
        Echoed:   #{inspect(Enum.join(echoed, " "))}
        Input:    #{inspect(input)}
        Response: #{inspect(response)}
        """
    end
  end

  defp find_longest_echo(input_words, response_words) do
    forward = longest_common_run(input_words, response_words)
    reverse = longest_common_run(response_words, input_words)

    if length(reverse) > length(forward), do: reverse, else: forward
  end

  defp longest_common_run(needle, haystack) do
    needle_len = length(needle)
    haystack_len = length(haystack)

    if haystack_len < needle_len do
      haystack
      |> Enum.with_index()
      |> Enum.reduce([], fn {_, start}, best ->
        window = Enum.slice(haystack, start, needle_len)
        run = count_prefix_match(needle, window, [])
        if length(run) > length(best), do: run, else: best
      end)
    else
      haystack
      |> Enum.chunk_every(needle_len, 1, :discard)
      |> Enum.reduce([], fn window, best ->
        run = count_prefix_match(needle, window, [])
        if length(run) > length(best), do: run, else: best
      end)
    end
  end

  defp count_prefix_match([], _, acc), do: Enum.reverse(acc)
  defp count_prefix_match(_, [], acc), do: Enum.reverse(acc)
  defp count_prefix_match([a | rest_a], [b | rest_b], acc) do
    if a == b, do: count_prefix_match(rest_a, rest_b, [a | acc]), else: Enum.reverse(acc)
  end

  # ──────────────────────────────────────────────────────────────────
  # Full pipeline: greeting
  # ──────────────────────────────────────────────────────────────────

  describe "full pipeline: greeting" do
    test "Hello produces a greeting-appropriate response" do
      model = analyze("Hello!")
      assert {:ok, response, metadata} = generate(model)

      {_input, _response, response_model} =
        assert_coherent_response("Hello!", response, metadata)

      assert metadata.method == :synthesis_pipeline

      social = find_primitive(metadata, :acknowledgment, :social)
      assert social != nil, "Expected a social acknowledgment primitive"
      assert social.rendered != nil and social.rendered != "",
        "Social acknowledgment was planned but never rendered"

      response_analysis = List.first(response_model.analyses)
      response_speech_act = response_analysis.speech_act || %{}
      response_category = Map.get(response_speech_act, :category, :unknown)

      assert response_category in [:expressive, :commissive, :assertive],
        """
        Response to a greeting should read as expressive, commissive, or assertive.
        Got speech act: #{inspect(response_category)}
        Response: #{inspect(response)}
        """

      assert %Score{} = metadata.score
      assert metadata.score.speech_act_alignment >= 0.7
    end

    test "conversational greeting produces analyzable discourse" do
      model = analyze("Hey there, how are you?")
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response("Hey there, how are you?", response, metadata)

      types = primitive_types(metadata)
      assert :acknowledgment in types or :follow_up in types,
        "Expected acknowledgment or follow_up for conversational greeting, got: #{inspect(types)}"
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Full pipeline: questions
  # ──────────────────────────────────────────────────────────────────

  describe "full pipeline: questions" do
    test "weather query produces informative response that re-analyzes cleanly" do
      model = analyze("What's the weather like today?")
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response("What's the weather like today?", response, metadata)

      types = primitive_types(metadata)
      assert :framing in types or :content in types,
        "Expected framing or content for a question, got: #{inspect(types)}"

      assert %Score{} = metadata.score
      assert metadata.score.speech_act_alignment >= 0.5
    end

    test "knowledge question response is built from content primitives" do
      model = analyze("What do you know about machine learning?")
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response("What do you know about machine learning?", response, metadata)

      has_content = Enum.any?(metadata.primitives, fn p ->
        p.type in [:content, :framing] and p.rendered != nil and p.rendered != ""
      end)
      assert has_content,
        "Expected at least one rendered content/framing primitive for a knowledge question"
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Full pipeline: negative sentiment
  # ──────────────────────────────────────────────────────────────────

  describe "full pipeline: negative sentiment" do
    test "strong negative signal produces empathetic response" do
      input = "I'm having a really terrible day and nothing is going right"
      model = analyze(input)

      # Override sentiment if the classifier didn't produce a strong enough
      # signal — this test validates pipeline wiring, not classifier quality.
      primary = List.first(model.analyses)
      sentiment = primary.sentiment || %{}

      model =
        if Map.get(sentiment, :label) != :negative or Map.get(sentiment, :confidence, 0) < 0.6 do
          updated = Enum.map(model.analyses, fn a ->
            %{a | sentiment: %{label: :negative, confidence: 0.85}}
          end)
          %{model | analyses: updated}
        else
          model
        end

      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)

      types = primitive_types(metadata)
      assert :attunement in types,
        "Expected attunement primitive for strong negative sentiment, got: #{inspect(types)}"

      attunement = find_primitive(metadata, :attunement)
      assert attunement.rendered != nil and attunement.rendered != "",
        "Attunement was planned but never rendered"
      assert String.contains?(response, attunement.rendered),
        "Response does not include its own empathetic attunement: #{inspect(response)}"
    end

    test "sad statement through real pipeline produces a coherent response" do
      input = "I'm having a really terrible day and nothing is going right"
      model = analyze(input)
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)

      primary = List.first(model.analyses)
      sentiment = primary.sentiment || %{}

      if Map.get(sentiment, :label) == :negative and Map.get(sentiment, :confidence, 0) > 0.6 do
        assert :attunement in primitive_types(metadata),
          "Classifier detected strong negative sentiment but no attunement was planned"
      end
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Full pipeline: low confidence
  # ──────────────────────────────────────────────────────────────────

  describe "full pipeline: low confidence" do
    test "low-confidence input produces hedging in the response" do
      input = "What is the quantum chromodynamic coupling constant at 91 GeV?"
      model = analyze(input)

      model =
        if model.analyses != [] do
          updated = Enum.map(model.analyses, &%{&1 | confidence: 0.15})
          %{model | analyses: updated}
        else
          model
        end

      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)

      types = primitive_types(metadata)
      assert :hedging in types,
        "Expected hedging primitive for low-confidence input, got: #{inspect(types)}"

      hedging = find_primitive(metadata, :hedging)
      assert hedging.content[:confidence_level] != nil
      assert hedging.content[:confidence_level] <= 0.4,
        "Hedging should reflect low confidence, got: #{hedging.content[:confidence_level]}"
      assert hedging.rendered != nil and hedging.rendered != "",
        "Hedging was planned but never rendered"
      assert String.contains?(response, hedging.rendered),
        "Response does not include its own hedging text: #{inspect(response)}"
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Stage-by-stage: verifying intermediate representations
  # ──────────────────────────────────────────────────────────────────

  describe "stage-by-stage: plan → specify" do
    test "planner produces well-formed Primitive structs" do
      model = analyze("Tell me about the weather in London")

      plan = DiscoursePlanner.plan(model)
      assert is_list(plan)
      assert length(plan) >= 1

      for p <- plan do
        assert is_struct(p, Primitive),
          "Planner must return Primitive structs, got: #{inspect(p)}"

        assert is_atom(p.type), "Primitive type must be an atom, got: #{inspect(p.type)}"

        assert p.type in [:acknowledgment, :framing, :hedging, :content,
                          :attunement, :follow_up, :contradiction_response, :transition],
          "Unexpected primitive type: #{inspect(p.type)}"
      end
    end

    test "specifier fills content maps without changing primitive count" do
      model = analyze("Tell me about the weather in London")
      primary = List.first(model.analyses)

      plan = DiscoursePlanner.plan(model)
      specified = ContentSpecifier.specify(plan, primary)

      assert length(specified) == length(plan),
        "Specifier must preserve primitive count (plan: #{length(plan)}, specified: #{length(specified)})"

      for {original, spec} <- Enum.zip(plan, specified) do
        assert spec.type == original.type,
          "Specifier must not change primitive types (was #{original.type}, became #{spec.type})"

        assert is_map(spec.content),
          "Primitive #{spec.type}/#{spec.variant} has no content map after specification"
      end
    end

    test "specified primitives are not yet rendered" do
      model = analyze("What is machine learning?")
      primary = List.first(model.analyses)

      plan = DiscoursePlanner.plan(model)
      specified = ContentSpecifier.specify(plan, primary)

      for p <- specified do
        refute Primitive.rendered?(p),
          "Primitives should not be rendered until SurfaceRealizer runs, " <>
          "but #{p.type}/#{p.variant} has rendered=#{inspect(p.rendered)}"
      end
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Refinement behavior
  # ──────────────────────────────────────────────────────────────────

  describe "refinement convergence" do
    test "high-confidence greeting converges with a coherent response" do
      input = "Hi!"
      model = analyze(input)
      assert {:ok, response, metadata} = generate(model, max_iterations: 5)

      assert_coherent_response(input, response, metadata)

      assert metadata.score.converged == true or metadata.iterations < 5,
        "Expected convergence for simple greeting, took #{metadata.iterations} iterations"

      assert metadata.score.overall >= 0.5,
        "Converged greeting should score >= 0.5, got: #{metadata.score.overall}"
    end

    test "single_pass produces a coherent scored response" do
      input = "What time is it?"
      model = analyze(input)
      assert {:ok, response, metadata} = RefinementLoop.single_pass(model)

      assert_coherent_response(input, response, metadata)

      assert metadata.iterations == 1
      assert %Score{} = metadata.score
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Multi-sentence / multi-chunk inputs
  # ──────────────────────────────────────────────────────────────────

  describe "multi-sentence input" do
    test "greeting + question produces a response that re-analyzes as coherent" do
      input = "Hey! What's the weather like?"
      model = analyze(input)
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)

      types = primitive_types(metadata)
      has_social = :acknowledgment in types
      has_informative = :framing in types or :content in types

      assert has_social or has_informative,
        "Expected social or informative primitives for greeting+question, got: #{inspect(types)}"

      assert length(metadata.primitives) >= 2,
        "Mixed input should produce >= 2 primitives, got: #{length(metadata.primitives)}"
    end

    test "statement + request produces coherent multi-primitive response" do
      input = "I'm feeling great today. Can you play some music?"
      model = analyze(input)
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)
      assert length(metadata.primitives) >= 1
    end
  end

  # ──────────────────────────────────────────────────────────────────
  # Evaluator dimension checks
  # ──────────────────────────────────────────────────────────────────

  describe "evaluator scoring dimensions" do
    test "all score dimensions are populated and bounded" do
      input = "Hello, can you help me with something?"
      model = analyze(input)
      assert {:ok, response, metadata} = generate(model)

      assert_coherent_response(input, response, metadata)

      score = metadata.score
      assert %Score{} = score

      dimensions = [
        {:speech_act_alignment, score.speech_act_alignment},
        {:confidence_alignment, score.confidence_alignment},
        {:content_coverage, score.content_coverage},
        {:naturalness, score.naturalness},
        {:overall, score.overall}
      ]

      for {name, value} <- dimensions do
        assert is_float(value), "#{name} should be a float, got: #{inspect(value)}"
        assert value >= 0.0 and value <= 1.0,
          "#{name} should be in [0.0, 1.0], got: #{value}"
      end

      assert score.overall > 0.0,
        "Overall score should be positive for a valid response"
    end
  end
end
