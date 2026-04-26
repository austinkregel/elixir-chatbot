defmodule Brain.ML.BenchmarkTest do
  @moduledoc """
  Benchmark tests for ML model accuracy.

  These tests verify that ML models meet minimum accuracy thresholds
  on known inputs. They focus on positive assertions - testing what
  the system SHOULD do correctly.

  Intent benchmarks run against the production feature-vector pipeline
  (MicroClassifiers :intent_full + analysis refinement) so failures
  indicate model drift/regression.

  Run with: mix test --only benchmark
  """

  alias Brain.Analysis.SpeechActClassifier
  alias Brain.ML.EntityExtractor
  alias Brain.Analysis.Pipeline
  alias Brain.ML
  use ExUnit.Case, async: false
  @moduletag :benchmark
  @moduletag timeout: 300_000

  alias ML.{Evaluation, EvaluationStore}

  setup_all do
    Brain.TestHelpers.require_services!(:ml_inference)

    unless ML.MicroClassifiers.ready?() do
      raise "MicroClassifiers not ready -- run `mix train_micro` first"
    end

    :ok
  end

  describe "intent classification" do
    @tag :benchmark
    test "correctly classifies weather queries" do
      weather_inputs = [
        "What's the weather like?",
        "Will it rain tomorrow?",
        "How's the weather in London?",
        "Is it going to snow?"
      ]

      results = classify_intents(weather_inputs)

      correct =
        Enum.count(results, fn {_text, intent} ->
          intent_starts_with?(intent, "weather.query") or intent_starts_with?(intent, "weather.condition")
        end)

      assert correct >= 3,
             "Expected at least 3/#{length(weather_inputs)} weather queries classified correctly, got #{correct}/#{length(weather_inputs)}: #{inspect(results)}"
    end

    @tag :benchmark
    test "correctly classifies greeting intents" do
      greeting_inputs = ["Hello", "Hi there", "Hey", "Good morning"]

      results = classify_intents(greeting_inputs)

      correct =
        Enum.count(results, fn {_text, intent} ->
          intent_starts_with?(intent, "smalltalk.greetings")
        end)

      assert correct >= 3,
             "Expected at least 3/#{length(greeting_inputs)} greetings classified correctly, got #{correct}/#{length(greeting_inputs)}: #{inspect(results)}"
    end

    @tag :benchmark
    test "correctly classifies music play intents" do
      music_inputs = ["Play some jazz music", "Can you play a song?", "Put on some rock music"]

      results = classify_intents(music_inputs)

      correct =
        Enum.count(results, fn {_text, intent} ->
          intent_starts_with?(intent, "music.play")
        end)

      assert correct >= 2,
             "Expected at least 2/#{length(music_inputs)} music queries classified correctly, got #{correct}/#{length(music_inputs)}: #{inspect(results)}"
    end

    @tag :benchmark
    test "gold standard accuracy meets minimum threshold" do
      gold = EvaluationStore.load_gold_standard("intent")

      if gold == [] do
        IO.puts(
          "  [SKIP] No gold standard data for intent (add to priv/evaluation/intent/gold_standard.json)"
        )
      else
        {predictions, actuals} = evaluate_intent_gold(gold)
        acc = Evaluation.accuracy(predictions, actuals)

        assert acc >= 0.6,
               "Intent classification accuracy #{Float.round(acc * 100, 1)}% is below minimum threshold of 60%"
      end
    end
  end

  describe "entity extraction" do
    @tag :benchmark
    test "extracts location entities from weather queries" do
      inputs_with_locations = [
        {"What's the weather in London?", "London"},
        {"How's the weather in New York?", "New York"},
        {"Is it raining in Paris?", "Paris"}
      ]

      results =
        Enum.map(inputs_with_locations, fn {text, expected_location} ->
          entities = extract_entities(text)

          location_found =
            Enum.any?(entities, fn entity ->
              entity_value = Map.get(entity, :value) || Map.get(entity, "value", "")
              entity_type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type", "")

              String.downcase(to_string(entity_value)) == String.downcase(expected_location) and
                to_string(entity_type) in ["location", "city", "country"]
            end)

          {text, expected_location, location_found}
        end)

      correct = Enum.count(results, fn {_, _, found} -> found end)

      assert correct >= 2,
             "Expected at least 2/#{length(inputs_with_locations)} location entities extracted, got #{correct}: #{inspect(results)}"
    end

    @tag :benchmark
    test "extracts person entities" do
      inputs = [{"Tell me about Albert Einstein", "Albert Einstein"}]

      results =
        Enum.map(inputs, fn {text, expected_name} ->
          entities = extract_entities(text)

          found =
            Enum.any?(entities, fn entity ->
              entity_value = Map.get(entity, :value) || Map.get(entity, "value", "")
              String.downcase(to_string(entity_value)) == String.downcase(expected_name)
            end)

          {text, expected_name, found}
        end)

      correct = Enum.count(results, fn {_, _, found} -> found end)

      assert correct >= 1,
             "Expected person entity extraction, got #{correct}: #{inspect(results)}"
    end
  end

  describe "speech act classification" do
    @tag :benchmark
    test "correctly classifies questions as directives" do
      questions = [
        "What time is it?",
        "Where is the nearest restaurant?",
        "How do I get to the airport?"
      ]

      results =
        Enum.map(questions, fn text ->
          result = SpeechActClassifier.classify(text)
          {text, result.category, result.is_question}
        end)

      question_count = Enum.count(results, fn {_, _, is_q} -> is_q end)

      assert question_count >= 2,
             "Expected at least 2/#{length(questions)} classified as questions, got #{question_count}: #{inspect(results)}"
    end

    @tag :benchmark
    test "correctly classifies commands as directives" do
      commands = ["Turn on the lights", "Play some music", "Set an alarm for 7am"]

      results =
        Enum.map(commands, fn text ->
          result = SpeechActClassifier.classify(text)
          {text, result.category}
        end)

      directive_count = Enum.count(results, fn {_, cat} -> cat == :directive end)

      assert directive_count >= 2,
             "Expected at least 2/#{length(commands)} classified as directives, got #{directive_count}: #{inspect(results)}"
    end

    @tag :benchmark
    test "correctly classifies greetings as expressives" do
      greetings = ["Hello!", "Good morning!", "Thanks a lot!"]

      results =
        Enum.map(greetings, fn text ->
          result = SpeechActClassifier.classify(text)
          {text, result.category}
        end)

      expressive_count = Enum.count(results, fn {_, cat} -> cat == :expressive end)

      assert expressive_count >= 2,
             "Expected at least 2/#{length(greetings)} classified as expressives, got #{expressive_count}: #{inspect(results)}"
    end
  end

  describe "full pipeline" do
    @tag :benchmark
    test "pipeline processes multi-sentence input with correct strategy" do
      result = Pipeline.process("Hello! What's the weather?")

      assert result.analyses != [], "Expected at least 1 analysis chunk"

      assert result.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ],
             "Expected actionable strategy, got #{result.overall_strategy}"
    end

    @tag :benchmark
    test "pipeline includes sentiment in analysis" do
      result = Pipeline.process("I'm really frustrated with this.")

      first_analysis = List.first(result.analyses)

      assert first_analysis != nil, "Expected at least one analysis"
      assert Map.has_key?(first_analysis, :sentiment), "Expected sentiment field in analysis"

      if first_analysis.sentiment do
        assert Map.has_key?(first_analysis.sentiment, :label), "Expected sentiment to have :label"

        assert Map.has_key?(first_analysis.sentiment, :confidence),
               "Expected sentiment to have :confidence"
      end
    end
  end

  defp classify_intents(texts) do
    Enum.map(texts, fn text ->
      intent =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          to_string(analysis.intent || "unknown")
        rescue
          _ -> "unknown"
        catch
          :exit, _ -> "unknown"
        end

      {text, intent}
    end)
  end

  defp intent_starts_with?(intent, prefix) do
    String.starts_with?(to_string(intent), prefix)
  end

  defp extract_entities(text) do
    try do
      EntityExtractor.extract_entities(text)
    rescue
      _ -> []
    catch
      :exit, _ -> []
    end
  end

  defp evaluate_intent_gold(gold) do
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["intent"]

      predicted =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          to_string(analysis.intent || "unknown")
        rescue
          _ -> "unknown"
        catch
          :exit, _ -> "unknown"
        end

      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end
end
