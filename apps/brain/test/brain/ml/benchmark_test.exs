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

  # These run the whole pipeline, which reads the knowledge graph. Without a
  # checked-out connection every one of those lookups fails, and the graph
  # features silently read as "unknown" — 2,960 failed lookups in one suite
  # run before this was added.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end

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

    # One `Pipeline.analyze_chunk/2` costs about 197 ms (measured 2026-09-22,
    # after the per-token POS fix; it was ~744 ms before). This used to sweep
    # the whole 4,870-row corpus in a single test -- 16 minutes, which no
    # timeout tolerates. It was killed at 300 s having measured nothing, while
    # the pipeline work it had started kept running without its sandbox
    # connection: that is where 1,777 failed graph lookups in one suite run
    # came from.
    #
    # It now reads the held-out split (500 rows, carved 2026-09-24), which is
    # both the honest partition and small enough to finish inside one timeout:
    #
    #   100 entries ~ 20 s   (one group)
    #   500 entries ~ 100 s  (the whole held-out split, five groups)
    #
    # Sample size is chosen for what it buys. The standard error of an accuracy
    # estimate is sqrt(p(1-p)/n): about 5.0% at n = 100 and 2.2% at n = 500.
    # Against a 60% threshold, 500 settles the question to within a couple of
    # points, so the default evaluates the split in full. GOLD_SAMPLE=<n>
    # takes a smaller stride sample when you want a faster answer.
    #
    # Entries are taken by a fixed stride rather than from the front, because
    # the file is grouped by intent -- the first N rows are not a sample of it.
    # Groups are evaluated one at a time so memory and time stay bounded, and
    # so a group that goes wrong reports its own numbers instead of taking the
    # whole measurement down with it.
    @gold_group_size 100
    @gold_default_sample 500

    @tag :benchmark
    @tag timeout: 180_000
    test "gold standard accuracy meets minimum threshold" do
      # Held-out only. ModelFactory trains the suite's :intent_full from the
      # :train partition, so scoring against the full corpus would be scoring
      # the model on rows it was just fitted to.
      gold = EvaluationStore.load_gold_standard("intent", :held_out)

      assert gold != [],
             "The held-out split for intent is empty. It is loaded by " <>
               "EvaluationStore.load_gold_standard/2 from priv/evaluation/intent/held_out.json; " <>
               "an empty list here means the measurement cannot be made, which is not the same as passing."

      entries = gold_sample(gold)

      {predictions, actuals, errors} =
        entries
        |> Enum.chunk_every(@gold_group_size)
        |> Enum.reduce({[], [], []}, fn group, {preds, acts, errs} ->
          {p, a, e} = evaluate_intent_gold(group)
          {preds ++ p, acts ++ a, errs ++ e}
        end)

      acc = Evaluation.accuracy(predictions, actuals)

      assert errors == [],
             "#{length(errors)}/#{length(entries)} entries failed to analyse rather than " <>
               "classifying wrongly: #{inspect(Enum.take(errors, 3))}"

      assert acc >= 0.6,
             "Intent classification accuracy #{Float.round(acc * 100, 1)}% over " <>
               "#{length(entries)} of #{length(gold)} gold entries is below the 60% threshold"
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

  # The entries this run measures. `GOLD_SAMPLE=all` takes the whole set (16
  # minutes); a number takes that many; the default is @gold_default_sample.
  # Taken by a fixed stride rather than from the front: the gold file is
  # grouped by intent, so its first 500 rows are not a sample of it.
  defp gold_sample(gold) do
    case System.get_env("GOLD_SAMPLE") do
      "all" ->
        gold

      value when is_binary(value) ->
        case Integer.parse(value) do
          {n, ""} when n > 0 -> stride_sample(gold, n)
          _ -> raise ArgumentError, "GOLD_SAMPLE must be a positive integer or \"all\", got #{inspect(value)}"
        end

      nil ->
        stride_sample(gold, @gold_default_sample)
    end
  end

  defp stride_sample(gold, wanted) do
    total = length(gold)

    if wanted >= total do
      gold
    else
      stride = div(total, wanted)

      gold
      |> Enum.with_index()
      |> Enum.filter(fn {_entry, index} -> rem(index, stride) == 0 end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.take(wanted)
    end
  end

  # Returns `{predictions, actuals, errors}`. An entry the pipeline could not
  # analyse is an error, not a wrong answer: counting a crash as a
  # misclassification quietly turns "the analyser broke" into "the model is
  # 2% less accurate".
  defp evaluate_intent_gold(gold) do
    {preds, acts, errors} =
      Enum.reduce(gold, {[], [], []}, fn example, {preds, acts, errors} ->
        text = example["text"]
        expected = example["intent"]

        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          {[to_string(analysis.intent || "unknown") | preds], [expected | acts], errors}
        rescue
          e -> {preds, acts, [{text, Exception.message(e)} | errors]}
        catch
          :exit, reason -> {preds, acts, [{text, {:exit, reason}} | errors]}
        end
      end)

    {Enum.reverse(preds), Enum.reverse(acts), Enum.reverse(errors)}
  end
end
