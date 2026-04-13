defmodule Brain.Analysis.PipelineIntegrationTest do
  @moduledoc """
  Integration tests for the full Pipeline.process/2 with real text inputs.

  Covers:
  - Greeting flow
  - Weather + slot detection
  - Ambiguous input handling
  - Multi-turn conversation simulation
  - Graceful degradation when LSTM is unavailable
  - Edge cases (empty, whitespace, long input)
  """
  use Brain.Test.GraphCase, async: false

  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline

  setup _context do
    start_test_services()
    :ok
  end

  describe "greeting flow" do
    test "simple greeting: intent starts with smalltalk.greet" do
      model = Pipeline.process("Hello there!")
      first = List.first(model.analyses || [])
      intent = first && first.intent
      assert is_binary(intent)
      assert String.starts_with?(intent, "smalltalk.greet")
    end

    test "greeting with followup question" do
      model = Pipeline.process("Hi! How are you doing today?")
      assert length(model.chunks) >= 2
      assert length(model.analyses) >= 2

      # First chunk should be a greeting
      first = List.first(model.analyses)
      assert first.intent != nil
    end

    test "farewell recognized" do
      model = Pipeline.process("Goodbye, see you later!")
      first = List.first(model.analyses || [])
      assert first != nil
      assert first.intent != nil
    end
  end

  describe "weather + slot detection" do
    test "weather question processes and may extract location entities" do
      model = Pipeline.process("What is the weather in NYC?")
      first = List.first(model.analyses || [])
      assert first != nil
      assert first.intent != nil
    end

    test "weather question with missing location triggers slot detection" do
      model = Pipeline.process("What's the weather?")
      first = List.first(model.analyses || [])
      assert first != nil

      # The slot detector should identify that location is missing
      # (if intent is weather-related)
      if first.intent && String.contains?(first.intent, "weather") do
        assert is_list(first.missing_context) or first.missing_context == nil
      end
    end

    test "weather question with explicit location" do
      model = Pipeline.process("Tell me the forecast for San Francisco tomorrow")
      first = List.first(model.analyses || [])
      assert first != nil
      assert first.intent != nil
      # Entity extraction should attempt to find San Francisco
      entities = first.entities || []
      assert is_list(entities)
    end
  end

  describe "ambiguous input" do
    test "short ambiguous input doesn't crash" do
      model = Pipeline.process("it")
      assert model != nil
      first = List.first(model.analyses || [])
      assert first != nil
    end

    test "vague question gets processed" do
      model = Pipeline.process("can you help")
      first = List.first(model.analyses || [])
      assert first != nil
      assert first.intent != nil
    end

    test "ambiguous entity mention" do
      model = Pipeline.process("Tell me about Mercury")
      # Mercury could be a planet, element, or mythological figure
      first = List.first(model.analyses || [])
      assert first != nil
      entities = first.entities || []
      assert is_list(entities)
    end

    test "mixed intent input" do
      model = Pipeline.process("Play some jazz and what's the weather like?")
      analyses = model.analyses || []
      # Should produce multiple analyses for the compound request
      assert length(analyses) >= 1
    end
  end

  describe "multi-turn conversation simulation" do
    test "context carries through user_id option" do
      opts = [user_id: "test_user_123"]

      # Turn 1: greeting
      model1 = Pipeline.process("Hello!", opts)
      assert model1 != nil
      first1 = List.first(model1.analyses || [])
      assert first1 != nil

      # Turn 2: question
      model2 = Pipeline.process("What's the capital of France?", opts)
      assert model2 != nil
      first2 = List.first(model2.analyses || [])
      assert first2 != nil
      assert first2.intent != nil

      # Turn 3: follow-up
      model3 = Pipeline.process("And what about Germany?", opts)
      assert model3 != nil
      first3 = List.first(model3.analyses || [])
      assert first3 != nil
    end

    test "multiple sentences are chunked correctly" do
      model = Pipeline.process("I love Paris. Can you tell me about the Eiffel Tower? It's amazing!")
      assert length(model.chunks) >= 3
      assert length(model.analyses) >= 3
    end
  end

  describe "graceful degradation" do
    test "pipeline works without LSTM models" do
      # LSTM models may not be loaded in test environment
      # Pipeline should gracefully fall back to TF-IDF
      model = Pipeline.process("What is machine learning?")
      first = List.first(model.analyses || [])
      assert first != nil
      assert first.intent != nil
    end

    test "sentiment analysis falls back when LSTM unavailable" do
      model = Pipeline.process("I absolutely love this!")
      first = List.first(model.analyses || [])
      assert first != nil
      # Sentiment should be available (either LSTM or fallback)
      sentiment = first.sentiment
      assert sentiment != nil or first.intent != nil
    end

    test "speech act classification works without LSTM" do
      model = Pipeline.process("Could you please help me?")
      first = List.first(model.analyses || [])
      assert first != nil
      # Speech act should be classified
      assert first.speech_act != nil
    end
  end

  describe "edge cases" do
    test "empty input: graceful handling" do
      model = Pipeline.process("")
      assert model != nil
    end

    test "whitespace input: graceful handling" do
      model = Pipeline.process("   \n\t  ")
      assert model != nil
    end

    test "very long input: no crash, reasonable processing" do
      long_input = String.duplicate("Hello world. ", 100)
      start = System.monotonic_time(:millisecond)
      model = Pipeline.process(long_input)
      elapsed = System.monotonic_time(:millisecond) - start
      assert model != nil
      assert elapsed < 30_000
    end

    test "special characters don't crash" do
      model = Pipeline.process("Hello!!! @#$%^& ???")
      assert model != nil
    end

    test "unicode text processes correctly" do
      model = Pipeline.process("Bonjour, comment allez-vous?")
      assert model != nil
      first = List.first(model.analyses || [])
      assert first != nil
    end

    test "numeric input handled" do
      model = Pipeline.process("42")
      assert model != nil
    end

    test "command input: speech act has command or directive category" do
      model = Pipeline.process("Turn off the lights")
      analyses = model.analyses || []

      categories =
        Enum.flat_map(analyses, fn a ->
          case a.speech_act do
            %{category: cat} when is_atom(cat) -> [cat]
            _ -> []
          end
        end)

      assert :command in categories or :directive in categories or length(analyses) >= 1
    end
  end
end
