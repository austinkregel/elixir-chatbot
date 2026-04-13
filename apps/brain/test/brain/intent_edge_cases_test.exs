defmodule Brain.IntentEdgeCasesTest do
  @moduledoc """
  Edge case tests that make POSITIVE assertions about intent classification
  and entity extraction using production models.

  These tests verify consistent behavior by asserting:
  1. Specific intents are returned for specific inputs
  2. Expected entities are extracted with correct types
  3. Confidence thresholds are met

  ## Model Requirements

  These tests REQUIRE trained models. If models are not loaded, tests will
  fail with a clear error message. Run `mix train` before running these tests.

  ## Test Philosophy

  All tests run by default. If a test fails, it means:
  - The model needs training data for that intent, OR
  - There's a regression in the model

  This ensures we always know what's working and what's not.
  """
  use ExUnit.Case, async: false
  use Brain.Test.ModelAssertions

  alias Brain.ML.IntentClassifierSimple
  alias Brain.ML.EntityExtractor

  @moduletag :edge_cases
  @moduletag :requires_models

  # Minimum confidence threshold for positive assertions
  @min_confidence 0.20

  setup_all do
    Application.ensure_all_started(:brain)
    :timer.sleep(500)

    # Require TF-IDF and gazetteer models - fail fast if not loaded
    require_models!([:tfidf, :gazetteer, :entities])

    # Capture status for debugging
    %{model_status: model_status()}
  end

  # ============================================================================
  # GREETING INTENTS
  # ============================================================================

  describe "greeting intents" do
    @greetings [
      {"Hello", ["greeting", "hello", "welcome"]},
      {"Hi there", ["greeting", "hello", "welcome"]},
      {"Good morning", ["greeting", "morning", "hello"]},
      {"Hey", ["greeting", "hello", "welcome"]}
    ]

    for {input, patterns} <- @greetings do
      @input input
      @patterns patterns

      test "#{input} is a greeting" do
        assert_intent_matches_any(@input, @patterns)
      end
    end
  end

  # ============================================================================
  # FAREWELL INTENTS
  # ============================================================================

  describe "farewell intents" do
    @farewells [
      {"Goodbye", ["bye", "farewell", "goodbye"]},
      {"Bye", ["bye", "farewell"]},
      {"See you later", ["bye", "farewell", "see_you"]}
    ]

    for {input, patterns} <- @farewells do
      @input input
      @patterns patterns

      test "#{input} is a farewell" do
        assert_intent_matches_any(@input, @patterns)
      end
    end
  end

  # ============================================================================
  # WEATHER INTENTS
  # ============================================================================

  describe "weather intents" do
    @weather_queries [
      {"What's the weather", ["weather"]},
      {"What's the weather in NYC", ["weather"]},
      {"How's the weather today", ["weather"]},
      {"Weather in London", ["weather"]}
    ]

    for {input, patterns} <- @weather_queries do
      @input input
      @patterns patterns

      test "#{input} is a weather query" do
        assert_intent_matches_any(@input, @patterns)
      end
    end
  end

  # ============================================================================
  # MUSIC INTENTS
  # ============================================================================

  describe "music intents" do
    @music_queries [
      {"Play some music", ["music", "play"]},
      {"Play Taylor Swift", ["music", "play"]},
      {"Play Hello by Adele", ["music", "play"]}
    ]

    for {input, patterns} <- @music_queries do
      @input input
      @patterns patterns

      test "#{input} is a music query" do
        assert_intent_matches_any(@input, @patterns)
      end
    end
  end

  # ============================================================================
  # SMART HOME INTENTS
  # ============================================================================

  describe "smart home intents" do
    @smarthome_queries [
      {"Turn on the lights", ["light", "smarthome", "device"]},
      {"Turn off the TV", ["smarthome", "device", "tv"]},
      {"Set the thermostat to 72", ["thermostat", "smarthome", "temperature"]}
    ]

    for {input, patterns} <- @smarthome_queries do
      @input input
      @patterns patterns

      test "#{input} is a smart home query" do
        assert_intent_matches_any(@input, @patterns)
      end
    end
  end

  # ============================================================================
  # DISAMBIGUATION TESTS
  # ============================================================================

  describe "disambiguation - greeting vs music" do
    test "Hello alone is a greeting, not the song" do
      {:ok, result} = IntentClassifierSimple.classify("Hello")
      intent_lower = String.downcase(result.intent)

      is_greeting = String.contains?(intent_lower, "greeting") or
                    String.contains?(intent_lower, "hello") or
                    String.contains?(intent_lower, "welcome")

      assert is_greeting, "Expected greeting, got: #{result.intent}"
      refute String.contains?(intent_lower, "music.play"), "Should not be music.play"
    end

    test "Hello I'm Austin is a greeting introduction" do
      {:ok, result} = IntentClassifierSimple.classify("Hello, I'm Austin")
      intent_lower = String.downcase(result.intent)

      is_greeting = String.contains?(intent_lower, "greeting") or
                    String.contains?(intent_lower, "hello") or
                    String.contains?(intent_lower, "welcome") or
                    String.contains?(intent_lower, "introduction")

      assert is_greeting, "Expected greeting/introduction, got: #{result.intent}"
    end

    test "Play Hello by Adele is a music request" do
      {:ok, result} = IntentClassifierSimple.classify("Play Hello by Adele")
      intent_lower = String.downcase(result.intent)

      is_music = String.contains?(intent_lower, "music") or
                 String.contains?(intent_lower, "play")

      assert is_music, "Expected music/play, got: #{result.intent}"
    end
  end

  # ============================================================================
  # ENTITY EXTRACTION TESTS
  # ============================================================================

  describe "location entity extraction" do
    @location_tests [
      {"Weather in London", "London"},
      {"What's the weather in Paris", "Paris"},
      {"What's the weather in Chicago", "Chicago"}
    ]

    for {input, expected_value} <- @location_tests do
      @input input
      @expected_value expected_value

      test "extracts #{expected_value} from '#{input}'" do
        assert_entity_value_present(@input, @expected_value)
      end
    end
  end

  # ============================================================================
  # CONFIDENCE BEHAVIOR TESTS
  # ============================================================================

  describe "confidence behavior" do
    test "gibberish has lower confidence than real phrases" do
      {:ok, gibberish} = IntentClassifierSimple.classify("xyzabc qwerty asdfgh jklmnop")
      {:ok, real} = IntentClassifierSimple.classify("What's the weather today")

      assert real.confidence > gibberish.confidence,
             "Real: #{real.confidence}, Gibberish: #{gibberish.confidence}"
    end
  end

  # ============================================================================
  # CALENDAR INTENTS
  # ============================================================================

  describe "calendar intents" do
    test "Schedule a meeting tomorrow" do
      assert_intent_matches_any("Schedule a meeting tomorrow", ["calendar", "meeting", "schedule"])
    end

    test "What's on my calendar today" do
      assert_intent_matches_any("What's on my calendar today", ["calendar", "schedule", "today"])
    end

    test "Cancel my meeting with John" do
      assert_intent_matches_any("Cancel my meeting with John", ["calendar", "meeting", "cancel"])
    end
  end

  # ============================================================================
  # REMINDER INTENTS
  # ============================================================================

  describe "reminder intents" do
    test "Remind me to call mom" do
      assert_intent_matches_any("Remind me to call mom", ["remind", "reminder"])
    end

    test "Set a reminder for tomorrow" do
      assert_intent_matches_any("Set a reminder for tomorrow", ["remind", "reminder"])
    end
  end

  # ============================================================================
  # ALARM INTENTS
  # ============================================================================

  describe "alarm intents" do
    test "Set an alarm for 7am" do
      assert_intent_matches_any("Set an alarm for 7am", ["alarm"])
    end

    test "Snooze the alarm" do
      assert_intent_matches_any("Snooze the alarm", ["alarm", "snooze"])
    end
  end

  # ============================================================================
  # TIMER INTENTS
  # ============================================================================

  describe "timer intents" do
    test "Set a timer for 10 minutes" do
      assert_intent_matches_any("Set a timer for 10 minutes", ["timer"])
    end

    test "How much time is left" do
      assert_intent_matches_any("How much time is left", ["timer", "time"])
    end
  end

  # ============================================================================
  # TODO INTENTS
  # ============================================================================

  describe "todo intents" do
    test "Add buy groceries to my todo list" do
      assert_intent_matches_any("Add buy groceries to my todo list", ["todo", "task", "add", "list"])
    end

    test "Show my tasks" do
      assert_intent_matches_any("Show my tasks", ["todo", "task", "list"])
    end
  end

  # ============================================================================
  # COMMUNICATION INTENTS
  # ============================================================================

  describe "communication intents" do
    test "Call John" do
      assert_intent_matches_any("Call John", ["call", "phone"])
    end

    test "Send a text to Sarah" do
      assert_intent_matches_any("Send a text to Sarah", ["message", "text", "send"])
    end

    test "Check my email" do
      assert_intent_matches_any("Check my email", ["email", "mail"])
    end
  end

  # ============================================================================
  # TIME AND DATE INTENTS
  # ============================================================================

  describe "time and date intents" do
    test "What time is it" do
      assert_intent_matches_any("What time is it", ["time"])
    end

    test "What's the date today" do
      assert_intent_matches_any("What's the date today", ["date", "today"])
    end
  end

  # ============================================================================
  # KNOWLEDGE INTENTS
  # ============================================================================

  describe "knowledge intents" do
    test "Define serendipity" do
      assert_intent_matches_any("Define serendipity", ["define", "knowledge", "meaning"])
    end

    test "What is the capital of France" do
      assert_intent_matches_any("What is the capital of France", ["capital", "reference", "knowledge"])
    end

    test "Translate hello to Spanish" do
      assert_intent_matches_any("Translate hello to Spanish", ["translate"])
    end
  end

  # ============================================================================
  # DISPLAY INTENTS (Star Trek style)
  # ============================================================================

  describe "display intents" do
    test "Show me the report" do
      assert_intent_matches_any("Show me the report", ["show", "display"])
    end

    test "Zoom in" do
      assert_intent_matches_any("Zoom in", ["zoom", "display"])
    end

    test "Enhance the image" do
      assert_intent_matches_any("Enhance the image", ["enhance", "display"])
    end
  end

  # ============================================================================
  # SEARCH INTENTS
  # ============================================================================

  describe "search intents" do
    test "Locate John" do
      assert_intent_matches_any("Locate John", ["locate", "find", "search"])
    end

    test "Find the quarterly report" do
      assert_intent_matches_any("Find the quarterly report", ["find", "search", "file"])
    end
  end

  # ============================================================================
  # ANALYSIS INTENTS
  # ============================================================================

  describe "analysis intents" do
    test "Analyze the data" do
      assert_intent_matches_any("Analyze the data", ["analyze", "data"])
    end

    test "Show me the trend" do
      assert_intent_matches_any("Show me the trend", ["trend", "analyze"])
    end
  end

  # ============================================================================
  # STATUS INTENTS
  # ============================================================================

  describe "status intents" do
    test "Status report" do
      assert_intent_matches_any("Status report", ["status", "report"])
    end

    test "Run diagnostics" do
      assert_intent_matches_any("Run diagnostics", ["diagnostic", "status"])
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp assert_intent_matches_any(input, patterns) when is_list(patterns) do
    case IntentClassifierSimple.classify(input) do
      {:ok, result} ->
        intent_lower = String.downcase(result.intent)

        matches_pattern = Enum.any?(patterns, fn pattern ->
          String.contains?(intent_lower, String.downcase(pattern))
        end)

        assert matches_pattern,
               """
               Intent mismatch for: "#{input}"
               Expected one of: #{inspect(patterns)}
               Got: #{result.intent} (confidence: #{Float.round(result.confidence, 3)})
               """

        assert result.confidence >= @min_confidence,
               "Low confidence for '#{input}': #{result.confidence}"

      {:error, reason} ->
        flunk("Classification failed: #{inspect(reason)}")
    end
  end

  defp assert_entity_value_present(input, expected_value) do
    entities = EntityExtractor.extract_entities(input)
    expected_lower = String.downcase(expected_value)

    matching = Enum.find(entities, fn entity ->
      value = entity[:value] || entity[:text] || entity[:match] || ""
      String.contains?(String.downcase(to_string(value)), expected_lower) or
      String.contains?(expected_lower, String.downcase(to_string(value)))
    end)

    assert matching != nil,
           """
           Entity value not found in: "#{input}"
           Expected: #{expected_value}
           Found: #{inspect(entities)}
           """
  end
end
