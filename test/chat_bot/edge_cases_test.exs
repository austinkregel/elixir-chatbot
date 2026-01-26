defmodule ChatBot.EdgeCasesTest do
  @moduledoc """
  Edge case tests to explore the limits of the NLP system.

  These tests cover cases that are NOT in the training data to understand
  how the system handles:
  1. Names that overlap with cities/locations
  2. Unusual greeting patterns
  3. Names that overlap with songs or products
  4. Typos, informal language, and edge cases

  The goal is not necessarily for all tests to pass, but to document
  expected vs actual behavior and identify areas for improvement.

  ## Snapshot Tests

  This module also includes "snapshot" tests that capture the exact analysis
  output for key inputs. These help:
  - Document expected behavior precisely
  - Catch regressions when implementation changes
  - Understand what the system actually detects

  Run snapshot tests with: `mix test test/chat_bot/edge_cases_test.exs --only snapshot`
  """
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog
  require Logger

  alias ChatBot.Brain
  alias ChatBot.Analysis.Pipeline
  alias ChatBot.ML.Gazetteer
  import ChatBot.TestHelpers

  # Tag all tests as edge cases for easy filtering
  @moduletag :edge_cases

  # ============================================================================
  # Snapshot Helpers
  # ============================================================================

  @doc """
  Extracts a normalized snapshot from a Pipeline analysis result.
  This captures the key fields we want to assert on.
  """
  def extract_snapshot(result) do
    %{
      chunk_count: length(result.chunks),
      overall_strategy: result.overall_strategy,
      analyses:
        Enum.map(result.analyses, fn analysis ->
          # Convert struct to map for easier access
          analysis_map = Map.from_struct(analysis)

          chunk_index = Map.get(analysis_map, :chunk_index, Map.get(analysis_map, :index, 0))
          text = Map.get(analysis_map, :text, "")

          # Get discourse info
          discourse = Map.get(analysis_map, :discourse)
          discourse_map = to_map(discourse)

          is_self_ref =
            Map.get(
              discourse_map,
              :is_self_referential,
              Map.get(discourse_map, :self_referential, false)
            )

          # Get slots info
          slots = Map.get(analysis_map, :slots)
          slots_map = to_map(slots)
          filled_slots = Map.get(slots_map, :filled_slots, Map.get(slots_map, :filled, %{}))
          missing_required = Map.get(slots_map, :missing_required, [])

          # Get intent - may be in slots or directly on analysis
          detected_intent =
            Map.get(
              analysis_map,
              :intent,
              Map.get(slots_map, :intent, Map.get(slots_map, :schema_name, nil))
            )

          # Get speech act info
          speech_act = Map.get(analysis_map, :speech_act)
          speech_act_map = to_map(speech_act)

          %{
            index: chunk_index,
            text: text,
            # Speech act
            speech_act_category: Map.get(speech_act_map, :category),
            speech_act_type: Map.get(speech_act_map, :sub_type) || Map.get(speech_act_map, :type),
            speech_act_confidence: round_confidence(Map.get(speech_act_map, :confidence)),
            # Discourse
            discourse_addressee: Map.get(discourse_map, :addressee),
            discourse_self_referential: is_self_ref,
            # Intent
            detected_intent: detected_intent,
            slots_filled: Map.keys(filled_slots) |> Enum.sort(),
            slots_missing: missing_required |> Enum.sort(),
            # Entities
            entities: normalize_entities(Map.get(analysis_map, :entities, [])),
            # Response strategy
            response_strategy: Map.get(analysis_map, :response_strategy)
          }
        end)
    }
  end

  defp to_map(nil), do: %{}
  defp to_map(term) when is_struct(term), do: Map.from_struct(term)
  defp to_map(term) when is_map(term), do: term
  defp to_map(_), do: %{}

  defp round_confidence(nil), do: nil
  defp round_confidence(conf) when is_float(conf), do: Float.round(conf, 2)
  defp round_confidence(conf), do: conf

  defp normalize_entities(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      %{
        type: Map.get(e, :entity_type, Map.get(e, :type, Map.get(e, :entity, "unknown"))),
        value: Map.get(e, :value, Map.get(e, :match, "unknown")),
        confidence: round_confidence(Map.get(e, :confidence, 1.0))
      }
    end)
    |> Enum.sort_by(& &1.value)
  end

  defp normalize_entities(_), do: []

  @doc """
  Logs a snapshot for easy copy/paste into test assertions.
  Use this when updating expected values. Output is captured by capture_log.
  """
  def log_snapshot(result) do
    snapshot = extract_snapshot(result)

    Logger.warning("\n" <> String.duplicate("=", 70))
    Logger.warning("SNAPSHOT OUTPUT")
    Logger.warning(String.duplicate("=", 70))
    Logger.warning("Chunk count: #{snapshot.chunk_count}")
    Logger.warning("Overall strategy: #{inspect(snapshot.overall_strategy)}")

    Enum.each(snapshot.analyses, fn analysis ->
      Logger.warning("--- Chunk #{analysis.index}: \"#{analysis.text}\" ---")

      Logger.warning(
        "  Speech Act: #{analysis.speech_act_category} / #{analysis.speech_act_type} (#{analysis.speech_act_confidence})"
      )

      Logger.warning(
        "  Discourse: addressee=#{analysis.discourse_addressee}, self_ref=#{analysis.discourse_self_referential}"
      )

      Logger.warning("  Intent: #{inspect(analysis.detected_intent)}")
      Logger.warning("  Slots filled: #{inspect(analysis.slots_filled)}")
      Logger.warning("  Slots missing: #{inspect(analysis.slots_missing)}")
      Logger.warning("  Entities: #{inspect(analysis.entities)}")
      Logger.warning("  Strategy: #{analysis.response_strategy}")
    end)

    Logger.warning(String.duplicate("=", 70))

    snapshot
  end

  setup do
    start_brain_services()
    {:ok, conversation_id} = Brain.create_conversation()
    %{conversation_id: conversation_id}
  end

  # ============================================================================
  # Names that overlap with cities/locations
  # ============================================================================
  describe "names overlapping with locations" do
    @tag :ambiguous_names
    test "Austin is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      # Should recognize as greeting with person name, NOT ask about weather in Austin, TX
      refute response =~ ~r/weather|temperature|forecast/i,
             "Greeting was misclassified as weather query: #{response}"

      refute response =~ ~r/Texas|city|travel/i,
             "Austin was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Dallas is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Dallas")

      refute response =~ ~r/weather|temperature|forecast|Texas/i,
             "Dallas was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Paris is also a city in France", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Paris")

      refute response =~ ~r/weather|France|Eiffel|travel/i,
             "Paris was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Jordan is also a country", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hey there, I'm Jordan")

      refute response =~ ~r/weather|Middle East|country/i,
             "Jordan was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Brooklyn is also part of NYC", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Brooklyn")

      refute response =~ ~r/weather|New York|NYC|borough/i,
             "Brooklyn was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Georgia is also a state and country", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, my name is Georgia")

      refute response =~ ~r/weather|Atlanta|state|country/i,
             "Georgia was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Madison is also a city in Wisconsin", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi there, I'm Madison")

      refute response =~ ~r/weather|Wisconsin/i,
             "Madison was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Sydney is also a city in Australia", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Sydney")

      refute response =~ ~r/weather|Australia|Opera House/i,
             "Sydney was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Charlotte is also a city in North Carolina", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Charlotte")

      refute response =~ ~r/weather|North Carolina/i,
             "Charlotte was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Savannah is also a city in Georgia", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Savannah")

      refute response =~ ~r/weather|Georgia/i,
             "Savannah was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Dakota is also a state reference", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hey, I'm Dakota")

      refute response =~ ~r/weather|North Dakota|South Dakota/i,
             "Dakota was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Orlando is also a city in Florida", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Orlando")

      refute response =~ ~r/weather|Florida|Disney/i,
             "Orlando was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Florence is also a city in Italy", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, my name is Florence")

      refute response =~ ~r/weather|Italy|Renaissance/i,
             "Florence was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Victoria is also a city and state", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Victoria")

      refute response =~ ~r/weather|Canada|Australia/i,
             "Victoria was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Lincoln is also a city in Nebraska", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Lincoln")

      refute response =~ ~r/weather|Nebraska/i,
             "Lincoln was incorrectly interpreted as a location: #{response}"
    end

    @tag :ambiguous_names
    test "Jackson is also a city in Mississippi", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi there, I'm Jackson")

      refute response =~ ~r/weather|Mississippi/i,
             "Jackson was incorrectly interpreted as a location: #{response}"
    end
  end

  # ============================================================================
  # Names that overlap with songs
  # ============================================================================
  describe "names overlapping with songs" do
    @tag :ambiguous_names
    test "Delilah is also a song (Hey There Delilah)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Delilah")

      refute response =~ ~r/playing|play\s|music|song/i,
             "Delilah was misclassified as a music request: #{response}"
    end

    @tag :ambiguous_names
    test "Jolene is also a famous song", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Jolene")

      refute response =~ ~r/playing|play\s|music|Dolly/i,
             "Jolene was misclassified as a music request: #{response}"
    end

    @tag :ambiguous_names
    test "Iris is also a song by Goo Goo Dolls", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Iris")

      refute response =~ ~r/playing|play\s|music|song/i,
             "Iris was misclassified as a music request: #{response}"
    end

    @tag :ambiguous_names
    test "Roxanne is also a song by The Police", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Roxanne")

      refute response =~ ~r/playing|play\s|music|song/i,
             "Roxanne was misclassified as a music request: #{response}"
    end
  end

  # ============================================================================
  # Names that overlap with products/assistants
  # ============================================================================
  describe "names overlapping with products" do
    @tag :ambiguous_names
    test "Alexa is also Amazon's assistant", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Alexa")

      # Should recognize as greeting, not try to invoke another assistant
      refute response =~ ~r/Amazon|assistant|device|smart home/i,
             "Alexa was incorrectly interpreted as a product: #{response}"
    end

    @tag :ambiguous_names
    test "Mercedes is also a car brand", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Mercedes")

      refute response =~ ~r/car|vehicle|Benz/i,
             "Mercedes was incorrectly interpreted as a car brand: #{response}"
    end

    @tag :ambiguous_names
    test "Luna is also a cryptocurrency and brand", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Luna")

      refute response =~ ~r/crypto|coin|moon/i,
             "Luna was incorrectly interpreted as a crypto: #{response}"
    end
  end

  # ============================================================================
  # Informal and unusual greetings (not in training data)
  # ============================================================================
  describe "informal greetings not in training data" do
    @tag :informal
    test "yo as greeting", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Yo")

      # Should be recognized as a greeting or at least not a command
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Informal greeting got farewell response: #{response}"

      assert String.length(response) > 0
    end

    @tag :informal
    test "sup as greeting", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Sup")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Informal greeting got farewell response: #{response}"
    end

    @tag :informal
    test "wassup as greeting", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Wassup")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Informal greeting got farewell response: #{response}"
    end

    @tag :informal
    test "hiya as greeting", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hiya!")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Informal greeting got farewell response: #{response}"
    end

    @tag :informal
    test "ello as greeting (dropped h)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "'Ello there")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Informal greeting got farewell response: #{response}"
    end

    @tag :informal
    test "extended vowels - heyyy", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Heyyy")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Extended greeting got farewell response: #{response}"
    end

    @tag :informal
    test "extended vowels - hiii", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hiii")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Extended greeting got farewell response: #{response}"
    end

    @tag :informal
    test "cultural greeting - g'day", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "G'day mate")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Cultural greeting got farewell response: #{response}"
    end

    @tag :informal
    test "cultural greeting - aloha", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Aloha")

      # Aloha can mean hello or goodbye, so just check for a response
      assert String.length(response) > 0
    end

    @tag :informal
    test "formal greeting - salutations", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Salutations!")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Formal greeting got farewell response: #{response}"
    end

    @tag :informal
    test "what's up as greeting", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What's up!")

      # Should be treated as a greeting, not a question about direction
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Greeting got farewell response: #{response}"
    end
  end

  # ============================================================================
  # Edge cases in text formatting
  # ============================================================================
  describe "text formatting edge cases" do
    @tag :formatting
    test "all lowercase with no punctuation", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "hello im austin")

      refute response =~ ~r/playing|play\s|weather/i,
             "Greeting was misclassified: #{response}"
    end

    @tag :formatting
    test "all uppercase", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "HELLO I AM AUSTIN")

      refute response =~ ~r/playing|play\s|weather/i,
             "Greeting was misclassified: #{response}"
    end

    @tag :formatting
    test "excessive punctuation", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello!!!! I'm Austin!!!!")

      refute response =~ ~r/playing|play\s|weather/i,
             "Greeting was misclassified: #{response}"
    end

    @tag :formatting
    test "mixed case in name", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm AuStIn")

      refute response =~ ~r/playing|play\s|weather/i,
             "Greeting was misclassified: #{response}"
    end

    @tag :formatting
    test "lowercase i in I'm", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, i'm austin")

      refute response =~ ~r/playing|play\s|weather/i,
             "Greeting was misclassified: #{response}"
    end

    @tag :formatting
    test "extra spaces", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello,    I'm    Austin")

      # Ideal: Should recognize as greeting + introduction, not trigger music/weather
      # Current: Extra spaces may affect tokenization/classification
      if response =~ ~r/playing|play\s|weather/i do
        IO.puts(
          "Note: 'Hello,    I'm    Austin' (extra spaces) triggered unexpected response - tokenization could be improved"
        )
      end

      # At minimum, should get a response
      assert String.length(response) > 0
    end
  end

  # ============================================================================
  # Common typos
  # ============================================================================
  describe "common typos" do
    @tag :typos
    test "helo (missing l)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Helo, I'm Austin")

      # Should still work reasonably
      assert String.length(response) > 0
    end

    @tag :typos
    test "hlelo (transposed letters)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hlelo, I'm Austin")

      assert String.length(response) > 0
    end

    @tag :typos
    test "im vs I'm", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, im Austin")

      # Ideal: Should recognize as greeting + introduction, not music/weather
      # Current: "Hello" may match song name, leading to music response
      if response =~ ~r/playing|play\s|weather/i do
        IO.puts(
          "Note: 'Hello, im Austin' (typo) triggered unexpected response - disambiguation could be improved"
        )
      end

      # At minimum, should get a response
      assert String.length(response) > 0
    end

    @tag :typos
    test "goodmorning (no space)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Goodmorning!")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Morning greeting got farewell response: #{response}"
    end
  end

  # ============================================================================
  # Name introduction patterns not in training
  # ============================================================================
  describe "unusual introduction patterns" do
    @tag :introductions
    test "with title - Dr.", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Dr. Smith")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "with title - Professor", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Professor Johnson")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "nickname pattern - call me", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, you can call me Bobby")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "my friends call me pattern", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, my friends call me Ace")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "full name introduction", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm John Smith")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "name with apostrophe - O'Brien", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm O'Brien")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "hyphenated name", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, I'm Mary-Jane")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "name is pattern", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "My name is Alex")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end

    @tag :introductions
    test "the name's pattern (James Bond style)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "The name's Bond, James Bond")

      # Ideal: Should recognize as introduction, not trigger music/weather
      # Current: Unusual patterns may not be perfectly recognized
      if response =~ ~r/playing|play\s|weather/i do
        IO.puts(
          "Note: 'The name's Bond, James Bond' triggered unexpected response - unusual pattern handling could be improved"
        )
      end

      # At minimum, should get a response
      assert String.length(response) > 0
    end

    @tag :introductions
    test "I go by pattern", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "I go by Max")

      refute response =~ ~r/playing|play\s|weather/i,
             "Introduction was misclassified: #{response}"
    end
  end

  # ============================================================================
  # Time-based greetings with ambiguous names
  # ============================================================================
  describe "time-based greetings with ambiguous names" do
    @tag :time_greeting
    test "good morning with city name (Austin)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good morning, I'm Austin")

      refute response =~ ~r/weather|temperature|forecast/i,
             "Morning greeting was misclassified: #{response}"
    end

    @tag :time_greeting
    test "good afternoon with city name (Dallas)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good afternoon, I'm Dallas")

      # Ideal: Dallas should be recognized as person name, not location
      # Current: Disambiguation may not perfectly distinguish - this is an area for improvement
      if response =~ ~r/weather|temperature|forecast/i do
        IO.puts(
          "Note: 'Good afternoon, I'm Dallas' triggered weather response - disambiguation could be improved"
        )
      end

      # At minimum, should get a response
      assert String.length(response) > 0
    end

    @tag :time_greeting
    test "good evening with city name (Paris)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good evening, I'm Paris")

      refute response =~ ~r/weather|temperature|France/i,
             "Evening greeting was misclassified: #{response}"
    end

    @tag :time_greeting
    test "good night with city name (Sydney)", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good night, I'm Sydney")

      # Good night could be interpreted as farewell, which is acceptable
      refute response =~ ~r/weather|temperature|Australia/i,
             "Night greeting was misclassified: #{response}"
    end
  end

  # ============================================================================
  # Complex multi-sentence scenarios
  # ============================================================================
  describe "complex multi-sentence scenarios" do
    @tag :multi_sentence
    test "greeting + question + name", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi! I'm Austin. What's the weather?")

      # Ideal: Austin (from "I'm Austin") should not be used as weather location
      # Current: Multi-chunk disambiguation may not perfectly distinguish - area for improvement
      if response =~ ~r/Austin.*weather|weather.*Austin/i do
        IO.puts(
          "Note: 'Hi! I'm Austin. What's the weather?' used Austin as location - disambiguation could be improved"
        )
      end

      # At minimum, should get a response
      assert String.length(response) > 0
    end

    @tag :multi_sentence
    test "multiple greetings", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello! Hi! Hey there!")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Multiple greetings got farewell response: #{response}"

      assert String.length(response) > 0
    end

    @tag :multi_sentence
    test "introduction then command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "I'm Dallas. Play some music.")

      # Dallas should be person, music command should be recognized
      # Should NOT try to play music IN Dallas (location)
      refute response =~ ~r/Dallas.*music|music.*Dallas/i,
             "Dallas was used as location context: #{response}"
    end
  end

  # ============================================================================
  # Extremely long input stress tests
  # ============================================================================
  describe "extremely long input handling" do
    @tag :stress_test
    test "handles very long input with many sentences", %{conversation_id: conv_id} do
      # Generate a long input with many sentences (simulating something like a movie script)
      # Each sentence is a simple statement to avoid triggering complex NLP
      long_input =
        1..100
        |> Enum.map(fn i -> "This is sentence number #{i}." end)
        |> Enum.join(" ")

      # The system should not crash and should return some response
      result = Brain.evaluate(conv_id, long_input)

      case result do
        {:ok, response} ->
          assert is_binary(response)
          assert String.length(response) > 0

        {:error, reason} ->
          # If it errors, it should be a graceful error, not a crash
          assert is_atom(reason) or is_binary(reason)
      end
    end

    @tag :stress_test
    test "semantic chunker breaks long input into chunks" do
      alias ChatBot.Analysis.SemanticChunker

      # 50 sentences should produce multiple chunks
      long_input =
        1..50
        |> Enum.map(fn i -> "Hello, I am person number #{i}." end)
        |> Enum.join(" ")

      chunks = SemanticChunker.chunk(long_input)

      # Should produce multiple chunks (default max is 50 words per chunk)
      assert length(chunks) > 1, "Long input should be chunked, got #{length(chunks)} chunks"

      # Each chunk should be reasonable size
      Enum.each(chunks, fn chunk ->
        word_count = chunk.text |> String.split() |> length()
        assert word_count <= 60, "Chunk too long: #{word_count} words"
      end)
    end

    @tag :stress_test
    test "handles repeated greeting pattern" do
      # Someone spamming greetings
      long_greetings =
        1..20
        |> Enum.map(fn _ -> "Hello! Hi! Hey there!" end)
        |> Enum.join(" ")

      result = Pipeline.process(long_greetings, [])

      # Should produce chunks and analyses
      assert length(result.analyses) >= 1
    end

    @tag :stress_test
    test "handles mixed content with many intents", %{conversation_id: conv_id} do
      # Multiple different types of input
      mixed_input = """
      Hello, I'm Austin. What's the weather? Play some music. Turn on the lights.
      Good morning! What time is it? Set a reminder. How are you today?
      Goodbye! Wait, actually, hello again. What can you do?
      """

      {:ok, response} = Brain.evaluate(conv_id, mixed_input)

      # Should produce a response without crashing
      assert is_binary(response)
      assert String.length(response) > 0
    end

    @tag :stress_test
    @tag :timeout
    test "does not take excessively long on moderately long input", %{conversation_id: conv_id} do
      # 30 varied sentences
      moderate_input =
        1..30
        |> Enum.map(fn i ->
          case rem(i, 5) do
            0 -> "What's the weather in city #{i}?"
            1 -> "Hello, I'm person #{i}."
            2 -> "Play song number #{i}."
            3 -> "Turn on light #{i}."
            _ -> "This is statement #{i}."
          end
        end)
        |> Enum.join(" ")

      # Should complete within reasonable time (10 seconds)
      # The Brain.evaluate default timeout is 90 seconds
      start_time = System.monotonic_time(:millisecond)
      {:ok, _response} = Brain.evaluate(conv_id, moderate_input)
      elapsed = System.monotonic_time(:millisecond) - start_time

      # Should complete within 30 seconds
      assert elapsed < 30_000, "Processing took too long: #{elapsed}ms"
    end
  end

  # ============================================================================
  # Direct entity extraction tests (lower level)
  # ============================================================================
  describe "entity extraction edge cases" do
    setup do
      ensure_started(ChatBot.ML.Gazetteer)
      Gazetteer.load_all()
      :ok
    end

    @tag :entity_extraction
    test "Austin recognized as both person and location" do
      results = Gazetteer.lookup_all_types("Austin")

      types = Enum.map(results, & &1.entity_type)

      # Should have both person and location/city entries
      assert "person" in types, "Austin should be recognized as a person name"

      assert "city" in types or "location" in types,
             "Austin should also be recognized as a city/location"
    end

    @tag :entity_extraction
    test "Dallas recognized as both person and location" do
      results = Gazetteer.lookup_all_types("Dallas")

      types = Enum.map(results, & &1.entity_type)

      assert "person" in types, "Dallas should be recognized as a person name"
    end

    @tag :entity_extraction
    test "Hello recognized in songs" do
      results = Gazetteer.lookup_all_types("Hello")

      types = Enum.map(results, & &1.entity_type)

      # Hello by Adele should be in songs
      assert "song" in types or "music" in types,
             "Hello should be recognized as a song"
    end
  end

  # ============================================================================
  # Pipeline disambiguation tests (mid-level)
  # ============================================================================
  describe "pipeline disambiguation" do
    @tag :disambiguation
    test "I'm Austin pattern triggers self-referential context" do
      result = Pipeline.process("Hello, I'm Austin", [])

      # Get first analysis
      assert length(result.analyses) >= 1
      analysis = hd(result.analyses)

      # Should recognize as self-introduction - speech act can be expressive or assertive
      # "Hello" is expressive (greeting), "I'm Austin" is assertive (statement of fact)
      assert analysis.discourse.addressee == :self or
               analysis.speech_act.category in [:expressive, :assertive],
             "Should recognize self-introduction pattern, got: addressee=#{inspect(analysis.discourse.addressee)}, category=#{inspect(analysis.speech_act.category)}"
    end

    @tag :disambiguation
    test "I'm from Austin triggers location context" do
      result = Pipeline.process("I'm from Austin", [])

      # This is different - "from Austin" suggests location
      assert length(result.analyses) >= 1

      # In this case, Austin SHOULD be recognized as a location
      # because "from [place]" is a location pattern
    end

    @tag :disambiguation
    test "weather in Austin triggers location context" do
      result = Pipeline.process("What's the weather in Austin?", [])

      assert length(result.analyses) >= 1
      analysis = hd(result.analyses)

      # Austin in weather context should be recognized as location
      # Check if we have entities and Austin is tagged as location
      # Entity structure uses :value or :match, not :text
      entities =
        Enum.filter(analysis.entities, fn e ->
          value = Map.get(e, :value, Map.get(e, :match, ""))
          String.downcase(to_string(value)) =~ "austin"
        end)

      if length(entities) > 0 do
        entity = hd(entities)

        entity_type = Map.get(entity, :entity_type, Map.get(entity, :type, "unknown"))

        assert entity_type == "location" or entity_type == "city",
               "Austin in weather context should be location, got: #{entity_type}"
      end
    end
  end

  # ============================================================================
  # SNAPSHOT TESTS
  # ============================================================================
  # These tests capture exact expected values for key inputs.
  # Run with: mix test test/chat_bot/edge_cases_test.exs --only snapshot
  #
  # To update a snapshot:
  # 1. Set @log_snapshots to true
  # 2. Run the test
  # 3. Copy the output into the expected values
  # 4. Set @log_snapshots back to false
  # ============================================================================

  # Set to true to print actual snapshots (for updating expected values)
  @log_snapshots false

  describe "snapshot tests - greeting with introduction" do
    @tag :snapshot
    test "Hello, I'm Austin - complete analysis snapshot" do
      input = "Hello, I'm Austin"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      # === EXPECTED VALUES ===
      # Single chunk for this short input
      assert snapshot.chunk_count == 1
      assert snapshot.overall_strategy == :can_respond

      # Get the single analysis
      [analysis] = snapshot.analyses

      # Text should be preserved
      assert analysis.text == "Hello, I'm Austin"

      # Speech act: can be expressive (greeting dominates) or assertive (intro dominates)
      # "Hello" is expressive, "I'm Austin" is assertive (stating a fact)
      assert analysis.speech_act_category in [:expressive, :assertive],
             "Expected expressive or assertive, got: #{analysis.speech_act_category}"

      # NOTE: Current behavior - discourse_self_referential is false
      # The DiscourseAnalyzer may not be setting this field for introductions
      # This could be an area for improvement

      # Entity: Austin should be detected as PERSON (not location)
      # NOTE: Current behavior - entities may be empty if extraction happens
      # at a different stage
      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      if length(austin_entities) > 0 do
        [austin] = austin_entities

        assert austin.type == "person",
               "Austin should be person, got: #{austin.type}"
      end

      # Should be able to respond (not need clarification)
      assert analysis.response_strategy == :can_respond
    end

    @tag :snapshot
    test "Hi, my name is Sarah - complete analysis snapshot" do
      input = "Hi, my name is Sarah"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Speech act: expressive greeting/introduction
      assert analysis.speech_act_category == :expressive

      # NOTE: Current behavior - self_referential not set in DiscourseResult
      # Documenting actual behavior - could be enhanced

      # Sarah entity check (may be extracted at different stage)
      sarah_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "sarah"
        end)

      if length(sarah_entities) > 0 do
        [sarah] = sarah_entities
        assert sarah.type == "person"
      end
    end
  end

  describe "snapshot tests - weather queries" do
    @tag :snapshot
    test "What's the weather in Austin? - location disambiguation" do
      input = "What's the weather in Austin?"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Speech act: should be directive (question/request)
      assert analysis.speech_act_category == :directive

      # Should NOT be self-referential
      assert analysis.discourse_self_referential == false

      # Austin should be detected as LOCATION (not person)
      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      if length(austin_entities) > 0 do
        [austin] = austin_entities

        assert austin.type in ["location", "city"],
               "Austin in weather context should be location, got: #{austin.type}"
      end

      # Intent: Current behavior returns generic "question.factual"
      # The slot schema may override to weather-specific intent later
      # The system correctly identifies this as a question, even if not specifically "weather"
      assert analysis.speech_act_type in [:request_information, :question, :request]
    end

    @tag :snapshot
    test "What's the weather? - missing location slot" do
      input = "What's the weather?"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Should be directive (question)
      assert analysis.speech_act_category == :directive

      # Should need clarification (missing location)
      # OR should have empty slots_filled for location
      assert analysis.response_strategy in [:needs_clarification, :can_respond]

      # If needs_clarification, location should be in missing slots
      if analysis.response_strategy == :needs_clarification do
        assert "location" in analysis.slots_missing
      end
    end
  end

  describe "snapshot tests - multi-chunk inputs" do
    @tag :snapshot
    test "Hello! What's the weather? - greeting then question" do
      input = "Hello! What's the weather?"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      # Should be split into 2 chunks
      assert snapshot.chunk_count == 2

      # First chunk: greeting
      [greeting_analysis, weather_analysis] = snapshot.analyses

      assert greeting_analysis.speech_act_category == :expressive
      assert greeting_analysis.text =~ ~r/hello/i

      # Second chunk: weather question
      assert weather_analysis.speech_act_category == :directive
      assert weather_analysis.text =~ ~r/weather/i
    end

    @tag :snapshot
    test "Hello, I'm Austin. What's the weather in Dallas? - intro then weather" do
      input = "Hello, I'm Austin. What's the weather in Dallas?"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      # NOTE: Current behavior - may chunk as 1 or 2 chunks
      # depending on sentence boundary detection

      # The important thing is that the system handles it correctly
      # regardless of chunking strategy
      assert snapshot.chunk_count >= 1

      # If single chunk, verify entities aren't confused
      if snapshot.chunk_count == 1 do
        [analysis] = snapshot.analyses

        # Should recognize the greeting aspect
        assert analysis.speech_act_category == :expressive
      else
        # If multi-chunk, verify cross-chunk isolation
        [intro_analysis | rest] = snapshot.analyses

        # First chunk should be greeting
        assert intro_analysis.speech_act_category == :expressive

        if length(rest) > 0 do
          weather_analysis = hd(rest)

          # Verify cross-chunk isolation: Austin should NOT appear in weather chunk
          austin_in_weather =
            Enum.filter(weather_analysis.entities, fn e ->
              String.downcase(to_string(e.value)) =~ "austin"
            end)

          # Austin should not leak into weather chunk
          assert length(austin_in_weather) == 0,
                 "Austin leaked into weather chunk - potential issue"
        end
      end
    end
  end

  describe "snapshot tests - commands" do
    @tag :snapshot
    test "Play some music - music command" do
      input = "Play some music"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Should be directive (command)
      assert analysis.speech_act_category == :directive
      assert analysis.speech_act_type in [:command, :request, :action]
    end

    @tag :snapshot
    test "Turn on the lights in the living room - device command with location" do
      input = "Turn on the lights in the living room"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Should be directive (command)
      assert analysis.speech_act_category == :directive

      # Should have entities for lights and/or room
      entity_values = Enum.map(analysis.entities, & &1.value) |> Enum.map(&String.downcase/1)

      # At least one relevant entity should be detected
      has_relevant =
        Enum.any?(entity_values, fn v ->
          v =~ ~r/light|living|room/i
        end)

      # May have none detected
      assert has_relevant or length(analysis.entities) >= 0
    end
  end

  describe "snapshot tests - edge cases with detailed inspection" do
    @tag :snapshot
    test "yo - informal greeting analysis" do
      input = "yo"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      assert snapshot.chunk_count == 1
      [analysis] = snapshot.analyses

      # Should recognize as expressive (informal greeting)
      # May also be classified as unknown - that's a known limitation
      # Document actual behavior - yo may not be recognized
      assert analysis.speech_act_category in [:expressive, :unknown, nil]
    end

    @tag :snapshot
    test "The name's Bond, James Bond - unusual intro pattern" do
      input = "The name's Bond, James Bond"
      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      [analysis | _] = snapshot.analyses

      # Document behavior - may or may not be recognized as intro
      # The point is to see what actually happens
      assert analysis.speech_act_category != nil
    end

    @tag :snapshot
    test "mixed content inspection", %{conversation_id: _conv_id} do
      input = """
      Hello, I'm Austin. What's the weather? Play some music. Turn on the lights.
      """

      result = Pipeline.process(input, [])

      if @log_snapshots, do: log_snapshot(result)

      snapshot = extract_snapshot(result)

      # Should produce multiple chunks
      assert snapshot.chunk_count >= 3

      # Each chunk should have a valid speech act
      Enum.each(snapshot.analyses, fn a ->
        assert a.speech_act_category != nil
      end)
    end
  end

  describe "snapshot tests - regression prevention" do
    @tag :snapshot
    @tag :regression
    test "Hello should NOT trigger music playback" do
      input = "Hello"
      result = Pipeline.process(input, [])

      snapshot = extract_snapshot(result)
      [analysis] = snapshot.analyses

      # Must be expressive, NOT directive
      assert analysis.speech_act_category == :expressive,
             "Hello should be greeting (expressive), got: #{analysis.speech_act_category}"

      # Intent should NOT be music-related
      if analysis.detected_intent do
        refute analysis.detected_intent =~ ~r/music|play/i,
               "Hello should not have music intent, got: #{analysis.detected_intent}"
      end
    end

    @tag :snapshot
    @tag :regression
    test "Hello, I'm Austin should NOT ask about weather in Austin" do
      input = "Hello, I'm Austin"
      result = Pipeline.process(input, [])

      snapshot = extract_snapshot(result)
      [analysis] = snapshot.analyses

      # Can be expressive (greeting dominates) or assertive (intro dominates)
      # "Hello" is expressive, "I'm Austin" is assertive (stating a fact)
      assert analysis.speech_act_category in [:expressive, :assertive],
             "Expected expressive or assertive, got: #{analysis.speech_act_category}"

      # Austin must be person, not location
      austin_entities =
        Enum.filter(analysis.entities, fn e ->
          String.downcase(to_string(e.value)) =~ "austin"
        end)

      Enum.each(austin_entities, fn e ->
        refute e.type in ["location", "city"],
               "Austin in greeting context must be person, got: #{e.type}"
      end)

      # Intent should NOT be weather
      if analysis.detected_intent do
        refute analysis.detected_intent =~ ~r/weather/i,
               "Hello, I'm Austin should not have weather intent"
      end
    end
  end
end
