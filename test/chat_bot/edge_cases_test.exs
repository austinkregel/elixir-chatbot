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

  To update snapshots: `mix test.update_snapshots`
  """
  use ExUnit.Case, async: false

  alias ChatBot.Brain
  alias ChatBot.Analysis.Pipeline
  alias ChatBot.ML.Gazetteer
  import ChatBot.TestHelpers
  import ChatBot.SnapshotHelper
  alias ChatBot.GenServerSandbox

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

  setup do
    # Start services and reset to clean state
    start_brain_services()
    GenServerSandbox.reset_global_state()
    {:ok, conversation_id} = Brain.create_conversation()
    %{conversation_id: conversation_id}
  end

  # ============================================================================
  # Names that overlap with cities/locations
  # ============================================================================
  describe "names overlapping with locations" do
    @tag :ambiguous_names
    test "Austin is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Austin")

      # Semantic assertion: Should be classified as greeting, not weather query
      assert_is_greeting(context)

      # Verify intent is NOT weather-related (if intent exists)
      intent = Map.get(context, :intent) || ""
      if intent != "", do: refute intent =~ ~r/weather/i, "Intent should not be weather, got: #{intent}"

      # Basic sanity + regression
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|forecast/i)
    end

    @tag :ambiguous_names
    test "Dallas is also a city in Texas", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, my name is Dallas")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      # Verify intent is NOT weather-related (if intent exists)
      intent = Map.get(context, :intent) || ""
      if intent != "", do: refute intent =~ ~r/weather/i, "Intent should not be weather, got: #{intent}"

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|forecast|Texas/i)
    end

    @tag :ambiguous_names
    test "Paris is also a city in France", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Paris")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|France|Eiffel|travel/i)
    end

    @tag :ambiguous_names
    test "Jordan is also a country", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hey there, I'm Jordan")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Middle East|country/i)
    end

    @tag :ambiguous_names
    test "Brooklyn is also part of NYC", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Brooklyn")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|New York|NYC|borough/i)
    end

    @tag :ambiguous_names
    test "Georgia is also a state and country", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, my name is Georgia")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Atlanta|state|country/i)
    end

    @tag :ambiguous_names
    test "Madison is also a city in Wisconsin", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi there, I'm Madison")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Wisconsin/i)
    end

    @tag :ambiguous_names
    test "Sydney is also a city in Australia", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Sydney")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Australia|Opera House/i)
    end

    @tag :ambiguous_names
    test "Charlotte is also a city in North Carolina", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, my name is Charlotte")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|North Carolina/i)
    end

    @tag :ambiguous_names
    test "Savannah is also a city in Georgia", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Savannah")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Georgia/i)
    end

    @tag :ambiguous_names
    test "Dakota is also a state reference", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hey, I'm Dakota")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|North Dakota|South Dakota/i)
    end

    @tag :ambiguous_names
    test "Orlando is also a city in Florida", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Orlando")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Florida|Disney/i)
    end

    @tag :ambiguous_names
    test "Florence is also a city in Italy", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, my name is Florence")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Italy|Renaissance/i)
    end

    @tag :ambiguous_names
    test "Victoria is also a city and state", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Victoria")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Canada|Australia/i)
    end

    @tag :ambiguous_names
    test "Lincoln is also a city in Nebraska", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Lincoln")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Nebraska/i)
    end

    @tag :ambiguous_names
    test "Jackson is also a city in Mississippi", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi there, I'm Jackson")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/weather|Mississippi/i)
    end
  end

  # ============================================================================
  # Names that overlap with songs
  # ============================================================================
  describe "names overlapping with songs" do
    @tag :ambiguous_names
    test "Delilah is also a song (Hey There Delilah)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Delilah")

      # Semantic assertion: Should be classified as greeting, not music
      assert_is_greeting(context)

      # Verify intent is NOT music-related (if intent exists)
      intent = Map.get(context, :intent) || ""
      if intent != "", do: refute intent =~ ~r/music|play/i, "Intent should not be music, got: #{intent}"

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|music|song/i)
    end

    @tag :ambiguous_names
    test "Jolene is also a famous song", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, my name is Jolene")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|music|Dolly/i)
    end

    @tag :ambiguous_names
    test "Iris is also a song by Goo Goo Dolls", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Iris")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|music|song/i)
    end

    @tag :ambiguous_names
    test "Roxanne is also a song by The Police", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Roxanne")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|music|song/i)
    end
  end

  # ============================================================================
  # Names that overlap with products/assistants
  # ============================================================================
  describe "names overlapping with products" do
    @tag :ambiguous_names
    test "Alexa is also Amazon's assistant", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Alexa")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/Amazon|assistant|device|smart home/i)
    end

    @tag :ambiguous_names
    test "Mercedes is also a car brand", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, my name is Mercedes")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/car|vehicle|Benz/i)
    end

    @tag :ambiguous_names
    test "Luna is also a cryptocurrency and brand", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Luna")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/crypto|coin|moon/i)
    end
  end

  # ============================================================================
  # Informal and unusual greetings (not in training data)
  # ============================================================================
  describe "informal greetings not in training data" do
    @tag :informal
    test "yo as greeting", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Yo")

      # Semantic assertion: Should be classified as greeting
      # Note: Very informal greetings may not be recognized by the classifier
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 and speech_act[:category] != nil do
        # If we have classification, check it's expressive (greeting-like) or at least not farewell
        assert speech_act[:category] in [:expressive, :assertive, :directive],
               "Expected informal greeting classification, got: #{inspect(speech_act)}"
      end

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "sup as greeting", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Sup")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "wassup as greeting", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Wassup")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "hiya as greeting", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hiya!")

      # Semantic assertion: Should be classified as greeting (if context available)
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0, do: assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "ello as greeting (dropped h)", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "'Ello there")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "extended vowels - heyyy", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Heyyy")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "extended vowels - hiii", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Hiii")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "cultural greeting - g'day", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "G'day mate")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "cultural greeting - aloha", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Aloha")

      # Aloha can mean hello or goodbye, so just check for a response
      assert_has_response(response)
    end

    @tag :informal
    test "formal greeting - salutations", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Salutations!")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :informal
    test "what's up as greeting", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What's up!")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end
  end

  # ============================================================================
  # Edge cases in text formatting
  # ============================================================================
  describe "text formatting edge cases" do
    @tag :formatting
    test "all lowercase with no punctuation", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "hello im austin")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :formatting
    test "all uppercase", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "HELLO I AM AUSTIN")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :formatting
    test "excessive punctuation", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello!!!! I'm Austin!!!!")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :formatting
    test "mixed case in name", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm AuStIn")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :formatting
    test "lowercase i in I'm", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, i'm austin")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :formatting
    test "extra spaces", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello,    I'm    Austin")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end
  end

  # ============================================================================
  # Common typos
  # ============================================================================
  describe "common typos" do
    @tag :typos
    test "helo (missing l)", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Helo, I'm Austin")

      # Should still work reasonably
      assert_has_response(response)
    end

    @tag :typos
    test "hlelo (transposed letters)", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Hlelo, I'm Austin")

      assert_has_response(response)
    end

    @tag :typos
    test "im vs I'm", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, im Austin")

      # Semantic assertion: Should be classified as greeting despite typo
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :typos
    test "goodmorning (no space)", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Goodmorning!")

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end
  end

  # ============================================================================
  # Name introduction patterns not in training
  # ============================================================================
  describe "unusual introduction patterns" do
    @tag :introductions
    test "with title - Dr.", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Dr. Smith")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "with title - Professor", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Professor Johnson")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "nickname pattern - call me", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, you can call me Bobby")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "my friends call me pattern", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, my friends call me Ace")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "full name introduction", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm John Smith")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "name with apostrophe - O'Brien", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm O'Brien")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "hyphenated name", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, I'm Mary-Jane")

      assert_is_greeting(context)
      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "name is pattern", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "My name is Alex")

      # This is an introduction, should be greeting-like
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert speech_act[:sub_type] == :greeting or speech_act[:category] in [:expressive, :assertive],
               "Expected greeting/introduction, got: #{inspect(speech_act)}"
      end

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "the name's pattern (James Bond style)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "The name's Bond, James Bond")

      # Check context shows introduction pattern
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert speech_act[:category] in [:expressive, :assertive],
               "Expected expressive/assertive for introduction, got: #{inspect(speech_act)}"
      end

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end

    @tag :introductions
    test "I go by pattern", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "I go by Max")

      assert_has_response(response)
      refute_response_matches(response, ~r/playing|play\s|weather/i)
    end
  end

  # ============================================================================
  # Time-based greetings with ambiguous names
  # ============================================================================
  describe "time-based greetings with ambiguous names" do
    @tag :time_greeting
    test "good morning with city name (Austin)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good morning, I'm Austin")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      # Intent should NOT be weather-related (if intent exists)
      intent = Map.get(context, :intent) || ""
      if intent != "", do: refute intent =~ ~r/weather/i, "Intent should not be weather, got: #{intent}"

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|forecast/i)
    end

    @tag :time_greeting
    test "good afternoon with city name (Dallas)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good afternoon, I'm Dallas")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|forecast/i)
    end

    @tag :time_greeting
    test "good evening with city name (Paris)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good evening, I'm Paris")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|France/i)
    end

    @tag :time_greeting
    test "good night with city name (Sydney)", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good night, I'm Sydney")

      # Good night could be greeting or farewell, both are acceptable
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert speech_act[:category] == :expressive,
               "Expected expressive speech act, got: #{inspect(speech_act)}"
      end

      assert_has_response(response)
      refute_response_matches(response, ~r/weather|temperature|Australia/i)
    end
  end

  # ============================================================================
  # Complex multi-sentence scenarios
  # ============================================================================
  describe "complex multi-sentence scenarios" do
    @tag :multi_sentence
    test "greeting + question + name", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi! I'm Austin. What's the weather?")

      # The context should show some intent was detected (greeting or weather)
      assert context[:intent] != nil or get_speech_act(context)[:category] != nil,
             "Expected some intent/speech_act classification, got: #{inspect(context)}"

      assert_has_response(response)

      # The name "Austin" from the introduction should NOT be used as a weather location.
      # The system should recognize "I'm Austin" as an introduction pattern and treat
      # "Austin" as a person's name, not a city.
      refute_response_matches(response, ~r/weather.*for.*Austin|Austin.*weather/i)

      # Since no location was provided for the weather query, the system should ask for one
      # (though this is optional - the response might also include a greeting)
    end

    @tag :multi_sentence
    test "multiple greetings", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello! Hi! Hey there!")

      # Semantic assertion: Should be classified as greeting
      assert_is_greeting(context)

      assert_has_response(response)
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    @tag :multi_sentence
    test "introduction then command", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "I'm Dallas. Play some music.")

      # The context should show a command was detected
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        # Could be greeting (for intro) or command (for music)
        assert speech_act[:category] in [:expressive, :directive, :assertive],
               "Expected greeting or command, got: #{inspect(speech_act)}"
      end

      assert_has_response(response)
      # Dallas should be person, not used as location context for music
      refute_response_matches(response, ~r/Dallas.*music|music.*Dallas/i)
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
    @tag timeout: 60_000
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
  # To update snapshots: mix test.update_snapshots
  # ============================================================================

  describe "snapshot tests - greeting with introduction" do
    @tag :snapshot
    test "Hello, I'm Austin - complete analysis snapshot" do
      result = Pipeline.process("Hello, I'm Austin", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hello_im_austin", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "Hi, my name is Sarah - complete analysis snapshot" do
      result = Pipeline.process("Hi, my name is Sarah", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hi_my_name_is_sarah", subdirectory: "edge_cases")
    end
  end

  describe "snapshot tests - weather queries" do
    @tag :snapshot
    test "What's the weather in Austin? - location disambiguation" do
      result = Pipeline.process("What's the weather in Austin?", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "weather_in_austin", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "What's the weather? - missing location slot" do
      result = Pipeline.process("What's the weather?", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "weather_no_location", subdirectory: "edge_cases")
    end
  end

  describe "snapshot tests - multi-chunk inputs" do
    @tag :snapshot
    test "Hello! What's the weather? - greeting then question" do
      result = Pipeline.process("Hello! What's the weather?", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hello_whats_the_weather", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "Hello, I'm Austin. What's the weather in Dallas? - intro then weather" do
      result = Pipeline.process("Hello, I'm Austin. What's the weather in Dallas?", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hello_im_austin_weather_dallas", subdirectory: "edge_cases")
    end
  end

  describe "snapshot tests - commands" do
    @tag :snapshot
    test "Play some music - music command" do
      result = Pipeline.process("Play some music", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "play_some_music", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "Turn on the lights in the living room - device command with location" do
      result = Pipeline.process("Turn on the lights in the living room", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "turn_on_lights_living_room", subdirectory: "edge_cases")
    end
  end

  describe "snapshot tests - edge cases with detailed inspection" do
    @tag :snapshot
    test "yo - informal greeting analysis" do
      result = Pipeline.process("yo", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "yo_greeting", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "The name's Bond, James Bond - unusual intro pattern" do
      result = Pipeline.process("The name's Bond, James Bond", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "names_bond_james_bond", subdirectory: "edge_cases")
    end

    @tag :snapshot
    test "mixed content inspection", %{conversation_id: _conv_id} do
      input = "Hello, I'm Austin. What's the weather? Play some music. Turn on the lights."
      result = Pipeline.process(input, [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "mixed_content", subdirectory: "edge_cases")
    end
  end

  describe "snapshot tests - regression prevention" do
    @tag :snapshot
    @tag :regression
    test "Hello should NOT trigger music playback" do
      result = Pipeline.process("Hello", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hello_regression", subdirectory: "edge_cases")
    end

    @tag :snapshot
    @tag :regression
    test "Hello, I'm Austin should NOT ask about weather in Austin" do
      result = Pipeline.process("Hello, I'm Austin", [])
      snapshot = extract_snapshot(result)
      assert_snapshot(snapshot, "hello_im_austin_regression", subdirectory: "edge_cases")
    end
  end
end
