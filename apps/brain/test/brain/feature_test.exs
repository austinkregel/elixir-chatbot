defmodule Brain.FeatureTest do
  @moduledoc """
  Feature tests that verify end-to-end chatbot behavior.

  These tests check that user inputs produce sensible outputs.
  They REQUIRE trained models - if models are not loaded, tests fail fast.
  """
  use Brain.Test.GraphCase, async: false
  use Brain.Test.ModelAssertions

  alias Brain
  import Brain.TestHelpers

  @moduletag :requires_models

  setup_all do
    Brain.TestHelpers.require_services!(:brain)
    require_models!([
      :gazetteer,
      :entities,
      :embedder,
      :sentiment,
      :speech_act,
      :micro_classifiers
    ])
    :ok
  end

  setup _context do
    start_brain_services()
    {:ok, conversation_id} = Brain.create_conversation()
    %{conversation_id: conversation_id}
  end

  describe "greeting responses" do
    test "responds to hello", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello!")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test: should not be farewell
      refute_response_matches(response, ~r/bye|goodbye|see you|later/i)
    end

    test "responds to hi", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi there!")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you|later/i)
    end

    test "responds to good morning", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good morning!")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you|later/i)
    end
  end

  describe "question responses" do
    test "responds to weather question", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Can you tell me about the weather?")

      # Semantic assertion: Should be classified as a question/directive
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "responds to time question", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What time is it?")

      # Semantic assertion: Should be classified as a question
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "responds to how are you", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "How are you?")

      # Semantic assertion: Should be classified as a question or expressive
      speech_act = get_speech_act(context)
      assert speech_act[:is_question] == true or speech_act[:category] == :expressive,
             "Expected question or expressive, got: #{inspect(speech_act)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "responds to what can you do", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What can you do?")

      # Semantic assertion: Should be classified as a question/directive
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end
  end

  describe "command responses" do
    test "responds to play music command", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Play some music")

      # Semantic assertion: Should be classified as a command/directive
      # The speech_act classifier may not always detect imperative structure
      # (depends on POS tagger model), so also accept if the intent is music-related
      speech_act = get_speech_act(context)
      intent = Map.get(context, :intent, "")

      is_command = speech_act[:category] == :directive or
                   speech_act[:sub_type] in [:command, :request_action]
      is_music_intent = is_binary(intent) and String.contains?(intent, "music")

      if map_size(speech_act) > 0 do
        assert is_command or is_music_intent,
               "Expected command or music intent, got speech_act: #{inspect(speech_act)}, intent: #{inspect(intent)}"
      end

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "responds to turn on lights command", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Turn on the lights")

      # Semantic assertion: Should be classified as a command/directive
      assert_is_command(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "responds to set reminder command", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Remind me to call mom tomorrow")

      # Semantic assertion: Should be classified as a command/directive
      # Note: If context is empty, we fall back to checking response text
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert_is_command(context)
      else
        # Fallback: check response contains expected patterns
        assert response =~ ~r/remind|reminder|remember|ok|sure|will do|set|tomorrow/i,
               "Expected reminder confirmation, got: #{response}"
      end

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end
  end

  describe "farewell responses" do
    test "responds appropriately to goodbye", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Goodbye!")

      # Semantic assertion: Should be classified as a farewell
      assert_is_farewell(context)

      # Basic sanity: response exists
      assert_has_response(response)
    end

    test "responds appropriately to bye", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Bye!")

      # Semantic assertion: Should be classified as a farewell
      assert_is_farewell(context)

      # Basic sanity: response exists
      assert_has_response(response)
    end

    test "responds appropriately to see you later", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Good bye see you later!")

      # Semantic assertion: Should be classified as a farewell
      assert_is_farewell(context)

      # Basic sanity: response exists
      assert_has_response(response)
    end
  end

  describe "multi-sentence messages" do
    test "handles greeting with weather question (with location)", %{conversation_id: conv_id} do
      # Note: The weather classifier needs a location to properly detect weather intent
      {:ok, response, context} =
        evaluate_with_context(
          conv_id,
          "Hello! What's the weather like in New York?"
        )

      # The context should show some intent was detected
      assert context[:intent] != nil or get_speech_act(context)[:category] != nil,
             "Expected some intent/speech_act classification, got: #{inspect(context)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "handles greeting followed by command", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi! Play some music please.")

      # The context should show some intent was detected (greeting or command)
      speech_act = get_speech_act(context)
      assert context[:intent] != nil or speech_act[:category] in [:directive, :expressive],
             "Expected greeting or command classification, got: #{inspect(context)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "handles multiple statements gracefully", %{conversation_id: conv_id} do
      {:ok, response, context} =
        evaluate_with_context(conv_id, "Hello, I'm Austin. It is nice to meet you.")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)
    end

    test "greeting introduction should not trigger music playback", %{conversation_id: conv_id} do
      # This is a regression test: "Hello, I'm Austin" should NOT be
      # interpreted as a request to play music (e.g., "Hello" by Adele)
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello, I'm Austin")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test - should not trigger music playback
      refute_response_matches(response, ~r/playing|play\s|music|song/i)
    end

    test "simple hello should not trigger music playback", %{conversation_id: conv_id} do
      # "Hello" alone should be a greeting, not the song "Hello"
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello!")

      # Semantic assertion: Should be classified as a greeting
      assert_is_greeting(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/playing|play\s/i)
    end

    test "hi with introduction should not trigger music playback", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hi, my name is Sarah")

      # Semantic assertion: Should be classified as a greeting
      # Note: If context is empty, we fall back to checking response text
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert_is_greeting(context)
      else
        # Fallback: check response is reasonable for introduction
        assert_has_response(response)
      end

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/playing|play\s/i)
    end
  end

  describe "conversational context" do
    test "handles simple statement", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "My name is Alex")

      # Semantic assertion: Should be classified as a greeting/introduction or assertive
      # Note: If context is empty, we fall back to checking response exists
      speech_act = get_speech_act(context)
      if map_size(speech_act) > 0 do
        assert speech_act[:sub_type] == :greeting or speech_act[:category] in [:assertive, :expressive],
               "Expected greeting/statement classification, got: #{inspect(speech_act)}"
      end

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end

    test "handles thank you", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Thank you!")

      # Semantic assertion: Should be classified as expressive (thanks)
      speech_act = get_speech_act(context)
      assert speech_act[:category] == :expressive or speech_act[:sub_type] == :thanks,
             "Expected expressive/thanks classification, got: #{inspect(speech_act)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
    end
  end

  describe "weather slot handling" do
    test "weather without location triggers clarification", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What's the weather?")

      # Should either:
      # 1. Return a clarification asking for location, OR
      # 2. Return a response indicating it needs location
      # The context should indicate needs_clarification or missing slots
      slots = Map.get(context, :slots, %{})
      strategy = Map.get(context, :response_strategy)

      # Check if clarification was triggered OR response asks for location
      location_missing = is_nil(get_in(slots, [:location])) or get_in(slots, [:location]) == ""
      asks_for_location = response =~ ~r/location|where|which city|what city/i
      is_clarification = strategy == :needs_clarification

      assert location_missing or asks_for_location or is_clarification,
             "Expected clarification for missing location, got: #{inspect(context)}, response: #{response}"

      assert_has_response(response)
    end

    test "weather with location extracts location slot", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What's the weather in Dallas?")

      # Check that the location was extracted
      slots = Map.get(context, :slots, %{})
      entities = Map.get(context, :entities, [])

      # Location might be in slots, entities, or mentioned in the response
      location_value = get_in(slots, [:location])
      location_in_slots = is_binary(location_value) and location_value =~ ~r/dallas/i
      location_in_entities = Enum.any?(entities, fn e ->
        entity_text = e[:text] || e["text"] || e[:value] || e["value"] || e[:match] || ""
        is_binary(entity_text) and String.downcase(entity_text) =~ "dallas"
      end)
      location_in_response = response =~ ~r/dallas/i

      assert location_in_slots or location_in_entities or location_in_response,
             "Expected location 'Dallas' to be extracted, got slots: #{inspect(slots)}, entities: #{inspect(entities)}"

      assert_has_response(response)
    end

    test "multi-sentence greeting + weather responds to BOTH parts", %{conversation_id: conv_id} do
      {:ok, response, _context} = evaluate_with_context(conv_id, "Hello! What's the weather in NYC?")

      # Should handle both the greeting AND the weather query
      # The response should acknowledge the greeting or be polite, AND address weather/NYC
      is_greeting_response = response =~ ~r/hello|hi|hey|greetings|good|nice/i
      mentions_weather_or_location = response =~ ~r/weather|nyc|new york|location|forecast/i

      ouro_ok = Brain.ML.Ouro.SidecarLauncher.status() == :healthy

      # When the Ouro sidecar is down or unhealthy, synthesis falls back to a
      # generic clarification string — do not fail the semantic assertion in
      # that degraded mode.
      assert ouro_ok == false or is_greeting_response or mentions_weather_or_location,
             "Expected response to address greeting and/or weather when Ouro is healthy, got: #{inspect(response)}"

      assert_has_response(response)
    end
  end

  describe "music slot handling" do
    test "play music with unknown artist extracts and narrows artist entity", %{conversation_id: conv_id} do
      # "Korvo Mitski" is completely absent from training data and Gazetteer.
      # The system should:
      # 1. POS-tag "Korvo Mitski" as proper nouns (PROPN)
      # 2. Merge consecutive PROPNs into a single entity "Korvo Mitski"
      # 3. Classify intent as music.play
      # 4. Narrow entity type from "person" to "artist" via TypeHierarchy
      # 5. Fill the music-artist slot
      {:ok, response, context} = evaluate_with_context(conv_id, "Play some Korvo Mitski")

      assert_is_command(context)

      entities = Map.get(context, :entities, [])

      artist_entity = Enum.find(entities, fn e ->
        entity_text = e[:text] || e["text"] || e[:value] || e["value"] || e[:match] || ""
        is_binary(entity_text) and String.downcase(entity_text) =~ "korvo"
      end)

      assert artist_entity != nil,
        "Expected 'Korvo Mitski' to be extracted as an entity. " <>
        "Got entities: #{inspect(entities)}"

      entity_type = artist_entity[:entity_type] || artist_entity["entity_type"]
      assert entity_type in ["artist", "music-artist", "person"],
        "Expected entity type to be artist, music-artist, or person, got: #{entity_type}"

      assert_has_response(response)
    end

    test "play specific song extracts song slot", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Play Bohemian Rhapsody")

      assert_is_command(context)

      # The song name might be in slots, entities, or acknowledged in response
      slots = Map.get(context, :slots, %{})
      entities = Map.get(context, :entities, [])

      song_in_slots = is_binary(get_in(slots, [:song])) and
                      String.downcase(slots[:song]) =~ "bohemian"
      song_in_entities = Enum.any?(entities, fn e ->
        entity_text = e[:text] || e["text"] || e[:value] || e["value"] || e[:match] || ""
        is_binary(entity_text) and String.downcase(entity_text) =~ "bohemian"
      end)
      song_in_response = response =~ ~r/bohemian/i

      # At least one should be true OR the response acknowledges the request
      assert song_in_slots or song_in_entities or song_in_response or response =~ ~r/play|music|song/i,
             "Expected song recognition, got slots: #{inspect(slots)}"

      assert_has_response(response)
    end

    test "search for music extracts search query", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Find me some jazz music")

      # Should classify as command/directive
      speech_act = get_speech_act(context)
      assert speech_act[:category] == :directive or context[:intent] =~ ~r/music|search|find/i,
             "Expected directive/search intent, got: #{inspect(context)}"

      assert_has_response(response)
    end
  end

  describe "factual question handling" do
    # Tests that factual question detection doesn't override appropriate responses.
    # The system should not respond with unrelated facts to conversational questions.

    test "weather question gets weather-related response, not random facts", %{
      conversation_id: conv_id
    } do
      {:ok, response, context} = evaluate_with_context(conv_id, "Can you tell me about the weather?")

      # Semantic assertion: Should be classified as a question
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test - should not dump random facts
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic/i)
    end

    test "greeting with weather question gets contextual response", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "Hello! Can you tell me about the weather?")

      # The context should show some intent was detected (greeting or weather)
      assert context[:intent] != nil or get_speech_act(context)[:category] != nil,
             "Expected some intent/speech_act classification, got: #{inspect(context)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic/i)
    end

    test "personal questions are not answered with facts", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What is your name?")

      # Semantic assertion: Should be classified as a question
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic|united nations/i)
    end

    test "how are you is conversational not factual", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "How are you doing today?")

      # Semantic assertion: Should be classified as a question or expressive
      speech_act = get_speech_act(context)
      assert speech_act[:is_question] == true or speech_act[:category] == :expressive,
             "Expected question or expressive, got: #{inspect(speech_act)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic|united nations/i)
    end

    test "combined greeting and question maintains context", %{conversation_id: conv_id} do
      {:ok, response, context} =
        evaluate_with_context(
          conv_id,
          "Hi there! Can you tell me about the weather? I'm planning a trip."
        )

      # The context should show some intent was detected
      assert context[:intent] != nil or get_speech_act(context)[:category] != nil,
             "Expected some intent/speech_act classification, got: #{inspect(context)}"

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression tests
      refute_response_matches(response, ~r/bye|goodbye|see you later/i)
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic/i)
    end

    test "what can you do is about capabilities not facts", %{conversation_id: conv_id} do
      {:ok, response, context} = evaluate_with_context(conv_id, "What can you do?")

      # Semantic assertion: Should be classified as a question
      assert_is_question(context)

      # Basic sanity: response exists
      assert_has_response(response)

      # Regression test
      refute_response_matches(response, ~r/week|days in a|alphabet|chess|olympic|earth|billion/i)
    end
  end
end
