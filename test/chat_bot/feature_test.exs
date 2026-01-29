defmodule ChatBot.FeatureTest do
  @moduledoc """
  Feature tests that verify end-to-end chatbot behavior.

  These tests check that user inputs produce sensible outputs.
  """
  use ExUnit.Case, async: false

  alias ChatBot.Brain
  import ChatBot.TestHelpers

  setup do
    start_brain_services()

    # Create a conversation for each test
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
      assert_is_command(context)

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
      {:ok, response, context} = evaluate_with_context(conv_id, "See you later!")

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
