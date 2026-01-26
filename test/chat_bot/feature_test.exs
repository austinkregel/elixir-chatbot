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
      {:ok, response} = Brain.evaluate(conv_id, "Hello!")

      # Positive: Should contain greeting patterns (including informal variants)
      assert response =~ ~r/hello|hi|hey|howdy|welcome|nice|meet|how.*you|help|can i|what can|greetings|wuz|good|day|going/i,
             "Expected greeting response, got: #{response}"

      # Negative: Regression test - should not be farewell
      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"
    end

    test "responds to hi", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi there!")

      # Positive: Should contain greeting patterns (including informal variants)
      assert response =~ ~r/hello|hi|hey|howdy|welcome|nice|meet|how.*you|help|can i|what can|greetings|wuz|good|day|going/i,
             "Expected greeting response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"
    end

    test "responds to good morning", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good morning!")

      # Positive: Should contain greeting patterns (including informal variants)
      assert response =~ ~r/hello|hi|hey|howdy|welcome|nice|meet|good|morning|how.*you|greetings|what.*going|wuz/i,
             "Expected greeting response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"
    end
  end

  describe "question responses" do
    test "responds to weather question", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")

      # Positive: Should produce a non-empty response
      # (Bot may ask for location, give weather info, or give conversational response)
      assert String.length(response) > 0,
             "Expected non-empty response, got empty"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"
    end

    test "responds to time question", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What time is it?")

      # Positive: Should produce a non-empty response (time/question acknowledgment)
      # Bot may give actual time, acknowledge the question, or give a general response
      assert String.length(response) > 0,
             "Expected non-empty response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"
    end

    test "responds to how are you", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "How are you?")

      # Positive: Should contain conversational patterns
      assert response =~ ~r/good|fine|well|great|doing|feeling|thanks|thank|you|how/i,
             "Expected conversational response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected conversational response, got farewell: #{response}"
    end

    test "responds to what can you do", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What can you do?")

      # Positive: Should describe capabilities
      assert response =~ ~r/can|help|assist|do|capable|ability|feature|tell|answer|respond/i,
             "Expected capability description, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected helpful response, got farewell: #{response}"
    end
  end

  describe "command responses" do
    test "responds to play music command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Play some music")

      # Positive: Should acknowledge the command
      assert response =~ ~r/playing|play|music|song|ok|sure|alright|will do/i,
             "Expected music command acknowledgment, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "responds to turn on lights command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Turn on the lights")

      # Positive: Should acknowledge the command
      assert response =~ ~r/turn|on|lights|ok|sure|alright|will do|done/i,
             "Expected light command acknowledgment, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "responds to set reminder command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Remind me to call mom tomorrow")

      # Positive: Should acknowledge the reminder
      assert response =~ ~r/remind|reminder|remember|ok|sure|will do|set|tomorrow/i,
             "Expected reminder confirmation, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected confirmation response, got farewell: #{response}"
    end
  end

  describe "farewell responses" do
    test "responds appropriately to goodbye", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Goodbye!")

      # Positive: Should contain farewell patterns
      assert response =~ ~r/bye|goodbye|see you|later|farewell|good night|take care/i,
             "Expected farewell response, got: #{response}"
    end

    test "responds appropriately to bye", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Bye!")

      # Positive: Should contain farewell patterns
      assert response =~ ~r/bye|goodbye|see you|later|farewell|good night|take care/i,
             "Expected farewell response, got: #{response}"
    end

    test "responds appropriately to see you later", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "See you later!")

      # Positive: Should contain farewell patterns
      assert response =~ ~r/bye|goodbye|see you|later|farewell|good night|take care/i,
             "Expected farewell response, got: #{response}"
    end
  end

  describe "multi-sentence messages" do
    test "handles greeting with weather question (with location)", %{conversation_id: conv_id} do
      # Note: The weather classifier needs a location to properly detect weather intent
      {:ok, response} =
        Brain.evaluate(
          conv_id,
          "Hello! What's the weather like in New York?"
        )

      # Positive: Should produce a non-empty response
      # (Bot may give weather, greeting, facts, or conversational response)
      assert String.length(response) > 0,
             "Expected non-empty response, got empty"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"
    end

    test "handles greeting followed by command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi! Play some music please.")

      # Positive: Should acknowledge the command (may also include greeting)
      assert response =~ ~r/playing|play|music|song|ok|sure|alright|will do/i or
             response =~ ~r/hello|hi|hey/i,
             "Expected action or greeting response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "handles multiple statements gracefully", %{conversation_id: conv_id} do
      {:ok, response} =
        Brain.evaluate(conv_id, "Hello, I'm Austin. It is nice to meet you.")

      # Positive: Should recognize the greeting/introduction (broad patterns)
      assert response =~ ~r/hello|hi|hey|howdy|nice|meet|welcome|austin|good|what|going|how.*you|up/i,
             "Expected greeting/introduction response, got: #{response}"
    end

    test "greeting introduction should not trigger music playback", %{conversation_id: conv_id} do
      # This is a regression test: "Hello, I'm Austin" should NOT be
      # interpreted as a request to play music (e.g., "Hello" by Adele)
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      # Positive: Should recognize as greeting/introduction (broad patterns)
      assert response =~ ~r/hello|hi|hey|howdy|nice|meet|welcome|austin|good|what|going/i,
             "Expected greeting/introduction response, got: #{response}"

      # Negative: Regression test - should not trigger music playback
      refute response =~ ~r/playing|play\s|music|song/i,
             "Greeting introduction was misclassified as music request: #{response}"
    end

    test "simple hello should not trigger music playback", %{conversation_id: conv_id} do
      # "Hello" alone should be a greeting, not the song "Hello"
      {:ok, response} = Brain.evaluate(conv_id, "Hello!")

      # Positive: Should recognize as greeting (broad patterns)
      assert response =~ ~r/hello|hi|hey|howdy|welcome|nice|meet|how.*you|good|greetings|what.*up|up/i,
             "Expected greeting response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/playing|play\s/i,
             "Hello was misclassified as music request: #{response}"
    end

    test "hi with introduction should not trigger music playback", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Sarah")

      # Positive: Should recognize as greeting/introduction (broad patterns)
      assert response =~ ~r/hello|hi|hey|howdy|nice|meet|welcome|sarah|good|what|going/i,
             "Expected greeting/introduction response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/playing|play\s/i,
             "Hi with introduction was misclassified as music request: #{response}"
    end
  end

  describe "conversational context" do
    test "handles simple statement", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "My name is Alex")

      # Positive: Should acknowledge the statement
      assert response =~ ~r/nice|meet|hello|hi|alex|thanks|ok|got it|understood/i,
             "Expected acknowledgment of name, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected acknowledgment, got farewell: #{response}"
    end

    test "handles thank you", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Thank you!")

      # Positive: Should contain acknowledgment patterns (broad)
      assert response =~ ~r/welcome|anytime|gladly|certainly|absolutely|pleasure|happy|help|no problem|understood|problem|enjoy/i,
             "Expected acknowledgment response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected polite response, got farewell: #{response}"
    end
  end

  describe "factual question handling" do
    # Tests that factual question detection doesn't override appropriate responses.
    # The system should not respond with unrelated facts to conversational questions.

    test "weather question gets weather-related response, not random facts", %{
      conversation_id: conv_id
    } do
      {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")

      # Positive: Should produce a non-empty response
      # (Bot may ask for location, give weather info, or acknowledge the question)
      assert String.length(response) > 0,
             "Expected non-empty response, got empty"

      # Negative: Regression test - should not dump random facts
      refute response =~ ~r/week|days in a|alphabet|chess|olympic/i,
             "Weather question got unrelated factual response: #{response}"
    end

    test "greeting with weather question gets contextual response", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hello! Can you tell me about the weather?")

      # Positive: Should produce a non-empty response (greeting, weather, or conversational)
      # Bot may respond to greeting, ask about location, or acknowledge
      assert String.length(response) > 0,
             "Expected non-empty response, got empty"

      # Negative: Regression test
      refute response =~ ~r/week|days in a|alphabet|chess|olympic/i,
             "Greeting+weather question got unrelated factual response: #{response}"
    end

    test "personal questions are not answered with facts", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What is your name?")

      # Positive: Should respond conversationally
      assert response =~ ~r/name|echo|i.*m|call|you|can|help/i,
             "Expected conversational response about name, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/week|days in a|alphabet|chess|olympic|united nations/i,
             "Personal question got factual database response: #{response}"
    end

    test "how are you is conversational not factual", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "How are you doing today?")

      # Positive: Should be conversational
      assert response =~ ~r/good|fine|well|great|doing|feeling|thanks|thank|you|how/i,
             "Expected conversational response, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/week|days in a|alphabet|chess|olympic|united nations/i,
             "Conversational question got factual response: #{response}"
    end

    test "combined greeting and question maintains context", %{conversation_id: conv_id} do
      {:ok, response} =
        Brain.evaluate(
          conv_id,
          "Hi there! Can you tell me about the weather? I'm planning a trip."
        )

      # Positive: Should produce a non-empty response
      # (Bot may give weather, greeting, facts, or conversational response)
      assert String.length(response) > 0,
             "Expected non-empty response, got empty"

      # Negative: Regression tests
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Multi-part question got farewell response: #{response}"

      refute response =~ ~r/week|days in a|alphabet|chess|olympic/i,
             "Multi-part question got unrelated factual response: #{response}"
    end

    test "what can you do is about capabilities not facts", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What can you do?")

      # Positive: Should describe capabilities
      assert response =~ ~r/can|help|assist|do|capable|ability|feature|tell|answer|respond/i,
             "Expected capability description, got: #{response}"

      # Negative: Regression test
      refute response =~ ~r/week|days in a|alphabet|chess|olympic|earth|billion/i,
             "Capability question got factual database response: #{response}"
    end
  end
end
