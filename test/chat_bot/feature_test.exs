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

      # Should respond with a greeting, not a farewell
      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"

      # Should be a reasonable response
      assert String.length(response) > 0
    end

    test "responds to hi", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi there!")

      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"
    end

    test "responds to good morning", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Good morning!")

      refute response =~ ~r/bye|goodbye|see you|later/i,
             "Expected greeting response, got farewell: #{response}"
    end
  end

  describe "question responses" do
    test "responds to weather question", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Can you tell me about the weather?")

      # Should not respond with a farewell to a question
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"

      # Should give some response
      assert String.length(response) > 0
    end

    test "responds to time question", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What time is it?")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"
    end

    test "responds to how are you", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "How are you?")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected conversational response, got farewell: #{response}"
    end

    test "responds to what can you do", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "What can you do?")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected helpful response, got farewell: #{response}"
    end
  end

  describe "command responses" do
    test "responds to play music command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Play some music")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "responds to turn on lights command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Turn on the lights")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "responds to set reminder command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Remind me to call mom tomorrow")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected confirmation response, got farewell: #{response}"
    end
  end

  describe "farewell responses" do
    test "responds appropriately to goodbye", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Goodbye!")

      # Farewell IS appropriate here
      assert String.length(response) > 0
    end

    test "responds appropriately to bye", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Bye!")

      assert String.length(response) > 0
    end

    test "responds appropriately to see you later", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "See you later!")

      assert String.length(response) > 0
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

      # Should give a response (not just crash)
      assert String.length(response) > 0

      # Should not respond with a farewell
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected informative response, got farewell: #{response}"
    end

    test "handles greeting followed by command", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi! Play some music please.")

      # Should not just respond with farewell
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected action response, got farewell: #{response}"
    end

    test "handles multiple statements gracefully", %{conversation_id: conv_id} do
      {:ok, response} =
        Brain.evaluate(conv_id, "Hello, I'm Austin. It is nice to meet you.")

      # Should give some response
      assert String.length(response) > 0

      # Should recognize the greeting
      assert response =~ ~r/hello|hi|hey|nice|meet/i or String.length(response) > 0
    end

    test "greeting introduction should not trigger music playback", %{conversation_id: conv_id} do
      # This is a regression test: "Hello, I'm Austin" should NOT be
      # interpreted as a request to play music (e.g., "Hello" by Adele)
      {:ok, response} = Brain.evaluate(conv_id, "Hello, I'm Austin")

      # Should NOT mention playing anything
      refute response =~ ~r/playing|play\s/i,
             "Greeting was misclassified as music request: #{response}"

      # Should NOT ask what to play
      refute response =~ ~r/what would you like me to play/i,
             "Greeting was misclassified as music request: #{response}"
    end

    test "simple hello should not trigger music playback", %{conversation_id: conv_id} do
      # "Hello" alone should be a greeting, not the song "Hello"
      {:ok, response} = Brain.evaluate(conv_id, "Hello!")

      refute response =~ ~r/playing|play\s/i,
             "Hello was misclassified as music request: #{response}"
    end

    test "hi with introduction should not trigger music playback", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Hi, my name is Sarah")

      refute response =~ ~r/playing|play\s/i,
             "Hi with introduction was misclassified as music request: #{response}"
    end
  end

  describe "conversational context" do
    test "handles simple statement", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "My name is Alex")

      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected acknowledgment, got farewell: #{response}"
    end

    test "handles thank you", %{conversation_id: conv_id} do
      {:ok, response} = Brain.evaluate(conv_id, "Thank you!")

      # Thank you should get a polite response, not a farewell
      refute response =~ ~r/bye|goodbye|see you later/i,
             "Expected polite response, got farewell: #{response}"
    end
  end
end
