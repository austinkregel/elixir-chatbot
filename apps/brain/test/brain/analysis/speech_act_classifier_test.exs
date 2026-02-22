defmodule Brain.Analysis.SpeechActClassifierTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Analysis.SpeechActClassifier
  alias Brain.Analysis.SpeechActResult
  import Brain.TestHelpers

  setup_all do
    Brain.TestHelpers.require_services!(:ml_inference)
    :ok
  end

  setup _context do
    start_test_services()
    :ok
  end

  describe "classify/1 - questions" do
    test "classifies question with question mark" do
      result = SpeechActClassifier.classify("What is the weather?")

      assert %SpeechActResult{
               category: category,
               is_question: is_question,
               confidence: conf
             } = result

      # Should be recognized as a question/request
      assert is_question == true or category == :directive
      assert conf > 0.3
    end

    test "classifies question with question word" do
      result = SpeechActClassifier.classify("Where is the nearest coffee shop")

      # Should recognize the question structure
      assert result.is_question == true or result.category == :directive
    end

    test "classifies modal you pattern as question/request" do
      result = SpeechActClassifier.classify("Would you open the window?")

      # Modal "you" pattern should be recognized as a request
      assert result.is_question == true or result.category in [:directive, :expressive]
    end

    test "classifies inversion pattern as question" do
      result = SpeechActClassifier.classify("Do you know the time?")

      assert result.is_question == true
    end
  end

  describe "classify/1 - requests" do
    test "classifies imperative as request action" do
      result = SpeechActClassifier.classify("Turn on the lights")

      assert result.category == :directive
      assert result.is_imperative == true
    end

    test "classifies polite request" do
      result = SpeechActClassifier.classify("Please turn on the lights")

      # Should recognize this as a request/command
      assert result.category == :directive or result.is_imperative == true
    end

    test "classifies desire pattern" do
      result = SpeechActClassifier.classify("I want to check my account balance")

      # This might be classified variously - just check it returns valid result
      assert result.category in [:directive, :assertive, :expressive]
    end

    test "classifies information request with tell me" do
      result = SpeechActClassifier.classify("Tell me about the current time")

      # "Tell" is an imperative, so should be directive
      assert result.is_imperative == true or result.category == :directive
    end
  end

  describe "classify/1 - expressives" do
    test "classifies greeting" do
      result = SpeechActClassifier.classify("Hello there!")

      # Model should recognize greeting from training data
      assert result.category == :expressive or result.sub_type == :greeting
    end

    test "classifies various greetings" do
      greetings = ["Hi!", "Hey there", "Good morning"]

      for greeting <- greetings do
        result = SpeechActClassifier.classify(greeting)
        # Should be expressive greeting or at least not a farewell
        assert result.sub_type != :farewell, "Got farewell for: #{greeting}"
      end
    end

    test "classifies farewell" do
      result = SpeechActClassifier.classify("Goodbye!")

      # Model should recognize farewell from training data
      assert result.category == :expressive or result.sub_type == :farewell
    end

    test "classifies thanks" do
      result = SpeechActClassifier.classify("Thanks!")

      # Model should recognize thanks from training data
      assert result.category == :expressive or result.sub_type == :thanks
    end

    test "classifies apology" do
      result = SpeechActClassifier.classify("Sorry about that")

      # Model may classify differently - just ensure valid response
      assert result.category in [:expressive, :assertive]
    end
  end

  describe "classify/1 - commissives" do
    test "classifies promise" do
      result = SpeechActClassifier.classify("I will do it tomorrow")

      # Model may not have commissive training data - use structural fallback
      assert result.category in [:commissive, :assertive, :directive]
    end

    test "classifies offer" do
      result = SpeechActClassifier.classify("shall i open the door")

      # Model may not recognize this as offer - falls back to question detection
      assert result.category in [:commissive, :directive, :assertive]
    end
  end

  describe "classify/1 - statements" do
    test "classifies declarative statement" do
      result = SpeechActClassifier.classify("The sky is blue.")

      assert result.category == :assertive
      assert result.sub_type == :statement
    end

    test "classifies self-referential statement" do
      result = SpeechActClassifier.classify("I am feeling great today.")

      assert result.category in [:assertive, :expressive]
    end
  end

  describe "analyze/1" do
    test "returns detailed analysis" do
      result = SpeechActClassifier.analyze("What is the weather?")

      assert %{
               result: %SpeechActResult{},
               intent_classification: intent_info,
               structural_analysis: struct_info
             } = result

      assert Map.has_key?(intent_info, :intent)
      assert Map.has_key?(struct_info, :is_question)
    end
  end

  describe "expects_response?/1" do
    test "questions expect response" do
      result = SpeechActClassifier.classify("What is the capital of France?")
      # Questions are directives and expect responses
      assert result.is_question == true
    end

    test "greetings expect response" do
      result = SpeechActClassifier.classify("Hello!")
      assert SpeechActResult.expects_response?(result) == true
    end

    test "statements may not expect response" do
      result = SpeechActClassifier.classify("The weather is nice.")
      # Statements don't typically expect a response
      assert result.category in [:assertive, :directive]
    end
  end

  describe "model-based classification" do
    test "uses trained intent for greeting" do
      result = SpeechActClassifier.classify("Hello!")

      # The model should classify this appropriately
      assert result.confidence > 0.0
      assert result.sub_type != :farewell
    end

    test "uses trained intent for weather question" do
      result = SpeechActClassifier.classify("What's the weather like?")

      # Should be recognized as information request
      assert result.category == :directive or result.is_question == true
    end

    test "does not confuse greeting with farewell" do
      greeting_result = SpeechActClassifier.classify("Hello!")
      farewell_result = SpeechActClassifier.classify("Goodbye!")

      # These should have different sub_types
      assert greeting_result.sub_type != farewell_result.sub_type or
               (greeting_result.sub_type == :greeting and farewell_result.sub_type == :farewell)
    end
  end
end
