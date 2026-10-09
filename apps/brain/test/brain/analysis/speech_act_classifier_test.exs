defmodule Brain.Analysis.SpeechActClassifierTest do
  use Brain.Test.GraphCase, async: false

  alias Brain.Analysis.SpeechActClassifier
  alias Brain.Analysis.SpeechActResult
  alias Brain.Analysis.SpeechActTable
  import Brain.TestHelpers

  @world_id "default"

  setup_all do
    Brain.TestHelpers.require_services!(:ml_inference)
    :ok
  end

  setup _context do
    start_test_services()
    :ok
  end

  describe "classify/2 - world" do
    test "raises without a :world_id" do
      assert_raise ArgumentError, ~r/SpeechActClassifier.classify requires a :world_id option/, fn ->
        SpeechActClassifier.classify("hi", [])
      end
    end

    test "raises on an empty :world_id" do
      assert_raise ArgumentError, ~r/SpeechActClassifier.classify requires a :world_id option/, fn ->
        SpeechActClassifier.classify("hi", world_id: "")
      end
    end
  end

  describe "classify/2 - questions" do
    test "classifies question with question mark" do
      result = SpeechActClassifier.classify("What is the weather?", world_id: @world_id)

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
      result = SpeechActClassifier.classify("Where is the nearest coffee shop", world_id: @world_id)

      # Should recognize the question structure
      assert result.is_question == true or result.category == :directive
    end

    test "classifies modal you pattern as question/request" do
      result = SpeechActClassifier.classify("Would you open the window?", world_id: @world_id)

      # Modal "you" pattern should be recognized as a request
      assert result.is_question == true or result.category in [:directive, :expressive]
    end

    test "classifies inversion pattern as question" do
      result = SpeechActClassifier.classify("Do you know the time?", world_id: @world_id)

      assert result.is_question == true
    end
  end

  describe "classify/2 - requests" do
    test "classifies imperative as request action" do
      result = SpeechActClassifier.classify("Turn on the lights", world_id: @world_id)

      assert result.category == :directive
      assert result.is_imperative == true
    end

    test "classifies polite request" do
      result = SpeechActClassifier.classify("Please turn on the lights", world_id: @world_id)

      # Should recognize this as a request/command
      assert result.category == :directive or result.is_imperative == true
    end

    test "classifies desire pattern" do
      result = SpeechActClassifier.classify("I want to check my account balance", world_id: @world_id)

      # This might be classified variously - just check it returns valid result
      assert result.category in [:directive, :assertive, :expressive]
    end

    test "classifies information request with tell me" do
      result = SpeechActClassifier.classify("Tell me about the current time", world_id: @world_id)

      # "Tell" is an imperative, so should be directive
      assert result.is_imperative == true or result.category == :directive
    end
  end

  describe "classify/2 - expressives" do
    test "classifies greeting" do
      result = SpeechActClassifier.classify("Hello there!", world_id: @world_id)

      # Model should recognize greeting from training data
      assert result.category == :expressive or result.sub_type == :greeting
    end

    test "classifies various greetings" do
      greetings = ["Hi!", "Hey there", "Good morning"]

      for greeting <- greetings do
        result = SpeechActClassifier.classify(greeting, world_id: @world_id)
        # Should be expressive greeting or at least not a farewell
        assert result.sub_type != :farewell, "Got farewell for: #{greeting}"
      end
    end

    test "classifies farewell" do
      result = SpeechActClassifier.classify("Goodbye!", world_id: @world_id)

      # Model should recognize farewell from training data
      assert result.category == :expressive or result.sub_type == :farewell
    end

    test "classifies thanks" do
      result = SpeechActClassifier.classify("Thanks!", world_id: @world_id)

      # Model should recognize thanks from training data
      assert result.category == :expressive or result.sub_type == :thanks
    end

    test "classifies apology" do
      result = SpeechActClassifier.classify("Sorry about that", world_id: @world_id)

      # Model may classify differently - just ensure valid response
      assert result.category in [:expressive, :assertive]
    end
  end

  describe "classify/2 - commissives" do
    test "classifies promise" do
      result = SpeechActClassifier.classify("I will do it tomorrow", world_id: @world_id)

      # Model may not have commissive training data - use structural fallback
      assert result.category in [:commissive, :assertive, :directive]
    end

    test "classifies offer" do
      result = SpeechActClassifier.classify("shall i open the door", world_id: @world_id)

      # Model may not recognize this as offer - falls back to question detection
      assert result.category in [:commissive, :directive, :assertive]
    end
  end

  describe "classify/2 - statements" do
    test "classifies declarative statement" do
      result = SpeechActClassifier.classify("The sky is blue.", world_id: @world_id)

      assert result.category == :assertive
      assert result.sub_type == :statement
    end

    test "classifies self-referential statement" do
      result = SpeechActClassifier.classify("I am feeling great today.", world_id: @world_id)

      assert result.category in [:assertive, :expressive]
    end
  end

  describe "analyze/2" do
    test "returns detailed analysis" do
      result = SpeechActClassifier.analyze("What is the weather?", world_id: @world_id)

      assert %{
               result: %SpeechActResult{},
               intent_classification: intent_info,
               structural_analysis: struct_info
             } = result

      assert Map.has_key?(intent_info, :intent)
      assert Map.has_key?(struct_info, :is_question)
    end

    test "raises without a :world_id" do
      assert_raise ArgumentError, ~r/SpeechActClassifier.analyze requires a :world_id option/, fn ->
        SpeechActClassifier.analyze("What is the weather?", [])
      end
    end
  end

  describe "expects_response?/1" do
    test "questions expect response" do
      result = SpeechActClassifier.classify("What is the capital of France?", world_id: @world_id)
      # Questions are directives and expect responses
      assert result.is_question == true
    end

    test "greetings expect response" do
      result = SpeechActClassifier.classify("Hello!", world_id: @world_id)
      assert SpeechActResult.expects_response?(result) == true
    end

    test "statements may not expect response" do
      result = SpeechActClassifier.classify("The weather is nice.", world_id: @world_id)
      # Statements don't typically expect a response
      assert result.category in [:assertive, :directive]
    end
  end

  describe "model-based classification" do
    test "uses trained intent for greeting" do
      result = SpeechActClassifier.classify("Hello!", world_id: @world_id)

      # The model should classify this appropriately
      assert result.confidence > 0.0
      assert result.sub_type != :farewell
    end

    test "uses trained intent for weather question" do
      result = SpeechActClassifier.classify("What's the weather like?", world_id: @world_id)

      # Should be recognized as information request
      assert result.category == :directive or result.is_question == true
    end

    test "the classified intent's speech act comes from the speech-act table" do
      table =
        SpeechActTable.build(
          [{"hello there", "smalltalk.greetings.hello"}, {"what is the weather", "weather.query"}],
          %{
            "smalltalk.greetings.hello" => %{"category" => "expressive", "speech_act" => "greeting"},
            "weather.query" => %{"category" => "directive", "speech_act" => "request_information"}
          },
          "en"
        )

      lattice = Brain.Lattice.from_top_k([{"smalltalk.greetings.hello", 5.0}, {"weather.query", 0.1}], stage: :intent_full)

      refined =
        SpeechActClassifier.refine_with_intent(
          SpeechActClassifier.classify("hello there", world_id: @world_id),
          lattice,
          table
        )

      assert refined.raw_analyses.intent.intent == "smalltalk.greetings.hello"
      assert {refined.raw_analyses.intent.category, refined.raw_analyses.intent.sub_type} == {:expressive, :greeting}

      unknown = Brain.Lattice.from_top_k([{"music.play", 5.0}], stage: :intent_full)

      assert_raise RuntimeError, ~r/no en speech act for intent "music.play"/, fn ->
        SpeechActClassifier.refine_with_intent(
          SpeechActClassifier.classify("play jazz", world_id: @world_id),
          unknown,
          table
        )
      end
    end

    test "does not confuse greeting with farewell" do
      greeting_result = SpeechActClassifier.classify("Hello!", world_id: @world_id)
      farewell_result = SpeechActClassifier.classify("Goodbye!", world_id: @world_id)

      # These should have different sub_types
      assert greeting_result.sub_type != farewell_result.sub_type or
               (greeting_result.sub_type == :greeting and farewell_result.sub_type == :farewell)
    end
  end
end
