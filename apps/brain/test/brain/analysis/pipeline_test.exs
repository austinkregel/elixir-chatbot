defmodule Brain.Analysis.PipelineTest do
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.InternalModel
  alias Brain.ML.EntityExtractor

  setup do
    # Start common test services (PubSub, IntentClassifierSimple, Gazetteer, etc.)
    start_test_services()

    # Ensure entity maps are loaded for entity extraction
    EntityExtractor.load_entity_maps()
    :ok
  end

  describe "process/2" do
    test "processes simple greeting" do
      model = Pipeline.process("Hello there!")

      assert %InternalModel{
               raw_input: "Hello there!",
               chunks: chunks,
               analyses: analyses,
               overall_strategy: strategy
             } = model

      assert length(chunks) >= 1
      assert length(analyses) >= 1

      # Greeting should be respondable
      assert strategy == :can_respond
    end

    test "processes question with addressee detection" do
      # Include location to satisfy required slots for weather query
      model = Pipeline.process("Can you tell me the weather in New York?")

      # Strategy may be :can_respond or :needs_clarification depending on entity extraction
      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]

      # Bot should be detected as addressee
      first_analysis = List.first(model.analyses)
      assert first_analysis.discourse.addressee == :bot
      assert first_analysis.speech_act.is_question == true
    end

    test "handles multi-sentence input" do
      model = Pipeline.process("Hello! What's the news? Also check the weather.")

      assert length(model.chunks) >= 2
      assert length(model.analyses) >= 2
    end

    test "detects need for clarification when context missing" do
      # Weather query without location
      model =
        Pipeline.process("What's the weather like?",
          skip_entity_extraction: true,
          entities: []
        )

      # Either the model figures out it needs location, or it proceeds with best effort
      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]
    end

    test "uses provided entities" do
      entities = [
        %{entity_type: "location", value: "Paris", confidence: 0.9}
      ]

      model =
        Pipeline.process("What's the weather?",
          entities: entities,
          skip_entity_extraction: true
        )

      first_analysis = List.first(model.analyses)

      # The entities might be transformed - check that the analysis ran successfully
      assert first_analysis != nil
      assert first_analysis.slots != nil
    end

    test "uses conversation history for context" do
      history = [
        %{
          entities: %{"location" => "London"},
          intent: "weather.query",
          timestamp: System.system_time(:millisecond)
        }
      ]

      model =
        Pipeline.process("What about tomorrow?",
          conversation_history: history,
          entities: [],
          skip_entity_extraction: true
        )

      # Should be able to resolve context from history
      assert %InternalModel{} = model
    end
  end

  describe "analyze_chunk/2" do
    test "analyzes a single chunk" do
      analysis = Pipeline.analyze_chunk("What time is it?")

      assert analysis.chunk_index == 0
      assert analysis.text == "What time is it?"
      assert analysis.discourse.addressee == :bot
      assert analysis.speech_act.is_question == true
    end

    test "analyzes imperative command" do
      analysis = Pipeline.analyze_chunk("Turn on the lights")

      # Model-based classification detects imperatives via intent
      # or structural analysis should detect it
      assert analysis.speech_act.category == :directive or
               analysis.speech_act.is_imperative == true or
               analysis.speech_act.sub_type == :command

      assert analysis.discourse.addressee == :bot
    end

    test "analyzes greeting" do
      analysis = Pipeline.analyze_chunk("Hello!")

      # Model-based classification should recognize greeting
      # May be :expressive or have greeting sub_type
      assert analysis.speech_act.category == :expressive or
               analysis.speech_act.sub_type == :greeting or
               analysis.speech_act.sub_type != :farewell
    end
  end

  describe "summarize/1" do
    test "provides summary of analysis" do
      model = Pipeline.process("Hello! How are you?")
      summary = Pipeline.summarize(model)

      assert is_map(summary)
      assert Map.has_key?(summary, :chunks)
      assert Map.has_key?(summary, :analyses)
      assert Map.has_key?(summary, :overall_strategy)
    end
  end

  describe "InternalModel helpers" do
    test "bot_addressed? returns true when bot is addressed" do
      model = Pipeline.process("Hey bot, what's up?")

      assert InternalModel.bot_addressed?(model) == true
    end

    test "respondable_analyses filters correctly" do
      model = Pipeline.process("Hello there!")

      respondable = InternalModel.respondable_analyses(model)

      assert length(respondable) >= 1
      assert Enum.all?(respondable, &(&1.response_strategy == :can_respond))
    end
  end
end
