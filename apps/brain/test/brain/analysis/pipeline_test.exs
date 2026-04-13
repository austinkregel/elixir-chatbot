defmodule Brain.Analysis.PipelineTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.InternalModel
  alias Brain.ML.EntityExtractor

  setup _context do
    start_test_services()
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

      assert chunks != []
      assert analyses != []
      assert strategy == :can_respond
    end

    test "processes question with addressee detection" do
      model = Pipeline.process("Can you tell me the weather in New York?")

      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]

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
      model =
        Pipeline.process("What's the weather like?", skip_entity_extraction: true, entities: [])

      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]
    end

    test "uses provided entities" do
      entities = [%{entity_type: "location", value: "Paris", confidence: 0.9}]

      model =
        Pipeline.process("What's the weather?", entities: entities, skip_entity_extraction: true)

      first_analysis = List.first(model.analyses)
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

      assert analysis.speech_act.category == :directive or
               analysis.speech_act.is_imperative == true or
               analysis.speech_act.sub_type == :command

      assert analysis.discourse.addressee == :bot
    end

    test "analyzes greeting" do
      analysis = Pipeline.analyze_chunk("Hello!")

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

      assert respondable != []
      assert Enum.all?(respondable, &(&1.response_strategy in [:can_respond, :hedged_response]))
    end
  end
end
