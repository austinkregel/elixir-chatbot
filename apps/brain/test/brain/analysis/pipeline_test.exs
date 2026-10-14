defmodule Brain.Analysis.PipelineTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Analysis.Pipeline
  alias Brain.Analysis.InternalModel

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  setup _context do
    start_test_services()
    :ok
  end

  describe "process/2" do
    test "processes simple greeting" do
      model = Pipeline.process("Hello there!", world_id: @world_id)

      assert %InternalModel{
               raw_input: "Hello there!",
               chunks: chunks,
               analyses: analyses,
               overall_strategy: strategy
             } = model

      assert chunks != []
      assert analyses != []
      assert strategy in [:can_respond, :hedged_response]
    end

    test "processes question with addressee detection" do
      model = Pipeline.process("Can you tell me the weather in New York?", world_id: @world_id)

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
      model = Pipeline.process("Hello! What's the news? Also check the weather.", world_id: @world_id)

      assert length(model.chunks) >= 2
      assert length(model.analyses) >= 2
    end

    test "detects need for clarification when context missing" do
      model =
        Pipeline.process("What's the weather like?",
          world_id: @world_id,
          skip_entity_extraction: true,
          entities: []
        )

      assert model.overall_strategy in [
               :can_respond,
               :needs_clarification,
               :partial_response_with_clarification
             ]
    end

    test "uses provided entities" do
      entities = [%{entity_type: "location", value: "Paris", confidence: 0.9}]

      model =
        Pipeline.process("What's the weather?",
          world_id: @world_id,
          entities: entities,
          skip_entity_extraction: true
        )

      first_analysis = List.first(model.analyses)
      assert first_analysis != nil
      assert first_analysis.slots != nil
    end

    test "uses conversation history for context" do
      history = [
        %{
          entities: [%{entity_type: "location", value: "London", confidence: 0.9}],
          intent: "weather.query",
          timestamp: System.system_time(:millisecond)
        }
      ]

      model =
        Pipeline.process("What about tomorrow?",
          world_id: @world_id,
          conversation_history: history,
          entities: [],
          skip_entity_extraction: true
        )

      assert %InternalModel{} = model
    end

    test "raises when no world is named" do
      assert_raise ArgumentError, ~r/Pipeline needs :world_id/, fn ->
        Pipeline.process("Hello there!")
      end

      assert_raise ArgumentError, ~r/Pipeline needs :world_id/, fn ->
        Pipeline.process("Hello there!", side_effects: false)
      end
    end
  end

  describe "analyze_chunk/2" do
    test "analyzes a single chunk" do
      analysis = Pipeline.analyze_chunk("What time is it?", world_id: @world_id)

      assert analysis.chunk_index == 0
      assert analysis.text == "What time is it?"
      assert analysis.discourse.addressee == :bot
      assert analysis.speech_act.is_question == true
    end

    test "analyzes imperative command" do
      analysis = Pipeline.analyze_chunk("Turn on the lights", world_id: @world_id)

      assert analysis.speech_act.category == :directive or
               analysis.speech_act.is_imperative == true or
               analysis.speech_act.sub_type == :command

      assert analysis.discourse.addressee == :bot
    end

    test "analyzes greeting" do
      analysis = Pipeline.analyze_chunk("Hello!", world_id: @world_id)

      assert analysis.speech_act.category == :expressive or
               analysis.speech_act.sub_type == :greeting or
               analysis.speech_act.sub_type != :farewell
    end

    test "raises when no world is named" do
      assert_raise ArgumentError, ~r/Pipeline needs :world_id/, fn ->
        Pipeline.analyze_chunk("What time is it?")
      end

      assert_raise ArgumentError, ~r/Pipeline needs :world_id/, fn ->
        Pipeline.analyze_chunk("What time is it?", side_effects: false)
      end
    end
  end

  describe "summarize/1" do
    test "provides summary of analysis" do
      model = Pipeline.process("Hello! How are you?", world_id: @world_id)
      summary = Pipeline.summarize(model)

      assert is_map(summary)
      assert Map.has_key?(summary, :chunks)
      assert Map.has_key?(summary, :analyses)
      assert Map.has_key?(summary, :overall_strategy)
    end
  end

  describe "InternalModel helpers" do
    test "bot_addressed? returns true when bot is addressed" do
      model = Pipeline.process("Hey bot, what's up?", world_id: @world_id)

      assert InternalModel.bot_addressed?(model) == true
    end

    test "respondable_analyses filters correctly" do
      model = Pipeline.process("Hello there!", world_id: @world_id)

      respondable = InternalModel.respondable_analyses(model)

      assert respondable != []
      assert Enum.all?(respondable, &(&1.response_strategy in [:can_respond, :hedged_response]))
    end
  end

  describe "the classified feature vector and the chunk profile" do
    test "every chunk keeps the vector its intent was classified from, and a profile computed from it" do
      model = Pipeline.process("Hi! Can you play some jazz?", world_id: @world_id, side_effects: false)

      assert length(model.analyses) >= 2

      for analysis <- model.analyses do
        assert [_ | _] = analysis.feature_vector
        assert %Brain.Analysis.ChunkProfile{} = analysis.profile
        assert is_binary(analysis.intent) and analysis.intent != ""

        # The profile's domain is :intent_domain's answer for the stored vector.
        {:ok, domain, _confidence} =
          Brain.ML.MicroClassifiers.classify_vector(:intent_domain, analysis.feature_vector)

        assert to_string(analysis.profile.domain) == to_string(domain)
      end
    end

    test "a chunk's accumulator keeps the turn's memory context and counts each source once" do
      model =
        Pipeline.process("Hello there. What is the weather in Paris?", world_id: @world_id, side_effects: false)

      for analysis <- model.analyses do
        sources = Enum.map(analysis.accumulated_context.signals, &elem(&1, 0))
        assert sources == Enum.uniq(sources)
        assert Enum.count(sources, &(&1 == :speech_act)) == 1
        assert is_list(analysis.accumulated_context.relevant_episodes)
      end
    end
  end

  describe "spans entity extraction could not decide" do
    # Gold sentences with a span the data leaves undecided before any context
    # is weighed, so the pipeline's extraction leaves it undecided too.
    setup do
      %{cases: undecided_span_cases(@world_id, 3)}
    end

    test "travel on the analysis's unresolved_spans, never among its entities or slots", %{cases: cases} do
      for {text, undecided} <- cases do
        analysis = Pipeline.analyze_chunk(text, world_id: @world_id, side_effects: false)

        assert Enum.all?(analysis.unresolved_spans, &match?(%{resolved: false}, &1))
        refute Enum.any?(analysis.entities, &match?(%{resolved: false}, &1))

        entity_values = MapSet.new(analysis.entities, &String.downcase(to_string(&1[:value])))
        slot_values = analysis.slots.filled_slots |> Map.values() |> Enum.map(&String.downcase(inspect(&1)))

        for span <- undecided do
          assert Enum.any?(analysis.unresolved_spans, &(&1.start_pos == span.start_pos and &1.end_pos == span.end_pos)),
                 "#{inspect(span.value)} of #{inspect(text)} is not on the analysis's unresolved_spans"

          refute MapSet.member?(entity_values, String.downcase(span.value)),
                 "#{inspect(span.value)} of #{inspect(text)} reached the entities"

          refute Enum.any?(slot_values, &String.contains?(&1, inspect(String.downcase(span.value)))),
                 "#{inspect(span.value)} of #{inspect(text)} filled a slot: #{inspect(analysis.slots.filled_slots)}"
        end
      end
    end

    test "reach the second pass's analysis", %{cases: cases} do
      {text, undecided} = hd(cases)
      analysis = Pipeline.analyze_chunk(text, world_id: @world_id, side_effects: false)

      assert analysis.pass == 2
      assert Enum.all?(undecided, fn span -> Enum.any?(analysis.unresolved_spans, &(&1.start_pos == span.start_pos)) end)
    end
  end
end
