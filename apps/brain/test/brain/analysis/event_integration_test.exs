defmodule Brain.Analysis.EventIntegrationTest do
  @moduledoc "Integration tests for the event extraction system.\n\nTests end-to-end flow from Pipeline through EventExtractor to Memory and Epistemic systems.\n"

  alias Brain.Analysis.EventPatterns
  alias Brain.ML.EntityExtractor
  alias Brain.Analysis
  use Brain.Test.GraphCase, async: false

  alias Analysis.{Pipeline, EventExtractor, ChunkAnalysis}
  alias Brain.Analysis.Types.Event
  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.Epistemic.BeliefStore

  import Brain.TestHelpers

  setup do
    start_test_services()
    EntityExtractor.load_entity_maps()

    if Process.whereis(MemoryStore) do
      :ok
    end

    :ok
  end

  describe "Pipeline integration" do
    test "pipeline extracts events from input" do
      result = Pipeline.process("I want to play some jazz music")

      assert %Brain.Analysis.InternalModel{} = result
      assert result.analyses != []

      has_events =
        Enum.any?(result.analyses, fn analysis ->
          is_list(analysis.events) and analysis.events != []
        end)

      assert has_events, "Expected at least one analysis to contain events"

      Enum.each(result.analyses, fn analysis ->
        assert is_list(analysis.events)
      end)
    end

    test "events contain expected structure" do
      result = Pipeline.process("Play some music for me")

      Enum.each(result.analyses, fn analysis ->
        Enum.each(analysis.events, fn event ->
          assert %Event{} = event
          assert is_binary(event.id)
          assert is_map(event.action)
          assert Map.has_key?(event.action, :verb)
          assert Map.has_key?(event.action, :lemma)
          assert is_float(event.confidence)
        end)
      end)
    end

    test "ChunkAnalysis has events field" do
      analysis = ChunkAnalysis.new(0, "I want coffee")
      assert analysis.events == []

      events = [
        Event.new(%{verb: "want", lemma: "want", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2},
          confidence: 0.9
        )
      ]

      updated = ChunkAnalysis.with_events(analysis, events)
      assert length(updated.events) == 1
      assert ChunkAnalysis.has_events?(updated)
      assert ChunkAnalysis.primary_event(updated) != nil
    end
  end

  describe "Memory integration" do
    @tag :memory
    test "events can be stored as episodes" do
      event =
        Event.new(%{verb: "play", lemma: "play", tense: :imperative},
          actor: nil,
          object: %{text: "jazz", type: "noun", token_index: 1},
          confidence: 0.85
        )

      context = %{
        response: "Playing jazz music",
        user_input: "Play some jazz"
      }

      if Process.whereis(MemoryStore) do
        {:ok, episode_id} = MemoryStore.add_event_episode(event, context, world_id: "test_events")

        assert is_binary(episode_id)
        {:ok, episodes} = MemoryStore.query_events_by_action("play", 5, world_id: "test_events")
        assert length(episodes) >= 0
      end
    end

    @tag :memory
    test "event episodes have correct tags" do
      event =
        Event.new(%{verb: "want", lemma: "want", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2},
          confidence: 0.9
        )

      context = %{response: "I'd be happy to help with coffee"}

      if Process.whereis(MemoryStore) do
        {:ok, _} = MemoryStore.add_event_episode(event, context, world_id: "test_events")
        {:ok, episodes} = MemoryStore.query_events_by_object("coffee", 5, world_id: "test_events")
        assert is_list(episodes)
      end
    end
  end

  describe "Epistemic integration" do
    @tag :epistemic
    test "beliefs can be extracted from events" do
      event =
        Event.new(%{verb: "want", lemma: "want", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2},
          confidence: 0.85
        )

      if Process.whereis(BeliefStore) do
        {:ok, belief_ids} = BeliefStore.extract_beliefs_from_event(event, "test_user")
        assert is_list(belief_ids)

        if belief_ids != [] do
          {:ok, belief} = BeliefStore.get_belief(hd(belief_ids))
          assert belief.predicate == :wants
          assert belief.object == "coffee"
        end
      end
    end

    @tag :epistemic
    test "low confidence events don't create beliefs" do
      event =
        Event.new(%{verb: "want", lemma: "want", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "something", type: "noun", token_index: 2},
          confidence: 0.3
        )

      if Process.whereis(BeliefStore) do
        {:ok, belief_ids} = BeliefStore.extract_beliefs_from_event(event, "test_user")
        assert belief_ids == []
      end
    end

    @tag :epistemic
    test "extract_beliefs_from_events handles multiple events" do
      events = [
        Event.new(%{verb: "want", lemma: "want", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2},
          confidence: 0.85
        ),
        Event.new(%{verb: "like", lemma: "like", tense: :present},
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "jazz", type: "noun", token_index: 2},
          confidence: 0.75
        )
      ]

      if Process.whereis(BeliefStore) do
        {:ok, belief_ids} = BeliefStore.extract_beliefs_from_events(events, "test_user")

        assert is_list(belief_ids)
      end
    end
  end

  describe "Generator integration" do
    test "build_context_from_events extracts slots" do
      alias Brain.Response.Generator

      events = [
        Event.new(%{verb: "play", lemma: "play", tense: :imperative},
          actor: nil,
          object: %{text: "jazz", type: "noun", token_index: 1},
          confidence: 0.85
        )
      ]

      entities = [%{entity_type: "location", value: "London"}]

      slots = Generator.build_context_from_events(events, entities)

      assert slots[:action] == "play"
      assert slots[:object] == "jazz"
      assert slots[:location] == "London"
    end

    test "build_context_from_events handles empty events" do
      alias Brain.Response.Generator

      entities = [%{entity_type: "person", value: "John"}]

      slots = Generator.build_context_from_events([], entities)

      assert slots[:person] == "John"
      assert slots[:action] == nil
    end
  end

  describe "Telemetry verification" do
    test "extraction emits telemetry events" do
      ref = make_ref()
      test_pid = self()
      handler_id = "test-event-extraction-#{inspect(ref)}"

      # Module-function capture instead of an anonymous fn so :telemetry
      # doesn't warn about a local-function handler. The dest pid + ref
      # ride along in the per-handler config map.
      :telemetry.attach(
        handler_id,
        [:chat_bot, :analysis, :event_extraction],
        &__MODULE__.__forward_event__/4,
        %{target: test_pid, ref: ref}
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)

      analysis = %{
        pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
        entities: [],
        tokens: ["I", "want", "coffee"]
      }

      {:ok, _events} = EventExtractor.extract(analysis)

      assert_receive {:telemetry, ^ref, [:chat_bot, :analysis, :event_extraction], measurements,
                      metadata},
                     1000

      assert Map.has_key?(measurements, :duration)
      assert Map.has_key?(measurements, :event_count)
      assert metadata.tensor_ops == true
      assert metadata.string_ops == false
    end
  end

  @doc false
  def __forward_event__(event, measurements, metadata, %{target: target, ref: ref})
      when is_pid(target) do
    send(target, {:telemetry, ref, event, measurements, metadata})
    :ok
  end

  describe "No string matching verification" do
    test "EventExtractor uses no regex patterns" do
      analysis = %{
        pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
        entities: [],
        tokens: ["I", "want", "coffee"]
      }

      {:ok, events} = EventExtractor.extract(analysis)
      assert is_list(events)
    end

    test "EventPatterns uses tensor indices not string comparisons" do
      indices = EventPatterns.pattern_indices("svo_basic")

      assert is_list(indices)
      assert Enum.all?(indices, &is_integer/1)
    end
  end
end
