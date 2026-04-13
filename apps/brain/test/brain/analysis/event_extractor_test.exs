defmodule Brain.Analysis.EventExtractorTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.EventExtractor
  alias Brain.Analysis.Types.Event
  alias Brain.Analysis.EventPatterns

  describe "extract/2 basic functionality" do
    test "extracts actor-verb-object from POS tags" do
      analysis = %{
        pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
        entities: [],
        tokens: ["I", "want", "coffee"]
      }

      {:ok, events} = EventExtractor.extract(analysis)

      assert length(events) == 1
      [event] = events

      assert %Event{} = event
      assert event.action.verb == "want"
      assert event.action.lemma == "want"
      assert event.actor.text == "I"
      assert event.object.text == "coffee"
    end

    test "extracts events from imperative sentences" do
      analysis = %{
        pos_tags: [{"Play", "VERB"}, {"music", "NOUN"}],
        entities: [],
        tokens: ["Play", "music"]
      }

      {:ok, events} = EventExtractor.extract(analysis)

      assert events != []
      [event | _] = events

      assert event.action.verb == "Play"
      assert event.action.tense == :imperative
      assert event.object.text == "music"
    end

    test "links entities to event roles" do
      analysis = %{
        pos_tags: [{"play", "VERB"}, {"Beatles", "PROPN"}],
        entities: [%{text: "Beatles", type: "artist", start: 1, end: 1}],
        tokens: ["play", "Beatles"]
      }

      {:ok, events} = EventExtractor.extract(analysis)

      assert events != []
      [event | _] = events

      assert event.action.verb == "play"
      assert event.object.text == "Beatles"
      assert event.object.entity_type == "artist"
    end

    test "handles multiple verbs" do
      analysis = %{
        pos_tags: [
          {"I", "PRON"},
          {"want", "VERB"},
          {"to", "PART"},
          {"play", "VERB"},
          {"music", "NOUN"}
        ],
        entities: [],
        tokens: ["I", "want", "to", "play", "music"]
      }

      {:ok, events} = EventExtractor.extract(analysis, max_events: 5)
      assert events != []
    end

    test "returns empty list for no verbs" do
      analysis = %{
        pos_tags: [{"the", "DET"}, {"coffee", "NOUN"}],
        entities: [],
        tokens: ["the", "coffee"]
      }

      {:ok, events} = EventExtractor.extract(analysis)

      assert events == []
    end

    test "respects min_confidence option" do
      analysis = %{
        pos_tags: [{"want", "VERB"}],
        entities: [],
        tokens: ["want"]
      }

      {:ok, events} = EventExtractor.extract(analysis, min_confidence: 0.9)
      assert events == []
    end

    test "handles invalid input gracefully" do
      {:error, :invalid_input} = EventExtractor.extract(%{})
    end
  end

  describe "tensor operations" do
    test "pos_tags_to_tensor converts tags to indices" do
      pos_tags = [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}]
      tensor = EventExtractor.pos_tags_to_tensor(pos_tags)
      indices = Nx.to_flat_list(tensor)
      assert indices == [1, 2, 3]
    end

    test "find_verb_positions finds VERB indices" do
      pos_tensor = Nx.tensor([1, 2, 3, 2, 3], type: :s32)
      verb_mask = EventExtractor.find_verb_positions(pos_tensor)
      mask_list = Nx.to_flat_list(verb_mask)
      assert mask_list == [0, 1, 0, 1, 0]
    end

    test "find_actor_positions finds PRON/NOUN/PROPN indices" do
      pos_tensor = Nx.tensor([1, 2, 3, 4, 2], type: :s32)
      actor_mask = EventExtractor.find_actor_positions(pos_tensor)

      mask_list = Nx.to_flat_list(actor_mask)
      assert mask_list == [1, 0, 1, 1, 0]
    end

    test "find_nearest_before finds correct position" do
      mask = Nx.tensor([1, 0, 1, 0, 0], type: :u8)
      target_idx = 3

      nearest = EventExtractor.find_nearest_before(mask, target_idx)
      assert Nx.to_number(nearest) == 2
    end

    test "find_nearest_after finds correct position" do
      mask = Nx.tensor([0, 0, 0, 1, 1], type: :u8)
      target_idx = 1

      nearest = EventExtractor.find_nearest_after(mask, target_idx)
      assert Nx.to_number(nearest) == 3
    end
  end

  describe "Event struct" do
    test "new/2 creates event with generated ID" do
      action = %{verb: "want", lemma: "want", tense: :present}
      event = Event.new(action, object: %{text: "coffee", type: "noun", token_index: 2})

      assert event.id =~ ~r/^evt_[a-f0-9]{16}$/
      assert event.action == action
      assert event.object.text == "coffee"
    end

    test "complete?/1 checks for actor and object" do
      action = %{verb: "want", lemma: "want", tense: :present}

      complete =
        Event.new(action,
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2}
        )

      incomplete = Event.new(action, object: %{text: "coffee", type: "noun", token_index: 2})

      assert Event.complete?(complete)
      refute Event.complete?(incomplete)
    end

    test "imperative?/1 checks tense" do
      imperative = Event.new(%{verb: "Play", lemma: "play", tense: :imperative})
      declarative = Event.new(%{verb: "want", lemma: "want", tense: :present})

      assert Event.imperative?(imperative)
      refute Event.imperative?(declarative)
    end

    test "to_description/1 formats event" do
      action = %{verb: "want", lemma: "want", tense: :present}

      event =
        Event.new(action,
          actor: %{text: "I", type: "pronoun", token_index: 0},
          object: %{text: "coffee", type: "noun", token_index: 2}
        )

      assert Event.to_description(event) == "I want coffee"
    end
  end

  describe "EventPatterns module" do
    test "loads patterns from JSON" do
      patterns = EventPatterns.patterns()
      assert is_list(patterns)
      assert patterns != []
    end

    test "pos_to_index converts tags" do
      assert EventPatterns.pos_to_index("VERB") == 2
      assert EventPatterns.pos_to_index("NOUN") == 3
      assert EventPatterns.pos_to_index("UNKNOWN") == 0
    end

    test "pattern_indices returns index list" do
      indices = EventPatterns.pattern_indices("svo_basic")
      assert indices == [1, 2, 3]
    end

    test "is_action_index? identifies verb indices" do
      assert EventPatterns.is_action_index?(2)
      refute EventPatterns.is_action_index?(1)
      refute EventPatterns.is_action_index?(3)
    end

    test "is_actor_index? identifies actor indices" do
      assert EventPatterns.is_actor_index?(1)
      assert EventPatterns.is_actor_index?(3)
      assert EventPatterns.is_actor_index?(4)
      refute EventPatterns.is_actor_index?(2)
    end
  end

  describe "parallel extraction" do
    test "extract_parallel processes multiple chunks" do
      chunks = [
        %{
          pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
          entities: [],
          tokens: ["I", "want", "coffee"]
        },
        %{
          pos_tags: [{"She", "PRON"}, {"plays", "VERB"}, {"music", "NOUN"}],
          entities: [],
          tokens: ["She", "plays", "music"]
        }
      ]

      {:ok, events} = EventExtractor.extract_parallel(chunks)
      assert events != []
    end

    test "extract_parallel handles empty chunks" do
      chunks = []
      {:ok, events} = EventExtractor.extract_parallel(chunks)
      assert events == []
    end

    test "extract_parallel handles mixed valid/invalid chunks" do
      chunks = [
        %{
          pos_tags: [{"I", "PRON"}, {"want", "VERB"}, {"coffee", "NOUN"}],
          entities: [],
          tokens: ["I", "want", "coffee"]
        },
        %{invalid: true}
      ]

      {:ok, events} = EventExtractor.extract_parallel(chunks)
      assert events != []
    end
  end

  describe "backend verification" do
    test "backend_info returns current backend" do
      info = EventExtractor.backend_info()

      assert Map.has_key?(info, :backend)
      assert Map.has_key?(info, :exla_available)
    end
  end
end