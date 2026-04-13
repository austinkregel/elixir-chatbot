defmodule Brain.Analysis.EventLinkerExtendedTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.EventLinker

  describe "link/4 edge cases" do
    test "empty events returns empty list" do
      result = EventLinker.link([], [], [], [])
      assert result == []
    end

    test "events with no matching entities" do
      events = [%{action: %{verb: "run"}, source_tokens: [0, 1]}]
      entities = []
      tokens = [%{text: "they", normalized: "they"}, %{text: "run", normalized: "run"}]
      pos_tags = ["PRP", "VB"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert length(frames) == 1
      assert hd(frames).trigger == "run"
      assert is_list(hd(frames).arguments)
    end

    test "multiple events with same trigger" do
      events = [
        %{action: %{verb: "run"}, source_tokens: [1]},
        %{action: %{verb: "run"}, source_tokens: [4]}
      ]

      entities = [%{text: "John", type: :person, start_pos: 0}]
      tokens = Enum.map(0..5, &%{text: "t#{&1}", normalized: "t#{&1}"})
      pos_tags = List.duplicate("NN", 6)

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert length(frames) == 2
    end

    test "event with temporal entity assigns argm_tmp role" do
      events = [%{action: %{verb: "met"}, source_tokens: [1]}]
      entities = [
        %{text: "John", type: :person, start_pos: 0},
        %{text: "yesterday", type: :temporal, start_pos: 2}
      ]

      tokens = [
        %{text: "John", normalized: "john"},
        %{text: "met", normalized: "met"},
        %{text: "yesterday", normalized: "yesterday"}
      ]
      pos_tags = ["NNP", "VBD", "NN"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert length(frames) == 1
      frame = hd(frames)

      temporal_args = Enum.filter(frame.arguments, fn arg ->
        arg.role == :argm_tmp
      end)

      assert length(temporal_args) >= 1
    end

    test "POS tags shorter than tokens" do
      events = [%{action: %{verb: "go"}, source_tokens: [0]}]
      entities = []
      tokens = [%{text: "go", normalized: "go"}, %{text: "now", normalized: "now"}]
      pos_tags = ["VB"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert is_list(frames)
    end

    test "event with malformed action map" do
      events = [%{action: %{}, source_tokens: [0]}]
      entities = []
      tokens = [%{text: "word", normalized: "word"}]
      pos_tags = ["NN"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert is_list(frames)
    end

    test "entities with various type formats" do
      events = [%{action: %{verb: "saw"}, source_tokens: [1]}]
      entities = [
        %{text: "Berlin", type: "location", start_pos: 2},
        %{text: "John", type: :person, start_pos: 0}
      ]

      tokens = [
        %{text: "John", normalized: "john"},
        %{text: "saw", normalized: "saw"},
        %{text: "Berlin", normalized: "berlin"}
      ]
      pos_tags = ["NNP", "VBD", "NNP"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)
      assert length(frames) == 1
      assert length(hd(frames).arguments) >= 1
    end
  end

  describe "assign_argument_roles/4" do
    test "returns empty list for no entities" do
      event = %{action: %{verb: "go"}, source_tokens: [0]}

      result = EventLinker.assign_argument_roles(
        event, [],
        [%{text: "go", normalized: "go"}],
        ["VB"]
      )
      assert result == []
    end

    test "filters out temporal entities" do
      event = %{action: %{verb: "walked"}, source_tokens: [2]}
      entities = [
        %{text: "today", type: :temporal, start_pos: 0},
        %{text: "John", type: :person, start_pos: 1}
      ]

      result = EventLinker.assign_argument_roles(
        event, entities,
        [%{text: "today", normalized: "today"}, %{text: "John", normalized: "john"}, %{text: "walked", normalized: "walked"}],
        ["NN", "NNP", "VBD"]
      )

      temporal_in_result = Enum.filter(result, fn arg ->
        arg.entity_type == :temporal
      end)

      assert length(temporal_in_result) == 0
    end
  end
end
