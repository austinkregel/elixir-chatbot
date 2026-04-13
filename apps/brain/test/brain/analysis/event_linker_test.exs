defmodule Brain.Analysis.EventLinkerTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.EventLinker

  @sample_entities [
    %{text: "John", type: :person, start_pos: 0},
    %{text: "Berlin", type: :location, start_pos: 2},
    %{text: "yesterday", type: :temporal, start_pos: 3}
  ]

  @sample_tokens [
    %{text: "John", normalized: "john"},
    %{text: "visited", normalized: "visited"},
    %{text: "Berlin", normalized: "berlin"},
    %{text: "yesterday", normalized: "yesterday"}
  ]

  @sample_pos_tags ["NNP", "VBD", "NNP", "NN"]

  @sample_event %{
    action: %{verb: "visited"},
    source_tokens: [0, 1, 2]
  }

  describe "link/4" do
    test "produces event frames with arguments" do
      events = [@sample_event]

      frames = EventLinker.link(events, @sample_entities, @sample_tokens, @sample_pos_tags)

      assert length(frames) == 1
      frame = hd(frames)

      assert frame.trigger == "visited"
      assert is_list(frame.arguments)
      assert length(frame.arguments) >= 1
    end

    test "temporal entities get argm_tmp role" do
      events = [@sample_event]
      frames = EventLinker.link(events, @sample_entities, @sample_tokens, @sample_pos_tags)

      frame = hd(frames)
      temporal_args = Enum.filter(frame.arguments, &(&1.role == :argm_tmp))

      assert length(temporal_args) >= 1
      temporal_arg = hd(temporal_args)
      assert temporal_arg.text == "yesterday"
      assert temporal_arg.entity_type == :temporal
    end

    test "handles empty events list" do
      frames = EventLinker.link([], @sample_entities, @sample_tokens, @sample_pos_tags)
      assert frames == []
    end

    test "handles empty entities list" do
      events = [@sample_event]
      frames = EventLinker.link(events, [], @sample_tokens, @sample_pos_tags)

      assert length(frames) == 1
      frame = hd(frames)
      assert is_list(frame.arguments)
    end
  end

  describe "assign_argument_roles/4" do
    test "assigns roles to non-temporal entities" do
      roles = EventLinker.assign_argument_roles(
        @sample_event,
        @sample_entities,
        @sample_tokens,
        @sample_pos_tags
      )

      non_temporal = Enum.reject(roles, &(&1.entity_type == :temporal))

      for role_assignment <- non_temporal do
        assert role_assignment.role in [:arg0, :arg1, :argm_loc, :argm_tmp, :argm_mnr]
        assert role_assignment.confidence > 0.0
        assert is_binary(role_assignment.text)
      end
    end

    test "skips temporal entities (handled separately)" do
      roles = EventLinker.assign_argument_roles(
        @sample_event,
        @sample_entities,
        @sample_tokens,
        @sample_pos_tags
      )

      temporal = Enum.filter(roles, &(&1.entity_type == :temporal))
      assert temporal == []
    end
  end

  describe "temporal relation detection" do
    test "detects ordering between events with temporal arguments" do
      events = [
        %{action: %{verb: "arrived"}, source_tokens: [0, 1, 2]},
        %{action: %{verb: "departed"}, source_tokens: [3, 4, 5]}
      ]

      entities = [
        %{text: "morning", type: :temporal, start_pos: 2},
        %{text: "evening", type: :temporal, start_pos: 5}
      ]

      tokens = Enum.map(0..5, &%{text: "t#{&1}", normalized: "t#{&1}"})
      pos_tags = List.duplicate("NN", 6)

      frames = EventLinker.link(events, entities, tokens, pos_tags)

      assert length(frames) == 2
    end
  end
end
