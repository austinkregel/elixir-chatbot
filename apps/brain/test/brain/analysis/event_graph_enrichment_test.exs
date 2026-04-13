defmodule Brain.Analysis.EventGraphEnrichmentTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.Analysis.EventLinker

  describe "event graph enrichment pipeline" do
    test "events are enriched with argument roles and temporal info" do
      events = [
        %{action: %{verb: "visited"}, source_tokens: [0, 1, 2]},
        %{action: %{verb: "returned"}, source_tokens: [4, 5, 6]}
      ]

      entities = [
        %{text: "The president", type: :person, start_pos: 0},
        %{text: "Berlin", type: :location, start_pos: 2},
        %{text: "yesterday", type: :temporal, start_pos: 3},
        %{text: "today", type: :temporal, start_pos: 6}
      ]

      tokens = Enum.map(0..6, &%{text: "t#{&1}", normalized: "t#{&1}"})
      pos_tags = List.duplicate("NN", 7)

      frames = EventLinker.link(events, entities, tokens, pos_tags)

      assert length(frames) == 2

      for frame <- frames do
        assert is_binary(frame.trigger)
        assert is_list(frame.arguments)
        assert is_list(frame.temporal_relations)
        assert is_list(frame.sub_events)

        for arg <- frame.arguments do
          assert Map.has_key?(arg, :text)
          assert Map.has_key?(arg, :role)
          assert Map.has_key?(arg, :confidence)
          assert arg.role in [:arg0, :arg1, :argm_loc, :argm_tmp, :argm_mnr]
        end
      end
    end

    test "enriched events produce graph-writable structure" do
      events = [%{action: %{verb: "announced"}, source_tokens: [0, 1, 2]}]

      entities = [
        %{text: "CEO", type: :person, start_pos: 0},
        %{text: "merger", type: :thing, start_pos: 2}
      ]

      tokens = [
        %{text: "CEO", normalized: "ceo"},
        %{text: "announced", normalized: "announced"},
        %{text: "merger", normalized: "merger"}
      ]

      pos_tags = ["NNP", "VBD", "NN"]

      frames = EventLinker.link(events, entities, tokens, pos_tags)

      frame = hd(frames)

      assert frame.trigger == "announced"
      assert length(frame.arguments) >= 1

      # Verify the frame structure is suitable for graph writing
      assert is_binary(frame.trigger)
      assert is_integer(frame.trigger_index)

      # Each argument should have the fields needed for graph edges
      for arg <- frame.arguments do
        assert is_binary(arg.text)
        assert is_atom(arg.role)
        assert is_float(arg.confidence) or is_integer(arg.confidence)
      end
    end
  end
end
