defmodule Brain.Graph.WriterTier3Test do
  @moduledoc """
  Tests for the Tier 3 graph writing functions: write_srl_triples and write_event_frames.
  Each test writes into its own world's graphs (`Brain.Test.GraphCase`).
  """

  use Brain.Test.GraphCase, async: false

  alias Brain.Graph.Writer

  describe "write_srl_triples/2" do
    test "raises without a world" do
      assert_raise ArgumentError, fn -> Writer.write_srl_triples([], nil) end
    end

    test "handles empty list", %{graph_world_id: world} do
      assert :ok = Writer.write_srl_triples([], world)
    end

    test "handles nil input", %{graph_world_id: world} do
      assert :ok = Writer.write_srl_triples(nil, world)
    end

    test "handles frames with no arguments (produces no triples)", %{graph_world_id: world} do
      frames = [%{predicate: "run", arguments: []}]
      assert :ok = Writer.write_srl_triples(frames, world)
    end

    test "handles frames with arguments", %{graph_world_id: world} do
      frames = [%{
        predicate: "visited",
        arguments: [
          %{role: "ARG0", text: "John"},
          %{role: "ARG1", text: "Berlin"}
        ]
      }]

      assert :ok = Writer.write_srl_triples(frames, world)
    end

    test "handles multiple frames", %{graph_world_id: world} do
      frames = [
        %{predicate: "ate", arguments: [
          %{role: "ARG0", text: "Alice"},
          %{role: "ARG1", text: "cake"}
        ]},
        %{predicate: "drank", arguments: [
          %{role: "ARG0", text: "Bob"},
          %{role: "ARG1", text: "coffee"}
        ]}
      ]

      assert :ok = Writer.write_srl_triples(frames, world)
    end
  end

  describe "write_event_frames/2" do
    test "raises without a world" do
      assert_raise ArgumentError, fn -> Writer.write_event_frames([], nil) end
    end

    test "handles empty list", %{graph_world_id: world} do
      assert :ok = Writer.write_event_frames([], world)
    end

    test "handles nil input", %{graph_world_id: world} do
      assert :ok = Writer.write_event_frames(nil, world)
    end

    test "handles frame with no arguments", %{graph_world_id: world} do
      frames = [%{trigger: "run", arguments: []}]
      assert :ok = Writer.write_event_frames(frames, world)
    end

    test "handles frame with arguments", %{graph_world_id: world} do
      frames = [%{
        trigger: "gave",
        arguments: [
          %{text: "John", role: "ARG0"},
          %{text: "book", role: "ARG1"},
          %{text: "Mary", role: "ARG2"}
        ]
      }]

      assert :ok = Writer.write_event_frames(frames, world)
      Process.sleep(200)

      assert_node_exists(Graph.world_graph("knowledge_graph", world), "LinkedEvent", "event_gave")
    end

    test "handles frames with missing trigger", %{graph_world_id: world} do
      frames = [%{arguments: [%{text: "test", role: "ARG0"}]}]
      assert :ok = Writer.write_event_frames(frames, world)
    end

    test "handles frames with missing arguments key", %{graph_world_id: world} do
      frames = [%{trigger: "test"}]
      assert :ok = Writer.write_event_frames(frames, world)
    end
  end

  describe "write_analysis/2 with enriched model" do
    test "handles model with all Tier 3 fields", %{graph_world_id: world} do
      model = %{
        analyses: [
          %{
            entities: [%{entity_type: "person", value: "John"}],
            events: [],
            event_frames: [%{trigger: "visited", arguments: [%{text: "Berlin", role: "ARG1"}]}],
            srl_frames: [%{predicate: "visited", arguments: [
              %{role: "ARG0", text: "John"},
              %{role: "ARG1", text: "Berlin"}
            ]}],
            pos_tags: []
          }
        ]
      }

      assert :ok = Writer.write_analysis(model, world)
    end

    test "handles model with empty Tier 3 fields", %{graph_world_id: world} do
      model = %{
        analyses: [
          %{
            entities: [],
            events: [],
            event_frames: [],
            srl_frames: [],
            pos_tags: []
          }
        ]
      }

      assert :ok = Writer.write_analysis(model, world)
    end

    test "handles model with missing Tier 3 fields", %{graph_world_id: world} do
      model = %{
        analyses: [
          %{
            entities: [],
            events: [],
            pos_tags: []
          }
        ]
      }

      assert :ok = Writer.write_analysis(model, world)
    end
  end
end
