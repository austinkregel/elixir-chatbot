defmodule Brain.Graph.WriterTier3Test do
  @moduledoc """
  Tests for the Tier 3 graph writing functions: write_srl_triples and write_event_frames.
  """

  use ExUnit.Case, async: false

  alias Brain.Graph.Writer

  describe "write_srl_triples/1" do
    test "handles empty list" do
      assert :ok = Writer.write_srl_triples([])
    end

    test "handles nil input" do
      assert :ok = Writer.write_srl_triples(nil)
    end

    test "handles frames with no arguments (produces no triples)" do
      frames = [%{predicate: "run", arguments: []}]
      assert :ok = Writer.write_srl_triples(frames)
    end

    test "handles frames with arguments" do
      frames = [%{
        predicate: "visited",
        arguments: [
          %{role: "ARG0", text: "John"},
          %{role: "ARG1", text: "Berlin"}
        ]
      }]

      assert :ok = Writer.write_srl_triples(frames)
    end

    test "handles multiple frames" do
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

      assert :ok = Writer.write_srl_triples(frames)
    end
  end

  describe "write_event_frames/1" do
    test "handles empty list" do
      assert :ok = Writer.write_event_frames([])
    end

    test "handles nil input" do
      assert :ok = Writer.write_event_frames(nil)
    end

    test "handles frame with no arguments" do
      frames = [%{trigger: "run", arguments: []}]
      assert :ok = Writer.write_event_frames(frames)
    end

    test "handles frame with arguments" do
      frames = [%{
        trigger: "gave",
        arguments: [
          %{text: "John", role: "ARG0"},
          %{text: "book", role: "ARG1"},
          %{text: "Mary", role: "ARG2"}
        ]
      }]

      assert :ok = Writer.write_event_frames(frames)
    end

    test "handles frames with missing trigger" do
      frames = [%{arguments: [%{text: "test", role: "ARG0"}]}]
      assert :ok = Writer.write_event_frames(frames)
    end

    test "handles frames with missing arguments key" do
      frames = [%{trigger: "test"}]
      assert :ok = Writer.write_event_frames(frames)
    end
  end

  describe "write_analysis/1 with enriched model" do
    test "handles model with all Tier 3 fields" do
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

      assert :ok = Writer.write_analysis(model)
    end

    test "handles model with empty Tier 3 fields" do
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

      assert :ok = Writer.write_analysis(model)
    end

    test "handles model with missing Tier 3 fields" do
      model = %{
        analyses: [
          %{
            entities: [],
            events: [],
            pos_tags: []
          }
        ]
      }

      assert :ok = Writer.write_analysis(model)
    end
  end
end
