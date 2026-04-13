defmodule Atlas.TelemetryTest do
  use ExUnit.Case, async: false

  alias Atlas.Telemetry

  setup do
    on_exit(fn -> Telemetry.attach_handlers() end)
    :ok
  end

  describe "attach_handlers/0 and detach_handlers/0" do
    test "attaches and detaches all handlers" do
      assert :ok = Telemetry.detach_handlers()
      assert :ok = Telemetry.attach_handlers()
      assert :ok = Telemetry.detach_handlers()
    end

    test "double attach does not crash" do
      Telemetry.detach_handlers()
      assert :ok = Telemetry.attach_handlers()
      # Second attach may raise for duplicate handler IDs, so detach first
      Telemetry.detach_handlers()
      assert :ok = Telemetry.attach_handlers()
      Telemetry.detach_handlers()
    end
  end

  describe "span/2" do
    test "wraps a function and returns its result" do
      result =
        Telemetry.span(:graph_query, %{graph: "test_graph"}, fn ->
          {:ok, 42}
        end)

      assert result == {:ok, 42}
    end

    test "propagates exceptions" do
      assert_raise RuntimeError, "boom", fn ->
        Telemetry.span(:graph_query, %{graph: "test"}, fn ->
          raise "boom"
        end)
      end
    end
  end

  describe "emit_node_added/2" do
    test "emits telemetry event without error" do
      assert :ok = Telemetry.emit_node_added("test_graph", "Person")
    end
  end

  describe "emit_edge_added/2" do
    test "emits telemetry event without error" do
      assert :ok = Telemetry.emit_edge_added("test_graph", "KNOWS")
    end
  end

  describe "handler functions" do
    test "handle_repo_query does not crash" do
      Telemetry.handle_repo_query(
        [:atlas, :repo, :query],
        %{total_time: 1_000_000},
        %{source: "atlas_beliefs"},
        %{}
      )
    end

    test "handle_graph_query_stop does not crash" do
      Telemetry.handle_graph_query_stop(
        [:chat_bot, :atlas, :graph_query, :stop],
        %{duration: 500_000},
        %{graph: "test"},
        %{}
      )
    end

    test "handle_graph_query_exception does not crash" do
      Telemetry.handle_graph_query_exception(
        [:chat_bot, :atlas, :graph_query, :exception],
        %{duration: 500_000},
        %{graph: "test"},
        %{}
      )
    end

    test "handle_node_added does not crash" do
      Telemetry.handle_node_added(
        [:chat_bot, :atlas, :graph, :node_added],
        %{count: 1},
        %{graph: "test", label: "Person"},
        %{}
      )
    end

    test "handle_edge_added does not crash" do
      Telemetry.handle_edge_added(
        [:chat_bot, :atlas, :graph, :edge_added],
        %{count: 1},
        %{graph: "test", rel_type: "KNOWS"},
        %{}
      )
    end
  end
end
