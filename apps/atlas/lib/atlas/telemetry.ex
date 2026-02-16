defmodule Atlas.Telemetry do
  @moduledoc """
  Telemetry events for Atlas database and graph operations.

  ## Ecto Built-in Events (automatic)

  - `[:atlas, :repo, :query]` -- every Ecto query with duration, source table, result

  ## Custom Atlas Events

  - `[:chat_bot, :atlas, :graph_query, :start | :stop | :exception]` -- Cypher query timing
  - `[:chat_bot, :atlas, :graph, :node_added]` -- node creation events
  - `[:chat_bot, :atlas, :graph, :edge_added]` -- edge creation events
  """

  require Logger

  @graph_query [:chat_bot, :atlas, :graph_query]
  @node_added [:chat_bot, :atlas, :graph, :node_added]
  @edge_added [:chat_bot, :atlas, :graph, :edge_added]

  @doc """
  Attaches all Atlas telemetry handlers. Called during application startup.
  """
  def attach_handlers do
    handlers = [
      {"atlas-repo-query",
       [:atlas, :repo, :query],
       &__MODULE__.handle_repo_query/4, %{}},

      {"atlas-graph-query-stop",
       @graph_query ++ [:stop],
       &__MODULE__.handle_graph_query_stop/4, %{}},

      {"atlas-graph-query-exception",
       @graph_query ++ [:exception],
       &__MODULE__.handle_graph_query_exception/4, %{}},

      {"atlas-graph-node-added",
       @node_added,
       &__MODULE__.handle_node_added/4, %{}},

      {"atlas-graph-edge-added",
       @edge_added,
       &__MODULE__.handle_edge_added/4, %{}}
    ]

    Enum.each(handlers, fn {id, event, handler, config} ->
      :telemetry.attach(id, event, handler, config)
    end)

    :ok
  end

  @doc """
  Detaches all Atlas telemetry handlers.
  """
  def detach_handlers do
    handler_ids = [
      "atlas-repo-query",
      "atlas-graph-query-stop",
      "atlas-graph-query-exception",
      "atlas-graph-node-added",
      "atlas-graph-edge-added"
    ]

    Enum.each(handler_ids, fn id ->
      :telemetry.detach(id)
    end)

    :ok
  end

  @doc """
  Wraps a function with graph query telemetry span measurement.
  """
  def span(:graph_query, metadata, fun) do
    :telemetry.span(@graph_query, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  @doc "Emits a node creation event."
  def emit_node_added(graph, label) do
    :telemetry.execute(
      @node_added,
      %{count: 1},
      %{graph: graph, label: label, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc "Emits an edge creation event."
  def emit_edge_added(graph, rel_type) do
    :telemetry.execute(
      @edge_added,
      %{count: 1},
      %{graph: graph, rel_type: rel_type, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  # ============================================================================
  # Handler Functions
  # ============================================================================

  @doc false
  def handle_repo_query(_event, measurements, metadata, _config) do
    if Process.whereis(Atlas.Stats.Collector) do
      duration_ms = div(measurements.total_time || 0, 1_000_000)

      GenServer.cast(
        Atlas.Stats.Collector,
        {:record_query, metadata[:source], duration_ms, metadata}
      )
    end
  end

  @doc false
  def handle_graph_query_stop(_event, measurements, metadata, _config) do
    if Process.whereis(Atlas.Stats.Collector) do
      duration_ms = System.convert_time_unit(measurements[:duration] || 0, :native, :millisecond)

      GenServer.cast(
        Atlas.Stats.Collector,
        {:record_graph_query, metadata[:graph], duration_ms}
      )
    end
  end

  @doc false
  def handle_graph_query_exception(_event, measurements, metadata, _config) do
    if Process.whereis(Atlas.Stats.Collector) do
      duration_ms = System.convert_time_unit(measurements[:duration] || 0, :native, :millisecond)

      GenServer.cast(
        Atlas.Stats.Collector,
        {:record_graph_error, metadata[:graph], duration_ms}
      )
    end
  end

  @doc false
  def handle_node_added(_event, _measurements, metadata, _config) do
    if Process.whereis(Atlas.Stats.Collector) do
      GenServer.cast(
        Atlas.Stats.Collector,
        {:record_node_added, metadata[:graph], metadata[:label]}
      )
    end
  end

  @doc false
  def handle_edge_added(_event, _measurements, metadata, _config) do
    if Process.whereis(Atlas.Stats.Collector) do
      GenServer.cast(
        Atlas.Stats.Collector,
        {:record_edge_added, metadata[:graph], metadata[:rel_type]}
      )
    end
  end
end
