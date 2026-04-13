defmodule Atlas.Stats.Collector do
  @moduledoc """
  GenServer that collects Atlas telemetry metrics via cast messages.

  Aggregates Ecto query durations, graph query durations, node/edge counts,
  and exposes them to the dashboard via `Atlas.Stats.get_overview/0`.
  """

  use GenServer
  require Logger

  @initial_state %{
    query_metrics: %{},
    graph_query_metrics: %{},
    graph_errors: %{},
    nodes_added: %{},
    edges_added: %{},
    total_queries: 0,
    total_graph_queries: 0
  }

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Get all collected metrics."
  def get_metrics do
    if Process.whereis(__MODULE__) do
      GenServer.call(__MODULE__, :get_metrics, 5_000)
    else
      @initial_state
    end
  end

  @doc "Reset all collected metrics."
  def reset do
    if Process.whereis(__MODULE__) do
      GenServer.cast(__MODULE__, :reset)
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    {:ok, @initial_state}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_cast({:record_query, source, duration_ms, _metadata}, state) do
    source_key = source || "unknown"

    query_metrics =
      Map.update(state.query_metrics, source_key, new_metric(duration_ms), fn existing ->
        update_metric(existing, duration_ms)
      end)

    {:noreply, %{state | query_metrics: query_metrics, total_queries: state.total_queries + 1}}
  end

  def handle_cast({:record_graph_query, graph, duration_ms}, state) do
    graph_key = graph || "unknown"

    graph_query_metrics =
      Map.update(state.graph_query_metrics, graph_key, new_metric(duration_ms), fn existing ->
        update_metric(existing, duration_ms)
      end)

    {:noreply,
     %{
       state
       | graph_query_metrics: graph_query_metrics,
         total_graph_queries: state.total_graph_queries + 1
     }}
  end

  def handle_cast({:record_graph_error, graph, _duration_ms}, state) do
    graph_key = graph || "unknown"

    graph_errors =
      Map.update(state.graph_errors, graph_key, 1, &(&1 + 1))

    {:noreply, %{state | graph_errors: graph_errors}}
  end

  def handle_cast({:record_node_added, graph, label}, state) do
    key = {graph || "unknown", label || "unknown"}

    nodes_added =
      Map.update(state.nodes_added, key, 1, &(&1 + 1))

    {:noreply, %{state | nodes_added: nodes_added}}
  end

  def handle_cast({:record_edge_added, graph, rel_type}, state) do
    key = {graph || "unknown", rel_type || "unknown"}

    edges_added =
      Map.update(state.edges_added, key, 1, &(&1 + 1))

    {:noreply, %{state | edges_added: edges_added}}
  end

  def handle_cast(:reset, _state) do
    {:noreply, @initial_state}
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp new_metric(duration_ms) do
    %{
      count: 1,
      total_ms: duration_ms,
      min_ms: duration_ms,
      max_ms: duration_ms,
      avg_ms: duration_ms / 1
    }
  end

  defp update_metric(existing, duration_ms) do
    count = existing.count + 1
    total = existing.total_ms + duration_ms

    %{
      existing
      | count: count,
        total_ms: total,
        min_ms: min(existing.min_ms, duration_ms),
        max_ms: max(existing.max_ms, duration_ms),
        avg_ms: total / count
    }
  end
end
