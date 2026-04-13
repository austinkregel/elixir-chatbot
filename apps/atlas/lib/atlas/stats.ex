defmodule Atlas.Stats do
  @moduledoc """
  Collects Atlas operational stats for the dashboard.

  Results are cached for `@cache_ttl_ms` so that multiple LiveView connections
  polling every few seconds share a single set of queries instead of each
  connection issuing its own.

  Provides a unified view of:
  - Database connection pool status
  - Row counts per relational table
  - Graph node/edge counts per graph schema
  - Migration status
  - Query performance metrics from telemetry
  """

  require Logger

  @tables ~w(atlas_credentials atlas_beliefs atlas_episodes atlas_semantic_facts atlas_review_candidates atlas_learned_facts)

  @cache_ttl_ms 5_000

  @doc """
  Get a complete overview of Atlas operational stats.

  Returns cached results if they are less than #{@cache_ttl_ms}ms old.
  Otherwise runs the queries once and caches the result for all callers.
  """
  def get_overview do
    case read_cache() do
      {:ok, cached} ->
        cached

      :miss ->
        result = fetch_overview()
        write_cache(result)
        result
    end
  end

  @doc "Check if Atlas.Repo is connected and responding (no cache)."
  def connected? do
    if Process.whereis(Atlas.Repo) do
      try do
        case Ecto.Adapters.SQL.query(Atlas.Repo, "SELECT 1", []) do
          {:ok, _} -> true
          _ -> false
        end
      rescue
        _ -> false
      end
    else
      false
    end
  end

  @doc "Get row counts for all relational tables."
  def repo_stats do
    repo_stats(connected?())
  end

  @doc "Get migration status as a list of {version, name, status} tuples."
  def migration_status do
    migration_status(connected?())
  end

  @doc "Get telemetry-collected query metrics (always live, no DB queries)."
  def query_metrics do
    Atlas.Stats.Collector.get_metrics()
  end

  # ============================================================================
  # Cache (uses :persistent_term for lock-free reads across processes)
  # ============================================================================

  defp read_cache do
    case :persistent_term.get({__MODULE__, :overview}, nil) do
      {timestamp, data} ->
        age = System.monotonic_time(:millisecond) - timestamp

        if age < @cache_ttl_ms do
          {:ok, data}
        else
          :miss
        end

      nil ->
        :miss
    end
  rescue
    _ -> :miss
  end

  defp write_cache(data) do
    :persistent_term.put({__MODULE__, :overview}, {System.monotonic_time(:millisecond), data})
  rescue
    _ -> :ok
  end

  # ============================================================================
  # Fetch (actually runs the DB queries)
  # ============================================================================

  defp fetch_overview do
    is_connected = connected?()

    %{
      connected: is_connected,
      repo: repo_stats(is_connected),
      graphs: graph_stats(is_connected),
      connection_pool: pool_stats(),
      migrations: migration_status(is_connected),
      query_metrics: query_metrics()
    }
  rescue
    e ->
      Logger.debug("Atlas.Stats.get_overview failed: #{Exception.message(e)}")

      %{
        connected: false,
        repo: empty_repo_stats(),
        graphs: empty_graph_stats(),
        connection_pool: %{pool_size: 0, checked_out: 0, idle: 0},
        migrations: [],
        query_metrics: %{}
      }
  end

  defp repo_stats(true) do
    Map.new(@tables, fn table ->
      count = count_table(table)
      key = table |> String.replace("atlas_", "") |> String.to_atom()
      {key, count}
    end)
  end

  defp repo_stats(false), do: empty_repo_stats()

  defp graph_stats(true) do
    Map.new(Atlas.Graph.graphs(), fn graph ->
      nodes =
        case Atlas.Graph.count_nodes(graph) do
          {:ok, n} -> n
          _ -> 0
        end

      edges =
        case Atlas.Graph.count_edges(graph) do
          {:ok, n} -> n
          _ -> 0
        end

      {graph, %{nodes: nodes, edges: edges}}
    end)
  rescue
    _ -> empty_graph_stats()
  end

  defp graph_stats(false), do: empty_graph_stats()

  defp pool_stats do
    if Process.whereis(Atlas.Repo) do
      try do
        %{pool_size: pool_size()} |> Map.merge(checkout_stats())
      rescue
        _ -> %{pool_size: 0, checked_out: 0, idle: 0}
      end
    else
      %{pool_size: 0, checked_out: 0, idle: 0}
    end
  end

  defp migration_status(true) do
    try do
      migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
      Ecto.Migrator.migrations(Atlas.Repo, migrations_path)
    rescue
      _ -> []
    end
  end

  defp migration_status(false), do: []

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp count_table(table) do
    case Ecto.Adapters.SQL.query(Atlas.Repo, "SELECT COUNT(*) FROM #{table}", []) do
      {:ok, %{rows: [[count]]}} -> count
      _ -> 0
    end
  rescue
    _ -> 0
  end

  defp pool_size do
    config = Atlas.Repo.config()
    Keyword.get(config, :pool_size, 10)
  end

  defp checkout_stats do
    %{checked_out: 0, idle: pool_size()}
  end

  defp empty_repo_stats do
    Map.new(@tables, fn table ->
      key = table |> String.replace("atlas_", "") |> String.to_atom()
      {key, 0}
    end)
  end

  defp empty_graph_stats do
    Map.new(Atlas.Graph.graphs(), fn graph ->
      {graph, %{nodes: 0, edges: 0}}
    end)
  end
end
