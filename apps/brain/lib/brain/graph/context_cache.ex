defmodule Brain.Graph.ContextCache do
  @moduledoc """
  Turn-scoped ETS cache for `Reader.entity_context` and `relationship_path` results.

  Deduplicates the identical Cypher lookups that occur when the disambiguation
  pipeline and EntityGraphEnricher both query the same entity within a single
  pipeline turn. Uses Process dictionary keys to track which ETS entries belong
  to the current process so they can be purged at turn end.

  Keys include `{world_id, graph, label, name}` for world scoping (Rule 7).
  """

  @ets_table :graph_context_cache
  @process_key :graph_context_cache_keys

  @doc "Ensures the ETS table exists. Called once at application start."
  def init do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])
    end

    :ok
  end

  @doc """
  Look up a cached entity context result.

  Returns `{:ok, {node, neighbors}}` on hit, `:miss` on miss.
  """
  def get(world_id, graph, label, name) do
    key = {world_id, graph, label, name}

    case :ets.lookup(@ets_table, key) do
      [{^key, value}] -> {:ok, value}
      _ -> :miss
    end
  rescue
    ArgumentError -> :miss
  end

  @doc """
  Store a result in the cache, associated with the calling process.
  """
  def put(world_id, graph, label, name, value) do
    key = {world_id, graph, label, name}
    :ets.insert(@ets_table, {key, value})
    track_key(key)
    :ok
  rescue
    ArgumentError -> :ok
  end

  @doc """
  Purge all cache entries created by the calling process.

  Call this at the end of a pipeline turn to prevent stale data
  from leaking across turns.
  """
  def purge_process_entries do
    keys = Process.delete(@process_key) || []

    Enum.each(keys, fn key ->
      :ets.delete(@ets_table, key)
    end)

    :ok
  rescue
    ArgumentError -> :ok
  end

  defp track_key(key) do
    existing = Process.get(@process_key, [])
    Process.put(@process_key, [key | existing])
  end
end
