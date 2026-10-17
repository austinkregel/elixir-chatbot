defmodule Brain.ML.KnowledgeGraph.EntityVectorCache do
  @moduledoc """
  Caches 128-dimensional entity and entity-type embeddings from
  `KnowledgeGraph.Embedder`, per world.

  A vector is computed from a world's own triple scorer
  (`Brain.ML.KnowledgeGraph.TripleScorer.loaded!/1`), so every entry belongs to
  the world whose scorer computed it: entities are keyed
  `{world_id, {:entity, name}}` and types `{world_id, {:type, name}}`. A world
  with no copy of the scorer raises when a vector is computed for it.

  Vectors are computed lazily, in the caller, on first access, and kept in a
  public ETS table this process owns. Seeding or dropping a world's scorer
  forgets that world's vectors (`forget_world/1`).

  LRU eviction at `@max_entries` (default 50_000) to bound memory.
  """

  use GenServer
  require Logger

  alias Brain.ML.KnowledgeGraph.{Embedder, TripleScorer}

  @ets_table :entity_vector_cache
  @max_entries 50_000

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Get or compute the world's embedding vector for an entity.

  Returns `{:ok, tensor}`. Raises when the world has no copy of the triple
  scorer, or when the cache is not started.
  """
  @spec get_or_compute(String.t(), String.t()) :: {:ok, Nx.Tensor.t()}
  def get_or_compute(world_id, entity_name) when is_binary(world_id) do
    get_or_compute_key({world_id, {:entity, entity_name}}, fn ->
      {model, params, vocab} = extraction_model!(world_id)
      Embedder.encode_entity(entity_name, model, params, vocab)
    end)
  end

  @doc """
  Get or compute the world's concept vector for an entity type.

  Returns `{:ok, tensor}`. Raises when the world has no copy of the triple
  scorer, or when the cache is not started.
  """
  @spec get_or_compute_type(String.t(), String.t()) :: {:ok, Nx.Tensor.t()}
  def get_or_compute_type(world_id, type_name) when is_binary(world_id) do
    get_or_compute_key({world_id, {:type, type_name}}, fn ->
      {model, params, vocab} = extraction_model!(world_id)
      parents = Brain.Analysis.TypeHierarchy.parents_of(type_name)
      Embedder.encode_entity_type(type_name, parents, model, params, vocab)
    end)
  end

  @doc """
  Fast ETS-only lookup for the world's type vectors. Returns `{:ok, tensor}` or
  `:miss`.

  Does NOT compute on miss -- returns `:miss` immediately. Use this on hot
  paths (intent scoring loops) where computing per iteration would kill
  throughput.
  """
  @spec get_cached_type(String.t(), String.t()) :: {:ok, Nx.Tensor.t()} | :miss
  def get_cached_type(world_id, type_name) when is_binary(world_id) do
    key = {world_id, {:type, type_name}}

    case :ets.lookup(table!(), key) do
      [{^key, tensor, _access_time}] -> {:ok, tensor}
      [] -> :miss
    end
  end

  @doc """
  Pre-compute the world's type vectors for all entity types from the hierarchy.

  Raises when the world has no copy of the triple scorer.
  """
  def warm_up_entity_types(world_id) when is_binary(world_id) do
    types = Brain.Analysis.TypeHierarchy.all_types()

    Enum.each(types, fn type_name ->
      {:ok, _vector} = get_or_compute_type(world_id, type_name)
    end)

    Logger.info("EntityVectorCache: warmed up #{length(types)} type vectors for world #{world_id}")
  end

  @doc """
  Forgets every vector computed for the world, for a world whose triple scorer
  is seeded or dropped. When the cache is not started (a Mix task seeding
  worlds), no vector has been computed in this VM, so there is nothing to forget.
  """
  @spec forget_world(String.t()) :: :ok
  def forget_world(world_id) when is_binary(world_id) do
    if :ets.whereis(@ets_table) != :undefined do
      :ets.match_delete(@ets_table, {{world_id, :_}, :_, :_})

      :telemetry.execute([:brain, :kg_signal, :cache_invalidated], %{reason: :world_scorer_replaced}, %{
        world_id: world_id
      })
    end

    :ok
  end

  @doc """
  Returns cache statistics.
  """
  def stats(name \\ __MODULE__) do
    try do
      GenServer.call(name, :stats, 1_000)
    catch
      :exit, _ -> %{size: 0, ready: false}
    end
  end

  @doc """
  Manually clear the cache.
  """
  def clear(name \\ __MODULE__) do
    GenServer.cast(name, :clear)
  end

  # --- GenServer callbacks ---

  @impl true
  def init(_opts) do
    table =
      if :ets.whereis(@ets_table) != :undefined do
        @ets_table
      else
        :ets.new(@ets_table, [:named_table, :set, :public, read_concurrency: true])
      end

    {:ok, %{table: table, hits: 0, misses: 0}}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    size =
      try do
        :ets.info(state.table, :size)
      catch
        _, _ -> 0
      end

    stats = %{
      size: size,
      max_entries: @max_entries,
      hits: state.hits,
      misses: state.misses,
      ready: true
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast(:clear, state) do
    :ets.delete_all_objects(state.table)
    {:noreply, %{state | hits: 0, misses: 0}}
  end

  def handle_cast(:hit, state), do: {:noreply, %{state | hits: state.hits + 1}}
  def handle_cast(:miss, state), do: {:noreply, %{state | misses: state.misses + 1}}

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # --- Private ---

  defp get_or_compute_key(key, compute) do
    table = table!()

    case :ets.lookup(table, key) do
      [{^key, tensor, _access_time}] ->
        :telemetry.execute([:brain, :kg_signal, :cache_hit], %{}, %{})
        GenServer.cast(__MODULE__, :hit)
        :ets.update_element(table, key, {3, System.monotonic_time()})
        {:ok, tensor}

      [] ->
        :telemetry.execute([:brain, :kg_signal, :cache_miss], %{}, %{})
        GenServer.cast(__MODULE__, :miss)
        tensor = compute.()
        maybe_evict(table)
        :ets.insert(table, {key, tensor, System.monotonic_time()})
        {:ok, tensor}
    end
  end

  # The world's scorer, cut at its `dense1` layer, which encodes an entity.
  defp extraction_model!(world_id) do
    scorer = TripleScorer.loaded!(world_id)

    model =
      Embedder.build_extraction_model(
        scorer.config.vocab_size,
        embedding_dim: scorer.config.embedding_dim,
        hidden_dim: scorer.config.hidden_dim,
        max_seq_length: Map.get(scorer.config, :max_seq_length, 32)
      )

    {model, scorer.params, scorer.vocab}
  end

  defp table! do
    if :ets.whereis(@ets_table) == :undefined do
      raise "EntityVectorCache: the #{inspect(@ets_table)} table does not exist; " <>
              "Brain.ML.KnowledgeGraph.EntityVectorCache creates it when it starts"
    end

    @ets_table
  end

  defp maybe_evict(table) do
    size = :ets.info(table, :size)

    if size >= @max_entries do
      evict_oldest(table, div(@max_entries, 10))
    end
  rescue
    _ -> :ok
  end

  defp evict_oldest(table, count) do
    entries =
      :ets.tab2list(table)
      |> Enum.sort_by(fn {_key, _tensor, access_time} -> access_time end)
      |> Enum.take(count)

    Enum.each(entries, fn {key, _, _} ->
      :ets.delete(table, key)
    end)
  rescue
    _ -> :ok
  end
end
