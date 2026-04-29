defmodule Brain.ML.KnowledgeGraph.EntityVectorCache do
  @moduledoc """
  Caches 128-dimensional entity embeddings from `KnowledgeGraph.Embedder`.

  Uses ETS for fast concurrent reads, keyed by `{world_id, entity_name}`.
  Lazily computes embeddings on first access via `Embedder.encode_entity/4`.

  Subscribes to `world_models:status` and invalidates on TripleScorer reload,
  since embeddings depend on the trained model weights.

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
  Get or compute the entity embedding vector.

  Returns `{:ok, tensor}` or `{:error, reason}`.
  """
  def get_or_compute(world_id, entity_name, name \\ __MODULE__) do
    key = {world_id, entity_name}

    case ets_lookup(key) do
      {:ok, tensor} ->
        update_access_time(key)
        {:ok, tensor}

      :miss ->
        try do
          GenServer.call(name, {:compute, world_id, entity_name}, 5_000)
        catch
          :exit, _ -> {:error, :not_ready}
        end
    end
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
  Get or compute the type concept vector (globally cached, not per-world).

  Type vectors use `{:global, {:type, type_name}}` keys to avoid
  redundant computation across worlds.
  """
  def get_or_compute_type(type_name, name \\ __MODULE__) do
    key = {:global, {:type, type_name}}

    case ets_lookup(key) do
      {:ok, tensor} ->
        update_access_time(key)
        {:ok, tensor}

      :miss ->
        try do
          GenServer.call(name, {:compute_type, type_name}, 5_000)
        catch
          :exit, _ -> {:error, :not_ready}
        end
    end
  end

  @doc """
  Pre-compute type vectors for all entity types from the hierarchy.

  Requires TripleScorer to be loaded. Raises if not available.
  """
  def warm_up_entity_types(name \\ __MODULE__) do
    unless TripleScorer.ready?() do
      raise "TripleScorer must be loaded for entity type vector warm-up"
    end

    types =
      if Code.ensure_loaded?(Brain.Analysis.TypeHierarchy) and
           function_exported?(Brain.Analysis.TypeHierarchy, :all_types, 0) do
        Brain.Analysis.TypeHierarchy.all_types()
      else
        []
      end

    Enum.each(types, fn type_name ->
      get_or_compute_type(type_name, name)
    end)

    Logger.info("EntityVectorCache: warmed up #{length(types)} type vectors")
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

    Phoenix.PubSub.subscribe(Brain.PubSub, "world_models:status")

    {:ok, %{table: table, hits: 0, misses: 0}}
  end

  @impl true
  def handle_call({:compute, world_id, entity_name}, _from, state) do
    key = {world_id, entity_name}

    case ets_lookup(key) do
      {:ok, tensor} ->
        {:reply, {:ok, tensor}, %{state | hits: state.hits + 1}}

      :miss ->
        case compute_embedding(world_id, entity_name) do
          {:ok, tensor} ->
            maybe_evict(state.table)
            :ets.insert(state.table, {key, tensor, System.monotonic_time()})
            {:reply, {:ok, tensor}, %{state | misses: state.misses + 1}}

          error ->
            {:reply, error, %{state | misses: state.misses + 1}}
        end
    end
  end

  def handle_call({:compute_type, type_name}, _from, state) do
    key = {:global, {:type, type_name}}

    case ets_lookup(key) do
      {:ok, tensor} ->
        {:reply, {:ok, tensor}, %{state | hits: state.hits + 1}}

      :miss ->
        case get_scorer_internals("default") do
          {:ok, model, params, vocab} ->
            parent =
              if Code.ensure_loaded?(Brain.Analysis.TypeHierarchy) and
                   function_exported?(Brain.Analysis.TypeHierarchy, :parent_of, 1) do
                Brain.Analysis.TypeHierarchy.parent_of(type_name)
              end

            vector = Embedder.encode_entity_type(type_name, parent, model, params, vocab)
            maybe_evict(state.table)
            :ets.insert(state.table, {key, vector, System.monotonic_time()})
            {:reply, {:ok, vector}, %{state | misses: state.misses + 1}}

          {:error, reason} ->
            {:reply, {:error, reason}, %{state | misses: state.misses + 1}}
        end
    end
  end

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
      ready: TripleScorer.ready?()
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast(:clear, state) do
    :ets.delete_all_objects(state.table)
    {:noreply, %{state | hits: 0, misses: 0}}
  end

  @impl true
  def handle_info({:triple_scorer_reloaded, _old_version, _new_version}, state) do
    Logger.info("EntityVectorCache: invalidating cache on TripleScorer reload")
    :ets.delete_all_objects(state.table)
    :telemetry.execute([:brain, :kg_signal, :cache_invalidated], %{reason: :model_reload}, %{})
    {:noreply, %{state | hits: 0, misses: 0}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # --- Private ---

  defp ets_lookup(key) do
    case :ets.lookup(@ets_table, key) do
      [{^key, tensor, _access_time}] ->
        :telemetry.execute([:brain, :kg_signal, :cache_hit], %{}, %{})
        {:ok, tensor}

      _ ->
        :telemetry.execute([:brain, :kg_signal, :cache_miss], %{}, %{})
        :miss
    end
  rescue
    ArgumentError -> :miss
  end

  defp update_access_time(key) do
    :ets.update_element(@ets_table, key, {3, System.monotonic_time()})
  rescue
    _ -> :ok
  end

  defp compute_embedding(world_id, entity_name) do
    unless TripleScorer.ready?() do
      {:error, :scorer_not_ready}
    else
      case TripleScorer.current_model_version() do
        {:ok, _version} ->
          case get_scorer_internals(world_id) do
            {:ok, model, params, vocab} ->
              embedding = Embedder.encode_entity(entity_name, model, params, vocab)
              {:ok, embedding}

            error ->
              error
          end

        _ ->
          {:error, :no_model_version}
      end
    end
  rescue
    e -> {:error, {:compute_failed, Exception.message(e)}}
  end

  defp get_scorer_internals(_world_id) do
    path = scorer_model_path("default")

    case TripleScorer.load_model(path) do
      {:ok, data} ->
        model =
          Embedder.build_extraction_model(
            data.config.vocab_size,
            embedding_dim: data.config.embedding_dim,
            hidden_dim: data.config.hidden_dim,
            max_seq_length: Map.get(data.config, :max_seq_length, 32)
          )

        {:ok, model, data.params, data.vocab}

      {:error, reason} ->
        {:error, {:model_load_failed, reason}}
    end
  end

  defp scorer_model_path(world_id) do
    priv = :code.priv_dir(:brain) |> to_string()
    Path.join([priv, "ml_models", world_id, "kg_lstm", "triple_scorer.term"])
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
