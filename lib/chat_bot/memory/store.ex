defmodule ChatBot.Memory.Store do
  @moduledoc """
  Storage layer for the cognitive memory system.

  Ported from the Rust cognitive_memory_system MemoryStore.

  Manages collections of episodic and semantic memories with separate
  vector indices for efficient retrieval. Supports persistence to disk.
  """

  use GenServer

  alias ChatBot.Memory.Types.{Episode, SemanticFact}
  alias ChatBot.Memory.{Embedder, VectorIndex}

  require Logger

  @default_persistence_path "priv/data/memory_store.term"

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Add a new episode to the store.
  The episode is embedded and indexed for similarity search.
  """
  def add_episode(state, action, outcome, tags) do
    GenServer.call(__MODULE__, {:add_episode, state, action, outcome, tags})
  end

  @doc """
  Add a pre-built episode to the store.
  """
  def add_episode_direct(episode) when is_struct(episode, Episode) do
    GenServer.call(__MODULE__, {:add_episode_direct, episode})
  end

  @doc """
  Query for episodes similar to the given text.
  Returns top k episodes with similarity scores.
  """
  def query_similar(text, k \\ 5) do
    # Wrap with telemetry span for async, non-blocking metrics
    ChatBot.Telemetry.span(:memory_query, %{k: k}, fn ->
      GenServer.call(__MODULE__, {:query_similar, text, k})
    end)
  end

  @doc """
  Query for episodes with specific tags.
  """
  def query_by_tags(tags, limit \\ 10) do
    GenServer.call(__MODULE__, {:query_by_tags, tags, limit})
  end

  @doc """
  Get a specific episode by ID.
  """
  def get_episode(id) do
    GenServer.call(__MODULE__, {:get_episode, id})
  end

  @doc """
  Add a semantic fact to the store.
  """
  def add_semantic(semantic) when is_struct(semantic, SemanticFact) do
    GenServer.call(__MODULE__, {:add_semantic, semantic})
  end

  @doc """
  Query for semantic facts similar to the given text.
  """
  def query_semantic(text, k \\ 5) do
    GenServer.call(__MODULE__, {:query_semantic, text, k})
  end

  @doc """
  Get a specific semantic fact by ID.
  """
  def get_semantic(id) do
    GenServer.call(__MODULE__, {:get_semantic, id})
  end

  @doc """
  Update an episode's semantic_id after consolidation.
  """
  def link_episode_to_semantic(episode_id, semantic_id) do
    GenServer.call(__MODULE__, {:link_episode, episode_id, semantic_id})
  end

  @doc """
  Get all episodes (for consolidation).
  """
  def all_episodes do
    GenServer.call(__MODULE__, :all_episodes)
  end

  @doc """
  Get all semantic facts.
  """
  def all_semantics do
    GenServer.call(__MODULE__, :all_semantics)
  end

  @doc """
  Get store statistics.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Persist the store to disk.
  """
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc """
  Clear all data from the store.
  """
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    persistence_path = Keyword.get(opts, :persistence_path, @default_persistence_path)

    # Create ETS tables for vector indices
    episode_index = VectorIndex.new(:memory_episode_index)
    semantic_index = VectorIndex.new(:memory_semantic_index)

    state = %{
      episodes: %{},
      semantics: %{},
      episode_index: episode_index,
      semantic_index: semantic_index,
      persistence_path: persistence_path
    }

    # Try to load from disk
    state = maybe_load_from_disk(state)

    Logger.info("Memory store initialized",
      episodes: map_size(state.episodes),
      semantics: map_size(state.semantics)
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:add_episode, text, action, outcome, tags}, _from, state) do
    # Check if embedder is ready before attempting to embed
    if Embedder.ready?() do
      case Embedder.embed(text) do
        {:ok, embedding} ->
          episode = Episode.new(text, action, outcome, tags, embedding)
          VectorIndex.insert(state.episode_index, episode.id, embedding)
          new_episodes = Map.put(state.episodes, episode.id, episode)
          new_state = %{state | episodes: new_episodes}

          {:reply, {:ok, episode.id}, new_state}

        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    else
      # Embedder not ready - store episode without embedding for now
      # This prevents blocking during vocabulary building
      episode = Episode.new(text, action, outcome, tags, [])
      new_episodes = Map.put(state.episodes, episode.id, episode)
      new_state = %{state | episodes: new_episodes}

      {:reply, {:ok, episode.id}, new_state}
    end
  end

  @impl true
  def handle_call({:add_episode_direct, episode}, _from, state) do
    VectorIndex.insert(state.episode_index, episode.id, episode.embedding)
    new_episodes = Map.put(state.episodes, episode.id, episode)
    new_state = %{state | episodes: new_episodes}
    {:reply, {:ok, episode.id}, new_state}
  end

  @impl true
  def handle_call({:query_similar, text, k}, _from, state) do
    # Check if embedder is ready before attempting to embed
    if Embedder.ready?() do
      case Embedder.embed(text) do
        {:ok, query_embedding} ->
          results =
            VectorIndex.search(state.episode_index, query_embedding, k)
            |> Enum.map(fn {id, similarity} ->
              episode = Map.get(state.episodes, id)
              {episode, similarity}
            end)
            |> Enum.filter(fn {ep, _} -> ep != nil end)

          {:reply, {:ok, results}, state}

        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    else
      {:reply, {:error, :embedder_not_ready}, state}
    end
  end

  @impl true
  def handle_call({:query_by_tags, tags, limit}, _from, state) do
    tag_set = MapSet.new(tags)

    results =
      state.episodes
      |> Map.values()
      |> Enum.filter(fn ep ->
        ep_tags = MapSet.new(ep.tags)
        not MapSet.disjoint?(tag_set, ep_tags)
      end)
      |> Enum.sort_by(fn ep -> -ep.timestamp end)
      |> Enum.take(limit)

    {:reply, {:ok, results}, state}
  end

  @impl true
  def handle_call({:get_episode, id}, _from, state) do
    case Map.get(state.episodes, id) do
      nil -> {:reply, {:error, :not_found}, state}
      episode -> {:reply, {:ok, episode}, state}
    end
  end

  @impl true
  def handle_call({:add_semantic, semantic}, _from, state) do
    VectorIndex.insert(state.semantic_index, semantic.id, semantic.embedding)
    new_semantics = Map.put(state.semantics, semantic.id, semantic)
    new_state = %{state | semantics: new_semantics}
    {:reply, {:ok, semantic.id}, new_state}
  end

  @impl true
  def handle_call({:query_semantic, text, k}, _from, state) do
    # Check if embedder is ready before attempting to embed
    if Embedder.ready?() do
      case Embedder.embed(text) do
        {:ok, query_embedding} ->
          results =
            VectorIndex.search(state.semantic_index, query_embedding, k)
            |> Enum.map(fn {id, similarity} ->
              semantic = Map.get(state.semantics, id)
              {semantic, similarity}
            end)
            |> Enum.filter(fn {s, _} -> s != nil end)

          {:reply, {:ok, results}, state}

        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    else
      {:reply, {:error, :embedder_not_ready}, state}
    end
  end

  @impl true
  def handle_call({:get_semantic, id}, _from, state) do
    case Map.get(state.semantics, id) do
      nil -> {:reply, {:error, :not_found}, state}
      semantic -> {:reply, {:ok, semantic}, state}
    end
  end

  @impl true
  def handle_call({:link_episode, episode_id, semantic_id}, _from, state) do
    case Map.get(state.episodes, episode_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      episode ->
        updated = %{episode | semantic_id: semantic_id}
        new_episodes = Map.put(state.episodes, episode_id, updated)
        {:reply, :ok, %{state | episodes: new_episodes}}
    end
  end

  @impl true
  def handle_call(:all_episodes, _from, state) do
    {:reply, {:ok, Map.values(state.episodes)}, state}
  end

  @impl true
  def handle_call(:all_semantics, _from, state) do
    {:reply, {:ok, Map.values(state.semantics)}, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      episode_count: map_size(state.episodes),
      semantic_count: map_size(state.semantics),
      episode_index_size: VectorIndex.count(state.episode_index),
      semantic_index_size: VectorIndex.count(state.semantic_index)
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    result = persist_to_disk(state)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    VectorIndex.clear(state.episode_index)
    VectorIndex.clear(state.semantic_index)

    new_state = %{state | episodes: %{}, semantics: %{}}
    {:reply, :ok, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp maybe_load_from_disk(state) do
    path = state.persistence_path

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          try do
            data = :erlang.binary_to_term(binary)
            episodes = Map.get(data, :episodes, %{})
            semantics = Map.get(data, :semantics, %{})

            # Rebuild indices
            Enum.each(episodes, fn {id, ep} ->
              if is_list(ep.embedding) and length(ep.embedding) > 0 do
                VectorIndex.insert(state.episode_index, id, ep.embedding)
              end
            end)

            Enum.each(semantics, fn {id, sem} ->
              if is_list(sem.embedding) and length(sem.embedding) > 0 do
                VectorIndex.insert(state.semantic_index, id, sem.embedding)
              end
            end)

            Logger.info("Loaded memory store from disk",
              episodes: map_size(episodes),
              semantics: map_size(semantics)
            )

            %{state | episodes: episodes, semantics: semantics}
          rescue
            e ->
              Logger.warning("Failed to load memory store: #{inspect(e)}")
              state
          end

        {:error, reason} ->
          Logger.warning("Could not read memory store file: #{inspect(reason)}")
          state
      end
    else
      state
    end
  end

  defp persist_to_disk(state) do
    path = state.persistence_path

    # Ensure directory exists
    path |> Path.dirname() |> File.mkdir_p!()

    data = %{
      episodes: state.episodes,
      semantics: state.semantics
    }

    binary = :erlang.term_to_binary(data)

    case File.write(path, binary) do
      :ok ->
        Logger.info("Memory store persisted to disk",
          episodes: map_size(state.episodes),
          semantics: map_size(state.semantics)
        )

        :ok

      {:error, reason} ->
        Logger.error("Failed to persist memory store: #{inspect(reason)}")
        {:error, reason}
    end
  end
end
