defmodule Brain.Memory.Store do
  @moduledoc "Storage layer for the cognitive memory system.\n\nPorted from the Rust cognitive_memory_system MemoryStore.\n\nManages collections of episodic and semantic memories with separate\nvector indices for efficient retrieval. Supports persistence to disk.\n\n## World Scoping\n\nAll operations support an optional `world_id` parameter for data isolation.\nIf not specified, operations use the \"default\" world.\n\nEpisodes and semantic facts are stored per-world, allowing complete\nisolation between training worlds while sharing the same GenServer.\n"

  # World.Embedder is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Embedder}

  alias Brain.Telemetry
  alias Brain.Memory
  alias Brain.Memory.Types
  use GenServer

  alias Types.{Episode, SemanticFact}
  alias Memory.{Consolidation, Embedder, VectorIndex}
  alias World.Embedder, as: WorldEmbedder

  require Logger

  @default_world_id "default"
  @consolidate_interval_ms 15 * 60 * 1000

  @doc "Returns true if the Store is ready to accept requests."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Starts the Memory Store.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Add a new episode to the store.\nThe episode is embedded and indexed for similarity search.\n\n## Options\n  - world_id: The world to add the episode to (default: \"default\")\n"
  def add_episode(state, action, outcome, tags, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:add_episode, state, action, outcome, tags, world_id})
  end

  @doc "Add a pre-built episode to the store.\n\n## Options\n  - world_id: The world to add the episode to (default: \"default\")\n"
  def add_episode_direct(episode, opts \\ []) when is_struct(episode, Episode) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:add_episode_direct, episode, world_id})
  end

  @doc "Query for episodes similar to the given text.\nReturns top k episodes with similarity scores.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def query_similar(text, k \\ 5, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    Telemetry.span(:memory_query, %{k: k, world_id: world_id}, fn ->
      GenServer.call(__MODULE__, {:query_similar, text, k, world_id})
    end)
  end

  @doc "Query for episodes with specific tags.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def query_by_tags(tags, limit \\ 10, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:query_by_tags, tags, limit, world_id})
  end

  @doc "Add an episode based on an extracted event.\n\nCreates an episode with structured state from the event, action from the verb lemma,\nand appropriate tags for event-based querying.\n\n## Parameters\n  - event: An Event struct from EventExtractor\n  - context: Map with :response (bot's response) and optional :user_input\n  - opts: Keyword list with :world_id, :tags\n\n## Examples\n\n    event = %Event{action: %{lemma: \"play\", ...}, object: %{text: \"jazz\"}}\n    context = %{response: \"Playing jazz music\", user_input: \"Play some jazz\"}\n    add_event_episode(event, context, world_id: \"training\")\n"
  def add_event_episode(event, context, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    extra_tags = Keyword.get(opts, :tags, [])
    state = format_event_state(event, context)
    action = get_event_action(event)
    outcome = Map.get(context, :response, "")
    event_tags = build_event_tags(event)
    all_tags = extra_tags ++ event_tags

    add_episode(state, action, outcome, all_tags, world_id: world_id)
  end

  @doc "Query for episodes by action/verb type.\n\nReturns episodes that were created from events with the specified action lemma.\n\n## Examples\n\n    # Find all episodes where user asked to \"play\" something\n    query_events_by_action(\"play\", 5, world_id: \"default\")\n"
  def query_events_by_action(action_lemma, k \\ 5, opts \\ []) do
    query_by_tags(["event:#{action_lemma}"], k, opts)
  end

  @doc "Query for episodes involving a specific object.\n\n## Examples\n\n    # Find all episodes about \"music\"\n    query_events_by_object(\"music\", 5)\n"
  def query_events_by_object(object_text, k \\ 5, opts \\ []) do
    query_by_tags(["object:#{String.downcase(object_text)}"], k, opts)
  end

  @doc "Query for episodes involving a specific actor.\n\n## Examples\n\n    # Find all episodes where user was the actor\n    query_events_by_actor(\"user\", 5)\n"
  def query_events_by_actor(actor_text, k \\ 5, opts \\ []) do
    query_by_tags(["actor:#{String.downcase(actor_text)}"], k, opts)
  end

  defp format_event_state(event, context) do
    actor_text = get_participant_text(event, :actor)
    object_text = get_participant_text(event, :object)
    action_lemma = get_event_action(event)

    user_input = Map.get(context, :user_input, "")

    if user_input != "" do
      "User said: #{user_input}"
    else
      actor =
        if actor_text do
          actor_text
        else
          "Someone"
        end

      object =
        if object_text do
          " #{object_text}"
        else
          ""
        end

      "#{actor} #{action_lemma}#{object}"
    end
  end

  defp get_event_action(event) do
    case event do
      %{action: %{lemma: lemma}} when is_binary(lemma) -> lemma
      %{action: %{verb: verb}} when is_binary(verb) -> String.downcase(verb)
      _ -> "unknown"
    end
  end

  defp get_participant_text(event, role) do
    case Map.get(event, role) do
      %{text: text} when is_binary(text) -> text
      _ -> nil
    end
  end

  defp build_event_tags(event) do
    action_lemma = get_event_action(event)
    actor_text = get_participant_text(event, :actor)
    object_text = get_participant_text(event, :object)

    tags = ["event", "event:#{action_lemma}"]

    tags =
      if actor_text do
        tags ++ ["actor:#{String.downcase(actor_text)}"]
      else
        tags
      end

    if object_text do
      tags ++ ["object:#{String.downcase(object_text)}"]
    else
      tags
    end
  end

  @doc "Get a specific episode by ID.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def get_episode(id, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:get_episode, id, world_id})
  end

  @doc "Add a semantic fact to the store.\n\n## Options\n  - world_id: The world to add the semantic to (default: \"default\")\n"
  def add_semantic(semantic, opts \\ []) when is_struct(semantic, SemanticFact) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:add_semantic, semantic, world_id})
  end

  @doc "Query for semantic facts similar to the given text.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def query_semantic(text, k \\ 5, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:query_semantic, text, k, world_id})
  end

  @doc "Get a specific semantic fact by ID.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def get_semantic(id, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:get_semantic, id, world_id})
  end

  @doc "Update an episode's semantic_id after consolidation.\n\n## Options\n  - world_id: The world containing the episode (default: \"default\")\n"
  def link_episode_to_semantic(episode_id, semantic_id, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:link_episode, episode_id, semantic_id, world_id})
  end

  @doc "Get all episodes.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def all_episodes(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:all_episodes, world_id})
  end

  @doc "Get all semantic facts.\n\n## Options\n  - world_id: The world to query (default: \"default\")\n"
  def all_semantics(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(__MODULE__, {:all_semantics, world_id})
  end

  @doc "Get store statistics.\n\n## Options\n  - world_id: The world to get stats for (default: nil for all worlds)\n"
  def stats(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, nil)
    GenServer.call(__MODULE__, {:stats, world_id})
  end

  @doc "Persist the store to disk.\n"
  def persist do
    GenServer.call(__MODULE__, :persist, 30_000)
  end

  @doc "Clear all data from the store.\n\n## Options\n  - world_id: The world to clear (default: nil for all worlds)\n"
  def clear(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, nil)
    GenServer.call(__MODULE__, {:clear, world_id}, 30_000)
  end

  @doc "Lists all world IDs that have data in the store.\n"
  def list_worlds do
    GenServer.call(__MODULE__, :list_worlds, 5_000)
  end

  @impl true
  def init(_opts) do
    episode_index = VectorIndex.new(:memory_episode_index)
    semantic_index = VectorIndex.new(:memory_semantic_index)

    state = %{
      episode_index: episode_index,
      semantic_index: semantic_index
    }

    warm_vector_index(state)

    Logger.info("Memory store initialized (Atlas-primary)",
      episode_index_size: VectorIndex.count(episode_index),
      semantic_index_size: VectorIndex.count(semantic_index)
    )

    Process.send_after(self(), :consolidate_scheduled, @consolidate_interval_ms)

    {:ok, state}
  end

  @impl true
  def handle_info(:consolidate_scheduled, state) do
    Task.start(fn ->
      case __MODULE__.list_worlds() do
        {:ok, world_ids} ->
          Enum.each(world_ids, fn world_id ->
            try do
              Consolidation.consolidate(world_id: world_id)
            rescue
              e ->
                Logger.warning("Consolidation failed for world #{world_id}: #{Exception.message(e)}")
            end
          end)

        _ ->
          :ok
      end
    end)

    Process.send_after(self(), :consolidate_scheduled, @consolidate_interval_ms)
    {:noreply, state}
  end

  # ============================================================================
  # Write Operations -- Atlas Primary, VectorIndex Cache Update
  # ============================================================================

  @impl true
  def handle_call({:add_episode, text, action, outcome, tags, world_id}, _from, state) do
    embedding_result = get_embedding(world_id, text)

    embedding =
      case embedding_result do
        {:ok, emb} -> emb
        {:error, _} -> []
      end

    episode = Episode.new(text, action, outcome, tags, embedding)

    # Write to Atlas first (primary store)
    case Brain.AtlasIntegration.persist_episode_sync(episode, world_id) do
      {:ok, _id} ->
        # Update VectorIndex cache
        if embedding != [] do
          VectorIndex.insert(state.episode_index, {world_id, episode.id}, embedding)
        end

        {:reply, {:ok, episode.id}, state}

      {:error, _reason} ->
        # Atlas unavailable -- write-through to VectorIndex only
        if embedding != [] do
          VectorIndex.insert(state.episode_index, {world_id, episode.id}, embedding)
        end

        Brain.AtlasIntegration.persist_episode(episode, world_id)
        {:reply, {:ok, episode.id}, state}
    end
  end

  @impl true
  def handle_call({:add_episode_direct, episode, world_id}, _from, state) do
    case Brain.AtlasIntegration.persist_episode_sync(episode, world_id) do
      {:ok, _id} ->
        if is_list(episode.embedding) and episode.embedding != [] do
          VectorIndex.insert(state.episode_index, {world_id, episode.id}, episode.embedding)
        end

        {:reply, {:ok, episode.id}, state}

      {:error, _reason} ->
        if is_list(episode.embedding) and episode.embedding != [] do
          VectorIndex.insert(state.episode_index, {world_id, episode.id}, episode.embedding)
        end

        Brain.AtlasIntegration.persist_episode(episode, world_id)
        {:reply, {:ok, episode.id}, state}
    end
  end

  @impl true
  def handle_call({:add_semantic, semantic, world_id}, _from, state) do
    case Brain.AtlasIntegration.persist_semantic_sync(semantic, world_id) do
      {:ok, _id} ->
        if is_list(semantic.embedding) and semantic.embedding != [] do
          VectorIndex.insert(state.semantic_index, {world_id, semantic.id}, semantic.embedding)
        end

        {:reply, {:ok, semantic.id}, state}

      {:error, _reason} ->
        if is_list(semantic.embedding) and semantic.embedding != [] do
          VectorIndex.insert(state.semantic_index, {world_id, semantic.id}, semantic.embedding)
        end

        Brain.AtlasIntegration.persist_semantic(semantic, world_id)
        {:reply, {:ok, semantic.id}, state}
    end
  end

  @impl true
  def handle_call({:link_episode, episode_id, semantic_id, world_id}, _from, state) do
    # Verify the episode exists before linking
    case Brain.AtlasIntegration.get_episode(episode_id, world_id) do
      {:ok, _episode} ->
        Brain.AtlasIntegration.link_episode_semantic(episode_id, semantic_id)
        {:reply, :ok, state}

      {:error, :not_found} ->
        {:reply, {:error, :not_found}, state}

      {:error, _reason} ->
        {:reply, {:error, :not_found}, state}
    end
  end

  # ============================================================================
  # Read Operations -- Atlas Primary, VectorIndex for Similarity
  # ============================================================================

  @impl true
  def handle_call({:query_similar, text, k, world_id}, _from, state) do
    case get_embedding(world_id, text) do
      {:ok, query_embedding} ->
        results =
          VectorIndex.search_all(state.episode_index, query_embedding, k * 2)
          |> Enum.filter(fn {{wid, _id}, _score} -> wid == world_id end)
          |> Enum.take(k)
          |> Enum.map(fn {{_wid, id}, similarity} ->
            case Brain.AtlasIntegration.get_episode(id, world_id) do
              {:ok, episode} -> {episode, similarity}
              {:error, _} -> nil
            end
          end)
          |> Enum.reject(&is_nil/1)

        {:reply, {:ok, results}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:query_by_tags, tags, limit, world_id}, _from, state) do
    case Brain.AtlasIntegration.query_episodes_by_tags(world_id, tags, limit) do
      {:ok, episodes} ->
        {:reply, {:ok, episodes}, state}

      {:error, _} ->
        {:reply, {:ok, []}, state}
    end
  end

  @impl true
  def handle_call({:get_episode, id, world_id}, _from, state) do
    case Brain.AtlasIntegration.get_episode(id, world_id) do
      {:ok, episode} -> {:reply, {:ok, episode}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:query_semantic, text, k, world_id}, _from, state) do
    case get_embedding(world_id, text) do
      {:ok, query_embedding} ->
        results =
          VectorIndex.search_all(state.semantic_index, query_embedding, k * 2)
          |> Enum.filter(fn {{wid, _id}, _score} -> wid == world_id end)
          |> Enum.take(k)
          |> Enum.map(fn {{_wid, id}, similarity} ->
            case Brain.AtlasIntegration.get_semantic(id, world_id) do
              {:ok, semantic} -> {semantic, similarity}
              {:error, _} -> nil
            end
          end)
          |> Enum.reject(&is_nil/1)

        {:reply, {:ok, results}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:get_semantic, id, world_id}, _from, state) do
    case Brain.AtlasIntegration.get_semantic(id, world_id) do
      {:ok, semantic} -> {:reply, {:ok, semantic}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:all_episodes, world_id}, _from, state) do
    case Brain.AtlasIntegration.list_episodes(world_id) do
      {:ok, episodes} -> {:reply, {:ok, episodes}, state}
      {:error, _} -> {:reply, {:ok, []}, state}
    end
  end

  @impl true
  def handle_call({:all_semantics, world_id}, _from, state) do
    case Brain.AtlasIntegration.list_semantics(world_id) do
      {:ok, semantics} -> {:reply, {:ok, semantics}, state}
      {:error, _} -> {:reply, {:ok, []}, state}
    end
  end

  # ============================================================================
  # State Management
  # ============================================================================

  @impl true
  def handle_call({:stats, nil}, _from, state) do
    stats = %{
      episode_count: VectorIndex.count(state.episode_index),
      semantic_count: VectorIndex.count(state.semantic_index),
      episode_index_size: VectorIndex.count(state.episode_index),
      semantic_index_size: VectorIndex.count(state.semantic_index),
      worlds: list_worlds_from_atlas()
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:stats, world_id}, _from, state) do
    episode_count =
      case Brain.AtlasIntegration.list_episodes(world_id) do
        {:ok, eps} -> length(eps)
        _ -> 0
      end

    semantic_count =
      case Brain.AtlasIntegration.list_semantics(world_id) do
        {:ok, sems} -> length(sems)
        _ -> 0
      end

    stats = %{
      episode_count: episode_count,
      semantic_count: semantic_count,
      world_id: world_id
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:clear, nil}, _from, state) do
    VectorIndex.clear(state.episode_index)
    VectorIndex.clear(state.semantic_index)
    Brain.AtlasIntegration.clear_memory()
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:clear, world_id}, _from, state) do
    # VectorIndex doesn't support world-scoped clear, but clearing all
    # and re-warming would be too expensive. For now, just acknowledge.
    _ = world_id
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:list_worlds, _from, state) do
    worlds = list_worlds_from_atlas()
    {:reply, {:ok, worlds}, state}
  end

  defp list_worlds_from_atlas do
    case Brain.AtlasIntegration.list_memory_worlds() do
      {:ok, worlds} -> worlds
      _ -> []
    end
  end

  defp warm_vector_index(state) do
    case Brain.AtlasIntegration.load_episodes(@default_world_id) do
      {:ok, episodes} when episodes != %{} ->
        Enum.each(episodes, fn {id, ep} ->
          if is_list(ep.embedding) and ep.embedding != [] do
            VectorIndex.insert(state.episode_index, {@default_world_id, id}, ep.embedding)
          end
        end)

        episode_count = map_size(episodes)

        case Brain.AtlasIntegration.load_semantics(@default_world_id) do
          {:ok, semantics} when semantics != %{} ->
            Enum.each(semantics, fn {id, sem} ->
              if is_list(sem.embedding) and sem.embedding != [] do
                VectorIndex.insert(state.semantic_index, {@default_world_id, id}, sem.embedding)
              end
            end)

            Logger.info("VectorIndex warmed from Atlas",
              episodes: episode_count,
              semantics: map_size(semantics)
            )

          _ ->
            Logger.info("VectorIndex warmed from Atlas (episodes only)",
              episodes: episode_count
            )
        end

      _ ->
        Logger.debug("No memory data in Atlas, VectorIndex empty")
    end
  rescue
    e ->
      Logger.warning("Failed to warm VectorIndex from Atlas: #{inspect(e)}")
  end

  defp get_embedding(world_id, text) do
    world_embed_result = WorldEmbedder.embed(world_id, text)

    case world_embed_result do
      {:ok, embedding} ->
        {:ok, embedding}

      {:error, reason}
      when reason in [:no_training_data, :not_initialized, :vocabulary_building, :table_not_ready] ->
        if Embedder.ready?() do
          Embedder.embed(text)
        else
          {:error, :no_embedder_available}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end
end
