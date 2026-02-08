defmodule Brain.Memory.Store do
  @moduledoc "Storage layer for the cognitive memory system.\n\nPorted from the Rust cognitive_memory_system MemoryStore.\n\nManages collections of episodic and semantic memories with separate\nvector indices for efficient retrieval. Supports persistence to disk.\n\n## World Scoping\n\nAll operations support an optional `world_id` parameter for data isolation.\nIf not specified, operations use the \"default\" world.\n\nEpisodes and semantic facts are stored per-world, allowing complete\nisolation between training worlds while sharing the same GenServer.\n"

  alias Brain.Telemetry
  alias Brain.Memory
  alias Brain.Memory.Types
  use GenServer

  alias Types.{Episode, SemanticFact}
  alias Memory.{Embedder, VectorIndex}
  alias World.Embedder, as: WorldEmbedder

  require Logger

  defp default_persistence_path do
    Brain.priv_path("data/memory_store.term")
  end

  @default_world_id "default"

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
    GenServer.call(__MODULE__, :persist)
  end

  @doc "Clear all data from the store.\n\n## Options\n  - world_id: The world to clear (default: nil for all worlds)\n"
  def clear(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, nil)
    GenServer.call(__MODULE__, {:clear, world_id})
  end

  @doc "Lists all world IDs that have data in the store.\n"
  def list_worlds do
    GenServer.call(__MODULE__, :list_worlds)
  end

  @impl true
  def init(opts) do
    config_path = Application.get_env(:brain, :memory_store_path, default_persistence_path())
    persistence_path = Keyword.get(opts, :persistence_path, config_path)
    episode_index = VectorIndex.new(:memory_episode_index)
    semantic_index = VectorIndex.new(:memory_semantic_index)

    state = %{
      episodes: %{},
      semantics: %{},
      episode_index: episode_index,
      semantic_index: semantic_index,
      persistence_path: persistence_path
    }

    state = maybe_load_from_disk(state)

    total_episodes = count_all_episodes(state.episodes)
    total_semantics = count_all_semantics(state.semantics)

    Logger.info("Memory store initialized",
      episodes: total_episodes,
      semantics: total_semantics,
      worlds: map_size(state.episodes)
    )

    {:ok, state}
  end

  @impl true
  def handle_call({:add_episode, text, action, outcome, tags, world_id}, _from, state) do
    embedding_result = get_embedding(world_id, text)

    case embedding_result do
      {:ok, embedding} ->
        episode = Episode.new(text, action, outcome, tags, embedding)
        VectorIndex.insert(state.episode_index, {world_id, episode.id}, embedding)

        world_episodes = Map.get(state.episodes, world_id, %{})
        new_world_episodes = Map.put(world_episodes, episode.id, episode)
        new_episodes = Map.put(state.episodes, world_id, new_world_episodes)
        new_state = %{state | episodes: new_episodes}

        {:reply, {:ok, episode.id}, new_state}

      {:error, _reason} ->
        episode = Episode.new(text, action, outcome, tags, [])
        world_episodes = Map.get(state.episodes, world_id, %{})
        new_world_episodes = Map.put(world_episodes, episode.id, episode)
        new_episodes = Map.put(state.episodes, world_id, new_world_episodes)
        new_state = %{state | episodes: new_episodes}

        {:reply, {:ok, episode.id}, new_state}
    end
  end

  @impl true
  def handle_call({:add_episode_direct, episode, world_id}, _from, state) do
    VectorIndex.insert(state.episode_index, {world_id, episode.id}, episode.embedding)

    world_episodes = Map.get(state.episodes, world_id, %{})
    new_world_episodes = Map.put(world_episodes, episode.id, episode)
    new_episodes = Map.put(state.episodes, world_id, new_world_episodes)
    new_state = %{state | episodes: new_episodes}

    {:reply, {:ok, episode.id}, new_state}
  end

  @impl true
  def handle_call({:query_similar, text, k, world_id}, _from, state) do
    case get_embedding(world_id, text) do
      {:ok, query_embedding} ->
        world_episodes = Map.get(state.episodes, world_id, %{})

        results =
          VectorIndex.search_all(state.episode_index, query_embedding, k * 2)
          |> Enum.filter(fn {{wid, _id}, _score} -> wid == world_id end)
          |> Enum.take(k)
          |> Enum.map(fn {{_wid, id}, similarity} ->
            episode = Map.get(world_episodes, id)
            {episode, similarity}
          end)
          |> Enum.filter(fn {ep, _} -> ep != nil end)

        {:reply, {:ok, results}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:query_by_tags, tags, limit, world_id}, _from, state) do
    tag_set = MapSet.new(tags)
    world_episodes = Map.get(state.episodes, world_id, %{})

    results =
      world_episodes
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
  def handle_call({:get_episode, id, world_id}, _from, state) do
    world_episodes = Map.get(state.episodes, world_id, %{})

    case Map.get(world_episodes, id) do
      nil -> {:reply, {:error, :not_found}, state}
      episode -> {:reply, {:ok, episode}, state}
    end
  end

  @impl true
  def handle_call({:add_semantic, semantic, world_id}, _from, state) do
    VectorIndex.insert(state.semantic_index, {world_id, semantic.id}, semantic.embedding)

    world_semantics = Map.get(state.semantics, world_id, %{})
    new_world_semantics = Map.put(world_semantics, semantic.id, semantic)
    new_semantics = Map.put(state.semantics, world_id, new_world_semantics)
    new_state = %{state | semantics: new_semantics}

    {:reply, {:ok, semantic.id}, new_state}
  end

  @impl true
  def handle_call({:query_semantic, text, k, world_id}, _from, state) do
    case get_embedding(world_id, text) do
      {:ok, query_embedding} ->
        world_semantics = Map.get(state.semantics, world_id, %{})

        results =
          VectorIndex.search_all(state.semantic_index, query_embedding, k * 2)
          |> Enum.filter(fn {{wid, _id}, _score} -> wid == world_id end)
          |> Enum.take(k)
          |> Enum.map(fn {{_wid, id}, similarity} ->
            semantic = Map.get(world_semantics, id)
            {semantic, similarity}
          end)
          |> Enum.filter(fn {s, _} -> s != nil end)

        {:reply, {:ok, results}, state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:get_semantic, id, world_id}, _from, state) do
    world_semantics = Map.get(state.semantics, world_id, %{})

    case Map.get(world_semantics, id) do
      nil -> {:reply, {:error, :not_found}, state}
      semantic -> {:reply, {:ok, semantic}, state}
    end
  end

  @impl true
  def handle_call({:link_episode, episode_id, semantic_id, world_id}, _from, state) do
    world_episodes = Map.get(state.episodes, world_id, %{})

    case Map.get(world_episodes, episode_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      episode ->
        updated = %{episode | semantic_id: semantic_id}
        new_world_episodes = Map.put(world_episodes, episode_id, updated)
        new_episodes = Map.put(state.episodes, world_id, new_world_episodes)
        {:reply, :ok, %{state | episodes: new_episodes}}
    end
  end

  @impl true
  def handle_call({:all_episodes, world_id}, _from, state) do
    world_episodes = Map.get(state.episodes, world_id, %{})
    {:reply, {:ok, Map.values(world_episodes)}, state}
  end

  @impl true
  def handle_call({:all_semantics, world_id}, _from, state) do
    world_semantics = Map.get(state.semantics, world_id, %{})
    {:reply, {:ok, Map.values(world_semantics)}, state}
  end

  @impl true
  def handle_call({:stats, nil}, _from, state) do
    stats = %{
      episode_count: count_all_episodes(state.episodes),
      semantic_count: count_all_semantics(state.semantics),
      episode_index_size: VectorIndex.count(state.episode_index),
      semantic_index_size: VectorIndex.count(state.semantic_index),
      worlds: Map.keys(state.episodes) |> Enum.uniq()
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call({:stats, world_id}, _from, state) do
    world_episodes = Map.get(state.episodes, world_id, %{})
    world_semantics = Map.get(state.semantics, world_id, %{})

    stats = %{
      episode_count: map_size(world_episodes),
      semantic_count: map_size(world_semantics),
      world_id: world_id
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    result = persist_to_disk(state)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear, nil}, _from, state) do
    VectorIndex.clear(state.episode_index)
    VectorIndex.clear(state.semantic_index)

    new_state = %{state | episodes: %{}, semantics: %{}}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:clear, world_id}, _from, state) do
    new_episodes = Map.delete(state.episodes, world_id)
    new_semantics = Map.delete(state.semantics, world_id)

    new_state = %{state | episodes: new_episodes, semantics: new_semantics}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:list_worlds, _from, state) do
    episode_worlds = Map.keys(state.episodes)
    semantic_worlds = Map.keys(state.semantics)
    all_worlds = Enum.uniq(episode_worlds ++ semantic_worlds)
    {:reply, {:ok, all_worlds}, state}
  end

  defp count_all_episodes(episodes) do
    episodes
    |> Map.values()
    |> Enum.reduce(0, fn world_eps, acc -> acc + map_size(world_eps) end)
  end

  defp count_all_semantics(semantics) do
    semantics
    |> Map.values()
    |> Enum.reduce(0, fn world_sems, acc -> acc + map_size(world_sems) end)
  end

  defp maybe_load_from_disk(state) do
    path = state.persistence_path

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          try do
            data = :erlang.binary_to_term(binary)
            {episodes, semantics} = migrate_data_format(data)

            Enum.each(episodes, fn {world_id, world_episodes} ->
              Enum.each(world_episodes, fn {id, ep} ->
                if is_list(ep.embedding) and ep.embedding != [] do
                  VectorIndex.insert(state.episode_index, {world_id, id}, ep.embedding)
                end
              end)
            end)

            Enum.each(semantics, fn {world_id, world_semantics} ->
              Enum.each(world_semantics, fn {id, sem} ->
                if is_list(sem.embedding) and sem.embedding != [] do
                  VectorIndex.insert(state.semantic_index, {world_id, id}, sem.embedding)
                end
              end)
            end)

            Logger.info("Loaded memory store from disk",
              episodes: count_all_episodes(episodes),
              semantics: count_all_semantics(semantics),
              worlds: map_size(episodes)
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

  defp migrate_data_format(data) do
    episodes = Map.get(data, :episodes, %{})
    semantics = Map.get(data, :semantics, %{})

    episodes =
      if is_old_format?(episodes) do
        %{@default_world_id => episodes}
      else
        episodes
      end

    semantics =
      if is_old_format?(semantics) do
        %{@default_world_id => semantics}
      else
        semantics
      end

    {episodes, semantics}
  end

  defp is_old_format?(data) when is_map(data) do
    case Map.values(data) |> List.first() do
      %Episode{} -> true
      %SemanticFact{} -> true
      _ -> false
    end
  end

  defp is_old_format?(_) do
    false
  end

  defp persist_to_disk(state) do
    path = state.persistence_path
    path |> Path.dirname() |> File.mkdir_p!()

    data = %{
      episodes: state.episodes,
      semantics: state.semantics,
      version: 2
    }

    binary = :erlang.term_to_binary(data)

    case File.write(path, binary) do
      :ok ->
        Logger.info("Memory store persisted to disk",
          episodes: count_all_episodes(state.episodes),
          semantics: count_all_semantics(state.semantics),
          worlds: map_size(state.episodes)
        )

        :ok

      {:error, reason} ->
        Logger.error("Failed to persist memory store: #{inspect(reason)}")
        {:error, reason}
    end
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