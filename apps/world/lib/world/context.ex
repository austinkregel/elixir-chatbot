defmodule World.Context do
  @moduledoc "Provides a unified API for world-scoped data access.\n\nAll data lookups (entities, memories, intents, knowledge) go through this module,\nand each reads only the world it is given. A world's `base_world` names the\nworld it was seeded from; it is not read through at runtime.\n\nThere is no implicit world: a nil world, or a world the manager does not know,\nraises ArgumentError.\n"

  alias World.Manager, as: WorldManager
  alias Brain.ML.Gazetteer

  alias Brain.Memory.Store
  alias Brain.KnowledgeStore
  require Logger

  @doc "Resolves data in a world.\n\n## Parameters\n  - world_id: The world to look in\n  - data_type: One of :entity, :episode, :semantic, :intent, :knowledge\n  - lookup_fn: Function that takes a world_id and returns {:ok, result} or {:error, _}\n\nReturns `{:ok, result, world_id}` when the lookup succeeds, otherwise `{:error, :not_found}`.\n"
  def resolve(world_id, _data_type, lookup_fn) when is_function(lookup_fn, 1) do
    world_id = known_world!(world_id)

    case lookup_fn.(world_id) do
      {:ok, result} -> {:ok, result, world_id}
      _ -> {:error, :not_found}
    end
  end

  @doc "Looks up an entity in the world's gazetteer.\n"
  def lookup_entity(world_id, text) do
    world_id = known_world!(world_id)

    case Gazetteer.lookup(text, world_id) do
      {:ok, result} -> {:ok, result, world_id}
      error -> error
    end
  end

  @doc "Gets all entities in a world's gazetteer overlay.\n"
  def get_all_entities(world_id) do
    world_id
    |> known_world!()
    |> Gazetteer.get_world_overlay()
    |> Map.new()
  end

  @doc "Gets the episodes stored in a world.\n"
  def get_episodes(world_id) do
    world_id
    |> known_world!()
    |> get_world_episodes_direct()
  end

  @doc "Gets the semantic facts stored in a world.\n"
  def get_semantics(world_id) do
    world_id
    |> known_world!()
    |> get_world_semantics_direct()
  end

  @doc "Gets a world's knowledge, optionally for one category.\n"
  def get_knowledge(world_id, category \\ nil) do
    world_id
    |> known_world!()
    |> get_world_knowledge_direct(category)
  end

  @doc "Classifies intent using the world's model.\n"
  def classify_intent(world_id, text) do
    world_id
    |> known_world!()
    |> classify_with_world_model(text)
  end

  @doc "Adds an episode to a specific world.\n"
  def add_episode(world_id, state, action, outcome, tags) do
    Store.add_episode(state, action, outcome, tags, world_id: world_id)
  end

  @doc "Adds a semantic fact to a specific world.\n"
  def add_semantic(world_id, semantic) do
    Store.add_semantic(semantic, world_id: world_id)
  end

  @doc "Adds knowledge to a specific world.\n"
  def add_knowledge(world_id, category, key, value) do
    KnowledgeStore.add_to_world(world_id, category, key, value)
  end

  @doc "Queries for similar episodes in a world.\n"
  def query_similar(world_id, text, k \\ 5) do
    world_id
    |> known_world!()
    |> query_world_similar(text, k)
  end

  defp known_world!(nil) do
    raise ArgumentError, "World.Context needs a world id, got nil"
  end

  defp known_world!(world_id) when is_binary(world_id) do
    case WorldManager.get(world_id) do
      {:ok, _world} ->
        world_id

      {:error, reason} ->
        raise ArgumentError, "World.Context: world #{inspect(world_id)} is not known: #{inspect(reason)}"
    end
  end

  defp get_world_episodes_direct(world_id) do
    case Store.all_episodes(world_id: world_id) do
      {:ok, episodes} -> episodes
      _ -> []
    end
  end

  defp get_world_semantics_direct(world_id) do
    case Store.all_semantics(world_id: world_id) do
      {:ok, semantics} -> semantics
      _ -> []
    end
  end

  defp get_world_knowledge_direct(world_id, category) do
    case KnowledgeStore.get_world_knowledge(world_id, category) do
      knowledge when is_map(knowledge) -> knowledge
      _ -> %{}
    end
  end

  defp query_world_similar(world_id, text, k) do
    Store.query_similar(text, k, world_id: world_id)
  end

  defp classify_with_world_model(world_id, text) do
    if Code.ensure_loaded?(Brain.ML.MicroClassifiers) and Brain.ML.MicroClassifiers.ready?() do
      try do
        analysis = Brain.Analysis.Pipeline.analyze_chunk(text, world_id: world_id)
        {feature_vector, _word_feats} = Brain.Analysis.FeatureExtractor.extract(analysis)

        case Brain.ML.MicroClassifiers.classify_vector(:intent_full, feature_vector) do
          {:ok, intent, confidence} ->
            {:ok, %{intent: intent, confidence: confidence}}

          _ ->
            {:ok, %{intent: "unknown", confidence: 0.0}}
        end
      rescue
        _ -> {:ok, %{intent: "unknown", confidence: 0.0}}
      end
    else
      {:ok, %{intent: "unknown", confidence: 0.0}}
    end
  end
end
