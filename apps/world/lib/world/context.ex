defmodule World.Context do
  @moduledoc "Provides unified API for world-scoped data access with inheritance.\n\nAll data lookups (entities, memories, intents, knowledge) go through this module,\nwhich resolves data through the world's inheritance chain:\n\n  1. World-specific data\n  2. Base world data (if world has a base_world)\n  3. Empty/default\n\nThis ensures complete data isolation between worlds while allowing\ntemplate-based inheritance for efficient world creation.\n"

  alias World.Manager, as: WorldManager
  alias Brain.ML.Gazetteer

  alias Brain.ML.IntentClassifierSimple
  alias Brain.Memory.Store
  alias Brain.KnowledgeStore
  require Logger

  @default_world_id "default"

  @doc "Returns the default world ID.\n"
  def default_world_id do
    @default_world_id
  end

  @doc "Gets the inheritance chain for a world (world -> base -> base's base -> ...).\nReturns a list of world IDs starting with the given world.\n"
  def get_inheritance_chain(nil) do
    [@default_world_id]
  end

  def get_inheritance_chain(world_id) when is_binary(world_id) do
    build_chain(world_id, [], MapSet.new())
  end

  defp build_chain(world_id, chain, visited) do
    if MapSet.member?(visited, world_id) do
      Enum.reverse(chain)
    else
      case WorldManager.get(world_id) do
        {:ok, world} ->
          new_chain = [world_id | chain]
          new_visited = MapSet.put(visited, world_id)

          case world.base_world do
            nil -> Enum.reverse(new_chain)
            base_id -> build_chain(base_id, new_chain, new_visited)
          end

        {:error, _} ->
          if chain == [] do
            [@default_world_id]
          else
            Enum.reverse(chain)
          end
      end
    end
  end

  @doc "Resolves data through the world inheritance chain.\n\n## Parameters\n  - world_id: The world to start the lookup from\n  - data_type: One of :entity, :episode, :semantic, :intent, :knowledge\n  - lookup_fn: Function that takes a world_id and returns {:ok, result} or {:error, _}\n\nReturns the first successful result from the inheritance chain.\n"
  def resolve(world_id, _data_type, lookup_fn) when is_function(lookup_fn, 1) do
    chain = get_inheritance_chain(world_id)

    Enum.find_value(chain, {:error, :not_found}, fn wid ->
      case lookup_fn.(wid) do
        {:ok, result} -> {:ok, result, wid}
        _ -> nil
      end
    end)
  end

  @doc "Looks up an entity in the world's gazetteer (with inheritance).\n"
  def lookup_entity(world_id, text) do
    chain = get_inheritance_chain(world_id)

    Enum.find_value(chain, nil, fn wid ->
      case Gazetteer.lookup(text, wid) do
        {:ok, result} -> {:ok, result, wid}
        _ -> nil
      end
    end) ||
      case Gazetteer.lookup(text) do
        {:ok, result} -> {:ok, result, :base}
        error -> error
      end
  end

  @doc "Gets all entities for a world (merged from inheritance chain).\n"
  def get_all_entities(world_id) do
    chain = get_inheritance_chain(world_id)

    chain
    |> Enum.reverse()
    |> Enum.reduce(%{}, fn wid, acc ->
      overlay = Gazetteer.get_world_overlay(wid)

      Enum.reduce(overlay, acc, fn {key, value}, map ->
        Map.put(map, key, value)
      end)
    end)
  end

  @doc "Gets episodes for a world (with optional inheritance).\n\nOptions:\n  - inherit: boolean, whether to include episodes from base worlds (default: false)\n"
  def get_episodes(world_id, opts \\ []) do
    inherit = Keyword.get(opts, :inherit, false)

    if inherit do
      chain = get_inheritance_chain(world_id)

      Enum.flat_map(chain, fn wid ->
        get_world_episodes_direct(wid)
      end)
    else
      get_world_episodes_direct(world_id)
    end
  end

  @doc "Gets semantic facts for a world (with optional inheritance).\n"
  def get_semantics(world_id, opts \\ []) do
    inherit = Keyword.get(opts, :inherit, false)

    if inherit do
      chain = get_inheritance_chain(world_id)

      Enum.flat_map(chain, fn wid ->
        get_world_semantics_direct(wid)
      end)
    else
      get_world_semantics_direct(world_id)
    end
  end

  @doc "Gets knowledge for a world (with inheritance - child overrides parent).\n"
  def get_knowledge(world_id, category \\ nil) do
    chain = get_inheritance_chain(world_id)

    chain
    |> Enum.reverse()
    |> Enum.reduce(%{}, fn wid, acc ->
      world_knowledge = get_world_knowledge_direct(wid, category)
      deep_merge(acc, world_knowledge)
    end)
  end

  @doc "Classifies intent using world-specific model (with inheritance fallback).\n"
  def classify_intent(world_id, text) do
    chain = get_inheritance_chain(world_id)

    Enum.find_value(chain, {:error, :no_classifier}, fn wid ->
      case classify_with_world_model(wid, text) do
        {:ok, result} -> {:ok, result}
        _ -> nil
      end
    end)
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

  @doc "Queries for similar episodes in a world (with inheritance).\n"
  def query_similar(world_id, text, k \\ 5, opts \\ []) do
    inherit = Keyword.get(opts, :inherit, true)

    if inherit do
      chain = get_inheritance_chain(world_id)

      results =
        chain
        |> Enum.flat_map(fn wid ->
          case query_world_similar(wid, text, k) do
            {:ok, episodes} -> episodes
            _ -> []
          end
        end)
        |> Enum.sort_by(fn {_ep, score} -> -score end)
        |> Enum.take(k)

      {:ok, results}
    else
      query_world_similar(world_id, text, k)
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
    IntentClassifierSimple.classify(text, world_id: world_id)
  end

  defp deep_merge(left, right) when is_map(left) and is_map(right) do
    Map.merge(left, right, fn _key, l, r ->
      if is_map(l) and is_map(r) do
        deep_merge(l, r)
      else
        r
      end
    end)
  end

  defp deep_merge(_left, right) do
    right
  end
end
