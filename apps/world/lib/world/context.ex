defmodule World.Context do
  @moduledoc """
  Provides unified API for world-scoped data access with inheritance.

  All data lookups (entities, memories, intents, knowledge) go through this module,
  which resolves data through the world's inheritance chain:

    1. World-specific data
    2. Base world data (if world has a base_world)
    3. Empty/default

  This ensures complete data isolation between worlds while allowing
  template-based inheritance for efficient world creation.
  """

  alias World.Manager, as: WorldManager, as: WorldManager
  alias Brain.ML.Gazetteer

  require Logger

  @default_world_id "default"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Returns the default world ID.
  """
  def default_world_id, do: @default_world_id

  @doc """
  Gets the inheritance chain for a world (world -> base -> base's base -> ...).
  Returns a list of world IDs starting with the given world.
  """
  def get_inheritance_chain(nil), do: [@default_world_id]

  def get_inheritance_chain(world_id) when is_binary(world_id) do
    build_chain(world_id, [], MapSet.new())
  end

  defp build_chain(world_id, chain, visited) do
    if MapSet.member?(visited, world_id) do
      # Circular reference, stop here
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
          # World not found, return what we have
          if chain == [] do
            [@default_world_id]
          else
            Enum.reverse(chain)
          end
      end
    end
  end

  @doc """
  Resolves data through the world inheritance chain.

  ## Parameters
    - world_id: The world to start the lookup from
    - data_type: One of :entity, :episode, :semantic, :intent, :knowledge
    - lookup_fn: Function that takes a world_id and returns {:ok, result} or {:error, _}

  Returns the first successful result from the inheritance chain.
  """
  def resolve(world_id, _data_type, lookup_fn) when is_function(lookup_fn, 1) do
    chain = get_inheritance_chain(world_id)

    Enum.find_value(chain, {:error, :not_found}, fn wid ->
      case lookup_fn.(wid) do
        {:ok, result} -> {:ok, result, wid}
        _ -> nil
      end
    end)
  end

  @doc """
  Looks up an entity in the world's gazetteer (with inheritance).
  """
  def lookup_entity(world_id, text) do
    chain = get_inheritance_chain(world_id)

    # Try world overlays first (in inheritance order)
    # Fall back to base gazetteer
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

  @doc """
  Gets all entities for a world (merged from inheritance chain).
  """
  def get_all_entities(world_id) do
    chain = get_inheritance_chain(world_id)

    # Collect overlays from all worlds in chain (child overrides parent)
    chain
    # Start from base, apply overrides
    |> Enum.reverse()
    |> Enum.reduce(%{}, fn wid, acc ->
      overlay = Gazetteer.get_world_overlay(wid)

      Enum.reduce(overlay, acc, fn {key, value}, map ->
        Map.put(map, key, value)
      end)
    end)
  end

  @doc """
  Gets episodes for a world (with optional inheritance).

  Options:
    - inherit: boolean, whether to include episodes from base worlds (default: false)
  """
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

  @doc """
  Gets semantic facts for a world (with optional inheritance).
  """
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

  @doc """
  Gets knowledge for a world (with inheritance - child overrides parent).
  """
  def get_knowledge(world_id, category \\ nil) do
    chain = get_inheritance_chain(world_id)

    # Merge from base to child (child overrides)
    chain
    |> Enum.reverse()
    |> Enum.reduce(%{}, fn wid, acc ->
      world_knowledge = get_world_knowledge_direct(wid, category)
      deep_merge(acc, world_knowledge)
    end)
  end

  @doc """
  Classifies intent using world-specific model (with inheritance fallback).
  """
  def classify_intent(world_id, text) do
    chain = get_inheritance_chain(world_id)

    # Try each world's classifier in order
    Enum.find_value(chain, {:error, :no_classifier}, fn wid ->
      case classify_with_world_model(wid, text) do
        {:ok, result} -> {:ok, result}
        _ -> nil
      end
    end)
  end

  @doc """
  Adds an episode to a specific world.
  """
  def add_episode(world_id, state, action, outcome, tags) do
    # Use world-scoped memory store
    Brain.Memory.Store.add_episode(state, action, outcome, tags, world_id: world_id)
  end

  @doc """
  Adds a semantic fact to a specific world.
  """
  def add_semantic(world_id, semantic) do
    Brain.Memory.Store.add_semantic(semantic, world_id: world_id)
  end

  @doc """
  Adds knowledge to a specific world.
  """
  def add_knowledge(world_id, category, key, value) do
    Brain.KnowledgeStore.add_to_world(world_id, category, key, value)
  end

  @doc """
  Queries for similar episodes in a world (with inheritance).
  """
  def query_similar(world_id, text, k \\ 5, opts \\ []) do
    inherit = Keyword.get(opts, :inherit, true)

    if inherit do
      chain = get_inheritance_chain(world_id)

      # Query each world and merge results
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

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp get_world_episodes_direct(world_id) do
    case Brain.Memory.Store.all_episodes(world_id: world_id) do
      {:ok, episodes} -> episodes
      _ -> []
    end
  end

  defp get_world_semantics_direct(world_id) do
    case Brain.Memory.Store.all_semantics(world_id: world_id) do
      {:ok, semantics} -> semantics
      _ -> []
    end
  end

  defp get_world_knowledge_direct(world_id, category) do
    case Brain.KnowledgeStore.get_world_knowledge(world_id, category) do
      knowledge when is_map(knowledge) -> knowledge
      _ -> %{}
    end
  end

  defp query_world_similar(world_id, text, k) do
    Brain.Memory.Store.query_similar(text, k, world_id: world_id)
  end

  defp classify_with_world_model(world_id, text) do
    Brain.ML.IntentClassifierSimple.classify(text, world_id: world_id)
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

  defp deep_merge(_left, right), do: right
end
