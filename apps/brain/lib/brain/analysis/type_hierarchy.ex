defmodule Brain.Analysis.TypeHierarchy do
  @moduledoc """
  Entity type hierarchy with IS-A relationships.

  The hierarchy is the static one in `priv/analysis/entity_types.json`,
  loaded at startup and shared by every world.

  It is not read back from Atlas. Knowledge graphs are per world, and this
  hierarchy is not; the only IS_A edges between EntityType nodes in any
  knowledge graph are the ones `sync_to_atlas/2` copies there from the same
  static file, so reading them back would add nothing.

  This enables type narrowing: a "person" entity can be narrowed to "artist"
  when the intent context expects an artist, because artist IS-A person.

  ## API

      TypeHierarchy.is_a?("artist", "person")  # => true
      TypeHierarchy.specializations("person")   # => ["artist", "name", "given_name", "last_name"]
      TypeHierarchy.compatible?("person", "artist")  # => true
      TypeHierarchy.narrowing_candidates("person", ["artist", "song"]) # => ["artist"]
  """

  use GenServer
  alias Atlas.Graph.EdgeLabels
  alias Brain.Provenance
  require Logger

  @ets_table :type_hierarchy
  @entity_types_path Path.join(:code.priv_dir(:brain), "analysis/entity_types.json")

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Check if `child_type` IS-A `parent_type`.

  Returns true if `child_type` sits under `parent_type`, directly or through
  intermediate types, in the hierarchy of `entity_types.json`. Nothing else
  relates two types: a type the file does not declare is related to nothing
  but itself.

  ## Examples

      iex> TypeHierarchy.is_a?("artist", "person")
      true
      iex> TypeHierarchy.is_a?("person", "artist")
      false
  """
  def is_a?(child_type, parent_type) when is_binary(child_type) and is_binary(parent_type) do
    children = specializations(parent_type)

    child_type in children or Enum.any?(children, &is_a?(child_type, &1))
  end

  @doc """
  Get all specializations (children) of a parent type.

  Reads from ETS for fast concurrent access.

  ## Examples

      iex> TypeHierarchy.specializations("person")
      ["artist", "name", "given_name", "last_name"]
  """
  def specializations(parent_type) when is_binary(parent_type) do
    case safe_ets_lookup(parent_type) do
      [{^parent_type, children}] -> children
      _ -> []
    end
  end

  @doc """
  Check if `entity_type` is compatible with `expected_type`.

  Compatible means either:
  - They are the same type (exact match)
  - `entity_type` is a parent of `expected_type` (entity could be narrowed)
  - `entity_type` is a child of `expected_type` (entity is already more specific)

  ## Examples

      iex> TypeHierarchy.compatible?("person", "artist")
      true   # person could be narrowed to artist
      iex> TypeHierarchy.compatible?("artist", "person")
      true   # artist is already a person
      iex> TypeHierarchy.compatible?("person", "song")
      false  # person cannot become a song
  """
  def compatible?(entity_type, expected_type) when is_binary(entity_type) and is_binary(expected_type) do
    entity_type == expected_type or
      is_a?(expected_type, entity_type) or
      is_a?(entity_type, expected_type)
  end

  @doc """
  Given an entity's current type and a list of expected types,
  return which expected types the entity could be narrowed to.

  Only returns types that are specializations of the entity's current type.

  ## Examples

      iex> TypeHierarchy.narrowing_candidates("person", ["artist", "song", "album"])
      ["artist"]
  """
  def narrowing_candidates(entity_type, expected_types) when is_binary(entity_type) and is_list(expected_types) do
    Enum.filter(expected_types, &is_a?(&1, entity_type))
  end

  @doc """
  Check if a type is a parent type (has children in the hierarchy).
  """
  def parent_type?(entity_type) when is_binary(entity_type) do
    specializations(entity_type) != []
  end

  @doc """
  Every parent type of a child type, sorted; `[]` for a type with none.

  A type can sit under several parents (`color` is both `information` and a
  property of a `device`), and each is kept: choosing one would drop the others.

  ## Examples

      iex> TypeHierarchy.parents_of("artist")
      ["person"]
      iex> TypeHierarchy.parents_of("person")
      []
  """
  def parents_of(child_type) when is_binary(child_type) do
    case safe_ets_lookup(:_all_parents) do
      [{:_all_parents, parent_map}] -> Map.get(parent_map, child_type, [])
      _ -> []
    end
  end

  @doc """
  Every entity type `entity_types.json` declares, sorted: each type its
  hierarchy lists as a parent, including a root listed with no kinds under it
  (`"event": []`), and each type listed under one.
  """
  def all_types do
    case safe_ets_lookup(:_all_types) do
      [{:_all_types, types}] -> types
      _ -> []
    end
  end

  @doc """
  Returns just the parent type names from the hierarchy (sorted).

  These are the types that have children in the type_hierarchy.
  """
  def parent_types do
    case safe_ets_lookup(:_all_parents) do
      [{:_all_parents, parent_map}] ->
        parent_map |> Map.values() |> List.flatten() |> Enum.uniq() |> Enum.sort()

      _ ->
        []
    end
  end

  @doc """
  Look up a config value from the `config` section of entity_types.json.

  Falls back to `default` if the key is not found or ETS is unavailable.
  Nested keys can be accessed by passing a list of keys.

  ## Examples

      iex> TypeHierarchy.config("default_propn_type")
      "person"

      iex> TypeHierarchy.config(["pos_tag_roles", "proper_noun"])
      "PROPN"

      iex> TypeHierarchy.config("nonexistent", "fallback")
      "fallback"
  """
  # Three outcomes that used to be one. A key the file declares, a key it does
  # not, and a table that could not be read at all are different facts: the last
  # means the store is down, which is not the same as a value being unset, and
  # returning the caller's fallback for both is how a dead lookup table reads as
  # a configuration choice. `Brain.Provenance` separates them for whoever is
  # looking; the returned value is unchanged either way.
  def config(key, default \\ nil)

  def config(key, default) when is_binary(key) do
    case safe_ets_lookup(:_config) do
      [{:_config, config_map}] ->
        case Map.fetch(config_map, key) do
          {:ok, value} -> declared(key, value)
          :error -> defaulted(key, default)
        end

      _ ->
        unavailable(key, default)
    end
  end

  def config(keys, default) when is_list(keys) do
    case safe_ets_lookup(:_config) do
      [{:_config, config_map}] ->
        case get_in(config_map, keys) do
          nil -> defaulted(Enum.join(keys, "."), default)
          value -> declared(Enum.join(keys, "."), value)
        end

      _ ->
        unavailable(Enum.join(keys, "."), default)
    end
  end

  defp declared(key, value) do
    Provenance.record(["config", key], value, :declared,
      source: "TypeHierarchy.config/2",
      meta: %{"file" => "priv/analysis/entity_types.json"}
    )
  end

  defp defaulted(key, default) do
    Provenance.record(["config", key], default, :default,
      source: "TypeHierarchy.config/2",
      meta: %{"reason" => "entity_types.json declares no #{key}"}
    )
  end

  defp unavailable(key, default) do
    Provenance.record(["config", key], default, :unavailable,
      source: "TypeHierarchy.config/2",
      meta: %{"reason" => "the TypeHierarchy ETS table could not be read"}
    )
  end

  @doc "Check if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Reload the hierarchy from its file."
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 5_000)
  end

  @doc """
  Seed the world's Atlas knowledge_graph with IS_A edges for the static
  hierarchy.

  Creates EntityType nodes and IS_A relationships so the hierarchy
  is queryable via Cypher in that world's graph. Raises for a missing or
  malformed world.
  """
  def sync_to_atlas(world_id, name \\ __MODULE__) do
    graph = Atlas.Graph.world_graph("knowledge_graph", world_id)
    GenServer.cast(name, {:sync_to_atlas, graph})
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    ensure_ets_table()
    load_hierarchy()
    {:ok, %{loaded: true}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.loaded, state}
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    load_hierarchy()
    {:reply, :ok, %{loaded: true}}
  end

  @impl true
  def handle_cast({:sync_to_atlas, graph}, state) do
    do_sync_to_atlas(graph)
    {:noreply, state}
  end

  # ============================================================================
  # Private
  # ============================================================================

  defp ensure_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end
  end

  defp load_hierarchy do
    {merged, config} = load_static_hierarchy_and_config()

    parent_map = build_parent_map(merged)

    Enum.each(merged, fn {parent, children} ->
      :ets.insert(@ets_table, {parent, children})
    end)

    :ets.insert(@ets_table, {:_all_parents, parent_map})
    :ets.insert(@ets_table, {:_all_types, (Map.keys(merged) ++ Map.keys(parent_map)) |> Enum.uniq() |> Enum.sort()})
    :ets.insert(@ets_table, {:_config, config})

    total_types = Enum.reduce(merged, 0, fn {_p, children}, acc -> acc + length(children) end)
    Logger.debug("TypeHierarchy loaded: #{map_size(merged)} parent types, #{total_types} child types")
  end

  defp load_static_hierarchy_and_config do
    case File.read(@entity_types_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} ->
            hierarchy = Map.get(data, "type_hierarchy", %{})
            config = Map.get(data, "config", %{})
            {hierarchy, config}

          {:error, reason} ->
            Logger.warning("TypeHierarchy: failed to parse entity_types.json: #{inspect(reason)}")
            {%{}, %{}}
        end

      {:error, reason} ->
        Logger.warning("TypeHierarchy: failed to read entity_types.json: #{inspect(reason)}")
        {%{}, %{}}
    end
  end

  # child => every parent it is listed under, sorted.
  defp build_parent_map(hierarchy) do
    hierarchy
    |> Enum.reduce(%{}, fn {parent, children}, acc ->
      Enum.reduce(children, acc, fn child, inner_acc ->
        Map.update(inner_acc, child, [parent], &[parent | &1])
      end)
    end)
    |> Map.new(fn {child, parents} -> {child, parents |> Enum.uniq() |> Enum.sort()} end)
  end

  defp do_sync_to_atlas(graph) do
    if Brain.AtlasIntegration.available?() do
      {static, _config} = load_static_hierarchy_and_config()

      Enum.each(static, fn {parent, children} ->
        Brain.AtlasIntegration.sync(fn ->
          {:ok, parent_node} =
            Brain.AtlasIntegration.ensure_node(graph, "EntityType", %{name: parent})

          Brain.AtlasIntegration.enrich_existing_node(graph, parent_node.id, parent)

          Enum.each(children, fn child ->
            {:ok, child_node} =
              Brain.AtlasIntegration.ensure_node(graph, "EntityType", %{name: child})

            Brain.AtlasIntegration.enrich_existing_node(graph, child_node.id, child)

            parent_id = parent_node.id
            child_id = child_node.id

            Atlas.Graph.cypher(graph,
              "MATCH (c:EntityType), (p:EntityType) WHERE id(c) = #{child_id} AND id(p) = #{parent_id} " <>
              "MERGE (c)-[:#{EdgeLabels.is_a()}]->(p) RETURN c, p"
            )
          end)
        end)
      end)

      Logger.info("TypeHierarchy: synced #{map_size(static)} parent types to Atlas #{graph}")
    end
  rescue
    e ->
      Logger.warning("TypeHierarchy: Atlas sync failed: #{Exception.message(e)}")
  end

  defp safe_ets_lookup(key) do
    if :ets.whereis(@ets_table) != :undefined do
      :ets.lookup(@ets_table, key)
    else
      []
    end
  rescue
    ArgumentError -> []
  end
end
