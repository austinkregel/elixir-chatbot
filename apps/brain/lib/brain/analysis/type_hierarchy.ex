defmodule Brain.Analysis.TypeHierarchy do
  @moduledoc """
  Entity type hierarchy with IS-A relationships.

  Provides a unified view of entity type relationships from two sources:

  1. **Static hierarchy** from `priv/analysis/entity_types.json` (loaded at startup)
  2. **Learned hierarchy** from Atlas knowledge_graph IS_A edges (enriched at runtime)

  This enables type narrowing: a "person" entity can be narrowed to "artist"
  when the intent context expects an artist, because artist IS-A person.

  ## API

      TypeHierarchy.is_a?("artist", "person")  # => true
      TypeHierarchy.specializations("person")   # => ["artist", "music-artist", "name"]
      TypeHierarchy.compatible?("person", "artist")  # => true
      TypeHierarchy.narrowing_candidates("person", ["artist", "song"]) # => ["artist"]
  """

  use GenServer
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

  Returns true if there is a direct IS-A relationship in the hierarchy.
  Falls back to WordNet hypernym chains when no static or Atlas edge exists.

  ## Examples

      iex> TypeHierarchy.is_a?("artist", "person")
      true
      iex> TypeHierarchy.is_a?("person", "artist")
      false
  """
  def is_a?(child_type, parent_type) when is_binary(child_type) and is_binary(parent_type) do
    children = specializations(parent_type)

    child_type in children or
      Enum.any?(children, &is_a?(child_type, &1)) or
      wordnet_is_a?(child_type, parent_type)
  end

  @doc """
  Get all specializations (children) of a parent type.

  Reads from ETS for fast concurrent access.

  ## Examples

      iex> TypeHierarchy.specializations("person")
      ["artist", "music-artist", "name"]
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
  Get the parent type for a given child type, if any.

  ## Examples

      iex> TypeHierarchy.parent_of("artist")
      "person"
      iex> TypeHierarchy.parent_of("person")
      nil
  """
  def parent_of(child_type) when is_binary(child_type) do
    case safe_ets_lookup(:_all_parents) do
      [{:_all_parents, parent_map}] -> Map.get(parent_map, child_type)
      _ -> nil
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
  def config(key, default \\ nil)

  def config(key, default) when is_binary(key) do
    case safe_ets_lookup(:_config) do
      [{:_config, config_map}] -> Map.get(config_map, key, default)
      _ -> default
    end
  end

  def config(keys, default) when is_list(keys) do
    case safe_ets_lookup(:_config) do
      [{:_config, config_map}] -> get_in(config_map, keys) || default
      _ -> default
    end
  end

  @doc "Check if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Reload hierarchy from file and Atlas."
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 5_000)
  end

  @doc """
  Seed the Atlas knowledge_graph with IS_A edges for the static hierarchy.

  Creates EntityType nodes and IS_A relationships so the hierarchy
  is queryable via Cypher and can be extended with learned relationships.
  """
  def sync_to_atlas(name \\ __MODULE__) do
    GenServer.cast(name, :sync_to_atlas)
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
  def handle_cast(:sync_to_atlas, state) do
    do_sync_to_atlas()
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
    {static, config} = load_static_hierarchy_and_config()
    atlas = load_atlas_hierarchy()

    merged = merge_hierarchies(static, atlas)

    parent_map = build_parent_map(merged)

    Enum.each(merged, fn {parent, children} ->
      :ets.insert(@ets_table, {parent, children})
    end)

    :ets.insert(@ets_table, {:_all_parents, parent_map})
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

  defp load_atlas_hierarchy do
    if Brain.AtlasIntegration.available?() do
      case Brain.AtlasIntegration.sync(fn ->
        Atlas.Graph.cypher("knowledge_graph",
          "MATCH (child:EntityType)-[:IS_A]->(parent:EntityType) RETURN parent.name, child.name"
        )
      end) do
        {:ok, {:ok, rows}} ->
          Enum.reduce(rows, %{}, fn [parent_name, child_name], acc ->
            parent = to_string(parent_name)
            child = to_string(child_name)
            Map.update(acc, parent, [child], fn children ->
              if child in children, do: children, else: [child | children]
            end)
          end)

        _ ->
          %{}
      end
    else
      %{}
    end
  end

  defp merge_hierarchies(static, atlas) do
    Map.merge(static, atlas, fn _parent, static_children, atlas_children ->
      Enum.uniq(static_children ++ atlas_children)
    end)
  end

  defp build_parent_map(hierarchy) do
    Enum.reduce(hierarchy, %{}, fn {parent, children}, acc ->
      Enum.reduce(children, acc, fn child, inner_acc ->
        Map.put(inner_acc, child, parent)
      end)
    end)
  end

  defp do_sync_to_atlas do
    if Brain.AtlasIntegration.available?() do
      {static, _config} = load_static_hierarchy_and_config()

      Enum.each(static, fn {parent, children} ->
        Brain.AtlasIntegration.sync(fn ->
          {:ok, parent_node} =
            Brain.AtlasIntegration.ensure_node("knowledge_graph", "EntityType", %{name: parent})

          Enum.each(children, fn child ->
            {:ok, child_node} =
              Brain.AtlasIntegration.ensure_node("knowledge_graph", "EntityType", %{name: child})

            parent_id = parent_node.id
            child_id = child_node.id

            Atlas.Graph.cypher("knowledge_graph",
              "MATCH (c:EntityType), (p:EntityType) WHERE id(c) = #{child_id} AND id(p) = #{parent_id} " <>
              "MERGE (c)-[:IS_A]->(p) RETURN c, p"
            )
          end)
        end)
      end)

      Logger.info("TypeHierarchy: synced #{map_size(static)} parent types to Atlas knowledge_graph")
    end
  rescue
    e ->
      Logger.warning("TypeHierarchy: Atlas sync failed: #{Exception.message(e)}")
  end

  @wordnet_type_roots %{
    "person" => ~w(person individual someone somebody mortal human),
    "location" => ~w(location place area region),
    "organization" => ~w(organization institution establishment),
    "artifact" => ~w(artifact artefact),
    "event" => ~w(event happening occurrence),
    "concept" => ~w(concept conception idea abstraction),
    "animal" => ~w(animal animate_being beast brute creature fauna),
    "plant" => ~w(plant flora plant_life),
    "food" => ~w(food nutrient),
    "substance" => ~w(substance matter)
  }

  defp wordnet_is_a?(child_type, parent_type) do
    if Process.whereis(Brain.ML.Lexicon) do
      chain = Brain.ML.Lexicon.hypernym_chain(child_type, :noun, max_depth: 10)
      root_words = Map.get(@wordnet_type_roots, parent_type, [parent_type])

      Enum.any?(chain, fn word ->
        word in root_words or word == parent_type
      end)
    else
      false
    end
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
