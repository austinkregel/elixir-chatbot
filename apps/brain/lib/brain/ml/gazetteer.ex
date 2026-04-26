defmodule Brain.ML.Gazetteer do
  @moduledoc """
  Fast entity lookup using ETS tables.

  The gazetteer provides efficient lookup of entities from various sources:
  - Cities (from world-cities.csv)
  - Music artists (from Global Music Artists.csv)
  - Devices, rooms, and other entities (from entities/*.json)
  - Emojis (from emojis.csv)

  Uses ETS tables for O(1) average lookup time and concurrent read access.
  Also supports prefix matching for multi-word entity detection.
  """

  use GenServer
  require Logger

  alias Brain.ML.DataLoaders
  alias Brain.Telemetry

  # Default table names for the global instance
  @default_table_name :gazetteer_entities
  @default_prefix_table :gazetteer_prefixes
  @default_stats_table :gazetteer_stats
  @default_world_overlay_table :gazetteer_world_overlays

  # Module attribute aliases for backward compatibility with direct ETS access
  @table_name @default_table_name
  @prefix_table @default_prefix_table
  @stats_table @default_stats_table
  @world_overlay_table @default_world_overlay_table

  @type entity_match :: %{
          entity_type: String.t(),
          value: String.t(),
          confidence: float(),
          metadata: map()
        }

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the Gazetteer GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
    - `:table_prefix` - Prefix for ETS table names (default: nil, uses global tables)
      When set, creates isolated ETS tables named `{prefix}_entities`, etc.
      This is useful for test isolation.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Initialize and load all gazetteers from data files.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def load_all(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :load_all, :infinity)
  end

  @doc """
  Returns true if the gazetteer is loaded and ready.

  ## Options
    - `:server` - The server to check (default: `#{__MODULE__}`)
  """
  def is_loaded?(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    table = get_table_name(server, :entities)

    Process.whereis(server) != nil and
      :ets.info(table) != :undefined
  rescue
    _ -> false
  end

  @doc """
  Gets the ETS table name for a specific table type.
  Useful for isolated instances that use custom table names.
  """
  def get_table_name(server \\ __MODULE__, table_type) do
    case GenServer.call(server, {:get_table_name, table_type}) do
      {:ok, name} -> name
      _ -> default_table_name(table_type)
    end
  catch
    :exit, _ -> default_table_name(table_type)
  end

  defp default_table_name(:entities), do: @default_table_name
  defp default_table_name(:prefixes), do: @default_prefix_table
  defp default_table_name(:stats), do: @default_stats_table
  defp default_table_name(:world_overlays), do: @default_world_overlay_table

  @doc """
  Look up an entity by exact match (case-insensitive).
  Returns {:ok, entity_info} or {:ok, [entity_info, ...]} or :not_found.

  When multiple entity types exist for the same key (e.g., "Austin" as both
  a person name and a city), returns a list of all matching types.
  """
  def lookup(text) when is_binary(text) do
    normalized = normalize(text)

    case :ets.lookup(@table_name, normalized) do
      [{^normalized, entity_infos}] when is_list(entity_infos) ->
        {:ok, entity_infos}

      [{^normalized, entity_info}] when is_map(entity_info) ->
        # Legacy format - single entity
        {:ok, entity_info}

      [] ->
        :not_found
    end
  rescue
    ArgumentError -> :not_found
  end

  @doc """
  Look up an entity and return all possible types.
  Always returns a list (empty if not found).
  """
  def lookup_all_types(text) when is_binary(text) do
    case lookup(text) do
      {:ok, infos} when is_list(infos) -> infos
      {:ok, info} when is_map(info) -> [info]
      :not_found -> []
    end
  end

  @doc """
  Look up multiple potential entity spans from a list of tokens.
  Returns a list of {start_index, end_index, entity_info} tuples.
  Prioritizes longer matches (multi-word entities).

  ## Options
    - `:domain` - Domain atom (e.g. `:weather`, `:music`) to rank multi-type results
    - `:intent` - Intent string (e.g. `"weather.query"`) for Poincare/domain scoring
    - `:world_id` - World ID for overlay lookups
  """
  def lookup_spans(tokens, opts \\ [])

  def lookup_spans(tokens, opts) when is_list(tokens) do
    Telemetry.span(:gazetteer_lookup, %{token_count: length(tokens)}, fn ->
      do_lookup_spans(tokens, opts)
    end)
  end

  @doc """
  Look up all span hypotheses including overlapping candidates with scores.
  Returns a list of {start_index, end_index, entity_infos, score} tuples.
  Overlapping spans are retained for the caller to resolve.

  ## Options
    Same as `lookup_spans/2`.
  """
  def lookup_spans_lattice(tokens, opts \\ []) when is_list(tokens) do
      Telemetry.span(:gazetteer_lookup, %{token_count: length(tokens)}, fn ->
      token_count = length(tokens)
      domain = Keyword.get(opts, :domain)
      intent = Keyword.get(opts, :intent)

      find_all_spans_raw(tokens, 0, token_count, [])
      |> Enum.map(fn {start_idx, end_idx, entity_infos} ->
        span_len = end_idx - start_idx + 1
        ranked = rank_types_with_context(entity_infos, domain: domain, intent: intent)
        score = compute_span_score(ranked, span_len, domain, intent)
        {start_idx, end_idx, ranked, score}
      end)
      |> Enum.sort_by(fn {start, _end, _infos, score} -> {-score, start} end)
    end)
  rescue
    ArgumentError -> []
  end

  @doc """
  Record that a type was successfully used for a surface form in a domain context.
  Adjusts future rankings for this key+domain combination.
  """
  def record_type_preference(surface_form, preferred_type, domain)
      when is_binary(surface_form) and is_binary(preferred_type) do
    normalized = normalize(surface_form)

    case :ets.lookup(@table_name, normalized) do
      [{^normalized, infos}] when is_list(infos) ->
        {preferred, rest} =
          Enum.split_with(infos, fn info ->
            (Map.get(info, :entity_type) || Map.get(info, :type)) == preferred_type
          end)

        if preferred != [] do
          boosted =
            Enum.map(preferred, fn info ->
              domain_prefs = Map.get(info, :domain_preferences, %{})
              updated_prefs = Map.update(domain_prefs, domain, 1, &(&1 + 1))
              Map.put(info, :domain_preferences, updated_prefs)
            end)

          :ets.insert(@table_name, {normalized, boosted ++ rest})
          :ok
        else
          :not_found
        end

      _ ->
        :not_found
    end
  rescue
    _ -> :error
  end

  defp do_lookup_spans(tokens, opts) do
    token_count = length(tokens)
    domain = Keyword.get(opts, :domain)
    intent = Keyword.get(opts, :intent)

    spans = find_all_spans_raw(tokens, 0, token_count, [])

    ranked_spans =
      if domain != nil or intent != nil do
        Enum.map(spans, fn {start_idx, end_idx, entity_infos} ->
          ranked = rank_types_with_context(entity_infos, domain: domain, intent: intent)
          {start_idx, end_idx, ranked}
        end)
      else
        spans
      end

    ranked_spans
    |> Enum.sort_by(fn {start, end_idx, _info} -> {-(end_idx - start), start} end)
    |> remove_overlapping_spans([])
  rescue
    ArgumentError -> []
  end

  @doc """
  Check if a text is a known prefix of any entity.
  Useful for multi-word entity detection during streaming.
  """
  def is_prefix?(text) when is_binary(text) do
    normalized = normalize(text)

    case :ets.lookup(@prefix_table, normalized) do
      [{^normalized, true}] -> true
      [] -> false
    end
  rescue
    ArgumentError -> false
  end

  @doc """
  Get statistics about loaded gazetteers.
  """
  def stats do
    case :ets.lookup(@stats_table, :stats) do
      [{:stats, stats}] -> stats
      [] -> %{}
    end
  rescue
    ArgumentError -> %{}
  end

  @doc """
  Check if gazetteers are loaded.
  """
  def loaded? do
    case stats() do
      %{loaded: true} -> true
      _ -> false
    end
  end

  @doc """
  Add a new entity to the gazetteer dynamically.

  ## Parameters
    - name: The entity name (e.g., "New York City")
    - entity_type: The type (e.g., "location", "city", "device")
    - metadata: Optional additional metadata

  ## Returns
    - {:ok, normalized_key} on success
    - {:error, reason} on failure
  """
  def add_entry(name, entity_type, metadata \\ %{})
      when is_binary(name) and is_binary(entity_type) do
    GenServer.call(__MODULE__, {:add_entry, name, entity_type, metadata})
  end

  @doc """
  Remove an entity from the gazetteer.
  """
  def remove_entry(name) when is_binary(name) do
    GenServer.call(__MODULE__, {:remove_entry, name})
  end

  @doc """
  Clear all entities of a given type.
  Returns {:ok, count} with the number of entries removed.
  """
  def clear_by_type(entity_type) when is_binary(entity_type) do
    GenServer.call(__MODULE__, {:clear_by_type, entity_type}, 60_000)
  end

  @doc """
  Clear all admin-added entities.
  Returns {:ok, count} with the number of entries removed.
  """
  def clear_admin_entries do
    GenServer.call(__MODULE__, :clear_admin_entries, 60_000)
  end

  @doc """
  Clear all entities from the gazetteer.
  Returns {:ok, count} with the number of entries removed.
  """
  def clear_all do
    GenServer.call(__MODULE__, :clear_all, 60_000)
  end

  @doc """
  Check if an entity already exists in the gazetteer.
  Returns {true, entity_infos} if it exists (list of all types), false otherwise.
  """
  def exists?(name) when is_binary(name) do
    try do
      normalized_key = normalize(name)

      case :ets.lookup(@table_name, normalized_key) do
        [{^normalized_key, infos}] when is_list(infos) -> {true, infos}
        [{^normalized_key, info}] when is_map(info) -> {true, [info]}
        [] -> false
      end
    rescue
      _ -> false
    end
  end

  @doc """
  List all entities of a given type.
  Returns a list of {name, entity_info} tuples.
  """
  def list_by_type(entity_type) when is_binary(entity_type) do
    try do
      :ets.tab2list(@table_name)
      |> Enum.flat_map(fn {key, infos} ->
        # Handle both list and single entity formats
        info_list = if is_list(infos), do: infos, else: [infos]

        info_list
        |> Enum.filter(fn info ->
          Map.get(info, :entity_type) == entity_type or
            Map.get(info, :type) == entity_type
        end)
        |> Enum.map(fn info -> {key, info} end)
      end)
      |> Enum.sort_by(fn {key, _} -> key end)
    rescue
      _ -> []
    end
  end

  @doc """
  List all entity types in the gazetteer.
  """
  def list_types do
    try do
      :ets.tab2list(@table_name)
      |> Enum.flat_map(fn {_key, infos} ->
        # Handle both list and single entity formats
        info_list = if is_list(infos), do: infos, else: [infos]

        Enum.map(info_list, fn info ->
          Map.get(info, :entity_type) || Map.get(info, :type) || "unknown"
        end)
      end)
      |> Enum.uniq()
      |> Enum.sort()
    rescue
      _ -> []
    end
  end

  @doc """
  Search entities by partial name match.
  """
  def search(query) when is_binary(query) do
    normalized_query = normalize(query)

    try do
      :ets.tab2list(@table_name)
      |> Enum.filter(fn {key, _info} ->
        String.contains?(key, normalized_query)
      end)
      |> Enum.take(50)
      |> Enum.sort_by(fn {key, _} -> key end)
    rescue
      _ -> []
    end
  end

  # ============================================================================
  # World Overlay API - For Training Worlds
  # ============================================================================

  @doc """
  Look up an entity, checking the world overlay first if provided.
  Falls back to base gazetteer if not found in overlay.
  """
  def lookup(text, world_id) when is_binary(text) and is_binary(world_id) do
    normalized = normalize(text)

    # Check world overlay first
    case lookup_world_overlay(normalized, world_id) do
      {:ok, result} -> {:ok, result}
      :not_found -> lookup(text)
    end
  end

  @doc """
  Look up all types for an entity, including world overlay.
  """
  def lookup_all_types(text, world_id) when is_binary(text) and is_binary(world_id) do
    normalized = normalize(text)

    # Get from world overlay
    overlay_types =
      case lookup_world_overlay(normalized, world_id) do
        {:ok, infos} when is_list(infos) -> infos
        {:ok, info} when is_map(info) -> [info]
        :not_found -> []
      end

    # Get from base gazetteer
    base_types = lookup_all_types(text)

    # Merge, preferring overlay (more recent/specific)
    merge_entity_types(overlay_types, base_types)
  end

  @doc """
  Creates an overlay namespace for a training world.
  """
  def create_world_overlay(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:create_world_overlay, world_id})
  end

  @doc """
  Destroys the overlay namespace for a training world.
  """
  def destroy_world_overlay(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:destroy_world_overlay, world_id})
  end

  @doc """
  Adds an entity to a world's overlay (not the base gazetteer).
  """
  def add_to_world(world_id, text, entity_type, metadata \\ %{})
      when is_binary(world_id) and is_binary(text) and is_binary(entity_type) do
    GenServer.call(__MODULE__, {:add_to_world, world_id, text, entity_type, metadata})
  end

  @doc """
  Gets all entities in a world's overlay.
  """
  def get_world_overlay(world_id) when is_binary(world_id) do
    try do
      :ets.match_object(@world_overlay_table, {{world_id, :_}, :_})
      |> Enum.map(fn {{_world_id, key}, info} -> {key, info} end)
    rescue
      ArgumentError -> []
    end
  end

  @doc """
  Restores a world overlay from saved data.
  """
  def restore_world_overlay(world_id, overlay_data)
      when is_binary(world_id) and is_list(overlay_data) do
    GenServer.call(__MODULE__, {:restore_world_overlay, world_id, overlay_data})
  end

  @doc """
  Removes an entity from a world's overlay.
  """
  def remove_from_world(world_id, text) when is_binary(world_id) and is_binary(text) do
    GenServer.call(__MODULE__, {:remove_from_world, world_id, text})
  end

  defp lookup_world_overlay(normalized_key, world_id) do
    try do
      case :ets.lookup(@world_overlay_table, {world_id, normalized_key}) do
        [{{^world_id, ^normalized_key}, entity_infos}] when is_list(entity_infos) ->
          {:ok, entity_infos}

        [{{^world_id, ^normalized_key}, entity_info}] when is_map(entity_info) ->
          {:ok, entity_info}

        [] ->
          :not_found
      end
    rescue
      ArgumentError -> :not_found
    end
  end

  defp merge_entity_types(overlay_types, base_types) do
    # Get entity types from overlay
    overlay_type_set =
      overlay_types
      |> Enum.map(&(Map.get(&1, :entity_type) || Map.get(&1, :type)))
      |> MapSet.new()

    # Filter base types to exclude those already in overlay
    filtered_base =
      Enum.reject(base_types, fn info ->
        type = Map.get(info, :entity_type) || Map.get(info, :type)
        MapSet.member?(overlay_type_set, type)
      end)

    overlay_types ++ filtered_base
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(opts) do
    # Determine table names - use prefix for isolation or defaults for global
    table_prefix = Keyword.get(opts, :table_prefix)

    tables =
      if table_prefix do
        # Isolated tables for testing
        %{
          entities: :"#{table_prefix}_entities",
          prefixes: :"#{table_prefix}_prefixes",
          stats: :"#{table_prefix}_stats",
          world_overlays: :"#{table_prefix}_world_overlays"
        }
      else
        # Default global tables
        %{
          entities: @default_table_name,
          prefixes: @default_prefix_table,
          stats: @default_stats_table,
          world_overlays: @default_world_overlay_table
        }
      end

    # Create ETS tables (with isolation support)
    create_tables(tables)

    state = %{
      loaded: false,
      tables: tables,
      isolated: table_prefix != nil
    }

    {:ok, state}
  end

  @impl true
  def handle_call({:get_table_name, table_type}, _from, state) do
    table_name =
      case table_type do
        :entities -> state.tables.entities
        :prefixes -> state.tables.prefixes
        :stats -> state.tables.stats
        :world_overlays -> state.tables.world_overlays
        _ -> nil
      end

    {:reply, {:ok, table_name}, state}
  end

  @impl true
  def handle_call(:load_all, _from, state) do
    Logger.info("Loading all gazetteers...")

    stats = %{
      entities: 0,
      prefixes: 0,
      cities: 0,
      artists: 0,
      emojis: 0,
      us_cities: 0,
      entity_types: 0,
      loaded: false,
      load_time_ms: 0
    }

    start_time = System.monotonic_time(:millisecond)

    # Load and index entities from JSON files
    stats =
      case DataLoaders.load_all_entities() do
        {:ok, entities} ->
          entity_lookup = DataLoaders.build_entity_lookup(entities)
          indexed = index_entities(entity_lookup, "json_entity")
          %{stats | entities: stats.entities + indexed, entity_types: map_size(entities)}

        {:error, _} ->
          stats
      end

    # Load and index world cities
    stats =
      case DataLoaders.load_cities() do
        {:ok, cities} ->
          city_lookup = DataLoaders.build_city_lookup(cities)
          indexed = index_entities(city_lookup, :cities)
          %{stats | entities: stats.entities + indexed, cities: length(cities)}

        {:error, _} ->
          stats
      end

    # Load and index US cities (comprehensive dataset)
    stats =
      case DataLoaders.load_us_cities() do
        {:ok, us_cities} ->
          us_city_lookup = DataLoaders.build_us_city_lookup(us_cities)
          indexed = index_entities(us_city_lookup, :us_cities)
          Logger.info("Loaded US cities", %{count: length(us_cities), indexed: indexed})
          %{stats | entities: stats.entities + indexed, us_cities: length(us_cities)}

        {:error, _} ->
          stats
      end

    # Load and index artists
    stats =
      case DataLoaders.load_artists() do
        {:ok, artists} ->
          artist_lookup = DataLoaders.build_artist_lookup(artists)
          indexed = index_entities(artist_lookup, "artist")
          %{stats | entities: stats.entities + indexed, artists: length(artists)}

        {:error, _} ->
          stats
      end

    # Load and index emojis
    stats =
      case DataLoaders.load_emojis() do
        {:ok, emojis} ->
          emoji_lookup = DataLoaders.build_emoji_lookup(emojis)
          indexed = index_entities(emoji_lookup, "emoji")
          %{stats | entities: stats.entities + indexed, emojis: length(emojis)}

        {:error, _} ->
          stats
      end

    # Enrich from Atlas knowledge_graph entities
    atlas_synced = sync_from_atlas()

    stats = %{stats | entities: stats.entities + atlas_synced}

    # Build prefix index
    prefix_count = build_prefix_index()

    end_time = System.monotonic_time(:millisecond)
    load_time = end_time - start_time

    final_stats = %{stats | prefixes: prefix_count, loaded: true, load_time_ms: load_time}

    :ets.insert(@stats_table, {:stats, final_stats})

    Logger.info("Gazetteers loaded", %{
      entities: final_stats.entities,
      prefixes: final_stats.prefixes,
      cities: final_stats.cities,
      artists: final_stats.artists,
      emojis: final_stats.emojis,
      atlas_synced: atlas_synced,
      load_time_ms: load_time
    })

    {:reply, {:ok, final_stats}, %{state | loaded: true}}
  end

  @impl true
  def handle_call({:add_entry, name, entity_type, metadata}, _from, state) do
    normalized_key = normalize(name)

    # Check if entry already exists
    case :ets.lookup(@table_name, normalized_key) do
      [{^normalized_key, existing_info}] ->
        existing_type = existing_info[:entity_type] || existing_info[:type]
        {:reply, {:error, {:duplicate, existing_type}}, state}

      [] ->
        entity_info =
          metadata
          |> Map.put(:entity_type, entity_type)
          |> Map.put(:type, entity_type)
          |> Map.put(:value, name)
          |> Map.put(:original_name, name)
          |> Map.put(:source, :admin)
          |> Map.put(:added_at, System.system_time(:second))

        :ets.insert(@table_name, {normalized_key, entity_info})

        # Update prefixes if multi-word
        words = String.split(normalized_key)

        if length(words) > 1 do
          prefixes =
            1..(length(words) - 1)
            |> Enum.map(fn n -> Enum.take(words, n) |> Enum.join(" ") end)

          Enum.each(prefixes, fn prefix ->
            :ets.insert(@prefix_table, {prefix, true})
          end)
        end

        # Update stats
        case :ets.lookup(@stats_table, :stats) do
          [{:stats, current_stats}] ->
            new_stats = Map.update(current_stats, :entities, 1, &(&1 + 1))
            :ets.insert(@stats_table, {:stats, new_stats})

          _ ->
            :ok
        end

        Logger.info("Added gazetteer entry", %{name: name, type: entity_type})
        {:reply, {:ok, normalized_key}, state}
    end
  end

  @impl true
  def handle_call({:remove_entry, name}, _from, state) do
    normalized_key = normalize(name)

    case :ets.lookup(@table_name, normalized_key) do
      [{^normalized_key, _info}] ->
        :ets.delete(@table_name, normalized_key)

        # Update stats
        case :ets.lookup(@stats_table, :stats) do
          [{:stats, current_stats}] ->
            new_stats = Map.update(current_stats, :entities, 0, &max(&1 - 1, 0))
            :ets.insert(@stats_table, {:stats, new_stats})

          _ ->
            :ok
        end

        Logger.info("Removed gazetteer entry", %{name: name})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:clear_by_type, entity_type}, _from, state) do
    # Find and delete all entries of the given type
    entries_to_delete =
      :ets.tab2list(@table_name)
      |> Enum.filter(fn {_key, info} ->
        Map.get(info, :entity_type) == entity_type or
          Map.get(info, :type) == entity_type
      end)

    count = length(entries_to_delete)

    Enum.each(entries_to_delete, fn {key, _info} ->
      :ets.delete(@table_name, key)
    end)

    # Update stats
    update_entity_count(-count)

    Logger.info("Cleared gazetteer entries by type", %{type: entity_type, count: count})
    {:reply, {:ok, count}, state}
  end

  @impl true
  def handle_call(:clear_admin_entries, _from, state) do
    # Find and delete all admin-added entries
    entries_to_delete =
      :ets.tab2list(@table_name)
      |> Enum.filter(fn {_key, info} ->
        Map.get(info, :source) == :admin
      end)

    count = length(entries_to_delete)

    Enum.each(entries_to_delete, fn {key, _info} ->
      :ets.delete(@table_name, key)
    end)

    # Update stats
    update_entity_count(-count)

    Logger.info("Cleared admin-added gazetteer entries", %{count: count})
    {:reply, {:ok, count}, state}
  end

  @impl true
  def handle_call(:clear_all, _from, state) do
    # Count before clearing
    count = :ets.info(@table_name, :size) || 0

    # Clear all tables
    :ets.delete_all_objects(@table_name)
    :ets.delete_all_objects(@prefix_table)

    # Reset stats
    :ets.insert(@stats_table, {:stats, %{entities: 0, loaded: false}})

    Logger.info("Cleared all gazetteer entries", %{count: count})
    {:reply, {:ok, count}, state}
  end

  # World Overlay Handlers

  @impl true
  def handle_call({:create_world_overlay, world_id}, _from, state) do
    # Just mark that this world exists - entries are added individually
    :ets.insert(
      @world_overlay_table,
      {{world_id, :_meta}, %{created_at: System.system_time(:second)}}
    )

    Logger.debug("Created world overlay", %{world_id: world_id})
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:destroy_world_overlay, world_id}, _from, state) do
    # Delete all entries for this world
    entries = :ets.match_object(@world_overlay_table, {{world_id, :_}, :_})
    Enum.each(entries, fn {key, _} -> :ets.delete(@world_overlay_table, key) end)

    Logger.debug("Destroyed world overlay", %{
      world_id: world_id,
      entries_removed: length(entries)
    })

    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:add_to_world, world_id, text, entity_type, metadata}, _from, state) do
    normalized_key = normalize(text)
    key = {world_id, normalized_key}

    entity_info =
      metadata
      |> Map.put(:entity_type, entity_type)
      |> Map.put(:type, entity_type)
      |> Map.put(:value, text)
      |> Map.put(:original_name, text)
      |> Map.put(:source, :world_learning)
      |> Map.put(:world_id, world_id)
      |> Map.put(:added_at, System.system_time(:second))

    # Check if entry already exists
    case :ets.lookup(@world_overlay_table, key) do
      [{^key, existing_infos}] when is_list(existing_infos) ->
        # Check if this type already exists
        already_has_type =
          Enum.any?(existing_infos, fn info ->
            (Map.get(info, :entity_type) || Map.get(info, :type)) == entity_type
          end)

        if already_has_type do
          {:reply, {:error, {:duplicate, entity_type}}, state}
        else
          :ets.insert(@world_overlay_table, {key, [entity_info | existing_infos]})
          {:reply, {:ok, normalized_key}, state}
        end

      [{^key, existing_info}] when is_map(existing_info) ->
        existing_type = Map.get(existing_info, :entity_type) || Map.get(existing_info, :type)

        if existing_type == entity_type do
          {:reply, {:error, {:duplicate, entity_type}}, state}
        else
          :ets.insert(@world_overlay_table, {key, [entity_info, existing_info]})
          {:reply, {:ok, normalized_key}, state}
        end

      [] ->
        :ets.insert(@world_overlay_table, {key, [entity_info]})
        {:reply, {:ok, normalized_key}, state}
    end
  end

  @impl true
  def handle_call({:restore_world_overlay, world_id, overlay_data}, _from, state) do
    # First clean any existing overlay
    entries = :ets.match_object(@world_overlay_table, {{world_id, :_}, :_})
    Enum.each(entries, fn {key, _} -> :ets.delete(@world_overlay_table, key) end)

    # Restore all entries
    Enum.each(overlay_data, fn {key, info} ->
      :ets.insert(@world_overlay_table, {{world_id, key}, info})
    end)

    Logger.debug("Restored world overlay", %{world_id: world_id, entries: length(overlay_data)})
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:remove_from_world, world_id, text}, _from, state) do
    normalized_key = normalize(text)
    key = {world_id, normalized_key}

    case :ets.lookup(@world_overlay_table, key) do
      [{^key, _}] ->
        :ets.delete(@world_overlay_table, key)
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp sync_from_atlas do
    case Brain.Graph.Training.collect_gazetteer_entries() do
      {:ok, entries} ->
        count =
          Enum.count(entries, fn {name, entity_type, metadata} ->
            insert_entry_direct(name, entity_type, metadata)
          end)

        Logger.debug("Gazetteer enriched from Atlas knowledge_graph (#{count} entries)")
        count

      {:error, reason} ->
        Logger.debug("Atlas Gazetteer sync skipped: #{inspect(reason)}")
        0
    end
  rescue
    e ->
      Logger.debug("Atlas Gazetteer sync unavailable: #{inspect(e)}")
      0
  end

  defp insert_entry_direct(name, entity_type, metadata) when is_binary(name) and name != "" do
    normalized_key = normalize(name)

    entity_info =
      metadata
      |> Map.put(:entity_type, entity_type)
      |> Map.put(:type, entity_type)
      |> Map.put(:value, name)
      |> Map.put(:original_name, name)
      |> Map.put(:source, :atlas)
      |> Map.put(:added_at, System.system_time(:second))

    case :ets.lookup(@table_name, normalized_key) do
      [{^normalized_key, existing}] ->
        existing_list =
          case existing do
            l when is_list(l) -> l
            m when is_map(m) -> [m]
          end

        already_has_type =
          Enum.any?(existing_list, fn entry ->
            Map.get(entry, :entity_type) == entity_type
          end)

        if already_has_type do
          false
        else
          :ets.insert(@table_name, {normalized_key, [entity_info | existing_list]})
          true
        end

      [] ->
        :ets.insert(@table_name, {normalized_key, entity_info})

        words = String.split(normalized_key)

        if length(words) > 1 do
          prefixes =
            1..(length(words) - 1)
            |> Enum.map(fn n -> Enum.take(words, n) |> Enum.join(" ") end)

          Enum.each(prefixes, fn prefix ->
            :ets.insert(@prefix_table, {prefix, true})
          end)
        end

        true
    end
  end

  defp insert_entry_direct(_, _, _), do: false

  defp update_entity_count(delta) do
    case :ets.lookup(@stats_table, :stats) do
      [{:stats, current_stats}] ->
        new_count = max((current_stats[:entities] || 0) + delta, 0)
        new_stats = Map.put(current_stats, :entities, new_count)
        :ets.insert(@stats_table, {:stats, new_stats})

      _ ->
        :ok
    end
  end

  defp create_tables(tables) do
    # Main entity lookup table
    create_table(tables.entities)

    # Prefix table for multi-word entity detection
    create_table(tables.prefixes)

    # Stats table
    create_table(tables.stats)

    # World overlay table for training worlds
    create_table(tables.world_overlays)
  end

  defp create_table(table_name) do
    if :ets.whereis(table_name) != :undefined do
      :ets.delete(table_name)
    end

    :ets.new(table_name, [:set, :public, :named_table, read_concurrency: true])
  end

  defp index_entities(lookup_map, source) do
    Enum.reduce(lookup_map, 0, fn {normalized_key, entity_info}, count ->
      # Handle both single entity and list of entities (from expanded ambiguous entries)
      entities_to_add =
        case entity_info do
          infos when is_list(infos) ->
            Enum.map(infos, &Map.put(&1, :source, source))

          info when is_map(info) ->
            [Map.put(info, :source, source)]
        end

      # Append to existing entries instead of overwriting
      # This allows multiple entity types per key (e.g., "Austin" as person AND location)
      existing =
        case :ets.lookup(@table_name, normalized_key) do
          [{^normalized_key, infos}] when is_list(infos) -> infos
          [{^normalized_key, info}] when is_map(info) -> [info]
          [] -> []
        end

      # Only add entities whose type isn't already present
      new_entries =
        Enum.filter(entities_to_add, fn enriched_info ->
          entity_type = Map.get(enriched_info, :entity_type) || Map.get(enriched_info, :type)

          not Enum.any?(existing, fn ex ->
            ex_type = Map.get(ex, :entity_type) || Map.get(ex, :type)
            ex_type == entity_type
          end)
        end)

      if new_entries != [] do
        :ets.insert(@table_name, {normalized_key, new_entries ++ existing})
      end

      count + length(entities_to_add)
    end)
  end

  defp build_prefix_index do
    # Build prefixes for all multi-word entities
    # This enables efficient lookup of entities like "New York"

    entities = :ets.tab2list(@table_name)

    prefixes =
      Enum.flat_map(entities, fn {key, _infos} ->
        # Handle both list and single entity formats
        words = String.split(key)

        if length(words) > 1 do
          # Generate all prefixes
          1..(length(words) - 1)
          |> Enum.map(fn n -> Enum.take(words, n) |> Enum.join(" ") end)
        else
          []
        end
      end)
      |> Enum.uniq()

    Enum.each(prefixes, fn prefix ->
      :ets.insert(@prefix_table, {prefix, true})
    end)

    length(prefixes)
  end

  defp find_all_spans_raw(_tokens, start_idx, token_count, acc) when start_idx >= token_count do
    acc
  end

  defp find_all_spans_raw(tokens, start_idx, token_count, acc) do
    max_span = min(5, token_count - start_idx)

    new_acc = find_all_matches_at(tokens, start_idx, max_span, acc)
    find_all_spans_raw(tokens, start_idx + 1, token_count, new_acc)
  end

  defp find_all_matches_at(tokens, start_idx, max_span, acc) do
    if max_span < 1 do
      acc
    else
      max_span..1//-1
      |> Enum.reduce(acc, fn span_len, inner_acc ->
        span_tokens = Enum.slice(tokens, start_idx, span_len)
        phrase = Enum.join(span_tokens, " ")
        normalized = normalize(phrase)

        case :ets.lookup(@table_name, normalized) do
          [{^normalized, entity_infos}] when is_list(entity_infos) ->
            end_idx = start_idx + span_len - 1
            [{start_idx, end_idx, entity_infos} | inner_acc]

          [{^normalized, entity_info}] when is_map(entity_info) ->
            end_idx = start_idx + span_len - 1
            [{start_idx, end_idx, [entity_info]} | inner_acc]

          [] ->
            inner_acc
        end
      end)
    end
  end

  defp remove_overlapping_spans([], resolved), do: Enum.reverse(resolved)

  defp remove_overlapping_spans([span | rest], resolved) do
    {start_idx, end_idx, _info} = span

    # Check if this span overlaps with any resolved span
    overlaps =
      Enum.any?(resolved, fn {res_start, res_end, _} ->
        # Overlaps if ranges intersect
        start_idx <= res_end and end_idx >= res_start
      end)

    if overlaps do
      # Skip this span (a longer span was already added)
      remove_overlapping_spans(rest, resolved)
    else
      # Add this span
      remove_overlapping_spans(rest, [span | resolved])
    end
  end

  @doc """
  Rank a list of entity type infos using contextual signals.
  Returns the same list reordered so the best match for the context is first.
  """
  def rank_types_with_context(infos, opts \\ [])

  def rank_types_with_context(infos, _opts) when length(infos) <= 1, do: infos

  def rank_types_with_context(infos, opts) when is_list(infos) do
    domain = Keyword.get(opts, :domain)
    intent = Keyword.get(opts, :intent)

    if domain == nil and intent == nil do
      infos
    else
      adjustments = Brain.Analysis.TypeHierarchy.config("confidence_adjustments", %{})
      source_priority = Brain.Analysis.TypeHierarchy.config("source_priority", %{})

      Enum.sort_by(infos, fn info ->
        entity_type = Map.get(info, :entity_type) || Map.get(info, :type) || "unknown"
        source = Map.get(info, :source, :unknown) |> to_string()

        domain_score = domain_alignment_score(entity_type, domain)
        poincare_score = poincare_proximity_score(entity_type, intent)
        conceptnet_score = conceptnet_isa_score(entity_type, domain)
        learned_score = learned_preference_score(info, domain)

        type_score = Map.get(adjustments, entity_type, 0.0)
        source_score = Map.get(source_priority, source, 0)

        total = domain_score * 3.0 + poincare_score * 2.0 +
                conceptnet_score * 1.5 + learned_score * 2.0 +
                source_score * 0.5 + type_score * 0.3

        -total
      end)
    end
  end

  defp domain_alignment_score(_entity_type, nil), do: 0.0

  defp domain_alignment_score(entity_type, domain) do
    groups = load_disambiguation_groups()
    domain_str = to_string(domain)

    matching_group =
      Enum.find(groups, fn {group_name, types} ->
        String.contains?(to_string(group_name), domain_str) and entity_type in types
      end)

    if matching_group, do: 1.0, else: 0.0
  end

  defp poincare_proximity_score(_entity_type, nil), do: 0.0

  defp poincare_proximity_score(entity_type, intent) do
    if Code.ensure_loaded?(Brain.ML.Poincare.Embeddings) and
         function_exported?(Brain.ML.Poincare.Embeddings, :entity_distance, 2) do
      try do
        case Brain.ML.Poincare.Embeddings.entity_distance(entity_type, intent) do
          {:ok, distance} when is_number(distance) ->
            1.0 / (1.0 + distance)

          _ ->
            0.0
        end
      catch
        _, _ -> 0.0
      end
    else
      0.0
    end
  end

  defp conceptnet_isa_score(_entity_type, nil), do: 0.0

  defp conceptnet_isa_score(entity_type, domain) do
    if Code.ensure_loaded?(Brain.Lexicon.ConceptNet) do
      try do
        isa_relations = Brain.Lexicon.ConceptNet.related(entity_type, "IsA")
        domain_str = to_string(domain)

        if Enum.any?(isa_relations, fn related ->
             String.contains?(String.downcase(related), domain_str) or
               related == domain_str
           end) do
          1.0
        else
          0.0
        end
      catch
        _, _ -> 0.0
      end
    else
      0.0
    end
  end

  defp learned_preference_score(_info, nil), do: 0.0

  defp learned_preference_score(info, domain) do
    domain_prefs = Map.get(info, :domain_preferences, %{})
    count = Map.get(domain_prefs, domain, 0)
    min(count * 0.5, 2.0)
  end

  defp compute_span_score(ranked_infos, span_len, domain, intent) do
    length_bonus = span_len * 0.5

    type_score =
      case ranked_infos do
        [best | _] ->
          entity_type = Map.get(best, :entity_type) || Map.get(best, :type) || "unknown"
          domain_alignment_score(entity_type, domain) +
            poincare_proximity_score(entity_type, intent) * 0.5
        [] -> 0.0
      end

    length_bonus + type_score
  end

  defp load_disambiguation_groups do
    path = Path.join(:code.priv_dir(:brain), "analysis/entity_types.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} -> Map.get(data, "disambiguation_groups", %{})
          _ -> %{}
        end

      _ ->
        %{}
    end
  rescue
    _ -> %{}
  end

  defp normalize(text) when is_binary(text) do
    text
    |> String.downcase()
    |> String.trim()
  end
end
