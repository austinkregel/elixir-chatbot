defmodule Brain.Code.CodeGazetteer do
  @moduledoc """
  Stores and retrieves code symbols (functions, classes, variables, etc.).

  This module provides a specialized gazetteer for code entities, separate
  from the natural language entity gazetteer. It supports:

  - World-scoped symbol storage for isolation
  - Qualified name lookups (e.g., `Module.function`)
  - Symbol type filtering
  - Relationship tracking (callers, callees, dependencies)

  ## Entity Types

  The following code entity types are supported:

  | Type | Description | Examples |
  |------|-------------|----------|
  | `code.function` | Function/method definitions | `def foo`, `function bar` |
  | `code.class` | Class/struct/module definitions | `class User`, `defmodule App` |
  | `code.variable` | Variable declarations | `let x`, `$name` |
  | `code.type` | Type definitions/annotations | `int`, `String`, `List[T]` |
  | `code.keyword` | Language keywords | `if`, `def`, `public` |
  | `code.import` | Import/require statements | `import os`, `use GenServer` |
  | `code.constant` | Constants | `MAX_SIZE`, `PI` |
  | `code.parameter` | Function parameters | `def foo(x, y)` - x, y |
  | `code.field` | Class/struct fields | `this.name`, `@name` |
  | `code.namespace` | Namespaces/packages | `namespace App`, `package main` |

  ## Architecture

  Uses ETS tables for O(1) lookups:
  - `:code_gazetteer_symbols` - Main symbol storage
  - `:code_gazetteer_qualified` - Qualified name index
  - `:code_gazetteer_by_type` - Type-based index
  - `:code_gazetteer_relations` - Symbol relationships
  """

  use GenServer
  require Logger
  alias Brain.Telemetry

  @ets_symbols :code_gazetteer_symbols
  @ets_qualified :code_gazetteer_qualified
  @ets_by_type :code_gazetteer_by_type
  @ets_relations :code_gazetteer_relations
  @ets_stats :code_gazetteer_stats

  # All supported code entity types
  @entity_types [
    "code.function",
    "code.class",
    "code.variable",
    "code.type",
    "code.keyword",
    "code.import",
    "code.constant",
    "code.parameter",
    "code.field",
    "code.namespace",
    "code.method",
    "code.interface",
    "code.enum",
    "code.macro"
  ]

  @type symbol :: %{
          name: String.t(),
          qualified_name: String.t(),
          entity_type: String.t(),
          language: atom(),
          file_path: String.t() | nil,
          line: non_neg_integer() | nil,
          column: non_neg_integer() | nil,
          metadata: map(),
          world_id: String.t()
        }

  @type relation_type :: :calls | :called_by | :extends | :implements | :imports | :uses

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the CodeGazetteer GenServer.
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Adds a symbol to the gazetteer.

  ## Parameters
    - `world_id` - The world to add the symbol to
    - `symbol` - A map with symbol information

  ## Required symbol fields
    - `:name` - The symbol name
    - `:entity_type` - One of the supported entity types
    - `:language` - The programming language

  ## Optional symbol fields
    - `:qualified_name` - Full qualified name (defaults to name)
    - `:file_path` - Source file path
    - `:line` - Line number
    - `:column` - Column number
    - `:metadata` - Additional metadata

  ## Examples

      CodeGazetteer.add_symbol("world_123", %{
        name: "calculate_tax",
        qualified_name: "Billing.calculate_tax",
        entity_type: "code.function",
        language: :elixir,
        file_path: "lib/billing.ex",
        line: 42,
        metadata: %{arity: 2, visibility: :public}
      })
  """
  @spec add_symbol(String.t(), map()) :: {:ok, String.t()} | {:error, term()}
  def add_symbol(world_id, symbol) when is_binary(world_id) and is_map(symbol) do
    Telemetry.span(:code_gazetteer_add, %{world_id: world_id, entity_type: Map.get(symbol, :entity_type)}, fn ->
      GenServer.call(__MODULE__, {:add_symbol, world_id, symbol})
    end)
  end

  @doc """
  Looks up a symbol by name within a world.

  Returns all symbols matching the name (there may be multiple
  with different qualified names or types).
  """
  @spec lookup(String.t(), String.t()) :: {:ok, [symbol()]} | :not_found
  def lookup(world_id, name) when is_binary(world_id) and is_binary(name) do
    Telemetry.span(:code_gazetteer_lookup, %{world_id: world_id, name: name}, fn ->
      normalized = normalize_name(name)
      key = {world_id, normalized}

      try do
        case :ets.lookup(@ets_symbols, key) do
          [{^key, symbols}] when is_list(symbols) -> {:ok, symbols}
          [{^key, symbol}] -> {:ok, [symbol]}
          [] -> :not_found
        end
      rescue
        ArgumentError -> :not_found
      end
    end)
  end

  @doc """
  Looks up a symbol by qualified name.

  ## Examples

      CodeGazetteer.lookup_qualified("world_123", "Billing.calculate_tax")
  """
  @spec lookup_qualified(String.t(), String.t()) :: {:ok, symbol()} | :not_found
  def lookup_qualified(world_id, qualified_name) when is_binary(world_id) and is_binary(qualified_name) do
    Telemetry.span(:code_gazetteer_lookup, %{world_id: world_id, qualified_name: qualified_name}, fn ->
      normalized = normalize_name(qualified_name)
      key = {world_id, normalized}

      try do
        case :ets.lookup(@ets_qualified, key) do
          [{^key, symbol}] -> {:ok, symbol}
          [] -> :not_found
        end
      rescue
        ArgumentError -> :not_found
      end
    end)
  end

  @doc """
  Lists all symbols of a specific type within a world.

  ## Examples

      CodeGazetteer.list_by_type("world_123", "code.function")
  """
  @spec list_by_type(String.t(), String.t()) :: [symbol()]
  def list_by_type(world_id, entity_type) when is_binary(world_id) and is_binary(entity_type) do
    key = {world_id, entity_type}

    try do
      case :ets.lookup(@ets_by_type, key) do
        [{^key, symbols}] -> symbols
        [] -> []
      end
    rescue
      ArgumentError -> []
    end
  end

  @doc """
  Lists all symbols in a file.
  """
  @spec list_by_file(String.t(), String.t()) :: [symbol()]
  def list_by_file(world_id, file_path) when is_binary(world_id) and is_binary(file_path) do
    try do
      :ets.tab2list(@ets_qualified)
      |> Enum.filter(fn {{wid, _}, symbol} ->
        wid == world_id and Map.get(symbol, :file_path) == file_path
      end)
      |> Enum.map(fn {_, symbol} -> symbol end)
      |> Enum.sort_by(&Map.get(&1, :line, 0))
    rescue
      ArgumentError -> []
    end
  end

  @doc """
  Searches for symbols matching a query.

  ## Options
    - `:entity_type` - Filter by entity type
    - `:language` - Filter by language
    - `:limit` - Maximum results (default: 50)
  """
  @spec search(String.t(), String.t(), keyword()) :: [symbol()]
  def search(world_id, query, opts \\ []) when is_binary(world_id) and is_binary(query) do
    entity_type = Keyword.get(opts, :entity_type)
    language = Keyword.get(opts, :language)
    limit = Keyword.get(opts, :limit, 50)

    query_lower = String.downcase(query)

    try do
      :ets.tab2list(@ets_qualified)
      |> Enum.filter(fn {{wid, _}, symbol} ->
        wid == world_id and
          matches_query?(symbol, query_lower) and
          (entity_type == nil or symbol.entity_type == entity_type) and
          (language == nil or symbol.language == language)
      end)
      |> Enum.map(fn {_, symbol} -> symbol end)
      |> Enum.take(limit)
    rescue
      ArgumentError -> []
    end
  end

  @doc """
  Adds a relationship between symbols.

  ## Relation types
    - `:calls` - Function A calls function B
    - `:called_by` - Function A is called by function B
    - `:extends` - Class A extends class B
    - `:implements` - Class A implements interface B
    - `:imports` - Module A imports module B
    - `:uses` - Symbol A uses symbol B
  """
  @spec add_relation(String.t(), String.t(), relation_type(), String.t()) :: :ok
  def add_relation(world_id, from_qualified, relation_type, to_qualified) do
    GenServer.cast(__MODULE__, {:add_relation, world_id, from_qualified, relation_type, to_qualified})
  end

  @doc """
  Gets relationships for a symbol.

  ## Examples

      CodeGazetteer.get_relations("world_123", "Billing.calculate_tax", :calls)
      # => ["Tax.compute", "Rate.lookup"]
  """
  @spec get_relations(String.t(), String.t(), relation_type()) :: [String.t()]
  def get_relations(world_id, qualified_name, relation_type) do
    key = {world_id, qualified_name, relation_type}

    try do
      case :ets.lookup(@ets_relations, key) do
        [{^key, targets}] -> targets
        [] -> []
      end
    rescue
      ArgumentError -> []
    end
  end

  @doc """
  Returns all entity types.
  """
  @spec entity_types() :: [String.t()]
  def entity_types, do: @entity_types

  @doc """
  Returns statistics for a world.
  """
  @spec stats(String.t()) :: map()
  def stats(world_id) do
    try do
      case :ets.lookup(@ets_stats, world_id) do
        [{^world_id, stats}] ->
          # Normalize files to a count
          files = Map.get(stats, :files, MapSet.new())
          file_count = if is_struct(files, MapSet), do: MapSet.size(files), else: 0

          # Normalize languages to a count
          languages = Map.get(stats, :languages, MapSet.new())
          language_count = if is_struct(languages, MapSet), do: MapSet.size(languages), else: 0

          %{
            symbols: Map.get(stats, :symbols, 0),
            relations: Map.get(stats, :relations, 0),
            files: file_count,
            languages: language_count,
            file_set: files,
            language_set: languages
          }
        [] -> %{symbols: 0, relations: 0, files: 0, languages: 0, file_set: MapSet.new(), language_set: MapSet.new()}
      end
    rescue
      ArgumentError -> %{symbols: 0, relations: 0, files: 0, languages: 0, file_set: MapSet.new(), language_set: MapSet.new()}
    end
  end

  @doc """
  Returns aggregate statistics across all worlds.
  """
  @spec stats() :: map()
  def stats do
    try do
      all_stats = :ets.tab2list(@ets_stats)

      total_symbols = Enum.reduce(all_stats, 0, fn {_world_id, s}, acc -> acc + Map.get(s, :symbols, 0) end)
      total_relations = Enum.reduce(all_stats, 0, fn {_world_id, s}, acc -> acc + Map.get(s, :relations, 0) end)

      total_files = Enum.reduce(all_stats, 0, fn {_world_id, s}, acc ->
        files = Map.get(s, :files, MapSet.new())
        acc + (if is_struct(files, MapSet), do: MapSet.size(files), else: 0)
      end)

      # Collect all unique languages across worlds
      all_languages = Enum.reduce(all_stats, MapSet.new(), fn {_world_id, s}, acc ->
        languages = Map.get(s, :languages, MapSet.new())
        if is_struct(languages, MapSet), do: MapSet.union(acc, languages), else: acc
      end)

      %{
        symbols: total_symbols,
        relations: total_relations,
        files: total_files,
        languages: MapSet.size(all_languages),
        language_list: MapSet.to_list(all_languages),
        worlds: length(all_stats)
      }
    rescue
      ArgumentError -> %{symbols: 0, relations: 0, files: 0, languages: 0, language_list: [], worlds: 0}
    end
  end

  @doc """
  Clears all symbols for a world.
  """
  @spec clear_world(String.t()) :: :ok
  def clear_world(world_id) do
    GenServer.call(__MODULE__, {:clear_world, world_id})
  end

  @doc """
  Loads language keywords into the gazetteer.
  """
  @spec load_language_keywords(String.t(), atom()) :: {:ok, non_neg_integer()} | {:error, term()}
  def load_language_keywords(world_id, language) do
    GenServer.call(__MODULE__, {:load_keywords, world_id, language})
  end

  @doc """
  Checks if the gazetteer is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    try do
      :ets.info(@ets_symbols) != :undefined
    rescue
      ArgumentError -> false
    end
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    create_tables()
    Logger.info("CodeGazetteer started")
    {:ok, %{initialized: true}}
  end

  @impl true
  def handle_call({:add_symbol, world_id, symbol}, _from, state) do
    result = do_add_symbol(world_id, symbol)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:clear_world, world_id}, _from, state) do
    do_clear_world(world_id)
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:load_keywords, world_id, language}, _from, state) do
    result = do_load_keywords(world_id, language)
    {:reply, result, state}
  end

  @impl true
  def handle_cast({:add_relation, world_id, from, relation_type, to}, state) do
    do_add_relation(world_id, from, relation_type, to)
    {:noreply, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp create_tables do
    tables = [
      {@ets_symbols, [:set, :public, :named_table, read_concurrency: true]},
      {@ets_qualified, [:set, :public, :named_table, read_concurrency: true]},
      {@ets_by_type, [:set, :public, :named_table, read_concurrency: true]},
      {@ets_relations, [:set, :public, :named_table, read_concurrency: true]},
      {@ets_stats, [:set, :public, :named_table, read_concurrency: true]}
    ]

    Enum.each(tables, fn {name, opts} ->
      if :ets.whereis(name) == :undefined do
        :ets.new(name, opts)
      end
    end)
  end

  defp do_add_symbol(world_id, symbol) do
    # Validate required fields
    with {:ok, name} <- get_required(symbol, :name),
         {:ok, entity_type} <- get_required(symbol, :entity_type),
         {:ok, language} <- get_required(symbol, :language) do
      # Build the full symbol
      qualified_name = Map.get(symbol, :qualified_name, name)

      full_symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: entity_type,
        language: language,
        file_path: Map.get(symbol, :file_path),
        line: Map.get(symbol, :line),
        column: Map.get(symbol, :column),
        metadata: Map.get(symbol, :metadata, %{}),
        world_id: world_id,
        inserted_at: DateTime.utc_now()
      }

      # Insert into main table (by name)
      name_key = {world_id, normalize_name(name)}
      existing = case :ets.lookup(@ets_symbols, name_key) do
        [{^name_key, symbols}] when is_list(symbols) -> symbols
        [{^name_key, symbol}] -> [symbol]
        [] -> []
      end
      :ets.insert(@ets_symbols, {name_key, [full_symbol | existing]})

      # Insert into qualified name index
      qualified_key = {world_id, normalize_name(qualified_name)}
      :ets.insert(@ets_qualified, {qualified_key, full_symbol})

      # Insert into type index
      type_key = {world_id, entity_type}
      existing_by_type = case :ets.lookup(@ets_by_type, type_key) do
        [{^type_key, symbols}] -> symbols
        [] -> []
      end
      :ets.insert(@ets_by_type, {type_key, [full_symbol | existing_by_type]})

      # Update stats
      update_stats(world_id, :add)

      # Track file if present
      if file_path = Map.get(symbol, :file_path) do
        update_stats(world_id, {:add_file, file_path})
      end

      # Track language
      update_stats(world_id, {:add_language, language})

      {:ok, qualified_name}
    end
  end

  defp get_required(map, key) do
    case Map.get(map, key) do
      nil -> {:error, {:missing_field, key}}
      value -> {:ok, value}
    end
  end

  defp normalize_name(name) when is_binary(name) do
    String.downcase(name)
  end

  defp matches_query?(symbol, query) do
    name_match = String.contains?(String.downcase(symbol.name), query)
    qualified_match = String.contains?(String.downcase(symbol.qualified_name), query)
    name_match or qualified_match
  end

  defp do_add_relation(world_id, from, relation_type, to) do
    key = {world_id, from, relation_type}

    existing = case :ets.lookup(@ets_relations, key) do
      [{^key, targets}] -> targets
      [] -> []
    end

    unless to in existing do
      :ets.insert(@ets_relations, {key, [to | existing]})
      update_stats(world_id, :add_relation)
    end
  end

  defp do_clear_world(world_id) do
    # Clear from all tables
    [@ets_symbols, @ets_qualified, @ets_by_type, @ets_relations]
    |> Enum.each(fn table ->
      try do
        :ets.tab2list(table)
        |> Enum.filter(fn {{wid, _}, _} -> wid == world_id; {key, _} when is_tuple(key) -> elem(key, 0) == world_id; _ -> false end)
        |> Enum.each(fn {key, _} -> :ets.delete(table, key) end)
      rescue
        ArgumentError -> :ok
      end
    end)

    :ets.delete(@ets_stats, world_id)
  end

  defp update_stats(world_id, action) do
    current = case :ets.lookup(@ets_stats, world_id) do
      [{^world_id, stats}] -> stats
      [] -> %{symbols: 0, relations: 0, files: MapSet.new(), languages: MapSet.new()}
    end

    # Ensure files and languages are MapSets (handle legacy data)
    files = case Map.get(current, :files, MapSet.new()) do
      f when is_struct(f, MapSet) -> f
      _ -> MapSet.new()
    end

    languages = case Map.get(current, :languages, MapSet.new()) do
      l when is_struct(l, MapSet) -> l
      _ -> MapSet.new()
    end

    current = %{current | files: files, languages: languages}

    updated = case action do
      :add ->
        %{current | symbols: Map.get(current, :symbols, 0) + 1}

      :add_relation ->
        %{current | relations: Map.get(current, :relations, 0) + 1}

      {:add_file, path} ->
        %{current | files: MapSet.put(current.files, path)}

      {:add_language, lang} when is_atom(lang) and not is_nil(lang) ->
        %{current | languages: MapSet.put(current.languages, lang)}

      {:add_language, _} ->
        current
    end

    :ets.insert(@ets_stats, {world_id, updated})
  end

  defp do_load_keywords(world_id, language) do
    keywords_path = keywords_file_path(language)

    if File.exists?(keywords_path) do
      case File.read(keywords_path) do
        {:ok, content} ->
          case Jason.decode(content) do
            {:ok, data} ->
              count = load_keywords_from_data(world_id, language, data)
              {:ok, count}

            {:error, reason} ->
              {:error, {:json_parse_failed, reason}}
          end

        {:error, reason} ->
          {:error, {:file_read_failed, reason}}
      end
    else
      # No keywords file yet, that's okay
      {:ok, 0}
    end
  end

  defp keywords_file_path(language) do
    case :code.priv_dir(:brain) do
      {:error, _} ->
        Path.join(["apps", "brain", "priv", "code", "languages", "#{language}.json"])

      priv_dir ->
        Path.join([priv_dir, "code", "languages", "#{language}.json"])
    end
  end

  defp load_keywords_from_data(world_id, language, data) do
    keywords = Map.get(data, "keywords", [])
    builtins = Map.get(data, "builtins", [])
    types = Map.get(data, "types", [])

    count = 0

    count = Enum.reduce(keywords, count, fn keyword, acc ->
      do_add_symbol(world_id, %{
        name: keyword,
        entity_type: "code.keyword",
        language: language,
        metadata: %{source: :language_definition}
      })
      acc + 1
    end)

    count = Enum.reduce(builtins, count, fn builtin, acc ->
      do_add_symbol(world_id, %{
        name: builtin,
        entity_type: "code.function",
        language: language,
        metadata: %{source: :language_definition, builtin: true}
      })
      acc + 1
    end)

    Enum.reduce(types, count, fn type_name, acc ->
      do_add_symbol(world_id, %{
        name: type_name,
        entity_type: "code.type",
        language: language,
        metadata: %{source: :language_definition, builtin: true}
      })
      acc + 1
    end)
  end
end
