defmodule World.CodeContext do
  @moduledoc """
  World-scoped API for code-specific data.

  This module provides a unified interface for querying code symbols,
  relationships, and analysis results within a training world context.

  ## Features

  - Query functions, classes, and other symbols
  - Find callers and callees of functions
  - Get module dependencies
  - Search symbols by name or pattern
  - Access code analysis results

  ## Usage

      # Get a function from the world
      {:ok, function} = World.CodeContext.get_function(world_id, "calculate_tax")

      # Find all callers of a function
      callers = World.CodeContext.get_callers(world_id, "Billing.calculate_tax")

      # Search for symbols
      results = World.CodeContext.search_symbols(world_id, "user")

      # Get module dependencies
      deps = World.CodeContext.get_dependencies(world_id, "MyApp.Billing")
  """

  require Logger

  alias Brain.Code.{CodeGazetteer, RelationMapper, Pipeline}

  # ============================================================================
  # Public API - Symbol Queries
  # ============================================================================

  @doc """
  Gets a function by name from the world.

  Returns the first matching function, or searches by qualified name
  if multiple matches exist.

  ## Examples

      World.CodeContext.get_function("world_123", "calculate_tax")
      World.CodeContext.get_function("world_123", "Billing.calculate_tax")
  """
  @spec get_function(String.t(), String.t()) :: {:ok, map()} | :not_found
  def get_function(world_id, name) do
    get_symbol_by_type(world_id, name, "code.function")
  end

  @doc """
  Gets a class/module by name from the world.
  """
  @spec get_class(String.t(), String.t()) :: {:ok, map()} | :not_found
  def get_class(world_id, name) do
    get_symbol_by_type(world_id, name, "code.class")
  end

  @doc """
  Gets a type definition by name.
  """
  @spec get_type(String.t(), String.t()) :: {:ok, map()} | :not_found
  def get_type(world_id, name) do
    get_symbol_by_type(world_id, name, "code.type")
  end

  @doc """
  Gets any symbol by name, regardless of type.
  """
  @spec get_symbol(String.t(), String.t()) :: {:ok, map()} | :not_found
  def get_symbol(world_id, name) do
    # Try qualified name first
    case CodeGazetteer.lookup_qualified(world_id, name) do
      {:ok, symbol} -> {:ok, symbol}
      :not_found ->
        # Fall back to simple name lookup
        case CodeGazetteer.lookup(world_id, name) do
          {:ok, [symbol | _]} -> {:ok, symbol}
          {:ok, symbol} when is_map(symbol) -> {:ok, symbol}
          :not_found -> :not_found
        end
    end
  end

  @doc """
  Lists all functions in the world.

  ## Options
    - `:limit` - Maximum results (default: 100)
    - `:language` - Filter by language
  """
  @spec list_functions(String.t(), keyword()) :: [map()]
  def list_functions(world_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 100)
    language = Keyword.get(opts, :language)

    CodeGazetteer.list_by_type(world_id, "code.function")
    |> maybe_filter_by_language(language)
    |> Enum.take(limit)
  end

  @doc """
  Lists all classes in the world.
  """
  @spec list_classes(String.t(), keyword()) :: [map()]
  def list_classes(world_id, opts \\ []) do
    limit = Keyword.get(opts, :limit, 100)
    language = Keyword.get(opts, :language)

    CodeGazetteer.list_by_type(world_id, "code.class")
    |> maybe_filter_by_language(language)
    |> Enum.take(limit)
  end

  @doc """
  Lists all symbols in a file.
  """
  @spec list_file_symbols(String.t(), String.t()) :: [map()]
  def list_file_symbols(world_id, file_path) do
    CodeGazetteer.list_by_file(world_id, file_path)
  end

  @doc """
  Searches for symbols matching a query.

  ## Options
    - `:entity_type` - Filter by type (e.g., "code.function")
    - `:language` - Filter by language
    - `:limit` - Maximum results (default: 50)
  """
  @spec search_symbols(String.t(), String.t(), keyword()) :: [map()]
  def search_symbols(world_id, query, opts \\ []) do
    CodeGazetteer.search(world_id, query, opts)
  end

  # ============================================================================
  # Public API - Relationship Queries
  # ============================================================================

  @doc """
  Gets all functions that call the specified function.

  ## Examples

      World.CodeContext.get_callers("world_123", "Billing.calculate_tax")
      # => ["Order.process", "Invoice.generate"]
  """
  @spec get_callers(String.t(), String.t()) :: [String.t()]
  def get_callers(world_id, function_name) do
    RelationMapper.find_callers(world_id, function_name)
  end

  @doc """
  Gets all functions called by the specified function.

  ## Examples

      World.CodeContext.get_callees("world_123", "Order.process")
      # => ["Billing.calculate_tax", "Inventory.check"]
  """
  @spec get_callees(String.t(), String.t()) :: [String.t()]
  def get_callees(world_id, function_name) do
    RelationMapper.find_callees(world_id, function_name)
  end

  @doc """
  Gets the inheritance hierarchy for a class.

  Returns a list of ancestor classes, from immediate parent to root.
  """
  @spec get_ancestors(String.t(), String.t()) :: [String.t()]
  def get_ancestors(world_id, class_name) do
    RelationMapper.find_ancestors(world_id, class_name)
  end

  @doc """
  Gets all classes that extend the specified class.
  """
  @spec get_descendants(String.t(), String.t()) :: [String.t()]
  def get_descendants(world_id, class_name) do
    RelationMapper.find_descendants(world_id, class_name)
  end

  @doc """
  Gets the dependencies (imports) for a module/file.

  ## Examples

      World.CodeContext.get_dependencies("world_123", "lib/billing.ex")
      # => ["Decimal", "MyApp.Tax", "MyApp.Rate"]
  """
  @spec get_dependencies(String.t(), String.t()) :: [String.t()]
  def get_dependencies(world_id, module_or_file) do
    CodeGazetteer.get_relations(world_id, module_or_file, :imports)
  end

  @doc """
  Gets modules that depend on (import) the specified module.
  """
  @spec get_dependents(String.t(), String.t()) :: [String.t()]
  def get_dependents(world_id, module_name) do
    # Find all imports that reference this module
    CodeGazetteer.list_by_type(world_id, "code.import")
    |> Enum.filter(fn import -> import.name == module_name end)
    |> Enum.map(& &1.file_path)
    |> Enum.uniq()
  end

  @doc """
  Builds a complete dependency graph for the world.

  Returns a map of module/file -> [dependencies].
  """
  @spec build_dependency_graph(String.t()) :: map()
  def build_dependency_graph(world_id) do
    RelationMapper.build_dependency_graph(world_id)
  end

  # ============================================================================
  # Public API - Analysis
  # ============================================================================

  @doc """
  Analyzes source code and stores results in the world.

  This is a convenience wrapper around Brain.Code.Pipeline.process/3.
  """
  @spec analyze_code(String.t(), String.t(), atom(), keyword()) :: {:ok, map()} | {:error, term()}
  def analyze_code(world_id, source_code, language, opts \\ []) do
    opts = Keyword.merge(opts, [world_id: world_id, store: true])
    Pipeline.process(source_code, language, opts)
  end

  @doc """
  Analyzes a source file and stores results in the world.
  """
  @spec analyze_file(String.t(), String.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def analyze_file(world_id, file_path, opts \\ []) do
    opts = Keyword.merge(opts, [world_id: world_id, store: true])
    Pipeline.process_file(file_path, opts)
  end

  @doc """
  Analyzes a directory of source files.
  """
  @spec analyze_directory(String.t(), String.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def analyze_directory(world_id, dir_path, opts \\ []) do
    opts = Keyword.merge(opts, [world_id: world_id, store: true])
    Pipeline.process_directory(dir_path, opts)
  end

  # ============================================================================
  # Public API - Statistics
  # ============================================================================

  @doc """
  Gets code analysis statistics for the world.
  """
  @spec stats(String.t()) :: map()
  def stats(world_id) do
    base_stats = CodeGazetteer.stats(world_id)

    # Count by entity type
    type_counts = CodeGazetteer.entity_types()
      |> Enum.map(fn type ->
        {type, length(CodeGazetteer.list_by_type(world_id, type))}
      end)
      |> Enum.into(%{})

    Map.merge(base_stats, %{
      by_type: type_counts,
      functions: Map.get(type_counts, "code.function", 0),
      classes: Map.get(type_counts, "code.class", 0),
      imports: Map.get(type_counts, "code.import", 0)
    })
  end

  @doc """
  Clears all code analysis data for a world.
  """
  @spec clear(String.t()) :: :ok
  def clear(world_id) do
    CodeGazetteer.clear_world(world_id)
  end

  # ============================================================================
  # Public API - Language Keywords
  # ============================================================================

  @doc """
  Loads language keywords and builtins for a language.

  This populates the world's code gazetteer with language-specific
  keywords and built-in functions.
  """
  @spec load_language_keywords(String.t(), atom()) :: {:ok, non_neg_integer()} | {:error, term()}
  def load_language_keywords(world_id, language) do
    CodeGazetteer.load_language_keywords(world_id, language)
  end

  @doc """
  Checks if a name is a language keyword.
  """
  @spec is_keyword?(String.t(), String.t()) :: boolean()
  def is_keyword?(world_id, name) do
    case CodeGazetteer.lookup(world_id, name) do
      {:ok, symbols} when is_list(symbols) ->
        Enum.any?(symbols, fn s -> s.entity_type == "code.keyword" end)

      {:ok, symbol} when is_map(symbol) ->
        symbol.entity_type == "code.keyword"

      :not_found ->
        false
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_symbol_by_type(world_id, name, entity_type) do
    # Try qualified name first
    case CodeGazetteer.lookup_qualified(world_id, name) do
      {:ok, symbol} when symbol.entity_type == entity_type ->
        {:ok, symbol}

      _ ->
        # Fall back to simple name lookup
        case CodeGazetteer.lookup(world_id, name) do
          {:ok, symbols} when is_list(symbols) ->
            case Enum.find(symbols, fn s -> s.entity_type == entity_type end) do
              nil -> :not_found
              symbol -> {:ok, symbol}
            end

          {:ok, symbol} when is_map(symbol) and symbol.entity_type == entity_type ->
            {:ok, symbol}

          _ ->
            :not_found
        end
    end
  end

  defp maybe_filter_by_language(symbols, nil), do: symbols
  defp maybe_filter_by_language(symbols, language) do
    Enum.filter(symbols, fn s -> s.language == language end)
  end
end
