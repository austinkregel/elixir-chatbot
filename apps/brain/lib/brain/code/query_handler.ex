defmodule Brain.Code.QueryHandler do
  @moduledoc """
  Handles natural language queries about code in the codebase.

  This module provides the bridge between user questions and the CodeGazetteer,
  allowing users to ask questions like:
  - "What does the process function do?"
  - "Who calls Brain.evaluate?"
  - "Show me the functions in the Parser module"

  ## Usage

      # Explain a symbol
      {:ok, response} = QueryHandler.explain("process", world_id: "my_world")

      # Find usages
      {:ok, response} = QueryHandler.find_usages("evaluate", world_id: "my_world")

      # List symbols
      {:ok, response} = QueryHandler.list_symbols("Brain.Code", world_id: "my_world")

      # Handle any code query based on intent
      {:ok, response} = QueryHandler.handle("code.explain", entities, world_id: "my_world")
  """

  require Logger

  alias Brain.Code.CodeGazetteer

  # Default world for code queries when none specified
  @default_world_id "default"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Handles a code query based on intent and entities.

  ## Parameters
    - `intent` - The classified intent (e.g., "code.explain", "code.find_usage")
    - `entities` - Extracted entities from the query
    - `opts` - Options including `:world_id` and `:query_text`

  ## Returns
    `{:ok, response}` or `:not_handled`
  """
  @spec handle(String.t(), list(), keyword()) :: {:ok, String.t()} | :not_handled
  def handle(intent, entities, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    query_text = Keyword.get(opts, :query_text)

    case intent do
      "code.explain" ->
        handle_explain(entities, world_id, query_text)

      "code.find_usage" ->
        handle_find_usage(entities, world_id)

      "code.signature" ->
        handle_signature(entities, world_id)

      "code.list" ->
        handle_list(entities, world_id)

      "code.search" ->
        handle_search(entities, world_id, query_text)

      "code" <> _ ->
        # Generic code intent - try to understand from context
        handle_generic_code_query(entities, world_id, query_text)

      _ ->
        :not_handled
    end
  end

  @doc """
  Explains a code symbol (function, class, module, etc.).
  """
  @spec explain(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def explain(symbol_name, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    case find_symbol(world_id, symbol_name) do
      {:ok, symbol} ->
        {:ok, generate_explanation(symbol, world_id)}

      :not_found ->
        {:ok, "I couldn't find `#{symbol_name}` in the analyzed code. " <>
              "Make sure the codebase has been analyzed with `World.DocumentIngestor.ingest_codebase/2`."}
    end
  end

  @doc """
  Finds all usages/callers of a symbol.
  """
  @spec find_usages(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def find_usages(symbol_name, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    case find_symbol(world_id, symbol_name) do
      {:ok, symbol} ->
        callers = CodeGazetteer.get_relations(world_id, symbol.qualified_name, :called_by)
        {:ok, format_usages(symbol.qualified_name, callers)}

      :not_found ->
        # Try direct relation lookup anyway
        callers = CodeGazetteer.get_relations(world_id, symbol_name, :called_by)
        if callers != [] do
          {:ok, format_usages(symbol_name, callers)}
        else
          {:ok, "I couldn't find `#{symbol_name}` or any calls to it."}
        end
    end
  end

  @doc """
  Gets the signature/definition of a symbol.
  """
  @spec signature(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def signature(symbol_name, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    case find_symbol(world_id, symbol_name) do
      {:ok, symbol} ->
        {:ok, generate_signature(symbol)}

      :not_found ->
        {:ok, "I couldn't find the definition of `#{symbol_name}`."}
    end
  end

  @doc """
  Lists symbols matching a pattern or in a module/namespace.
  """
  @spec list_symbols(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def list_symbols(pattern, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    limit = Keyword.get(opts, :limit, 20)

    symbols = CodeGazetteer.search(world_id, pattern, limit: limit)

    if symbols == [] do
      {:ok, "No symbols matching `#{pattern}` were found."}
    else
      {:ok, format_symbol_list(pattern, symbols)}
    end
  end

  @doc """
  Searches for symbols by name or description.
  """
  @spec search(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def search(query, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    limit = Keyword.get(opts, :limit, 10)

    symbols = CodeGazetteer.search(world_id, query, limit: limit)

    if symbols == [] do
      {:ok, "No code matching \"#{query}\" was found."}
    else
      {:ok, format_search_results(query, symbols)}
    end
  end

  @doc """
  Gets statistics about the analyzed codebase.
  """
  @spec stats(keyword()) :: {:ok, String.t()} | {:error, term()}
  def stats(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    stats = CodeGazetteer.stats(world_id)

    response = """
    **Codebase Statistics:**
    - **Symbols:** #{Map.get(stats, :symbols, 0)}
    - **Relations:** #{Map.get(stats, :relations, 0)}
    - **Files:** #{Map.get(stats, :files, 0)}
    - **Languages:** #{Map.get(stats, :languages, 0)}
    """

    {:ok, String.trim(response)}
  end

  # ============================================================================
  # Intent Handlers
  # ============================================================================

  defp handle_explain(entities, world_id, query_text) do
    case extract_symbol_name(entities, query_text) do
      nil ->
        {:ok, "What code would you like me to explain? Please specify a function, class, or module name."}

      symbol_name ->
        explain(symbol_name, world_id: world_id)
    end
  end

  defp handle_find_usage(entities, world_id) do
    case extract_symbol_name(entities, nil) do
      nil ->
        {:ok, "What symbol would you like me to find usages for?"}

      symbol_name ->
        find_usages(symbol_name, world_id: world_id)
    end
  end

  defp handle_signature(entities, world_id) do
    case extract_symbol_name(entities, nil) do
      nil ->
        {:ok, "What function or method would you like me to describe?"}

      symbol_name ->
        signature(symbol_name, world_id: world_id)
    end
  end

  defp handle_list(entities, world_id) do
    case extract_symbol_name(entities, nil) do
      nil ->
        # Return general stats
        stats(world_id: world_id)

      pattern ->
        list_symbols(pattern, world_id: world_id)
    end
  end

  defp handle_search(entities, world_id, query_text) do
    query = extract_symbol_name(entities, nil) || query_text || ""
    search(query, world_id: world_id)
  end

  defp handle_generic_code_query(entities, world_id, query_text) do
    symbol_name = extract_symbol_name(entities, query_text)

    cond do
      # If we have a symbol, try to explain it
      symbol_name ->
        explain(symbol_name, world_id: world_id)

      # If we have query text, try to search
      query_text && String.length(query_text) > 3 ->
        search(query_text, world_id: world_id)

      true ->
        stats(world_id: world_id)
    end
  end

  # ============================================================================
  # Symbol Finding
  # ============================================================================

  defp find_symbol(world_id, name) do
    # Try exact qualified name match first
    case CodeGazetteer.lookup_qualified(world_id, name) do
      {:ok, symbol} ->
        {:ok, symbol}

      :not_found ->
        # Try simple name lookup
        case CodeGazetteer.lookup(world_id, name) do
          {:ok, [symbol | _]} ->
            {:ok, symbol}

          {:ok, []} ->
            # Try search as fallback
            case CodeGazetteer.search(world_id, name, limit: 1) do
              [symbol | _] -> {:ok, symbol}
              [] -> :not_found
            end

          :not_found ->
            :not_found
        end
    end
  end

  # ============================================================================
  # Entity Extraction
  # ============================================================================

  defp extract_symbol_name(entities, query_text) do
    # Look for explicit code.symbol entity
    symbol = Enum.find(entities, fn e ->
      entity_type = e[:entity_type] || e["entity_type"]
      entity_type in ["code.symbol", "symbol", "code.file"]
    end)

    cond do
      symbol ->
        symbol[:value] || symbol["value"]

      # Try to extract from any entity
      entities != [] ->
        entity = List.first(entities)
        entity[:value] || entity["value"]

      # Try to extract code-like patterns from query
      query_text ->
        extract_code_pattern(query_text)

      true ->
        nil
    end
  end

  defp extract_code_pattern(text) do
    # Look for patterns that look like code symbols
    patterns = [
      # Module.function pattern (e.g., Brain.evaluate)
      ~r/\b([A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)*(?:\.[a-z_][a-z0-9_]*)?)\b/,
      # snake_case function names
      ~r/\b([a-z_][a-z0-9_]+)\b(?:\s+function|\s+method)?/,
      # CamelCase class names
      ~r/\b([A-Z][a-zA-Z0-9]+)\b(?:\s+class|\s+module)?/
    ]

    Enum.find_value(patterns, fn pattern ->
      case Regex.run(pattern, text) do
        [_, match] when byte_size(match) > 2 -> match
        _ -> nil
      end
    end)
  end

  # ============================================================================
  # Response Formatting
  # ============================================================================

  defp generate_explanation(symbol, world_id) do
    entity_type = symbol.entity_type
    qualified = symbol.qualified_name
    language = symbol.language
    metadata = symbol.metadata || %{}

    type_label = type_to_label(entity_type)

    parts = ["**#{qualified}** is a #{language} #{type_label}"]

    # Add location
    parts = if symbol.file_path && symbol.line do
      parts ++ ["Defined in `#{Path.basename(symbol.file_path)}` at line #{symbol.line}"]
    else
      parts
    end

    # Add metadata details
    parts = add_metadata_details(parts, entity_type, metadata)

    # Add relationship info
    parts = add_relationship_info(parts, world_id, symbol.qualified_name, entity_type)

    Enum.join(parts, ". ") <> "."
  end

  defp add_metadata_details(parts, "code.function", metadata) do
    arity = Map.get(metadata, :arity, "unknown")
    visibility = Map.get(metadata, :visibility, :public)
    parts ++ ["It takes #{arity} parameter(s) and has #{visibility} visibility"]
  end

  defp add_metadata_details(parts, "code.class", metadata) do
    if superclass = Map.get(metadata, :superclass) do
      parts ++ ["It extends `#{superclass}`"]
    else
      parts
    end
  end

  defp add_metadata_details(parts, _type, _metadata), do: parts

  defp add_relationship_info(parts, world_id, qualified_name, entity_type) do
    # Get callers for functions
    if entity_type in ["code.function", "code.method"] do
      callers = CodeGazetteer.get_relations(world_id, qualified_name, :called_by)
      if callers != [] do
        count = length(callers)
        parts ++ ["It is called by #{count} other function(s)"]
      else
        parts
      end
    else
      # Get what the symbol calls
      calls = CodeGazetteer.get_relations(world_id, qualified_name, :calls)
      if calls != [] do
        count = length(calls)
        parts ++ ["It calls #{count} other function(s)"]
      else
        parts
      end
    end
  end

  defp format_usages(symbol_name, callers) when callers == [] do
    "`#{symbol_name}` doesn't appear to be called from anywhere in the analyzed code."
  end

  defp format_usages(symbol_name, callers) do
    count = length(callers)

    if count <= 5 do
      caller_list = callers |> Enum.map(&"`#{&1}`") |> Enum.join(", ")
      "`#{symbol_name}` is called by: #{caller_list}."
    else
      shown = callers |> Enum.take(5) |> Enum.map(&"`#{&1}`") |> Enum.join(", ")
      "`#{symbol_name}` is called #{count} times, including: #{shown}, and #{count - 5} more."
    end
  end

  defp generate_signature(symbol) do
    name = symbol.qualified_name
    entity_type = symbol.entity_type
    metadata = symbol.metadata || %{}
    language = symbol.language

    case entity_type do
      "code.function" ->
        arity = Map.get(metadata, :arity, 0)
        visibility = Map.get(metadata, :visibility, :public)
        params = if arity > 0, do: " (#{arity} parameters)", else: ""
        "**#{name}**#{params} - #{visibility} #{language} function"

      "code.method" ->
        arity = Map.get(metadata, :arity, 0)
        visibility = Map.get(metadata, :visibility, :public)
        params = if arity > 0, do: " (#{arity} parameters)", else: ""
        "**#{name}**#{params} - #{visibility} method"

      "code.class" ->
        superclass = Map.get(metadata, :superclass)
        if superclass do
          "**#{name}** - #{language} class extending `#{superclass}`"
        else
          "**#{name}** - #{language} class"
        end

      "code.namespace" ->
        "**#{name}** - #{language} module/namespace"

      _ ->
        "**#{name}** - #{type_to_label(entity_type)}"
    end
  end

  defp format_symbol_list(pattern, symbols) do
    count = length(symbols)
    header = if count == 1 do
      "Found 1 symbol matching `#{pattern}`:"
    else
      "Found #{count} symbols matching `#{pattern}`:"
    end

    items = symbols
    |> Enum.take(15)
    |> Enum.map(fn s ->
      type_icon = type_to_icon(s.entity_type)
      "- #{type_icon} **#{s.qualified_name}** (#{s.language} #{type_to_label(s.entity_type)})"
    end)
    |> Enum.join("\n")

    more = if count > 15, do: "\n\n_...and #{count - 15} more._", else: ""

    header <> "\n\n" <> items <> more
  end

  defp format_search_results(query, symbols) do
    count = length(symbols)
    header = "Found #{count} result(s) for \"#{query}\":"

    items = symbols
    |> Enum.take(10)
    |> Enum.map(fn s ->
      location = if s.file_path do
        " - `#{Path.basename(s.file_path)}`"
      else
        ""
      end
      "- **#{s.qualified_name}** (#{type_to_label(s.entity_type)})#{location}"
    end)
    |> Enum.join("\n")

    header <> "\n\n" <> items
  end

  defp type_to_label("code.function"), do: "function"
  defp type_to_label("code.class"), do: "class"
  defp type_to_label("code.method"), do: "method"
  defp type_to_label("code.variable"), do: "variable"
  defp type_to_label("code.constant"), do: "constant"
  defp type_to_label("code.type"), do: "type"
  defp type_to_label("code.namespace"), do: "module"
  defp type_to_label("code.interface"), do: "interface"
  defp type_to_label("code.enum"), do: "enum"
  defp type_to_label("code.import"), do: "import"
  defp type_to_label("code.macro"), do: "macro"
  defp type_to_label(type), do: type |> String.replace("code.", "")

  defp type_to_icon("code.function"), do: "fn"
  defp type_to_icon("code.class"), do: "C"
  defp type_to_icon("code.method"), do: "M"
  defp type_to_icon("code.variable"), do: "v"
  defp type_to_icon("code.constant"), do: "#"
  defp type_to_icon("code.namespace"), do: "N"
  defp type_to_icon(_), do: "*"
end
