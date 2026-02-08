defmodule Brain.Code.QueryHandler do
  @moduledoc "Handles natural language queries about code in the codebase.\n\nThis module provides the bridge between user questions and the CodeGazetteer,\nallowing users to ask questions like:\n- \"What does the process function do?\"\n- \"Who calls Brain.evaluate?\"\n- \"Show me the functions in the Parser module\"\n\n## Usage\n\n    # Explain a symbol\n    {:ok, response} = QueryHandler.explain(\"process\", world_id: \"my_world\")\n\n    # Find usages\n    {:ok, response} = QueryHandler.find_usages(\"evaluate\", world_id: \"my_world\")\n\n    # List symbols\n    {:ok, response} = QueryHandler.list_symbols(\"Brain.Code\", world_id: \"my_world\")\n\n    # Handle any code query based on intent\n    {:ok, response} = QueryHandler.handle(\"code.explain\", entities, world_id: \"my_world\")\n"

  require Logger

  alias Brain.Code.CodeGazetteer
  @default_world_id "default"

  @doc "Handles a code query based on intent and entities.\n\n## Parameters\n  - `intent` - The classified intent (e.g., \"code.explain\", \"code.find_usage\")\n  - `entities` - Extracted entities from the query\n  - `opts` - Options including `:world_id` and `:query_text`\n\n## Returns\n  `{:ok, response}` or `:not_handled`\n"
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
        handle_generic_code_query(entities, world_id, query_text)

      _ ->
        :not_handled
    end
  end

  @doc "Explains a code symbol (function, class, module, etc.).\n"
  @spec explain(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def explain(symbol_name, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    case find_symbol(world_id, symbol_name) do
      {:ok, symbol} ->
        {:ok, generate_explanation(symbol, world_id)}

      :not_found ->
        {:ok,
         "I couldn't find `#{symbol_name}` in the analyzed code. " <>
           "Make sure the codebase has been analyzed with `World.DocumentIngestor.ingest_codebase/2`."}
    end
  end

  @doc "Finds all usages/callers of a symbol.\n"
  @spec find_usages(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def find_usages(symbol_name, opts \\ []) do
    world_id = Keyword.get(opts, :world_id, @default_world_id)

    case find_symbol(world_id, symbol_name) do
      {:ok, symbol} ->
        callers = CodeGazetteer.get_relations(world_id, symbol.qualified_name, :called_by)
        {:ok, format_usages(symbol.qualified_name, callers)}

      :not_found ->
        callers = CodeGazetteer.get_relations(world_id, symbol_name, :called_by)

        if callers != [] do
          {:ok, format_usages(symbol_name, callers)}
        else
          {:ok, "I couldn't find `#{symbol_name}` or any calls to it."}
        end
    end
  end

  @doc "Gets the signature/definition of a symbol.\n"
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

  @doc "Lists symbols matching a pattern or in a module/namespace.\n"
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

  @doc "Searches for symbols by name or description.\n"
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

  @doc "Gets statistics about the analyzed codebase.\n"
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

  defp handle_explain(entities, world_id, query_text) do
    case extract_symbol_name(entities, query_text) do
      nil ->
        {:ok,
         "What code would you like me to explain? Please specify a function, class, or module name."}

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
      symbol_name ->
        explain(symbol_name, world_id: world_id)

      query_text && String.length(query_text) > 3 ->
        search(query_text, world_id: world_id)

      true ->
        stats(world_id: world_id)
    end
  end

  defp find_symbol(world_id, name) do
    case CodeGazetteer.lookup_qualified(world_id, name) do
      {:ok, symbol} ->
        {:ok, symbol}

      :not_found ->
        case CodeGazetteer.lookup(world_id, name) do
          {:ok, [symbol | _]} ->
            {:ok, symbol}

          {:ok, []} ->
            case CodeGazetteer.search(world_id, name, limit: 1) do
              [symbol | _] -> {:ok, symbol}
              [] -> :not_found
            end

          :not_found ->
            :not_found
        end
    end
  end

  defp extract_symbol_name(entities, query_text) do
    symbol =
      Enum.find(entities, fn e ->
        entity_type = e[:entity_type] || e["entity_type"]
        entity_type in ["code.symbol", "symbol", "code.file"]
      end)

    cond do
      symbol ->
        symbol[:value] || symbol["value"]

      entities != [] ->
        entity = List.first(entities)
        entity[:value] || entity["value"]

      query_text ->
        extract_code_pattern(query_text)

      true ->
        nil
    end
  end

  defp extract_code_pattern(text) do
    patterns = [
      ~r/\b([A-Z][a-zA-Z0-9]*(?:\.[A-Z][a-zA-Z0-9]*)*(?:\.[a-z_][a-z0-9_]*)?)\b/,
      ~r/\b([a-z_][a-z0-9_]+)\b(?:\s+function|\s+method)?/,
      ~r/\b([A-Z][a-zA-Z0-9]+)\b(?:\s+class|\s+module)?/
    ]

    Enum.find_value(patterns, fn pattern ->
      case Regex.run(pattern, text) do
        [_, match] when byte_size(match) > 2 -> match
        _ -> nil
      end
    end)
  end

  defp generate_explanation(symbol, world_id) do
    entity_type = symbol.entity_type
    qualified = symbol.qualified_name
    language = symbol.language
    metadata = symbol.metadata || %{}

    type_label = type_to_label(entity_type)

    parts = ["**#{qualified}** is a #{language} #{type_label}"]

    parts =
      if symbol.file_path && symbol.line do
        parts ++ ["Defined in `#{Path.basename(symbol.file_path)}` at line #{symbol.line}"]
      else
        parts
      end

    parts = add_metadata_details(parts, entity_type, metadata)
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

  defp add_metadata_details(parts, _type, _metadata) do
    parts
  end

  defp add_relationship_info(parts, world_id, qualified_name, entity_type) do
    if entity_type in ["code.function", "code.method"] do
      callers = CodeGazetteer.get_relations(world_id, qualified_name, :called_by)

      if callers != [] do
        count = length(callers)
        parts ++ ["It is called by #{count} other function(s)"]
      else
        parts
      end
    else
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
      caller_list = callers |> Enum.map_join(", ", &"`#{&1}`")
      "`#{symbol_name}` is called by: #{caller_list}."
    else
      shown = callers |> Enum.take(5) |> Enum.map_join(", ", &"`#{&1}`")
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

        params =
          if arity > 0 do
            " (#{arity} parameters)"
          else
            ""
          end

        "**#{name}**#{params} - #{visibility} #{language} function"

      "code.method" ->
        arity = Map.get(metadata, :arity, 0)
        visibility = Map.get(metadata, :visibility, :public)

        params =
          if arity > 0 do
            " (#{arity} parameters)"
          else
            ""
          end

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

    header =
      if count == 1 do
        "Found 1 symbol matching `#{pattern}`:"
      else
        "Found #{count} symbols matching `#{pattern}`:"
      end

    items =
      symbols
      |> Enum.take(15)
      |> Enum.map_join(
        "\n",
        fn s ->
          type_icon = type_to_icon(s.entity_type)
          "- #{type_icon} **#{s.qualified_name}** (#{s.language} #{type_to_label(s.entity_type)})"
        end
      )

    more =
      if count > 15 do
        "

_...and #{count - 15} more._"
      else
        ""
      end

    header <> "\n\n" <> items <> more
  end

  defp format_search_results(query, symbols) do
    count = length(symbols)
    header = "Found #{count} result(s) for \"#{query}\":"

    items =
      symbols
      |> Enum.take(10)
      |> Enum.map_join(
        "\n",
        fn s ->
          location =
            if s.file_path do
              " - `#{Path.basename(s.file_path)}`"
            else
              ""
            end

          "- **#{s.qualified_name}** (#{type_to_label(s.entity_type)})#{location}"
        end
      )

    header <> "\n\n" <> items
  end

  defp type_to_label("code.function") do
    "function"
  end

  defp type_to_label("code.class") do
    "class"
  end

  defp type_to_label("code.method") do
    "method"
  end

  defp type_to_label("code.variable") do
    "variable"
  end

  defp type_to_label("code.constant") do
    "constant"
  end

  defp type_to_label("code.type") do
    "type"
  end

  defp type_to_label("code.namespace") do
    "module"
  end

  defp type_to_label("code.interface") do
    "interface"
  end

  defp type_to_label("code.enum") do
    "enum"
  end

  defp type_to_label("code.import") do
    "import"
  end

  defp type_to_label("code.macro") do
    "macro"
  end

  defp type_to_label(type) do
    type |> String.replace("code.", "")
  end

  defp type_to_icon("code.function") do
    "fn"
  end

  defp type_to_icon("code.class") do
    "C"
  end

  defp type_to_icon("code.method") do
    "M"
  end

  defp type_to_icon("code.variable") do
    "v"
  end

  defp type_to_icon("code.constant") do
    "#"
  end

  defp type_to_icon("code.namespace") do
    "N"
  end

  defp type_to_icon(_) do
    "*"
  end
end