defmodule Brain.Code.Summarizer do
  @moduledoc """
  Generates natural language descriptions from code constructs.

  This module analyzes code symbols and their relationships to produce
  human-readable explanations of what code does. It uses:

  - Symbol names and naming conventions
  - Function signatures and parameters
  - Class hierarchies and relationships
  - Import dependencies
  - Call patterns

  ## Usage

      # Summarize a function
      summary = Brain.Code.Summarizer.summarize_function(function_symbol)
      # => "Calculates the tax amount for an order based on the region and subtotal"

      # Summarize a class
      summary = Brain.Code.Summarizer.summarize_class(class_symbol, world_id)
      # => "Manages user authentication, providing login, logout, and session handling"

      # Summarize a file
      summary = Brain.Code.Summarizer.summarize_file(world_id, file_path)
      # => "Defines the billing module with 5 functions for payment processing"
  """

  require Logger

  alias Brain.Code.{CodeGazetteer, RelationMapper}
  alias World.CodeContext

  # Common naming patterns and their meanings
  @verb_patterns [
    {"get", "retrieves"},
    {"fetch", "fetches"},
    {"set", "sets"},
    {"update", "updates"},
    {"delete", "deletes"},
    {"remove", "removes"},
    {"create", "creates"},
    {"add", "adds"},
    {"build", "builds"},
    {"make", "creates"},
    {"init", "initializes"},
    {"start", "starts"},
    {"stop", "stops"},
    {"run", "runs"},
    {"execute", "executes"},
    {"process", "processes"},
    {"handle", "handles"},
    {"validate", "validates"},
    {"check", "checks"},
    {"verify", "verifies"},
    {"parse", "parses"},
    {"format", "formats"},
    {"convert", "converts"},
    {"transform", "transforms"},
    {"calculate", "calculates"},
    {"compute", "computes"},
    {"find", "finds"},
    {"search", "searches"},
    {"filter", "filters"},
    {"sort", "sorts"},
    {"save", "saves"},
    {"load", "loads"},
    {"read", "reads"},
    {"write", "writes"},
    {"send", "sends"},
    {"receive", "receives"},
    {"connect", "connects"},
    {"disconnect", "disconnects"},
    {"render", "renders"},
    {"display", "displays"},
    {"show", "shows"},
    {"hide", "hides"},
    {"enable", "enables"},
    {"disable", "disables"}
  ]

  # Common suffix patterns
  @suffix_meanings %{
    "er" => "that performs",
    "or" => "that performs",
    "able" => "that can be",
    "ible" => "that can be",
    "tion" => "for",
    "sion" => "for",
    "ment" => "for",
    "ness" => "representing",
    "ity" => "representing"
  }

  # Class naming conventions
  @class_patterns [
    {"Controller", "handles HTTP requests for"},
    {"Service", "provides services for"},
    {"Manager", "manages"},
    {"Handler", "handles"},
    {"Factory", "creates"},
    {"Builder", "builds"},
    {"Parser", "parses"},
    {"Validator", "validates"},
    {"Repository", "provides data access for"},
    {"Store", "stores"},
    {"Cache", "caches"},
    {"Client", "connects to"},
    {"Server", "serves"},
    {"Worker", "processes"},
    {"Processor", "processes"},
    {"Provider", "provides"},
    {"Adapter", "adapts"},
    {"Wrapper", "wraps"},
    {"Helper", "assists with"},
    {"Util", "provides utilities for"},
    {"Utils", "provides utilities for"},
    {"Config", "configures"},
    {"Settings", "manages settings for"}
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Generates a natural language summary for a function symbol.

  ## Parameters
    - `symbol` - The function symbol from CodeGazetteer
    - `opts` - Options
      - `:world_id` - For fetching additional context
      - `:include_callers` - Include who calls this function

  ## Returns
    A human-readable description string
  """
  @spec summarize_function(map(), keyword()) :: String.t()
  def summarize_function(symbol, opts \\ []) do
    world_id = Keyword.get(opts, :world_id)
    include_callers = Keyword.get(opts, :include_callers, false)

    name = symbol.name
    qualified_name = symbol.qualified_name
    metadata = symbol.metadata || %{}

    # Analyze the function name
    {verb, object} = parse_function_name(name)

    # Build base description
    base = cond do
      verb && object ->
        "#{verb} #{object}"

      verb ->
        verb

      true ->
        "performs an operation"
    end

    # Add parameter info
    arity = Map.get(metadata, :arity, 0)
    param_desc = case arity do
      0 -> ""
      1 -> " taking one parameter"
      n -> " taking #{n} parameters"
    end

    # Add visibility
    visibility = Map.get(metadata, :visibility, :public)
    vis_desc = if visibility == :private, do: " (private)", else: ""

    # Add caller info if requested
    caller_desc = if include_callers && world_id do
      callers = CodeContext.get_callers(world_id, qualified_name)
      if length(callers) > 0 do
        ", called by #{length(callers)} other functions"
      else
        ""
      end
    else
      ""
    end

    "#{capitalize_first(base)}#{param_desc}#{vis_desc}#{caller_desc}."
  end

  @doc """
  Generates a natural language summary for a class symbol.
  """
  @spec summarize_class(map(), keyword()) :: String.t()
  def summarize_class(symbol, opts \\ []) do
    world_id = Keyword.get(opts, :world_id)

    name = symbol.name
    metadata = symbol.metadata || %{}

    # Analyze class name pattern
    purpose = analyze_class_name(name)

    # Build description parts
    superclass = Map.get(metadata, :superclass)
    
    inheritance_part = if superclass, do: "extending #{superclass}", else: nil
    
    method_part = if world_id do
      methods = CodeContext.search_symbols(world_id, name, entity_type: "code.function", limit: 100)
      method_count = length(methods)
      if method_count > 0, do: "with #{method_count} methods", else: nil
    end

    parts = [purpose, inheritance_part, method_part]
      |> Enum.filter(& &1)

    Enum.join(parts, " ") <> "."
  end

  @doc """
  Generates a summary for a file's contents.
  """
  @spec summarize_file(String.t(), String.t()) :: String.t()
  def summarize_file(world_id, file_path) do
    symbols = CodeContext.list_file_symbols(world_id, file_path)

    if symbols == [] do
      "Empty or unanalyzed file."
    else
      # Count by type
      functions = Enum.count(symbols, &(&1.entity_type == "code.function"))
      classes = Enum.count(symbols, &(&1.entity_type == "code.class"))
      imports = Enum.count(symbols, &(&1.entity_type == "code.import"))

      parts = []

      parts = if classes > 0 do
        class_names = symbols
          |> Enum.filter(&(&1.entity_type == "code.class"))
          |> Enum.map(& &1.name)
          |> Enum.take(3)
          |> Enum.join(", ")

        parts ++ ["Defines #{pluralize(classes, "class", "classes")} (#{class_names})"]
      else
        parts
      end

      parts = if functions > 0 do
        parts ++ ["Contains #{pluralize(functions, "function", "functions")}"]
      else
        parts
      end

      parts = if imports > 0 do
        parts ++ ["Imports #{imports} dependencies"]
      else
        parts
      end

      if parts == [] do
        "Contains #{length(symbols)} symbols."
      else
        Enum.join(parts, ". ") <> "."
      end
    end
  end

  @doc """
  Generates a summary for a module/namespace.
  """
  @spec summarize_module(String.t(), String.t()) :: String.t()
  def summarize_module(world_id, module_name) do
    # Get all symbols in this module
    symbols = CodeContext.search_symbols(world_id, module_name, limit: 100)

    if symbols == [] do
      "Module #{module_name} - no symbols found."
    else
      # Analyze the module
      public_functions = symbols
        |> Enum.filter(&(&1.entity_type == "code.function"))
        |> Enum.filter(fn s ->
          visibility = get_in(s, [:metadata, :visibility])
          visibility != :private
        end)

      # Find the main purpose from function names
      verbs = public_functions
        |> Enum.map(fn s -> s.name end)
        |> Enum.map(&parse_function_name/1)
        |> Enum.map(fn {v, _} -> v end)
        |> Enum.filter(& &1)
        |> Enum.frequencies()
        |> Enum.sort_by(fn {_, count} -> -count end)
        |> Enum.take(3)
        |> Enum.map(fn {verb, _} -> verb end)

      verb_summary = case verbs do
        [] -> "provides various utilities"
        [v] -> "primarily #{v}"
        [v1, v2] -> "#{v1} and #{v2}"
        [v1, v2, v3] -> "#{v1}, #{v2}, and #{v3}"
        _ -> "provides various operations"
      end

      "Module **#{module_name}** #{verb_summary}, with #{length(public_functions)} public functions."
    end
  end

  @doc """
  Generates a summary of a codebase (all symbols in a world).
  """
  @spec summarize_codebase(String.t()) :: String.t()
  def summarize_codebase(world_id) do
    stats = CodeContext.stats(world_id)

    functions = stats[:functions] || 0
    classes = stats[:classes] || 0
    imports = stats[:imports] || 0
    total = stats[:symbols] || 0

    if total == 0 do
      "No code has been analyzed yet."
    else
      base = "Analyzed codebase contains **#{total}** symbols"

      details = []
      details = if functions > 0, do: details ++ ["#{functions} functions"], else: details
      details = if classes > 0, do: details ++ ["#{classes} classes"], else: details
      details = if imports > 0, do: details ++ ["#{imports} imports"], else: details

      if details != [] do
        "#{base} (#{Enum.join(details, ", ")})."
      else
        "#{base}."
      end
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp parse_function_name(name) do
    # Convert from camelCase or snake_case to words
    words = name
      |> String.replace(~r/([a-z])([A-Z])/, "\\1_\\2")
      |> String.downcase()
      |> String.split(~r/[_\s]+/)

    case words do
      [first | rest] ->
        verb = find_verb_meaning(first)
        object = if rest != [], do: Enum.join(rest, " "), else: nil
        {verb, object}

      _ ->
        {nil, nil}
    end
  end

  defp find_verb_meaning(word) do
    Enum.find_value(@verb_patterns, fn {prefix, meaning} ->
      if String.starts_with?(word, prefix), do: meaning
    end) || word
  end

  defp analyze_class_name(name) do
    # Check for common class patterns
    pattern_match = Enum.find_value(@class_patterns, fn {suffix, meaning} ->
      if String.ends_with?(name, suffix) do
        base = String.replace_trailing(name, suffix, "")
        words = camel_to_words(base)
        "#{meaning} #{words}"
      end
    end)

    if pattern_match do
      capitalize_first(pattern_match)
    else
      # Default: try to describe based on the name
      words = camel_to_words(name)
      "Represents #{words}"
    end
  end

  defp camel_to_words(name) do
    name
    |> String.replace(~r/([a-z])([A-Z])/, "\\1 \\2")
    |> String.downcase()
  end

  defp capitalize_first(string) do
    case String.graphemes(string) do
      [first | rest] -> String.upcase(first) <> Enum.join(rest)
      _ -> string
    end
  end

  defp pluralize(1, singular, _plural), do: "1 #{singular}"
  defp pluralize(n, _singular, plural), do: "#{n} #{plural}"
end
