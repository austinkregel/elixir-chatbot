defmodule Brain.Code.RelationMapper do
  @moduledoc """
  Maps relationships between code symbols.

  This module analyzes ASTs to identify relationships between symbols:
  - Function calls (who calls whom)
  - Inheritance and composition
  - Module dependencies
  - Variable usage (definition to references)

  ## Relationship Types

  | Type | Description | Example |
  |------|-------------|---------|
  | `:calls` | Function A calls function B | `foo()` calling `bar()` |
  | `:called_by` | Inverse of calls | `bar()` called by `foo()` |
  | `:extends` | Class inheritance | `class B extends A` |
  | `:implements` | Interface implementation | `class B implements I` |
  | `:imports` | Module import | `import os` |
  | `:uses` | General usage | Variable reference |
  | `:contains` | Scope containment | Class contains method |
  | `:instantiates` | Object creation | `new MyClass()` |

  ## Architecture

  The mapper works in two passes:
  1. **Definition Pass**: Collects all symbol definitions
  2. **Reference Pass**: Finds references to known symbols
  """

  require Logger

  alias Brain.Code.CodeGazetteer

  @type relation :: %{
          from: String.t(),
          to: String.t(),
          type: atom(),
          location: {non_neg_integer(), non_neg_integer()} | nil
        }

  @type mapping_result :: %{
          relations: [relation()],
          unresolved: [String.t()],
          stats: map()
        }

  # Call expression patterns per language
  @call_patterns %{
    elixir: ["call", "function_call", "remote_call"],
    python: ["call", "call_expression"],
    ruby: ["call", "method_call"],
    go: ["call_expression"],
    java: ["method_invocation"],
    c: ["call_expression"],
    cpp: ["call_expression"],
    csharp: ["invocation_expression"],
    php: ["function_call_expression", "member_call_expression"]
  }

  # Inheritance patterns
  @extends_patterns %{
    elixir: [],  # Elixir uses behaviours, not inheritance
    python: ["argument_list"],  # class Foo(Bar)
    ruby: ["superclass"],
    go: [],  # Go uses composition
    java: ["superclass", "extends"],
    cpp: ["base_class_clause"],
    csharp: ["base_list"],
    php: ["base_clause"]
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Maps all relationships in an AST.

  ## Parameters
    - `ast` - The parsed AST
    - `symbols` - List of known symbols (from SymbolExtractor)
    - `language` - The programming language
    - `opts` - Options

  ## Options
    - `:world_id` - World ID for storing relations
    - `:store` - Whether to store in CodeGazetteer (default: false)
    - `:resolve_external` - Try to resolve calls to external modules (default: false)

  ## Returns
    A map with `:relations`, `:unresolved`, and `:stats`
  """
  @spec map_relations(map(), [map()], atom(), keyword()) :: mapping_result()
  def map_relations(ast, symbols, language, opts \\ []) do
    world_id = Keyword.get(opts, :world_id)
    store = Keyword.get(opts, :store, false)

    # Build symbol lookup table
    symbol_table = build_symbol_table(symbols)

    # Initialize context
    context = %{
      language: language,
      symbol_table: symbol_table,
      current_scope: [],
      relations: [],
      unresolved: [],
      stats: %{calls: 0, extends: 0, implements: 0, imports: 0, uses: 0}
    }

    # Walk AST for relationships
    result = walk_for_relations(ast, context)

    # Store if requested
    if store and world_id do
      store_relations(world_id, result.relations)
    end

    %{
      relations: Enum.reverse(result.relations),
      unresolved: Enum.uniq(Enum.reverse(result.unresolved)),
      stats: result.stats
    }
  end

  @doc """
  Finds all callers of a function.

  ## Examples

      RelationMapper.find_callers("world_123", "Billing.calculate_tax")
      # => ["Order.process", "Invoice.generate"]
  """
  @spec find_callers(String.t(), String.t()) :: [String.t()]
  def find_callers(world_id, function_name) do
    CodeGazetteer.get_relations(world_id, function_name, :called_by)
  end

  @doc """
  Finds all functions called by a function.

  ## Examples

      RelationMapper.find_callees("world_123", "Order.process")
      # => ["Billing.calculate_tax", "Inventory.check"]
  """
  @spec find_callees(String.t(), String.t()) :: [String.t()]
  def find_callees(world_id, function_name) do
    CodeGazetteer.get_relations(world_id, function_name, :calls)
  end

  @doc """
  Finds the inheritance hierarchy for a class.

  Returns a list of ancestor classes.
  """
  @spec find_ancestors(String.t(), String.t()) :: [String.t()]
  def find_ancestors(world_id, class_name) do
    find_ancestors_recursive(world_id, class_name, [])
  end

  @doc """
  Finds all classes that extend a given class.
  """
  @spec find_descendants(String.t(), String.t()) :: [String.t()]
  def find_descendants(world_id, class_name) do
    # This requires searching for all classes that extend this one
    CodeGazetteer.search(world_id, class_name, entity_type: "code.class")
    |> Enum.filter(fn symbol ->
      superclass = get_in(symbol, [:metadata, :superclass])
      superclass == class_name
    end)
    |> Enum.map(& &1.qualified_name)
  end

  @doc """
  Builds a dependency graph for a module.

  Returns a map of module -> [dependencies].
  """
  @spec build_dependency_graph(String.t()) :: map()
  def build_dependency_graph(world_id) do
    # Get all import relations
    imports = CodeGazetteer.list_by_type(world_id, "code.import")

    imports
    |> Enum.group_by(&Map.get(&1, :file_path))
    |> Enum.map(fn {file, import_symbols} ->
      {file, Enum.map(import_symbols, & &1.name)}
    end)
    |> Enum.into(%{})
  end

  # ============================================================================
  # Private Functions - AST Walking
  # ============================================================================

  defp walk_for_relations(node, context) when is_map(node) do
    node_type = Map.get(node, :type, "")

    # Check for call expressions
    context = if matches_pattern?(node_type, get_patterns(@call_patterns, context.language)) do
      extract_call_relation(node, context)
    else
      context
    end

    # Check for extends/inheritance
    context = if matches_pattern?(node_type, get_patterns(@extends_patterns, context.language)) do
      extract_extends_relation(node, context)
    else
      context
    end

    # Check for new/instantiation
    context = if String.contains?(node_type, "new") or String.contains?(node_type, "object_creation") do
      extract_instantiation_relation(node, context)
    else
      context
    end

    # Update scope if entering a class/function
    context = maybe_push_scope(node, node_type, context)

    # Recurse into children
    children = Map.get(node, :children, [])

    result = Enum.reduce(children, context, fn child, ctx ->
      walk_for_relations(child, ctx)
    end)

    # Pop scope if we pushed one
    maybe_pop_scope(node, node_type, result, context)
  end

  defp walk_for_relations(_, context), do: context

  defp get_patterns(patterns_map, language) do
    Map.get(patterns_map, language, [])
  end

  defp matches_pattern?(_node_type, []), do: false

  defp matches_pattern?(node_type, patterns) do
    Enum.any?(patterns, fn pattern ->
      String.contains?(node_type, pattern)
    end)
  end

  # ============================================================================
  # Relation Extraction
  # ============================================================================

  defp extract_call_relation(node, context) do
    # Find the function being called
    callee_name = find_callee_name(node, context.language)

    if callee_name && callee_name != "" do
      # Get current scope (caller)
      caller = current_scope_name(context)

      # Check if callee is in our symbol table
      resolved = resolve_symbol(callee_name, context.symbol_table, context)

      if resolved do
        relation = %{
          from: caller,
          to: resolved,
          type: :calls,
          location: get_location(node)
        }

        # Also add inverse relation
        inverse = %{
          from: resolved,
          to: caller,
          type: :called_by,
          location: get_location(node)
        }

        stats = Map.update(context.stats, :calls, 1, &(&1 + 1))
        %{context | relations: [inverse, relation | context.relations], stats: stats}
      else
        # Unresolved call
        %{context | unresolved: [callee_name | context.unresolved]}
      end
    else
      context
    end
  end

  defp extract_extends_relation(node, context) do
    # Find the superclass
    superclass_name = find_superclass_name(node, context.language)

    if superclass_name && superclass_name != "" do
      # Current class (the one extending)
      current_class = current_scope_name(context)

      relation = %{
        from: current_class,
        to: superclass_name,
        type: :extends,
        location: get_location(node)
      }

      stats = Map.update(context.stats, :extends, 1, &(&1 + 1))
      %{context | relations: [relation | context.relations], stats: stats}
    else
      context
    end
  end

  defp extract_instantiation_relation(node, context) do
    # Find the class being instantiated
    class_name = find_instantiated_class(node, context.language)

    if class_name && class_name != "" do
      caller = current_scope_name(context)

      relation = %{
        from: caller,
        to: class_name,
        type: :instantiates,
        location: get_location(node)
      }

      %{context | relations: [relation | context.relations]}
    else
      context
    end
  end

  # ============================================================================
  # Name Finding Helpers
  # ============================================================================

  defp find_callee_name(node, language) do
    children = Map.get(node, :children, [])

    # Look for identifier or member access
    callee = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "identifier") or
        String.contains?(type, "member") or
        String.contains?(type, "attribute") or
        String.contains?(type, "field")
    end)

    if callee do
      Map.get(callee, :text, "")
    else
      # Try to extract from node text
      text = Map.get(node, :text, "")
      extract_function_name_from_call(text, language)
    end
  end

  defp extract_function_name_from_call(text, _language) do
    # Simple extraction - get the function name before parenthesis
    case String.split(text, "(", parts: 2) do
      [name | _] -> String.trim(name) |> String.split(".") |> List.last()
      _ -> nil
    end
  end

  defp find_superclass_name(node, _language) do
    children = Map.get(node, :children, [])

    # Look for identifier in superclass/extends node
    super_node = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "identifier") or String.contains?(type, "type")
    end)

    if super_node do
      Map.get(super_node, :text, "")
    end
  end

  defp find_instantiated_class(node, _language) do
    children = Map.get(node, :children, [])

    # Look for type identifier after "new"
    type_node = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "type") or String.contains?(type, "identifier")
    end)

    if type_node do
      Map.get(type_node, :text, "")
    end
  end

  # ============================================================================
  # Symbol Resolution
  # ============================================================================

  defp build_symbol_table(symbols) do
    symbols
    |> Enum.flat_map(fn symbol ->
      name = symbol.name
      qualified = symbol.qualified_name

      [{String.downcase(name), qualified}, {String.downcase(qualified), qualified}]
    end)
    |> Enum.into(%{})
  end

  defp resolve_symbol(name, symbol_table, context) do
    normalized = String.downcase(name)

    # Try exact match first
    case Map.get(symbol_table, normalized) do
      nil ->
        # Try with current scope prefix
        scoped_name = build_scoped_name(name, context.current_scope)
        Map.get(symbol_table, String.downcase(scoped_name))

      resolved ->
        resolved
    end
  end

  defp build_scoped_name(name, []), do: name
  defp build_scoped_name(name, scope) do
    Enum.join(Enum.reverse(scope) ++ [name], ".")
  end

  # ============================================================================
  # Scope Management
  # ============================================================================

  defp current_scope_name(context) do
    case context.current_scope do
      [] -> "global"
      scope -> Enum.join(Enum.reverse(scope), ".")
    end
  end

  defp maybe_push_scope(node, node_type, context) do
    # Check if this is a scope-creating node (class, function, module)
    if is_scope_node?(node_type) do
      name = find_node_name(node)
      if name do
        %{context | current_scope: [name | context.current_scope]}
      else
        context
      end
    else
      context
    end
  end

  defp maybe_pop_scope(_node, node_type, current_context, original_context) do
    if is_scope_node?(node_type) do
      # Restore original scope
      %{current_context | current_scope: original_context.current_scope}
    else
      current_context
    end
  end

  defp is_scope_node?(node_type) do
    String.contains?(node_type, "function") or
      String.contains?(node_type, "method") or
      String.contains?(node_type, "class") or
      String.contains?(node_type, "module") or
      String.contains?(node_type, "namespace")
  end

  defp find_node_name(node) do
    children = Map.get(node, :children, [])

    name_node = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "identifier") or String.contains?(type, "name")
    end)

    if name_node do
      Map.get(name_node, :text)
    end
  end

  # ============================================================================
  # Utility Functions
  # ============================================================================

  defp get_location(node) do
    case Map.get(node, :start_point) do
      {line, col} -> {line + 1, col}
      _ -> nil
    end
  end

  defp find_ancestors_recursive(world_id, class_name, acc) do
    case CodeGazetteer.lookup_qualified(world_id, class_name) do
      {:ok, symbol} ->
        superclass = get_in(symbol, [:metadata, :superclass])
        if superclass && superclass not in acc do
          find_ancestors_recursive(world_id, superclass, [superclass | acc])
        else
          Enum.reverse(acc)
        end

      :not_found ->
        Enum.reverse(acc)
    end
  end

  defp store_relations(world_id, relations) do
    Enum.each(relations, fn relation ->
      CodeGazetteer.add_relation(world_id, relation.from, relation.type, relation.to)
    end)
  end
end
