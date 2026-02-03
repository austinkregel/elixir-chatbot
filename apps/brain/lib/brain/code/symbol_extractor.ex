defmodule Brain.Code.SymbolExtractor do
  @moduledoc """
  Extracts symbols from parsed ASTs.

  This module walks through AST nodes and extracts meaningful symbols
  such as functions, classes, variables, and imports. It understands
  language-specific patterns for each supported language.

  ## Extraction Process

  1. Parse source code into AST (via Brain.Code.Parser)
  2. Walk the AST recursively
  3. Match language-specific node patterns
  4. Extract symbol metadata (name, type, location, etc.)
  5. Build qualified names based on scope

  ## Supported Constructs

  | Language | Functions | Classes | Variables | Imports |
  |----------|-----------|---------|-----------|---------|
  | Elixir   | def/defp  | defmodule | = | alias/import/use |
  | Python   | def       | class   | = | import/from |
  | Ruby     | def       | class/module | = | require |
  | Go       | func      | struct/interface | := | import |
  | Java     | method    | class/interface | var | import |
  | C        | function  | struct  | declaration | #include |
  | C++      | function  | class/struct | declaration | #include |
  | C#       | method    | class/interface | var | using |
  | PHP      | function  | class   | $ | use/require |
  """

  require Logger

  alias Brain.Code.CodeGazetteer
  alias Brain.Telemetry

  @type extraction_result :: %{
          symbols: [map()],
          relations: [{String.t(), atom(), String.t()}],
          errors: [String.t()]
        }

  # Node type patterns for each language
  # These match both tree-sitter AST node types and our fallback parser types
  @patterns %{
    elixir: %{
      module: ["module_definition", "call"],
      function: ["function_definition", "macro_definition", "call"],
      variable: ["match_operator", "assignment", "module_attribute"],
      import: ["alias", "import", "use", "require"]
    },
    python: %{
      function: ["function_definition", "function_def"],
      class: ["class_definition", "class_def"],
      variable: ["assignment", "annotated_assignment"],
      import: ["import_statement", "import_from_statement"]
    },
    ruby: %{
      function: ["method", "method_definition", "singleton_method"],
      class: ["class", "module"],
      variable: ["assignment", "lhs"],
      import: ["call"]
    },
    go: %{
      function: ["function_declaration", "method_declaration"],
      type: ["type_declaration", "type_spec"],
      variable: ["short_var_declaration", "var_declaration"],
      import: ["import_declaration", "import_spec"],
      namespace: ["package_clause"]
    },
    java: %{
      method: ["method_declaration"],
      class: ["class_declaration", "interface_declaration", "enum_declaration"],
      variable: ["local_variable_declaration", "field_declaration"],
      import: ["import_declaration"]
    },
    c: %{
      function: ["function_definition", "function_declarator"],
      struct: ["struct_specifier", "class_specifier"],
      variable: ["declaration", "init_declarator"],
      include: ["preproc_include"]
    },
    cpp: %{
      function: ["function_definition", "function_declarator"],
      class: ["class_specifier", "struct_specifier"],
      variable: ["declaration", "init_declarator"],
      include: ["preproc_include"],
      namespace: ["namespace_definition"]
    },
    csharp: %{
      method: ["method_declaration"],
      class: ["class_declaration", "interface_declaration", "struct_declaration"],
      variable: ["variable_declaration", "field_declaration"],
      using: ["using_directive"],
      namespace: ["namespace_declaration"]
    },
    php: %{
      function: ["function_definition", "method_declaration"],
      class: ["class_declaration", "interface_declaration", "trait_declaration"],
      variable: ["simple_variable", "property_declaration"],
      use: ["namespace_use_declaration"],
      namespace: ["namespace_definition"]
    }
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Extracts all symbols from an AST.

  ## Parameters
    - `ast` - The parsed AST (from Brain.Code.Parser)
    - `language` - The programming language
    - `opts` - Options

  ## Options
    - `:file_path` - Source file path for location tracking
    - `:world_id` - World ID for storing symbols (optional)
    - `:store` - Whether to store in CodeGazetteer (default: false)

  ## Returns
    A map with `:symbols`, `:relations`, and `:errors`
  """
  @spec extract(map(), atom(), keyword()) :: extraction_result()
  def extract(ast, language, opts \\ []) when is_map(ast) and is_atom(language) do
    Telemetry.span(:code_extract, %{language: language}, fn ->
      file_path = Keyword.get(opts, :file_path)
      world_id = Keyword.get(opts, :world_id)
      store = Keyword.get(opts, :store, false)

      # Initialize extraction context
      context = %{
        language: language,
        file_path: file_path,
        scope_stack: [],
        symbols: [],
        relations: [],
        errors: []
      }

      # Walk the AST
      result = walk_ast(ast, context)

      # Store if requested
      if store and world_id do
        store_symbols(world_id, result.symbols)
        store_relations(world_id, result.relations)
      end

      %{
        symbols: Enum.reverse(result.symbols),
        relations: Enum.reverse(result.relations),
        errors: Enum.reverse(result.errors)
      }
    end)
  end

  @doc """
  Extracts symbols from source code directly.

  Convenience function that parses and extracts in one step.
  """
  @spec extract_from_source(String.t(), atom(), keyword()) :: {:ok, extraction_result()} | {:error, term()}
  def extract_from_source(source_code, language, opts \\ []) do
    case Brain.Code.Parser.parse(source_code, language) do
      {:ok, ast} ->
        result = extract(ast, language, opts)
        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Extracts symbols from a file.
  """
  @spec extract_from_file(String.t(), keyword()) :: {:ok, extraction_result()} | {:error, term()}
  def extract_from_file(file_path, opts \\ []) do
    case Brain.Code.Parser.parse_file(file_path) do
      {:ok, ast} ->
        language = Map.get(ast, :language)
        opts = Keyword.put(opts, :file_path, file_path)
        result = extract(ast, language, opts)
        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Gets the qualified name for a symbol given the current scope.
  """
  @spec build_qualified_name(String.t(), [String.t()], atom()) :: String.t()
  def build_qualified_name(name, scope_stack, language) do
    separator = scope_separator(language)

    case scope_stack do
      [] -> name
      scopes -> Enum.join(Enum.reverse(scopes) ++ [name], separator)
    end
  end

  # ============================================================================
  # Private Functions - AST Walking
  # ============================================================================

  defp walk_ast(node, context) when is_map(node) do
    # Get language patterns
    patterns = Map.get(@patterns, context.language, %{})

    # Check what type of node this is
    node_type = Map.get(node, :type, "")

    # Try to extract based on node type
    context = extract_from_node(node, node_type, patterns, context)

    # Recurse into children
    children = Map.get(node, :children, [])

    Enum.reduce(children, context, fn child, ctx ->
      walk_ast(child, ctx)
    end)
  end

  defp walk_ast(_, context), do: context

  defp extract_from_node(node, node_type, patterns, context) do
    cond do
      # Check for function definitions
      matches_pattern?(node_type, Map.get(patterns, :function, [])) ->
        extract_function(node, context)

      matches_pattern?(node_type, Map.get(patterns, :method, [])) ->
        extract_function(node, context)

      # Check for class/module definitions
      matches_pattern?(node_type, Map.get(patterns, :class, [])) ->
        extract_class(node, context)

      matches_pattern?(node_type, Map.get(patterns, :module, [])) ->
        extract_module(node, context)

      matches_pattern?(node_type, Map.get(patterns, :struct, [])) ->
        extract_struct(node, context)

      # Check for variable declarations
      matches_pattern?(node_type, Map.get(patterns, :variable, [])) ->
        extract_variable(node, context)

      # Check for imports
      matches_pattern?(node_type, Map.get(patterns, :import, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :include, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :using, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :use, [])) ->
        extract_import(node, context)

      # Check for type definitions
      matches_pattern?(node_type, Map.get(patterns, :type, [])) ->
        extract_type(node, context)

      # Check for namespace
      matches_pattern?(node_type, Map.get(patterns, :namespace, [])) ->
        extract_namespace(node, context)

      true ->
        context
    end
  end

  defp matches_pattern?(_node_type, []), do: false

  defp matches_pattern?(node_type, patterns) when is_list(patterns) do
    Enum.any?(patterns, fn pattern ->
      String.contains?(node_type, pattern)
    end)
  end

  # ============================================================================
  # Symbol Extraction Functions
  # ============================================================================

  defp extract_function(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.function",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: extract_function_metadata(node, context.language)
      }

      %{context | symbols: [symbol | context.symbols]}
    else
      context
    end
  end

  defp extract_class(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.class",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: extract_class_metadata(node, context.language)
      }

      # Push class to scope for nested definitions
      new_context = %{context |
        symbols: [symbol | context.symbols],
        scope_stack: [name | context.scope_stack]
      }

      # Process children with updated scope, then pop scope
      # Note: This is handled by the recursive walk, not here
      new_context
    else
      context
    end
  end

  defp extract_module(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.namespace",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{}
      }

      %{context |
        symbols: [symbol | context.symbols],
        scope_stack: [name | context.scope_stack]
      }
    else
      context
    end
  end

  defp extract_struct(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.class",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{kind: :struct}
      }

      %{context | symbols: [symbol | context.symbols]}
    else
      context
    end
  end

  defp extract_variable(node, context) do
    name = find_variable_name(node, context.language)

    if name && name != "" && not is_parameter?(name) do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.variable",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{}
      }

      %{context | symbols: [symbol | context.symbols]}
    else
      context
    end
  end

  defp extract_import(node, context) do
    import_target = find_import_target(node, context.language)

    if import_target && import_target != "" do
      symbol = %{
        name: import_target,
        qualified_name: import_target,
        entity_type: "code.import",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{}
      }

      # Also record as a relation
      current_module = List.first(context.scope_stack) || context.file_path || "unknown"
      relation = {current_module, :imports, import_target}

      %{context |
        symbols: [symbol | context.symbols],
        relations: [relation | context.relations]
      }
    else
      context
    end
  end

  defp extract_type(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      qualified_name = build_qualified_name(name, context.scope_stack, context.language)

      symbol = %{
        name: name,
        qualified_name: qualified_name,
        entity_type: "code.type",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{}
      }

      %{context | symbols: [symbol | context.symbols]}
    else
      context
    end
  end

  defp extract_namespace(node, context) do
    name = find_name_in_node(node, context.language)

    if name && name != "" do
      symbol = %{
        name: name,
        qualified_name: name,
        entity_type: "code.namespace",
        language: context.language,
        file_path: context.file_path,
        line: get_line(node),
        column: get_column(node),
        metadata: %{}
      }

      %{context |
        symbols: [symbol | context.symbols],
        scope_stack: [name | context.scope_stack]
      }
    else
      context
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

  defp find_name_in_node(node, _language) do
    # Look for identifier children or name field
    cond do
      Map.has_key?(node, :name) ->
        node.name

      true ->
        # Search children for identifier
        children = Map.get(node, :children, [])

        Enum.find_value(children, fn child ->
          type = Map.get(child, :type, "")

          if String.contains?(type, "identifier") or String.contains?(type, "name") do
            Map.get(child, :text, "")
          end
        end)
    end
  end

  defp find_variable_name(node, language) do
    # Language-specific variable name extraction
    case language do
      :php ->
        # PHP variables start with $
        text = Map.get(node, :text, "")
        if String.starts_with?(text, "$"), do: text, else: find_name_in_node(node, language)

      _ ->
        find_name_in_node(node, language)
    end
  end

  defp find_import_target(node, _language) do
    # Look for the imported module/path
    text = Map.get(node, :text, "")

    # Try to extract the module name from import statement
    children = Map.get(node, :children, [])

    import_child = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "identifier") or
        String.contains?(type, "dotted_name") or
        String.contains?(type, "string")
    end)

    if import_child do
      Map.get(import_child, :text, "")
    else
      # Fall back to trying to parse from text
      extract_import_from_text(text)
    end
  end

  defp extract_import_from_text(text) do
    # Simple extraction - get first identifier-like string after import keyword
    text
    |> String.split(~r/\s+/)
    |> Enum.drop(1)
    |> List.first()
    |> case do
      nil -> nil
      s -> String.trim(s, "\"'")
    end
  end

  defp extract_function_metadata(node, language) do
    children = Map.get(node, :children, [])

    # Try to find parameters
    params = Enum.find(children, fn child ->
      type = Map.get(child, :type, "")
      String.contains?(type, "parameter") or String.contains?(type, "arguments")
    end)

    arity = if params do
      param_children = Map.get(params, :children, [])
      length(param_children)
    else
      0
    end

    # Check visibility (language-specific)
    visibility = case language do
      :elixir ->
        text = Map.get(node, :text, "")
        if String.contains?(text, "defp"), do: :private, else: :public

      :python ->
        name = find_name_in_node(node, language) || ""
        if String.starts_with?(name, "_"), do: :private, else: :public

      _ ->
        :public
    end

    %{arity: arity, visibility: visibility}
  end

  defp extract_class_metadata(node, language) do
    children = Map.get(node, :children, [])

    # Try to find superclass/extends
    superclass = Enum.find_value(children, fn child ->
      type = Map.get(child, :type, "")
      if String.contains?(type, "superclass") or String.contains?(type, "extends") do
        find_name_in_node(child, language)
      end
    end)

    %{superclass: superclass}
  end

  defp get_line(node) do
    case Map.get(node, :start_point) do
      {line, _col} -> line + 1
      _ -> nil
    end
  end

  defp get_column(node) do
    case Map.get(node, :start_point) do
      {_line, col} -> col
      _ -> nil
    end
  end

  defp is_parameter?(name) do
    # Simple heuristic - parameters are usually short and common names
    name in ["self", "this", "cls", "_", "__"]
  end

  defp scope_separator(language) do
    case language do
      :elixir -> "."
      :python -> "."
      :ruby -> "::"
      :go -> "."
      :java -> "."
      :csharp -> "."
      :cpp -> "::"
      :php -> "\\"
      _ -> "."
    end
  end

  defp store_symbols(world_id, symbols) do
    Enum.each(symbols, fn symbol ->
      CodeGazetteer.add_symbol(world_id, symbol)
    end)
  end

  defp store_relations(world_id, relations) do
    Enum.each(relations, fn {from, type, to} ->
      CodeGazetteer.add_relation(world_id, from, type, to)
    end)
  end
end
