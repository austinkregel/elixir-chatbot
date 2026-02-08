defmodule Brain.Code.SymbolExtractor do
  @moduledoc "Extracts symbols from parsed ASTs.\n\nThis module walks through AST nodes and extracts meaningful symbols\nsuch as functions, classes, variables, and imports. It understands\nlanguage-specific patterns for each supported language.\n\n## Extraction Process\n\n1. Parse source code into AST (via Brain.Code.Parser)\n2. Walk the AST recursively\n3. Match language-specific node patterns\n4. Extract symbol metadata (name, type, location, etc.)\n5. Build qualified names based on scope\n\n## Supported Constructs\n\n| Language | Functions | Classes | Variables | Imports |\n|----------|-----------|---------|-----------|---------|\n| Elixir   | def/defp  | defmodule | = | alias/import/use |\n| Python   | def       | class   | = | import/from |\n| Ruby     | def       | class/module | = | require |\n| Go       | func      | struct/interface | := | import |\n| Java     | method    | class/interface | var | import |\n| C        | function  | struct  | declaration | #include |\n| C++      | function  | class/struct | declaration | #include |\n| C#       | method    | class/interface | var | using |\n| PHP      | function  | class   | $ | use/require |\n"

  alias Brain.Code.Parser
  require Logger

  alias Brain.Code.CodeGazetteer
  alias Brain.Telemetry

  @type extraction_result :: %{
          symbols: [map()],
          relations: [{String.t(), atom(), String.t()}],
          errors: [String.t()]
        }
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

  @doc "Extracts all symbols from an AST.\n\n## Parameters\n  - `ast` - The parsed AST (from Brain.Code.Parser)\n  - `language` - The programming language\n  - `opts` - Options\n\n## Options\n  - `:file_path` - Source file path for location tracking\n  - `:world_id` - World ID for storing symbols (optional)\n  - `:store` - Whether to store in CodeGazetteer (default: false)\n\n## Returns\n  A map with `:symbols`, `:relations`, and `:errors`\n"
  @spec extract(map(), atom(), keyword()) :: extraction_result()
  def extract(ast, language, opts \\ []) when is_map(ast) and is_atom(language) do
    Telemetry.span(:code_extract, %{language: language}, fn ->
      file_path = Keyword.get(opts, :file_path)
      world_id = Keyword.get(opts, :world_id)
      store = Keyword.get(opts, :store, false)

      context = %{
        language: language,
        file_path: file_path,
        scope_stack: [],
        symbols: [],
        relations: [],
        errors: []
      }

      result = walk_ast(ast, context)

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

  @doc "Extracts symbols from source code directly.\n\nConvenience function that parses and extracts in one step.\n"
  @spec extract_from_source(String.t(), atom(), keyword()) ::
          {:ok, extraction_result()} | {:error, term()}
  def extract_from_source(source_code, language, opts \\ []) do
    case Parser.parse(source_code, language) do
      {:ok, ast} ->
        result = extract(ast, language, opts)
        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Extracts symbols from a file.\n"
  @spec extract_from_file(String.t(), keyword()) :: {:ok, extraction_result()} | {:error, term()}
  def extract_from_file(file_path, opts \\ []) do
    case Parser.parse_file(file_path) do
      {:ok, ast} ->
        language = Map.get(ast, :language)
        opts = Keyword.put(opts, :file_path, file_path)
        result = extract(ast, language, opts)
        {:ok, result}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Gets the qualified name for a symbol given the current scope.\n"
  @spec build_qualified_name(String.t(), [String.t()], atom()) :: String.t()
  def build_qualified_name(name, scope_stack, language) do
    separator = scope_separator(language)

    case scope_stack do
      [] -> name
      scopes -> Enum.join(Enum.reverse(scopes) ++ [name], separator)
    end
  end

  defp walk_ast(node, context) when is_map(node) do
    patterns = Map.get(@patterns, context.language, %{})
    node_type = Map.get(node, :type, "")
    context = extract_from_node(node, node_type, patterns, context)
    children = Map.get(node, :children, [])

    Enum.reduce(children, context, fn child, ctx ->
      walk_ast(child, ctx)
    end)
  end

  defp walk_ast(_, context) do
    context
  end

  defp extract_from_node(node, node_type, patterns, context) do
    cond do
      matches_pattern?(node_type, Map.get(patterns, :function, [])) ->
        extract_function(node, context)

      matches_pattern?(node_type, Map.get(patterns, :method, [])) ->
        extract_function(node, context)

      matches_pattern?(node_type, Map.get(patterns, :class, [])) ->
        extract_class(node, context)

      matches_pattern?(node_type, Map.get(patterns, :module, [])) ->
        extract_module(node, context)

      matches_pattern?(node_type, Map.get(patterns, :struct, [])) ->
        extract_struct(node, context)

      matches_pattern?(node_type, Map.get(patterns, :variable, [])) ->
        extract_variable(node, context)

      matches_pattern?(node_type, Map.get(patterns, :import, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :include, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :using, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :use, [])) ->
        extract_import(node, context)

      matches_pattern?(node_type, Map.get(patterns, :type, [])) ->
        extract_type(node, context)

      matches_pattern?(node_type, Map.get(patterns, :namespace, [])) ->
        extract_namespace(node, context)

      true ->
        context
    end
  end

  defp matches_pattern?(_node_type, []) do
    false
  end

  defp matches_pattern?(node_type, patterns) when is_list(patterns) do
    Enum.any?(patterns, fn pattern ->
      String.contains?(node_type, pattern)
    end)
  end

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

      new_context = %{
        context
        | symbols: [symbol | context.symbols],
          scope_stack: [name | context.scope_stack]
      }

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

      %{context | symbols: [symbol | context.symbols], scope_stack: [name | context.scope_stack]}
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

      current_module = List.first(context.scope_stack) || context.file_path || "unknown"
      relation = {current_module, :imports, import_target}

      %{context | symbols: [symbol | context.symbols], relations: [relation | context.relations]}
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

      %{context | symbols: [symbol | context.symbols], scope_stack: [name | context.scope_stack]}
    else
      context
    end
  end

  defp find_name_in_node(node, _language) do
    cond do
      Map.has_key?(node, :name) ->
        node.name

      true ->
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
    case language do
      :php ->
        text = Map.get(node, :text, "")

        if String.starts_with?(text, "$") do
          text
        else
          find_name_in_node(node, language)
        end

      _ ->
        find_name_in_node(node, language)
    end
  end

  defp find_import_target(node, _language) do
    text = Map.get(node, :text, "")
    children = Map.get(node, :children, [])

    import_child =
      Enum.find(children, fn child ->
        type = Map.get(child, :type, "")

        String.contains?(type, "identifier") or
          String.contains?(type, "dotted_name") or
          String.contains?(type, "string")
      end)

    if import_child do
      Map.get(import_child, :text, "")
    else
      extract_import_from_text(text)
    end
  end

  defp extract_import_from_text(text) do
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

    params =
      Enum.find(children, fn child ->
        type = Map.get(child, :type, "")
        String.contains?(type, "parameter") or String.contains?(type, "arguments")
      end)

    arity =
      if params do
        param_children = Map.get(params, :children, [])
        length(param_children)
      else
        0
      end

    visibility =
      case language do
        :elixir ->
          text = Map.get(node, :text, "")

          if String.contains?(text, "defp") do
            :private
          else
            :public
          end

        :python ->
          name = find_name_in_node(node, language) || ""

          if String.starts_with?(name, "_") do
            :private
          else
            :public
          end

        _ ->
          :public
      end

    %{arity: arity, visibility: visibility}
  end

  defp extract_class_metadata(node, language) do
    children = Map.get(node, :children, [])

    superclass =
      Enum.find_value(children, fn child ->
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