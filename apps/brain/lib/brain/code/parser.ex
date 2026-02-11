defmodule Brain.Code.Parser do
  @moduledoc """
  Parses source code into ASTs using tree-sitter.

  This module provides the core parsing functionality for the code analysis
  system. It uses tree-sitter grammars to generate concrete syntax trees
  that can be analyzed for symbols, relationships, and semantic meaning.

  ## Supported Languages

  - C family: C, C++, Java, C#, PHP
  - Scripting: Python, Ruby
  - Modern: Elixir, Go

  ## Usage

      # Parse a string of code
      {:ok, ast} = Brain.Code.Parser.parse("def hello, do: :world", :elixir)

      # Parse a file (auto-detects language)
      {:ok, ast} = Brain.Code.Parser.parse_file("/path/to/file.py")

      # Get just the language detection
      :python = Brain.Code.Parser.detect_language("script.py")
  """

  # TreeSitter is an optional dependency (not currently installed).
  # See mix.exs for installation instructions.
  @compile {:no_warn_undefined, TreeSitter}

  require Logger

  alias Brain.Code.LanguageGrammar
  alias Brain.Telemetry

  @type language :: :c | :cpp | :java | :csharp | :php | :python | :ruby | :elixir | :go
  @type ast_node :: %{
          type: String.t(),
          text: String.t(),
          start_byte: non_neg_integer(),
          end_byte: non_neg_integer(),
          start_point: {non_neg_integer(), non_neg_integer()},
          end_point: {non_neg_integer(), non_neg_integer()},
          children: [ast_node()]
        }
  @type parse_result :: {:ok, ast_node()} | {:error, term()}

  # Extension to language mapping
  @extension_map %{
    ".c" => :c,
    ".h" => :c,
    ".cpp" => :cpp,
    ".cc" => :cpp,
    ".cxx" => :cpp,
    ".hpp" => :cpp,
    ".hxx" => :cpp,
    ".java" => :java,
    ".cs" => :csharp,
    ".php" => :php,
    ".py" => :python,
    ".pyw" => :python,
    ".rb" => :ruby,
    ".ex" => :elixir,
    ".exs" => :elixir,
    ".go" => :go
  }

  # All supported languages
  @supported_languages [:c, :cpp, :java, :csharp, :php, :python, :ruby, :elixir, :go]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Parses source code into an AST.

  ## Parameters
    - `source_code` - The source code string to parse
    - `language` - The programming language (atom)

  ## Returns
    - `{:ok, ast_node}` - The parsed AST
    - `{:error, reason}` - If parsing failed

  ## Examples

      iex> Brain.Code.Parser.parse("def add(a, b), do: a + b", :elixir)
      {:ok, %{type: "source", children: [...]}}
  """
  @spec parse(String.t(), language()) :: parse_result()
  def parse(source_code, language) when is_binary(source_code) and is_atom(language) do
    Telemetry.span(:code_parse, %{language: language, size: byte_size(source_code)}, fn ->
      if language in @supported_languages do
        case LanguageGrammar.get_parser(language) do
          {:ok, parser} ->
            do_parse(parser, source_code, language)

          {:error, reason} ->
            {:error, {:grammar_not_available, language, reason}}
        end
      else
        {:error, {:unsupported_language, language}}
      end
    end)
  end

  def parse(_, language) when not is_atom(language) do
    {:error, {:invalid_language, language}}
  end

  def parse(source_code, _) when not is_binary(source_code) do
    {:error, :invalid_source_code}
  end

  @doc """
  Parses a file, auto-detecting the language from the extension.

  ## Parameters
    - `file_path` - Path to the source file

  ## Returns
    - `{:ok, ast_node}` - The parsed AST with language metadata
    - `{:error, reason}` - If parsing failed

  ## Examples

      iex> Brain.Code.Parser.parse_file("/path/to/script.py")
      {:ok, %{type: "source", language: :python, ...}}
  """
  @spec parse_file(String.t()) :: parse_result()
  def parse_file(file_path) when is_binary(file_path) do
    case detect_language(file_path) do
      :unknown ->
        {:error, {:unknown_language, file_path}}

      language ->
        case File.read(file_path) do
          {:ok, content} ->
            case parse(content, language) do
              {:ok, ast} ->
                {:ok, Map.put(ast, :language, language)}

              error ->
                error
            end

          {:error, reason} ->
            {:error, {:file_read_failed, reason}}
        end
    end
  end

  @doc """
  Detects the programming language from a file path.

  Uses file extension for detection. Returns `:unknown` if the
  extension is not recognized.

  ## Examples

      iex> Brain.Code.Parser.detect_language("main.py")
      :python

      iex> Brain.Code.Parser.detect_language("lib/myapp.ex")
      :elixir

      iex> Brain.Code.Parser.detect_language("unknown.xyz")
      :unknown
  """
  @spec detect_language(String.t()) :: language() | :unknown
  def detect_language(file_path) when is_binary(file_path) do
    extension = Path.extname(file_path) |> String.downcase()
    Map.get(@extension_map, extension, :unknown)
  end

  @doc """
  Returns the list of supported languages.

  ## Examples

      iex> Brain.Code.Parser.supported_languages()
      [:c, :cpp, :java, :csharp, :php, :python, :ruby, :elixir, :go]
  """
  @spec supported_languages() :: [language()]
  def supported_languages, do: @supported_languages

  @doc """
  Checks if a language is supported.

  ## Examples

      iex> Brain.Code.Parser.language_supported?(:python)
      true

      iex> Brain.Code.Parser.language_supported?(:cobol)
      false
  """
  @spec language_supported?(atom()) :: boolean()
  def language_supported?(language), do: language in @supported_languages

  @doc """
  Returns file extensions for a given language.

  ## Examples

      iex> Brain.Code.Parser.extensions_for(:python)
      [".py", ".pyw"]
  """
  @spec extensions_for(language()) :: [String.t()]
  def extensions_for(language) do
    @extension_map
    |> Enum.filter(fn {_ext, lang} -> lang == language end)
    |> Enum.map(fn {ext, _lang} -> ext end)
  end

  @doc """
  Parses source code and returns a simplified tree structure.

  This is useful for debugging and visualization.

  ## Options
    - `:max_depth` - Maximum tree depth to return (default: 10)
    - `:include_text` - Include source text in nodes (default: false)
  """
  @spec parse_simplified(String.t(), language(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def parse_simplified(source_code, language, opts \\ []) do
    case parse(source_code, language) do
      {:ok, ast} ->
        max_depth = Keyword.get(opts, :max_depth, 10)
        include_text = Keyword.get(opts, :include_text, false)
        {:ok, simplify_ast(ast, max_depth, include_text, 0)}

      error ->
        error
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  # Perform the actual parsing using tree-sitter
  defp do_parse(parser, source_code, language) do
    try do
      # Use tree-sitter to parse the code
      case apply_parser(parser, source_code) do
        {:ok, tree} ->
          ast = tree_to_ast(tree, source_code)
          {:ok, Map.put(ast, :language, language)}

        {:error, reason} ->
          {:error, {:parse_failed, reason}}
      end
    rescue
      e ->
        Logger.warning("Parse error for #{language}: #{inspect(e)}")
        {:error, {:parse_exception, Exception.message(e)}}
    end
  end

  # Apply the parser to source code
  # This wraps the tree-sitter NIF call
  defp apply_parser(parser, source_code) do
    # Check if tree_sitter module is available
    if Code.ensure_loaded?(TreeSitter) do
      TreeSitter.parse(parser, source_code)
    else
      # Fallback to internal simple parser when tree-sitter is not available
      fallback_parse(source_code)
    end
  end

  # Fallback parser for when tree-sitter is not available
  # Provides basic tokenization-based AST
  # Smart fallback parser that extracts symbols using regex patterns
  # This is used when tree-sitter is not available
  defp fallback_parse(source_code) do
    lines = String.split(source_code, "\n")

    # Extract structured nodes from lines
    children =
      lines
      |> Enum.with_index()
      |> Enum.flat_map(fn {line, idx} ->
        extract_nodes_from_line(line, idx)
      end)

    tree = %{
      type: "source",
      text: source_code,
      start_byte: 0,
      end_byte: byte_size(source_code),
      start_point: {0, 0},
      end_point: {length(lines) - 1, String.length(List.last(lines) || "")},
      children: children
    }

    {:ok, tree}
  end

  # Extract AST-like nodes from a single line using pattern matching
  defp extract_nodes_from_line(line, line_num) do
    trimmed = String.trim(line)
    nodes = []

    nodes = nodes ++ extract_elixir_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_python_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_ruby_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_go_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_java_csharp_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_c_cpp_nodes(trimmed, line, line_num)
    nodes = nodes ++ extract_php_nodes(trimmed, line, line_num)

    # If no specific nodes found, return a generic line node
    if nodes == [] do
      [make_line_node(line, line_num)]
    else
      nodes
    end
  end

  defp make_line_node(line, line_num) do
    %{
      type: "line",
      text: line,
      start_byte: 0,
      end_byte: byte_size(line),
      start_point: {line_num, 0},
      end_point: {line_num, String.length(line)},
      children: []
    }
  end

  defp make_node(type, name, line, line_num, opts \\ []) do
    children = Keyword.get(opts, :children, [])
    %{
      type: type,
      text: line,
      name: name,
      start_byte: 0,
      end_byte: byte_size(line),
      start_point: {line_num, 0},
      end_point: {line_num, String.length(line)},
      children: children
    }
  end

  # Elixir pattern extraction
  defp extract_elixir_nodes(trimmed, line, line_num) do
    cond do
      # Module definition
      String.starts_with?(trimmed, "defmodule ") ->
        case Regex.run(~r/defmodule\s+([A-Z][A-Za-z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("module_definition", name, line, line_num)]
          _ -> []
        end

      # Function definition
      String.starts_with?(trimmed, "def ") or String.starts_with?(trimmed, "defp ") ->
        case Regex.run(~r/def(?:p)?\s+([a-z_][a-z0-9_?!]*)/, trimmed) do
          [_, name] -> [make_node("function_definition", name, line, line_num)]
          _ -> []
        end

      # Macro definition
      String.starts_with?(trimmed, "defmacro ") or String.starts_with?(trimmed, "defmacrop ") ->
        case Regex.run(~r/defmacro(?:p)?\s+([a-z_][a-z0-9_?!]*)/, trimmed) do
          [_, name] -> [make_node("macro_definition", name, line, line_num)]
          _ -> []
        end

      # Alias/import/use
      String.starts_with?(trimmed, "alias ") ->
        case Regex.run(~r/alias\s+([A-Z][A-Za-z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("alias", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "import ") ->
        case Regex.run(~r/import\s+([A-Z][A-Za-z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("import", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "use ") ->
        case Regex.run(~r/use\s+([A-Z][A-Za-z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("use", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "require ") ->
        case Regex.run(~r/require\s+([A-Z][A-Za-z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("require", name, line, line_num)]
          _ -> []
        end

      # Module attribute
      String.starts_with?(trimmed, "@") and not String.starts_with?(trimmed, "@doc") and not String.starts_with?(trimmed, "@moduledoc") ->
        case Regex.run(~r/@([a-z_][a-z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("module_attribute", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # Python pattern extraction
  defp extract_python_nodes(trimmed, line, line_num) do
    cond do
      String.starts_with?(trimmed, "def ") ->
        case Regex.run(~r/def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, trimmed) do
          [_, name] -> [make_node("function_definition", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "class ") ->
        case Regex.run(~r/class\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("class_definition", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "import ") ->
        case Regex.run(~r/import\s+([a-zA-Z_][a-zA-Z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("import_statement", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "from ") ->
        case Regex.run(~r/from\s+([a-zA-Z_][a-zA-Z0-9_.]+)\s+import/, trimmed) do
          [_, name] -> [make_node("import_from_statement", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # Ruby pattern extraction
  defp extract_ruby_nodes(trimmed, line, line_num) do
    cond do
      String.starts_with?(trimmed, "def ") ->
        case Regex.run(~r/def\s+([a-z_][a-z0-9_?!]*)/, trimmed) do
          [_, name] -> [make_node("method_definition", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "class ") ->
        case Regex.run(~r/class\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("class", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "module ") ->
        case Regex.run(~r/module\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("module", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "require ") or String.starts_with?(trimmed, "require_relative ") ->
        case Regex.run(~r/require(?:_relative)?\s+['"]([^'"]+)['"]/, trimmed) do
          [_, name] -> [make_node("call", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # Go pattern extraction
  defp extract_go_nodes(trimmed, line, line_num) do
    cond do
      String.starts_with?(trimmed, "func ") ->
        case Regex.run(~r/func\s+(?:\([^)]+\)\s+)?([a-zA-Z_][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("function_declaration", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "type ") ->
        case Regex.run(~r/type\s+([A-Z][a-zA-Z0-9_]*)\s+(?:struct|interface)/, trimmed) do
          [_, name] -> [make_node("type_declaration", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "package ") ->
        case Regex.run(~r/package\s+([a-z][a-z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("package_clause", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # Java/C# pattern extraction
  defp extract_java_csharp_nodes(trimmed, line, line_num) do
    cond do
      # Class/interface definition
      Regex.match?(~r/(public|private|protected|internal)?\s*(abstract|static|sealed)?\s*(class|interface|enum)\s+/, trimmed) ->
        case Regex.run(~r/(?:class|interface|enum)\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("class_declaration", name, line, line_num)]
          _ -> []
        end

      # Method definition (simplified pattern)
      Regex.match?(~r/(public|private|protected)\s+.*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(/, trimmed) ->
        case Regex.run(~r/\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, trimmed) do
          [_, name] when name not in ["if", "for", "while", "switch", "catch"] ->
            [make_node("method_declaration", name, line, line_num)]
          _ -> []
        end

      # Import/using
      String.starts_with?(trimmed, "import ") ->
        case Regex.run(~r/import\s+([a-zA-Z_][a-zA-Z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("import_declaration", name, line, line_num)]
          _ -> []
        end

      String.starts_with?(trimmed, "using ") and not String.contains?(trimmed, "(") ->
        case Regex.run(~r/using\s+([A-Z][a-zA-Z0-9_.]+)/, trimmed) do
          [_, name] -> [make_node("using_directive", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # C/C++ pattern extraction
  defp extract_c_cpp_nodes(trimmed, line, line_num) do
    cond do
      # Include directive
      String.starts_with?(trimmed, "#include") ->
        case Regex.run(~r/#include\s*[<"]([^>"]+)[>"]/, trimmed) do
          [_, name] -> [make_node("preproc_include", name, line, line_num)]
          _ -> []
        end

      # Class/struct definition
      Regex.match?(~r/^(class|struct)\s+[A-Z]/, trimmed) ->
        case Regex.run(~r/(?:class|struct)\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("class_specifier", name, line, line_num)]
          _ -> []
        end

      # Function definition (simplified - looks for type name(
      Regex.match?(~r/^[a-zA-Z_][a-zA-Z0-9_*&\s]+\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(/, trimmed) and
          not String.starts_with?(trimmed, "if") and
          not String.starts_with?(trimmed, "for") and
          not String.starts_with?(trimmed, "while") ->
        case Regex.run(~r/\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/, trimmed) do
          [_, name] -> [make_node("function_definition", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # PHP pattern extraction
  defp extract_php_nodes(trimmed, line, line_num) do
    cond do
      # Function definition
      String.starts_with?(trimmed, "function ") or Regex.match?(~r/(public|private|protected)\s+function\s+/, trimmed) ->
        case Regex.run(~r/function\s+([a-zA-Z_][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("function_definition", name, line, line_num)]
          _ -> []
        end

      # Class definition
      Regex.match?(~r/^(abstract\s+)?class\s+/, trimmed) ->
        case Regex.run(~r/class\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("class_declaration", name, line, line_num)]
          _ -> []
        end

      # Interface/trait
      String.starts_with?(trimmed, "interface ") or String.starts_with?(trimmed, "trait ") ->
        case Regex.run(~r/(?:interface|trait)\s+([A-Z][a-zA-Z0-9_]*)/, trimmed) do
          [_, name] -> [make_node("interface_declaration", name, line, line_num)]
          _ -> []
        end

      # Namespace
      String.starts_with?(trimmed, "namespace ") ->
        case Regex.run(~r/namespace\s+([A-Za-z][A-Za-z0-9_\\]+)/, trimmed) do
          [_, name] -> [make_node("namespace_definition", name, line, line_num)]
          _ -> []
        end

      # Use statement
      String.starts_with?(trimmed, "use ") ->
        case Regex.run(~r/use\s+([A-Za-z][A-Za-z0-9_\\]+)/, trimmed) do
          [_, name] -> [make_node("namespace_use_declaration", name, line, line_num)]
          _ -> []
        end

      true -> []
    end
  end

  # Convert tree-sitter tree to our AST format
  defp tree_to_ast(tree, _source_code) when is_map(tree) do
    # Already in map format
    tree
  end

  defp tree_to_ast(tree, source_code) do
    # Convert from tree-sitter's internal format
    %{
      type: get_node_type(tree),
      text: get_node_text(tree, source_code),
      start_byte: get_start_byte(tree),
      end_byte: get_end_byte(tree),
      start_point: get_start_point(tree),
      end_point: get_end_point(tree),
      children: get_children(tree, source_code)
    }
  end

  # Node accessors - these handle both map format and tree-sitter format
  defp get_node_type(%{type: type}), do: type
  defp get_node_type(node) when is_tuple(node), do: elem(node, 0) |> to_string()
  defp get_node_type(_), do: "unknown"

  defp get_node_text(%{text: text}, _source), do: text

  defp get_node_text(node, source_code) do
    start_byte = get_start_byte(node)
    end_byte = get_end_byte(node)
    binary_part(source_code, start_byte, end_byte - start_byte)
  rescue
    _ -> ""
  end

  defp get_start_byte(%{start_byte: b}), do: b
  defp get_start_byte(_), do: 0

  defp get_end_byte(%{end_byte: b}), do: b
  defp get_end_byte(_), do: 0

  defp get_start_point(%{start_point: p}), do: p
  defp get_start_point(_), do: {0, 0}

  defp get_end_point(%{end_point: p}), do: p
  defp get_end_point(_), do: {0, 0}

  defp get_children(%{children: children}, source_code) do
    Enum.map(children, &tree_to_ast(&1, source_code))
  end

  defp get_children(_, _), do: []

  # Simplify AST for debugging/display
  defp simplify_ast(_node, max_depth, _include_text, depth) when depth >= max_depth do
    %{type: "...", truncated: true}
  end

  defp simplify_ast(node, max_depth, include_text, depth) do
    base = %{type: node.type}

    base =
      if include_text and node.text != "" do
        Map.put(base, :text, String.slice(node.text, 0, 50))
      else
        base
      end

    case node.children do
      [] ->
        base

      children ->
        simplified_children =
          Enum.map(children, &simplify_ast(&1, max_depth, include_text, depth + 1))

        Map.put(base, :children, simplified_children)
    end
  end
end
