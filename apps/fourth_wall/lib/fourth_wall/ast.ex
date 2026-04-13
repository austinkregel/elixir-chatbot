defmodule FourthWall.AST do
  @moduledoc """
  AST parsing, transformation, and serialization utilities for FourthWall.

  This module provides the core functionality for AST-based code transformations.
  It wraps Elixir's built-in Code and Macro modules with error handling and
  formatting integration.

  ## Usage

      # Parse source code
      {:ok, ast} = FourthWall.AST.parse(source_code)

      # Transform AST nodes
      {new_ast, acc} = FourthWall.AST.transform(ast, [], fn
        {:length, meta, [arg]}, acc ->
          # Transform length(x) to something else
          {{:new_form, meta, [arg]}, acc}
        node, acc ->
          {node, acc}
      end)

      # Convert back to source
      new_source = FourthWall.AST.to_source(new_ast)

  ## Round-Trip Guarantees

  The parse -> transform -> to_source pipeline preserves:
  - Semantic meaning of the code
  - Module structure and attributes
  - Function definitions and their bodies
  - String interpolation
  - Pipes and captures

  Note: Exact formatting may differ from the original due to `mix format`.
  """

  @doc """
  Parse source code into an AST.

  Uses `Code.string_to_quoted/2` with metadata preservation enabled.

  ## Options

  The underlying parser options include:
  - `:columns` - Include column information in metadata
  - `:token_metadata` - Include token metadata for formatting

  ## Examples

      iex> FourthWall.AST.parse("def foo, do: :ok")
      {:ok, {:def, [line: 1], [{:foo, [line: 1], nil}, [do: :ok]]}}

      iex> FourthWall.AST.parse("def foo(")
      {:error, {[line: 1, column: 9], "missing terminator: )", ""}}
  """
  @spec parse(String.t()) :: {:ok, Macro.t()} | {:error, term()}
  def parse(source) when is_binary(source) do
    case Code.string_to_quoted(source, columns: true, token_metadata: true) do
      {:ok, ast} -> {:ok, ast}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Convert an AST back to formatted source code.

  Uses `Macro.to_string/1` followed by `Code.format_string!/1` for consistent
  formatting.

  ## Examples

      iex> ast = quote do: def(foo, do: :ok)
      iex> FourthWall.AST.to_source(ast)
      "def foo do\\n  :ok\\nend\\n"
  """
  @spec to_source(Macro.t()) :: String.t()
  def to_source(ast) do
    ast
    |> Macro.to_string()
    |> Code.format_string!()
    |> IO.iodata_to_binary()
  end

  @doc """
  Transform an AST by walking it and applying a function to each node.

  Uses `Macro.prewalk/3` to traverse the AST depth-first, pre-order.
  The transformation function receives each node and an accumulator,
  and should return `{new_node, new_acc}`.

  ## Examples

      # Count all atoms
      {_ast, count} = FourthWall.AST.transform(ast, 0, fn
        atom, count when is_atom(atom) -> {atom, count + 1}
        node, count -> {node, count}
      end)

      # Replace all occurrences of :foo with :bar
      {new_ast, _} = FourthWall.AST.transform(ast, nil, fn
        :foo, acc -> {:bar, acc}
        node, acc -> {node, acc}
      end)
  """
  @spec transform(Macro.t(), acc, (Macro.t(), acc -> {Macro.t(), acc})) :: {Macro.t(), acc}
        when acc: term()
  def transform(ast, acc, fun) when is_function(fun, 2) do
    Macro.prewalk(ast, acc, fun)
  end

  @doc """
  Transform a file's AST and return the new source code.

  Reads the file, parses it, applies the transformation, and returns
  the formatted source code. Does not write to the file.

  ## Examples

      {:ok, new_source} = FourthWall.AST.transform_file("lib/example.ex", fn
        {:length, meta, [arg]}, acc ->
          {{:new_form, meta, [arg]}, acc}
        node, acc ->
          {node, acc}
      end)
  """
  @spec transform_file(Path.t(), (Macro.t(), acc -> {Macro.t(), acc})) ::
          {:ok, String.t()} | {:error, term()}
        when acc: term()
  def transform_file(path, fun, initial_acc \\ nil) do
    with {:ok, source} <- File.read(path),
         {:ok, ast} <- parse(source) do
      {new_ast, _acc} = transform(ast, initial_acc, fun)
      {:ok, to_source(new_ast)}
    else
      {:error, :enoent} -> {:error, :enoent}
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Check if an AST node matches a pattern.

  This is a convenience function for pattern matching in transformations.
  Useful for checking if a node is a specific function call, operator, etc.

  ## Examples

      # Check if node is a length/1 call
      FourthWall.AST.matches?(node, {:length, :_, [:_]})
  """
  @spec matches?(Macro.t(), Macro.t()) :: boolean()
  def matches?(node, pattern) do
    match_ast?(node, pattern)
  end

  # Pattern matching helper
  defp match_ast?(_node, :_), do: true
  defp match_ast?(node, [:_]) when is_list(node), do: true

  defp match_ast?({form1, _, args1}, {form2, :_, args2}) do
    match_ast?(form1, form2) and match_ast?(args1, args2)
  end

  defp match_ast?([h1 | t1], [h2 | t2]) do
    match_ast?(h1, h2) and match_ast?(t1, t2)
  end

  defp match_ast?([], []), do: true
  defp match_ast?(a, a), do: true
  defp match_ast?(_, _), do: false
end
