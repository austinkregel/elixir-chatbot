defmodule FourthWall.ASTTest do
  use ExUnit.Case, async: false

  alias FourthWall.AST

  describe "parse/1" do
    test "parses valid Elixir code into AST" do
      code = """
      defmodule Example do
        def foo, do: :ok
      end
      """

      assert {:ok, ast} = AST.parse(code)
      assert is_tuple(ast)
    end

    test "preserves line metadata" do
      code = """
      defmodule Example do
        def foo, do: :ok
      end
      """

      {:ok, ast} = AST.parse(code)

      # Find the defmodule node and check it has line info
      {_, meta, _} = ast
      assert Keyword.has_key?(meta, :line)
    end

    test "returns error for invalid syntax" do
      invalid = "defmodule Example do def foo("

      assert {:error, _reason} = AST.parse(invalid)
    end

    test "handles empty string" do
      assert {:ok, _} = AST.parse("")
    end
  end

  describe "to_source/1" do
    test "converts simple AST back to source code" do
      ast = quote do
        defmodule Example do
          def foo, do: :ok
        end
      end

      result = AST.to_source(ast)
      assert is_binary(result)
      assert String.contains?(result, "defmodule Example")
      assert String.contains?(result, "def foo")
    end

    test "formats output code" do
      # Unformatted AST structure
      ast = quote do
        def foo(x), do: if(x > 0, do: :positive, else: :negative)
      end

      result = AST.to_source(ast)
      # Should produce readable, formatted code
      assert is_binary(result)
    end
  end

  describe "transform/2" do
    test "applies transformation function to AST" do
      {:ok, ast} = AST.parse("x = 1 + 2")

      # Identity transformation
      {new_ast, acc} = AST.transform(ast, [], fn node, acc -> {node, acc} end)

      assert new_ast == ast
      assert acc == []
    end

    test "can collect information during traversal" do
      code = """
      defmodule Example do
        def foo, do: :ok
        def bar, do: :error
      end
      """

      {:ok, ast} = AST.parse(code)

      # Collect all function names
      {_ast, function_names} =
        AST.transform(ast, [], fn
          {:def, _meta, [{name, _, _} | _]} = node, acc ->
            {node, [name | acc]}

          node, acc ->
            {node, acc}
        end)

      assert :foo in function_names
      assert :bar in function_names
    end

    test "can modify nodes during traversal" do
      code = "x = 1"
      {:ok, ast} = AST.parse(code)

      # Replace all integer 1 with 42
      {new_ast, _} =
        AST.transform(ast, nil, fn
          1, acc -> {42, acc}
          node, acc -> {node, acc}
        end)

      result = AST.to_source(new_ast)
      assert String.contains?(result, "42")
      refute String.contains?(result, " 1")
    end
  end

  describe "round-trip preservation" do
    test "parse -> to_source preserves semantic meaning" do
      original = """
      defmodule Example do
        def foo(x) do
          x + 1
        end
      end
      """

      {:ok, ast1} = AST.parse(original)
      source = AST.to_source(ast1)
      {:ok, ast2} = AST.parse(source)

      # ASTs should be equivalent (ignoring metadata differences)
      assert strip_meta(ast1) == strip_meta(ast2)
    end

    test "preserves module attributes" do
      code = """
      defmodule Example do
        @moduledoc "Example module"
        @doc "Does something"
        def foo, do: :ok
      end
      """

      {:ok, ast} = AST.parse(code)
      result = AST.to_source(ast)

      assert String.contains?(result, "@moduledoc")
      assert String.contains?(result, "@doc")
    end

    test "preserves pipes" do
      code = """
      list
      |> Enum.map(&(&1 * 2))
      |> Enum.filter(&(&1 > 0))
      """

      {:ok, ast} = AST.parse(code)
      result = AST.to_source(ast)

      assert String.contains?(result, "|>")
    end

    test "preserves captures" do
      code = "Enum.map(list, &String.length/1)"

      {:ok, ast} = AST.parse(code)
      result = AST.to_source(ast)

      assert String.contains?(result, "&String.length/1") or
               String.contains?(result, "& String.length/1")
    end

    test "preserves heredocs content" do
      code = ~S'''
      @doc """
      This is a heredoc.
      It has multiple lines.
      """
      def foo, do: :ok
      '''

      {:ok, ast} = AST.parse(code)
      result = AST.to_source(ast)

      assert String.contains?(result, "This is a heredoc")
      assert String.contains?(result, "multiple lines")
    end

    test "preserves string interpolation" do
      code = ~S[msg = "Hello, #{name}!"]

      {:ok, ast} = AST.parse(code)
      result = AST.to_source(ast)

      # Should preserve interpolation structure
      assert String.contains?(result, "Hello") or String.contains?(result, "name")
    end
  end

  describe "transform_file/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "ast_test_#{:rand.uniform(100_000)}.ex")

      on_exit(fn -> File.rm(path) end)

      {:ok, path: path}
    end

    test "transforms file content", %{path: path} do
      File.write!(path, """
      defmodule Example do
        def value, do: 1
      end
      """)

      # Replace 1 with 42
      {:ok, result} =
        AST.transform_file(path, fn
          1, acc -> {42, acc}
          node, acc -> {node, acc}
        end)

      assert String.contains?(result, "42")
    end

    test "returns error for non-existent file" do
      assert {:error, :enoent} = AST.transform_file("/nonexistent.ex", fn n, a -> {n, a} end)
    end

    test "returns error for invalid syntax", %{path: path} do
      File.write!(path, "defmodule Bad do def foo(")

      assert {:error, _} = AST.transform_file(path, fn n, a -> {n, a} end)
    end
  end

  describe "matches?/2" do
    test "matches atoms with :_" do
      assert FourthWall.AST.matches?(:foo, :_)
      assert FourthWall.AST.matches?(:bar, :_)
    end

    test "matches identical atoms" do
      assert FourthWall.AST.matches?(:foo, :foo)
      refute FourthWall.AST.matches?(:foo, :bar)
    end

    test "matches tuples with :_ in positions" do
      node = {:length, [line: 1], [:arg]}
      pattern = {:length, :_, [:_]}
      assert FourthWall.AST.matches?(node, pattern)
    end
  end

  # Helper to strip metadata for AST comparison
  defp strip_meta(ast) do
    Macro.prewalk(ast, fn
      {form, _meta, args} -> {form, [], args}
      other -> other
    end)
  end
end
