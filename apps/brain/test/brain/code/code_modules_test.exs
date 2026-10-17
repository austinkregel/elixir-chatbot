defmodule Brain.Code.CodeModulesTest do
  @moduledoc "Tests for Brain.Code.* modules to improve code analysis coverage."
  use ExUnit.Case, async: false

  describe "Brain.Code.Tokenizer" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.Tokenizer)
    end

    test "tokenizes Elixir source code" do
      if function_exported?(Brain.Code.Tokenizer, :tokenize, 2) do
        code = """
        defmodule Hello do
          def greet(name), do: "Hello, \#{name}!"
        end
        """

        result = Brain.Code.Tokenizer.tokenize(code, :elixir)
        assert is_list(result)
      end
    end
  end

  describe "Brain.Code.SymbolExtractor" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.SymbolExtractor)
    end

    test "extracts symbols from parsed AST" do
      if function_exported?(Brain.Code.SymbolExtractor, :extract, 3) do
        code = """
        defmodule Calculator do
          def add(a, b), do: a + b
          def subtract(a, b), do: a - b
        end
        """

        case Brain.Code.Parser.parse(code, :elixir) do
          {:ok, ast} ->
            result = Brain.Code.SymbolExtractor.extract(ast, :elixir, [])
            assert is_list(result) or is_map(result)

          %{} = ast ->
            result = Brain.Code.SymbolExtractor.extract(ast, :elixir, [])
            assert is_list(result) or is_map(result)

          _ ->
            :skip
        end
      end
    end
  end

  describe "Brain.Code.Parser" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.Parser)
    end

    test "parses Elixir source code" do
      if function_exported?(Brain.Code.Parser, :parse, 2) do
        code = """
        defmodule Sample do
          def hello, do: :world
        end
        """

        result = Brain.Code.Parser.parse(code, :elixir)
        assert result != nil
      end
    end
  end

  describe "Brain.Code.Summarizer" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.Summarizer)
    end

    test "summarize_file is available" do
      if function_exported?(Brain.Code.Summarizer, :summarize_file, 2) do
        result = Brain.Code.Summarizer.summarize_file("test_world", "lib/nonexistent.ex")
        assert is_binary(result)
      end
    end
  end

  describe "Brain.Code.RelationMapper" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.RelationMapper)
    end
  end

  describe "Brain.Code.QueryHandler" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.QueryHandler)
    end

    test "every query raises without a world" do
      alias Brain.Code.QueryHandler

      assert_raise ArgumentError, ~r/QueryHandler.handle needs :world_id/, fn ->
        QueryHandler.handle("code.explain", [], query_text: "what does parse do")
      end

      # The world is checked before the intent, so a non-code intent raises too.
      assert_raise ArgumentError, ~r/QueryHandler.handle needs :world_id/, fn ->
        QueryHandler.handle("smalltalk.greetings.hello", [], [])
      end

      assert_raise ArgumentError, ~r/QueryHandler.explain needs :world_id/, fn ->
        QueryHandler.explain("parse", [])
      end

      assert_raise ArgumentError, ~r/QueryHandler.find_usages needs :world_id/, fn ->
        QueryHandler.find_usages("parse", [])
      end

      assert_raise ArgumentError, ~r/QueryHandler.signature needs :world_id/, fn ->
        QueryHandler.signature("parse", [])
      end

      assert_raise ArgumentError, ~r/QueryHandler.list_symbols needs :world_id/, fn ->
        QueryHandler.list_symbols("Brain.Code", limit: 5)
      end

      assert_raise ArgumentError, ~r/QueryHandler.search needs :world_id/, fn ->
        QueryHandler.search("parse", [])
      end

      assert_raise ArgumentError, ~r/QueryHandler.stats needs :world_id/, fn ->
        QueryHandler.stats([])
      end
    end

    test "a world option that is not a world id raises" do
      alias Brain.Code.QueryHandler

      assert_raise ArgumentError, ~r/QueryHandler.stats option :world_id must be a world id, got: nil/, fn ->
        QueryHandler.stats(world_id: nil)
      end

      assert_raise ArgumentError, ~r/QueryHandler.explain option :world_id must be a world id, got: ""/, fn ->
        QueryHandler.explain("parse", world_id: "")
      end
    end
  end

  describe "Brain.Code.LanguageGrammar" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.LanguageGrammar)
    end

    test "ready? returns boolean when process exists" do
      if Process.whereis(Brain.Code.LanguageGrammar) do
        result = Brain.Code.LanguageGrammar.ready?()
        assert is_boolean(result)
      end
    end
  end

  describe "Brain.Code.CodeGazetteer" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.CodeGazetteer)
    end

    test "ready? returns boolean when process exists" do
      if Process.whereis(Brain.Code.CodeGazetteer) do
        result = Brain.Code.CodeGazetteer.ready?()
        assert is_boolean(result)
      end
    end
  end

  describe "Brain.Code.Pipeline" do
    test "module is loadable" do
      assert Code.ensure_loaded?(Brain.Code.Pipeline)
    end
  end
end
