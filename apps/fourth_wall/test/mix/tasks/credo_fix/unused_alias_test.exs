defmodule Mix.Tasks.CredoFix.UnusedAliasTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.UnusedAlias

  describe "parse_unused_alias_warnings/2" do
    test "parses single unused alias warning" do
      output = """
          warning: unused alias Analysis
           │
        11 │   alias Brain.Analysis
           │   ~
           │
           └─ lib/brain/analysis/outcome_learner.ex:4:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Analysis"
      assert entry.line == 4
      assert String.contains?(entry.file, "outcome_learner.ex")
    end

    test "parses multiple unused alias warnings" do
      output = """
          warning: unused alias Analysis
           │
        11 │   alias Brain.Analysis
           │   ~
           │
           └─ lib/brain/analysis/response_gate.ex:4:3

          warning: unused alias Types
           │
        10 │   alias Brain.Epistemic.Types
           │   ~
           │
           └─ lib/brain/epistemic/disclosure_policy.ex:6:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert length(result) == 2
      alias_names = Enum.map(result, & &1.alias_name) |> Enum.sort()
      assert alias_names == ["Analysis", "Types"]
    end

    test "filters warnings by paths" do
      output = """
          warning: unused alias Foo
           │
           └─ apps/brain/lib/foo.ex:5:3

          warning: unused alias Bar
           │
           └─ apps/world/lib/bar.ex:10:3
      """

      # Only apps/brain should match
      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/brain"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Foo"
    end

    test "ignores non-alias warnings" do
      output = """
          warning: variable x is unused
           │
           └─ lib/foo.ex:5:3

          warning: unused import Enum
           │
           └─ lib/bar.ex:10:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert Enum.empty?(result)
    end

    test "handles app-prefixed paths" do
      output = """
          warning: unused alias Tokenizer
           │
        10 │   alias Brain.ML.Tokenizer
           │   ~
           │
           └─ (brain 0.1.0) lib/brain/ml/micro_classifiers.ex:153:7
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Tokenizer"
      assert String.contains?(entry.file, "micro_classifiers.ex")
    end

    test "excludes paths containing apps/fourth_wall/test/" do
      output = """
          warning: unused alias TestHelper
           │
           └─ apps/fourth_wall/test/support/some_helper.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/"])

      assert result == []
    end

    test "file_in_paths? matches path with ./ prefix against lib/" do
      output = """
          warning: unused alias Foo
           │
           └─ ./lib/brain/foo.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Foo"
    end

    test "file_in_paths? matches lib/ path when path is apps/ catch-all" do
      output = """
          warning: unused alias Bar
           │
           └─ lib/brain/foo.ex:10:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Bar"
    end

    test "file_in_paths? matches lib/ path when path is specific app apps/brain" do
      output = """
          warning: unused alias Baz
           │
           └─ lib/brain/foo.ex:7:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/brain"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Baz"
    end

    test "find_location parses simpler format without column (file:line)" do
      output = """
          warning: unused alias Simple
           │
           └─ lib/simple.ex:12
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])

      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Simple"
      assert entry.line == 12
    end
  end

  describe "remove_unused_aliases/4" do
    test "removes simple unused alias" do
      input = """
      defmodule Foo do
        alias Some.Module
        alias Other.Thing

        def bar, do: Thing.call()
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["Module"], false, "test.ex")

      assert count == 1
      refute String.contains?(output, "alias Some.Module")
      assert String.contains?(output, "alias Other.Thing")
    end

    test "removes multiple unused aliases" do
      input = """
      defmodule Foo do
        alias A.First
        alias B.Second
        alias C.Third

        def bar, do: :ok
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["First", "Second", "Third"], false, "test.ex")

      assert count == 3
      refute String.contains?(output, "alias A.First")
      refute String.contains?(output, "alias B.Second")
      refute String.contains?(output, "alias C.Third")
    end

    test "preserves used aliases" do
      input = """
      defmodule Foo do
        alias A.First
        alias B.Second

        def bar, do: Second.call()
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["First"], false, "test.ex")

      assert count == 1
      refute String.contains?(output, "alias A.First")
      assert String.contains?(output, "alias B.Second")
    end

    test "handles grouped aliases - removes one" do
      input = """
      defmodule Foo do
        alias Brain.Analysis.{First, Second, Third}

        def bar do
          Second.call()
          Third.call()
        end
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["First"], false, "test.ex")

      assert count == 1
      # Should still have grouped alias but without First
      assert String.contains?(output, "alias Brain.Analysis.{")
      refute String.contains?(output, "First")
      assert String.contains?(output, "Second")
      assert String.contains?(output, "Third")
    end

    test "handles grouped aliases - removes multiple leaving one" do
      input = """
      defmodule Foo do
        alias Brain.Analysis.{First, Second, Third}

        def bar, do: Third.call()
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["First", "Second"], false, "test.ex")

      assert count == 2
      # Should convert to simple alias since only one remains
      assert String.contains?(output, "alias Brain.Analysis.Third")
      refute String.contains?(output, "{")
    end

    test "handles grouped aliases - removes all" do
      input = """
      defmodule Foo do
        alias Brain.Analysis.{First, Second}

        def bar, do: :ok
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["First", "Second"], false, "test.ex")

      assert count == 2
      refute String.contains?(output, "alias Brain.Analysis")
    end

    test "handles alias with :as option" do
      input = """
      defmodule Foo do
        alias Very.Long.Module.Name, as: Short

        def bar, do: :ok
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["Short"], false, "test.ex")

      assert count == 1
      refute String.contains?(output, "alias Very.Long.Module.Name")
    end

    test "returns 0 count when no aliases match" do
      input = """
      defmodule Foo do
        alias Some.Module

        def bar, do: Module.call()
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["NonExistent"], false, "test.ex")

      assert count == 0
      assert output == input
    end

    test "handles code without aliases" do
      input = """
      defmodule Foo do
        def bar, do: :ok
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["Module"], false, "test.ex")

      assert count == 0
      assert output == input
    end

    test "preserves module structure and formatting" do
      input = """
      defmodule Foo do
        @moduledoc "Some docs"

        alias Used.Module
        alias Unused.Thing

        @some_attr :value

        def bar, do: Module.call()
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["Thing"], false, "test.ex")

      assert count == 1
      assert String.contains?(output, "@moduledoc")
      assert String.contains?(output, "alias Used.Module")
      assert String.contains?(output, "@some_attr :value")
      assert String.contains?(output, "def bar")
    end
  end

  describe "process_file/4" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "ua_test_#{:rand.uniform(100_000)}.ex")

      on_exit(fn ->
        File.rm(path)
      end)

      {:ok, path: path, tmp_dir: tmp_dir}
    end

    test "removes unused aliases in dry-run mode without modifying file", %{path: path} do
      content = """
      defmodule Test do
        alias Some.Unused

        def foo, do: :ok
      end
      """

      File.write!(path, content)

      aliases = [%{file: path, line: 2, alias_name: "Unused"}]
      count = UnusedAlias.process_file(path, aliases, true, false)

      assert count == 1
      # File should be unchanged in dry-run
      assert File.read!(path) == content
    end

    test "removes unused aliases when apply mode is on", %{path: path} do
      content = """
      defmodule Test do
        alias Some.Unused

        def foo, do: :ok
      end
      """

      File.write!(path, content)

      aliases = [%{file: path, line: 2, alias_name: "Unused"}]
      count = UnusedAlias.process_file(path, aliases, false, false)

      assert count == 1
      new_content = File.read!(path)
      refute String.contains?(new_content, "alias Some.Unused")
    end

    test "handles non-existent file gracefully", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "nonexistent.ex")
      aliases = [%{file: path, line: 2, alias_name: "Unused"}]

      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        count = UnusedAlias.process_file(path, aliases, false, true)
        assert count == 0
      end)

      assert output =~ "Failed to read"
    end

    test "handles parse errors gracefully", %{path: path} do
      # Invalid Elixir syntax
      File.write!(path, "defmodule Test do def foo end")

      aliases = [%{file: path, line: 1, alias_name: "Unused"}]

      # Note: parse error may go to stderr or be silent
      _output = capture_io(:stderr, fn ->
        count = UnusedAlias.process_file(path, aliases, false, true)
        assert count == 0
      end)

      # Test passes if it doesn't crash - parse error handling is internal
    end

    test "verbose mode prints each alias being removed", %{path: path} do
      content = """
      defmodule Test do
        alias Some.Unused

        def foo, do: :ok
      end
      """

      File.write!(path, content)
      aliases = [%{file: path, line: 2, alias_name: "Unused"}]

      output = capture_io(fn ->
        count = UnusedAlias.process_file(path, aliases, false, true)
        assert count == 1
      end)

      assert output =~ "removing unused alias Unused"
      assert output =~ path
    end

    test "remove_unused_aliases parse error with verbose prints error message" do
      invalid_content = "defmodule Broken { invalid syntax"
      alias_names = ["Unused"]

      output = capture_io(:stderr, fn ->
        result = UnusedAlias.remove_unused_aliases(invalid_content, alias_names, true, "broken.ex")
        assert match?({:error, _}, result)
      end)

      assert output =~ "Failed to parse"
      assert output =~ "broken.ex"
    end

    test "read error without verbose returns 0 silently", %{tmp_dir: tmp_dir} do
      path = Path.join(tmp_dir, "nonexistent.ex")
      aliases = [%{file: path, line: 2, alias_name: "Unused"}]

      stderr = capture_io(:stderr, fn ->
        count = UnusedAlias.process_file(path, aliases, false, false)
        assert count == 0
      end)

      assert stderr == ""
    end
  end

  describe "run/1 integration" do
    # These tests use async: false due to Mix.shell and filesystem operations
    @describetag :capture_log

    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "ua_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "displays help message in dry-run mode" do
      output = capture_io(fn ->
        assert {:ok, _count} = UnusedAlias.run(["--dry-run"])
      end)

      assert output =~ "DRY RUN" or output =~ "Scanning"
    end
  end

  describe "edge cases" do
    test "handles deeply nested module aliases" do
      input = """
      defmodule Foo do
        alias Very.Deeply.Nested.Module.Path

        def bar, do: :ok
      end
      """

      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, ["Path"], false, "test.ex")

      assert count == 1
      refute String.contains?(output, "alias Very.Deeply")
    end

    test "handles multiple grouped alias statements" do
      input = """
      defmodule Foo do
        alias Brain.Analysis.{First, Second}
        alias Brain.ML.{Third, Fourth}

        def bar do
          Second.call()
          Fourth.call()
        end
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["First", "Third"], false, "test.ex")

      assert count == 2
      # Both groups should be converted to simple aliases
      assert String.contains?(output, "alias Brain.Analysis.Second")
      assert String.contains?(output, "alias Brain.ML.Fourth")
    end

    test "handles mixed simple and grouped aliases" do
      input = """
      defmodule Foo do
        alias Simple.One
        alias Brain.Analysis.{First, Second}
        alias Another.Simple

        def bar, do: Second.call()
      end
      """

      {:ok, output, count} =
        UnusedAlias.remove_unused_aliases(input, ["One", "First", "Simple"], false, "test.ex")

      assert count == 3
      refute String.contains?(output, "alias Simple.One")
      refute String.contains?(output, "alias Another.Simple")
      assert String.contains?(output, "alias Brain.Analysis.Second")
    end

    test "does not remove alias if name appears in code (false negative protection)" do
      # This tests that we only remove what the compiler told us is unused
      # The fixer relies on compiler output, not its own analysis
      input = """
      defmodule Foo do
        alias Some.Module

        def bar, do: Module.call()
      end
      """

      # Empty list means compiler didn't flag anything
      {:ok, output, count} = UnusedAlias.remove_unused_aliases(input, [], false, "test.ex")

      assert count == 0
      assert String.contains?(output, "alias Some.Module")
    end
  end
end
