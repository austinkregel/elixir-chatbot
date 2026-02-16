defmodule Mix.Tasks.CredoFixTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix

  describe "module availability" do
    test "all fixer modules are compiled and loadable" do
      fixer_modules = [
        Mix.Tasks.CredoFix.LargeNumbers,
        Mix.Tasks.CredoFix.TrailingWhitespace,
        Mix.Tasks.CredoFix.AliasUsage,
        Mix.Tasks.CredoFix.LengthCheck,
        Mix.Tasks.CredoFix.MapJoin,
        Mix.Tasks.CredoFix.UnusedAlias
      ]

      for module <- fixer_modules do
        assert Code.ensure_loaded?(module), "Fixer module #{inspect(module)} should be available"
      end
    end
  end

  describe "run/1 argument parsing" do
    test "defaults to dry-run mode" do
      output = capture_io(fn ->
        CredoFix.run(["--only", "trailing_whitespace", "nonexistent_path/"])
      end)

      assert output =~ "DRY RUN"
    end

    test "accepts --apply flag" do
      output = capture_io(fn ->
        CredoFix.run(["--apply", "--only", "trailing_whitespace", "nonexistent_path/"])
      end)

      assert output =~ "Running Credo fixers"
    end

    test "accepts --verbose flag" do
      output = capture_io(fn ->
        CredoFix.run(["--verbose", "--only", "trailing_whitespace", "nonexistent_path/"])
      end)

      assert output =~ "Trailing Whitespace"
    end

    test "accepts --exclude flag" do
      output = capture_io(fn ->
        CredoFix.run(["--exclude", "alias_usage", "nonexistent_path/"])
      end)

      assert output =~ "Running Credo fixers"
    end
  end
end
