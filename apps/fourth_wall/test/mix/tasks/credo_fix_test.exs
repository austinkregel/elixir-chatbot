defmodule Mix.Tasks.CredoFixTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix

  describe "run/1 argument parsing" do
    test "defaults to dry-run mode" do
      output = capture_io(fn ->
        CredoFix.run(["--only", "trailing_whitespace", "nonexistent_path/"])
      end)

      assert output =~ "[DRY RUN]"
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
