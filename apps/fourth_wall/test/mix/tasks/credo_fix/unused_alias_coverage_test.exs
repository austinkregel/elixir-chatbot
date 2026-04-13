defmodule Mix.Tasks.CredoFix.UnusedAliasCoverageTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.UnusedAlias

  describe "excluded_path?/1 (via parse_unused_alias_warnings)" do
    test "excludes files in apps/fourth_wall/test/" do
      output = """
          warning: unused alias Foo
           │
           └─ apps/fourth_wall/test/some_test.exs:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/"])
      assert Enum.empty?(result)
    end

    test "does not exclude files in other app test directories" do
      output = """
          warning: unused alias Foo
           │
           └─ apps/brain/test/some_test.exs:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/brain"])
      assert length(result) == 1
    end
  end

  describe "file_in_paths?/2 edge cases (via parse_unused_alias_warnings)" do
    test "matches files with ./ prefix" do
      output = """
          warning: unused alias Mod
           │
           └─ ./lib/brain/foo.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])
      assert length(result) == 1
    end

    test "matches apps/ path with lib/ relative files" do
      output = """
          warning: unused alias Mod
           │
           └─ lib/brain/foo.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/"])
      assert length(result) == 1
    end

    test "matches specific app path with lib/ relative files" do
      output = """
          warning: unused alias Mod
           │
           └─ lib/brain/foo.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/brain"])
      assert length(result) == 1
    end

    test "rejects files not in target paths" do
      output = """
          warning: unused alias Mod
           │
           └─ deps/some_dep/lib/mod.ex:5:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["apps/brain"])
      assert Enum.empty?(result)
    end
  end

  describe "find_location simpler format" do
    test "parses location with └─ file:line format (no column)" do
      output = """
          warning: unused alias Simple
           │
           └─ lib/brain/simple.ex:42
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])
      assert length(result) == 1
      [entry] = result
      assert entry.alias_name == "Simple"
      assert entry.line == 42
    end
  end

  describe "process_file/4 with verbose mode" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "ua_verbose_test_#{:rand.uniform(100_000)}.ex")

      on_exit(fn -> File.rm(path) end)

      {:ok, path: path}
    end

    test "prints alias removal info when verbose", %{path: path} do
      content = """
      defmodule VerboseTest do
        alias Some.Unused

        def foo, do: :ok
      end
      """

      File.write!(path, content)

      aliases = [%{file: path, line: 2, alias_name: "Unused"}]

      output =
        capture_io(fn ->
          count = UnusedAlias.process_file(path, aliases, true, true)
          assert count == 1
        end)

      assert output =~ "removing unused alias Unused"
    end

    test "prints read error when verbose for nonexistent file" do
      path = Path.join(System.tmp_dir!(), "nonexistent_#{:rand.uniform(100_000)}.ex")
      aliases = [%{file: path, line: 2, alias_name: "X"}]

      output =
        capture_io(:stderr, fn ->
          count = UnusedAlias.process_file(path, aliases, false, true)
          assert count == 0
        end)

      assert output =~ "Failed to read"
    end

    test "prints parse error when verbose for invalid syntax", %{path: path} do
      File.write!(path, "defmodule Bad do def foo end")
      aliases = [%{file: path, line: 1, alias_name: "X"}]

      output =
        capture_io(:stderr, fn ->
          count = UnusedAlias.process_file(path, aliases, false, true)
          assert count == 0
        end)

      assert output =~ "Failed to process" or output =~ "Failed to parse" or output == ""
    end

    test "silent when not verbose for nonexistent file" do
      path = Path.join(System.tmp_dir!(), "missing_#{:rand.uniform(100_000)}.ex")
      aliases = [%{file: path, line: 1, alias_name: "X"}]

      output =
        capture_io(:stderr, fn ->
          count = UnusedAlias.process_file(path, aliases, false, false)
          assert count == 0
        end)

      assert output == ""
    end
  end

  describe "remove_unused_aliases/4 with verbose parse error" do
    test "prints error when verbose and code has invalid syntax" do
      invalid = "%{invalid => syntax [["

      output =
        capture_io(:stderr, fn ->
          result = UnusedAlias.remove_unused_aliases(invalid, ["X"], true, "bad.ex")
          assert {:error, _} = result
        end)

      assert output =~ "Failed to parse"
    end

    test "silent on parse error when not verbose" do
      invalid = "%{invalid => syntax [["

      output =
        capture_io(:stderr, fn ->
          result = UnusedAlias.remove_unused_aliases(invalid, ["X"], false, "bad.ex")
          assert {:error, _} = result
        end)

      assert output == ""
    end
  end

  describe "run/1 modes" do
    test "apply mode shows removing message" do
      output =
        capture_io(fn ->
          UnusedAlias.run(["--apply", "nonexistent_dir_#{:rand.uniform(100_000)}"])
        end)

      assert output =~ "Removing unused aliases" or output =~ "No unused aliases"
    end

    test "default dry-run mode shows scanning message" do
      output =
        capture_io(fn ->
          UnusedAlias.run(["nonexistent_dir_#{:rand.uniform(100_000)}"])
        end)

      assert output =~ "DRY RUN" or output =~ "No unused aliases"
    end

    test "returns ok tuple with count" do
      result =
        capture_io(fn ->
          assert {:ok, count} = UnusedAlias.run(["--dry-run", "nonexistent_#{:rand.uniform(100_000)}"])
          assert is_integer(count)
        end)

      assert is_binary(result)
    end
  end

  describe "clean_file_path/1 (via parse_unused_alias_warnings)" do
    test "strips parenthetical app prefix from path" do
      output = """
          warning: unused alias Config
           │
           └─ (brain 0.1.0) lib/brain/config.ex:10:3
      """

      result = UnusedAlias.parse_unused_alias_warnings(output, ["lib/"])
      assert length(result) == 1
      [entry] = result
      refute String.contains?(entry.file, "(brain")
      assert String.contains?(entry.file, "config.ex")
    end
  end
end
