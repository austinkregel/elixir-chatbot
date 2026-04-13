defmodule FourthWall.SafetyTest do
  use ExUnit.Case, async: false

  alias FourthWall.Safety

  describe "check_output/1" do
    test "returns :ok for clean code" do
      clean_code = """
      defmodule Example do
        def foo(list) do
          list == []
        end
      end
      """

      assert {:ok, ^clean_code} = Safety.check_output(clean_code)
    end

    test "detects stray digit after empty list ([]0 corruption)" do
      corrupted = """
      defmodule Example do
        def foo(list) do
          list == []0
        end
      end
      """

      assert {:error, issues} = Safety.check_output(corrupted)
      assert length(issues) >= 1
      assert Enum.any?(issues, fn {_pattern, desc} ->
        String.contains?(desc, "stray digit")
      end)
    end

    test "detects []1, []2, etc. variations" do
      for digit <- 0..9 do
        corrupted = "all_responses == []#{digit}"
        assert {:error, _} = Safety.check_output(corrupted),
               "Should detect []#{digit} corruption"
      end
    end

    test "detects corrupted String.length (String.x == [])" do
      corrupted = """
      defmodule Example do
        def foo(s) do
          String.x == []
        end
      end
      """

      assert {:error, issues} = Safety.check_output(corrupted)
      assert Enum.any?(issues, fn {_pattern, desc} ->
        String.contains?(desc, "String.length")
      end)
    end

    test "detects corrupted String.length with !=" do
      corrupted = "String.query != []"

      assert {:error, issues} = Safety.check_output(corrupted)
      assert Enum.any?(issues, fn {_pattern, desc} ->
        String.contains?(desc, "String.length")
      end)
    end

    test "detects corrupted capture operator (String.&1)" do
      corrupted = """
      defmodule Example do
        def foo do
          Enum.filter(list, &(String.&1 != []))
        end
      end
      """

      assert {:error, issues} = Safety.check_output(corrupted)
      assert Enum.any?(issues, fn {_pattern, desc} ->
        String.contains?(desc, "capture")
      end)
    end

    test "detects multiple corruption patterns in same code" do
      corrupted = """
      defmodule Example do
        def foo(list) do
          list == []0
          String.x == []
          String.&1 != []
        end
      end
      """

      assert {:error, issues} = Safety.check_output(corrupted)
      # Should detect all three patterns
      assert length(issues) >= 3
    end

    test "does not flag legitimate empty list comparisons" do
      # These are all valid Elixir code
      valid_examples = [
        "list == []",
        "[] == list",
        "if list != [], do: :ok",
        "String.length(x) == 0",
        "Enum.empty?(list)"
      ]

      for code <- valid_examples do
        assert {:ok, ^code} = Safety.check_output(code),
               "Should not flag: #{code}"
      end
    end

    test "does not flag legitimate String module calls" do
      valid_examples = [
        "String.length(x)",
        "String.trim(s)",
        "String.upcase(s)",
        "String.split(s, \",\")"
      ]

      for code <- valid_examples do
        assert {:ok, ^code} = Safety.check_output(code),
               "Should not flag: #{code}"
      end
    end
  end

  describe "patterns/0" do
    test "returns list of known corruption patterns" do
      patterns = Safety.patterns()
      assert is_list(patterns)
      assert length(patterns) >= 3

      # Each pattern should be a {regex, description} tuple
      for {regex, desc} <- patterns do
        assert %Regex{} = regex
        assert is_binary(desc)
      end
    end
  end

  describe "scan_file/1" do
    setup do
      # Create a temporary file for testing
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "safety_test_#{:rand.uniform(100_000)}.ex")

      on_exit(fn ->
        File.rm(path)
      end)

      {:ok, path: path}
    end

    test "returns :ok for clean file", %{path: path} do
      File.write!(path, """
      defmodule CleanModule do
        def foo, do: :ok
      end
      """)

      assert {:ok, _} = Safety.scan_file(path)
    end

    test "returns error for corrupted file", %{path: path} do
      File.write!(path, """
      defmodule Corrupted do
        def foo(list), do: list == []0
      end
      """)

      assert {:error, issues} = Safety.scan_file(path)
      assert length(issues) >= 1
    end

    test "returns error for non-existent file" do
      assert {:error, :enoent} = Safety.scan_file("/nonexistent/path.ex")
    end

    test "returns error for other file errors", %{path: path} do
      # Create a directory where we expect a file - this causes :eisdir
      dir_path = path <> "_dir"
      File.mkdir_p!(dir_path)
      on_exit(fn -> File.rm_rf!(dir_path) end)

      assert {:error, :eisdir} = Safety.scan_file(dir_path)
    end
  end

  describe "check_output!/1" do
    test "returns code unchanged when clean" do
      clean_code = "list == []"
      assert ^clean_code = Safety.check_output!(clean_code)
    end

    test "raises on corruption" do
      corrupted = "list == []0"

      assert_raise RuntimeError, ~r/Corruption detected/, fn ->
        Safety.check_output!(corrupted)
      end
    end

    test "raises with descriptive message for multiple issues" do
      corrupted = """
      list == []0
      String.x == []
      """

      error = catch_error(Safety.check_output!(corrupted))
      assert %RuntimeError{message: message} = error
      assert String.contains?(message, "stray digit")
      assert String.contains?(message, "String.length")
    end
  end
end
