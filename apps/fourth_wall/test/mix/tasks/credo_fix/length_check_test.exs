defmodule Mix.Tasks.CredoFix.LengthCheckTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.LengthCheck

  describe "run/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "lc_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "runs in dry-run mode by default", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "if length(x) == 0, do: :empty\n")

      output = capture_io(fn -> LengthCheck.run([test_dir]) end)

      # File should NOT be modified
      assert File.read!(path) == "if length(x) == 0, do: :empty\n"
      assert output =~ "[DRY RUN]"
      assert output =~ "Would fix"
    end

    test "applies changes with --apply flag", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "if length(x) == 0, do: :empty\n")

      output = capture_io(fn -> LengthCheck.run(["--apply", test_dir]) end)

      # File SHOULD be modified
      content = File.read!(path)
      assert String.contains?(content, "x == []")
      assert output =~ "Fixed"
    end

    test "verbose mode shows details", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "if length(x) == 0, do: :empty\n")

      output = capture_io(fn -> LengthCheck.run(["--verbose", test_dir]) end)

      assert output =~ "length check"
    end

    test "respects --style=enum_empty option", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "if length(x) == 0, do: :empty\n")

      output = capture_io(fn -> LengthCheck.run(["--apply", "--style=enum_empty", test_dir]) end)

      # File SHOULD be modified with Enum.empty?
      content = File.read!(path)
      assert String.contains?(content, "Enum.empty?")
      assert output =~ "Fixed"
    end

    test "handles directory with no elixir files" do
      output = capture_io(fn -> LengthCheck.run(["nonexistent_path/"]) end)

      assert output =~ "No Elixir files found"
    end
  end

  describe "process_file/4" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "lc_test_#{:rand.uniform(100_000)}.ex")
      on_exit(fn -> File.rm(path) end)
      {:ok, path: path}
    end

    test "processes file with length checks (dry-run)", %{path: path} do
      File.write!(path, "if length(x) == 0, do: :empty\n")

      # Non-verbose mode produces no output
      {result_path, changes} = LengthCheck.process_file(path, true, false, "pattern_match")

      assert result_path == path
      assert changes == 1
      # File should NOT be modified in dry-run
      assert File.read!(path) == "if length(x) == 0, do: :empty\n"
    end

    test "processes file with length checks (apply)", %{path: path} do
      File.write!(path, "if length(x) == 0, do: :empty\n")

      # Non-verbose mode produces no output
      {result_path, changes} = LengthCheck.process_file(path, false, false, "pattern_match")

      assert result_path == path
      assert changes == 1
      # File SHOULD be modified
      content = File.read!(path)
      assert String.contains?(content, "x == []")
    end

    test "processes file with verbose mode", %{path: path} do
      File.write!(path, "if length(x) == 0, do: :empty\n")

      output = capture_io(fn ->
        {result_path, changes} = LengthCheck.process_file(path, true, true, "pattern_match")
        assert result_path == path
        assert changes == 1
      end)

      assert output =~ "length check"
    end

    test "handles file with no changes", %{path: path} do
      File.write!(path, "if x == [], do: :empty\n")

      {result_path, changes} = LengthCheck.process_file(path, false, false, "pattern_match")

      assert result_path == path
      assert changes == 0
    end

    test "handles non-existent file" do
      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        {_path, changes} = LengthCheck.process_file("/nonexistent/file.ex", true, false, "pattern_match")
        assert changes == 0
      end)

      assert output =~ "Failed to read"
    end
  end

  # =============================================================================
  # Basic transformations (pattern_match style)
  # Note: AST transformation normalizes formatting, so we check for the
  # semantically correct transformation rather than exact string matching
  # =============================================================================

  describe "fix_length_checks/4 with pattern_match style" do
    test "fixes length(x) == 0" do
      input = "if length(items) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # AST transformation may reformat, but the result should have the correct pattern
      assert changes == 1
      assert String.contains?(output, "items == []")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) > 0" do
      input = "if length(items) > 0, do: :has_items"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "items != []")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) != 0" do
      input = "if length(list) != 0, do: :ok"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "list != []")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) >= 1 in function" do
      # Use a valid context for the guard
      input = "def foo(x) when length(x) >= 1, do: x"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "x != []")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) < 1" do
      input = "if length(list) < 1, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "list == []")
      refute String.contains?(output, "length(")
    end

    test "fixes 0 == length(x)" do
      input = "if 0 == length(items), do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "items == []")
      refute String.contains?(output, "length(")
    end

    test "fixes 0 < length(x)" do
      input = "if 0 < length(items), do: :has_items"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "items != []")
      refute String.contains?(output, "length(")
    end

    test "fixes 1 <= length(x)" do
      input = "if 1 <= length(items), do: :has_items"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "items != []")
      refute String.contains?(output, "length(")
    end
  end

  # =============================================================================
  # Enum.empty? style transformations
  # =============================================================================

  describe "fix_length_checks/4 with enum_empty style" do
    test "fixes length(x) == 0 with Enum.empty?" do
      input = "if length(items) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "enum_empty")

      assert changes == 1
      assert String.contains?(output, "Enum.empty?(items)")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) > 0 with not Enum.empty?" do
      input = "if length(items) > 0, do: :has_items"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "enum_empty")

      assert changes == 1
      assert String.contains?(output, "Enum.empty?(items)")
      assert String.contains?(output, "not ")
      refute String.contains?(output, "length(")
    end

    test "fixes length(x) != 0 with not Enum.empty?" do
      input = "if length(items) != 0, do: :ok"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "enum_empty")

      assert changes == 1
      assert String.contains?(output, "Enum.empty?(items)")
      assert String.contains?(output, "not ")
      refute String.contains?(output, "length(")
    end
  end

  # =============================================================================
  # CRITICAL: Must NOT corrupt code - regression tests for prior bugs
  # =============================================================================

  describe "corruption prevention - must NOT corrupt code" do
    test "does not leave trailing characters after replacement (the []0 bug)" do
      # This was a REAL bug: length(x) == 0 became x == []0
      input = "if length(all_responses) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "all_responses == []")

      # Verify no stray characters - this is the actual regression test
      refute String.contains?(output, "[]0"), "Found []0 corruption"
      refute String.contains?(output, "[]1"), "Found []1 corruption"
    end

    test "does not corrupt String.length expressions" do
      # This was a REAL bug: String.length(x) != 0 became String.x != []
      input = "if String.length(str) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # String.length is NOT the same as length/1 - must not be changed
      assert changes == 0
      assert String.contains?(output, "String.length(str)")

      # Critical: must not corrupt into nonsense
      refute String.contains?(output, "String.str"), "Found String.x corruption"
      refute String.contains?(output, "String. =="), "Found corrupted String."
    end

    test "does not corrupt capture operators" do
      # This was a REAL bug: &(String.length(&1) != 0) got corrupted
      input = "Enum.filter(list, &(String.length(&1) != 0))"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # String.length should not be touched
      assert changes == 0
      assert String.contains?(output, "String.length")

      # Critical corruption patterns to check for
      refute String.contains?(output, "String.&1"), "Found String.&1 corruption"
    end

    test "does transform bare length in capture" do
      # Bare length(&1) in a capture SHOULD be transformed since it's bare length
      input = "Enum.filter(list, &(length(&1) == 0))"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "&1 == []") or String.contains?(output, "== []")
      refute String.contains?(output, "[]0"), "Found []0 corruption"
    end

    test "output never contains []<digit> pattern" do
      inputs = [
        "if length(x) == 0, do: :ok",
        "if length(x) > 0, do: :ok",
        "if length(x) != 0, do: :ok",
        "def f(x) when length(x) >= 1, do: x",
        "if length(x) < 1, do: :ok",
        "if 0 == length(x), do: :ok",
        "if 0 < length(x), do: :ok",
        "if 1 <= length(x), do: :ok"
      ]

      for input <- inputs do
        {output, _} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

        # Safety check: no corruption patterns
        refute Regex.match?(~r/\[\]\d/, output),
               "Found []<digit> corruption in output for input: #{input}"
      end
    end

    test "uses Safety module to verify output" do
      # The fixer should use FourthWall.Safety.check_output/1 internally
      input = "if length(list) == 0, do: :empty"
      {output, _} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # The output should pass safety checks
      assert {:ok, _} = FourthWall.Safety.check_output(output)
    end
  end

  # =============================================================================
  # Edge cases that must NOT be modified
  # =============================================================================

  describe "patterns that must NOT be modified" do
    test "does not modify String.length" do
      input = "if String.length(str) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "does not modify byte_size" do
      input = "if byte_size(data) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "does not modify tuple_size" do
      input = "if tuple_size(t) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "does not modify map_size" do
      input = "if map_size(m) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "does not modify Enum.count" do
      # Enum.count has different semantics - handled differently by Credo
      input = "if Enum.count(list) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "does not modify length comparisons with non-zero values" do
      inputs = [
        "if length(items) == 5, do: :five",
        "if length(items) > 1, do: :multiple",
        "if length(items) >= 2, do: :at_least_two",
        "if length(items) < 3, do: :few",
        "if length(items) <= 10, do: :limited"
      ]

      for input <- inputs do
        {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")
        assert output == input, "Should not modify: #{input}"
        assert changes == 0
      end
    end

    test "skips comment lines" do
      input = "# if length(items) == 0"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end

    test "skips code inside string literals" do
      input = ~S[msg = "length(items) == 0 is bad"]
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert output == input
      assert changes == 0
    end
  end

  # =============================================================================
  # Complex expressions
  # =============================================================================

  describe "complex expressions" do
    test "handles variable names with underscores" do
      input = "if length(all_responses) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "all_responses == []")
      refute String.contains?(output, "length(")
    end

    test "handles module-qualified variables" do
      input = "if length(state.items) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "state.items == []")
      refute String.contains?(output, "length(")
    end

    test "handles nested access" do
      input = "if length(map.nested.list) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "map.nested.list == []")
      refute String.contains?(output, "length(")
    end

    test "handles length check followed by other code on same line" do
      input = "if length(list) == 0, do: handle_empty(), else: process(list)"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "list == []")
      refute String.contains?(output, "length(")
    end

    test "handles multiple length checks in same file" do
      input = """
      def empty?(list) do
        length(list) == 0
      end

      def has_items?(list) do
        length(list) > 0
      end
      """

      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 2
      assert String.contains?(output, "list == []")
      assert String.contains?(output, "list != []")
      refute String.contains?(output, "length(")
    end

    test "handles function call as argument" do
      input = "if length(get_list()) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      assert changes == 1
      assert String.contains?(output, "get_list() == []")
      refute String.contains?(output, "length(")
    end

    test "handles pipe as argument" do
      input = "if length(data |> filter()) == 0, do: :empty"
      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # The pipe expression should be preserved
      assert changes == 1
      assert String.contains?(output, "== []")
      refute String.contains?(output, "length(")
    end
  end

  # =============================================================================
  # Round-trip safety
  # =============================================================================

  describe "round-trip safety" do
    test "output is valid Elixir syntax" do
      # Use complete, valid Elixir expressions
      inputs = [
        "if length(x) == 0, do: :ok",
        "if length(list) > 0, do: :ok",
        "def f(x) when length(x) >= 1, do: x",
        "if 0 == length(items), do: :ok"
      ]

      for input <- inputs do
        {output, _} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

        # Try to parse the output - it should be valid Elixir
        case Code.string_to_quoted(output) do
          {:ok, _ast} ->
            :ok

          {:error, reason} ->
            flunk("Output is not valid Elixir syntax for input '#{input}': #{inspect(reason)}")
        end
      end
    end

    test "full module transformation produces valid code" do
      input = """
      defmodule Example do
        def empty?(list) do
          length(list) == 0
        end

        def has_items?(list) do
          length(list) > 0
        end

        # String.length should not be touched
        def blank?(str) do
          String.length(str) == 0
        end
      end
      """

      {output, changes} = LengthCheck.fix_length_checks(input, false, "test.ex", "pattern_match")

      # Should have transformed 2 length calls (not String.length)
      assert changes == 2

      # Output should be valid Elixir
      assert {:ok, _} = Code.string_to_quoted(output)

      # String.length should be preserved
      assert String.contains?(output, "String.length")
    end
  end
end
