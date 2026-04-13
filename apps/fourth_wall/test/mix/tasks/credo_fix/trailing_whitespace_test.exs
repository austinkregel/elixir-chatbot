defmodule Mix.Tasks.CredoFix.TrailingWhitespaceTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.TrailingWhitespace

  describe "run/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "tw_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "runs in dry-run mode by default", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "hello  \nworld\n")

      output = capture_io(fn -> TrailingWhitespace.run([test_dir]) end)

      # File should NOT be modified
      assert File.read!(path) == "hello  \nworld\n"
      assert output =~ "[DRY RUN]"
      assert output =~ "Would fix"
    end

    test "applies changes with --apply flag", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "hello  \nworld\n")

      output = capture_io(fn -> TrailingWhitespace.run(["--apply", test_dir]) end)

      # File SHOULD be modified
      assert File.read!(path) == "hello\nworld\n"
      assert output =~ "Fixed"
    end

    test "verbose mode shows details", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "hello  \nworld\n")

      output = capture_io(fn -> TrailingWhitespace.run(["--verbose", test_dir]) end)

      assert output =~ "test.ex"
    end

    test "list-files mode lists files with issues", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "hello  \nworld\n")

      output = capture_io(fn ->
        result = TrailingWhitespace.run(["--list-files", test_dir])
        assert is_map(result)
        assert length(result.files_with_issues) == 1
      end)

      assert output =~ "test.ex"
    end

    test "report-clean mode shows clean files", %{test_dir: test_dir} do
      path = Path.join(test_dir, "clean.ex")
      File.write!(path, "hello\nworld\n")

      output = capture_io(fn -> TrailingWhitespace.run(["--report-clean", test_dir]) end)

      assert output =~ "no trailing whitespace"
    end

    test "handles directory with no elixir files" do
      output = capture_io(fn -> TrailingWhitespace.run(["nonexistent_path/"]) end)

      assert output =~ "No Elixir files found"
    end

    test "handles glob patterns", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "hello  \n")

      output = capture_io(fn -> TrailingWhitespace.run([Path.join(test_dir, "*.ex")]) end)

      assert output =~ "DRY RUN" or output =~ "Scanning"
    end
  end

  describe "process_file/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "tw_test_#{:rand.uniform(100_000)}.ex")
      on_exit(fn -> File.rm(path) end)
      {:ok, path: path}
    end

    test "processes file with trailing whitespace (dry-run)", %{path: path} do
      File.write!(path, "hello  \nworld\n")

      # Non-verbose mode produces no output
      {result_path, changes} = TrailingWhitespace.process_file(path, true, false)

      assert result_path == path
      assert changes == 1
      # File should NOT be modified in dry-run
      assert File.read!(path) == "hello  \nworld\n"
    end

    test "processes file with trailing whitespace (apply)", %{path: path} do
      File.write!(path, "hello  \nworld\n")

      {result_path, changes} = TrailingWhitespace.process_file(path, false, false)

      assert result_path == path
      assert changes == 1
      # File SHOULD be modified when not dry-run
      assert File.read!(path) == "hello\nworld\n"
    end

    test "processes file with verbose mode", %{path: path} do
      File.write!(path, "hello  \nworld\n")

      output = capture_io(fn ->
        {result_path, changes} = TrailingWhitespace.process_file(path, true, true)
        assert result_path == path
        assert changes == 1
      end)

      assert output =~ "1 lines"
    end

    test "handles file with no changes", %{path: path} do
      File.write!(path, "hello\nworld\n")

      {result_path, changes} = TrailingWhitespace.process_file(path, false, false)

      assert result_path == path
      assert changes == 0
    end

    test "handles non-existent file" do
      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        {_path, changes} = TrailingWhitespace.process_file("/nonexistent/file.ex", true, false)
        assert changes == 0
      end)

      assert output =~ "Failed to read"
    end
  end

  describe "fix_trailing_whitespace/1" do
    test "removes trailing spaces from lines" do
      input = "hello  \nworld\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello\nworld\n"
      assert changes == 1
    end

    test "removes trailing tabs from lines" do
      input = "hello\t\t\nworld\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello\nworld\n"
      assert changes == 1
    end

    test "handles lines with only whitespace (preserves newline)" do
      input = "  \n\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "\n\n"
      assert changes == 1
    end

    test "does not modify clean lines" do
      input = "hello\nworld\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello\nworld\n"
      assert changes == 0
    end

    test "preserves leading indentation" do
      input = "  def foo do  \n    :ok\n  end\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "  def foo do\n    :ok\n  end\n"
      assert changes == 1
    end

    test "handles multiple lines with trailing whitespace" do
      input = "line1  \nline2  \nline3  \n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "line1\nline2\nline3\n"
      assert changes == 3
    end

    test "handles files without trailing newline" do
      input = "hello  "
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello"
      assert changes == 1
    end

    test "handles Windows line endings (CRLF)" do
      input = "hello  \r\nworld\r\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello\r\nworld\r\n"
      assert changes == 1
    end

    test "handles mixed content" do
      # Build input with explicit trailing spaces
      input =
        "defmodule Foo do  \n" <>
          "  def bar do\n" <>
          "    :ok  \n" <>
          "  end\n" <>
          "end\n"

      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      expected =
        "defmodule Foo do\n" <>
          "  def bar do\n" <>
          "    :ok\n" <>
          "  end\n" <>
          "end\n"

      assert output == expected
      assert changes == 2
    end

    test "handles empty string" do
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace("")
      assert output == ""
      assert changes == 0
    end

    test "handles mixed spaces and tabs" do
      input = "hello \t \t\nworld\n"
      {output, changes} = TrailingWhitespace.fix_trailing_whitespace(input)

      assert output == "hello\nworld\n"
      assert changes == 1
    end
  end

  describe "has_trailing_whitespace?/1" do
    test "returns true for content with trailing spaces" do
      assert TrailingWhitespace.has_trailing_whitespace?("hello  \n")
    end

    test "returns true for content with trailing tabs" do
      assert TrailingWhitespace.has_trailing_whitespace?("hello\t\n")
    end

    test "returns false for clean content" do
      refute TrailingWhitespace.has_trailing_whitespace?("hello\nworld\n")
    end

    test "returns true for trailing whitespace at EOF without newline" do
      assert TrailingWhitespace.has_trailing_whitespace?("hello  ")
    end
  end
end
