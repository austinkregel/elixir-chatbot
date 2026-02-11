defmodule Mix.Tasks.CredoFix.LargeNumbersTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.LargeNumbers

  describe "run/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "ln_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "runs in dry-run mode by default", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "x = 10000\n")

      output = capture_io(fn -> LargeNumbers.run([test_dir]) end)

      # File should NOT be modified
      assert File.read!(path) == "x = 10000\n"
      assert output =~ "[DRY RUN]"
      assert output =~ "Would fix"
    end

    test "applies changes with --apply flag", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "x = 10000\n")

      output = capture_io(fn -> LargeNumbers.run(["--apply", test_dir]) end)

      # File SHOULD be modified
      assert File.read!(path) == "x = 10_000\n"
      assert output =~ "Fixed"
    end

    test "verbose mode shows details", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, "x = 10000\n")

      output = capture_io(fn -> LargeNumbers.run(["--verbose", test_dir]) end)

      assert output =~ "10000 -> 10_000"
    end

    test "handles directory with no elixir files" do
      output = capture_io(fn -> LargeNumbers.run(["nonexistent_path/"]) end)

      assert output =~ "No Elixir files found"
    end
  end

  describe "process_file/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "ln_test_#{:rand.uniform(100_000)}.ex")
      on_exit(fn -> File.rm(path) end)
      {:ok, path: path}
    end

    test "processes file with large numbers (dry-run)", %{path: path} do
      File.write!(path, "x = 10000\n")

      # Non-verbose mode produces no output
      {result_path, changes} = LargeNumbers.process_file(path, true, false)

      assert result_path == path
      assert changes == 1
      # File should NOT be modified in dry-run
      assert File.read!(path) == "x = 10000\n"
    end

    test "processes file with large numbers (apply)", %{path: path} do
      File.write!(path, "x = 10000\n")

      # Non-verbose mode produces no output
      {result_path, changes} = LargeNumbers.process_file(path, false, false)

      assert result_path == path
      assert changes == 1
      # File SHOULD be modified
      assert File.read!(path) == "x = 10_000\n"
    end

    test "processes file with verbose mode", %{path: path} do
      File.write!(path, "x = 10000\n")

      output = capture_io(fn ->
        {result_path, changes} = LargeNumbers.process_file(path, true, true)
        assert result_path == path
        assert changes == 1
      end)

      assert output =~ "10000 -> 10_000"
    end

    test "handles file with no changes", %{path: path} do
      File.write!(path, "x = 100\n")

      {result_path, changes} = LargeNumbers.process_file(path, false, false)

      assert result_path == path
      assert changes == 0
    end

    test "handles non-existent file" do
      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        {_path, changes} = LargeNumbers.process_file("/nonexistent/file.ex", true, false)
        assert changes == 0
      end)

      assert output =~ "Failed to read"
    end
  end

  describe "format_integer/1" do
    test "formats 5-digit numbers" do
      assert LargeNumbers.format_integer("10000") == "10_000"
      assert LargeNumbers.format_integer("99999") == "99_999"
    end

    test "formats 6-digit numbers" do
      assert LargeNumbers.format_integer("100000") == "100_000"
      assert LargeNumbers.format_integer("999999") == "999_999"
    end

    test "formats 7-digit numbers" do
      assert LargeNumbers.format_integer("1000000") == "1_000_000"
    end

    test "formats 4-digit numbers" do
      assert LargeNumbers.format_integer("1234") == "1_234"
    end

    test "does not modify small numbers" do
      assert LargeNumbers.format_integer("123") == "123"
      assert LargeNumbers.format_integer("12") == "12"
      assert LargeNumbers.format_integer("1") == "1"
    end
  end

  describe "format_float/1" do
    test "formats integer part of floats" do
      assert LargeNumbers.format_float("10000.5") == "10_000.5"
      assert LargeNumbers.format_float("1000000.123") == "1_000_000.123"
    end

    test "preserves decimal part" do
      assert LargeNumbers.format_float("10000.123456") == "10_000.123456"
    end

    test "handles integers passed as float format" do
      assert LargeNumbers.format_float("10000") == "10_000"
    end
  end

  describe "fix_large_numbers/3" do
    test "fixes integers in code" do
      input = "x = 10000"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "x = 10_000"
      assert changes == 1
    end

    test "fixes multiple numbers on same line" do
      input = "range = 10000..99999"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "range = 10_000..99_999"
      assert changes == 2
    end

    test "skips numbers in comments" do
      input = "# x = 10000"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "# x = 10000"
      assert changes == 0
    end

    test "skips numbers inside strings" do
      input = ~s(msg = "Error code: 10000")
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == ~s(msg = "Error code: 10000")
      assert changes == 0
    end

    test "handles real code patterns" do
      # Use explicit string concatenation to prevent heredoc from being modified
      # by the credo fixer (which would add underscores to the test input!)
      input = "defmodule Config do\n  @timeout 30000\n  @max_size 100000\nend\n"

      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      expected = "defmodule Config do\n  @timeout 30_000\n  @max_size 100_000\nend\n"

      assert output == expected
      assert changes == 2
    end

    test "handles floats with large integer parts" do
      input = "x = 10000.5"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "x = 10_000.5"
      assert changes == 1
    end

    test "does not modify numbers already with underscores" do
      input = "x = 10_000"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "x = 10_000"
      assert changes == 0
    end

    test "does not modify small numbers" do
      input = "x = 1000"
      {output, changes} = LargeNumbers.fix_large_numbers(input, false, "test.ex")

      assert output == "x = 1000"
      assert changes == 0
    end

    test "handles empty content" do
      {output, changes} = LargeNumbers.fix_large_numbers("", false, "test.ex")

      assert output == ""
      assert changes == 0
    end
  end
end
