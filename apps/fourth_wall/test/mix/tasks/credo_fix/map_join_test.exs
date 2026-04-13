defmodule Mix.Tasks.CredoFix.MapJoinTest do
  alias FourthWall.Safety
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.MapJoin

  describe "run/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "mj_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "runs in dry-run mode by default", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      output = capture_io(fn -> MapJoin.run([test_dir]) end)

      content = File.read!(path)
      assert String.contains?(content, "Enum.map(")
      assert output =~ "[DRY RUN]"
      assert output =~ "Would fix"
    end

    test "applies changes with --apply flag", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      output = capture_io(fn -> MapJoin.run(["--apply", test_dir]) end)

      content = File.read!(path)
      assert String.contains?(content, "Enum.map_join")
      assert output =~ "Fixed"
    end

    test "verbose mode shows details", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      output = capture_io(fn -> MapJoin.run(["--verbose", test_dir]) end)

      assert output =~ "map/join pattern"
    end

    test "handles directory with no elixir files" do
      output = capture_io(fn -> MapJoin.run(["nonexistent_path/"]) end)

      assert output =~ "No Elixir files found"
    end
  end

  describe "process_file/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "mj_test_#{:rand.uniform(100_000)}.ex")
      on_exit(fn -> File.rm(path) end)
      {:ok, path: path}
    end

    test "processes file with map/join patterns (dry-run)", %{path: path} do
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      # Non-verbose mode produces no output
      {result_path, changes} = MapJoin.process_file(path, true, false)

      assert result_path == path
      assert changes == 1
      assert String.contains?(File.read!(path), "Enum.map(")
    end

    test "processes file with map/join patterns (apply)", %{path: path} do
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      # Non-verbose mode produces no output
      {result_path, changes} = MapJoin.process_file(path, false, false)

      assert result_path == path
      assert changes == 1
      content = File.read!(path)
      assert String.contains?(content, "Enum.map_join")
    end

    test "processes file with verbose mode", %{path: path} do
      File.write!(path, ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")])

      output = capture_io(fn ->
        {result_path, changes} = MapJoin.process_file(path, true, true)
        assert result_path == path
        assert changes == 1
      end)

      assert output =~ "map/join pattern"
    end

    test "handles file with no changes", %{path: path} do
      File.write!(path, ~S[Enum.map_join(list, ",", &to_string/1)])

      {result_path, changes} = MapJoin.process_file(path, false, false)

      assert result_path == path
      assert changes == 0
    end

    test "handles non-existent file" do
      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        {_path, changes} = MapJoin.process_file("/nonexistent/file.ex", true, false)
        assert changes == 0
      end)

      assert output =~ "Failed to read"
    end
  end

  describe "fix_map_join/3 basic pipe patterns" do
    test "fixes piped map |> join pattern" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
      refute String.contains?(output, "Enum.map(") or String.contains?(output, "|> Enum.join")
    end

    test "fixes direct Enum.join(Enum.map(...), sep) pattern" do
      input = ~S[Enum.join(Enum.map(list, &to_string/1), ",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end

    test "fixes x |> Enum.map(fn) |> Enum.join(sep)" do
      input = ~S[data |> Enum.map(fn x -> x * 2 end) |> Enum.join("-")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end

    test "handles join with empty string separator" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.join("")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end

    test "handles join with no separator (defaults to empty string)" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.join()]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end
  end

  describe "patterns that should not be modified" do
    test "does not modify already correct Enum.map_join" do
      input = ~S[Enum.map_join(list, ",", &to_string/1)]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert output == input
      assert changes == 0
    end

    test "does not modify filter |> join (only map |> join)" do
      input = ~S[list |> Enum.filter(&valid?/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert output == input
      assert changes == 0
    end

    test "does not modify map |> filter |> join chain" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.filter(&valid?/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")
      assert output == input
      assert changes == 0
    end

    test "does not modify code in comments" do
      input = ~S[# list |> Enum.map(&to_string/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert output == input
      assert changes == 0
    end

    test "does not modify code in strings" do
      input = ~S[msg = "list |> Enum.map(&to_string/1) |> Enum.join(",")"]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert output == input
      assert changes == 0
    end

    test "does not modify Stream.map |> Enum.join" do
      input = ~S[list |> Stream.map(&to_string/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")
      assert output == input
      assert changes == 0
    end
  end

  describe "complex expressions" do
    test "preserves surrounding code before pipe chain" do
      input = ~S[result = data |> Enum.map(&to_string/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "result =")
      assert String.contains?(output, "Enum.map_join")
    end

    test "preserves code after join" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.join(",") |> String.upcase()]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
      assert String.contains?(output, "String.upcase")
    end

    test "handles multiline pipe chains" do
      input = "list\n|> Enum.map(&to_string/1)\n|> Enum.join(\",\")\n"

      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end

    test "handles multiple map/join patterns in same file" do
      input =
        "def foo(list) do\n  list |> Enum.map(&to_string/1) |> Enum.join(\",\")\nend\n\ndef bar(list) do\n  list |> Enum.map(&inspect/1) |> Enum.join(\"-\")\nend\n"

      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 2
      assert String.contains?(output, "Enum.map_join")
    end

    test "handles capture functions" do
      input = ~S[list |> Enum.map(&String.upcase/1) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")

      assert String.contains?(output, "&String.upcase/1") or
               String.contains?(output, "& String.upcase/1")
    end

    test "handles anonymous functions" do
      input = ~S[list |> Enum.map(fn x -> x * 2 end) |> Enum.join(",")]
      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert String.contains?(output, "Enum.map_join")
    end
  end

  describe "output validation" do
    test "output is valid Elixir syntax" do
      inputs = [
        ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")],
        ~S[Enum.join(Enum.map(list, &to_string/1), ",")],
        ~S[data |> Enum.map(fn x -> x * 2 end) |> Enum.join("-")]
      ]

      for input <- inputs do
        {output, _} = MapJoin.fix_map_join(input, false, "test.ex")

        case Code.string_to_quoted(output) do
          {:ok, _ast} ->
            :ok

          {:error, reason} ->
            flunk("Output is not valid Elixir syntax for input '#{input}': #{inspect(reason)}")
        end
      end
    end

    test "passes safety checks" do
      input = ~S[list |> Enum.map(&to_string/1) |> Enum.join(",")]
      {output, _} = MapJoin.fix_map_join(input, false, "test.ex")

      assert {:ok, _} = Safety.check_output(output)
    end

    test "full module transformation produces valid code" do
      input =
        "defmodule Example do\n  def format_list(list) do\n    list\n    |> Enum.map(&to_string/1)\n    |> Enum.join(\", \")\n  end\nend\n"

      {output, changes} = MapJoin.fix_map_join(input, false, "test.ex")

      assert changes == 1
      assert {:ok, _} = Code.string_to_quoted(output)
      assert String.contains?(output, "Enum.map_join")
    end
  end
end