defmodule Mix.Tasks.CredoFix.AliasUsageTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureIO

  alias Mix.Tasks.CredoFix.AliasUsage

  describe "run/1" do
    setup do
      tmp_dir = System.tmp_dir!()
      test_dir = Path.join(tmp_dir, "au_run_test_#{:rand.uniform(100_000)}")
      File.mkdir_p!(test_dir)

      on_exit(fn ->
        File.rm_rf!(test_dir)
      end)

      {:ok, test_dir: test_dir}
    end

    test "runs in dry-run mode by default", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")

      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      output = capture_io(fn -> AliasUsage.run([test_dir]) end)

      # File should NOT be modified
      assert File.read!(path) == content
      assert output =~ "[DRY RUN]"
      assert output =~ "Would add"
    end

    test "applies changes with --apply flag", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")

      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      output = capture_io(fn -> AliasUsage.run(["--apply", test_dir]) end)

      # File SHOULD be modified
      new_content = File.read!(path)
      assert String.contains?(new_content, "alias MyApp.Services.UserService")
      assert output =~ "Added"
    end

    test "verbose mode shows details", %{test_dir: test_dir} do
      path = Path.join(test_dir, "test.ex")

      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      output = capture_io(fn -> AliasUsage.run(["--verbose", test_dir]) end)

      assert output =~ "aliasing"
    end

    test "handles directory with no elixir files" do
      output = capture_io(fn -> AliasUsage.run(["nonexistent_path/"]) end)

      assert output =~ "No Elixir files found"
    end
  end

  describe "process_file/3" do
    setup do
      tmp_dir = System.tmp_dir!()
      path = Path.join(tmp_dir, "au_test_#{:rand.uniform(100_000)}.ex")
      on_exit(fn -> File.rm(path) end)
      {:ok, path: path}
    end

    test "processes file with nested module usages (dry-run)", %{path: path} do
      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      # Non-verbose mode produces no output
      {result_path, changes} = AliasUsage.process_file(path, true, false)

      assert result_path == path
      assert changes >= 1
      # File should NOT be modified in dry-run
      assert File.read!(path) == content
    end

    test "processes file with nested module usages (apply)", %{path: path} do
      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      # Non-verbose mode produces no output
      {result_path, changes} = AliasUsage.process_file(path, false, false)

      assert result_path == path
      assert changes >= 1
      # File SHOULD be modified
      new_content = File.read!(path)
      assert String.contains?(new_content, "alias MyApp.Services.UserService")
    end

    test "processes file with verbose mode", %{path: path} do
      content = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      File.write!(path, content)

      output = capture_io(fn ->
        {result_path, changes} = AliasUsage.process_file(path, true, true)
        assert result_path == path
        assert changes >= 1
      end)

      assert output =~ "aliasing"
    end

    test "handles file with no changes", %{path: path} do
      content = """
      defmodule Example do
        def foo do
          String.upcase("hello")
        end
      end
      """

      File.write!(path, content)

      {result_path, changes} = AliasUsage.process_file(path, false, false)

      assert result_path == path
      assert changes == 0
    end

    test "handles non-existent file" do
      # Error output goes to stderr
      output = capture_io(:stderr, fn ->
        {_path, changes} = AliasUsage.process_file("/nonexistent/file.ex", true, false)
        assert changes == 0
      end)

      assert output =~ "Failed to read"
    end
  end

  # =============================================================================
  # Basic alias insertion
  # =============================================================================

  describe "fix_alias_usage/3 basic patterns" do
    test "adds alias for nested module and shortens usage" do
      input = """
      defmodule MyApp.Web.Search do
        def twitter_mentions do
          MyApp.External.TwitterAPI.search("elixir")
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes >= 1
      assert String.contains?(output, "alias MyApp.External.TwitterAPI")
      assert String.contains?(output, "TwitterAPI.search")
      # Should not have the full path anymore (in usage, not alias)
      refute String.contains?(output, "MyApp.External.TwitterAPI.search")
    end

    test "adds alias for multiple usages of same module" do
      input = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
          MyApp.Services.UserService.update_user(1, %{})
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes >= 1
      assert String.contains?(output, "alias MyApp.Services.UserService")
      # Both usages should be shortened
      refute String.contains?(output, "MyApp.Services.UserService.get_user")
      refute String.contains?(output, "MyApp.Services.UserService.update_user")
    end

    test "handles multiple different modules" do
      input = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
          MyApp.External.TwitterAPI.search("test")
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes >= 2
      assert String.contains?(output, "alias MyApp.Services.UserService")
      assert String.contains?(output, "alias MyApp.External.TwitterAPI")
    end
  end

  # =============================================================================
  # Patterns that should NOT be modified
  # =============================================================================

  describe "patterns that should not be modified" do
    test "does not add alias for single-part modules" do
      input = """
      defmodule Example do
        def foo do
          String.upcase("hello")
          Enum.map([], &to_string/1)
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes == 0
      assert output == input
    end

    test "does not add alias for already aliased modules" do
      input = """
      defmodule Example do
        alias MyApp.Services.UserService

        def foo do
          UserService.get_user(1)
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes == 0
      # Should not duplicate the alias
      refute String.contains?(output, "alias MyApp.Services.UserService\n  alias")
    end

    test "does not add alias for excluded namespaces (File, IO, etc.)" do
      input = """
      defmodule Example do
        def foo do
          File.read!("test.txt")
          IO.puts("hello")
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes == 0
      assert output == input, "Excluded namespaces should not be modified"
    end

    test "skips code in strings" do
      input = """
      defmodule Example do
        def foo do
          "MyApp.Services.UserService.get_user(1)"
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes == 0
      assert output == input
    end
  end

  # =============================================================================
  # Alias placement
  # =============================================================================

  describe "alias placement" do
    test "places alias after defmodule and any moduledoc" do
      input = """
      defmodule Example do
        @moduledoc "Example module"

        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes >= 1
      # The alias should appear after moduledoc
      assert Regex.match?(~r/@moduledoc.*alias/s, output)
    end

    test "places alias with existing aliases" do
      input = """
      defmodule Example do
        alias MyApp.Other.Thing

        def foo do
          MyApp.Services.UserService.get_user(1)
          Thing.do_something()
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes >= 1
      assert String.contains?(output, "alias MyApp.Services.UserService")
    end
  end

  # =============================================================================
  # Conflict handling
  # =============================================================================

  describe "conflict handling" do
    test "does not alias if last name conflicts with existing alias" do
      input = """
      defmodule Example do
        alias MyApp.Other.UserService

        def foo do
          MyApp.Services.UserService.get_user(1)
          UserService.other_thing()
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      # Should not alias because UserService would conflict
      assert changes == 0
      assert output == input, "Conflicting alias should not modify the file"
    end
  end

  # =============================================================================
  # Output validation
  # =============================================================================

  describe "output validation" do
    test "output is valid Elixir syntax" do
      input = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      {output, _} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert {:ok, _} = Code.string_to_quoted(output)
    end

    test "passes safety checks" do
      input = """
      defmodule Example do
        def foo do
          MyApp.Services.UserService.get_user(1)
        end
      end
      """

      {output, _} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert {:ok, _} = FourthWall.Safety.check_output(output)
    end
  end

  # =============================================================================
  # REGRESSION: Grouped alias corruption bug
  # =============================================================================

  describe "regression: grouped alias corruption" do
    test "does not corrupt existing grouped alias statements" do
      # This was a REAL bug: The fixer would see Brain.Analysis.InternalModel
      # inside a grouped alias like `alias Brain.Analysis.{InternalModel, ...}`
      # and incorrectly shorten it to `alias Analysis.{InternalModel, ...}`
      input = """
      defmodule Brain.Analysis.ResponseGate do
        alias Brain.Analysis.{InternalModel, SpeechActResult}

        def foo(%InternalModel{} = model) do
          model
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      # Should NOT modify existing alias statements
      assert changes == 0
      # The original grouped alias must be preserved
      assert String.contains?(output, "alias Brain.Analysis.{InternalModel")
      # Must NOT have corrupted alias
      refute String.contains?(output, "alias Analysis.{InternalModel"),
             "Found corrupted grouped alias - Brain prefix was stripped"
    end

    test "does not corrupt modules in struct patterns" do
      # Struct usage like %InternalModel{} should not trigger new aliasing
      # when the module is already aliased via grouped alias
      input = """
      defmodule Brain.Analysis.ResponseGate do
        alias Brain.Analysis.{InternalModel, SpeechActResult}

        def evaluate(%InternalModel{} = model) do
          %SpeechActResult{} = get_result(model)
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      assert changes == 0
      # Original structure preserved
      assert String.contains?(output, "alias Brain.Analysis.{InternalModel")
    end

    test "does not modify alias statements that are not function calls" do
      # The fixer should ONLY look at Module.function() calls, not alias statements
      input = """
      defmodule Example do
        alias MyApp.Services.{UserService, AuthService}

        def foo do
          UserService.get_user(1)
        end
      end
      """

      {output, changes} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      # No changes needed - UserService is already aliased via grouped alias
      assert changes == 0
      assert String.contains?(output, "alias MyApp.Services.{UserService, AuthService}")
    end

    test "produces compilable code for sibling module references" do
      # When in Brain.Analysis.ResponseGate, references to Brain.Analysis.InternalModel
      # should either be left alone (if grouped alias exists) or properly aliased
      input = """
      defmodule Brain.Analysis.ResponseGate do
        alias Brain.Analysis.{InternalModel, LearningStore}

        def evaluate(%InternalModel{} = model) do
          LearningStore.get_params()
          model
        end
      end
      """

      {output, _} = AliasUsage.fix_alias_usage(input, false, "test.ex")

      # Output must be valid Elixir that compiles
      case Code.string_to_quoted(output) do
        {:ok, _ast} ->
          :ok

        {:error, reason} ->
          flunk("Output is not valid Elixir syntax: #{inspect(reason)}\n\nOutput:\n#{output}")
      end
    end
  end
end
