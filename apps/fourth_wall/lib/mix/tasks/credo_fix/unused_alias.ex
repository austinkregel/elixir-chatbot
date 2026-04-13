defmodule Mix.Tasks.CredoFix.UnusedAlias do
  @moduledoc """
  Mix task to remove unused aliases detected by the Elixir compiler.

  This addresses compiler warnings like:
    warning: unused alias Analysis

  The fixer works by:
  1. Running `mix compile` to capture compiler warnings
  2. Parsing warnings to identify unused aliases and their locations
  3. Using AST transformation to remove the unused aliases

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.unused_alias

      # Actually apply the fixes
      mix credo_fix.unused_alias --apply

      # Fix specific directory
      mix credo_fix.unused_alias --apply apps/brain/lib

      # Show verbose output
      mix credo_fix.unused_alias --verbose

  ## Options

    --apply      Actually apply the fixes (required to modify files)
    --dry-run    Show what would be changed without modifying files (default)
    --verbose    Show each alias being removed

  NOTE: By default, this task runs in dry-run mode. You must pass --apply to modify files.
  """

  use Mix.Task
  require Logger

  alias FourthWall.AST
  alias FourthWall.Safety

  @shortdoc "Remove unused aliases detected by compiler"

  @default_paths ["apps/"]

  def run(args) do
    {opts, paths, _} =
      OptionParser.parse(args,
        strict: [
          apply: :boolean,
          dry_run: :boolean,
          verbose: :boolean
        ],
        aliases: [
          d: :dry_run,
          v: :verbose
        ]
      )

    apply? = Keyword.get(opts, :apply, false)
    dry_run? = not apply? or Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)

    paths = if Enum.empty?(paths), do: @default_paths, else: paths

    if dry_run? do
      Mix.shell().info("[DRY RUN] Scanning for unused aliases...")
      Mix.shell().info("Pass --apply to actually modify files.\n")
    else
      Mix.shell().info("Removing unused aliases...")
    end

    # Step 1: Get unused aliases from compiler
    unused_aliases = collect_unused_aliases(paths)

    if Enum.empty?(unused_aliases) do
      Mix.shell().info("No unused aliases found.")
      {:ok, 0}
    else
      # Group by file
      by_file = Enum.group_by(unused_aliases, & &1.file)

      total_removed =
        by_file
        |> Enum.map(fn {file, aliases} ->
          process_file(file, aliases, dry_run?, verbose?)
        end)
        |> Enum.sum()

      if dry_run? do
        Mix.shell().info("\n[DRY RUN] Would remove #{total_removed} unused aliases")
      else
        Mix.shell().info("\nRemoved #{total_removed} unused aliases")
      end

      {:ok, total_removed}
    end
  end

  @doc """
  Collects unused alias warnings from the compiler.

  Returns a list of maps with :file, :line, and :alias_name keys.
  """
  def collect_unused_aliases(paths) do
    # Force recompile to get fresh warnings
    compile_output = run_compile()

    # Parse warnings like:
    #   warning: unused alias Analysis
    #    │
    # 11 │   alias Brain.Analysis
    #    │   ~
    #    │
    #    └─ lib/brain/analysis/outcome_learner.ex:4:3

    parse_unused_alias_warnings(compile_output, paths)
  end

  defp run_compile do
    # We need to capture compiler warnings. The best way is to run mix compile
    # in a subprocess with stderr captured. We use `--force` to ensure we get
    # fresh warnings.
    #
    # Note: Running mix inside mix can be tricky. We use System.cmd which
    # spawns a new OS process, avoiding issues with the Elixir environment.
    #
    # First clean to ensure fresh compile
    _ = System.cmd("mix", ["clean"], stderr_to_stdout: true, cd: File.cwd!())

    # Now compile and capture output
    {output, _exit_code} =
      System.cmd("mix", ["compile"],
        stderr_to_stdout: true,
        cd: File.cwd!(),
        env: [{"MIX_QUIET", "false"}]
      )

    output
  end

  @doc """
  Parses compiler output to extract unused alias warnings.

  Returns a list of %{file: path, line: integer, alias_name: string}
  """
  def parse_unused_alias_warnings(output, paths) do
    # Split into lines for processing
    lines = String.split(output, "\n")

    # Find all "unused alias X" warnings and their locations
    parse_warnings(lines, [], paths)
  end

  defp parse_warnings([], acc, _paths), do: acc

  defp parse_warnings([line | rest], acc, paths) do
    case parse_unused_alias_line(line) do
      {:ok, alias_name} ->
        # Look ahead to find the file location (└─ file:line:col)
        case find_location(rest) do
          {:ok, file, line_num} ->
            # Resolve the file path (handles umbrella app paths)
            resolved_file = resolve_umbrella_path(file)

            # Check if this file is in the target paths
            if file_in_paths?(resolved_file, paths) and not excluded_path?(resolved_file) do
              entry = %{file: resolved_file, line: line_num, alias_name: alias_name}
              parse_warnings(rest, [entry | acc], paths)
            else
              parse_warnings(rest, acc, paths)
            end

          :not_found ->
            parse_warnings(rest, acc, paths)
        end

      :no_match ->
        parse_warnings(rest, acc, paths)
    end
  end

  defp parse_unused_alias_line(line) do
    # Match "warning: unused alias SomeModule"
    case Regex.run(~r/warning: unused alias (\w+)/, line) do
      [_, alias_name] -> {:ok, alias_name}
      _ -> :no_match
    end
  end

  defp find_location(lines) do
    # Look for the └─ line with file:line:col format
    Enum.find_value(lines, :not_found, fn line ->
      case Regex.run(~r/└─.*?([^\s]+):(\d+):\d+/, line) do
        [_, file_path, line_num] ->
          # Extract just the relative path if it's wrapped
          clean_path = clean_file_path(file_path)
          {:ok, clean_path, String.to_integer(line_num)}

        _ ->
          # Also try simpler format: └─ file:line
          case Regex.run(~r/└─\s+([^\s:]+):(\d+)/, line) do
            [_, file_path, line_num] ->
              clean_path = clean_file_path(file_path)
              {:ok, clean_path, String.to_integer(line_num)}

            _ ->
              nil
          end
      end
    end)
  end

  defp clean_file_path(path) do
    # Remove any parenthetical prefixes like "(brain 0.1.0)"
    path
    |> String.replace(~r/^\([^)]+\)\s*/, "")
    |> String.trim()
  end

  # Resolve paths from compiler output to actual file paths.
  # Compiler outputs paths like "lib/brain/foo.ex" which are relative to
  # the umbrella app directory (apps/brain/lib/brain/foo.ex from root).
  defp resolve_umbrella_path(path) do
    cond do
      # Already a full path from umbrella root
      String.starts_with?(path, "apps/") ->
        path

      # Relative path like "lib/brain/foo.ex" - need to find the app
      String.starts_with?(path, "lib/") ->
        # Try to find which umbrella app this belongs to by checking
        # if the path exists under any of them
        find_umbrella_app_path(path)

      true ->
        path
    end
  end

  defp find_umbrella_app_path(relative_path) do
    # List umbrella apps and check which one contains this file
    apps_dir = Path.join(File.cwd!(), "apps")

    if File.dir?(apps_dir) do
      case File.ls(apps_dir) do
        {:ok, apps} ->
          Enum.find_value(apps, relative_path, fn app ->
            full_path = Path.join(["apps", app, relative_path])

            if File.exists?(full_path), do: full_path, else: nil
          end)

        _ ->
          relative_path
      end
    else
      relative_path
    end
  end

  defp file_in_paths?(file, paths) do
    # When running from umbrella, paths might be "apps/" but compiler output
    # gives relative paths like "lib/..." from each app's perspective.
    # Also handle "(app_name x.y.z) lib/..." format.
    Enum.any?(paths, fn path ->
      cond do
        # Direct match
        String.starts_with?(file, path) ->
          true

        # With ./ prefix
        String.starts_with?(file, "./#{path}") ->
          true

        # Contains the path segment
        String.contains?(file, "/#{path}") ->
          true

        # For umbrella apps: when path is "apps/", accept "lib/" paths
        # because the compiler outputs relative to each app
        path == "apps/" and String.starts_with?(file, "lib/") ->
          true

        # Specific app path like "apps/brain"
        String.starts_with?(path, "apps/") and String.starts_with?(file, "lib/") ->
          true

        true ->
          false
      end
    end)
  end

  @doc """
  Processes a single file, removing the specified unused aliases.

  Returns the number of aliases removed.
  """
  def process_file(file, aliases, dry_run?, verbose?) do
    case File.read(file) do
      {:ok, content} ->
        alias_names = Enum.map(aliases, & &1.alias_name) |> Enum.uniq()

        case remove_unused_aliases(content, alias_names, verbose?, file) do
          {:ok, new_content, removed_count} when removed_count > 0 ->
            if verbose? do
              Enum.each(alias_names, fn name ->
                Mix.shell().info("  #{file}: removing unused alias #{name}")
              end)
            end

            if not dry_run? do
              # Safety check before writing
              case Safety.check_output(new_content) do
                {:ok, _} ->
                  File.write!(file, new_content)

                {:error, issues} ->
                  descriptions = Enum.map_join(issues, ", ", fn {_p, d} -> d end)
                  Mix.shell().error("Safety check failed for #{file}: #{descriptions}")
                  0
              end
            end

            removed_count

          {:ok, _content, 0} ->
            0

          {:error, reason} ->
            if verbose? do
              Mix.shell().error("Failed to process #{file}: #{inspect(reason)}")
            end

            0
        end

      {:error, reason} ->
        if verbose? do
          Mix.shell().error("Failed to read #{file}: #{inspect(reason)}")
        end

        0
    end
  end

  @doc """
  Removes unused aliases from the given code string.

  Uses AST transformation to:
  1. Find all alias statements
  2. Remove those that match the unused alias names
  3. For grouped aliases like `alias Foo.{Bar, Baz}`, remove only the unused parts

  Returns {:ok, new_content, removed_count} or {:error, reason}
  """
  def remove_unused_aliases(content, alias_names, verbose?, file) do
    case AST.parse(content) do
      {:ok, ast} ->
        {new_ast, removed_count} = remove_aliases_from_ast(ast, alias_names)

        if removed_count > 0 do
          new_content = AST.to_source(new_ast)
          {:ok, new_content, removed_count}
        else
          {:ok, content, 0}
        end

      {:error, reason} ->
        if verbose? do
          Mix.shell().error("Failed to parse #{file}: #{inspect(reason)}")
        end

        {:error, reason}
    end
  end

  defp remove_aliases_from_ast(ast, alias_names) do
    alias_set = MapSet.new(alias_names)

    {new_ast, removed_count} =
      Macro.prewalk(ast, 0, fn
        # Simple alias: alias Foo.Bar
        {:alias, _meta, [{:__aliases__, _, parts}]} = node, acc ->
          last_part = parts |> List.last() |> to_string()

          if MapSet.member?(alias_set, last_part) do
            # Remove by replacing with nil (will be filtered)
            {nil, acc + 1}
          else
            {node, acc}
          end

        # Alias with :as option: alias Foo.Bar, as: Baz
        {:alias, _meta, [{:__aliases__, _, _parts}, [as: {:__aliases__, _, [as_name]}]]} = node,
        acc ->
          if MapSet.member?(alias_set, to_string(as_name)) do
            {nil, acc + 1}
          else
            {node, acc}
          end

        # Grouped alias: alias Foo.{Bar, Baz, Qux}
        {:alias, meta,
         [{{:., dot_meta, [{:__aliases__, prefix_meta, prefix_parts}, :{}]}, group_meta, suffixes}]} =
            node,
        acc ->
          # Filter out unused suffixes
          {remaining, removed} =
            Enum.reduce(suffixes, {[], 0}, fn
              {:__aliases__, _, suffix_parts} = suffix, {keep, count} ->
                last_part = suffix_parts |> List.last() |> to_string()

                if MapSet.member?(alias_set, last_part) do
                  {keep, count + 1}
                else
                  {[suffix | keep], count}
                end

              other, {keep, count} ->
                {[other | keep], count}
            end)

          remaining = Enum.reverse(remaining)

          cond do
            # All aliases were removed
            Enum.empty?(remaining) ->
              {nil, acc + removed}

            # Only one remaining - convert to simple alias
            length(remaining) == 1 ->
              [{:__aliases__, _, suffix_parts}] = remaining
              full_parts = prefix_parts ++ suffix_parts
              new_node = {:alias, meta, [{:__aliases__, prefix_meta, full_parts}]}
              {new_node, acc + removed}

            # Some aliases remain
            removed > 0 ->
              new_node =
                {:alias, meta,
                 [
                   {{:., dot_meta, [{:__aliases__, prefix_meta, prefix_parts}, :{}]}, group_meta,
                    remaining}
                 ]}

              {new_node, acc + removed}

            # Nothing changed
            true ->
              {node, acc}
          end

        node, acc ->
          {node, acc}
      end)

    # Filter out nil nodes (removed aliases)
    new_ast = filter_nil_nodes(new_ast)
    {new_ast, removed_count}
  end

  defp filter_nil_nodes(ast) do
    Macro.prewalk(ast, fn
      {:__block__, meta, statements} ->
        filtered = Enum.reject(statements, &is_nil/1)
        {:__block__, meta, filtered}

      node ->
        node
    end)
  end

  # Exclude fourth_wall test files to prevent self-modification of test fixtures
  defp excluded_path?(path) do
    String.contains?(path, "apps/fourth_wall/test/")
  end
end
