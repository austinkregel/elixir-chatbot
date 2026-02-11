defmodule Mix.Tasks.CredoFix.TrailingWhitespace do
  @moduledoc """
  Mix task to fix trailing whitespace issues flagged by Credo.

  This addresses Credo.Check.Readability.TrailingWhiteSpace by removing
  trailing spaces/tabs from line endings while preserving the newline.

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.trailing_whitespace

      # Actually apply the fixes
      mix credo_fix.trailing_whitespace --apply

      # Fix specific directory
      mix credo_fix.trailing_whitespace --apply apps/brain/lib

      # Fix specific file
      mix credo_fix.trailing_whitespace --apply apps/brain/lib/brain.ex

  ## Options

    --apply           Actually apply the fixes (required to modify files)
    --dry-run         Show what would be changed without modifying files (default)
    --verbose         Show each line being fixed
    --report-clean    Report files that were scanned but had no issues
    --list-files      Only list files that have issues (one per line)

  NOTE: By default, this task runs in dry-run mode. You must pass --apply to modify files.

  ## How it works

  For each line in a file:
  1. If the line ends with whitespace (spaces or tabs) before the newline,
     strip that trailing whitespace
  2. Preserve the newline character(s) (\\n or \\r\\n)
  3. Empty lines remain empty (just newline, no spaces)
  """

  use Mix.Task
  require Logger

  @shortdoc "Fix trailing whitespace in Elixir files"

  @default_paths ["apps/"]
  @extensions [".ex", ".exs"]

  def run(args) do
    {opts, paths, _} =
      OptionParser.parse(args,
        strict: [
          apply: :boolean,
          dry_run: :boolean,
          verbose: :boolean,
          report_clean: :boolean,
          list_files: :boolean
        ],
        aliases: [
          d: :dry_run,
          v: :verbose
        ]
      )

    apply? = Keyword.get(opts, :apply, false)
    # Default to dry-run unless --apply is explicitly passed
    dry_run? = not apply? or Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)
    report_clean? = Keyword.get(opts, :report_clean, false)
    list_files? = Keyword.get(opts, :list_files, false)

    target_paths = if Enum.empty?(paths), do: @default_paths, else: paths

    files = collect_files(target_paths)

    if Enum.empty?(files) do
      Mix.shell().info("No Elixir files found in #{inspect(target_paths)}")
    else
      if dry_run? do
        Mix.shell().info("[DRY RUN] Scanning #{length(files)} files for trailing whitespace...")
        Mix.shell().info("Pass --apply to actually modify files.\n")
      else
        Mix.shell().info("Scanning #{length(files)} files for trailing whitespace...")
      end

      # For --list-files mode, just scan and list without full processing
      if list_files? do
        files_with_issues =
          files
          |> Enum.filter(fn path ->
            case File.read(path) do
              {:ok, content} -> has_trailing_whitespace?(content)
              _ -> false
            end
          end)

        Enum.each(files_with_issues, &Mix.shell().info/1)
        %{files_with_issues: files_with_issues, total: length(files_with_issues)}
      else
        all_results =
          files
          |> Enum.map(&process_file(&1, dry_run?, verbose?))

        results_with_changes = Enum.filter(all_results, fn {_, changes} -> changes > 0 end)
        clean_files = Enum.filter(all_results, fn {_, changes} -> changes == 0 end)

        total_files = length(results_with_changes)
        total_changes = Enum.reduce(results_with_changes, 0, fn {_, c}, acc -> acc + c end)

        if dry_run? do
          Mix.shell().info("\n[DRY RUN] Would fix #{total_changes} lines in #{total_files} files")
        else
          Mix.shell().info("\nFixed #{total_changes} lines in #{total_files} files")
        end

        if report_clean? do
          Mix.shell().info("\nFiles with no trailing whitespace (#{length(clean_files)}):")

          clean_files
          |> Enum.each(fn {path, _} ->
            Mix.shell().info("  #{path}")
          end)
        end

        # Return the results for programmatic use
        %{
          files_with_issues: Enum.map(results_with_changes, fn {path, count} -> {path, count} end),
          clean_files: Enum.map(clean_files, fn {path, _} -> path end),
          total_changes: total_changes
        }
      end
    end
  end

  @doc """
  Check if content has any trailing whitespace.
  """
  def has_trailing_whitespace?(content) when is_binary(content) do
    String.match?(content, ~r/[ \t]+(\r?\n)/) or
      (String.match?(content, ~r/[ \t]+$/) and not String.ends_with?(content, "\n"))
  end

  @doc """
  Process a single file to remove trailing whitespace.

  Returns `{path, num_changes}` where num_changes is the count of lines modified.
  """
  def process_file(path, dry_run?, verbose?) do
    case File.read(path) do
      {:ok, content} ->
        {fixed_content, changes} = fix_trailing_whitespace(content)

        if changes > 0 do
          if verbose? do
            Mix.shell().info("  #{path}: #{changes} lines")
          end

          unless dry_run? do
            File.write!(path, fixed_content)
          end
        end

        {path, changes}

      {:error, reason} ->
        Mix.shell().error("Failed to read #{path}: #{inspect(reason)}")
        {path, 0}
    end
  end

  @doc """
  Fix trailing whitespace in content string.

  Returns `{fixed_content, num_changes}`.

  Matches pattern: any spaces/tabs immediately before a newline.
  This is what Credo flags as trailing whitespace.

  ## Examples

      iex> Mix.Tasks.CredoFix.TrailingWhitespace.fix_trailing_whitespace("hello  \\n")
      {"hello\\n", 1}

      iex> Mix.Tasks.CredoFix.TrailingWhitespace.fix_trailing_whitespace("  \\n\\n")
      {"\\n\\n", 1}

      iex> Mix.Tasks.CredoFix.TrailingWhitespace.fix_trailing_whitespace("clean\\n")
      {"clean\\n", 0}
  """
  def fix_trailing_whitespace(content) when is_binary(content) do
    # Pattern: one or more spaces/tabs followed by a newline (capturing the newline)
    # We replace with just the newline, removing the trailing whitespace
    pattern = ~r/[ \t]+(\r?\n)/

    # Count matches first
    matches = Regex.scan(pattern, content)
    changes = length(matches)

    # Replace all occurrences: strip the whitespace, keep the newline
    fixed_content = Regex.replace(pattern, content, "\\1")

    # Also handle trailing whitespace at end of file (no newline after)
    {fixed_content, extra_changes} =
      if String.match?(content, ~r/[ \t]+$/) and not String.ends_with?(content, "\n") do
        {String.replace(fixed_content, ~r/[ \t]+$/, ""), 1}
      else
        {fixed_content, 0}
      end

    {fixed_content, changes + extra_changes}
  end

  defp collect_files(paths) do
    paths
    |> Enum.flat_map(&expand_path/1)
    |> Enum.filter(&elixir_file?/1)
    |> Enum.reject(&excluded_path?/1)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # Exclude fourth_wall test files to prevent self-modification of test fixtures
  defp excluded_path?(path) do
    String.contains?(path, "apps/fourth_wall/test/")
  end

  defp expand_path(path) do
    cond do
      File.regular?(path) ->
        [path]

      File.dir?(path) ->
        Path.wildcard(Path.join(path, "**/*"))
        |> Enum.filter(&File.regular?/1)

      true ->
        # Could be a glob pattern
        Path.wildcard(path)
        |> Enum.filter(&File.regular?/1)
    end
  end

  defp elixir_file?(path) do
    ext = Path.extname(path)
    ext in @extensions
  end
end
