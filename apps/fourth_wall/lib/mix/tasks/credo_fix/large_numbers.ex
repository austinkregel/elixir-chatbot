defmodule Mix.Tasks.CredoFix.LargeNumbers do
  @moduledoc """
  Mix task to fix large number formatting issues flagged by Credo.

  This addresses Credo.Check.Readability.LargeNumbers by adding underscores
  to integers >= 10_000 and floats with >= 5 digits before the decimal.

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.large_numbers

      # Actually apply the fixes
      mix credo_fix.large_numbers --apply

      # Fix specific directory
      mix credo_fix.large_numbers --apply apps/brain/lib

      # Show verbose output
      mix credo_fix.large_numbers --verbose

  ## Options

    --apply      Actually apply the fixes (required to modify files)
    --dry-run    Show what would be changed without modifying files (default)
    --verbose    Show each number being fixed

  NOTE: By default, this task runs in dry-run mode. You must pass --apply to modify files.

  ## Examples of fixes

      10_000      -> 10_000
      100_000     -> 100_000
      1_000_000    -> 1_000_000
      10_000.5    -> 10_000.5
      1_000_000.12 -> 1_000_000.12

  ## What is NOT fixed

  - Numbers in strings, comments, or heredocs
  - Hex/octal/binary literals (0xFF, 0o777, 0b1010)
  - Numbers that are part of identifiers
  - Numbers already containing underscores
  """

  use Mix.Task
  require Logger

  @shortdoc "Fix large number formatting in Elixir files"

  @default_paths ["apps/"]
  @extensions [".ex", ".exs"]

  # Match integers >= 10000 that don't already have underscores
  # Negative lookbehind: not preceded by underscore or word char
  # Negative lookahead: not followed by underscore, word char
  @integer_pattern ~r/(?<![_\w])(\d{5,})(?![_\w])/

  # Match floats with large integer parts (>= 5 digits before decimal)
  @float_int_pattern ~r/(?<![_\w])(\d{5,})\.(\d+)(?![_\w])/

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
    # Default to dry-run unless --apply is explicitly passed
    dry_run? = not apply? or Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)

    target_paths = if Enum.empty?(paths), do: @default_paths, else: paths

    files = collect_files(target_paths)

    if Enum.empty?(files) do
      Mix.shell().info("No Elixir files found in #{inspect(target_paths)}")
    else
      if dry_run? do
        Mix.shell().info("[DRY RUN] Scanning #{length(files)} files for large numbers...")
        Mix.shell().info("Pass --apply to actually modify files.\n")
      else
        Mix.shell().info("Scanning #{length(files)} files for large numbers...")
      end

      results =
        files
        |> Enum.map(&process_file(&1, dry_run?, verbose?))
        |> Enum.filter(fn {_, changes} -> changes > 0 end)

      total_files = length(results)
      total_changes = Enum.reduce(results, 0, fn {_, c}, acc -> acc + c end)

      if dry_run? do
        Mix.shell().info("\n[DRY RUN] Would fix #{total_changes} numbers in #{total_files} files")
      else
        Mix.shell().info("\nFixed #{total_changes} numbers in #{total_files} files")
      end
    end
  end

  @doc """
  Process a single file to fix large number formatting.
  """
  def process_file(path, dry_run?, verbose?) do
    case File.read(path) do
      {:ok, content} ->
        {fixed_content, changes} = fix_large_numbers(content, verbose?, path)

        if changes > 0 do
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
  Fix large numbers in content string.

  Returns `{fixed_content, num_changes}`.

  ## Examples

      iex> {content, _} = Mix.Tasks.CredoFix.LargeNumbers.fix_large_numbers("x = 10000", false, "")
      iex> content
      "x = 10_000"

      iex> {content, _} = Mix.Tasks.CredoFix.LargeNumbers.fix_large_numbers("x = 1000000", false, "")
      iex> content
      "x = 1_000_000"
  """
  def fix_large_numbers(content, verbose?, path) when is_binary(content) do
    lines = String.split(content, "\n")

    {fixed_lines, total_changes} =
      lines
      |> Enum.with_index(1)
      |> Enum.map_reduce(0, fn {line, line_num}, acc ->
        # Skip comment-only lines
        if String.match?(line, ~r/^\s*#/) do
          {line, acc}
        else
          {fixed_line, changes} = fix_numbers_in_line(line, verbose?, path, line_num)
          {fixed_line, acc + changes}
        end
      end)

    {Enum.join(fixed_lines, "\n"), total_changes}
  end

  defp fix_numbers_in_line(line, verbose?, path, line_num) do
    # First, handle floats with large integer parts
    # Process from right to left so index positions stay valid after replacements
    {line1, float_changes} =
      Regex.scan(@float_int_pattern, line, return: :index)
      |> Enum.reverse()
      |> Enum.reduce({line, 0}, fn [{start, len} | _], {current_line, count} ->
        original = String.slice(current_line, start, len)

        # Only fix if not already underscored and not in a string
        if not String.contains?(original, "_") and not in_string_context?(current_line, start) do
          fixed = format_float(original)

          if fixed != original do
            if verbose? do
              Mix.shell().info("  #{path}:#{line_num}: #{original} -> #{fixed}")
            end

            new_line =
              String.slice(current_line, 0, start) <>
                fixed <>
                String.slice(current_line, start + len, String.length(current_line))

            {new_line, count + 1}
          else
            {current_line, count}
          end
        else
          {current_line, count}
        end
      end)

    # Then, handle plain integers (also right to left)
    {line2, int_changes} =
      Regex.scan(@integer_pattern, line1, return: :index)
      |> Enum.reverse()
      |> Enum.reduce({line1, 0}, fn [{start, len} | _], {current_line, count} ->
        original = String.slice(current_line, start, len)

        # Check what follows - skip if it's a float (single dot followed by digit)
        after_num = String.slice(current_line, start + len, 2)
        is_float = String.match?(after_num, ~r/^\.\d/)

        # Only fix if not already underscored, not in a string, and not a float
        if not String.contains?(original, "_") and
             not in_string_context?(current_line, start) and
             not is_float do
          fixed = format_integer(original)

          if fixed != original do
            if verbose? do
              Mix.shell().info("  #{path}:#{line_num}: #{original} -> #{fixed}")
            end

            new_line =
              String.slice(current_line, 0, start) <>
                fixed <>
                String.slice(current_line, start + len, String.length(current_line))

            {new_line, count + 1}
          else
            {current_line, count}
          end
        else
          {current_line, count}
        end
      end)

    {line2, float_changes + int_changes}
  end

  @doc """
  Format an integer string with underscores every 3 digits from the right.

  ## Examples

      iex> Mix.Tasks.CredoFix.LargeNumbers.format_integer("10000")
      "10_000"

      iex> Mix.Tasks.CredoFix.LargeNumbers.format_integer("1000000")
      "1_000_000"

      iex> Mix.Tasks.CredoFix.LargeNumbers.format_integer("123")
      "123"
  """
  def format_integer(num_str) when is_binary(num_str) do
    num_str
    |> String.graphemes()
    |> Enum.reverse()
    |> Enum.chunk_every(3)
    |> Enum.map(&Enum.reverse/1)
    |> Enum.reverse()
    |> Enum.map_join("_", &Enum.join/1)
  end

  @doc """
  Format a float string with underscores in the integer part.

  ## Examples

      iex> Mix.Tasks.CredoFix.LargeNumbers.format_float("10000.5")
      "10_000.5"

      iex> Mix.Tasks.CredoFix.LargeNumbers.format_float("1000000.123")
      "1_000_000.123"
  """
  def format_float(num_str) when is_binary(num_str) do
    case String.split(num_str, ".") do
      [int_part, dec_part] ->
        format_integer(int_part) <> "." <> dec_part

      [int_part] ->
        format_integer(int_part)
    end
  end

  # Simple heuristic to detect if position is inside a string literal
  # Count quotes before this position - if odd, we're inside a string
  defp in_string_context?(line, position) do
    prefix = String.slice(line, 0, position)

    # Count unescaped double quotes
    # This is a simplification - doesn't handle all edge cases
    quote_count =
      Regex.scan(~r/(?<!\\)"/, prefix)
      |> length()

    rem(quote_count, 2) == 1
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
        Path.wildcard(path)
        |> Enum.filter(&File.regular?/1)
    end
  end

  defp elixir_file?(path) do
    ext = Path.extname(path)
    ext in @extensions
  end
end
