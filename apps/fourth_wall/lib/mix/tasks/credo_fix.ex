defmodule Mix.Tasks.CredoFix do
  @moduledoc "Master task to run Credo auto-fixers.\n\nThis task orchestrates the individual credo_fix.* tasks to fix common\nCredo issues automatically.\n\n## IMPORTANT: Dry-Run by Default\n\nThis task runs in dry-run mode by default. You MUST pass `--apply` to\nactually modify files. This is a safety measure to prevent accidental\ncode corruption.\n\n## Usage\n\n    # Preview all fixes (dry run - default, safe)\n    mix credo_fix\n\n    # Actually apply the fixes (requires --apply)\n    mix credo_fix --apply\n\n    # Run specific fixers only\n    mix credo_fix --apply --only trailing_whitespace,large_numbers\n\n    # Exclude specific fixers\n    mix credo_fix --apply --exclude map_join\n\n    # Target specific paths\n    mix credo_fix --apply apps/brain/lib\n\n## Available fixers\n\n  - trailing_whitespace  Removes trailing spaces/tabs from lines (regex-based)\n  - large_numbers        Adds underscores to numbers >= 10_000 (regex-based)\n  - length_check         Replaces length(x) == 0 with x == [] (AST-based)\n  - map_join             Converts Enum.map |> Enum.join to Enum.map_join (AST-based)\n  - alias_usage          Adds aliases for frequently used modules (AST-based)\n\n## Options\n\n  --apply                Actually apply the fixes (required to modify files)\n  --dry-run              Show what would be changed (default)\n  --verbose              Show detailed output for each fix\n  --only <fixers>        Comma-separated list of fixers to run\n  --exclude <fixers>     Comma-separated list of fixers to skip\n  --length-style <s>     Style for length fixes: \"pattern_match\" or \"enum_empty\"\n\n## Examples\n\n    # Check what would be fixed (default, safe)\n    mix credo_fix\n\n    # Actually apply fixes\n    mix credo_fix --apply\n\n    # Apply only trailing whitespace fixes\n    mix credo_fix --apply --only trailing_whitespace\n\n    # Apply everything except map_join (which may need review)\n    mix credo_fix --apply --exclude map_join\n\n    # Use Enum.empty? style for length checks\n    mix credo_fix --apply --only length_check --length-style enum_empty\n"

  alias Mix.Tasks.CredoFix.AliasUsage
  alias Mix.Tasks.CredoFix.MapJoin
  alias Mix.Tasks.CredoFix.LengthCheck
  alias Mix.Tasks.CredoFix.LargeNumbers
  alias Mix.Tasks.CredoFix.TrailingWhitespace
  use Mix.Task
  require Logger

  @shortdoc "Run Credo auto-fixers (dry-run by default)"
  @available_fixers [:trailing_whitespace, :large_numbers, :length_check, :map_join, :alias_usage]
  @planned_fixers []

  def run(args) do
    {opts, paths, _} =
      OptionParser.parse(args,
        strict: [
          apply: :boolean,
          dry_run: :boolean,
          verbose: :boolean,
          only: :string,
          exclude: :string,
          length_style: :string
        ],
        aliases: [d: :dry_run, v: :verbose]
      )

    apply? = Keyword.get(opts, :apply, false)
    dry_run? = not apply? or Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)
    length_style = Keyword.get(opts, :length_style, "pattern_match")

    allowed_fixer_strings = Enum.map(@available_fixers ++ @planned_fixers, &Atom.to_string/1)

    only =
      case Keyword.get(opts, :only) do
        nil -> nil
        str ->
          str
          |> String.split(",", trim: true)
          |> Enum.map(&String.trim/1)
          |> Enum.filter(&(&1 in allowed_fixer_strings))
          |> Enum.map(&String.to_existing_atom/1)
      end

    exclude =
      case Keyword.get(opts, :exclude) do
        nil -> []
        str ->
          str
          |> String.split(",", trim: true)
          |> Enum.map(&String.trim/1)
          |> Enum.filter(&(&1 in allowed_fixer_strings))
          |> Enum.map(&String.to_existing_atom/1)
      end

    if only do
      unimplemented = Enum.filter(only, &(&1 in @planned_fixers))

      if Enum.any?(unimplemented) do
        Mix.shell().info(
          "Note: The following fixers are planned but not yet implemented: #{Enum.join(unimplemented, ", ")}"
        )
      end
    end

    fixers =
      cond do
        only -> Enum.filter(@available_fixers, &(&1 in only))
        true -> Enum.reject(@available_fixers, &(&1 in exclude))
      end

    if Enum.empty?(fixers) do
      Mix.shell().error("No fixers selected. Available: #{Enum.join(@available_fixers, ", ")}")

      Mix.shell().info("Planned (not yet implemented): #{Enum.join(@planned_fixers, ", ")}")
      exit({:shutdown, 1})
    end

    Mix.shell().info("Running Credo fixers: #{Enum.join(fixers, ", ")}")

    if dry_run? do
      Mix.shell().info("[DRY RUN MODE - no files will be modified]")
      Mix.shell().info("Pass --apply to actually modify files.\n")
    end

    Enum.each(fixers, fn fixer ->
      Mix.shell().info("
=== #{format_fixer_name(fixer)} ===")
      run_fixer(fixer, paths, dry_run?, verbose?, length_style)
    end)

    Mix.shell().info("\nDone!")

    unless dry_run? do
      Mix.shell().info("Run `mix credo --strict` to verify fixes.")
    end
  end

  defp run_fixer(:trailing_whitespace, paths, dry_run?, verbose?, _) do
    args = build_args(paths, dry_run?, verbose?)
    TrailingWhitespace.run(args)
  end

  defp run_fixer(:large_numbers, paths, dry_run?, verbose?, _) do
    args = build_args(paths, dry_run?, verbose?)
    LargeNumbers.run(args)
  end

  defp run_fixer(:length_check, paths, dry_run?, verbose?, length_style) do
    args = build_args(paths, dry_run?, verbose?) ++ ["--style", length_style]
    LengthCheck.run(args)
  end

  defp run_fixer(:map_join, paths, dry_run?, verbose?, _) do
    args = build_args(paths, dry_run?, verbose?)
    MapJoin.run(args)
  end

  defp run_fixer(:alias_usage, paths, dry_run?, verbose?, _) do
    args = build_args(paths, dry_run?, verbose?)
    AliasUsage.run(args)
  end

  defp build_args(paths, dry_run?, verbose?) do
    args = paths

    args =
      if dry_run? do
        ["--dry-run" | args]
      else
        ["--apply" | args]
      end

    args =
      if verbose? do
        ["--verbose" | args]
      else
        args
      end

    args
  end

  defp format_fixer_name(:trailing_whitespace) do
    "Trailing Whitespace"
  end

  defp format_fixer_name(:large_numbers) do
    "Large Numbers"
  end

  defp format_fixer_name(:length_check) do
    "Length Check"
  end

  defp format_fixer_name(:map_join) do
    "Map Join"
  end

  defp format_fixer_name(:alias_usage) do
    "Alias Usage"
  end
end
