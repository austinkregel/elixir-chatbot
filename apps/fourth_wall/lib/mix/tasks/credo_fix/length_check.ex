defmodule Mix.Tasks.CredoFix.LengthCheck do
  @moduledoc """
  AST-based Mix task to fix expensive length checks flagged by Credo.

  This addresses Credo.Check.Warning.ExpensiveEmptyEnumCheck by replacing
  length(list) == 0 patterns with more efficient alternatives.

  ## IMPORTANT: AST-Based Implementation

  This fixer uses AST transformation (not regex) to ensure correctness.
  It uses the exact same pattern that Credo uses for detection:

      {:length, _, [_]}

  This pattern matches ONLY bare `length/1` calls, NOT:
  - `String.length/1`
  - `byte_size/1`
  - `tuple_size/1`
  - Any other qualified function calls

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.length_check

      # Actually apply the fixes
      mix credo_fix.length_check --apply

      # Use Enum.empty? style instead of pattern matching
      mix credo_fix.length_check --apply --style enum_empty

  ## Options

    --apply      Actually apply the fixes (required to modify files)
    --dry-run    Show what would be changed without modifying files (default)
    --verbose    Show each fix being applied
    --style      "pattern_match" (default) or "enum_empty"

  ## Transformations (pattern_match style)

      length(x) == 0  ->  x == []
      length(x) > 0   ->  x != []
      length(x) != 0  ->  x != []
      length(x) >= 1  ->  x != []
      length(x) < 1   ->  x == []
      0 == length(x)  ->  x == []
      0 < length(x)   ->  x != []
      1 <= length(x)  ->  x != []

  ## Transformations (enum_empty style)

      length(x) == 0  ->  Enum.empty?(x)
      length(x) > 0   ->  not Enum.empty?(x)
      length(x) != 0  ->  not Enum.empty?(x)
  """

  use Mix.Task
  require Logger

  alias FourthWall.AST
  alias FourthWall.Safety

  @shortdoc "Fix expensive length checks (AST-based)"

  @default_paths ["apps/"]
  @extensions [".ex", ".exs"]

  def run(args) do
    {opts, paths, _} =
      OptionParser.parse(args,
        strict: [
          apply: :boolean,
          dry_run: :boolean,
          verbose: :boolean,
          style: :string
        ],
        aliases: [
          d: :dry_run,
          v: :verbose,
          s: :style
        ]
      )

    apply? = Keyword.get(opts, :apply, false)
    dry_run? = not apply? or Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)
    style = Keyword.get(opts, :style, "pattern_match")

    target_paths = if Enum.empty?(paths), do: @default_paths, else: paths
    files = collect_files(target_paths)

    if Enum.empty?(files) do
      Mix.shell().info("No Elixir files found in #{inspect(target_paths)}")
    else
      if dry_run? do
        Mix.shell().info("[DRY RUN] Scanning #{length(files)} files for length checks...")
        Mix.shell().info("Pass --apply to actually modify files.\n")
      else
        Mix.shell().info("Scanning #{length(files)} files for length checks...")
      end

      results =
        files
        |> Enum.map(&process_file(&1, dry_run?, verbose?, style))
        |> Enum.filter(fn {_, changes} -> changes > 0 end)

      total_files = length(results)
      total_changes = Enum.reduce(results, 0, fn {_, c}, acc -> acc + c end)

      if dry_run? do
        Mix.shell().info("\n[DRY RUN] Would fix #{total_changes} length checks in #{total_files} files")
      else
        Mix.shell().info("\nFixed #{total_changes} length checks in #{total_files} files")
      end
    end
  end

  @doc """
  Process a single file to fix length checks.
  """
  def process_file(path, dry_run?, verbose?, style) do
    case File.read(path) do
      {:ok, content} ->
        {fixed_content, changes} = fix_length_checks(content, verbose?, path, style)

        if changes > 0 do
          unless dry_run? do
            # Safety check before writing
            case Safety.check_output(fixed_content) do
              {:ok, _} ->
                File.write!(path, fixed_content)

              {:error, issues} ->
                descriptions = Enum.map_join(issues, ", ", fn {_, desc} -> desc end)
                Mix.shell().error("Safety check failed for #{path}: #{descriptions}")
                Mix.shell().error("File NOT modified.")
            end
          end
        end

        {path, changes}

      {:error, reason} ->
        Mix.shell().error("Failed to read #{path}: #{inspect(reason)}")
        {path, 0}
    end
  end

  @doc """
  Fix length checks in content string using AST transformation.

  Returns `{fixed_content, num_changes}`.

  ## Examples

      iex> {output, 1} = Mix.Tasks.CredoFix.LengthCheck.fix_length_checks(
      ...>   "if length(list) == 0, do: :empty",
      ...>   false, "test.ex", "pattern_match"
      ...> )
      iex> output
      "if list == [], do: :empty"
  """
  def fix_length_checks(content, verbose?, path, style) when is_binary(content) do
    case AST.parse(content) do
      {:ok, ast} ->
        {new_ast, changes} = transform_length_checks(ast, style, verbose?, path)

        if changes > 0 do
          fixed_content = AST.to_source(new_ast)

          # Final safety check
          case Safety.check_output(fixed_content) do
            {:ok, safe_content} ->
              {safe_content, changes}

            {:error, _issues} ->
              # If safety check fails, return original content unchanged
              {content, 0}
          end
        else
          {content, 0}
        end

      {:error, _reason} ->
        # If parsing fails, return unchanged
        {content, 0}
    end
  end

  # Transform the AST to fix length checks
  defp transform_length_checks(ast, style, verbose?, path) do
    AST.transform(ast, 0, fn node, count ->
      case transform_node(node, style) do
        {:transformed, new_node} ->
          if verbose? do
            line = get_line(node)
            Mix.shell().info("  #{path}:#{line}: length check fixed")
          end

          {new_node, count + 1}

        :skip ->
          {node, count}
      end
    end)
  end

  # Pattern: length(x) == 0 -> x == [] or Enum.empty?(x)
  defp transform_node({:==, meta, [{:length, _, [arg]}, 0]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :empty)}
  end

  # Pattern: length(x) != 0 -> x != [] or not Enum.empty?(x)
  defp transform_node({:!=, meta, [{:length, _, [arg]}, 0]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :not_empty)}
  end

  # Pattern: length(x) > 0 -> x != [] or not Enum.empty?(x)
  defp transform_node({:>, meta, [{:length, _, [arg]}, 0]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :not_empty)}
  end

  # Pattern: length(x) < 1 -> x == [] or Enum.empty?(x)
  defp transform_node({:<, meta, [{:length, _, [arg]}, 1]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :empty)}
  end

  # Pattern: length(x) >= 1 -> x != [] or not Enum.empty?(x)
  defp transform_node({:>=, meta, [{:length, _, [arg]}, 1]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :not_empty)}
  end

  # Pattern: 0 == length(x) -> x == [] or Enum.empty?(x)
  defp transform_node({:==, meta, [0, {:length, _, [arg]}]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :empty)}
  end

  # Pattern: 0 < length(x) -> x != [] or not Enum.empty?(x)
  defp transform_node({:<, meta, [0, {:length, _, [arg]}]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :not_empty)}
  end

  # Pattern: 1 <= length(x) -> x != [] or not Enum.empty?(x)
  defp transform_node({:<=, meta, [1, {:length, _, [arg]}]}, style) do
    {:transformed, build_empty_check(arg, meta, style, :not_empty)}
  end

  # No match - skip this node
  defp transform_node(_node, _style) do
    :skip
  end

  # Build the replacement expression based on style and check type
  defp build_empty_check(arg, meta, "pattern_match", :empty) do
    {:==, meta, [arg, []]}
  end

  defp build_empty_check(arg, meta, "pattern_match", :not_empty) do
    {:!=, meta, [arg, []]}
  end

  defp build_empty_check(arg, meta, "enum_empty", :empty) do
    {{:., meta, [{:__aliases__, meta, [:Enum]}, :empty?]}, meta, [arg]}
  end

  defp build_empty_check(arg, meta, "enum_empty", :not_empty) do
    {:not, meta, [{{:., meta, [{:__aliases__, meta, [:Enum]}, :empty?]}, meta, [arg]}]}
  end

  defp build_empty_check(arg, meta, _unknown_style, check_type) do
    # Default to pattern_match style
    build_empty_check(arg, meta, "pattern_match", check_type)
  end

  defp get_line({_, meta, _}) when is_list(meta), do: Keyword.get(meta, :line, "?")
  defp get_line(_), do: "?"

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
      File.regular?(path) -> [path]
      File.dir?(path) -> Path.wildcard(Path.join(path, "**/*")) |> Enum.filter(&File.regular?/1)
      true -> Path.wildcard(path) |> Enum.filter(&File.regular?/1)
    end
  end

  defp elixir_file?(path), do: Path.extname(path) in @extensions
end
