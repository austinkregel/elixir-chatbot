defmodule Mix.Tasks.CredoFix.MapJoin do
  @moduledoc """
  AST-based Mix task to fix Enum.map |> Enum.join patterns flagged by Credo.

  This addresses Credo.Check.Refactor.MapJoin by converting
  `Enum.map(...) |> Enum.join(...)` to `Enum.map_join(...)`.

  ## IMPORTANT: AST-Based Implementation

  This fixer uses AST transformation (not regex) to ensure correctness.
  It matches the same patterns that Credo uses for detection.

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.map_join

      # Actually apply the fixes
      mix credo_fix.map_join --apply

  ## Options

    --apply      Actually apply the fixes (required to modify files)
    --dry-run    Show what would be changed without modifying files (default)
    --verbose    Show each fix being applied

  ## Transformations

      list |> Enum.map(fn) |> Enum.join(sep)
        -> list |> Enum.map_join(sep, fn)

      Enum.join(Enum.map(list, fn), sep)
        -> Enum.map_join(list, sep, fn)
  """

  use Mix.Task
  require Logger

  alias FourthWall.AST
  alias FourthWall.Safety

  @shortdoc "Fix Enum.map |> Enum.join patterns (AST-based)"

  @default_paths ["apps/"]
  @extensions [".ex", ".exs"]

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

    target_paths = if Enum.empty?(paths), do: @default_paths, else: paths
    files = collect_files(target_paths)

    if Enum.empty?(files) do
      Mix.shell().info("No Elixir files found in #{inspect(target_paths)}")
    else
      if dry_run? do
        Mix.shell().info("[DRY RUN] Scanning #{length(files)} files for map/join patterns...")
        Mix.shell().info("Pass --apply to actually modify files.\n")
      else
        Mix.shell().info("Scanning #{length(files)} files for map/join patterns...")
      end

      results =
        files
        |> Enum.map(&process_file(&1, dry_run?, verbose?))
        |> Enum.filter(fn {_, changes} -> changes > 0 end)

      total_files = length(results)
      total_changes = Enum.reduce(results, 0, fn {_, c}, acc -> acc + c end)

      if dry_run? do
        Mix.shell().info("\n[DRY RUN] Would fix #{total_changes} map/join patterns in #{total_files} files")
      else
        Mix.shell().info("\nFixed #{total_changes} map/join patterns in #{total_files} files")
      end
    end
  end

  @doc """
  Process a single file to fix map/join patterns.
  """
  def process_file(path, dry_run?, verbose?) do
    case File.read(path) do
      {:ok, content} ->
        {fixed_content, changes} = fix_map_join(content, verbose?, path)

        if changes > 0 do
          unless dry_run? do
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
  Fix map/join patterns in content string using AST transformation.

  Returns `{fixed_content, num_changes}`.
  """
  def fix_map_join(content, verbose?, path) when is_binary(content) do
    case AST.parse(content) do
      {:ok, ast} ->
        {new_ast, changes} = transform_map_join(ast, verbose?, path)

        if changes > 0 do
          fixed_content = AST.to_source(new_ast)

          case Safety.check_output(fixed_content) do
            {:ok, safe_content} ->
              {safe_content, changes}

            {:error, _issues} ->
              {content, 0}
          end
        else
          {content, 0}
        end

      {:error, _reason} ->
        {content, 0}
    end
  end

  defp transform_map_join(ast, verbose?, path) do
    AST.transform(ast, 0, fn node, count ->
      case transform_node(node) do
        {:transformed, new_node} ->
          if verbose? do
            line = get_line(node)
            Mix.shell().info("  #{path}:#{line}: map/join pattern fixed")
          end

          {new_node, count + 1}

        :skip ->
          {node, count}
      end
    end)
  end

  # Pattern 1: x |> Enum.map(fn) |> Enum.join(sep)
  # The innermost pipe is: {:|>, _, [x, Enum.map(fn)]}
  # The full expression is: {:|>, _, [{:|>, _, [x, Enum.map(fn)]}, Enum.join(sep)]}
  defp transform_node(
         {:|>, meta,
          [
            {:|>, _,
             [
               source,
               {{:., _, [{:__aliases__, _, [:Enum]}, :map]}, _, [map_fn]}
             ]},
            {{:., _, [{:__aliases__, _, [:Enum]}, :join]}, _, join_args}
          ]}
       ) do
    separator = extract_separator(join_args)

    new_node =
      {:|>, meta,
       [
         source,
         {{:., meta, [{:__aliases__, meta, [:Enum]}, :map_join]}, meta, [separator, map_fn]}
       ]}

    {:transformed, new_node}
  end

  # Pattern 2: Enum.map(x, fn) |> Enum.join(sep)
  defp transform_node(
         {:|>, meta,
          [
            {{:., _, [{:__aliases__, _, [:Enum]}, :map]}, _, [source, map_fn]},
            {{:., _, [{:__aliases__, _, [:Enum]}, :join]}, _, join_args}
          ]}
       ) do
    separator = extract_separator(join_args)

    new_node =
      {{:., meta, [{:__aliases__, meta, [:Enum]}, :map_join]}, meta,
       [source, separator, map_fn]}

    {:transformed, new_node}
  end

  # Pattern 3: Enum.join(Enum.map(source, fn), sep)
  defp transform_node(
         {{:., meta, [{:__aliases__, _, [:Enum]}, :join]}, _,
          [
            {{:., _, [{:__aliases__, _, [:Enum]}, :map]}, _, [source, map_fn]},
            separator
          ]}
       ) do
    new_node =
      {{:., meta, [{:__aliases__, meta, [:Enum]}, :map_join]}, meta,
       [source, separator, map_fn]}

    {:transformed, new_node}
  end

  # Pattern 4: list |> Enum.map(fn) directly piped to Enum.join(sep)
  # where join only has separator as argument (the mapped result is piped in)
  defp transform_node(
         {{:., meta, [{:__aliases__, _, [:Enum]}, :join]}, _,
          [
            {:|>, _, [source, {{:., _, [{:__aliases__, _, [:Enum]}, :map]}, _, [map_fn]}]},
            separator
          ]}
       ) do
    new_node =
      {{:., meta, [{:__aliases__, meta, [:Enum]}, :map_join]}, meta,
       [source, separator, map_fn]}

    {:transformed, new_node}
  end

  # No match
  defp transform_node(_node) do
    :skip
  end

  # Extract separator from join arguments
  # Enum.join() with no args defaults to ""
  # Enum.join(sep) uses the provided separator
  defp extract_separator([]), do: ""
  defp extract_separator([sep]), do: sep
  defp extract_separator(_), do: ""

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
