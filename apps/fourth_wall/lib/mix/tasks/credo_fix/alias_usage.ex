defmodule Mix.Tasks.CredoFix.AliasUsage do
  @moduledoc """
  AST-based Mix task to fix nested module alias issues flagged by Credo.

  This addresses Credo.Check.Design.AliasUsage by adding alias statements
  for nested modules and replacing full module paths with short names.

  ## IMPORTANT: AST-Based Implementation

  This fixer uses AST transformation to:
  1. Find all nested module references (e.g., MyApp.Services.UserService)
  2. Add alias statements at the top of the module
  3. Replace full paths with short names

  ## Usage

      # Preview changes (dry run - default)
      mix credo_fix.alias_usage

      # Actually apply the fixes
      mix credo_fix.alias_usage --apply

  ## Options

    --apply      Actually apply the fixes (required to modify files)
    --dry-run    Show what would be changed without modifying files (default)
    --verbose    Show each fix being applied

  ## Excluded Modules

  The following are not aliased (matching Credo's defaults):
  - Single-part modules (String, Enum, etc.)
  - Erlang modules (:ets, :gen_server, etc.)
  - Standard library namespaces (File, IO, Kernel, etc.)
  """

  use Mix.Task
  require Logger

  alias FourthWall.AST
  alias FourthWall.Safety

  @shortdoc "Fix nested module alias usage (AST-based)"

  @default_paths ["apps/"]
  @extensions [".ex", ".exs"]

  # Modules that should not be aliased (similar to Credo defaults)
  @excluded_namespaces ~w[File IO Inspect Kernel Macro Supervisor Task Version]
  @excluded_lastnames ~w[Access Agent Application Atom Base Behaviour
                         Bitwise Code Date DateTime Dict Enum Exception
                         File Float GenEvent GenServer HashDict HashSet
                         Integer IO Kernel Keyword List Macro Map MapSet
                         Module NaiveDateTime Node OptionParser Path Port
                         Process Protocol Range Record Regex Registry Set
                         Stream String StringIO Supervisor System Task Time
                         Tuple URI Version]

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
        Mix.shell().info("[DRY RUN] Scanning #{length(files)} files for alias usage...")
        Mix.shell().info("Pass --apply to actually modify files.\n")
      else
        Mix.shell().info("Scanning #{length(files)} files for alias usage...")
      end

      results =
        files
        |> Enum.map(&process_file(&1, dry_run?, verbose?))
        |> Enum.filter(fn {_, changes} -> changes > 0 end)

      total_files = length(results)
      total_changes = Enum.reduce(results, 0, fn {_, c}, acc -> acc + c end)

      if dry_run? do
        Mix.shell().info("\n[DRY RUN] Would add #{total_changes} aliases in #{total_files} files")
      else
        Mix.shell().info("\nAdded #{total_changes} aliases in #{total_files} files")
      end
    end
  end

  def process_file(path, dry_run?, verbose?) do
    case File.read(path) do
      {:ok, content} ->
        {fixed_content, changes} = fix_alias_usage(content, verbose?, path)

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
  Fix alias usage in content string using AST transformation.

  Returns `{fixed_content, num_changes}`.
  """
  def fix_alias_usage(content, verbose?, path) when is_binary(content) do
    case AST.parse(content) do
      {:ok, ast} ->
        {new_ast, changes} = transform_modules(ast, verbose?, path)

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

  defp transform_modules(ast, verbose?, path) do
    # Process each defmodule separately
    AST.transform(ast, 0, fn
      {:defmodule, meta, [module_name, [do: body]]} = _node, count ->
        {new_body, new_count} = process_module_body(body, verbose?, path)
        {{:defmodule, meta, [module_name, [do: new_body]]}, count + new_count}

      node, count ->
        {node, count}
    end)
  end

  defp process_module_body(body, verbose?, path) do
    # Step 1: Find existing aliases
    existing_aliases = find_existing_aliases(body)
    existing_lastnames = MapSet.new(existing_aliases, &get_lastname/1)

    # Step 2: Find all nested module usages
    module_usages = find_module_usages(body)

    # Step 3: Filter to aliasable modules
    aliasable =
      module_usages
      |> Enum.filter(&should_alias?(&1, existing_lastnames))
      |> Enum.uniq_by(&get_lastname/1)

    if Enum.empty?(aliasable) do
      {body, 0}
    else
      if verbose? do
        for mod <- aliasable do
          Mix.shell().info("  #{path}: aliasing #{inspect(mod)}")
        end
      end

      # Step 4: Build alias statements
      alias_stmts = Enum.map(aliasable, &build_alias_stmt/1)

      # Step 5: Replace usages with short names
      replacements = Map.new(aliasable, fn mod -> {mod, get_lastname(mod)} end)
      new_body = replace_module_usages(body, replacements)

      # Step 6: Insert aliases at appropriate location
      final_body = insert_aliases(new_body, alias_stmts)

      {final_body, length(aliasable)}
    end
  end

  defp find_existing_aliases(body) do
    {_, aliases} =
      AST.transform(body, [], fn
        # Simple alias: alias Brain.Analysis.InternalModel
        {:alias, _, [{:__aliases__, _, parts}]}, acc when is_list(parts) ->
          {{:alias, [], [{:__aliases__, [], parts}]}, [parts | acc]}

        # alias with :as option: alias Brain.Analysis.InternalModel, as: Model
        {:alias, _, [{:__aliases__, _, parts}, _opts]}, acc when is_list(parts) ->
          {{:alias, [], [{:__aliases__, [], parts}]}, [parts | acc]}

        # Grouped alias: alias Brain.Analysis.{InternalModel, SpeechActResult}
        # AST: {:alias, _, [{{:., _, [{:__aliases__, _, prefix}, :{}]}, _, suffixes}]}
        {:alias, _,
         [
           {{:., _, [{:__aliases__, _, prefix_parts}, :{}]}, _, suffix_aliases}
         ]},
        acc
        when is_list(prefix_parts) and is_list(suffix_aliases) ->
          # Extract full module paths for each suffix
          expanded_aliases =
            Enum.map(suffix_aliases, fn
              {:__aliases__, _, suffix_parts} when is_list(suffix_parts) ->
                prefix_parts ++ suffix_parts

              _ ->
                nil
            end)
            |> Enum.reject(&is_nil/1)

          {{:alias, [], []}, expanded_aliases ++ acc}

        node, acc ->
          {node, acc}
      end)

    aliases
  end

  defp find_module_usages(body) do
    {_, usages} =
      AST.transform(body, [], fn
        # Skip alias statements entirely - they're not "usages" to be aliased
        {:alias, _, _} = node, acc ->
          {node, acc}

        # Module.function() call - only match actual function calls
        # Exclude :{} which is the special atom for grouped aliases
        {{:., _, [{:__aliases__, _, parts}, fun]}, _, _} = node, acc
        when is_list(parts) and is_atom(fun) and fun != :{} ->
          if length(parts) > 1 do
            {node, [parts | acc]}
          else
            {node, acc}
          end

        node, acc ->
          {node, acc}
      end)

    usages |> Enum.uniq()
  end

  defp should_alias?(parts, existing_lastnames) when is_list(parts) do
    # Must have at least 2 parts
    length(parts) > 1 and
      # First part not in excluded namespaces
      not Enum.member?(@excluded_namespaces, to_string(List.first(parts))) and
      # Last part not in excluded lastnames
      not Enum.member?(@excluded_lastnames, to_string(List.last(parts))) and
      # Last name not already aliased
      not MapSet.member?(existing_lastnames, List.last(parts))
  end

  defp get_lastname(parts) when is_list(parts), do: List.last(parts)

  defp build_alias_stmt(parts) do
    {:alias, [], [{:__aliases__, [], parts}]}
  end

  defp replace_module_usages(body, replacements) do
    # We need to manually walk the AST to skip alias nodes entirely
    # because Macro.prewalk continues into children even after we return
    do_replace_module_usages(body, replacements)
  end

  # Skip alias statements entirely - don't traverse into them
  defp do_replace_module_usages({:alias, meta, args}, _replacements) do
    {:alias, meta, args}
  end

  # Module.function() call - replace the module part
  defp do_replace_module_usages(
         {{:., dot_meta, [{:__aliases__, alias_meta, parts}, fun]}, call_meta, args},
         replacements
       )
       when is_list(parts) and is_atom(fun) do
    # Recursively process the arguments first
    new_args = Enum.map(args, &do_replace_module_usages(&1, replacements))

    case Map.get(replacements, parts) do
      nil ->
        {{:., dot_meta, [{:__aliases__, alias_meta, parts}, fun]}, call_meta, new_args}

      short_name ->
        {{:., dot_meta, [{:__aliases__, alias_meta, [short_name]}, fun]}, call_meta, new_args}
    end
  end

  # Tuple nodes (most AST nodes) - recursively process
  defp do_replace_module_usages({form, meta, args}, replacements) when is_list(args) do
    new_form = do_replace_module_usages(form, replacements)
    new_args = Enum.map(args, &do_replace_module_usages(&1, replacements))
    {new_form, meta, new_args}
  end

  # Two-tuples (keyword lists, do blocks)
  defp do_replace_module_usages({key, value}, replacements) do
    {do_replace_module_usages(key, replacements), do_replace_module_usages(value, replacements)}
  end

  # Lists
  defp do_replace_module_usages(list, replacements) when is_list(list) do
    Enum.map(list, &do_replace_module_usages(&1, replacements))
  end

  # Atoms, numbers, strings, etc. - return as is
  defp do_replace_module_usages(other, _replacements), do: other

  defp insert_aliases(body, alias_stmts) do
    # Try to insert after @moduledoc if present, otherwise at the start
    case body do
      {:__block__, meta, statements} ->
        # Find the insertion point (after moduledoc/doc attributes)
        {before, after_docs} = split_at_insertion_point(statements)
        {:__block__, meta, before ++ alias_stmts ++ after_docs}

      single_expr ->
        # Single expression body, wrap in block with aliases
        {:__block__, [], alias_stmts ++ [single_expr]}
    end
  end

  defp split_at_insertion_point(statements) do
    # Split after @moduledoc and @doc attributes, and existing aliases
    Enum.split_while(statements, fn
      {:@, _, [{:moduledoc, _, _}]} -> true
      {:@, _, [{:doc, _, _}]} -> true
      {:alias, _, _} -> true
      _ -> false
    end)
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
      File.regular?(path) -> [path]
      File.dir?(path) -> Path.wildcard(Path.join(path, "**/*")) |> Enum.filter(&File.regular?/1)
      true -> Path.wildcard(path) |> Enum.filter(&File.regular?/1)
    end
  end

  defp elixir_file?(path), do: Path.extname(path) in @extensions
end
