defmodule Mix.Tasks.MigrateGoldStandard do
  @shortdoc "Migrate intent data into gold standard evaluation files"
  @moduledoc "Migrates intent training data from multiple sources into the gold standard\nevaluation files. Supports destructive mode to delete source files.\n\n## Data Sources\n\n- `data/intents/*_usersays_en.json` - Dialogflow usersays files\n- `data/training/intents/*.json` - Enriched training data\n\n## Context Variants\n\nContext variants (e.g., `account.balance.check.context_.balance`) are follow-up\nutterances that require prior conversational context. By default, they are kept\nas separate intents because they have different semantic meanings:\n\n- Base intent: \"check my credit card balance\" (self-contained)\n- Context variant: \"how much money\" (requires prior context)\n\nUse `--merge-contexts` only if you want to flatten them into base intents.\n\n## Usage\n\n    mix migrate_gold_standard                    # Migrate all intents\n    mix migrate_gold_standard --preview          # Preview without writing\n    mix migrate_gold_standard --destructive      # Migrate and delete source files\n    mix migrate_gold_standard --merge-contexts   # Merge context variants into base intents\n    mix migrate_gold_standard --limit 5          # Max 5 examples per intent\n    mix migrate_gold_standard --select lights    # Only intents containing \"lights\"\n    mix migrate_gold_standard --no-ner           # Skip NER gold standard\n    mix migrate_gold_standard --append           # Append instead of replace\n    mix migrate_gold_standard --list             # List available intents\n\n## Examples\n\n    # See what's available\n    mix migrate_gold_standard --list\n\n    # Preview migration of lighting intents\n    mix migrate_gold_standard --select lights --preview\n\n    # Full destructive migration (migrate all, delete sources)\n    mix migrate_gold_standard --destructive\n\n    # Merge context variants into base intents\n    mix migrate_gold_standard --merge-contexts --preview\n\n    # Extract intent metadata to intent_registry.json\n    mix migrate_gold_standard --extract-metadata --preview\n    mix migrate_gold_standard --extract-metadata\n\n    # Extract response templates to templates.json\n    mix migrate_gold_standard --extract-templates --preview\n    mix migrate_gold_standard --extract-templates\n\n    # Delete source directories after migration is complete\n    mix migrate_gold_standard --cleanup-sources --preview\n    mix migrate_gold_standard --cleanup-sources\n"

  use Mix.Task

  alias Brain.ML.GoldStandardMigrator

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    list? = "--list" in args
    preview? = "--preview" in args
    no_ner? = "--no-ner" in args
    append? = "--append" in args
    destructive? = "--destructive" in args
    merge_contexts? = "--merge-contexts" in args
    extract_metadata? = "--extract-metadata" in args
    extract_templates? = "--extract-templates" in args
    cleanup? = "--cleanup-sources" in args

    limit = parse_limit(args)
    select_filter = parse_select(args)

    cond do
      list? ->
        list_intents()

      extract_metadata? ->
        extract_and_merge_metadata(preview?)

      extract_templates? ->
        extract_response_templates(preview?)

      cleanup? ->
        cleanup_source_directories(preview?)

      preview? ->
        preview_migration(select_filter, limit, merge_contexts?)

      true ->
        run_migration(select_filter, limit, !no_ner?, append?, destructive?, merge_contexts?)
    end
  end

  defp list_intents do
    grouped = GoldStandardMigrator.list_available_intents_grouped()
    stats = GoldStandardMigrator.gold_standard_stats()

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("AVAILABLE INTENTS FROM ALL SOURCES")
    IO.puts(String.duplicate("=", 60))

    total_intents = 0
    total_examples = 0

    {total_intents, total_examples} =
      Enum.reduce(grouped, {total_intents, total_examples}, fn {group, intents}, {ti, te} ->
        group_examples = Enum.sum(Enum.map(intents, & &1.example_count))
        IO.puts("
  #{group} (#{length(intents)} intents, #{group_examples} examples)")

        Enum.each(intents, fn intent ->
          sources = intent.sources |> Enum.map_join(", ", &to_string/1)
          IO.puts("    #{intent.name} (#{intent.example_count} examples) [#{sources}]")
        end)

        {ti + length(intents), te + group_examples}
      end)

    IO.puts("\n" <> String.duplicate("-", 60))
    IO.puts("Total: #{total_intents} intents, #{total_examples} examples")

    IO.puts("\nCurrent gold standard sizes:")

    Enum.each(stats, fn {task, count} ->
      IO.puts("  #{task}: #{count} examples")
    end)

    IO.puts("")
  end

  defp preview_migration(select_filter, limit, merge_contexts?) do
    intent_names = resolve_intent_names(select_filter)

    IO.puts("
Previewing migration for #{length(intent_names)} intent(s)...")

    if merge_contexts? do
      IO.puts("NOTE: Context variants will be merged into base intents")
    end

    opts = [merge_context_variants: merge_contexts?]

    opts =
      if limit do
        Keyword.put(opts, :limit, limit)
      else
        opts
      end

    {intent_examples, entity_examples, source_files} =
      GoldStandardMigrator.preview(intent_names, opts)

    non_empty_ner = Enum.count(entity_examples, fn e -> e["expected"] != [] end)
    unique_intents = intent_examples |> Enum.map(& &1["intent"]) |> Enum.uniq() |> length()

    context_variant_count =
      intent_examples
      |> Enum.map(& &1["intent"])
      |> Enum.count(&String.contains?(&1, "context_"))

    IO.puts("\nWould migrate:")
    IO.puts("  Intent examples: #{length(intent_examples)} (#{unique_intents} unique intents)")

    if context_variant_count > 0 and not merge_contexts? do
      IO.puts(
        "  Context variants: #{context_variant_count} examples (use --merge-contexts to merge)"
      )
    end

    IO.puts("  NER examples:    #{non_empty_ner}")
    IO.puts("  Source files:    #{length(source_files)}")

    if intent_examples != [] do
      IO.puts("\nSample intent examples:")

      intent_examples
      |> Enum.take(10)
      |> Enum.each(fn ex ->
        richness =
          if ex["tokens"] do
            " [enriched]"
          else
            ""
          end

        IO.puts("  [#{ex["intent"]}] #{ex["text"]}#{richness}")
      end)

      if length(intent_examples) > 10 do
        IO.puts("  ... and #{length(intent_examples) - 10} more")
      end
    end

    IO.puts("")
  end

  defp extract_and_merge_metadata(preview?) do
    IO.puts("\nExtracting intent metadata from definition files...")

    extracted = GoldStandardMigrator.extract_intent_metadata()

    IO.puts("Extracted metadata for #{map_size(extracted)} intents")

    if preview? do
      IO.puts("\nSample extracted metadata:")

      extracted
      |> Enum.take(5)
      |> Enum.each(fn {intent_name, metadata} ->
        required = Map.get(metadata, "required", [])
        optional = Map.get(metadata, "optional", [])
        templates = Map.get(metadata, "clarification_templates", %{})

        IO.puts("
  #{intent_name}:")
        IO.puts("    required: #{inspect(required)}")
        IO.puts("    optional: #{inspect(optional)}")

        if map_size(templates) > 0 do
          IO.puts("    clarifications: #{map_size(templates)}")
        end
      end)

      if map_size(extracted) > 5 do
        IO.puts("
  ... and #{map_size(extracted) - 5} more")
      end

      IO.puts("\nRun without --preview to write to intent_registry.json")
    else
      {:ok, merged} = GoldStandardMigrator.merge_into_intent_registry(extracted, write: true)
      IO.puts("Merged into intent_registry.json (#{map_size(merged)} total intents)")
      IO.puts("\nRun `mix migrate_gold_standard --destructive` to complete migration.\n")
    end
  end

  defp extract_response_templates(preview?) do
    IO.puts("\nExtracting response templates from definition files...")

    templates = GoldStandardMigrator.extract_response_templates()

    total_templates =
      templates
      |> Map.values()
      |> Enum.map(fn entry -> length(Map.get(entry, "templates", [])) end)
      |> Enum.sum()

    IO.puts("Extracted #{total_templates} templates for #{map_size(templates)} intents")

    if preview? do
      IO.puts("\nSample extracted templates:")

      templates
      |> Enum.take(5)
      |> Enum.each(fn {intent_name, entry} ->
        tpls = Map.get(entry, "templates", [])
        sources = tpls |> Enum.map(& &1["source"]) |> Enum.uniq() |> Enum.join(", ")

        IO.puts("
  #{intent_name}: (#{length(tpls)} templates, sources: #{sources})")

        tpls
        |> Enum.take(2)
        |> Enum.each(fn t ->
          text = String.slice(t["text"], 0, 60)

          text =
            if String.length(t["text"]) > 60 do
              text <> "..."
            else
              text
            end

          IO.puts("    - #{text}")
        end)

        if length(tpls) > 2 do
          IO.puts("    - ... and #{length(tpls) - 2} more")
        end
      end)

      if map_size(templates) > 5 do
        IO.puts("
  ... and #{map_size(templates) - 5} more intents")
      end

      IO.puts("\nRun without --preview to write to priv/response/templates.json")
    else
      {:ok, path} = GoldStandardMigrator.write_templates_json(templates, write: true)
      IO.puts("Wrote to #{path}")
      IO.puts("\nRun `mix migrate_gold_standard --destructive` to complete migration.\n")
    end
  end

  defp cleanup_source_directories(preview?) do
    IO.puts("\nSource directories to be deleted:")

    project_root =
      Application.app_dir(:brain)
      |> Path.join("../../../..")
      |> Path.expand()

    dirs = [
      Path.join(project_root, "data/intents"),
      Path.join(project_root, "data/training/intents"),
      Path.join(project_root, "data/legacy/intents")
    ]

    custom_file = Path.join(project_root, "data/customSmalltalkResponses_en.json")

    existing_dirs = Enum.filter(dirs, &File.dir?/1)
    custom_exists = File.exists?(custom_file)

    if existing_dirs == [] and not custom_exists do
      IO.puts("  No source directories found - already cleaned up?")
    else
      Enum.each(existing_dirs, fn dir ->
        file_count = count_files(dir)
        IO.puts("  #{dir} (#{file_count} files)")
      end)

      if custom_exists do
        IO.puts("  #{custom_file}")
      end

      if preview? do
        IO.puts("\nRun without --preview to delete these directories.\n")
      else
        IO.puts("\nWARNING: This will permanently delete these directories!")
        Process.sleep(2000)

        {:ok, deleted} = GoldStandardMigrator.delete_source_directories()

        IO.puts("
Deleted #{length(deleted)} items:")

        Enum.each(deleted, fn item ->
          IO.puts("  - #{item}")
        end)

        IO.puts("")
      end
    end
  end

  defp count_files(dir) do
    try do
      Path.wildcard(Path.join(dir, "**/*.json")) |> length()
    rescue
      _ -> 0
    end
  end

  defp run_migration(select_filter, limit, include_ner?, append?, destructive?, merge_contexts?) do
    intent_names = resolve_intent_names(select_filter)

    mode =
      if destructive? do
        "DESTRUCTIVE"
      else
        "non-destructive"
      end

    IO.puts("
Migrating #{length(intent_names)} intent(s) [#{mode} mode]...")

    if merge_contexts? do
      IO.puts("NOTE: Context variants will be merged into base intents")
    end

    if destructive? do
      IO.puts("WARNING: Source files will be deleted after successful migration!")
      Process.sleep(1000)
    end

    opts = [
      include_ner: include_ner?,
      append: append?,
      destructive: destructive?,
      merge_context_variants: merge_contexts?
    ]

    opts =
      if limit do
        Keyword.put(opts, :limit, limit)
      else
        opts
      end

    {:ok, %{intent_count: ic, ner_count: nc, deleted_files: deleted}} =
      GoldStandardMigrator.migrate_intents(intent_names, opts)

    IO.puts("\nMigration complete:")
    IO.puts("  Intent examples: #{ic}")
    IO.puts("  NER examples:    #{nc}")

    if deleted != [] do
      IO.puts("  Files deleted:   #{length(deleted)}")
    end

    stats = GoldStandardMigrator.gold_standard_stats()
    IO.puts("\nUpdated gold standard sizes:")

    Enum.each(stats, fn {task, count} ->
      IO.puts("  #{task}: #{count} examples")
    end)

    IO.puts("\nRun `mix evaluate --save` to evaluate against the new gold standard.\n")
  end

  defp resolve_intent_names(nil) do
    :all
  end

  defp resolve_intent_names(filter) do
    filter_lower = String.downcase(filter)

    GoldStandardMigrator.list_available_intents()
    |> Enum.filter(fn intent ->
      String.downcase(intent.name) |> String.contains?(filter_lower)
    end)
    |> Enum.map(& &1.name)
  end

  defp parse_limit(args) do
    case Enum.find_index(args, &(&1 == "--limit")) do
      nil ->
        nil

      idx ->
        case Enum.at(args, idx + 1) do
          nil -> nil
          val -> parse_int(val)
        end
    end
  end

  defp parse_select(args) do
    case Enum.find_index(args, &(&1 == "--select")) do
      nil -> nil
      idx -> Enum.at(args, idx + 1)
    end
  end

  defp parse_int(val) do
    case Integer.parse(val) do
      {n, _} when n > 0 -> n
      _ -> nil
    end
  end
end