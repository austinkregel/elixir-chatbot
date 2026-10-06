defmodule Mix.Tasks.MigrateGoldStandard do
  @shortdoc "Report what a migration into the gold standard would produce (read-only)"
  @moduledoc """
  Reports what a migration from `data/intents/` and `data/training/intents/` into
  the gold standard, the intent registry and the response templates would produce.

  **This task no longer writes anything.** Three other tasks own those artifacts
  and build them from the same export under conventions this one contradicts:

  | artifact | owner |
  |---|---|
  | `priv/evaluation/intent/gold_standard.json` | `mix rebuild_gold_standard` |
  | `priv/analysis/intent_registry.json` | `mix registry.derive` |
  | `priv/response/templates.json` | `mix templates.reconcile` |

  Intents are named through `Brain.Corpus.Label.canonical/1`, the same rule the
  corpus uses: a ` - context: …` suffix names a context of an intent rather than an
  intent, and `smarthome.lights.*` folds to `smarthome.device.*`. The export holds
  242 usersays files, 40 of them suffixed, against the corpus's 194 intents, and
  `Pipeline.registered_intent?/1` gates accuracy on that vocabulary.

  Each writing path is refused with the reason and the task that owns the artifact.
  One artifact gets one writer.

  ## Usage

      mix migrate_gold_standard --list                        # intents across all sources
      mix migrate_gold_standard --preview                      # what a migration would produce
      mix migrate_gold_standard --select lights --preview      # only matching intents
      mix migrate_gold_standard --extract-metadata --preview   # slots and clarification prompts
      mix migrate_gold_standard --extract-templates --preview  # responses held in the export
      mix migrate_gold_standard --cleanup-sources --preview    # what deletion would remove

  `--limit N`, `--no-ner`, `--append`, `--exclude-context-variants` still shape
  what the reports show.
  """

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
    exclude_context_variants? = "--exclude-context-variants" in args
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
        preview_migration(select_filter, limit, exclude_context_variants?)

      true ->
        run_migration(select_filter, limit, !no_ner?, append?, destructive?, exclude_context_variants?)
    end
  end

  # This task and `mix rebuild_gold_standard` / `mix registry.derive` /
  # `mix templates.reconcile` generate the same three artifacts from the same
  # export, under conventions that contradict each other: this one keeps a
  # ` - context: …` variant as a separate intent, `root_label/1` folds it into its
  # base. The export holds 242 usersays files, 40 of them suffixed; the corpus
  # holds 194 intents. `registered_intent?/1` gates pipeline accuracy on that
  # vocabulary, so a registry built the other way rejects the labels the deployed
  # classifier emits.
  #
  # Every reporting path here still runs. The writing paths refuse, because each
  # one silently replaces a file another task owns:
  #
  #   --extract-templates   writes the whole map, not a merge. It extracts 94
  #                         intents / 241 templates against the 198 / 596 in
  #                         templates.json, so it removes 104 intents' responses.
  #   --extract-metadata    produces `required`, `optional` and
  #                         `clarification_templates`, all of which
  #                         `mix registry.derive` derives from the same export.
  #   --cleanup-sources     deletes data/intents, the canonical export. `data/` is
  #                         gitignored, so it cannot be recovered.
  defp refuse!(flag, owner, detail) do
    successor =
      case owner do
        nil -> "There is no replacement, because the artifact should not be deleted."
        task -> "#{task} owns this artifact now. Run that instead."
      end

    Mix.raise("""
    migrate_gold_standard #{flag} is disabled.

    #{detail}

    #{successor}

    The reporting paths still work: add --preview to see what this task would have
    produced, or use --list.
    """)
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

  defp preview_migration(select_filter, limit, exclude_context_variants?) do
    intent_names = resolve_intent_names(select_filter)

    IO.puts("
Previewing migration for #{length(intent_names)} intent(s)...")

    if exclude_context_variants? do
      IO.puts("NOTE: Context-variant usersays files will be excluded")
    end

    opts = [exclude_context_variants: exclude_context_variants?]

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

    # Nothing should reach here carrying a context suffix: intents are named by
    # `Brain.Corpus.Label.canonical/1` where they are read. A non-zero count means
    # a source spells one in a way the export cannot account for, which is worth
    # seeing rather than passing over.
    unresolved =
      intent_examples
      |> Enum.map(& &1["intent"])
      |> Enum.filter(&String.contains?(&1, "context_"))
      |> Enum.uniq()

    IO.puts("\nWould migrate:")
    IO.puts("  Intent examples: #{length(intent_examples)} (#{unique_intents} unique intents)")

    unless unresolved == [] do
      IO.puts("  UNRESOLVED labels (#{length(unresolved)}), still carrying a context suffix:")
      Enum.each(unresolved, fn label -> IO.puts("    #{label}") end)
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

      IO.puts("\nThis is a report only. Writing to intent_registry.json is disabled.")
    else
      refuse!(
        "--extract-metadata",
        "`mix registry.derive`",
        "It derives required, optional and clarification_templates from the same\n" <>
          "parameter definitions, plus the category/speech_act/description judgements\n" <>
          "this task cannot supply."
      )
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

      IO.puts("\nThis is a report only. Writing to priv/response/templates.json is disabled.")
    else
      refuse!(
        "--extract-templates",
        "`mix templates.reconcile`",
        "write_templates_json/2 replaces the file rather than merging into it, and\n" <>
          "this extraction covers fewer intents than the file already holds, so the\n" <>
          "difference would be deleted."
      )
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
        IO.puts("\nThis is a report only. Deleting these directories is disabled.\n")
      else
        refuse!(
          "--cleanup-sources",
          nil,
          "data/intents is the canonical Dialogflow export that mix rebuild_gold_standard\n" <>
            "reads, and data/ is gitignored, so deleting it cannot be undone. The annotated\n" <>
            "utterances under data/training/intents that no longer have a source are archived\n" <>
            "in data/archive/intent_annotations/, which is tracked."
        )
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

  defp run_migration(select_filter, limit, include_ner?, append?, destructive?, exclude_context_variants?) do
    refuse!(
      "migration",
      "`mix rebuild_gold_standard`",
      "Both rebuild priv/evaluation/intent/gold_standard.json from the same export.\n" <>
        "They now agree on the vocabulary -- both name intents through\n" <>
        "Brain.Corpus.Label.canonical/1 -- but one artifact still gets one writer,\n" <>
        "and rebuild_gold_standard is it: this path has no provenance, no\n" <>
        "needs_review for ambiguous texts, and no comparison against the previous\n" <>
        "corpus."
    )

    intent_names = resolve_intent_names(select_filter)

    mode =
      if destructive? do
        "DESTRUCTIVE"
      else
        "non-destructive"
      end

    intent_label =
      case intent_names do
        :all -> "all"
        names -> "#{length(names)}"
      end

    IO.puts("
Migrating #{intent_label} intent(s) [#{mode} mode]...")

    if exclude_context_variants? do
      IO.puts("NOTE: Context-variant usersays files will be excluded")
    end

    if destructive? do
      IO.puts("WARNING: Source files will be deleted after successful migration!")
      Process.sleep(1000)
    end

    opts = [
      include_ner: include_ner?,
      append: append?,
      destructive: destructive?,
      exclude_context_variants: exclude_context_variants?
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