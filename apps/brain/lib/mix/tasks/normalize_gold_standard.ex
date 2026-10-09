defmodule Mix.Tasks.NormalizeGoldStandard do
  @moduledoc "Normalizes the intent gold standard by:\n1. Removing entries with empty text\n2. Deduplicating entries (keeping unique text, intent and follow-up triples)\n\nA follow-up utterance keeps its `followup_context` and `followup_variant`, so it\nis never merged with the same text said on its own. An intent label containing\n`\" - \"` is a contextual-variant name that was not collapsed into its intent, and\nraises.\n\n## Usage\n\n    # Preview changes (dry run)\n    mix normalize_gold_standard --dry-run\n\n    # Apply normalization\n    mix normalize_gold_standard\n\n    # Show detailed stats\n    mix normalize_gold_standard --stats\n"

  use Mix.Task
  require Logger

  alias Brain.ML.EvaluationStore

  @shortdoc "Normalize intent gold standard: drop empty texts and duplicates"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [dry_run: :boolean, stats: :boolean],
        aliases: [d: :dry_run, s: :stats]
      )

    dry_run? = Keyword.get(opts, :dry_run, false)
    stats_only? = Keyword.get(opts, :stats, false)

    Mix.Task.run("app.start")

    examples = EvaluationStore.load_gold_standard("intent")
    reject_variant_labels!(examples)

    if stats_only? do
      show_stats(examples)
    else
      normalize(examples, dry_run?)
    end
  end

  defp show_stats(examples) do
    IO.puts("\n=== Intent Gold Standard Statistics ===\n")

    intent_counts =
      examples
      |> Enum.group_by(& &1["intent"])
      |> Enum.map(fn {intent, items} -> {intent, length(items)} end)
      |> Enum.sort_by(fn {_, count} -> -count end)

    IO.puts("Total examples: #{length(examples)}")
    IO.puts("Unique intents: #{length(intent_counts)}")
    IO.puts("Follow-up examples: #{Enum.count(examples, &Map.has_key?(&1, "followup_context"))}\n")

    empty_count = Enum.count(examples, fn e -> e["text"] == "" or e["text"] == nil end)
    IO.puts("Empty text examples: #{empty_count}\n")
  end

  defp normalize(examples, dry_run?) do
    IO.puts("\n=== Normalizing Intent Gold Standard ===\n")
    non_empty = Enum.filter(examples, fn e -> e["text"] != "" and e["text"] != nil end)
    empty_removed = length(examples) - length(non_empty)
    IO.puts("Removed #{empty_removed} empty text entries")

    deduplicated =
      Enum.uniq_by(non_empty, fn e -> {e["text"], e["intent"], e["followup_context"], e["followup_variant"]} end)

    duplicates_removed = length(non_empty) - length(deduplicated)
    IO.puts("Removed #{duplicates_removed} duplicate entries")
    IO.puts("\n--- Summary ---")
    IO.puts("Original examples: #{length(examples)}")
    IO.puts("Final examples: #{length(deduplicated)}")
    IO.puts("Total removed: #{length(examples) - length(deduplicated)}")
    show_empty_intents(examples)

    if dry_run? do
      IO.puts("\n[DRY RUN] No changes written. Run without --dry-run to apply.")
    else
      path = EvaluationStore.gold_standard_path("intent")
      File.write!(path, Jason.encode!(deduplicated, pretty: true))
      IO.puts("\n✓ Gold standard updated: #{path}")
    end
  end

  defp reject_variant_labels!(examples) do
    case examples |> Enum.map(& &1["intent"]) |> Enum.filter(&(is_binary(&1) and String.contains?(&1, " - "))) |> Enum.uniq() do
      [] ->
        :ok

      labels ->
        Mix.raise("""
        normalize_gold_standard: #{length(labels)} intent label(s) name a contextual variant:

        #{Enum.map_join(labels, "\n", &"  #{&1}")}

        A follow-up belongs to the intent its action names, carrying followup_context
        and followup_variant; collapse it in data/intents and rebuild.
        """)
    end
  end

  defp show_empty_intents(original) do
    empty_intents =
      original
      |> Enum.filter(fn e -> e["text"] == "" or e["text"] == nil end)
      |> Enum.group_by(& &1["intent"])
      |> Enum.sort_by(fn {_, items} -> -length(items) end)

    if empty_intents != [] do
      IO.puts("\n--- Intents with Empty Text (removed) ---")

      Enum.each(empty_intents, fn {intent, items} ->
        IO.puts("  #{intent}: #{length(items)} empty examples removed")
      end)
    end
  end
end
