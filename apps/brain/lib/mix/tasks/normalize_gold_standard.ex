defmodule Mix.Tasks.NormalizeGoldStandard do
  @moduledoc """
  Normalizes the intent gold standard by:
  1. Removing context variant suffixes (e.g., "music.play - context_play-music" -> "music.play")
  2. Removing entries with empty text
  3. Deduplicating entries (keeping unique text-intent pairs)

  ## Usage

      # Preview changes (dry run)
      mix normalize_gold_standard --dry-run

      # Apply normalization
      mix normalize_gold_standard

      # Show detailed stats
      mix normalize_gold_standard --stats
  """

  use Mix.Task
  require Logger

  alias Brain.ML.EvaluationStore

  @shortdoc "Normalize intent gold standard by merging context variants"

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

    if stats_only? do
      show_stats(examples)
    else
      normalize(examples, dry_run?)
    end
  end

  defp show_stats(examples) do
    IO.puts("\n=== Intent Gold Standard Statistics ===\n")

    # Count by intent
    intent_counts =
      examples
      |> Enum.group_by(& &1["intent"])
      |> Enum.map(fn {intent, items} -> {intent, length(items)} end)
      |> Enum.sort_by(fn {_, count} -> -count end)

    IO.puts("Total examples: #{length(examples)}")
    IO.puts("Unique intents: #{length(intent_counts)}\n")

    # Context variants
    context_variants =
      Enum.filter(intent_counts, fn {intent, _} ->
        String.contains?(intent, " - context")
      end)

    IO.puts("Context variant intents: #{length(context_variants)}")

    context_example_count =
      context_variants |> Enum.map(fn {_, c} -> c end) |> Enum.sum()

    IO.puts("Context variant examples: #{context_example_count}\n")

    # Empty text
    empty_count = Enum.count(examples, fn e -> e["text"] == "" or e["text"] == nil end)
    IO.puts("Empty text examples: #{empty_count}\n")

    # Show context variants
    if length(context_variants) > 0 do
      IO.puts("--- Context Variant Intents ---")

      Enum.each(context_variants, fn {intent, count} ->
        base = extract_base_intent(intent)
        IO.puts("  #{intent}")
        IO.puts("    -> base: #{base}, examples: #{count}")
      end)
    end
  end

  defp normalize(examples, dry_run?) do
    IO.puts("\n=== Normalizing Intent Gold Standard ===\n")

    # Step 1: Remove empty text entries
    non_empty = Enum.filter(examples, fn e -> e["text"] != "" and e["text"] != nil end)
    empty_removed = length(examples) - length(non_empty)
    IO.puts("Removed #{empty_removed} empty text entries")

    # Step 2: Normalize context variants to base intents
    normalized =
      Enum.map(non_empty, fn example ->
        original_intent = example["intent"]
        base_intent = extract_base_intent(original_intent)

        if original_intent != base_intent do
          Map.put(example, "intent", base_intent)
        else
          example
        end
      end)

    context_normalized =
      Enum.count(non_empty, fn e -> String.contains?(e["intent"], " - context") end)

    IO.puts("Normalized #{context_normalized} context variant examples")

    # Step 3: Deduplicate by text+intent
    deduplicated =
      normalized
      |> Enum.uniq_by(fn e -> {e["text"], e["intent"]} end)

    duplicates_removed = length(normalized) - length(deduplicated)
    IO.puts("Removed #{duplicates_removed} duplicate entries")

    # Summary
    IO.puts("\n--- Summary ---")
    IO.puts("Original examples: #{length(examples)}")
    IO.puts("Final examples: #{length(deduplicated)}")
    IO.puts("Total removed/merged: #{length(examples) - length(deduplicated)}")

    # Show before/after intent counts for affected intents
    IO.puts("\n--- Affected Intents ---")
    show_affected_intents(examples, deduplicated)

    if dry_run? do
      IO.puts("\n[DRY RUN] No changes written. Run without --dry-run to apply.")
    else
      path = EvaluationStore.gold_standard_path("intent")
      File.write!(path, Jason.encode!(deduplicated, pretty: true))
      IO.puts("\n✓ Gold standard updated: #{path}")
    end
  end

  defp extract_base_intent(intent) do
    # Handle patterns like:
    # "music.play - context_play-music" -> "music.play"
    # "smarthome - context_room - comment_specify or change room" -> "smarthome"
    # "account.balance.check - context_ account" -> "account.balance.check"
    intent
    |> String.split(" - context")
    |> List.first()
    |> String.trim()
  end

  defp show_affected_intents(original, normalized) do
    original_counts =
      original
      |> Enum.group_by(& &1["intent"])
      |> Enum.into(%{}, fn {k, v} -> {k, length(v)} end)

    normalized_counts =
      normalized
      |> Enum.group_by(& &1["intent"])
      |> Enum.into(%{}, fn {k, v} -> {k, length(v)} end)

    # Find base intents that gained examples from context variants
    base_intents_affected =
      original_counts
      |> Map.keys()
      |> Enum.filter(&String.contains?(&1, " - context"))
      |> Enum.map(&extract_base_intent/1)
      |> Enum.uniq()

    Enum.each(base_intents_affected, fn base ->
      before_base = Map.get(original_counts, base, 0)
      after_base = Map.get(normalized_counts, base, 0)
      gained = after_base - before_base

      if gained > 0 do
        IO.puts("  #{base}: #{before_base} -> #{after_base} (+#{gained})")
      end
    end)

    # Also show intents with empty text that were removed
    empty_intents =
      original
      |> Enum.filter(fn e -> e["text"] == "" or e["text"] == nil end)
      |> Enum.group_by(& &1["intent"])
      |> Enum.sort_by(fn {_, items} -> -length(items) end)

    if length(empty_intents) > 0 do
      IO.puts("\n--- Intents with Empty Text (removed) ---")

      Enum.each(empty_intents, fn {intent, items} ->
        IO.puts("  #{intent}: #{length(items)} empty examples removed")
      end)
    end
  end
end
