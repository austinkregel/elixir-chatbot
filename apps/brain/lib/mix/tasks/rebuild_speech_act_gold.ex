defmodule Mix.Tasks.Rebuild.SpeechActGold do
  @shortdoc "Rebuild speech act gold standard with structural anchoring and confidence gating"
  @moduledoc """
  Rebuilds the speech act gold standard using a two-layer labeling strategy:

  1. **Structural anchoring** — high-confidence relabeling based on punctuation,
     imperative verbs, and commissive markers
  2. **Confidence-gated classifier** — for unanchored texts, use the speech act
     classifier when confidence >= 0.7, otherwise mark `"needs_review"`

  Examples with `"needs_review"` status are excluded from the output file and
  written separately to `needs_review.json`.

  ## Usage

      mix rebuild.speech_act_gold              # Rebuild
      mix rebuild.speech_act_gold --dry-run    # Report changes without writing
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.SpeechActClassifier
  alias Brain.ML.{EvaluationStore, MicroClassifiers, Tokenizer}

  @imperative_verbs ~w(tell show give bring let make do go take send put find get help call open close turn set play stop start)

  @greeting_emotion_words ~w(hello hi hey howdy yo greetings wow oh yay hooray congrats congratulations bravo cheers thanks thank awesome amazing great wonderful fantastic excellent beautiful lovely nice good)

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    dry_run? = "--dry-run" in args

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.\n")

    IO.puts(String.duplicate("=", 60))
    IO.puts("REBUILD SPEECH ACT GOLD STANDARD")
    IO.puts(String.duplicate("=", 60) <> "\n")

    gold = EvaluationStore.load_gold_standard("speech_act")

    if gold == [] do
      Mix.raise("No speech act gold standard data found. Cannot rebuild.")
    end

    IO.puts("  Gold standard: #{length(gold)} entries\n")

    before_dist = label_distribution(gold, "speech_act")

    {rebuilt, stats} = rebuild(gold)

    {accepted, needs_review} =
      Enum.split_with(rebuilt, fn entry -> entry["_status"] != "needs_review" end)

    clean_accepted = Enum.map(accepted, &Map.delete(&1, "_status"))
    clean_review = Enum.map(needs_review, &Map.delete(&1, "_status"))

    after_dist = label_distribution(clean_accepted, "speech_act")

    print_summary(stats, length(gold), before_dist, after_dist)

    unless dry_run? do
      write_output(clean_accepted, gold_path())
      write_output(clean_review, review_path())
    else
      IO.puts("  Dry-run mode. Omit --dry-run to write files.\n")
    end
  end

  # ---------------------------------------------------------------------------
  # Rebuild logic
  # ---------------------------------------------------------------------------

  defp rebuild(gold) do
    total = length(gold)

    gold
    |> Enum.with_index(1)
    |> Enum.map_reduce(
      %{anchored: 0, corrected: 0, confirmed: 0, reclassified: 0, needs_review: 0},
      fn {example, idx}, stats ->
        if rem(idx, 1000) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

        text = example["text"]
        gold_label = example["speech_act"]
        anchor = structural_anchor(text)
        {entry, stats} = decide(text, gold_label, anchor, stats)

        {{entry, stats}, stats}
      end
    )
    |> then(fn {indexed, final_stats} ->
      if total >= 1000, do: IO.write("\r  Progress: #{total}/#{total}\n")
      {Enum.map(indexed, fn {entry, _} -> entry end), final_stats}
    end)
  end

  defp decide(text, gold_label, {:anchor, anchor_label}, stats) do
    if anchor_label == gold_label do
      entry = %{"text" => text, "speech_act" => gold_label, "_status" => "anchored"}
      {entry, Map.update!(stats, :anchored, &(&1 + 1))}
    else
      entry = %{"text" => text, "speech_act" => anchor_label, "_status" => "corrected"}
      {entry, Map.update!(stats, :corrected, &(&1 + 1))}
    end
  end

  defp decide(text, gold_label, :no_anchor, stats) do
    result = classify(text)
    classifier_label = to_string(result.category)

    if classifier_label == gold_label do
      entry = %{"text" => text, "speech_act" => gold_label, "_status" => "confirmed"}
      {entry, Map.update!(stats, :confirmed, &(&1 + 1))}
    else
      if result.confidence > 0.7 do
        entry = %{"text" => text, "speech_act" => classifier_label, "_status" => "reclassified"}
        {entry, Map.update!(stats, :reclassified, &(&1 + 1))}
      else
        entry = %{"text" => text, "speech_act" => gold_label, "_status" => "needs_review"}
        {entry, Map.update!(stats, :needs_review, &(&1 + 1))}
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Structural anchoring
  # ---------------------------------------------------------------------------

  defp structural_anchor(text) do
    trimmed = String.trim(text)
    downcased = String.downcase(trimmed)
    tokens = Tokenizer.tokenize(downcased)
    first_token = case tokens do
      [t | _] -> t.text
      [] -> ""
    end

    cond do
      commissive_marker?(first_token, tokens) ->
        {:anchor, "commissive"}

      String.ends_with?(trimmed, "?") ->
        {:anchor, "directive"}

      imperative_verb?(first_token) ->
        {:anchor, "directive"}

      String.ends_with?(trimmed, "!") and has_greeting_emotion?(downcased) ->
        {:anchor, "expressive"}

      true ->
        :no_anchor
    end
  end

  defp imperative_verb?(first_token) do
    first_token in @imperative_verbs
  end

  defp has_greeting_emotion?(downcased_text) do
    words = Tokenizer.tokenize(downcased_text) |> Enum.map(& &1.text)
    Enum.any?(words, &(&1 in @greeting_emotion_words))
  end

  defp commissive_marker?(first_token, tokens) do
    first_two = tokens |> Enum.take(2) |> Enum.map(& &1.text) |> Enum.join(" ")
    first_three = tokens |> Enum.take(3) |> Enum.map(& &1.text) |> Enum.join(" ")
    first_four = tokens |> Enum.take(4) |> Enum.map(& &1.text) |> Enum.join(" ")
    first_five = tokens |> Enum.take(5) |> Enum.map(& &1.text) |> Enum.join(" ")

    first_token in ~w(i'll i\u2019ll we'll we\u2019ll) or
      first_two in ["i will", "i can", "i promise", "i shall", "we will", "let me"] or
      first_three in ["i ' ll", "we ' ll", "i'm going to"] or
      first_four == "i ' m going" or
      first_five == "i ' m going to"
  end

  # ---------------------------------------------------------------------------
  # Classification
  # ---------------------------------------------------------------------------

  defp classify(text) do
    SpeechActClassifier.classify(text)
  rescue
    e ->
      Logger.warning(
        "Speech act classification failed for \"#{truncate(text, 40)}\": #{Exception.message(e)}"
      )

      %{category: :unknown, confidence: 0.0}
  catch
    :exit, reason ->
      Logger.warning(
        "Speech act classification exit for \"#{truncate(text, 40)}\": #{inspect(reason)}"
      )

      %{category: :unknown, confidence: 0.0}
  end

  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  defp print_summary(stats, total, before_dist, after_dist) do
    IO.puts("\n--- Rebuild Summary ---\n")
    IO.puts("  Total examples:              #{total}")
    IO.puts("  Anchored (kept):             #{stats.anchored}")
    IO.puts("  Corrected (by anchor):       #{stats.corrected}")
    IO.puts("  Confirmed (by classifier):   #{stats.confirmed}")
    IO.puts("  Reclassified (by classifier): #{stats.reclassified}")
    IO.puts("  Needs review (excluded):     #{stats.needs_review}")

    IO.puts("\n  Label distribution BEFORE:")
    print_distribution(before_dist)

    IO.puts("\n  Label distribution AFTER:")
    print_distribution(after_dist)

    IO.puts("")
  end

  defp print_distribution(dist) do
    dist
    |> Enum.sort_by(fn {_label, count} -> -count end)
    |> Enum.each(fn {label, count} ->
      IO.puts("    #{String.pad_trailing(label, 20)} #{count}")
    end)
  end

  defp label_distribution(entries, key) do
    Enum.frequencies_by(entries, &Map.get(&1, key, "unknown"))
  end

  defp write_output(entries, path) do
    dir = Path.dirname(path)
    File.mkdir_p!(dir)

    if File.exists?(path) do
      backup = path <> ".bak"
      File.cp!(path, backup)
      IO.puts("  Backed up original to: #{backup}")
    end

    json = Jason.encode!(entries, pretty: true) <> "\n"
    File.write!(path, json)
    IO.puts("  Wrote #{length(entries)} entries to: #{path}")
  end

  defp gold_path do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/speech_act/gold_standard.json"
      priv -> Path.join(priv, "evaluation/speech_act/gold_standard.json")
    end
  end

  defp review_path do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/speech_act/needs_review.json"
      priv -> Path.join(priv, "evaluation/speech_act/needs_review.json")
    end
  end

  defp truncate(text, max_len) do
    if String.length(text) > max_len do
      String.slice(text, 0, max_len) <> "..."
    else
      text
    end
  end
end
