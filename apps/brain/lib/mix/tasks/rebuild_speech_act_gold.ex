defmodule Mix.Tasks.Rebuild.SpeechActGold do
  @shortdoc "Rebuild speech act gold standard with structural anchoring and confidence gating"
  @moduledoc """
  Rebuilds the speech act gold standard from structural anchoring only.

      mix rebuild.speech_act_gold           # dry run, prints the summary
      mix rebuild.speech_act_gold --save    # write it

  ## The two layers, and why only one of them writes

  1. **Structural anchoring** — punctuation, imperative verbs and commissive
     markers. Deterministic rules over the text, inspectable in this file. This
     layer may set a label, and every label it sets is stamped
     `labeled_by: "structural_anchor"`.

  2. **The speech act classifier** — consulted for texts no anchor matches. Its
     disagreements are written to `classifier_suggestions.json` as proposals, with
     the label the corpus currently holds and the confidence behind the proposal.
     **It never writes a label into the gold standard.**

  A model prediction written into a gold standard is indistinguishable from ground
  truth the moment it lands, and the model then scores partial credit for agreeing
  with itself. Dropping the rows a model is unsure about does the same damage by
  subtraction: it leaves an evaluation set the model already agrees with.

  So every input row appears in the output, with a label and a `labeled_by`. Rows
  whose origin was never recorded say `"unrecorded"` rather than claiming one, and
  the run aborts if fewer rows come out than went in.
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

    save? = "--save" in args

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

    {rebuilt, suggestions, stats} = rebuild(gold)

    # The corpus may not shrink. Dropping the rows a model is unsure about is how
    # an evaluation set quietly stops covering the cases the model is worst at.
    unless length(rebuilt) == length(gold) do
      Mix.raise(
        "rebuild.speech_act_gold: #{length(gold)} rows in, #{length(rebuilt)} out. " <>
          "Nothing was written."
      )
    end

    after_dist = label_distribution(rebuilt, "speech_act")

    print_summary(stats, length(gold), before_dist, after_dist, suggestions)

    if save? do
      write_output(rebuilt, gold_path())
      write_output(suggestions, suggestions_path())
    else
      IO.puts("  Dry run. Pass --save to write.\n")
    end
  end

  # ---------------------------------------------------------------------------
  # Rebuild logic
  # ---------------------------------------------------------------------------

  @empty_stats %{anchored: 0, corrected: 0, confirmed: 0, suggested: 0}

  defp rebuild(gold) do
    total = length(gold)

    {pairs, stats} =
      gold
      |> Enum.with_index(1)
      |> Enum.map_reduce(@empty_stats, fn {example, idx}, stats ->
        if rem(idx, 1000) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

        {entry, suggestion, stats} =
          decide(
            example["text"],
            example["speech_act"],
            structural_anchor(example["text"]),
            Map.get(example, "labeled_by", "unrecorded"),
            stats
          )

        {{entry, suggestion}, stats}
      end)

    if total >= 1000, do: IO.write("\r  Progress: #{total}/#{total}\n")

    {
      Enum.map(pairs, &elem(&1, 0)),
      pairs |> Enum.map(&elem(&1, 1)) |> Enum.reject(&is_nil/1),
      stats
    }
  end

  # Layer 1. Deterministic rules over the text, so this layer is allowed to set a
  # label -- and says so on every row it sets.
  defp decide(text, gold_label, {:anchor, anchor_label}, provenance, stats) do
    if anchor_label == gold_label do
      {row(text, gold_label, provenance), nil, bump(stats, :anchored)}
    else
      {row(text, anchor_label, "structural_anchor"), nil, bump(stats, :corrected)}
    end
  end

  # Layer 2. The classifier is consulted and its disagreement is recorded, but the
  # corpus keeps the label it already had. A proposal is not evidence.
  defp decide(text, gold_label, :no_anchor, provenance, stats) do
    result = classify(text)
    classifier_label = to_string(result.category)

    if classifier_label == gold_label do
      {row(text, gold_label, provenance), nil, bump(stats, :confirmed)}
    else
      suggestion = %{
        "text" => text,
        "current" => gold_label,
        "proposed" => classifier_label,
        "confidence" => result.confidence
      }

      {row(text, gold_label, provenance), suggestion, bump(stats, :suggested)}
    end
  end

  defp row(text, label, provenance) do
    %{"text" => text, "speech_act" => label, "labeled_by" => provenance}
  end

  defp bump(stats, key), do: Map.update!(stats, key, &(&1 + 1))

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

  # Splitting the suggestions by confidence tells a reviewer which end to start
  # from: a confident disagreement is a likely mislabel, an unconfident one is more
  # often the classifier being weak on that text.
  @review_threshold 0.7

  defp print_summary(stats, total, before_dist, after_dist, suggestions) do
    {confident, unsure} =
      Enum.split_with(suggestions, &(&1["confidence"] > @review_threshold))

    IO.puts("\n--- Rebuild Summary ---\n")
    IO.puts("  Total examples:                #{total}")
    IO.puts("  Anchored (label unchanged):    #{stats.anchored}")
    IO.puts("  Corrected by anchor (written): #{stats.corrected}")
    IO.puts("  Confirmed by classifier:       #{stats.confirmed}")
    IO.puts("  Classifier disagreed:          #{stats.suggested}")
    IO.puts("")
    IO.puts("  Rows written to gold:          #{total}  (never fewer than went in)")
    IO.puts("  Rows the classifier wrote:     0")
    IO.puts("  Suggestions for review:        #{length(suggestions)}")
    IO.puts("")
    IO.puts("  Suggestions by confidence:")
    IO.puts("    above #{@review_threshold}:  #{length(confident)}")
    IO.puts("    at or below:  #{length(unsure)}")

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

  defp suggestions_path do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/speech_act/classifier_suggestions.json"
      priv -> Path.join(priv, "evaluation/speech_act/classifier_suggestions.json")
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
