defmodule Mix.Tasks.Audit.SpeechActGold do
  @shortdoc "Audit speech act gold standard for mislabeled examples"
  @moduledoc """
  Audit the speech act gold standard for structural mismatches and mislabeling patterns.

  Loads the gold standard, runs each example through the production classifier,
  and reports label distribution, structural signal mismatches, classifier
  disagreements, and low-confidence examples.

  ## Usage

      mix audit.speech_act_gold              # Run audit
      mix audit.speech_act_gold --export     # Export flagged examples for review
  """

  use Mix.Task

  alias Brain.ML.{EvaluationStore, MicroClassifiers, Tokenizer}
  alias Brain.Analysis.SpeechActClassifier

  @imperative_stems ~w(tell show give let please help explain describe list find
                       get make set add remove create delete open close start stop
                       run send check look go come try do say ask call read write)

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    export? = "--export" in args

    gold = EvaluationStore.load_gold_standard("speech_act")

    if gold == [] do
      IO.puts("\nNo gold standard data found.")
      IO.puts("Add examples to: priv/evaluation/speech_act/gold_standard.json\n")
      exit(:normal)
    end

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.\n")

    IO.puts(String.duplicate("=", 70))
    IO.puts("SPEECH ACT GOLD STANDARD AUDIT (#{length(gold)} examples)")
    IO.puts(String.duplicate("=", 70))

    print_label_distribution(gold)

    classified = classify_all(gold)
    structural_flags = find_structural_mismatches(gold)

    print_structural_mismatches(structural_flags)
    print_classifier_disagreements(classified)
    print_low_confidence(classified)

    if export? do
      export_flagged(structural_flags)
    end

    IO.puts("")
  end

  defp print_label_distribution(gold) do
    IO.puts("\n--- Label Distribution ---\n")

    gold
    |> Enum.frequencies_by(fn ex -> ex["speech_act"] end)
    |> Enum.sort_by(fn {_label, count} -> -count end)
    |> Enum.each(fn {label, count} ->
      IO.puts("  #{String.pad_trailing(label, 20)} #{count}")
    end)

    IO.puts("")
  end

  defp classify_all(gold) do
    total = length(gold)

    gold
    |> Enum.with_index(1)
    |> Enum.map(fn {example, idx} ->
      if rem(idx, 1000) == 0, do: IO.write("\r  Classifying: #{idx}/#{total}")

      text = example["text"]
      gold_label = example["speech_act"]

      predicted =
        try do
          result = SpeechActClassifier.classify(text)
          to_string(result.category)
        rescue
          _ -> "unknown"
        catch
          :exit, _ -> "unknown"
        end

      %{text: text, gold_label: gold_label, predicted_label: predicted}
    end)
    |> tap(fn _ ->
      if total >= 1000, do: IO.write("\r  Classifying: #{total}/#{total}\n")
    end)
  end

  defp find_structural_mismatches(gold) do
    Enum.flat_map(gold, fn example ->
      text = example["text"]
      gold_label = example["speech_act"]
      trimmed = String.trim(text)

      flags = []

      flags =
        if String.ends_with?(trimmed, "?") and gold_label != "directive" do
          reason = "question ending with ? labeled as #{gold_label}"
          [%{text: text, gold_label: gold_label, reason: reason} | flags]
        else
          flags
        end

      flags =
        if starts_with_imperative?(trimmed) and gold_label != "directive" do
          reason = "imperative-starting text labeled as #{gold_label}"
          [%{text: text, gold_label: gold_label, reason: reason} | flags]
        else
          flags
        end

      flags =
        if gold_label == "expressive" and String.ends_with?(trimmed, "?") do
          reason = "question ending with ? labeled as expressive"
          [%{text: text, gold_label: gold_label, reason: reason} | flags]
        else
          flags
        end

      flags
    end)
    |> Enum.uniq_by(fn flag -> {flag.text, flag.reason} end)
  end

  defp starts_with_imperative?(text) do
    tokens = Tokenizer.tokenize(text)

    case tokens do
      [first | _] ->
        first_word = String.downcase(first.text)
        first_word in @imperative_stems

      [] ->
        false
    end
  end

  defp print_structural_mismatches(flags) do
    IO.puts("--- Structural Signal Mismatches (#{length(flags)} flagged) ---\n")

    flags
    |> Enum.frequencies_by(fn flag -> flag.reason end)
    |> Enum.sort_by(fn {_reason, count} -> -count end)
    |> Enum.each(fn {reason, count} ->
      IO.puts("  #{String.pad_trailing(reason, 55)} #{count}")
    end)

    IO.puts("")
  end

  defp print_classifier_disagreements(classified) do
    disagreements =
      classified
      |> Enum.filter(fn ex -> ex.gold_label != ex.predicted_label end)

    IO.puts("--- Classifier Disagreements (#{length(disagreements)}/#{length(classified)}) ---\n")

    if disagreements == [] do
      IO.puts("  No disagreements.\n")
    else
      disagreements
      |> Enum.frequencies_by(fn ex -> {ex.gold_label, ex.predicted_label} end)
      |> Enum.sort_by(fn {_pair, count} -> -count end)
      |> Enum.each(fn {{gold, predicted}, count} ->
        IO.puts(
          "  #{String.pad_trailing(gold, 20)} -> #{String.pad_trailing(predicted, 20)} (#{count}x)"
        )
      end)

      IO.puts("")
    end
  end

  defp print_low_confidence(classified) do
    low_conf =
      classified
      |> Enum.filter(fn ex ->
        ex.gold_label != ex.predicted_label and has_clear_structural_signal?(ex)
      end)

    IO.puts("--- Low-Confidence Structural Conflicts (#{length(low_conf)}) ---\n")

    low_conf
    |> Enum.frequencies_by(fn ex -> {ex.gold_label, ex.predicted_label} end)
    |> Enum.sort_by(fn {_pair, count} -> -count end)
    |> Enum.each(fn {{gold, predicted}, count} ->
      IO.puts(
        "  #{String.pad_trailing(gold, 20)} -> #{String.pad_trailing(predicted, 20)} (#{count}x)"
      )
    end)

    if low_conf == [] do
      IO.puts("  None.\n")
    else
      IO.puts("")
    end
  end

  defp has_clear_structural_signal?(example) do
    trimmed = String.trim(example.text)

    (String.ends_with?(trimmed, "?") and example.gold_label == "directive") or
      (String.ends_with?(trimmed, "!") and example.gold_label == "expressive")
  end

  defp export_flagged(flags) do
    output =
      Enum.map(flags, fn flag ->
        %{
          text: flag.text,
          gold_label: flag.gold_label,
          predicted_label: classify_single(flag.text),
          reason: flag.reason
        }
      end)

    dir = Path.join(:code.priv_dir(:brain), "evaluation/speech_act")
    File.mkdir_p!(dir)
    path = Path.join(dir, "flagged_for_review.json")

    File.write!(path, Jason.encode!(output, pretty: true))
    IO.puts("\nExported #{length(output)} flagged examples to: #{path}")
  end

  defp classify_single(text) do
    try do
      result = SpeechActClassifier.classify(text)
      to_string(result.category)
    rescue
      _ -> "unknown"
    catch
      :exit, _ -> "unknown"
    end
  end
end
