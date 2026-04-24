defmodule Mix.Tasks.Evaluate.Classifiers do
  @shortdoc "Evaluate intent classification against gold standard"
  @moduledoc """
  Runs TF-IDF and MicroClassifier against the intent gold standard
  and reports accuracy and confusion pairs.

  ## Usage

      mix evaluate.classifiers              # Full comparison
      mix evaluate.classifiers --verbose    # Show per-example disagreements
      mix evaluate.classifiers --save       # Save results to evaluation store
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    verbose? = "--verbose" in args
    save? = "--save" in args

    gold = EvaluationStore.load_gold_standard("intent")

    if gold == [] do
      IO.puts("\nNo gold standard data found.")
      IO.puts("Add examples to: priv/evaluation/intent/gold_standard.json\n")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("CLASSIFIER EVALUATION: TF-IDF (#{length(gold)} examples)")
    IO.puts(String.duplicate("=", 70) <> "\n")

    tfidf_ready = false

    results = evaluate_all(gold, tfidf_ready)

    print_summary(results, tfidf_ready)
    print_confusion_pairs(results, :tfidf, "TF-IDF")

    if verbose? do
      print_details(results)
    end

    if save? do
      save_results(results)
    end

    IO.puts("")
  end

  defp evaluate_all(gold, tfidf_ready) do
    examples =
      Enum.map(gold, fn example ->
        text = example["text"]
        expected = example["intent"]

        tfidf_result = if tfidf_ready, do: classify_tfidf(text), else: %{intent: nil, confidence: 0.0}

        tfidf_correct = tfidf_result.intent == expected

        %{
          text: text,
          expected: expected,
          tfidf: tfidf_result,
          tfidf_correct: tfidf_correct
        }
      end)

    %{
      examples: examples,
      total: length(examples),
      tfidf_correct: Enum.count(examples, & &1.tfidf_correct)
    }
  end

  defp classify_tfidf(_text) do
    %{intent: "unknown", confidence: 0.0}
  end

  defp print_summary(results, tfidf_ready) do
    IO.puts("--- Overall Accuracy ---\n")

    if tfidf_ready do
      tfidf_acc = results.tfidf_correct / max(results.total, 1) * 100
      IO.puts("  TF-IDF:  #{Float.round(tfidf_acc, 1)}% (#{results.tfidf_correct}/#{results.total})")
    end

    IO.puts("")
  end

  defp print_confusion_pairs(results, classifier, label) do
    wrong_examples =
      case classifier do
        :tfidf -> Enum.filter(results.examples, &(not &1.tfidf_correct))
      end

    if wrong_examples == [] do
      IO.puts("--- #{label} Confusion Pairs: None (perfect!) ---\n")
    else
      confusion_pairs =
        wrong_examples
        |> Enum.map(fn ex ->
          predicted = ex.tfidf.intent
          {ex.expected, predicted}
        end)
        |> Enum.frequencies()
        |> Enum.sort_by(fn {_pair, count} -> -count end)
        |> Enum.take(15)

      IO.puts("--- #{label} Top Confusion Pairs (expected -> predicted) ---\n")

      Enum.each(confusion_pairs, fn {{expected, predicted}, count} ->
        IO.puts("  #{String.pad_trailing(expected, 35)} -> #{String.pad_trailing(predicted, 35)} (#{count}x)")
      end)

      IO.puts("")
    end
  end

  defp print_details(results) do
    wrong =
      results.examples
      |> Enum.reject(& &1.tfidf_correct)

    if wrong == [] do
      IO.puts("--- No errors! ---\n")
    else
      IO.puts("--- Errors (#{length(wrong)} examples) ---\n")

      Enum.each(wrong, fn ex ->
        IO.puts("  \"#{truncate(ex.text, 60)}\"")
        IO.puts("    Expected: #{ex.expected}")
        IO.puts("    TF-IDF:   #{ex.tfidf.intent} (#{Float.round(ex.tfidf.confidence * 100, 1)}%) ✗")
        IO.puts("")
      end)
    end
  end

  defp save_results(results) do
    data = %{
      task: "classifier_evaluation",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      total: results.total,
      tfidf_accuracy: results.tfidf_correct / max(results.total, 1)
    }

    path = Path.join([
      Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models"),
      "evaluation",
      "classifier_evaluation_#{Date.utc_today()}.json"
    ])

    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(data, pretty: true))
    IO.puts("Results saved to: #{path}")
  end

  defp truncate(text, max_len) do
    if String.length(text) > max_len do
      String.slice(text, 0, max_len) <> "..."
    else
      text
    end
  end
end
