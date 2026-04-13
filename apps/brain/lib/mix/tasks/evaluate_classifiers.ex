defmodule Mix.Tasks.Evaluate.Classifiers do
  @shortdoc "Compare LSTM vs TF-IDF intent classification against gold standard"
  @moduledoc """
  Runs both LSTM (UnifiedModel) and TF-IDF (IntentClassifierSimple) classifiers
  against the intent gold standard and reports where they agree, where they
  disagree, and which confusion pairs each classifier struggles with.

  This directly feeds into understanding when the IntentArbitrator should
  trust one classifier over the other.

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
    IO.puts("CLASSIFIER COMPARISON: LSTM vs TF-IDF (#{length(gold)} examples)")
    IO.puts(String.duplicate("=", 70) <> "\n")

    lstm_ready = check_lstm_ready()
    tfidf_ready = check_tfidf_ready()

    unless lstm_ready and tfidf_ready do
      IO.puts("WARNING: Not all classifiers are ready!")
      unless lstm_ready, do: IO.puts("  - LSTM (UnifiedModel): NOT READY")
      unless tfidf_ready, do: IO.puts("  - TF-IDF (IntentClassifierSimple): NOT READY")
      IO.puts("")
    end

    results = evaluate_all(gold, lstm_ready, tfidf_ready)

    print_summary(results, lstm_ready, tfidf_ready)
    print_agreement_analysis(results)
    print_confusion_pairs(results, :lstm, "LSTM")
    print_confusion_pairs(results, :tfidf, "TF-IDF")

    if verbose? do
      print_disagreements(results)
    end

    if save? do
      save_results(results)
    end

    IO.puts("")
  end

  defp check_lstm_ready do
    try do
      Code.ensure_loaded?(Brain.ML.LSTM.UnifiedModel) and
        Brain.ML.LSTM.UnifiedModel.ready?()
    rescue
      _ -> false
    catch
      _, _ -> false
    end
  end

  defp check_tfidf_ready do
    try do
      Code.ensure_loaded?(Brain.ML.IntentClassifierSimple) and
        Brain.ML.IntentClassifierSimple.ready?()
    rescue
      _ -> false
    catch
      _, _ -> false
    end
  end

  defp evaluate_all(gold, lstm_ready, tfidf_ready) do
    examples =
      Enum.map(gold, fn example ->
        text = example["text"]
        expected = example["intent"]

        lstm_result = if lstm_ready, do: classify_lstm(text), else: %{intent: nil, confidence: 0.0}
        tfidf_result = if tfidf_ready, do: classify_tfidf(text), else: %{intent: nil, confidence: 0.0}

        lstm_correct = lstm_result.intent == expected
        tfidf_correct = tfidf_result.intent == expected
        agree = lstm_result.intent == tfidf_result.intent

        %{
          text: text,
          expected: expected,
          lstm: lstm_result,
          tfidf: tfidf_result,
          lstm_correct: lstm_correct,
          tfidf_correct: tfidf_correct,
          agree: agree,
          category: categorize(lstm_correct, tfidf_correct, agree)
        }
      end)

    %{
      examples: examples,
      total: length(examples),
      lstm_correct: Enum.count(examples, & &1.lstm_correct),
      tfidf_correct: Enum.count(examples, & &1.tfidf_correct),
      both_correct: Enum.count(examples, &(&1.lstm_correct and &1.tfidf_correct)),
      both_wrong: Enum.count(examples, &(not &1.lstm_correct and not &1.tfidf_correct)),
      lstm_only_correct: Enum.count(examples, &(&1.lstm_correct and not &1.tfidf_correct)),
      tfidf_only_correct: Enum.count(examples, &(not &1.lstm_correct and &1.tfidf_correct)),
      agreement_count: Enum.count(examples, & &1.agree)
    }
  end

  defp categorize(lstm_correct, tfidf_correct, _agree) do
    cond do
      lstm_correct and tfidf_correct -> :both_correct
      lstm_correct and not tfidf_correct -> :lstm_only
      not lstm_correct and tfidf_correct -> :tfidf_only
      true -> :both_wrong
    end
  end

  defp classify_lstm(text) do
    case Brain.ML.LSTM.UnifiedModel.classify_intent(text) do
      {:ok, %{label: intent, confidence: conf}} ->
        %{intent: intent, confidence: conf}

      {:ok, {intent, conf}} ->
        %{intent: intent, confidence: conf}

      _ ->
        %{intent: "unknown", confidence: 0.0}
    end
  rescue
    _ -> %{intent: "unknown", confidence: 0.0}
  catch
    _, _ -> %{intent: "unknown", confidence: 0.0}
  end

  defp classify_tfidf(text) do
    case Brain.ML.IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: conf}} ->
        %{intent: intent, confidence: conf}

      {:ok, {intent, conf}} ->
        %{intent: intent, confidence: conf}

      _ ->
        %{intent: "unknown", confidence: 0.0}
    end
  rescue
    _ -> %{intent: "unknown", confidence: 0.0}
  catch
    _, _ -> %{intent: "unknown", confidence: 0.0}
  end

  defp print_summary(results, lstm_ready, tfidf_ready) do
    IO.puts("--- Overall Accuracy ---\n")

    if lstm_ready do
      lstm_acc = results.lstm_correct / max(results.total, 1) * 100
      IO.puts("  LSTM:    #{Float.round(lstm_acc, 1)}% (#{results.lstm_correct}/#{results.total})")
    end

    if tfidf_ready do
      tfidf_acc = results.tfidf_correct / max(results.total, 1) * 100
      IO.puts("  TF-IDF:  #{Float.round(tfidf_acc, 1)}% (#{results.tfidf_correct}/#{results.total})")
    end

    IO.puts("")
  end

  defp print_agreement_analysis(results) do
    IO.puts("--- Agreement Analysis ---\n")

    agreement_rate = results.agreement_count / max(results.total, 1) * 100
    IO.puts("  Agree:              #{results.agreement_count}/#{results.total} (#{Float.round(agreement_rate, 1)}%)")
    IO.puts("  Both correct:       #{results.both_correct}")
    IO.puts("  Both wrong:         #{results.both_wrong}")
    IO.puts("  LSTM only correct:  #{results.lstm_only_correct}")
    IO.puts("  TF-IDF only correct: #{results.tfidf_only_correct}")
    IO.puts("")

    if results.tfidf_only_correct > 0 do
      IO.puts("  => TF-IDF beats LSTM on #{results.tfidf_only_correct} examples")
      IO.puts("     (these are cases the arbitrator should learn to prefer TF-IDF)")
    end

    if results.lstm_only_correct > 0 do
      IO.puts("  => LSTM beats TF-IDF on #{results.lstm_only_correct} examples")
    end

    IO.puts("")
  end

  defp print_confusion_pairs(results, classifier, label) do
    wrong_examples =
      case classifier do
        :lstm -> Enum.filter(results.examples, &(not &1.lstm_correct))
        :tfidf -> Enum.filter(results.examples, &(not &1.tfidf_correct))
      end

    if wrong_examples == [] do
      IO.puts("--- #{label} Confusion Pairs: None (perfect!) ---\n")
    else
      confusion_pairs =
        wrong_examples
        |> Enum.map(fn ex ->
          predicted = if classifier == :lstm, do: ex.lstm.intent, else: ex.tfidf.intent
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

  defp print_disagreements(results) do
    disagreements =
      results.examples
      |> Enum.reject(& &1.agree)
      |> Enum.sort_by(fn ex ->
        cond do
          ex.tfidf_correct and not ex.lstm_correct -> 0
          ex.lstm_correct and not ex.tfidf_correct -> 1
          true -> 2
        end
      end)

    if disagreements == [] do
      IO.puts("--- No disagreements! ---\n")
    else
      IO.puts("--- Disagreements (#{length(disagreements)} examples) ---\n")

      Enum.each(disagreements, fn ex ->
        winner =
          cond do
            ex.tfidf_correct and not ex.lstm_correct -> "TF-IDF wins"
            ex.lstm_correct and not ex.tfidf_correct -> "LSTM wins"
            true -> "both wrong"
          end

        IO.puts("  \"#{truncate(ex.text, 60)}\"")
        IO.puts("    Expected: #{ex.expected}")
        IO.puts("    LSTM:     #{ex.lstm.intent} (#{Float.round(ex.lstm.confidence * 100, 1)}%) #{if ex.lstm_correct, do: "✓", else: "✗"}")
        IO.puts("    TF-IDF:   #{ex.tfidf.intent} (#{Float.round(ex.tfidf.confidence * 100, 1)}%) #{if ex.tfidf_correct, do: "✓", else: "✗"}")
        IO.puts("    => #{winner}")
        IO.puts("")
      end)
    end
  end

  defp save_results(results) do
    data = %{
      task: "classifier_comparison",
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      total: results.total,
      lstm_accuracy: results.lstm_correct / max(results.total, 1),
      tfidf_accuracy: results.tfidf_correct / max(results.total, 1),
      agreement_rate: results.agreement_count / max(results.total, 1),
      both_correct: results.both_correct,
      both_wrong: results.both_wrong,
      lstm_only_correct: results.lstm_only_correct,
      tfidf_only_correct: results.tfidf_only_correct,
      disagreements:
        results.examples
        |> Enum.reject(& &1.agree)
        |> Enum.map(fn ex ->
          %{
            text: ex.text,
            expected: ex.expected,
            lstm_intent: ex.lstm.intent,
            lstm_confidence: ex.lstm.confidence,
            tfidf_intent: ex.tfidf.intent,
            tfidf_confidence: ex.tfidf.confidence,
            lstm_correct: ex.lstm_correct,
            tfidf_correct: ex.tfidf_correct
          }
        end)
    }

    path = Path.join([
      Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models"),
      "evaluation",
      "classifier_comparison_#{Date.utc_today()}.json"
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
