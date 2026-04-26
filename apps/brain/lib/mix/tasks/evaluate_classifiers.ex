defmodule Mix.Tasks.Evaluate.Classifiers do
  @shortdoc "Evaluate intent classification: feature-vector vs full pipeline"
  @moduledoc """
  Runs both the isolated feature-vector classifier and the full analysis
  pipeline against the intent gold standard, reporting side-by-side
  accuracy and confusion pairs.

  ## Usage

      mix evaluate.classifiers              # Full comparison
      mix evaluate.classifiers --verbose    # Show per-example disagreements
      mix evaluate.classifiers --save       # Save results to evaluation store
  """

  use Mix.Task

  alias Brain.Analysis.{FeatureExtractor, Pipeline}
  alias Brain.ML.{Evaluation, EvaluationStore, MicroClassifiers}

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

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.")

    IO.puts("\n" <> String.duplicate("=", 70))
    IO.puts("CLASSIFIER EVALUATION: Feature-Vector vs Full Pipeline (#{length(gold)} examples)")
    IO.puts(String.duplicate("=", 70) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    results = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    print_summary(results, duration_ms)
    print_confusion_pairs(results, :feature_vector, "Feature-Vector (isolated)")
    print_confusion_pairs(results, :pipeline, "Full Pipeline")

    if verbose? do
      print_details(results)
    end

    if save? do
      save_results(results, duration_ms)
    end

    IO.puts("")
  end

  defp evaluate_all(gold) do
    total = length(gold)

    examples =
      gold
      |> Enum.with_index(1)
      |> Enum.map(fn {example, idx} ->
        if rem(idx, 500) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

        text = example["text"]
        expected = example["intent"]

        {fv_intent, fv_confidence, pipeline_intent} = classify_both(text)

        %{
          text: text,
          expected: expected,
          feature_vector: %{intent: fv_intent, confidence: fv_confidence},
          pipeline: %{intent: pipeline_intent},
          fv_correct: fv_intent == expected,
          pipeline_correct: pipeline_intent == expected
        }
      end)

    if total >= 500, do: IO.write("\r  Progress: #{total}/#{total}\n")

    %{
      examples: examples,
      total: length(examples),
      fv_correct: Enum.count(examples, & &1.fv_correct),
      pipeline_correct: Enum.count(examples, & &1.pipeline_correct)
    }
  end

  defp classify_both(text) do
    try do
      analysis = Pipeline.analyze_chunk(text, side_effects: false)
      pipeline_intent = to_string(analysis.intent || "unknown")

      {fv_intent, fv_confidence} =
        try do
          {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)

          case MicroClassifiers.classify_vector(:intent_full, feature_vector) do
            {:ok, label, confidence} -> {to_string(label), confidence}
            _ -> {"unknown", 0.0}
          end
        rescue
          _ -> {"unknown", 0.0}
        end

      {fv_intent, fv_confidence, pipeline_intent}
    rescue
      _ -> {"unknown", 0.0, "unknown"}
    catch
      :exit, _ -> {"unknown", 0.0, "unknown"}
    end
  end

  defp print_summary(results, duration_ms) do
    IO.puts("--- Overall Accuracy ---\n")

    fv_acc = results.fv_correct / max(results.total, 1) * 100
    pipe_acc = results.pipeline_correct / max(results.total, 1) * 100

    IO.puts("  Feature-Vector:  #{Float.round(fv_acc, 1)}% (#{results.fv_correct}/#{results.total})")
    IO.puts("  Full Pipeline:   #{Float.round(pipe_acc, 1)}% (#{results.pipeline_correct}/#{results.total})")
    IO.puts("  Duration:        #{duration_ms}ms")

    delta = pipe_acc - fv_acc
    sign = if delta >= 0, do: "+", else: ""

    IO.puts(
      "\n  Pipeline delta:  #{sign}#{Float.round(delta, 1)}% " <>
        "(refinement #{if delta >= 0, do: "helps", else: "hurts"})"
    )

    both_correct = Enum.count(results.examples, &(&1.fv_correct and &1.pipeline_correct))
    fv_only = Enum.count(results.examples, &(&1.fv_correct and not &1.pipeline_correct))
    pipe_only = Enum.count(results.examples, &(not &1.fv_correct and &1.pipeline_correct))
    both_wrong = Enum.count(results.examples, &(not &1.fv_correct and not &1.pipeline_correct))

    IO.puts("\n  Agreement matrix:")
    IO.puts("    Both correct:      #{both_correct}")
    IO.puts("    FV only correct:   #{fv_only}")
    IO.puts("    Pipeline only:     #{pipe_only}")
    IO.puts("    Both wrong:        #{both_wrong}")

    confidences = Enum.map(results.examples, & &1.feature_vector.confidence)
    correct_confs = results.examples |> Enum.filter(& &1.fv_correct) |> Enum.map(& &1.feature_vector.confidence)
    wrong_confs = results.examples |> Enum.reject(& &1.fv_correct) |> Enum.map(& &1.feature_vector.confidence)

    IO.puts("\n  Feature-vector confidence:")
    IO.puts("    Mean (all):        #{format_conf(mean(confidences))}")
    IO.puts("    Mean (correct):    #{format_conf(mean(correct_confs))}")
    IO.puts("    Mean (incorrect):  #{format_conf(mean(wrong_confs))}")
    IO.puts("")
  end

  defp mean([]), do: 0.0
  defp mean(vals), do: Enum.sum(vals) / length(vals)

  defp format_conf(val), do: "#{Float.round(val * 100, 1)}%"

  defp print_confusion_pairs(results, classifier, label) do
    wrong_examples =
      case classifier do
        :feature_vector -> Enum.filter(results.examples, &(not &1.fv_correct))
        :pipeline -> Enum.filter(results.examples, &(not &1.pipeline_correct))
      end

    if wrong_examples == [] do
      IO.puts("--- #{label} Confusion Pairs: None (perfect!) ---\n")
    else
      confusion_pairs =
        wrong_examples
        |> Enum.map(fn ex ->
          predicted =
            case classifier do
              :feature_vector -> ex.feature_vector.intent
              :pipeline -> ex.pipeline.intent
            end

          {ex.expected, predicted}
        end)
        |> Enum.frequencies()
        |> Enum.sort_by(fn {_pair, count} -> -count end)
        |> Enum.take(15)

      IO.puts("--- #{label} Top Confusion Pairs (expected -> predicted) ---\n")

      Enum.each(confusion_pairs, fn {{expected, predicted}, count} ->
        IO.puts(
          "  #{String.pad_trailing(to_string(expected), 35)} -> " <>
            "#{String.pad_trailing(to_string(predicted), 35)} (#{count}x)"
        )
      end)

      IO.puts("")
    end
  end

  defp print_details(results) do
    disagreements =
      results.examples
      |> Enum.filter(&(&1.fv_correct != &1.pipeline_correct))

    if disagreements == [] do
      IO.puts("--- No disagreements between classifiers ---\n")
    else
      IO.puts("--- Disagreements (#{length(disagreements)} examples) ---\n")

      Enum.each(disagreements, fn ex ->
        fv_marker = if ex.fv_correct, do: "ok", else: "X"
        pipe_marker = if ex.pipeline_correct, do: "ok", else: "X"
        conf = format_conf(ex.feature_vector.confidence)

        IO.puts("  \"#{truncate(ex.text, 60)}\"")
        IO.puts("    Expected:       #{ex.expected}")
        IO.puts("    Feature-Vector: #{ex.feature_vector.intent} (#{conf}) [#{fv_marker}]")
        IO.puts("    Pipeline:       #{ex.pipeline.intent} [#{pipe_marker}]")
        IO.puts("")
      end)
    end
  end

  defp save_results(results, duration_ms) do
    fv_predictions = Enum.map(results.examples, & &1.feature_vector.intent)
    pipe_predictions = Enum.map(results.examples, & &1.pipeline.intent)
    actuals = Enum.map(results.examples, & &1.expected)

    fv_result =
      Evaluation.build_result("intent_feature_vector", fv_predictions, actuals,
        notes: "Isolated feature-vector classifier (:intent_full)"
      )

    pipe_result =
      Evaluation.build_result("intent_pipeline", pipe_predictions, actuals,
        notes: "Full analysis pipeline (feature-vector + refinement)"
      )

    {:ok, fv_path} = EvaluationStore.save(Map.put(fv_result, :duration_ms, duration_ms))
    {:ok, pipe_path} = EvaluationStore.save(Map.put(pipe_result, :duration_ms, duration_ms))
    IO.puts("Feature-vector results saved to: #{fv_path}")
    IO.puts("Pipeline results saved to: #{pipe_path}")
  end

  defp truncate(text, max_len) do
    if String.length(text) > max_len do
      String.slice(text, 0, max_len) <> "..."
    else
      text
    end
  end
end
