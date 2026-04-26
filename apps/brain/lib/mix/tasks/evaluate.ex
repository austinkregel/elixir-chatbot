defmodule Mix.Tasks.Evaluate do
  alias Brain.Analysis.{Pipeline, SpeechActClassifier}
  alias Brain.ML
  require Logger
  @shortdoc "Run all ML model evaluations"
  @moduledoc "Run evaluation against gold standard data for all ML tasks.\n\n## Usage\n\n    mix evaluate              # Run all evaluations\n    mix evaluate --save       # Save results for historical tracking\n    mix evaluate --compare    # Compare with previous run\n    mix evaluate --verbose    # Show per-class details\n"

  use Mix.Task

  alias ML.{Evaluation, EvaluationStore}

  @tasks ["intent", "sentiment", "speech_act"]

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    compare? = "--compare" in args
    verbose? = "--verbose" in args

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    ML.MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.\n")

    IO.puts(String.duplicate("=", 60))
    IO.puts("ML MODEL EVALUATION")
    IO.puts(String.duplicate("=", 60) <> "\n")

    results =
      Enum.map(@tasks, fn task ->
        gold = EvaluationStore.load_gold_standard(task)

        if gold == [] do
          IO.puts(
            "  #{task}: No gold standard data (add examples to priv/evaluation/#{task}/gold_standard.json)"
          )

          {task, nil}
        else
          IO.puts("  Evaluating #{task} (#{length(gold)} examples)...")
          {predictions, actuals, counts} = run_task_evaluation(task, gold)
          result = Evaluation.build_result(task, predictions, actuals)
          result = Map.put(result, :diagnostics, counts)

          acc = Float.round(result.accuracy * 100, 1)
          f1 = Float.round(result.macro_f1 * 100, 1)
          IO.puts("    Accuracy: #{acc}%  Macro-F1: #{f1}%")

          if verbose? do
            report =
              Evaluation.classification_report(Evaluation.confusion_matrix(predictions, actuals))

            IO.puts("")
            IO.puts(Evaluation.format_report(report))
            IO.puts("")
          end

          if save? do
            EvaluationStore.save(result)
          end

          {task, result}
        end
      end)

    ner_result = run_ner_evaluation(save?, verbose?)

    if compare? do
      IO.puts("\n--- Comparison with previous runs ---\n")

      all_results = if ner_result, do: results ++ [{"ner", ner_result}], else: results

      Enum.each(all_results, fn
        {task, nil} ->
          IO.puts("  #{task}: no data")

        {task, current} ->
          case EvaluationStore.list_runs(task) do
            [_current | [previous | _]] ->
              delta = EvaluationStore.compare(previous, current)

              sign =
                if delta.accuracy_delta >= 0 do
                  "+"
                else
                  ""
                end

              IO.puts(
                "  #{task}: #{sign}#{Float.round(delta.accuracy_delta * 100, 1)}% accuracy change"
              )

            _ ->
              IO.puts("  #{task}: no previous run to compare")
          end
      end)
    end

    IO.puts("")
  end

  defp run_ner_evaluation(save?, verbose?) do
    gold = EvaluationStore.load_gold_standard("ner")

    if gold == [] do
      IO.puts("  ner: No gold standard data (add examples to priv/evaluation/ner/gold_standard.json)")
      nil
    else
      IO.puts("  Evaluating ner (#{length(gold)} examples)...")
      {predictions, actuals, counts} = evaluate_ner(gold)
      result = Evaluation.build_result("ner", predictions, actuals)
      result = Map.put(result, :diagnostics, counts)

      acc = Float.round(result.accuracy * 100, 1)
      f1 = Float.round(result.macro_f1 * 100, 1)
      IO.puts("    Accuracy: #{acc}%  Macro-F1: #{f1}%")

      if counts.errored > 0 do
        IO.puts("    Diagnostics: ok=#{counts.ok} errored=#{counts.errored}")
      end

      if verbose? do
        report = Evaluation.classification_report(Evaluation.confusion_matrix(predictions, actuals))
        IO.puts("")
        IO.puts(Evaluation.format_report(report))
        IO.puts("")
      end

      if save?, do: EvaluationStore.save(result)

      result
    end
  end

  defp evaluate_ner(gold) do
    {results, counts} =
      Enum.reduce(gold, {[], %{ok: 0, errored: 0}}, fn example, {acc, counts} ->
        text = example["text"]
        expected_entities = example["entities"] || example["expected"] || []

        case extract_entities_safely(text) do
          {:ok, extracted} ->
            pairs = match_entities(expected_entities, extracted)
            {pairs ++ acc, Map.update!(counts, :ok, &(&1 + 1))}

          {:error, _} ->
            pairs = Enum.map(expected_entities, fn e -> {"none", e["type"]} end)
            {pairs ++ acc, Map.update!(counts, :errored, &(&1 + 1))}
        end
      end)

    error_rate = counts.errored / max(length(gold), 1)

    if error_rate > 0.10 do
      Mix.raise("""
      Evaluation aborted: NER error rate #{Float.round(error_rate * 100, 1)}% exceeds 10% threshold.
        ok: #{counts.ok}, errored: #{counts.errored}
      """)
    end

    results = Enum.reverse(results)
    predictions = Enum.map(results, &elem(&1, 0))
    actuals = Enum.map(results, &elem(&1, 1))

    {predictions, actuals, counts}
  end

  defp extract_entities_safely(text) do
    analysis = Pipeline.analyze_chunk(text, side_effects: false)
    {:ok, analysis.entities || []}
  rescue
    e ->
      Logger.warning("NER extraction crashed: #{Exception.message(e)}")
      {:error, e}
  catch
    :exit, reason ->
      Logger.warning("NER extraction exit: #{inspect(reason)}")
      {:error, reason}
  end

  defp match_entities(expected_entities, extracted) do
    Enum.map(expected_entities, fn expected ->
      expected_value = expected["value"]
      expected_type = expected["type"]

      match =
        Enum.find(extracted, fn ext ->
          ext_value = Map.get(ext, :value) || Map.get(ext, "value")
          normalize_value(ext_value) == normalize_value(expected_value)
        end)

      predicted_type =
        if match do
          to_string(Map.get(match, :entity_type) || Map.get(match, "entity_type") || "none")
        else
          "none"
        end

      {predicted_type, to_string(expected_type)}
    end)
  end

  defp normalize_value(nil), do: ""
  defp normalize_value(v) when is_binary(v), do: v |> String.downcase() |> String.trim()
  defp normalize_value(v), do: to_string(v)

  defp run_task_evaluation(task, gold) do
    classify_fn = classifier_for(task)

    {results, counts} =
      Enum.reduce(gold, {[], %{ok: 0, unknown: 0, errored: 0}}, fn example, {acc, counts} ->
        text = example["text"]
        expected = expected_label(task, example)

        case classify_fn.(text) do
          {:ok, predicted} ->
            {[{predicted, expected} | acc], Map.update!(counts, :ok, &(&1 + 1))}

          {:unknown, predicted} ->
            {[{predicted, expected} | acc], Map.update!(counts, :unknown, &(&1 + 1))}

          {:error, predicted} ->
            {[{predicted, expected} | acc], Map.update!(counts, :errored, &(&1 + 1))}
        end
      end)

    error_rate = (counts.errored + counts.unknown) / max(length(gold), 1)

    if error_rate > 0.10 do
      Mix.raise("""
      Evaluation aborted: #{task} error rate #{Float.round(error_rate * 100, 1)}% exceeds 10% threshold.
        ok: #{counts.ok}, unknown: #{counts.unknown}, errored: #{counts.errored}
      This likely indicates a systemic failure (models not loaded, pipeline crash).
      """)
    end

    if counts.unknown > 0 or counts.errored > 0 do
      IO.puts("    Diagnostics: ok=#{counts.ok} unknown=#{counts.unknown} errored=#{counts.errored}")
    end

    results = Enum.reverse(results)
    predictions = Enum.map(results, &elem(&1, 0))
    actuals = Enum.map(results, &elem(&1, 1))

    {predictions, actuals, counts}
  end

  defp expected_label("intent", example), do: example["intent"]
  defp expected_label("sentiment", example), do: example["sentiment"]
  defp expected_label("speech_act", example), do: example["speech_act"]

  defp classifier_for("intent"), do: &classify_intent/1
  defp classifier_for("sentiment"), do: &classify_sentiment/1
  defp classifier_for("speech_act"), do: &classify_speech_act/1

  defp classify_intent(text) do
    analysis = Pipeline.analyze_chunk(text, side_effects: false)
    predicted = to_string(analysis.intent || "unknown")

    if predicted == "unknown", do: {:unknown, predicted}, else: {:ok, predicted}
  rescue
    e ->
      Logger.warning("Intent classification crashed: #{Exception.message(e)}")
      {:error, "unknown"}
  catch
    :exit, reason ->
      Logger.warning("Intent classification exit: #{inspect(reason)}")
      {:error, "unknown"}
  end

  defp classify_sentiment(text) do
    case ML.SentimentClassifierSimple.classify(text) do
      {:ok, %{label: label}} -> {:ok, to_string(label)}
      {:ok, result} when is_map(result) -> {:ok, to_string(Map.get(result, :label, "neutral"))}
      _ -> {:unknown, "neutral"}
    end
  rescue
    _ -> {:error, "neutral"}
  end

  defp classify_speech_act(text) do
    case SpeechActClassifier.classify(text) do
      %{category: category} -> {:ok, to_string(category)}
      _ -> {:unknown, "unknown"}
    end
  rescue
    _ -> {:error, "unknown"}
  end
end