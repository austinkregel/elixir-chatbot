defmodule Mix.Tasks.Evaluate do
  alias Brain.Analysis.SpeechActClassifier
  alias Brain.ML
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

    IO.puts("\n" <> String.duplicate("=", 60))
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
          {predictions, actuals} = run_task_evaluation(task, gold)
          result = Evaluation.build_result(task, predictions, actuals)

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

    if compare? do
      IO.puts("\n--- Comparison with previous runs ---\n")

      Enum.each(results, fn
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

  defp run_task_evaluation("intent", gold) do
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["intent"]
      predicted = classify_intent(text)
      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end

  defp run_task_evaluation("sentiment", gold) do
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["sentiment"]
      predicted = classify_sentiment(text)
      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end

  defp run_task_evaluation("speech_act", gold) do
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["speech_act"]
      predicted = classify_speech_act(text)
      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end

  defp run_task_evaluation(_task, _gold) do
    {[], []}
  end

  defp classify_intent(text) do
    case SpeechActClassifier.classify(text) do
      %{indicators: indicators} ->
        indicators
        |> Enum.find_value("unknown", fn indicator ->
          case String.split(indicator, ":", parts: 2) do
            ["intent", intent] -> intent
            _ -> nil
          end
        end)

      _ ->
        "unknown"
    end
  end

  defp classify_sentiment(text) do
    case ML.SentimentClassifierSimple.classify(text) do
      {:ok, %{label: label}} -> to_string(label)
      {:ok, result} when is_map(result) -> to_string(Map.get(result, :label, "neutral"))
      _ -> "neutral"
    end
  end

  defp classify_speech_act(text) do
    case SpeechActClassifier.classify(text) do
      %{category: category} -> to_string(category)
      _ -> "unknown"
    end
  end
end