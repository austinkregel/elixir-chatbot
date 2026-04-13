defmodule Mix.Tasks.Evaluate.Sentiment do
  alias Brain.ML.LSTM.Integration
  alias Brain.ML
  @shortdoc "Evaluate sentiment analysis accuracy"
  @moduledoc "Evaluate sentiment classification against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy.\n\n## Usage\n\n    mix evaluate.sentiment              # Run evaluation\n    mix evaluate.sentiment --save       # Save results\n    mix evaluate.sentiment --verbose    # Show per-class details\n\n## Gold Standard Format\n\n    [{\"text\": \"I love this!\", \"sentiment\": \"positive\"}, ...]\n"

  use Mix.Task

  alias ML.{Evaluation, EvaluationStore}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args

    gold = EvaluationStore.load_gold_standard("sentiment")

    if gold == [] do
      IO.puts("\nNo gold standard data for sentiment analysis.")
      IO.puts("Add examples to: priv/evaluation/sentiment/gold_standard.json")
      IO.puts("")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("SENTIMENT ANALYSIS EVALUATION (#{length(gold)} examples)")
    IO.puts("(Using production pipeline)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    status = Integration.model_status()
    IO.puts("Model status: LSTM=#{status.lstm_unified}, TF-IDF=#{status.tfidf}\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals} = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    result = Evaluation.build_result("sentiment", predictions, actuals)

    IO.puts("Overall Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:         #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("Duration:         #{duration_ms}ms")
    IO.puts("")

    broadcast_result(result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("sentiment", %{
      accuracy: result.accuracy,
      macro_f1: result.macro_f1,
      weighted_f1: result.weighted_f1,
      total_examples: result.total_examples,
      duration_ms: duration_ms
    })

    if verbose? do
      cm = Evaluation.confusion_matrix(predictions, actuals)
      report = Evaluation.classification_report(cm)
      IO.puts(Evaluation.format_report(report))
      IO.puts("")
    end

    if save? do
      {:ok, path} = EvaluationStore.save(result)
      IO.puts("Results saved to: #{path}")
    end

    IO.puts("")
  end

  defp evaluate_all(gold) do
    total = length(gold)

    gold
    |> Enum.with_index(1)
    |> Enum.reduce({[], []}, fn {example, idx}, {preds, acts} ->
      text = example["text"]
      expected = example["sentiment"]

      if rem(idx, 1000) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

      predicted =
        try do
          case Integration.classify_sentiment(text) do
            {:ok, %{label: label}} -> to_string(label)
            _ -> "neutral"
          end
        rescue
          _ -> "neutral"
        catch
          :exit, _ -> "neutral"
        end

      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} ->
      IO.write("\r  Progress: #{total}/#{total}\n")
      {Enum.reverse(p), Enum.reverse(a)}
    end)
  end

  defp broadcast_result(result, duration_ms) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "evaluation:complete",
      {:evaluation_complete, %{
        task: "sentiment",
        accuracy: result.accuracy,
        macro_f1: result.macro_f1,
        weighted_f1: result.weighted_f1,
        total_examples: result.total_examples,
        duration_ms: duration_ms,
        timestamp: DateTime.utc_now()
      }})
  rescue
    _ -> :ok
  end
end
