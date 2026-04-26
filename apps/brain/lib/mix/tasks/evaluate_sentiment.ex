defmodule Mix.Tasks.Evaluate.Sentiment do
  alias Brain.ML
  @shortdoc "Evaluate sentiment analysis accuracy"
  @moduledoc "Evaluate sentiment classification against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy.\n\n## Usage\n\n    mix evaluate.sentiment              # Run evaluation\n    mix evaluate.sentiment --save       # Save results\n    mix evaluate.sentiment --verbose    # Show per-class details\n    mix evaluate.sentiment --audit      # Report high-confidence disagreements\n\n## Gold Standard Format\n\n    [{\"text\": \"I love this!\", \"sentiment\": \"positive\"}, ...]\n"

  use Mix.Task

  require Logger

  alias ML.{Evaluation, EvaluationStore, MicroClassifiers}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args
    audit? = "--audit" in args

    IO.puts("\nAwaiting MicroClassifiers readiness...")
    MicroClassifiers.await_ready(:infinity)
    IO.puts("MicroClassifiers ready.")

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

    IO.puts("Model: TF-IDF SentimentClassifierSimple\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals, counts, disagreements} = evaluate_all(gold, audit?)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    error_rate = (counts.errored + counts.unknown) / max(length(gold), 1)

    if error_rate > 0.10 do
      Mix.raise("""
      Evaluation aborted: sentiment error rate #{Float.round(error_rate * 100, 1)}% exceeds 10% threshold.
        ok: #{counts.ok}, unknown: #{counts.unknown}, errored: #{counts.errored}
      """)
    end

    if counts.unknown > 0 or counts.errored > 0 do
      IO.puts("Diagnostics: ok=#{counts.ok} unknown=#{counts.unknown} errored=#{counts.errored}\n")
    end

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
      duration_ms: duration_ms,
      diagnostics: counts
    })

    if verbose? do
      cm = Evaluation.confusion_matrix(predictions, actuals)
      report = Evaluation.classification_report(cm)
      IO.puts(Evaluation.format_report(report))
      IO.puts("")
    end

    if audit? and disagreements != [] do
      top = Enum.sort_by(disagreements, & &1.confidence, :desc) |> Enum.take(20)

      IO.puts("--- High-Confidence Disagreements (top #{length(top)}) ---\n")

      Enum.each(top, fn d ->
        IO.puts("  text:       #{d.text}")
        IO.puts("  expected:   #{d.expected}")
        IO.puts("  predicted:  #{d.predicted}")
        IO.puts("  confidence: #{Float.round(d.confidence, 4)}")
        IO.puts("")
      end)
    end

    if save? do
      {:ok, path} = EvaluationStore.save(result)
      IO.puts("Results saved to: #{path}")
    end

    IO.puts("")
  end

  defp evaluate_all(gold, audit?) do
    total = length(gold)

    gold
    |> Enum.with_index(1)
    |> Enum.reduce({[], [], %{ok: 0, unknown: 0, errored: 0}, []}, fn {example, idx},
                                                                       {preds, acts, counts, disagrees} ->
      text = example["text"]
      expected = example["sentiment"]

      if rem(idx, 1000) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

      {status, predicted, confidence} =
        try do
          case ML.SentimentClassifierSimple.classify(text) do
            {:ok, %{label: label, confidence: conf}} ->
              {:ok, to_string(label), conf}

            {:ok, %{label: label}} ->
              {:ok, to_string(label), nil}

            {:ok, result} when is_map(result) ->
              {:unknown, to_string(Map.get(result, :label, "neutral")), nil}

            other ->
              Logger.warning("Unexpected classify result: #{inspect(other)}")
              {:unknown, "neutral", nil}
          end
        rescue
          e ->
            Logger.warning("Sentiment classify crashed: #{Exception.message(e)}")
            {:error, "neutral", nil}
        catch
          :exit, reason ->
            Logger.warning("Sentiment classify exited: #{inspect(reason)}")
            {:error, "neutral", nil}
        end

      counts = Map.update!(counts, status_key(status), &(&1 + 1))

      disagrees =
        if audit? and status == :ok and confidence != nil and confidence > 0.9 and
             predicted != expected do
          [%{text: text, expected: expected, predicted: predicted, confidence: confidence} | disagrees]
        else
          disagrees
        end

      {[predicted | preds], [expected | acts], counts, disagrees}
    end)
    |> then(fn {p, a, counts, disagrees} ->
      IO.write("\r  Progress: #{total}/#{total}\n")
      {Enum.reverse(p), Enum.reverse(a), counts, Enum.reverse(disagrees)}
    end)
  end

  defp status_key(:ok), do: :ok
  defp status_key(:unknown), do: :unknown
  defp status_key(:error), do: :errored

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
