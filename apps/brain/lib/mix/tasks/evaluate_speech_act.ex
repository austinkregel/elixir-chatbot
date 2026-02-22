defmodule Mix.Tasks.Evaluate.SpeechAct do
  alias Brain.Analysis.Pipeline
  alias Brain.ML
  @shortdoc "Evaluate speech act classification accuracy"
  @moduledoc "Evaluate speech act classification against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy.\n\n## Usage\n\n    mix evaluate.speech_act              # Run evaluation\n    mix evaluate.speech_act --save       # Save results\n    mix evaluate.speech_act --verbose    # Show per-class details\n\n## Gold Standard Format\n\n    [{\"text\": \"What time is it?\", \"speech_act\": \"directive\"}, ...]\n"

  use Mix.Task

  alias ML.{Evaluation, EvaluationStore}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args

    gold = EvaluationStore.load_gold_standard("speech_act")

    if gold == [] do
      IO.puts("\nNo gold standard data for speech act classification.")
      IO.puts("Add examples to: priv/evaluation/speech_act/gold_standard.json")
      IO.puts("")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("SPEECH ACT EVALUATION (#{length(gold)} examples)")
    IO.puts("(Using production pipeline)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals} = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    result = Evaluation.build_result("speech_act", predictions, actuals)

    IO.puts("Overall Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:         #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("Duration:         #{duration_ms}ms")
    IO.puts("")

    broadcast_result(result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("speech_act", %{
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
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["speech_act"]

      predicted =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          to_string(analysis.speech_act.category)
        rescue
          _ -> "unknown"
        catch
          :exit, _ -> "unknown"
        end

      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end

  defp broadcast_result(result, duration_ms) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "evaluation:complete",
      {:evaluation_complete, %{
        task: "speech_act",
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
