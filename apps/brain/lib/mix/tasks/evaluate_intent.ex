defmodule Mix.Tasks.Evaluate.Intent do
  alias Brain.Analysis.Pipeline
  alias Brain.ML
  @shortdoc "Evaluate intent classification accuracy"
  @moduledoc "Evaluate intent classification against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy.\n\n## Usage\n\n    mix evaluate.intent              # Run evaluation\n    mix evaluate.intent --save       # Save results\n    mix evaluate.intent --compare    # Compare with previous\n    mix evaluate.intent --verbose    # Show confusion matrix\n"

  use Mix.Task

  alias ML.{Evaluation, EvaluationStore}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    compare? = "--compare" in args
    verbose? = "--verbose" in args

    gold = EvaluationStore.load_gold_standard("intent")

    if gold == [] do
      IO.puts("\nNo gold standard data for intent classification.")
      IO.puts("Add annotated examples to: priv/evaluation/intent/gold_standard.json")
      IO.puts("")
      IO.puts("Format: [{\"text\": \"What's the weather?\", \"intent\": \"weather.query\"}, ...]")
      IO.puts("")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("INTENT CLASSIFICATION EVALUATION (#{length(gold)} examples)")
    IO.puts("(Using production pipeline)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals} = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    cm = Evaluation.confusion_matrix(predictions, actuals)
    report = Evaluation.classification_report(cm)
    result = Evaluation.build_result("intent", predictions, actuals)

    IO.puts("Overall Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:         #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("Weighted F1:      #{Float.round(result.weighted_f1 * 100, 1)}%")
    IO.puts("Total Examples:   #{result.total_examples}")
    IO.puts("Duration:         #{duration_ms}ms")
    IO.puts("")

    IO.puts(Evaluation.format_report(report))
    IO.puts("")

    broadcast_result(result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("intent", %{
      accuracy: result.accuracy,
      macro_f1: result.macro_f1,
      weighted_f1: result.weighted_f1,
      total_examples: result.total_examples,
      duration_ms: duration_ms
    })

    if verbose? do
      IO.puts("\n--- Confusion Matrix ---\n")
      print_confusion_matrix(cm)
    end

    if save? do
      {:ok, path} = EvaluationStore.save(result)
      IO.puts("Results saved to: #{path}")
    end

    if compare? do
      runs = EvaluationStore.list_runs("intent")

      case runs do
        [_current | [previous | _]] ->
          delta = EvaluationStore.compare(previous, result)

          sign =
            if delta.accuracy_delta >= 0 do
              "+"
            else
              ""
            end

          IO.puts("\n--- Comparison with previous ---")
          IO.puts("  Accuracy: #{sign}#{Float.round(delta.accuracy_delta * 100, 1)}%")
          IO.puts("  Macro F1: #{sign}#{Float.round(delta.macro_f1_delta * 100, 1)}%")

        _ ->
          IO.puts("\nNo previous run to compare with.")
      end
    end

    IO.puts("")
  end

  defp evaluate_all(gold) do
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected = example["intent"]

      predicted =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          analysis.intent || "unknown"
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
        task: "intent",
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

  defp print_confusion_matrix(cm) do
    all_labels =
      (Map.keys(cm) ++ Enum.flat_map(Map.values(cm), &Map.keys/1))
      |> Enum.uniq()
      |> Enum.sort()

    label_width = all_labels |> Enum.map(&String.length(to_string(&1))) |> Enum.max(fn -> 10 end)
    label_width = max(label_width, 10)
    cell_width = 5
    header = String.pad_trailing("Actual \\ Pred", label_width + 2)

    header =
      header <>
        Enum.map_join(
          all_labels,
          " ",
          &String.pad_leading(String.slice(to_string(&1), 0, cell_width), cell_width)
        )

    IO.puts(header)
    IO.puts(String.duplicate("-", String.length(header)))

    Enum.each(all_labels, fn actual ->
      row = String.pad_trailing(to_string(actual), label_width + 2)

      cells =
        Enum.map_join(all_labels, " ", fn predicted ->
          count = get_in(cm, [actual, predicted]) || 0
          String.pad_leading(to_string(count), cell_width)
        end)

      IO.puts(row <> cells)
    end)
  end
end
