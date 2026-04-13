defmodule Mix.Tasks.Evaluate.Ner do
  alias Brain.Analysis.Pipeline
  alias Brain.ML
  @shortdoc "Evaluate named entity recognition accuracy"
  @moduledoc "Evaluate NER against gold standard data.\n\nUses the full production pipeline for evaluation to measure real-world accuracy,\nincluding type normalization and disambiguation.\n\n## Usage\n\n    mix evaluate.ner              # Run evaluation\n    mix evaluate.ner --save       # Save results\n    mix evaluate.ner --verbose    # Show per-type details\n\n## Gold Standard Format\n\nAccepts either `\"entities\"` or `\"expected\"` as the key for entity annotations:\n\n    [\n      {\n        \"text\": \"What's the weather in London?\",\n        \"entities\": [{\"value\": \"London\", \"type\": \"location\"}]\n      }\n    ]\n"

  use Mix.Task

  alias ML.{Evaluation, EvaluationStore}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args

    gold = EvaluationStore.load_gold_standard("ner")

    if gold == [] do
      IO.puts("\nNo gold standard data for NER.")
      IO.puts("Add examples to: priv/evaluation/ner/gold_standard.json")
      IO.puts("")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("NER EVALUATION (#{length(gold)} examples)")
    IO.puts("(Using production pipeline with type normalization)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    {predictions, actuals} = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    result = Evaluation.build_result("ner", predictions, actuals)

    IO.puts("Entity-level Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:              #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("Duration:              #{duration_ms}ms")
    IO.puts("")

    broadcast_result(result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("ner", %{
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
      expected_entities = example["entities"] || example["expected"] || []

      extracted =
        try do
          analysis = Pipeline.analyze_chunk(text, side_effects: false)
          analysis.entities || []
        rescue
          _ -> []
        catch
          :exit, _ -> []
        end

      {matched_preds, matched_acts} =
        Enum.reduce(expected_entities, {[], []}, fn expected, {p_acc, a_acc} ->
          expected_value = expected["value"]
          expected_type = expected["type"]

          match =
            Enum.find(extracted, fn ext ->
              ext_value = Map.get(ext, :value) || Map.get(ext, "value")
              normalize_entity_value(ext_value) == normalize_entity_value(expected_value)
            end)

          predicted_type =
            if match do
              Map.get(match, :entity_type) || Map.get(match, "entity_type") || "none"
            else
              "none"
            end

          {[to_string(predicted_type) | p_acc], [to_string(expected_type) | a_acc]}
        end)

      {Enum.reverse(matched_preds) ++ preds, Enum.reverse(matched_acts) ++ acts}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end

  defp normalize_entity_value(nil), do: ""

  defp normalize_entity_value(value) when is_binary(value) do
    value |> String.downcase() |> String.trim()
  end

  defp normalize_entity_value(value), do: to_string(value)

  defp broadcast_result(result, duration_ms) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "evaluation:complete",
      {:evaluation_complete, %{
        task: "ner",
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
