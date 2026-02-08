defmodule Mix.Tasks.Evaluate.Ner do
  alias Brain.ML.EntityExtractor
  alias Brain.ML
  @shortdoc "Evaluate named entity recognition accuracy"
  @moduledoc "Evaluate NER against gold standard data.\n\n## Usage\n\n    mix evaluate.ner              # Run evaluation\n    mix evaluate.ner --save       # Save results\n    mix evaluate.ner --verbose    # Show per-type details\n\n## Gold Standard Format\n\n    [\n      {\n        \"text\": \"What's the weather in London?\",\n        \"entities\": [{\"value\": \"London\", \"type\": \"location\"}]\n      }\n    ]\n"

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
    IO.puts(String.duplicate("=", 60) <> "\n")

    {predictions, actuals} = evaluate_all(gold)
    result = Evaluation.build_result("ner", predictions, actuals)

    IO.puts("Entity-level Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:              #{Float.round(result.macro_f1 * 100, 1)}%")
    IO.puts("")

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
      expected_entities = example["entities"] || []

      extracted =
        try do
          EntityExtractor.extract_entities(text)
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

  defp normalize_entity_value(nil) do
    ""
  end

  defp normalize_entity_value(value) when is_binary(value) do
    value
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_entity_value(value) do
    to_string(value)
  end
end