defmodule Mix.Tasks.Evaluate.Ner do
  @shortdoc "Evaluate named entity recognition accuracy"
  @moduledoc """
  Evaluate NER against gold standard data.

  ## Usage

      mix evaluate.ner              # Run evaluation
      mix evaluate.ner --save       # Save results
      mix evaluate.ner --verbose    # Show per-type details

  ## Gold Standard Format

      [
        {
          "text": "What's the weather in London?",
          "entities": [{"value": "London", "type": "location"}]
        }
      ]
  """

  use Mix.Task

  alias Brain.ML.{Evaluation, EvaluationStore}

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
    # For NER, we evaluate at entity level:
    # Each expected entity is either found (correct type) or missed
    # Each extracted entity is either correct or a false positive
    Enum.reduce(gold, {[], []}, fn example, {preds, acts} ->
      text = example["text"]
      expected_entities = example["entities"] || []

      extracted =
        try do
          Brain.ML.EntityExtractor.extract_entities(text)
        rescue
          _ -> []
        catch
          :exit, _ -> []
        end

      # Match extracted to expected by value
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
    value
    |> String.downcase()
    |> String.trim()
  end

  defp normalize_entity_value(value), do: to_string(value)
end
