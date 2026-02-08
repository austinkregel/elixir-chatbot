defmodule Mix.Tasks.Evaluate.SpeechAct do
  alias Brain.Analysis.SpeechActClassifier
  alias Brain.ML
  @shortdoc "Evaluate speech act classification accuracy"
  @moduledoc "Evaluate speech act classification against gold standard data.\n\n## Usage\n\n    mix evaluate.speech_act              # Run evaluation\n    mix evaluate.speech_act --save       # Save results\n    mix evaluate.speech_act --verbose    # Show per-class details\n\n## Gold Standard Format\n\n    [{\"text\": \"What time is it?\", \"speech_act\": \"directive\"}, ...]\n"

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
    IO.puts(String.duplicate("=", 60) <> "\n")

    {predictions, actuals} = evaluate_all(gold)
    result = Evaluation.build_result("speech_act", predictions, actuals)

    IO.puts("Overall Accuracy: #{Float.round(result.accuracy * 100, 1)}%")
    IO.puts("Macro F1:         #{Float.round(result.macro_f1 * 100, 1)}%")
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
      expected = example["speech_act"]

      predicted =
        case SpeechActClassifier.classify(text) do
          %{category: category} -> to_string(category)
          _ -> "unknown"
        end

      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end
end