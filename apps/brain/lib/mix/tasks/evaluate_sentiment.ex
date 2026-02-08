defmodule Mix.Tasks.Evaluate.Sentiment do
  @shortdoc "Evaluate sentiment analysis accuracy"
  @moduledoc """
  Evaluate sentiment classification against gold standard data.

  ## Usage

      mix evaluate.sentiment              # Run evaluation
      mix evaluate.sentiment --save       # Save results
      mix evaluate.sentiment --verbose    # Show per-class details

  ## Gold Standard Format

      [{"text": "I love this!", "sentiment": "positive"}, ...]
  """

  use Mix.Task

  alias Brain.ML.{Evaluation, EvaluationStore}

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
    IO.puts(String.duplicate("=", 60) <> "\n")

    unless Brain.ML.LSTM.UnifiedModel.ready?() do
      IO.puts("WARNING: UnifiedModel not ready. All predictions will be 'neutral'.\n")
    end

    {predictions, actuals} = evaluate_all(gold)
    result = Evaluation.build_result("sentiment", predictions, actuals)

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
      expected = example["sentiment"]

      predicted =
        if Brain.ML.LSTM.UnifiedModel.ready?() do
          case Brain.ML.LSTM.UnifiedModel.classify_sentiment(text) do
            {:ok, %{label: label}} -> to_string(label)
            _ -> "neutral"
          end
        else
          "neutral"
        end

      {[predicted | preds], [expected | acts]}
    end)
    |> then(fn {p, a} -> {Enum.reverse(p), Enum.reverse(a)} end)
  end
end
