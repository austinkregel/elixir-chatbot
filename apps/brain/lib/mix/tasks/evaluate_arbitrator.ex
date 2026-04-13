defmodule Mix.Tasks.Evaluate.Arbitrator do
  @shortdoc "Evaluate intent arbitrator accuracy"
  @moduledoc """
  Evaluate the intent arbitrator meta-learner against gold standard data.

  Runs both LSTM and TF-IDF classifiers on each example, lets the arbitrator
  choose, and compares the final intent against the gold standard. Also shows
  how each individual classifier performs for comparison.

  ## Usage

      mix evaluate.arbitrator              # Run evaluation
      mix evaluate.arbitrator --save       # Save results
      mix evaluate.arbitrator --verbose    # Show per-class details

  ## Gold Standard Format

  Uses the intent gold standard: `priv/evaluation/intent/gold_standard.json`

      [{"text": "What's the weather?", "intent": "weather.query"}, ...]
  """

  use Mix.Task
  require Logger

  alias Brain.ML.{IntentArbitrator, IntentClassifierSimple, Evaluation, EvaluationStore}

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    save? = "--save" in args
    verbose? = "--verbose" in args

    gold = EvaluationStore.load_gold_standard("intent")

    if gold == [] do
      IO.puts("\nNo gold standard data for intent classification.")
      IO.puts("Add examples to: priv/evaluation/intent/gold_standard.json")
      IO.puts("")
      exit(:normal)
    end

    unless IntentArbitrator.ready?() do
      IO.puts("\nIntentArbitrator is not ready.")
      IO.puts("Train it first: mix train_arbitrator")
      IO.puts("")
      exit(:normal)
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("INTENT ARBITRATOR EVALUATION (#{length(gold)} examples)")
    IO.puts(String.duplicate("=", 60) <> "\n")

    start_time = System.monotonic_time(:millisecond)
    results = evaluate_all(gold)
    duration_ms = System.monotonic_time(:millisecond) - start_time

    arb_preds = Enum.map(results, & &1.arbitrated)
    lstm_preds = Enum.map(results, & &1.lstm_intent)
    tfidf_preds = Enum.map(results, & &1.tfidf_intent)
    actuals = Enum.map(results, & &1.expected)

    arb_result = Evaluation.build_result("arbitrator", arb_preds, actuals)
    lstm_result = Evaluation.build_result("arbitrator_lstm_baseline", lstm_preds, actuals)
    tfidf_result = Evaluation.build_result("arbitrator_tfidf_baseline", tfidf_preds, actuals)

    choices = Enum.frequencies_by(results, & &1.choice)
    lstm_chosen = Map.get(choices, :lstm, 0)
    tfidf_chosen = Map.get(choices, :tfidf, 0)

    IO.puts("Arbitrator Accuracy:  #{pct(arb_result.accuracy)}%")
    IO.puts("  vs LSTM-only:       #{pct(lstm_result.accuracy)}%")
    IO.puts("  vs TF-IDF-only:     #{pct(tfidf_result.accuracy)}%")
    IO.puts("")
    IO.puts("Arbitrator Macro F1:  #{pct(arb_result.macro_f1)}%")
    IO.puts("  vs LSTM-only:       #{pct(lstm_result.macro_f1)}%")
    IO.puts("  vs TF-IDF-only:     #{pct(tfidf_result.macro_f1)}%")
    IO.puts("")
    IO.puts("Classifier Selection: LSTM=#{lstm_chosen}, TF-IDF=#{tfidf_chosen}")
    IO.puts("Duration:             #{duration_ms}ms")
    IO.puts("")

    broadcast_result(arb_result, duration_ms)
    Brain.Telemetry.emit_evaluation_complete("arbitrator", %{
      accuracy: arb_result.accuracy,
      macro_f1: arb_result.macro_f1,
      weighted_f1: arb_result.weighted_f1,
      total_examples: arb_result.total_examples,
      duration_ms: duration_ms,
      lstm_baseline_accuracy: lstm_result.accuracy,
      tfidf_baseline_accuracy: tfidf_result.accuracy
    })

    if verbose? do
      IO.puts("--- Arbitrator Per-Class Report ---\n")
      cm = Evaluation.confusion_matrix(arb_preds, actuals)
      report = Evaluation.classification_report(cm)
      IO.puts(Evaluation.format_report(report))
      IO.puts("")
    end

    if save? do
      {:ok, path} = EvaluationStore.save(arb_result)
      IO.puts("Results saved to: #{path}")
    end

    IO.puts("")
  end

  defp evaluate_all(gold) do
    total = length(gold)

    gold
    |> Enum.with_index(1)
    |> Enum.map(fn {example, idx} ->
      text = example["text"]
      expected = example["intent"]

      if rem(idx, 500) == 0, do: IO.write("\r  Progress: #{idx}/#{total}")

      lstm_result = classify_lstm(text)
      tfidf_result = classify_tfidf(text)

      features =
        IntentArbitrator.extract_features(%{
          lstm: lstm_result,
          tfidf: tfidf_result,
          text: text
        })

      {choice, _prob} =
        case IntentArbitrator.arbitrate(features) do
          {:ok, {choice, prob}} -> {choice, prob}
          {:ok, choice} when is_atom(choice) -> {choice, 0.5}
          _ -> {:tfidf, 0.5}
        end

      arbitrated_intent =
        case choice do
          :lstm -> to_string(lstm_result[:intent] || "unknown")
          :tfidf -> to_string(tfidf_result[:intent] || "unknown")
          _ -> to_string(tfidf_result[:intent] || "unknown")
        end

      %{
        expected: expected,
        arbitrated: arbitrated_intent,
        lstm_intent: to_string(lstm_result[:intent] || "unknown"),
        tfidf_intent: to_string(tfidf_result[:intent] || "unknown"),
        choice: choice
      }
    end)
    |> tap(fn _ -> IO.write("\r  Progress: #{total}/#{total}\n") end)
  end

  defp classify_lstm(text) do
    case Brain.ML.LSTM.UnifiedModel.classify_intent(text) do
      {:ok, %{label: intent, confidence: conf, scores: scores}} ->
        %{intent: intent, confidence: conf, scores: scores}

      {:ok, {intent, conf}} ->
        %{intent: intent, confidence: conf, scores: []}

      _ ->
        %{intent: nil, confidence: 0.0, scores: []}
    end
  rescue
    _ -> %{intent: nil, confidence: 0.0, scores: []}
  catch
    _, _ -> %{intent: nil, confidence: 0.0, scores: []}
  end

  defp classify_tfidf(text) do
    case IntentClassifierSimple.classify(text, with_details: true, top_k: 5) do
      {:ok, %{intent: intent, confidence: conf} = result} ->
        top_k = Map.get(result, :top_k, [])

        scores =
          Enum.map(top_k, fn
            {label, score} -> {label, score}
            %{intent: i, score: s} -> {i, s}
            _ -> {"unknown", 0.0}
          end)

        %{intent: intent, confidence: conf, scores: scores}

      {:ok, {intent, conf}} ->
        %{intent: intent, confidence: conf, scores: []}

      _ ->
        %{intent: nil, confidence: 0.0, scores: []}
    end
  rescue
    _ -> %{intent: nil, confidence: 0.0, scores: []}
  catch
    _, _ -> %{intent: nil, confidence: 0.0, scores: []}
  end

  defp pct(value), do: Float.round(value * 100, 1)

  defp broadcast_result(result, duration_ms) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "evaluation:complete",
      {:evaluation_complete, %{
        task: "arbitrator",
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
