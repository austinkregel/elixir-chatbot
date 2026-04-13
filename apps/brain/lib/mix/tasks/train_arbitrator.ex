defmodule Mix.Tasks.TrainArbitrator do
  @moduledoc """
  Train the intent arbitrator meta-learner.

  Generates training data by running both LSTM and TF-IDF classifiers on
  gold standard evaluation data, then trains the stacked meta-learner to
  decide which classifier to trust.

  By default uses 5-fold cross-validation to prevent TF-IDF data leakage.

  ## Usage

      mix train_arbitrator [options]

  ## Options

    --epochs N          Number of training epochs (default: 50)
    --batch-size N      Batch size (default: 16)
    --lr FLOAT          Learning rate (default: 0.001)
    --naive             Use naive (non-CV) training data generation
    --folds N           Number of CV folds (default: 5)

  ## Prerequisites

  Both the unified LSTM model and TF-IDF classifier must be trained first.
  Gold standard data must exist at `priv/evaluation/intent/gold_standard.json`.
  """

  use Mix.Task
  require Logger

  alias Brain.ML.{IntentArbitrator, EvaluationStore}

  @shortdoc "Train intent arbitrator meta-learner"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          epochs: :integer,
          batch_size: :integer,
          lr: :float,
          naive: :boolean,
          folds: :integer
        ]
      )

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Intent Arbitrator Meta-Learner Training")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")

    gold = EvaluationStore.load_gold_standard("intent")

    if gold == [] do
      Mix.shell().error("No gold standard data found at priv/evaluation/intent/gold_standard.json")
      Mix.shell().error("Run: mix evaluate.intent --save  to generate evaluation data first")
      System.halt(1)
    end

    Mix.shell().info("Gold standard examples: #{length(gold)}")

    use_naive = opts[:naive] || false
    folds = opts[:folds] || 5

    if use_naive do
      Mix.shell().info("Mode: naive (no cross-validation, possible data leakage)")
    else
      Mix.shell().info("Mode: #{folds}-fold cross-validation")
    end

    Mix.shell().info("")
    Mix.shell().info("Generating training data (running both classifiers on each example)...")

    training_data =
      if use_naive do
        IntentArbitrator.generate_training_data(gold)
      else
        IntentArbitrator.generate_training_data_cv(gold, folds: folds)
      end

    lstm_count = Enum.count(training_data, fn %{label: l} -> l == :lstm end)
    tfidf_count = Enum.count(training_data, fn %{label: l} -> l == :tfidf end)
    total = length(training_data)

    Mix.shell().info("")
    Mix.shell().info("Usable training examples: #{total}")
    Mix.shell().info("  LSTM preferred: #{lstm_count} (#{pct(lstm_count, total)}%)")
    Mix.shell().info("  TF-IDF preferred: #{tfidf_count} (#{pct(tfidf_count, total)}%)")
    Mix.shell().info("")

    if total < 10 do
      Mix.shell().error("Not enough training data (need at least 10 examples)")
      System.halt(1)
    end

    train_opts = []
    train_opts = if opts[:epochs], do: [{:epochs, opts[:epochs]} | train_opts], else: train_opts
    train_opts = if opts[:batch_size], do: [{:batch_size, opts[:batch_size]} | train_opts], else: train_opts
    train_opts = if opts[:lr], do: [{:learning_rate, opts[:lr]} | train_opts], else: train_opts

    start_time = System.monotonic_time(:second)

    case IntentArbitrator.train(training_data, train_opts) do
      {:ok, _model, params} ->
        duration = System.monotonic_time(:second) - start_time

        IntentArbitrator.save_model(params)

        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("  Duration: #{duration}s")
        Mix.shell().info("  Training examples: #{total}")
        Mix.shell().info("  LSTM preferred: #{lstm_count} (#{pct(lstm_count, total)}%)")
        Mix.shell().info("  TF-IDF preferred: #{tfidf_count} (#{pct(tfidf_count, total)}%)")
        Mix.shell().info("")
        Mix.shell().info("Model saved. Reload with: Brain.ML.IntentArbitrator.reload()")

      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp pct(_n, 0), do: "0.0"
  defp pct(n, total), do: Float.round(n / total * 100, 1) |> to_string()
end
