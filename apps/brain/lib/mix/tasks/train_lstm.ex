defmodule Mix.Tasks.TrainLstm do
  @moduledoc "Train LSTM models using Axon's optimized training loop with EXLA.\n\n## Usage\n\n    mix train_lstm [options]\n\n## Options\n\n  --epochs N       Number of training epochs (default: 5)\n  --batch-size N   Batch size (default: 32)\n  --hidden-size N  LSTM hidden dimension (default: 64)\n  --lr FLOAT       Learning rate (default: 0.001)\n  --name NAME      Experiment name for A/B testing\n  --compare        Show comparison of all experiments\n\n## Examples\n\n    # Train with defaults (fast, ~1-2 minutes)\n    mix train_lstm\n\n    # A/B test different configurations\n    mix train_lstm --epochs 10 --name \"baseline\"\n    mix train_lstm --epochs 10 --hidden-size 128 --name \"larger_hidden\"\n    mix train_lstm --epochs 20 --name \"more_epochs\"\n\n    # Compare all experiments\n    mix train_lstm --compare\n\nThis task uses EXLA for accelerated training. The first epoch may be slower\ndue to JIT compilation, but subsequent epochs will be much faster.\n"

  alias Brain.ML.LSTM.AxonTrainer
  alias Brain.ML.LSTM.ExperimentTracker
  alias Brain.ML.ModelStore
  use Mix.Task
  require Logger

  @shortdoc "Train LSTM intent classifier with EXLA acceleration"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          epochs: :integer,
          batch_size: :integer,
          hidden_size: :integer,
          embedding_size: :integer,
          lr: :float,
          name: :string,
          compare: :boolean,
          publish: :boolean
        ]
      )

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    if opts[:compare] do
      ExperimentTracker.print_comparison()
      return_ok()
    else
      run_training(opts)
    end
  end

  defp run_training(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Intent Classifier Training (EXLA Accelerated)")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    config = []

    config =
      if opts[:epochs] do
        [{:epochs, opts[:epochs]} | config]
      else
        config
      end

    config =
      if opts[:batch_size] do
        [{:batch_size, opts[:batch_size]} | config]
      else
        config
      end

    config =
      if opts[:hidden_size] do
        [{:hidden_size, opts[:hidden_size]} | config]
      else
        config
      end

    config =
      if opts[:embedding_size] do
        [{:embedding_size, opts[:embedding_size]} | config]
      else
        config
      end

    config =
      if opts[:lr] do
        [{:learning_rate, opts[:lr]} | config]
      else
        config
      end

    config =
      if opts[:name] do
        [{:name, opts[:name]} | config]
      else
        config
      end

    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs, 5)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size, 64)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")
    Mix.shell().info("  Backend: EXLA (#{get_exla_target()})")

    if opts[:name] do
      Mix.shell().info("  Experiment: #{opts[:name]}")
    end

    Mix.shell().info("")

    start_time = System.monotonic_time(:second)

    case AxonTrainer.train_intent_classifier(config) do
      {:ok, result} ->
        duration = System.monotonic_time(:second) - start_time
        metrics = result[:metrics] || %{}

        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")

        Mix.shell().info(
          "  Epochs completed: #{metrics[:epochs_completed] || result.config.epochs}"
        )

        Mix.shell().info("  Vocabulary size: #{map_size(result.vocabularies.token_vocab)}")
        Mix.shell().info("  Intent classes: #{map_size(result.vocabularies.intent_to_idx)}")
        Mix.shell().info("")

        if metrics[:best_val_accuracy] do
          Mix.shell().info(
            "  Best validation accuracy: #{Float.round(metrics.best_val_accuracy * 100, 1)}%"
          )

          Mix.shell().info(
            "  Final train accuracy: #{Float.round(metrics.final_train_accuracy * 100, 1)}%"
          )

          Mix.shell().info("  Final validation loss: #{Float.round(metrics.final_val_loss, 3)}")
          Mix.shell().info("")
        end

        model_path = Brain.priv_path("ml_models/lstm/axon_intent.term")
        Mix.shell().info("Model saved to #{model_path}")

        if opts[:publish] do
          remote_key = ModelStore.version_prefix() <> "lstm/axon_intent.term"
          ModelStore.publish(model_path, remote_key)
        end

        if opts[:name] do
          Mix.shell().info(
            "Experiment '#{opts[:name]}' recorded. Run 'mix train_lstm --compare' to compare."
          )
        end

        Mix.shell().info("")

      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp return_ok do
    :ok
  end

  defp get_exla_target do
    System.get_env("XLA_TARGET") ||
      cond do
        File.dir?("/opt/rocm") -> "rocm"
        File.dir?("/usr/local/cuda") -> "cuda"
        true -> "cpu"
      end
  end
end
