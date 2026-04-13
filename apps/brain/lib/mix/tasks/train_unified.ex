defmodule Mix.Tasks.TrainUnified do
  @moduledoc "Train the unified LSTM model for multi-task NLP.\n\n## Usage\n\n    mix train_unified [options]\n\n## Options\n\n  --epochs N          Number of training epochs (default: 20)\n  --batch-size N      Batch size (default: 32)\n  --hidden-size N     LSTM hidden dimension (default: 128)\n  --lr FLOAT          Learning rate (default: 0.001)\n  --name NAME         Experiment name for tracking (default: unified_YYYYMMDD_HHMMSS)\n  --compare           Print experiment comparison table after training\n  --min-examples N    Min examples per intent to include in training (default: 10)\n\n## Examples\n\n    # Train with defaults\n    mix train_unified\n\n    # Train with more epochs\n    mix train_unified --epochs 30 --name \"unified_30ep\"\n\n    # Lower intent threshold to include sparse intents\n    mix train_unified --min-examples 5\n\nThis trains a shared LSTM encoder that powers:\n- Intent classification\n- Named Entity Recognition (NER)\n- Sentiment analysis\n- Speech act classification\n"

  alias Brain.ML.LSTM.UnifiedModel
  alias Brain.ML.ModelStore
  use Mix.Task
  require Logger

  alias Brain.ML.LSTM.ExperimentTracker

  @shortdoc "Train unified multi-task LSTM model"

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
          min_examples: :integer,
          publish: :boolean
        ]
      )

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Unified Multi-Task LSTM Training (EXLA Accelerated)")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    Mix.shell().info("This model powers:")
    Mix.shell().info("  - Intent Classification")
    Mix.shell().info("  - Named Entity Recognition")
    Mix.shell().info("  - Sentiment Analysis")
    Mix.shell().info("  - Speech Act Classification")
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
        [{:hidden_size, 128} | config]
      end

    config =
      if opts[:embedding_size] do
        [{:embedding_size, opts[:embedding_size]} | config]
      else
        [{:embedding_size, 128} | config]
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

    config =
      if opts[:min_examples] do
        [{:min_examples_per_intent, opts[:min_examples]} | config]
      else
        config
      end

    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size)}")
    Mix.shell().info("  Embedding size: #{Keyword.get(config, :embedding_size)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")

    if opts[:name] do
      Mix.shell().info("  Experiment: #{opts[:name]}")
    end

    Mix.shell().info("")

    start_time = System.monotonic_time(:second)

    case UnifiedModel.train(config) do
      {:ok, result} ->
        duration = System.monotonic_time(:second) - start_time

        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")
        Mix.shell().info("  Vocabulary size: #{map_size(result.vocabularies.token_vocab)}")
        Mix.shell().info("  Intent classes: #{map_size(result.vocabularies.intent_to_idx)}")
        Mix.shell().info("")
        model_path = Brain.priv_path("ml_models/lstm/unified_model.term")
        Mix.shell().info("Model saved to #{model_path}")

        if opts[:publish] do
          remote_key = ModelStore.version_prefix() <> "lstm/unified_model.term"
          ModelStore.publish(model_path, remote_key)
        end

        Mix.shell().info("")
        Mix.shell().info("Usage:")
        Mix.shell().info("  Brain.ML.LSTM.UnifiedModel.analyze(\"What's the weather?\")")
        Mix.shell().info("")
        experiment_name = opts[:name] || generate_experiment_name("unified")

        ExperimentTracker.record(%{
          name: experiment_name,
          config: Enum.into(config, %{}),
          epochs_completed: Keyword.get(config, :epochs),
          training_time_seconds: duration,
          notes:
            "Unified multi-task LSTM. Vocab: #{map_size(result.vocabularies.token_vocab)}, Intents: #{map_size(result.vocabularies.intent_to_idx)}"
        })

        Mix.shell().info("  Experiment recorded: #{experiment_name}")

        if opts[:compare] do
          Mix.shell().info("")
          ExperimentTracker.print_comparison()
        end

      {:error, reason} ->
        Mix.shell().error("Training failed: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp generate_experiment_name(prefix) do
    now = NaiveDateTime.utc_now()

    ts =
      now
      |> NaiveDateTime.to_iso8601()
      |> String.slice(0, 19)
      |> String.replace("-", "")
      |> String.replace("T", "_")
      |> String.replace(":", "")

    "#{prefix}_#{ts}"
  end
end
