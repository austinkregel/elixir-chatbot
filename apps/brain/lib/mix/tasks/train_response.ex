defmodule Mix.Tasks.TrainResponse do
  @moduledoc "Train the LSTM response scoring model.\n\n## Usage\n\n    mix train_response [options]\n\n## Options\n\n  --epochs N       Number of training epochs (default: 15)\n  --batch-size N   Batch size (default: 32)\n  --hidden-size N  LSTM hidden dimension (default: 128)\n  --lr FLOAT       Learning rate (default: 0.001)\n  --name NAME      Experiment name for tracking (default: response_YYYYMMDD_HHMMSS)\n  --compare        Print experiment comparison table after training\n\n## What This Trains\n\nThe response scorer learns to evaluate query-response pairs:\n- Positive examples: actual responses from training data\n- Negative examples: mismatched query-response pairs\n\nAfter training, the scorer is used to:\n- Pick the best response from multiple candidates\n- Detect low-quality or irrelevant responses\n- Improve response selection accuracy\n\n## Examples\n\n    # Train with defaults\n    mix train_response\n\n    # Train with more epochs\n    mix train_response --epochs 25\n"

  alias Brain.Response.LSTMResponse
  alias Brain.ML.ModelStore
  use Mix.Task
  require Logger

  alias Brain.ML.LSTM.ExperimentTracker

  @shortdoc "Train LSTM response quality scorer"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          epochs: :integer,
          batch_size: :integer,
          hidden_size: :integer,
          lr: :float,
          name: :string,
          compare: :boolean,
          publish: :boolean
        ]
      )

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("LSTM Response Scorer Training")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    Mix.shell().info("This model learns to score query-response pairs")
    Mix.shell().info("enabling better response selection.")
    Mix.shell().info("")
    config = []

    config =
      if opts[:epochs] do
        [{:epochs, opts[:epochs]} | config]
      else
        [{:epochs, 15} | config]
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
      if opts[:lr] do
        [{:learning_rate, opts[:lr]} | config]
      else
        config
      end

    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{Keyword.get(config, :epochs)}")
    Mix.shell().info("  Batch size: #{Keyword.get(config, :batch_size, 32)}")
    Mix.shell().info("  Hidden size: #{Keyword.get(config, :hidden_size)}")
    Mix.shell().info("  Learning rate: #{Keyword.get(config, :learning_rate, 0.001)}")
    Mix.shell().info("")

    start_time = System.monotonic_time(:second)

    case LSTMResponse.train(config) do
      {:ok, _result} ->
        duration = System.monotonic_time(:second) - start_time

        Mix.shell().info("")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("Training Complete!")
        Mix.shell().info("=" |> String.duplicate(60))
        Mix.shell().info("")
        Mix.shell().info("  Total time: #{duration} seconds")
        Mix.shell().info("")
        model_path = Brain.priv_path("ml_models/lstm/response_scorer.term")
        Mix.shell().info("Model saved to #{model_path}")

        if opts[:publish] do
          remote_key = ModelStore.version_prefix() <> "lstm/response_scorer.term"
          ModelStore.publish(model_path, remote_key)
        end

        Mix.shell().info("")
        Mix.shell().info("Usage:")
        Mix.shell().info("  # Score a response")

        Mix.shell().info(
          "  LSTMResponse.score_response(\"What's the weather?\", \"It's sunny.\")"
        )

        Mix.shell().info("")
        Mix.shell().info("  # Generate best response")
        Mix.shell().info("  LSTMResponse.generate(query, intent, entities)")
        Mix.shell().info("")
        experiment_name = opts[:name] || generate_experiment_name("response")

        ExperimentTracker.record(%{
          name: experiment_name,
          config: Enum.into(config, %{}),
          epochs_completed: Keyword.get(config, :epochs),
          training_time_seconds: duration,
          notes: "Response scorer LSTM"
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
