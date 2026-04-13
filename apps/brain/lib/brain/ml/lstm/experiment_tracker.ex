defmodule Brain.ML.LSTM.ExperimentTracker do
  @moduledoc "Track and compare ML training experiments.\n\nStores experiment results so you can A/B test different configurations\nand compare accuracy, loss, and training time.\n\n## Usage\n\n    # Run experiments\n    mix train_lstm --epochs 10 --name \"baseline\"\n    mix train_lstm --epochs 10 --hidden-size 128 --name \"larger_hidden\"\n    mix train_lstm --epochs 20 --name \"more_epochs\"\n\n    # Compare results\n    ExperimentTracker.compare_all()\n    ExperimentTracker.best_by(:val_accuracy)\n"

  require Logger

  @experiments_file "experiments.json"

  defstruct [
    :name,
    :config,
    :final_train_accuracy,
    :final_train_loss,
    :final_val_accuracy,
    :final_val_loss,
    :best_val_accuracy,
    :best_val_loss,
    :epochs_completed,
    :training_time_seconds,
    :timestamp,
    :notes
  ]

  @doc "Record an experiment result.\n"
  def record(experiment) when is_struct(experiment, __MODULE__) do
    experiments = load_experiments()
    updated = [experiment_to_map(experiment) | experiments]
    save_experiments(updated)
    Logger.info("Recorded experiment: #{experiment.name}")
    :ok
  end

  def record(attrs) when is_map(attrs) do
    experiment =
      struct(__MODULE__, Map.put(attrs, :timestamp, DateTime.utc_now() |> DateTime.to_iso8601()))

    record(experiment)
  end

  @doc "List all experiments, sorted by timestamp (newest first).\n"
  def list_all do
    load_experiments()
    |> Enum.sort_by(& &1["timestamp"], :desc)
  end

  @doc "Get the best experiment by a given metric.\n\n## Examples\n\n    ExperimentTracker.best_by(:val_accuracy)  # highest validation accuracy\n    ExperimentTracker.best_by(:val_loss, :min)  # lowest validation loss\n"
  def best_by(metric, direction \\ :max) do
    experiments = load_experiments()

    key = metric_to_key(metric)

    experiments
    |> Enum.filter(& &1[key])
    |> case do
      [] ->
        nil

      exps ->
        case direction do
          :max -> Enum.max_by(exps, & &1[key])
          :min -> Enum.min_by(exps, & &1[key])
        end
    end
  end

  @doc "Compare all experiments in a table format.\nReturns a list of maps with key metrics for comparison.\n"
  def compare_all do
    experiments = load_experiments()

    experiments
    |> Enum.map(fn exp ->
      %{
        name: exp["name"],
        val_acc: format_percent(exp["best_val_accuracy"]),
        val_loss: format_float(exp["best_val_loss"]),
        train_acc: format_percent(exp["final_train_accuracy"]),
        epochs: exp["epochs_completed"],
        time: "#{exp["training_time_seconds"]}s",
        hidden: get_in(exp, ["config", "hidden_size"]),
        batch: get_in(exp, ["config", "batch_size"]),
        lr: get_in(exp, ["config", "learning_rate"]),
        timestamp: exp["timestamp"]
      }
    end)
    |> Enum.sort_by(& &1.val_acc, :desc)
  end

  @doc "Print a comparison table to the console.\n"
  def print_comparison do
    results = compare_all()

    if Enum.empty?(results) do
      IO.puts("\nNo experiments recorded yet.\n")
      IO.puts("Run: mix train_lstm --epochs 10 --name \"my_experiment\"\n")
    else
      IO.puts("\n" <> String.duplicate("=", 100))
      IO.puts("EXPERIMENT COMPARISON (sorted by validation accuracy)")
      IO.puts(String.duplicate("=", 100))
      IO.puts("")

      IO.puts(
        format_row([
          "Name",
          "Val Acc",
          "Val Loss",
          "Train Acc",
          "Epochs",
          "Time",
          "Hidden",
          "Batch",
          "LR"
        ])
      )

      IO.puts(String.duplicate("-", 100))

      Enum.each(results, fn r ->
        IO.puts(
          format_row([
            r.name || "unnamed",
            r.val_acc,
            r.val_loss,
            r.train_acc,
            r.epochs,
            r.time,
            r.hidden,
            r.batch,
            r.lr
          ])
        )
      end)

      IO.puts("")

      case best_by(:best_val_accuracy) do
        nil ->
          :ok

        best ->
          IO.puts(
            "Best experiment: #{best["name"]} with #{format_percent(best["best_val_accuracy"])} validation accuracy"
          )
      end

      IO.puts("")
    end
  end

  @doc "Delete all experiments.\n"
  def clear_all do
    save_experiments([])
    Logger.info("Cleared all experiments")
    :ok
  end

  @doc "Delete a specific experiment by name.\n"
  def delete(name) do
    experiments = load_experiments()
    updated = Enum.reject(experiments, &(&1["name"] == name))
    save_experiments(updated)
    Logger.info("Deleted experiment: #{name}")
    :ok
  end

  defp experiments_path do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)
    Path.join(lstm_path, @experiments_file)
  end

  defp load_experiments do
    case File.read(experiments_path()) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, experiments} -> experiments
          _ -> []
        end

      _ ->
        []
    end
  end

  defp save_experiments(experiments) do
    content = Jason.encode!(experiments, pretty: true)
    File.write!(experiments_path(), content)
  end

  defp experiment_to_map(experiment) do
    %{
      "name" => experiment.name,
      "config" => experiment.config,
      "final_train_accuracy" => experiment.final_train_accuracy,
      "final_train_loss" => experiment.final_train_loss,
      "final_val_accuracy" => experiment.final_val_accuracy,
      "final_val_loss" => experiment.final_val_loss,
      "best_val_accuracy" => experiment.best_val_accuracy,
      "best_val_loss" => experiment.best_val_loss,
      "epochs_completed" => experiment.epochs_completed,
      "training_time_seconds" => experiment.training_time_seconds,
      "timestamp" => experiment.timestamp,
      "notes" => experiment.notes
    }
  end

  defp metric_to_key(:val_accuracy) do
    "best_val_accuracy"
  end

  defp metric_to_key(:val_loss) do
    "best_val_loss"
  end

  defp metric_to_key(:train_accuracy) do
    "final_train_accuracy"
  end

  defp metric_to_key(:train_loss) do
    "final_train_loss"
  end

  defp metric_to_key(:best_val_accuracy) do
    "best_val_accuracy"
  end

  defp metric_to_key(:best_val_loss) do
    "best_val_loss"
  end

  defp metric_to_key(other) do
    to_string(other)
  end

  defp format_percent(nil) do
    "-"
  end

  defp format_percent(val) when is_float(val) do
    "#{Float.round(val * 100, 1)}%"
  end

  defp format_percent(val) do
    "#{val}%"
  end

  defp format_float(nil) do
    "-"
  end

  defp format_float(val) when is_float(val) do
    Float.round(val, 3) |> to_string()
  end

  defp format_float(val) do
    to_string(val)
  end

  defp format_row(items) do
    widths = [20, 10, 10, 10, 8, 8, 8, 8, 10]

    items
    |> Enum.zip(widths)
    |> Enum.map_join(
      " ",
      fn {item, width} ->
        String.pad_trailing(to_string(item || "-"), width)
      end
    )
  end
end