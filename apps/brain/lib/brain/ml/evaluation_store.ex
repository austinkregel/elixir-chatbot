defmodule Brain.ML.EvaluationStore do
  @moduledoc """
  Stores and retrieves ML evaluation results as JSON files.

  Results are stored in `priv/evaluation/results/` with filenames like
  `intent_2026-02-07T12:00:00Z.json`.

  ## Usage

      result = Evaluation.build_result("intent", predictions, actuals)
      EvaluationStore.save(result)

      EvaluationStore.latest("intent")
      EvaluationStore.trend("intent", :accuracy)
  """

  require Logger

  @results_dir "evaluation/results"

  @doc """
  Save an evaluation result to disk.
  """
  def save(result) when is_map(result) do
    dir = results_path()
    File.mkdir_p!(dir)

    task = Map.get(result, :task, "unknown")
    timestamp = Map.get(result, :timestamp, DateTime.utc_now() |> DateTime.to_iso8601())
    safe_timestamp = timestamp |> String.replace(":", "-")
    filename = "#{task}_#{safe_timestamp}.json"
    path = Path.join(dir, filename)

    content = Jason.encode!(result, pretty: true)
    File.write!(path, content)
    Logger.info("EvaluationStore: Saved #{task} evaluation to #{filename}")
    {:ok, path}
  end

  @doc """
  Get the most recent evaluation for a task.
  """
  def latest(task) do
    case list_runs(task) do
      [] -> nil
      runs -> List.first(runs)
    end
  end

  @doc """
  List all evaluation runs for a task, newest first.
  """
  def list_runs(task) do
    dir = results_path()

    if File.dir?(dir) do
      dir
      |> File.ls!()
      |> Enum.filter(&String.starts_with?(&1, "#{task}_"))
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.sort(:desc)
      |> Enum.map(fn filename ->
        path = Path.join(dir, filename)

        case File.read(path) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, data} -> data
              _ -> nil
            end

          _ ->
            nil
        end
      end)
      |> Enum.reject(&is_nil/1)
    else
      []
    end
  end

  @doc """
  Get the accuracy trend over time for a task.

  Returns a list of `%{timestamp: string, value: float}` sorted oldest to newest.
  """
  def trend(task, metric \\ :accuracy) do
    list_runs(task)
    |> Enum.reverse()
    |> Enum.map(fn run ->
      %{
        timestamp: run["timestamp"],
        value: get_metric(run, metric)
      }
    end)
    |> Enum.reject(fn entry -> is_nil(entry.value) end)
  end

  @doc """
  Compare two evaluation runs and return the differences.
  """
  def compare(run_a, run_b) when is_map(run_a) and is_map(run_b) do
    %{
      accuracy_delta: (run_b["accuracy"] || 0) - (run_a["accuracy"] || 0),
      macro_f1_delta: (run_b["macro_f1"] || 0) - (run_a["macro_f1"] || 0),
      weighted_f1_delta: (run_b["weighted_f1"] || 0) - (run_a["weighted_f1"] || 0),
      examples_delta: (run_b["total_examples"] || 0) - (run_a["total_examples"] || 0),
      run_a_timestamp: run_a["timestamp"],
      run_b_timestamp: run_b["timestamp"]
    }
  end

  @doc """
  Load gold standard data for a task.

  Returns a list of annotated examples or an empty list if no data exists.
  """
  def load_gold_standard(task) do
    path = gold_standard_path(task)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) -> data
          _ -> []
        end

      {:error, _} ->
        []
    end
  end

  @doc """
  Get the path to the gold standard file for a task.
  """
  def gold_standard_path(task) do
    Path.join([evaluation_base_path(), task, "gold_standard.json"])
  end

  # Private

  defp results_path do
    Path.join(evaluation_base_path(), "results")
  end

  defp evaluation_base_path do
    Brain.priv_path("evaluation")
  end

  defp get_metric(run, :accuracy), do: run["accuracy"]
  defp get_metric(run, :macro_f1), do: run["macro_f1"]
  defp get_metric(run, :weighted_f1), do: run["weighted_f1"]
  defp get_metric(run, metric), do: run[to_string(metric)]
end
