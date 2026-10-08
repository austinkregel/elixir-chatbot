defmodule Brain.ML.EvaluationStore do
  @moduledoc """
  Stores and retrieves ML evaluation results.

  Primary storage: Atlas `atlas_evaluation_results` table.
  Write-through: JSON files in `priv/evaluation/results/` for offline access.

  ## Usage

      result = Evaluation.build_result("intent", predictions, actuals)
      EvaluationStore.save(result)

      EvaluationStore.latest("intent")
      EvaluationStore.trend("intent", :accuracy)
  """

  require Logger

  @doc """
  Save an evaluation result to Atlas (primary) and disk (write-through).
  """
  def save(result) when is_map(result) do
    task = Map.get(result, :task, "unknown")

    # Primary: write to Atlas
    save_to_atlas(result)

    # Write-through: file-based for offline access
    file_result = save_to_file(result)

    # Emit telemetry
    Brain.Telemetry.emit_evaluation_complete(task, %{
      accuracy: Map.get(result, :accuracy, 0.0),
      macro_f1: Map.get(result, :macro_f1, 0.0),
      weighted_f1: Map.get(result, :weighted_f1, 0.0),
      total_examples: Map.get(result, :total_examples, 0),
      duration_ms: Map.get(result, :duration_ms, 0)
    })

    file_result
  end

  @doc """
  Get the most recent evaluation for a task.

  Queries Atlas first, falls back to file-based storage.
  """
  def latest(task) do
    case latest_from_atlas(task) do
      nil ->
        case list_runs_from_files(task) do
          [] -> nil
          runs -> List.first(runs)
        end

      result ->
        result
    end
  end

  @doc """
  List all evaluation runs for a task, newest first.

  Queries Atlas first, falls back to file-based storage.
  """
  def list_runs(task) do
    case list_runs_from_atlas(task) do
      [] -> list_runs_from_files(task)
      runs -> runs
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
        timestamp: run["timestamp"] || run[:timestamp],
        value: get_metric(run, metric)
      }
    end)
    |> Enum.reject(fn entry -> is_nil(entry.value) end)
  end

  @doc """
  Compare two evaluation runs and return the differences.
  """
  def compare(run_a, run_b) when is_map(run_a) and is_map(run_b) do
    get_val = fn map, key ->
      Map.get(map, key) || Map.get(map, to_string(key), 0)
    end

    %{
      accuracy_delta: get_val.(run_b, :accuracy) - get_val.(run_a, :accuracy),
      macro_f1_delta: get_val.(run_b, :macro_f1) - get_val.(run_a, :macro_f1),
      weighted_f1_delta: get_val.(run_b, :weighted_f1) - get_val.(run_a, :weighted_f1),
      examples_delta: get_val.(run_b, :total_examples) - get_val.(run_a, :total_examples),
      run_a_timestamp: Map.get(run_a, "timestamp") || Map.get(run_a, :timestamp),
      run_b_timestamp: Map.get(run_b, "timestamp") || Map.get(run_b, :timestamp)
    }
  end

  @doc """
  Load gold standard data for a task.

  `split` selects which partition of the corpus to return:

    * `:all` (default) -- every annotated example
    * `:train` -- everything except the held-out split
    * `:held_out` -- only the held-out split

  The split is applied **here, at read time**, against `held_out.json` beside
  the gold standard. It is deliberately not carved out of
  `gold_standard.json`: `mix rebuild_gold_standard`, `cleanup_gold_standard`,
  `normalize_gold_standard` and `augment_training_data` all regenerate that
  file, and any of them would silently undo a split stored in it.

  `:train` and `:held_out` require the split to exist. Without it there is no
  such thing as a training partition, and answering as though there were is how
  every accuracy figure in this repo came to be measured on its own training
  data. Use `:all` when you genuinely want the whole corpus.

  Raises on a missing or malformed file. This used to return `[]`, which made
  "no corpus" and "the corpus failed to load" indistinguishable.
  """
  @spec load_gold_standard(String.t(), :all | :train | :held_out) :: [map()]
  def load_gold_standard(task, split \\ :all)

  def load_gold_standard(task, :all), do: read_json_array!(gold_standard_path(task))

  def load_gold_standard(task, :held_out), do: read_json_array!(held_out_path!(task))

  def load_gold_standard(task, :train) do
    held = MapSet.new(read_json_array!(held_out_path!(task)))

    task
    |> load_gold_standard(:all)
    |> Enum.reject(&MapSet.member?(held, &1))
  end

  @doc """
  Get the path to the gold standard file for a task.
  """
  def gold_standard_path(task) do
    Path.join([evaluation_base_path(), task, "gold_standard.json"])
  end

  @doc "Where the held-out split for a task lives, whether or not it exists yet."
  @spec held_out_path(String.t()) :: Path.t()
  def held_out_path(task) do
    Path.join([evaluation_base_path(), task, "held_out.json"])
  end

  @doc "True when a held-out split has been carved for this task."
  @spec held_out?(String.t()) :: boolean()
  def held_out?(task), do: task |> held_out_path() |> File.exists?()

  defp held_out_path!(task) do
    path = held_out_path(task)

    if File.exists?(path) do
      path
    else
      raise """
      No held-out split for #{inspect(task)} at #{path}

      Without one there is no training partition to separate from an evaluation
      partition, so any accuracy measured here would be measured on the data the
      model was fitted to.

      Carve one with `mix split.held_out #{task}`, or ask for :all if you really
      do want the whole corpus.
      """
    end
  end

  defp read_json_array!(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} when is_list(data) ->
            data

          {:ok, other} ->
            raise "#{path} holds a #{type_name(other)}; an array of annotated examples was expected"

          {:error, reason} ->
            raise "#{path} is not valid JSON: #{Exception.message(reason)}"
        end

      {:error, reason} ->
        raise "Cannot read #{path}: #{:file.format_error(reason)}"
    end
  end

  defp type_name(v) when is_map(v), do: "JSON object"
  defp type_name(v) when is_binary(v), do: "JSON string"
  defp type_name(v) when is_number(v), do: "JSON number"
  defp type_name(v) when is_boolean(v), do: "JSON boolean"
  defp type_name(nil), do: "JSON null"
  defp type_name(_), do: "value of an unexpected type"

  # ============================================================================
  # Atlas Storage
  # ============================================================================

  defp save_to_atlas(result) do
    Brain.AtlasIntegration.async(fn ->
      attrs = %{
        task: Map.get(result, :task, "unknown"),
        accuracy: Map.get(result, :accuracy, 0.0),
        macro_f1: Map.get(result, :macro_f1, 0.0),
        weighted_f1: Map.get(result, :weighted_f1, 0.0),
        total_examples: Map.get(result, :total_examples, 0),
        duration_ms: Map.get(result, :duration_ms),
        per_class: Map.get(result, :per_class, %{}),
        confusion_matrix: Map.get(result, :confusion_matrix, %{})
      }

      %Atlas.Schemas.EvaluationResult{}
      |> Atlas.Schemas.EvaluationResult.changeset(attrs)
      |> Atlas.Repo.insert()
    end)
  rescue
    _ -> :ok
  end

  defp latest_from_atlas(task) do
    import Ecto.Query

    case Brain.AtlasIntegration.sync(fn ->
           Atlas.Schemas.EvaluationResult
           |> where([r], r.task == ^task)
           |> order_by([r], desc: r.inserted_at)
           |> limit(1)
           |> Atlas.Repo.one()
         end) do
      {:ok, nil} ->
        nil

      {:ok, row} ->
        atlas_row_to_result(row)

      {:error, _} ->
        nil
    end
  rescue
    _ -> nil
  end

  defp list_runs_from_atlas(task) do
    import Ecto.Query

    case Brain.AtlasIntegration.sync(fn ->
           Atlas.Schemas.EvaluationResult
           |> where([r], r.task == ^task)
           |> order_by([r], desc: r.inserted_at)
           |> Atlas.Repo.all()
         end) do
      {:ok, rows} ->
        Enum.map(rows, &atlas_row_to_result/1)

      {:error, _} ->
        []
    end
  rescue
    _ -> []
  end

  defp atlas_row_to_result(row) do
    %{
      "task" => row.task,
      "accuracy" => row.accuracy,
      "macro_f1" => row.macro_f1,
      "weighted_f1" => row.weighted_f1,
      "total_examples" => row.total_examples,
      "duration_ms" => row.duration_ms,
      "per_class" => row.per_class || %{},
      "confusion_matrix" => row.confusion_matrix || %{},
      "timestamp" => DateTime.to_iso8601(row.inserted_at)
    }
  end

  # ============================================================================
  # File-based Storage (write-through)
  # ============================================================================

  defp save_to_file(result) do
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
  rescue
    e ->
      Logger.warning("EvaluationStore: failed to save to file: #{inspect(e)}")
      {:error, e}
  end

  defp list_runs_from_files(task) do
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

  defp results_path do
    Path.join(evaluation_base_path(), "results")
  end

  defp evaluation_base_path do
    Brain.priv_path("evaluation")
  end

  defp get_metric(run, :accuracy), do: run["accuracy"] || run[:accuracy]
  defp get_metric(run, :macro_f1), do: run["macro_f1"] || run[:macro_f1]
  defp get_metric(run, :weighted_f1), do: run["weighted_f1"] || run[:weighted_f1]
  defp get_metric(run, metric), do: run[to_string(metric)] || run[metric]
end
