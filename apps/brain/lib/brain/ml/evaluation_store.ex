defmodule Brain.ML.EvaluationStore do
  @moduledoc """
  Stores and retrieves ML evaluation results.

  Atlas (`atlas_evaluation_results`) is the one record of them: every
  result is read from there. Each result is also written as a JSON file in
  `priv/evaluation/results/`, a copy for reading by hand; nothing here reads
  those files back, except `import_result_files!/1`, which imports the files
  saved before Atlas held every result.

  ## Usage

      result = Evaluation.build_result("intent", predictions, actuals)
      EvaluationStore.save(result)

      EvaluationStore.latest("intent")
      EvaluationStore.trend("intent", :accuracy)
  """

  import Ecto.Query
  require Logger

  alias Atlas.Schemas.EvaluationResult

  @doc """
  Saves an evaluation result to Atlas, writes its JSON copy, and emits the
  evaluation's completion (`Brain.Telemetry.emit_evaluation_complete/2`).
  Raises when Atlas refuses the result or cannot be reached.
  """
  def save(result) when is_map(result) do
    task = Map.get(result, :task, "unknown")

    save_to_atlas!(result)
    file_result = save_to_file(result)

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
  The most recent evaluation of a task, read from Atlas; `nil` when Atlas
  holds none. Raises when Atlas cannot be read.
  """
  def latest(task) do
    EvaluationResult
    |> where([r], r.task == ^task)
    |> order_by([r], desc: r.inserted_at)
    |> limit(1)
    |> Atlas.Repo.one()
    |> case do
      nil -> nil
      row -> atlas_row_to_result(row)
    end
  end

  @doc """
  Every evaluation of a task, newest first, read from Atlas. Raises when
  Atlas cannot be read.
  """
  def list_runs(task) do
    EvaluationResult
    |> where([r], r.task == ^task)
    |> order_by([r], desc: r.inserted_at)
    |> Atlas.Repo.all()
    |> Enum.map(&atlas_row_to_result/1)
  end

  @doc """
  Imports into Atlas the evaluation results saved as JSON files in `dir`
  (default: `priv/evaluation/results/`) that Atlas does not hold, each
  inserted at the time its file records, so the history reads in the order
  the evaluations ran.

  A file is held already when Atlas holds a result of the same task with the
  same accuracy, F1 scores, example count, per-class scores and confusion
  matrix; a file is imported for each such file beyond the results Atlas
  holds, so running it again imports nothing. Every file is read before
  anything is written, and all are written in one transaction: a file that
  cannot be read, decoded or inserted raises, and nothing is imported.

  Returns `%{files: count, imported: count, held: count}`.
  """
  @spec import_result_files!(Path.t()) :: %{files: non_neg_integer(), imported: non_neg_integer(), held: non_neg_integer()}
  def import_result_files!(dir \\ results_path()) do
    unless File.dir?(dir) do
      raise "Brain.ML.EvaluationStore: no evaluation results directory at #{dir}"
    end

    files =
      dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.sort()
      |> Enum.map(&read_result_file!(Path.join(dir, &1)))

    {:ok, imported} =
      Atlas.Repo.transaction(fn ->
        held =
          files
          |> Enum.map(& &1.task)
          |> Enum.uniq()
          |> Enum.flat_map(fn task -> EvaluationResult |> where([r], r.task == ^task) |> Atlas.Repo.all() end)
          |> Enum.frequencies_by(&result_key/1)

        {to_import, _held} =
          Enum.reduce(files, {[], held}, fn file, {to_import, held} ->
            key = result_key(file)

            case Map.get(held, key, 0) do
              0 -> {[file | to_import], held}
              n -> {to_import, Map.put(held, key, n - 1)}
            end
          end)

        Enum.each(Enum.reverse(to_import), fn file ->
          %EvaluationResult{}
          |> EvaluationResult.changeset(Map.delete(file, :inserted_at))
          |> Ecto.Changeset.put_change(:inserted_at, file.inserted_at)
          |> Ecto.Changeset.put_change(:updated_at, file.inserted_at)
          |> Atlas.Repo.insert!()
        end)

        length(to_import)
      end)

    %{files: length(files), imported: imported, held: length(files) - imported}
  end

  defp read_result_file!(path) do
    data = path |> File.read!() |> Jason.decode!()

    for key <- ~w(task timestamp accuracy macro_f1 total_examples), not Map.has_key?(data, key) do
      raise "#{path} holds no #{key}"
    end

    timestamp =
      case DateTime.from_iso8601(data["timestamp"]) do
        {:ok, timestamp, 0} -> timestamp
        other -> raise "#{path}: its timestamp #{inspect(data["timestamp"])} is not an ISO 8601 UTC time (#{inspect(other)})"
      end

    %{
      task: data["task"],
      accuracy: data["accuracy"],
      macro_f1: data["macro_f1"],
      weighted_f1: data["weighted_f1"],
      total_examples: data["total_examples"],
      duration_ms: data["duration_ms"],
      per_class: data["per_class"] || %{},
      confusion_matrix: data["confusion_matrix"] || %{},
      inserted_at: timestamp
    }
  end

  # What makes two results the same evaluation run's.
  defp result_key(result) do
    {result.task, result.accuracy * 1.0, result.macro_f1 * 1.0, result.weighted_f1 && result.weighted_f1 * 1.0,
     result.total_examples, result.per_class || %{}, result.confusion_matrix || %{}}
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

  defp save_to_atlas!(result) do
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

    %EvaluationResult{}
    |> EvaluationResult.changeset(attrs)
    |> Atlas.Repo.insert!()
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
  # The JSON copy
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
