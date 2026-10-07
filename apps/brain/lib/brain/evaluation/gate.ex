defmodule Brain.Evaluation.Gate do
  @moduledoc """
  The evaluation regression gate: the saved baseline and the per-task verdict
  against it.

  This is the one implementation. `mix evaluate.gate` reports these verdicts
  and fails the run on any failure; pages show the same verdicts through
  `ChatWeb.UI.macro_f1_gate/1`. Neither computes a second copy.

  ## What the gate judges

  The gate is not an accuracy target. It judges each task's latest saved
  evaluation (`Brain.ML.EvaluationStore.latest/1`) against a saved baseline:

    * macro-F1 may fall at most the task's allowance below the baseline:
      2 points for intent and sentiment, 1 for speech act, 3 for NER;
    * the count of unknown, errored and not-loaded predictions may not rise
      above the baseline's (the error canary);
    * a task with a baseline but no current result fails, because there is
      nothing to show it did not regress.

  The error canary needs `"diagnostics"` on both the baseline entry and the
  latest result. When either has none, the canary is `:not_measured`, its
  `new_errors` is nil and `diagnostics_absent` names the side or sides that
  had none: an absent count is not a count of zero. Macro-F1 is still judged.
  A verdict that judged nothing (no baseline entry, or no current result) has
  canary `:not_judged`.

  A task with no baseline entry, or no baseline file at all, has not been
  judged: its status is `:not_set`, never a pass.

  ## The baseline

  `mix evaluate.gate --baseline` writes the latest results to
  `baseline_path/0` as JSON: per task, `"macro_f1"`, `"accuracy"` and
  `"diagnostics"`, built by `baseline_entry/1`.
  """

  alias Brain.ML.EvaluationStore

  @tasks ["intent", "sentiment", "speech_act", "ner"]

  @allowances %{
    "intent" => 0.02,
    "sentiment" => 0.02,
    "speech_act" => 0.01,
    "ner" => 0.03
  }

  @type baseline :: %{optional(String.t()) => map()}

  @type verdict :: %{
          task: String.t(),
          status: :pass | :fail | :not_set,
          allowance: float(),
          baseline_macro_f1: float() | nil,
          current_macro_f1: float() | nil,
          delta: float() | nil,
          canary: :measured | :not_measured | :not_judged,
          new_errors: integer() | nil,
          diagnostics_absent: [:baseline | :current],
          failures: [String.t()]
        }

  @doc "The tasks the gate judges, in report order."
  @spec tasks() :: [String.t()]
  def tasks, do: @tasks

  @doc """
  How far macro-F1 may fall below the baseline for `task`, as a fraction of 1
  (0.02 is 2 points). Raises for a task the gate does not judge.
  """
  @spec allowance(String.t()) :: float()
  def allowance(task) do
    case Map.fetch(@allowances, task) do
      {:ok, allowance} ->
        allowance

      :error ->
        raise ArgumentError,
              "Brain.Evaluation.Gate: the gate does not judge task #{inspect(task)}. " <>
                "Its tasks are #{inspect(@tasks)}."
    end
  end

  @doc "Where the baseline is saved."
  @spec baseline_path() :: Path.t()
  def baseline_path, do: Brain.priv_path("evaluation/baseline.json")

  @doc """
  Reads the baseline at `path`: `{:ok, baseline}`, or `:not_set` when no
  baseline file exists.

  A file that exists but cannot be read, is not valid JSON, or is not a JSON
  object raises, because a gate that silently treats a broken baseline as
  unset would report "gate not set" over a baseline someone saved.
  """
  @spec read_baseline(Path.t()) :: {:ok, baseline()} | :not_set
  def read_baseline(path \\ baseline_path()) do
    case File.read(path) do
      {:ok, json} ->
        case Jason.decode!(json) do
          baseline when is_map(baseline) ->
            {:ok, baseline}

          other ->
            raise ArgumentError,
                  "Brain.Evaluation.Gate: the baseline at #{path} is not a JSON object, got #{inspect(other)}."
        end

      {:error, :enoent} ->
        :not_set

      {:error, reason} ->
        raise File.Error, reason: reason, action: "read the gate baseline", path: path
    end
  end

  @doc """
  The baseline entry for one saved evaluation result, as
  `Brain.ML.EvaluationStore.latest/1` returns it (string keys).
  """
  @spec baseline_entry(map()) :: map()
  def baseline_entry(result) when is_map(result) do
    %{
      "macro_f1" => macro_f1!(result, "the result being saved as baseline"),
      "accuracy" => Map.get(result, "accuracy"),
      "diagnostics" => Map.get(result, "diagnostics")
    }
  end

  @doc """
  Reads the baseline and judges `task`. The latest result is fetched only when
  the baseline has an entry for the task, since without one nothing is judged.
  """
  @spec check(String.t()) :: verdict()
  def check(task), do: check(task, read_baseline())

  @doc "Judges `task` against an already-read baseline (see `read_baseline/1`)."
  @spec check(String.t(), {:ok, baseline()} | :not_set) :: verdict()
  def check(task, {:ok, baseline}) when is_map(baseline) do
    if Map.has_key?(baseline, task) do
      verdict(task, {:ok, baseline}, EvaluationStore.latest(task))
    else
      verdict(task, {:ok, baseline}, nil)
    end
  end

  def check(task, :not_set), do: verdict(task, :not_set, nil)

  @doc """
  The gate's verdict on one task.

  `baseline` is what `read_baseline/1` returns; `current` the task's latest
  saved result (string keys, as `Brain.ML.EvaluationStore.latest/1` returns
  it), or nil when there is none.

  `delta` is current minus baseline macro-F1, as a fraction of 1. `failures`
  says, in words, every reason the task failed.
  """
  @spec verdict(String.t(), {:ok, baseline()} | :not_set, map() | nil) :: verdict()
  def verdict(task, baseline, current) do
    allowance = allowance(task)

    base =
      case baseline do
        {:ok, map} when is_map(map) -> Map.get(map, task)
        :not_set -> nil
      end

    judge(task, allowance, base, current)
  end

  defp judge(task, allowance, nil, _current) do
    %{
      task: task,
      status: :not_set,
      allowance: allowance,
      baseline_macro_f1: nil,
      current_macro_f1: nil,
      delta: nil,
      canary: :not_judged,
      new_errors: nil,
      diagnostics_absent: [],
      failures: []
    }
  end

  defp judge(task, allowance, base, nil) do
    %{
      task: task,
      status: :fail,
      allowance: allowance,
      baseline_macro_f1: macro_f1!(base, "the #{task} baseline"),
      current_macro_f1: nil,
      delta: nil,
      canary: :not_judged,
      new_errors: nil,
      diagnostics_absent: [],
      failures: ["no current evaluation results"]
    }
  end

  defp judge(task, allowance, base, current) do
    base_f1 = macro_f1!(base, "the #{task} baseline")
    current_f1 = macro_f1!(current, "the latest #{task} result")
    delta = current_f1 - base_f1

    {canary, new_errors, diagnostics_absent} =
      canary(Map.get(base, "diagnostics"), Map.get(current, "diagnostics"))

    failures =
      Enum.reject(
        [
          delta < -allowance &&
            "macro_f1 regressed #{points(abs(delta))}pp (allowance: #{points(allowance)}pp)",
          canary == :measured and new_errors > 0 &&
            "#{new_errors} new unknown/errored/not_loaded predictions"
        ],
        &(&1 == false)
      )

    %{
      task: task,
      status: if(failures == [], do: :pass, else: :fail),
      allowance: allowance,
      baseline_macro_f1: base_f1,
      current_macro_f1: current_f1,
      delta: delta,
      canary: canary,
      new_errors: new_errors,
      diagnostics_absent: diagnostics_absent,
      failures: failures
    }
  end

  # The error canary: new unknown, errored and not-loaded predictions, current
  # minus baseline. It is measured only when both sides saved diagnostics;
  # otherwise it names the sides that had none.
  defp canary(base_diagnostics, current_diagnostics) do
    absent =
      [baseline: base_diagnostics, current: current_diagnostics]
      |> Enum.filter(fn {_side, diagnostics} -> is_nil(diagnostics) end)
      |> Enum.map(fn {side, _} -> side end)

    case absent do
      [] ->
        {:measured, prediction_failures(current_diagnostics) - prediction_failures(base_diagnostics), []}

      sides ->
        {:not_measured, nil, sides}
    end
  end

  defp macro_f1!(map, what) do
    case Map.get(map, "macro_f1") do
      value when is_number(value) ->
        value

      other ->
        raise ArgumentError,
              "Brain.Evaluation.Gate: #{what} has no numeric \"macro_f1\", got #{inspect(other)}. " <>
                "The gate cannot judge a result without its macro-F1."
    end
  end

  # Unknown, errored and not-loaded predictions in a result's diagnostics
  # counts. Diagnostics that are not a map raise.
  defp prediction_failures(diagnostics) when is_map(diagnostics) do
    ["unknown", "errored", "not_loaded"]
    |> Enum.map(fn key -> Map.get(diagnostics, key, 0) end)
    |> Enum.sum()
  end

  defp points(fraction), do: Float.round(fraction * 100.0, 1)
end
