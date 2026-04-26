defmodule Mix.Tasks.Evaluate.Gate do
  @shortdoc "CI gate: fail if evaluation metrics regress beyond thresholds"
  @moduledoc """
  Compares the latest saved evaluation results against a baseline and fails
  if any metric regresses beyond configured thresholds.

  ## Thresholds

  - Intent macro-F1: regression > 2pp = fail
  - Sentiment macro-F1: regression > 2pp = fail
  - Speech-act macro-F1: regression > 1pp = fail
  - Any task: increase in unknown/not_loaded/errored > 0 from baseline = fail

  ## Usage

      mix evaluate.gate              # Check against previous run
      mix evaluate.gate --baseline   # Set current results as the baseline
  """

  use Mix.Task

  alias Brain.ML.EvaluationStore

  @thresholds %{
    "intent" => 0.02,
    "sentiment" => 0.02,
    "speech_act" => 0.01,
    "ner" => 0.03
  }

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    if "--baseline" in args do
      set_baseline()
    else
      check_gate()
    end
  end

  @gate_tasks ["intent", "sentiment", "speech_act", "ner"]

  defp set_baseline do
    baseline =
      Enum.reduce(@gate_tasks, %{}, fn task, acc ->
        case EvaluationStore.latest(task) do
          nil ->
            IO.puts("  #{task}: no results to baseline")
            acc

          result ->
            entry = %{
              "macro_f1" => result.macro_f1,
              "accuracy" => result.accuracy,
              "diagnostics" => Map.get(result, :diagnostics, %{ok: 0, unknown: 0, errored: 0})
            }

            IO.puts("  #{task}: macro_f1=#{Float.round(result.macro_f1 * 100, 1)}%")
            Map.put(acc, task, entry)
        end
      end)

    path = baseline_path()
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(baseline, pretty: true) <> "\n")
    IO.puts("\nBaseline saved to: #{path}")
  end

  defp check_gate do
    path = baseline_path()

    unless File.exists?(path) do
      Mix.raise("""
      No baseline found at #{path}.
      Run `mix evaluate.gate --baseline` to set the current results as baseline.
      """)
    end

    {:ok, json} = File.read(path)
    {:ok, baseline} = Jason.decode(json)

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("EVALUATION REGRESSION GATE")
    IO.puts(String.duplicate("=", 60) <> "\n")

    failures =
      Enum.reduce(@gate_tasks, [], fn task, failures ->
        case {Map.get(baseline, task), EvaluationStore.latest(task)} do
          {nil, _} ->
            IO.puts("  #{task}: no baseline, skipping")
            failures

          {_, nil} ->
            IO.puts("  #{task}: no current results")
            [{task, "no current evaluation results"} | failures]

          {base, current} ->
            check_task(task, base, current, failures)
        end
      end)

    IO.puts("")

    if failures == [] do
      IO.puts("GATE PASSED: All metrics within thresholds.\n")
    else
      IO.puts("GATE FAILED:\n")

      Enum.each(failures, fn {task, reason} ->
        IO.puts("  [FAIL] #{task}: #{reason}")
      end)

      IO.puts("")
      exit({:shutdown, 1})
    end
  end

  defp check_task(task, base, current, failures) do
    threshold = Map.get(@thresholds, task, 0.02)
    base_f1 = base["macro_f1"]
    current_f1 = current.macro_f1
    delta = current_f1 - base_f1

    status = if delta >= -threshold, do: "PASS", else: "FAIL"

    IO.puts(
      "  #{task}: macro_f1 #{Float.round(base_f1 * 100, 1)}% -> #{Float.round(current_f1 * 100, 1)}% " <>
        "(delta: #{if delta >= 0, do: "+"}#{Float.round(delta * 100, 1)}pp, threshold: #{Float.round(threshold * 100, 1)}pp) [#{status}]"
    )

    failures =
      if delta < -threshold do
        [{task, "macro_f1 regressed #{Float.round(abs(delta) * 100, 1)}pp (threshold: #{Float.round(threshold * 100, 1)}pp)"} | failures]
      else
        failures
      end

    base_diag = base["diagnostics"] || %{}
    current_diag = Map.get(current, :diagnostics, %{})

    current_unknown = Map.get(current_diag, :unknown, 0) + Map.get(current_diag, "unknown", 0)
    current_errored = Map.get(current_diag, :errored, 0) + Map.get(current_diag, "errored", 0)
    current_not_loaded = Map.get(current_diag, :not_loaded, 0) + Map.get(current_diag, "not_loaded", 0)
    base_unknown = Map.get(base_diag, "unknown", 0)
    base_errored = Map.get(base_diag, "errored", 0)

    new_errors = (current_unknown + current_errored + current_not_loaded) - (base_unknown + base_errored)

    if new_errors > 0 do
      IO.puts("    error canary: +#{new_errors} new unknown/errored/not_loaded predictions")
      [{task, "#{new_errors} new unknown/errored/not_loaded predictions"} | failures]
    else
      failures
    end
  end

  defp baseline_path do
    case :code.priv_dir(:brain) do
      {:error, _} -> "apps/brain/priv/evaluation/baseline.json"
      priv -> Path.join(priv, "evaluation/baseline.json")
    end
  end
end
