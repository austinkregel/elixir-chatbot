defmodule Mix.Tasks.Evaluate.Gate do
  @shortdoc "CI gate: fail if evaluation metrics regress beyond thresholds"
  @moduledoc """
  Compares the latest saved evaluation results against a baseline and fails
  if any metric regresses beyond the gate's allowances.

  The baseline and the verdicts are `Brain.Evaluation.Gate`'s; this task only
  reports them and sets the exit status, so the pages that show the gate reach
  the same answer.

  ## Allowances

  - Intent macro-F1: regression > 2pp = fail
  - Sentiment macro-F1: regression > 2pp = fail
  - Speech-act macro-F1: regression > 1pp = fail
  - NER macro-F1: regression > 3pp = fail
  - Any task: increase in unknown/not_loaded/errored > 0 from baseline = fail

  ## Usage

      mix evaluate.gate              # Check against the saved baseline
      mix evaluate.gate --baseline   # Set current results as the baseline
  """

  use Mix.Task

  alias Brain.Evaluation.Gate
  alias Brain.ML.EvaluationStore

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    if "--baseline" in args do
      set_baseline()
    else
      check_gate()
    end
  end

  defp set_baseline do
    baseline =
      Enum.reduce(Gate.tasks(), %{}, fn task, acc ->
        case EvaluationStore.latest(task) do
          nil ->
            IO.puts("  #{task}: no results to baseline")
            acc

          result ->
            entry = Gate.baseline_entry(result)
            IO.puts("  #{task}: macro_f1=#{percent(entry["macro_f1"])}%")
            Map.put(acc, task, entry)
        end
      end)

    path = Gate.baseline_path()
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(baseline, pretty: true) <> "\n")
    IO.puts("\nBaseline saved to: #{path}")
  end

  defp check_gate do
    path = Gate.baseline_path()

    baseline =
      case Gate.read_baseline(path) do
        {:ok, _} = baseline ->
          baseline

        :not_set ->
          Mix.raise("""
          No baseline found at #{path}.
          Run `mix evaluate.gate --baseline` to set the current results as baseline.
          """)
      end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("EVALUATION REGRESSION GATE")
    IO.puts(String.duplicate("=", 60) <> "\n")

    verdicts = Enum.map(Gate.tasks(), &Gate.check(&1, baseline))
    Enum.each(verdicts, &report/1)

    IO.puts("")

    failures = Enum.filter(verdicts, &(&1.status == :fail))

    if failures == [] do
      IO.puts("GATE PASSED: All metrics within thresholds.\n")
    else
      IO.puts("GATE FAILED:\n")

      for verdict <- failures, reason <- verdict.failures do
        IO.puts("  [FAIL] #{verdict.task}: #{reason}")
      end

      IO.puts("")
      exit({:shutdown, 1})
    end
  end

  defp report(%{status: :not_set, task: task}) do
    IO.puts("  #{task}: no baseline, skipping")
  end

  defp report(%{current_macro_f1: nil, task: task}) do
    IO.puts("  #{task}: no current results")
  end

  defp report(verdict) do
    sign = if verdict.delta >= 0, do: "+", else: ""

    IO.puts(
      "  #{verdict.task}: macro_f1 #{percent(verdict.baseline_macro_f1)}% -> #{percent(verdict.current_macro_f1)}% " <>
        "(delta: #{sign}#{percent(verdict.delta)}pp, threshold: #{percent(verdict.allowance)}pp) " <>
        "[#{verdict.status |> Atom.to_string() |> String.upcase()}]"
    )

    case verdict.canary do
      :measured ->
        if verdict.new_errors > 0 do
          IO.puts("    error canary: +#{verdict.new_errors} new unknown/errored/not_loaded predictions")
        end

      :not_measured ->
        IO.puts("    error canary not measured (no diagnostics in #{diagnostics_absent_words(verdict.diagnostics_absent)})")
    end
  end

  defp diagnostics_absent_words(sides) do
    case Enum.sort(sides) do
      [:baseline] -> "the baseline"
      [:current] -> "the latest result"
      [:baseline, :current] -> "the baseline or the latest result"
      other -> Mix.raise("evaluate.gate: an unmeasured canary names :baseline, :current or both; got #{inspect(other)}")
    end
  end

  defp percent(fraction), do: Float.round(fraction * 100.0, 1)
end
