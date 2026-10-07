defmodule Brain.Evaluation.GateTest do
  @moduledoc """
  The regression gate's verdicts: what a present baseline, an absent one, a
  regression within and beyond the allowance, new failed predictions, and an
  error canary that cannot be measured each produce. `mix evaluate.gate` and the pages report these same verdicts.
  """
  use ExUnit.Case, async: true

  alias Brain.Evaluation.Gate

  @moduletag :tmp_dir

  defp baseline(entries), do: {:ok, entries}

  defp entry(macro_f1, diagnostics \\ nil) do
    %{"macro_f1" => macro_f1, "accuracy" => 0.6, "diagnostics" => diagnostics}
  end

  describe "allowance/1" do
    test "each task's allowance, as a fraction of 1" do
      assert Gate.allowance("intent") == 0.02
      assert Gate.allowance("sentiment") == 0.02
      assert Gate.allowance("speech_act") == 0.01
      assert Gate.allowance("ner") == 0.03
    end

    test "a task the gate does not judge raises" do
      assert_raise ArgumentError, ~r/does not judge task "pos"/, fn -> Gate.allowance("pos") end
    end

    test "every gate task has an allowance" do
      for task <- Gate.tasks(), do: assert(is_float(Gate.allowance(task)))
    end
  end

  describe "verdict/3 with no baseline" do
    test "no baseline file is not set, never a pass" do
      verdict = Gate.verdict("intent", :not_set, %{"macro_f1" => 0.9})

      assert verdict.status == :not_set
      assert verdict.delta == nil
      assert verdict.failures == []
      assert verdict.allowance == 0.02
      assert verdict.canary == :not_judged
      assert verdict.new_errors == nil
    end

    test "a baseline with no entry for the task is not set for that task" do
      verdict = Gate.verdict("ner", baseline(%{"intent" => entry(0.5)}), %{"macro_f1" => 0.4})

      assert verdict.status == :not_set
    end
  end

  describe "verdict/3 with a baseline" do
    test "a fall within the allowance passes, with the change and the allowance" do
      verdict = Gate.verdict("intent", baseline(%{"intent" => entry(0.500)}), %{"macro_f1" => 0.496})

      assert verdict.status == :pass
      assert_in_delta verdict.delta, -0.004, 1.0e-9
      assert verdict.baseline_macro_f1 == 0.5
      assert verdict.current_macro_f1 == 0.496
      assert verdict.allowance == 0.02
      assert verdict.failures == []
    end

    test "a gain passes" do
      verdict = Gate.verdict("speech_act", baseline(%{"speech_act" => entry(0.3)}), %{"macro_f1" => 0.35})

      assert verdict.status == :pass
      assert verdict.delta > 0
    end

    test "a regression beyond the allowance fails and says by how much" do
      verdict = Gate.verdict("speech_act", baseline(%{"speech_act" => entry(0.31)}), %{"macro_f1" => 0.29})

      assert verdict.status == :fail
      assert_in_delta verdict.delta, -0.02, 1.0e-9
      assert verdict.failures == ["macro_f1 regressed 2.0pp (allowance: 1.0pp)"]
    end

    test "the same fall passes a task with a wider allowance" do
      verdict = Gate.verdict("ner", baseline(%{"ner" => entry(0.31)}), %{"macro_f1" => 0.29})

      assert verdict.status == :pass
    end

    test "new unknown, errored or not-loaded predictions fail even when macro-F1 holds" do
      base = baseline(%{"intent" => entry(0.5, %{"ok" => 90, "unknown" => 1, "errored" => 0})})
      current = %{"macro_f1" => 0.5, "diagnostics" => %{"ok" => 87, "unknown" => 2, "errored" => 1, "not_loaded" => 1}}

      verdict = Gate.verdict("intent", base, current)

      assert verdict.status == :fail
      assert verdict.new_errors == 3
      assert verdict.failures == ["3 new unknown/errored/not_loaded predictions"]
    end

    test "with diagnostics on both sides the canary is measured" do
      base = baseline(%{"intent" => entry(0.5, %{"ok" => 9, "errored" => 1})})
      current = %{"macro_f1" => 0.5, "diagnostics" => %{"ok" => 10, "errored" => 0}}

      verdict = Gate.verdict("intent", base, current)

      assert verdict.status == :pass
      assert verdict.canary == :measured
      assert verdict.new_errors == -1
      assert verdict.diagnostics_absent == []
    end

    test "a result with no diagnostics leaves the canary not measured, never zero" do
      base = baseline(%{"intent" => entry(0.5, %{"unknown" => 0})})

      verdict = Gate.verdict("intent", base, %{"macro_f1" => 0.5})

      assert verdict.canary == :not_measured
      assert verdict.new_errors == nil
      assert verdict.diagnostics_absent == [:current]
      assert verdict.status == :pass
      assert verdict.failures == []
    end

    test "a baseline with no diagnostics leaves the canary not measured" do
      base = baseline(%{"intent" => entry(0.5)})
      current = %{"macro_f1" => 0.5, "diagnostics" => %{"unknown" => 4}}

      verdict = Gate.verdict("intent", base, current)

      assert verdict.canary == :not_measured
      assert verdict.new_errors == nil
      assert verdict.diagnostics_absent == [:baseline]
    end

    test "neither side with diagnostics names both" do
      verdict = Gate.verdict("intent", baseline(%{"intent" => entry(0.5)}), %{"macro_f1" => 0.5})

      assert verdict.diagnostics_absent == [:baseline, :current]
    end

    test "an unmeasured canary still judges macro-F1" do
      verdict = Gate.verdict("intent", baseline(%{"intent" => entry(0.5)}), %{"macro_f1" => 0.4})

      assert verdict.status == :fail
      assert verdict.canary == :not_measured
      assert verdict.failures == ["macro_f1 regressed 10.0pp (allowance: 2.0pp)"]
    end

    test "diagnostics that are not a map raise" do
      base = baseline(%{"intent" => entry(0.5, %{"unknown" => 0})})

      assert_raise FunctionClauseError, fn ->
        Gate.verdict("intent", base, %{"macro_f1" => 0.5, "diagnostics" => [1, 2]})
      end
    end

    test "both failures are reported together" do
      base = baseline(%{"intent" => entry(0.5, %{"unknown" => 0})})
      current = %{"macro_f1" => 0.4, "diagnostics" => %{"unknown" => 2}}

      assert length(Gate.verdict("intent", base, current).failures) == 2
    end

    test "a baseline with no current result fails" do
      verdict = Gate.verdict("intent", baseline(%{"intent" => entry(0.5)}), nil)

      assert verdict.status == :fail
      assert verdict.failures == ["no current evaluation results"]
      assert verdict.delta == nil
      assert verdict.canary == :not_judged
      assert verdict.new_errors == nil
    end

    test "a result with no macro-F1 raises rather than judging it" do
      assert_raise ArgumentError, ~r/the latest intent result has no numeric "macro_f1"/, fn ->
        Gate.verdict("intent", baseline(%{"intent" => entry(0.5)}), %{"accuracy" => 0.7})
      end

      assert_raise ArgumentError, ~r/the intent baseline has no numeric "macro_f1"/, fn ->
        Gate.verdict("intent", baseline(%{"intent" => %{"accuracy" => 0.7}}), %{"macro_f1" => 0.5})
      end
    end
  end

  describe "read_baseline/1" do
    test "a missing file is not set", %{tmp_dir: dir} do
      assert Gate.read_baseline(Path.join(dir, "baseline.json")) == :not_set
    end

    test "a saved baseline is read as written", %{tmp_dir: dir} do
      path = Path.join(dir, "baseline.json")
      saved = %{"intent" => entry(0.5, %{"ok" => 9, "unknown" => 1, "errored" => 0})}
      File.write!(path, Jason.encode!(saved))

      assert Gate.read_baseline(path) == {:ok, saved}
    end

    test "a file that is not valid JSON raises", %{tmp_dir: dir} do
      path = Path.join(dir, "baseline.json")
      File.write!(path, "{not json")

      assert_raise Jason.DecodeError, fn -> Gate.read_baseline(path) end
    end

    test "a file that is not a JSON object raises", %{tmp_dir: dir} do
      path = Path.join(dir, "baseline.json")
      File.write!(path, "[1, 2]")

      assert_raise ArgumentError, ~r/is not a JSON object/, fn -> Gate.read_baseline(path) end
    end
  end

  describe "baseline_entry/1" do
    test "takes macro-F1, accuracy and diagnostics from a stored result" do
      result = %{"macro_f1" => 0.4, "accuracy" => 0.7, "diagnostics" => %{"unknown" => 2}, "per_class" => %{}}

      assert Gate.baseline_entry(result) == %{
               "macro_f1" => 0.4,
               "accuracy" => 0.7,
               "diagnostics" => %{"unknown" => 2}
             }
    end

    test "a result with no macro-F1 cannot be a baseline" do
      assert_raise ArgumentError, ~r/no numeric "macro_f1"/, fn -> Gate.baseline_entry(%{"accuracy" => 0.7}) end
    end
  end

  describe "check/2" do
    test "with no baseline it judges nothing and reads no result" do
      for task <- Gate.tasks() do
        assert Gate.check(task, :not_set).status == :not_set
      end
    end

    test "a baseline without the task is not set for it" do
      assert Gate.check("sentiment", {:ok, %{"intent" => entry(0.5)}}).status == :not_set
    end
  end

  test "the baseline lives in brain's priv evaluation directory" do
    assert Gate.baseline_path() == Brain.priv_path("evaluation/baseline.json")
  end
end
