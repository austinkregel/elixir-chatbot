defmodule ChatWeb.Harness.ResultPanelTest do
  @moduledoc """
  `Runner.result_panel/1` and `Runner.case_list/1`, rendered.

  These are the two components an eventual page is built out of, and until a
  page exists nothing else renders them. They are fed real values throughout —
  an actual `Runner.run/2` outcome, an actual
  `Atlas.Verification.Comparison.compare/3` result, an actual
  `Atlas.Schemas.VerificationCase` struct — because the shapes are the point.
  """
  use ExUnit.Case, async: true

  import Phoenix.LiveViewTest

  alias Atlas.Schemas.VerificationCase
  alias Atlas.Verification.Comparison
  alias Brain.Provenance
  alias ChatWeb.Harness.Runner

  defmodule Subject do
    @moduledoc false

    alias Brain.Provenance

    def succeed(input), do: %{intent: "greeting", text: input}

    def fail(_input), do: raise(ArgumentError, "the tagger was never loaded")

    def with_stand_ins(_input) do
      Provenance.record(["entity", "familiarity"], 0.76, :computed, source: "Subject.compute/1")

      Provenance.record(["config", "domain_lemmas"], %{}, :default,
        source: "TypeHierarchy.config/2",
        meta: %{"reason" => "entity_types.json declares no domain_lemmas"}
      )

      Provenance.record(["config", "gazetteer"], nil, :unavailable,
        source: "TypeHierarchy.config/2",
        meta: %{"reason" => "the TypeHierarchy ETS table could not be read"}
      )

      %{intent: "greeting", config: %{gazetteer: nil, domain_lemmas: %{}}}
    end
  end

  defp panel(outcome, attrs \\ []) do
    render_component(
      &Runner.result_panel/1,
      Map.merge(
        %{
          outcome: outcome,
          label: "x",
          world_id: "smart-home",
          expected_sources: ["Subject", "TypeHierarchy"]
        },
        Map.new(attrs)
      )
    )
  end

  describe "result_panel/1 for a call that returned" do
    test "shows what was called, that it returned, and how long it took" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = panel(outcome, label: "SpeechActClassifier.classify/1")

      assert html =~ "SpeechActClassifier.classify/1"
      assert html =~ "returned"
      assert html =~ "Raw term"
    end

    test "shows the world the run used" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = panel(outcome)

      assert html =~ ~s(data-world="smart-home")
      assert html =~ "world smart-home"
    end

    test "does not show an applied language, since nothing in the app provides one" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      refute panel(outcome) =~ "language"
    end

    test "renders the returned term" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = panel(outcome)

      assert html =~ "intent"
      assert html =~ "greeting"
    end
  end

  describe "result_panel/1 for a call that raised" do
    test "the exception is the headline, with its stacktrace" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      html = panel(outcome)

      assert html =~ "The subsystem raised"
      assert html =~ "ArgumentError"
      assert html =~ "the tagger was never loaded"
      assert html =~ "fail"
    end

    test "says plainly that this is a result and not a missing one" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      assert panel(outcome) =~ "This is the verification result, not a missing one"
    end

    test "no raw-term panel is rendered, since there is no value" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      refute panel(outcome) =~ "Raw term"
    end

    test "the world is shown for a raised call too" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      assert panel(outcome) =~ "world smart-home"
    end
  end

  describe "result_panel/1 provenance" do
    test "an empty trace says which modules recorded nothing, not a blank" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = panel(outcome, expected_sources: ["Brain.Analysis.TypeHierarchy"])

      assert html =~ ~s(data-empty="not_instrumented")
      assert html =~ "Not instrumented"
      assert html =~ "Brain.Analysis.TypeHierarchy"
      assert html =~ "a gap, not an all-clear"
    end

    test "expected sources that recorded nothing are named under a trace that has rows" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = panel(outcome, expected_sources: ["TypeHierarchy", "Brain.ML.Tokenizer"])

      assert html =~ "config.domain_lemmas"
      assert html =~ ~s(data-empty="not_instrumented")
      assert html =~ "Brain.ML.Tokenizer"
    end

    test "when every expected source reported, there is no empty state" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      refute panel(outcome, expected_sources: ["TypeHierarchy", "Subject"]) =~ "data-empty"
    end

    test "naming no expected source raises" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      assert_raise ArgumentError, ~r/must name at least one/, fn -> panel(outcome, expected_sources: []) end
    end

    test "recorded values render with origin, reason and the function responsible" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = panel(outcome)

      assert html =~ "entity.familiarity"
      assert html =~ "your input"
      assert html =~ "config.domain_lemmas"
      assert html =~ "fallback default"
      assert html =~ "entity_types.json declares no domain_lemmas"
      assert html =~ "TypeHierarchy.config/2"
    end

    test "an unreadable source renders as its own origin, not as a fallback" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      assert panel(outcome) =~ "source unreadable"
    end

    test "the raw term marks each stand-in by its own origin" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = panel(outcome)

      assert html =~ ~s(data-origin="unavailable")
      assert html =~ ~s(data-origin="default")
    end

    test "the stand-in count is in the header, counting grouped rows" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      assert panel(outcome) =~ "2 not from your input"
    end

    test "provenance recorded before a raise is still rendered" do
      outcome =
        Runner.run(
          fn _ ->
            Provenance.record(["config", "x"], nil, :unavailable, source: "T.config/2")
            raise "boom"
          end,
          nil
        )

      html = panel(outcome, expected_sources: ["T."])

      assert html =~ "The subsystem raised"
      assert html =~ "config.x"
    end
  end

  describe "result_panel/1 with a comparison" do
    test "renders the verdict against the saved expectation" do
      outcome = Runner.run(&Subject.succeed/1, "hello")
      comparison = Comparison.compare(%{intent: "farewell"}, outcome.value)

      html = panel(outcome, comparison: comparison)

      assert html =~ "Against the saved expectation"
      assert html =~ "Fail"
      assert html =~ "intent"
    end

    test "without a comparison that section is absent" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      refute panel(outcome) =~ "Against the saved expectation"
    end
  end

  describe "case_list/1" do
    defp a_case(attrs) do
      struct!(
        %VerificationCase{
          subsystem: "speech_act",
          name: "a plain greeting",
          world_id: "default",
          status: "pending",
          expected: %{"intent" => "greeting"},
          last_actual: nil,
          last_run_at: nil
        },
        attrs
      )
    end

    defp actual, do: Comparison.normalize(%{intent: "greeting", score: 0.9, text: "hi"})

    test "an empty list says nothing has been verified by hand" do
      html = render_component(&Runner.case_list/1, cases: [])

      assert html =~ "No saved cases for this subsystem yet"
      assert html =~ "Nothing here has been verified by hand"
    end

    test "a case that has never run says so rather than showing nothing" do
      html = render_component(&Runner.case_list/1, cases: [a_case(%{})])

      assert html =~ "a plain greeting"
      assert html =~ "Not run"
      assert html =~ "never run"
    end

    test "a case that has run shows its verdict and when" do
      html =
        render_component(&Runner.case_list/1,
          cases: [
            a_case(%{status: "pass", last_actual: actual(), last_run_at: ~U[2026-10-06 01:02:03.000000Z]})
          ]
        )

      assert html =~ "Pass"
      assert html =~ "2026-10-06"
    end

    test "a pass shows its coverage: asserted of returned, and what went unchecked" do
      html =
        render_component(&Runner.case_list/1,
          cases: [a_case(%{status: "pass", last_actual: actual(), last_run_at: ~U[2026-10-06 01:02:03.000000Z]})]
        )

      assert html =~ "data-coverage"
      assert html =~ "of 3"
      assert html =~ "2 unchecked"
    end

    test "every verdict carries coverage" do
      cases =
        for {status, index} <- Enum.with_index(["pending", "pass", "fail", "error"]) do
          last_actual =
            case status do
              "pending" -> nil
              "error" -> %{"__error__" => true, "kind" => "RuntimeError", "message" => "boom", "stacktrace" => []}
              _ -> actual()
            end

          a_case(%{status: status, name: "case #{index}", last_actual: last_actual})
        end

      html = render_component(&Runner.case_list/1, cases: cases)

      for label <- ["Not run", "Pass", "Fail", "Raised"], do: assert(html =~ label)
      assert length(Regex.scan(~r/data-coverage/, html)) == 4
    end

    test "a case that raised counts only what it asserts, since there is no output" do
      error = %{"__error__" => true, "kind" => "RuntimeError", "message" => "boom", "stacktrace" => []}

      html = render_component(&Runner.case_list/1, cases: [a_case(%{status: "error", last_actual: error})])

      assert html =~ "values asserted"
      refute html =~ "unchecked"
    end

    test "the world stamp is a mono badge" do
      html = render_component(&Runner.case_list/1, cases: [a_case(%{})])

      assert html =~ ~r/class="[^"]*text-ref[^"]*">\s*world default/
    end

    test "a pass with no stored output raises" do
      assert_raise ArgumentError, ~r/no stored output/, fn ->
        render_component(&Runner.case_list/1, cases: [a_case(%{status: "pass"})])
      end
    end

    test "a case that asserts nothing raises" do
      assert_raise ArgumentError, ~r/asserts nothing/, fn ->
        render_component(&Runner.case_list/1, cases: [a_case(%{expected: %{}})])
      end
    end
  end
end
