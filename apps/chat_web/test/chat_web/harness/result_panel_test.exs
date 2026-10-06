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

      %{intent: "greeting"}
    end
  end

  describe "result_panel/1 for a call that returned" do
    test "shows what was called, that it returned, and how long it took" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html =
        render_component(&Runner.result_panel/1,
          outcome: outcome,
          label: "SpeechActClassifier.classify/1"
        )

      assert html =~ "SpeechActClassifier.classify/1"
      assert html =~ "returned"
      assert html =~ "Raw term"
    end

    test "renders the returned term" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "intent"
      assert html =~ "greeting"
    end
  end

  describe "result_panel/1 for a call that raised" do
    test "the exception is the headline, with its stacktrace" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "The subsystem raised"
      assert html =~ "ArgumentError"
      assert html =~ "the tagger was never loaded"
      assert html =~ "fail"
    end

    test "says plainly that this is a result and not a missing one" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "This is the verification result, not a missing one"
    end

    test "no raw-term panel is rendered, since there is no value" do
      outcome = Runner.run(&Subject.fail/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      refute html =~ "Raw term"
    end
  end

  describe "result_panel/1 provenance" do
    test "an uninstrumented call says so rather than rendering an empty table" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "Nothing on this path is instrumented yet"
      assert html =~ "a gap, not an all-clear"
    end

    test "recorded values render with origin, reason and the function responsible" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "entity.familiarity"
      assert html =~ "your input"
      assert html =~ "config.domain_lemmas"
      assert html =~ "fallback default"
      assert html =~ "entity_types.json declares no domain_lemmas"
      assert html =~ "TypeHierarchy.config/2"
    end

    test "an unreadable source renders as its own origin, not as a fallback" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "source unreadable"
    end

    test "the stand-in count is in the header, counting grouped rows" do
      outcome = Runner.run(&Subject.with_stand_ins/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "2 not from your input"
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

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      assert html =~ "The subsystem raised"
      assert html =~ "config.x"
    end
  end

  describe "result_panel/1 with a comparison" do
    test "renders the verdict against the saved expectation" do
      outcome = Runner.run(&Subject.succeed/1, "hello")
      comparison = Comparison.compare(%{intent: "farewell"}, outcome.value)

      html =
        render_component(&Runner.result_panel/1,
          outcome: outcome,
          label: "x",
          comparison: comparison
        )

      assert html =~ "Against the saved expectation"
      assert html =~ "Fail"
      assert html =~ "intent"
    end

    test "without a comparison that section is absent" do
      outcome = Runner.run(&Subject.succeed/1, "hello")

      html = render_component(&Runner.result_panel/1, outcome: outcome, label: "x")

      refute html =~ "Against the saved expectation"
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
          last_run_at: nil
        },
        attrs
      )
    end

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
          cases: [a_case(%{status: "pass", last_run_at: ~U[2026-10-06 01:02:03.000000Z]})]
        )

      assert html =~ "Pass"
      assert html =~ "2026-10-06"
    end

    test "every status renders" do
      cases =
        for {status, index} <- Enum.with_index(["pending", "pass", "fail", "error"]) do
          a_case(%{status: status, name: "case #{index}"})
        end

      html = render_component(&Runner.case_list/1, cases: cases)

      for label <- ["Not run", "Pass", "Fail", "Raised"], do: assert(html =~ label)
    end
  end
end
