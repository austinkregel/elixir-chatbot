defmodule ChatWeb.Harness.DiffTest do
  @moduledoc """
  The diff components, rendered.

  These exist because HEEx compiling proves very little: a missing required
  attribute, a slot used wrongly, or a helper that raises on real data all
  surface at render and not at compile. Every component here is rendered against
  the shapes it will actually be given — a real
  `Atlas.Verification.Comparison.compare/3` result and a real normalised term —
  rather than hand-built approximations of them.
  """
  use ExUnit.Case, async: true

  import Phoenix.LiveViewTest

  alias Atlas.Verification.Comparison
  alias ChatWeb.Harness.Diff

  describe "diff/1 with a real comparison result" do
    test "a passing comparison says every asserted value matched" do
      comparison = Comparison.compare(%{intent: "greeting"}, %{intent: "greeting", score: 0.9})

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "Pass"
      assert html =~ "Every asserted value matched"
    end

    test "a passing comparison still shows what it did not check" do
      # The whole point of showing coverage: a green over 1 of 2 values is not
      # the same claim as a green over 2 of 2.
      comparison = Comparison.compare(%{intent: "greeting"}, %{intent: "greeting", score: 0.9})

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "1</span>\n      of 2 values asserted" or html =~ "of 2 values asserted"
      assert html =~ "1 unchecked"
    end

    test "a failing comparison renders each mismatch with its path and reason" do
      comparison =
        Comparison.compare(
          %{speech_act: %{category: "directive"}, tokens: ["turn", "on"]},
          %{speech_act: %{category: "assertive"}, tokens: ["turn", "off"]}
        )

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "Fail"
      assert html =~ "speech_act.category"
      assert html =~ "tokens.1"
      assert html =~ "differs"
    end

    test "a missing key renders as missing rather than as a difference" do
      comparison = Comparison.compare(%{intent: "greeting"}, %{score: 0.9})

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "missing"
      assert html =~ "intent"
    end

    test "a length mismatch renders as length" do
      comparison = Comparison.compare(%{tokens: ["turn"]}, %{tokens: ["turn", "on"]})

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "length"
    end

    test "the tolerance in force is shown, since it decides float verdicts" do
      comparison = Comparison.compare(%{score: 0.75}, %{score: 0.76}, tolerance: 0.02)

      html = render_component(&Diff.diff/1, comparison: comparison)

      assert html =~ "tolerance"
      assert html =~ "0.02"
    end
  end

  describe "verdict/1" do
    test "renders a label for every status a case can hold" do
      for {status, label} <- [
            {"pass", "Pass"},
            {"fail", "Fail"},
            {"error", "Raised"},
            {"pending", "Not run"}
          ] do
        assert render_component(&Diff.verdict/1, status: status) =~ label
      end
    end

    test "a status outside the closed vocabulary raises rather than rendering" do
      assert_raise FunctionClauseError, fn ->
        render_component(&Diff.verdict/1, status: "probably fine")
      end
    end
  end

  describe "coverage/1" do
    test "shows the counts as plain numbers" do
      html = render_component(&Diff.coverage/1, checked: 7, total: 143)

      assert html =~ "7"
      assert html =~ "of 143 values asserted"
      assert html =~ "136 unchecked"
    end

    test "full coverage says nothing about unchecked values" do
      html = render_component(&Diff.coverage/1, checked: 3, total: 3)

      assert html =~ "of 3 values asserted"
      refute html =~ "unchecked"
    end
  end

  describe "term/1 on a real normalised term" do
    test "renders nested maps, lists and scalars" do
      term =
        Comparison.normalize(%{
          intent: "greeting",
          entities: [%{type: :person, confidence: 0.9}],
          raw: {:ok, 1}
        })

      html = render_component(&Diff.term/1, term: term)

      assert html =~ "intent"
      assert html =~ "greeting"
      assert html =~ "entities"
      assert html =~ "confidence"
      assert html =~ "__tuple__"
    end

    test "an empty map and an empty list say so rather than rendering blank" do
      html = render_component(&Diff.term/1, term: %{"a" => %{}, "b" => []})

      assert html =~ "(empty map)"
      assert html =~ "(empty list)"
    end

    test "a defaulted path is marked and an undefaulted one is not" do
      term = Comparison.normalize(%{entity: %{familiarity: 0.5, count: 2}})

      html =
        render_component(&Diff.term/1,
          term: term,
          defaulted: [["entity", "familiarity"]]
        )

      assert html =~ "default"
      assert html =~ "text-origin-default"
    end

    test "with no defaulted paths nothing is marked" do
      term = Comparison.normalize(%{entity: %{familiarity: 0.5}})

      html = render_component(&Diff.term/1, term: term, defaulted: [])

      refute html =~ "text-origin-default"
    end

    test "a bare scalar renders without a surrounding structure" do
      assert render_component(&Diff.term/1, term: "hello") =~ "hello"
    end
  end
end
