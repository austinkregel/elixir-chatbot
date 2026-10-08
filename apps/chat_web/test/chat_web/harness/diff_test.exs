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

    test "a not-run verdict takes words saying why nothing was judged, keeping its mark" do
      html = render_component(&Diff.verdict/1, status: "pending", label: "gate not set")

      assert html =~ "gate not set"
      refute html =~ "Not run"
      assert html =~ ~s(data-mark="dashed_square")
      assert html =~ "text-verdict-pending"
    end

    test "pass, fail and raised keep their own words: a label on them raises" do
      for status <- ["pass", "fail", "error"] do
        assert_raise ArgumentError, ~r/label words only a non-blank not-run verdict/, fn ->
          render_component(&Diff.verdict/1, status: status, label: "gate not set")
        end
      end
    end

    test "a blank label raises" do
      assert_raise ArgumentError, ~r/non-blank not-run verdict/, fn ->
        render_component(&Diff.verdict/1, status: "pending", label: "")
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

    test "with no output to count against, it says how many values are asserted" do
      html = render_component(&Diff.coverage/1, checked: 2)

      assert html =~ "values asserted"
      refute html =~ " of "
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

    test "a stand-in path is marked and another path is not" do
      term = Comparison.normalize(%{entity: %{familiarity: 0.5, count: 2}})

      html =
        render_component(&Diff.term/1,
          term: term,
          stand_ins: [%{path: ["entity", "familiarity"], origin: :default}]
        )

      assert html =~ "fallback default"
      assert html =~ "text-origin-default"
      assert html =~ "decoration-dashed"
      assert length(Regex.scan(~r/data-origin=/, html)) == 1
    end

    test "each stand-in is labeled by its real origin, not default for all" do
      term = Comparison.normalize(%{memory: %{turns: 0}, config: %{gazetteer: nil}})

      html =
        render_component(&Diff.term/1,
          term: term,
          stand_ins: [
            %{path: ["memory", "turns"], origin: :absent},
            %{path: ["config", "gazetteer"], origin: :unavailable}
          ]
        )

      assert html =~ ~s(data-origin="absent")
      assert html =~ "no data — stand-in"
      assert html =~ "decoration-dotted"
      assert html =~ "bg-origin-absent-wash"

      assert html =~ ~s(data-origin="unavailable")
      assert html =~ "source unreadable"
      assert html =~ "decoration-wavy"
      assert html =~ "bg-origin-unavailable-wash"

      refute html =~ "fallback default"
    end

    test "a map or list that stood in whole is marked too" do
      term = Comparison.normalize(%{memory: %{context: []}, config: %{domain_lemmas: %{}}})

      html =
        render_component(&Diff.term/1,
          term: term,
          stand_ins: [
            %{path: ["memory", "context"], origin: :absent},
            %{path: ["config", "domain_lemmas"], origin: :default}
          ]
        )

      assert html =~ ~s(data-origin="absent")
      assert html =~ ~s(data-origin="default")
      assert html =~ "(empty list)"
      assert html =~ "(empty map)"
    end

    test "a path that stood in by two origins shows both" do
      term = Comparison.normalize(%{threshold: 0.4})

      html =
        render_component(&Diff.term/1,
          term: term,
          stand_ins: [
            %{path: ["threshold"], origin: :default},
            %{path: ["threshold"], origin: :absent}
          ]
        )

      assert html =~ ~s(data-origin="default")
      assert html =~ ~s(data-origin="absent")
    end

    test "an origin that is not a stand-in raises" do
      assert_raise ArgumentError, ~r/is not a stand-in origin/, fn ->
        render_component(&Diff.term/1,
          term: %{"a" => 1},
          stand_ins: [%{path: ["a"], origin: :computed}]
        )
      end
    end

    test "with no stand-ins nothing is marked" do
      term = Comparison.normalize(%{entity: %{familiarity: 0.5}})

      html = render_component(&Diff.term/1, term: term, stand_ins: [])

      refute html =~ "data-origin"
    end

    test "a bare scalar renders without a surrounding structure" do
      assert render_component(&Diff.term/1, term: "hello") =~ "hello"
    end
  end
end
