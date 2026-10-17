defmodule Atlas.Verification.ComparisonTest do
  @moduledoc """
  The comparison a verification case's pass or fail is decided by.

  Every assertion states the whole result it expects rather than probing one
  field of it, so a change in an unasserted part of the shape surfaces here
  instead of passing unnoticed.
  """
  use ExUnit.Case, async: true

  alias Atlas.Verification.Comparison

  defmodule Sample do
    @moduledoc false
    defstruct [:label, :score]
  end

  describe "normalize/1 on terms jsonb cannot hold" do
    test "atoms become strings, as keys and as values" do
      assert Comparison.normalize(%{intent: :smalltalk}) == %{"intent" => "smalltalk"}
    end

    test "a tuple is tagged, so it does not store identically to a list" do
      assert Comparison.normalize({:ok, 1}) == %{
               "__tuple__" => true,
               "elements" => ["ok", 1]
             }

      refute Comparison.normalize({:ok, 1}) == Comparison.normalize([:ok, 1])
    end

    test "a struct keeps the module it came from" do
      assert Comparison.normalize(%Sample{label: "greeting", score: 0.5}) == %{
               "__struct__" => "Atlas.Verification.ComparisonTest.Sample",
               "label" => "greeting",
               "score" => 0.5
             }
    end

    test "a MapSet is tagged and its members normalised" do
      assert Comparison.normalize(MapSet.new([:a])) == %{
               "__mapset__" => true,
               "members" => ["a"]
             }
    end

    test "date and time structs become ISO 8601 strings" do
      assert Comparison.normalize(~D[2026-10-06]) == "2026-10-06"
      assert Comparison.normalize(~U[2026-10-06 01:02:03.000000Z]) == "2026-10-06T01:02:03.000000Z"
    end

    test "a term with no JSON counterpart is recorded as its inspect string" do
      assert %{"__inspect__" => inspected} = Comparison.normalize(self())
      assert inspected =~ "PID"
    end

    test "nesting is normalised all the way down" do
      actual = Comparison.normalize(%{outer: [%{inner: {:ok, :done}}]})

      assert actual == %{
               "outer" => [
                 %{"inner" => %{"__tuple__" => true, "elements" => ["ok", "done"]}}
               ]
             }
    end

    test "numbers, strings, booleans and nil pass through unchanged" do
      assert Comparison.normalize(%{a: 1, b: 1.5, c: "s", d: true, e: nil}) ==
               %{"a" => 1, "b" => 1.5, "c" => "s", "d" => true, "e" => nil}
    end
  end

  describe "leaf_count/1" do
    test "counts values and not the structural markers" do
      # "__tuple__" describes the shape; "elements" holds the two values.
      assert Comparison.leaf_count(Comparison.normalize({:ok, 1})) == 2
      assert Comparison.leaf_count(Comparison.normalize(%Sample{label: "x", score: 1.0})) == 2
    end

    test "a scalar is one leaf and a nested map is the sum of its leaves" do
      assert Comparison.leaf_count("x") == 1
      assert Comparison.leaf_count(%{"a" => %{"b" => 1, "c" => 2}, "d" => [3, 4]}) == 4
    end
  end

  describe "compare/3 — a partial expectation" do
    test "an expectation matching every key it names passes, and reports its coverage" do
      result = Comparison.compare(%{intent: "greeting"}, %{intent: "greeting", score: 0.9})

      assert result == %{
               status: "pass",
               checked: 1,
               total: 2,
               mismatches: [],
               tolerance: Comparison.default_tolerance()
             }
    end

    test "a key the expectation does not name is never compared" do
      # `score` differs and the case still passes, which is the cost of a
      # partial expectation. checked-of-total is what makes that visible.
      result = Comparison.compare(%{intent: "greeting"}, %{intent: "greeting", score: 0.1})

      assert result.status == "pass"
      assert result.checked == 1
      assert result.total == 2
    end

    test "a differing value fails and names its path" do
      result = Comparison.compare(%{intent: "greeting"}, %{intent: "farewell"})

      assert result.status == "fail"

      assert result.mismatches == [
               %{path: ["intent"], expected: "greeting", actual: "farewell", reason: :not_equal}
             ]
    end

    test "a missing key fails with reason :missing rather than comparing against nil" do
      result = Comparison.compare(%{intent: "greeting"}, %{score: 0.9})

      assert result.mismatches == [
               %{path: ["intent"], expected: "greeting", actual: nil, reason: :missing}
             ]
    end

    test "a nested mismatch reports the full path" do
      result =
        Comparison.compare(
          %{speech_act: %{category: "directive"}},
          %{speech_act: %{category: "assertive"}}
        )

      assert result.mismatches == [
               %{
                 path: ["speech_act", "category"],
                 expected: "directive",
                 actual: "assertive",
                 reason: :not_equal
               }
             ]
    end

    test "every mismatch is reported, not just the first" do
      result = Comparison.compare(%{a: 1, b: 2}, %{a: 9, b: 8})

      assert length(result.mismatches) == 2
      assert Enum.map(result.mismatches, & &1.path) |> Enum.sort() == [["a"], ["b"]]
    end
  end

  describe "compare/3 — lists" do
    test "a list matching element for element passes" do
      assert Comparison.compare(%{tokens: ["turn", "on"]}, %{tokens: ["turn", "on"]}).status ==
               "pass"
    end

    test "a different length fails as :length rather than element by element" do
      result = Comparison.compare(%{tokens: ["turn"]}, %{tokens: ["turn", "on"]})

      assert result.mismatches == [
               %{
                 path: ["tokens"],
                 expected: ["turn"],
                 actual: ["turn", "on"],
                 reason: :length
               }
             ]
    end

    test "a differing element reports its index in the path" do
      result = Comparison.compare(%{tokens: ["turn", "on"]}, %{tokens: ["turn", "off"]})

      assert result.mismatches == [
               %{path: ["tokens", "1"], expected: "on", actual: "off", reason: :not_equal}
             ]
    end
  end

  describe "compare/3 — floats" do
    test "a difference within tolerance passes" do
      assert Comparison.compare(%{score: 0.75}, %{score: 0.7500001}).status == "pass"
    end

    test "a difference outside tolerance fails" do
      assert Comparison.compare(%{score: 0.75}, %{score: 0.76}).status == "fail"
    end

    test "a case may widen its own tolerance, and the result records which was used" do
      result = Comparison.compare(%{score: 0.75}, %{score: 0.76}, tolerance: 0.02)

      assert result.status == "pass"
      assert result.tolerance == 0.02
    end

    test "an integer and a float of the same value compare equal" do
      assert Comparison.compare(%{count: 1}, %{count: 1.0}).status == "pass"
    end

    test "the default tolerance is tight enough to catch a two-decimal change" do
      assert Comparison.default_tolerance() < 0.01
    end
  end

  describe "compare/3 — an expectation that asserts nothing" do
    test "raises rather than reporting a pass" do
      # A case with an empty expectation cannot fail, so reporting "pass" would
      # be a green result obtained by checking nothing.
      assert_raise ArgumentError, fn -> Comparison.compare(%{}, %{intent: "greeting"}) end

      message =
        try do
          Comparison.compare(%{}, %{intent: "greeting"})
        rescue
          e -> Exception.message(e)
        end

      assert message =~ "asserts nothing"
    end
  end

  describe "compare/3 — shape disagreements" do
    test "a map expectation against a scalar fails rather than raising" do
      result = Comparison.compare(%{a: %{b: 1}}, %{a: "scalar"})

      assert result.status == "fail"
      assert hd(result.mismatches).path == ["a"]
    end

    test "a struct is compared by its module as well as its fields" do
      result =
        Comparison.compare(
          %Sample{label: "greeting", score: 1.0},
          %{"label" => "greeting", "score" => 1.0}
        )

      assert result.status == "fail"

      assert hd(result.mismatches) == %{
               path: ["__struct__"],
               expected: "Atlas.Verification.ComparisonTest.Sample",
               actual: nil,
               reason: :missing
             }
    end

    test "raw subsystem output may be passed without normalising first" do
      assert Comparison.compare(%{intent: :greeting}, %{intent: :greeting}).status == "pass"
    end
  end
end
