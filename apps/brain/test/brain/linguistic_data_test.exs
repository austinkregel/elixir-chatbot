defmodule Brain.LinguisticDataTest do
  @moduledoc """
  Unit tests for `Brain.LinguisticData.negation?/1` and
  `Brain.LinguisticData.has_negation?/1`.

  Includes parity assertions against the legacy ad-hoc
  `Enum.any?(negation_words(), &String.contains?(text, &1))` style of check
  that previously lived in `Knowledge.Types`, `Academic.PaperModelBuilder`,
  and the local hardcoded MapSet in `ContradictionDetector`. The unified
  predicates must agree with the legacy approach on representative inputs
  that don't trigger known substring false positives.
  """
  use ExUnit.Case, async: true

  alias Brain.LinguisticData

  describe "negation?/1" do
    test "matches plain negation tokens" do
      for word <- ~w(not no never none cannot won't don't can't),
          do: assert(LinguisticData.negation?(word), "expected #{word} to be a negation")
    end

    test "is case-insensitive" do
      assert LinguisticData.negation?("NOT")
      assert LinguisticData.negation?("Never")
    end

    test "rejects non-negation words" do
      for word <- ~w(yes happy water table cat),
          do: refute(LinguisticData.negation?(word), "expected #{word} not to be a negation")
    end
  end

  describe "has_negation?/1" do
    test "detects negation tokens in a sentence" do
      assert LinguisticData.has_negation?("I do not like rain")
      assert LinguisticData.has_negation?("She has never been there")
      assert LinguisticData.has_negation?("There is no answer")
    end

    test "returns false for plain affirmative sentences" do
      refute LinguisticData.has_negation?("I like rain")
      refute LinguisticData.has_negation?("Water is a liquid")
      refute LinguisticData.has_negation?("Paris is in France")
    end

    test "is case-insensitive" do
      assert LinguisticData.has_negation?("I do NOT like this")
      assert LinguisticData.has_negation?("Never again!")
    end

    test "avoids substring false positives the legacy check produced" do
      # Legacy: Enum.any?(negation_words, &String.contains?("knot", &1)) == true
      # because "knot" contains "not". The token-level predicate must NOT match.
      refute LinguisticData.has_negation?("She tied a knot in the rope")
      refute LinguisticData.has_negation?("The cannon fired")
    end

    test "returns false for non-string input" do
      refute LinguisticData.has_negation?(nil)
      refute LinguisticData.has_negation?(123)
    end
  end

  describe "parity with legacy substring-based checks" do
    # These inputs are the kind both Knowledge.Types and PaperModelBuilder
    # used to handle correctly with the substring approach. The new predicate
    # must agree on each.
    @parity_cases [
      {"Paris is the capital of France", false},
      {"Paris is not the capital of France", true},
      {"Water boils at 100 degrees", false},
      {"Water does not boil at 100 degrees", true},
      {"He likes cats", false},
      {"He never likes cats", true}
    ]

    for {text, expected} <- @parity_cases do
      @text text
      @expected expected

      test "parity for: #{text}" do
        assert LinguisticData.has_negation?(@text) == @expected
      end
    end
  end
end
