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
  use ExUnit.Case, async: false

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

  describe "has_negation?/1 with contracted negation" do
    # Regression: has_negation?/1 split on ~r/\W+/, so "don't" became
    # ["don", "t"] and neither token is in the negation vocabulary, which lists
    # the negator itself. Every contracted negation was therefore reported as
    # affirmative. Measured on the cognitive-distortions corpus before the fix:
    # 55 of 206 negated sentences (26.7%) were missed.
    #
    # Every case in the describe blocks above uses negation spelled out in full,
    # which is why the defect survived the suite.
    @contracted [
      "I don't like rain",
      "I can't help it",
      "It won't work",
      "She didn't come to the party",
      "It isn't working",
      "They aren't here",
      "That wasn't flawless",
      "We weren't ready",
      "He doesn't like cats"
    ]

    for text <- @contracted do
      @text text

      test "detects contracted negation in: #{text}" do
        assert LinguisticData.has_negation?(@text),
               "expected contracted negation to be detected in #{inspect(@text)}"
      end
    end

    test "contracted and expanded forms agree" do
      pairs = [
        {"I don't like rain", "I do not like rain"},
        {"It won't work", "It will not work"},
        {"She didn't come", "She did not come"}
      ]

      for {contracted, expanded} <- pairs do
        assert LinguisticData.has_negation?(contracted) ==
                 LinguisticData.has_negation?(expanded),
               "#{inspect(contracted)} and #{inspect(expanded)} must agree"
      end
    end

    test "expanding contractions does not create false positives" do
      refute LinguisticData.has_negation?("I can help it")
      refute LinguisticData.has_negation?("She tied a knot in the rope")
      refute LinguisticData.has_negation?("The cannon fired")
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
