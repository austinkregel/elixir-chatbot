defmodule Brain.Knowledge.ContradictionDetectorTest do
  use ExUnit.Case, async: false

  alias Brain.Knowledge.ContradictionDetector

  describe "has_negation_difference?/2" do
    test "detects negation difference" do
      assert ContradictionDetector.has_negation_difference?(
               "Water is a liquid",
               "Water is not a liquid"
             )
    end

    test "no negation difference for identical texts" do
      refute ContradictionDetector.has_negation_difference?(
               "I like cats",
               "I like cats"
             )
    end

    test "no negation difference when both have negation" do
      refute ContradictionDetector.has_negation_difference?(
               "It is not raining",
               "It is not sunny"
             )
    end

    test "handles nil inputs gracefully" do
      refute ContradictionDetector.has_negation_difference?(nil, "text")
      refute ContradictionDetector.has_negation_difference?("text", nil)
    end

    test "detects contraction negation" do
      assert ContradictionDetector.has_negation_difference?(
               "Water is a liquid",
               "Water isn't a liquid"
             )
    end
  end

  describe "has_number_disagreement?/2" do
    test "detects number disagreement" do
      assert ContradictionDetector.has_number_disagreement?(
               "The population is 1000",
               "The population is 2000"
             )
    end

    test "no disagreement for close numbers" do
      refute ContradictionDetector.has_number_disagreement?(
               "The score is 95",
               "The score is 98"
             )
    end

    test "no disagreement when no numbers" do
      refute ContradictionDetector.has_number_disagreement?(
               "Paris is the capital",
               "London is the capital"
             )
    end

    test "handles nil inputs gracefully" do
      refute ContradictionDetector.has_number_disagreement?(nil, "100")
    end
  end

  describe "contradicts?/2" do
    test "detects negation contradiction" do
      assert ContradictionDetector.contradicts?(
               "Water is a liquid",
               "Water is not a liquid"
             )
    end

    test "detects number contradiction" do
      assert ContradictionDetector.contradicts?(
               "The population is 1000",
               "The population is 5000"
             )
    end

    test "no contradiction for agreeing texts" do
      refute ContradictionDetector.contradicts?(
               "The sky is blue",
               "The sky is blue"
             )
    end

    test "detects is/is-not opposition" do
      assert ContradictionDetector.contradicts?(
               "The earth is flat",
               "The earth is not flat"
             )
    end

    test "handles nil inputs" do
      refute ContradictionDetector.contradicts?(nil, nil)
    end
  end
end
