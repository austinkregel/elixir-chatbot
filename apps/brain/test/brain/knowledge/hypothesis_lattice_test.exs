defmodule Brain.Knowledge.HypothesisLatticeTest do
  use ExUnit.Case, async: true

  alias Brain.Knowledge.Types.{Hypothesis, Investigation}
  alias Brain.Lattice

  defp sample_investigation do
    h1 =
      Hypothesis.new("Claim A is true",
        entity: "A",
        derived_from: "Q1",
        prediction: "If A then sources agree."
      )

    h2 =
      Hypothesis.new("Claim B is true",
        entity: "B",
        derived_from: "Q2",
        prediction: "If B then sources agree."
      )

    h1 = %{h1 | confidence: 0.9, supporting_evidence: [%{}, %{}], contradicting_evidence: []}
    h2 = %{h2 | confidence: 0.2, supporting_evidence: [], contradicting_evidence: []}

    Investigation.new("topic", hypotheses: [h1, h2])
  end

  test "rank_hypotheses/1 builds a lattice ordered by confidence" do
    inv = sample_investigation()
    lat = Investigation.rank_hypotheses(inv)
    assert %Lattice{} = lat
    assert %Hypothesis{claim: c} = Lattice.best(lat).label
    assert c =~ "Claim A"
  end

  test "falsification_margin/1 returns non-negative margin" do
    inv = sample_investigation()
    assert Investigation.falsification_margin(inv) >= 0.0
  end

  test "formulate_competing_hypotheses/2 returns a lattice" do
    goal = Brain.Knowledge.Types.ResearchGoal.new("Research X")
    lat = Brain.Knowledge.Types.ResearchGoal.formulate_competing_hypotheses(goal, "TestEntity")
    assert %Lattice{} = lat
    assert length(lat.candidates) >= 3
  end
end
