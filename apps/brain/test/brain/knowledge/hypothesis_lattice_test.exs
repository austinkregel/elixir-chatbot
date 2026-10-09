defmodule Brain.Knowledge.HypothesisLatticeTest do
  use ExUnit.Case, async: true

  alias Brain.Epistemic.BeliefStore
  alias Brain.Knowledge.Types.{Hypothesis, Investigation, ResearchGoal}
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

  describe "formulate_competing_hypotheses/3" do
    setup do
      world = "test_hypothesis_lattice_#{System.unique_integer([:positive])}"
      other = "test_hypothesis_lattice_other_#{System.unique_integer([:positive])}"

      on_exit(fn ->
        :ok = BeliefStore.clear(world)
        :ok = BeliefStore.clear(other)
      end)

      {:ok, world: world, other: other}
    end

    test "returns a lattice", %{world: world} do
      goal = ResearchGoal.new("Research X")
      lat = ResearchGoal.formulate_competing_hypotheses(world, goal, "TestEntity")
      assert %Lattice{} = lat
      assert length(lat.candidates) >= 3
    end

    test "takes its priors from the world's own beliefs", %{world: world, other: other} do
      {:ok, _id} = BeliefStore.add_belief(world, "TestEntity", :is, "a prior", confidence: 0.9)
      goal = ResearchGoal.new("Research X")

      here = ResearchGoal.formulate_competing_hypotheses(world, goal, "TestEntity")
      there = ResearchGoal.formulate_competing_hypotheses(other, goal, "TestEntity")

      assert length(here.candidates) == length(there.candidates) + 1
    end

    test "raises without a world" do
      goal = ResearchGoal.new("Research X")

      for world <- [nil, ""] do
        assert_raise FunctionClauseError, fn ->
          ResearchGoal.formulate_competing_hypotheses(world, goal, "TestEntity")
        end
      end
    end
  end
end
