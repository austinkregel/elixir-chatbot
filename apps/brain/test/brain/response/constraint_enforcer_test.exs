defmodule Brain.Response.ConstraintEnforcerTest do
  @moduledoc """
  Tests for the contradiction-pre-emit gate added to
  `Brain.Response.ConstraintEnforcer.validate/3`.

  Covers:
  - Pass-through when no factual primitives are present.
  - Pass-through when factual primitives carry facts that don't contradict
    any existing belief.
  - Rejection (`{:rejected, _}`) when a factual primitive carries a fact that
    contradicts a belief seeded into `Brain.Epistemic.BeliefStore`. The
    rejection signal is what triggers `Brain.Response.RefinementLoop` to
    request another candidate (it bubbles through `OuroRealizer` as
    `{:error, {:constraint_rejected, _}}`).
  """
  use Brain.Test.GraphCase, async: false

  alias Brain.Response.ConstraintEnforcer
  alias Brain.Response.Primitive
  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.Types.Belief

  setup do
    # Ensure the predicate atom we'll use exists so
    # `Brain.FactDatabase.Integration.normalize_entity/1`'s
    # `String.to_existing_atom/1` succeeds.
    _ = String.to_atom("constraint_enforcer_test_paris")
    :ok
  end

  defp factual_primitive(opts) do
    facts = Keyword.get(opts, :facts, [])
    entities = Keyword.get(opts, :entities, [])

    Primitive.new(:content, :factual, %{
      facts: facts,
      entity_context: entities
    })
  end

  describe "validate/3 contradiction gate" do
    test "passes through when there are no factual primitives" do
      primitives = [Primitive.new(:greeting, :hello, %{})]

      assert {:ok, "Hello there friend"} =
               ConstraintEnforcer.validate("Hello there friend", primitives)
    end

    test "passes through when fact does not contradict any existing belief" do
      primitives = [
        factual_primitive(
          facts: [%{fact: "constraint_enforcer_test_paris is the capital of france"}],
          entities: [%{value: "constraint_enforcer_test_paris"}]
        )
      ]

      result =
        ConstraintEnforcer.validate(
          "constraint_enforcer_test_paris is the capital of france and a great city",
          primitives
        )

      # We don't seed a contradicting belief, so the contradiction gate
      # must not reject. Other checks may still reject for unrelated
      # reasons; assert specifically that the rejection isn't ours.
      case result do
        {:ok, _} ->
          :ok

        {:rejected, reason} ->
          refute String.contains?(reason, "Asserted fact contradicts")
      end
    end

    test "rejects when a primitive's fact contradicts a seeded belief" do
      # Seed a belief on subject :world, predicate :constraint_enforcer_test_paris
      # whose object asserts the affirmative form. Then submit a candidate
      # response whose factual primitive asserts the negation -- the
      # contradiction detector treats negation differences as contradictions.
      affirmative = "constraint_enforcer_test_paris is the capital of france"

      belief =
        Belief.new(
          :world,
          :constraint_enforcer_test_paris,
          affirmative,
          source: :curated_fact,
          confidence: 0.95
        )

      {:ok, _belief_id} = BeliefStore.add_belief(belief)

      contradictory = "constraint_enforcer_test_paris is not the capital of france"

      primitives = [
        factual_primitive(
          facts: [%{fact: contradictory}],
          entities: [%{value: "constraint_enforcer_test_paris"}]
        )
      ]

      # Use a response text that satisfies length and shares enough tokens
      # with the fact to pass fact_fidelity, so the contradiction gate is
      # actually reached.
      response_text =
        "Yes, constraint_enforcer_test_paris is not the capital of france as commonly assumed."

      assert {:rejected, reason} = ConstraintEnforcer.validate(response_text, primitives)
      assert reason =~ "Asserted fact contradicts"
    end
  end
end
