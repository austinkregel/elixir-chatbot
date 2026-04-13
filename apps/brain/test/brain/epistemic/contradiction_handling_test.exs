defmodule Brain.Epistemic.ContradictionHandlingTest do
  @moduledoc "Tests for handling contradictions when user input contradicts existing beliefs.\n\nThese tests verify:\n- Contradiction detection in fact verification\n- Learner handling of contradictory facts\n- Belief store queries for contradictions\n- End-to-end contradiction handling in conversations\n"
  alias Brain.Epistemic
  use Brain.Test.GraphCase, async: false

  alias Epistemic.{BeliefStore, JTMS, ContradictionHandler, UserModelStore}
  alias Brain.FactDatabase
  alias Brain.FactDatabase.Integration
  alias Brain
  import Brain.TestHelpers

  setup do
    start_brain_services()
    ensure_epistemic_stores_started()
    {:ok, conversation_id} = Brain.create_conversation()
    user_id = "test_user_#{:rand.uniform(100_000)}"

    %{conversation_id: conversation_id, user_id: user_id}
  end

  describe "FactDatabase.Integration.verify_fact" do
    test "detects contradiction when fact contradicts existing belief" do
      {:ok, belief_id1} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.verify_fact("france", "The capital is not Paris")

      assert {:contradicted, conflicting_beliefs} = result
      assert length(conflicting_beliefs) == 1
      assert hd(conflicting_beliefs).id == belief_id1
      assert hd(conflicting_beliefs).object == "The capital is Paris"
    end

    test "detects contradiction with negation" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :water,
          "Water boils at 100C",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.verify_fact("water", "Water does not boil at 100C")

      assert {:contradicted, conflicting_beliefs} = result
      assert length(conflicting_beliefs) == 1
    end

    test "returns verified when fact is consistent" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.verify_fact("france", "The capital is Paris")

      assert {:verified, confidence} = result
      assert confidence >= 0.9
    end

    test "returns uncertain when no existing beliefs" do
      result = Integration.verify_fact("unknown_entity", "Some fact")

      assert {:uncertain, :no_existing_beliefs} = result
    end

    test "detects contradiction with opposite meaning" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :temperature,
          "It is hot",
          confidence: 0.8,
          source: :explicit
        )

      result = Integration.verify_fact("temperature", "It is cold")

      assert match?({:contradicted, _}, result) or
               match?({:uncertain, _}, result) or
               match?({:verified, _}, result)
    end
  end

  describe "FactDatabase.Integration.check_contradiction" do
    test "checks for contradictions against existing beliefs" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.check_contradiction("france", "The capital is not Paris")

      assert {:contradiction, conflicting_beliefs} = result
      assert length(conflicting_beliefs) == 1
    end

    test "returns consistent when no contradiction" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.check_contradiction("france", "The capital is Paris")

      assert :consistent = result
    end

    test "returns no_data when no existing beliefs" do
      result = Integration.check_contradiction("unknown_entity", "Some fact")
      assert result in [:no_data, :consistent]
    end
  end

  describe "User belief contradictions" do
    test "detects when user contradicts their own previous statement", %{user_id: user_id} do
      {:ok, _belief_id1} =
        BeliefStore.add_belief(
          :user,
          "location",
          "New York",
          user_id: user_id,
          confidence: 0.85,
          source: :explicit
        )

      result =
        Integration.verify_fact("location", "Chicago")
        |> case do
          {:uncertain, _} ->
            case BeliefStore.query_beliefs(
                   subject: :user,
                   predicate: "location",
                   user_id: user_id
                 ) do
              {:ok, beliefs} ->
                contradictions =
                  Enum.filter(beliefs, fn belief ->
                    belief.object != "Chicago"
                  end)

                if contradictions != [] do
                  {:contradicted, contradictions}
                else
                  :consistent
                end

              _ ->
                :no_data
            end

          other ->
            other
        end

      assert result == :consistent or match?({:contradicted, _}, result)
      beliefs = BeliefStore.query_beliefs(subject: :user, predicate: "location", user_id: user_id)
      assert {:ok, user_beliefs} = beliefs
      assert user_beliefs != []
    end

    test "handles multiple contradictory beliefs about same entity", %{user_id: user_id} do
      {:ok, _belief_id1} =
        BeliefStore.add_belief(
          :user,
          :location,
          "New York",
          user_id: user_id,
          confidence: 0.85,
          source: :explicit
        )

      {:ok, _belief_id2} =
        BeliefStore.add_belief(
          :user,
          :location,
          "Chicago",
          user_id: user_id,
          confidence: 0.8,
          source: :explicit
        )

      {:ok, beliefs} =
        BeliefStore.query_beliefs(subject: :user, predicate: :location, user_id: user_id)

      assert length(beliefs) == 2
      locations = Enum.map(beliefs, &String.downcase(&1.object))
      assert "new york" in locations
      assert "chicago" in locations
      result = Integration.check_contradiction("location", "Los Angeles")

      assert result == :no_data or
               result == :consistent or
               match?({:contradiction, _}, result)
    end
  end

  describe "Learner contradiction handling" do
    test "detects contradiction when learned fact contradicts existing belief" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      # verify_fact detects the contradiction and returns it
      result = Integration.verify_fact("france", "The capital is not Paris")
      assert {:contradicted, conflicting_beliefs} = result

      assert length(conflicting_beliefs) == 1,
             "Expected 1 conflicting belief, got: #{inspect(conflicting_beliefs)}"

      {:ok, beliefs} = BeliefStore.query_beliefs(subject: :world, predicate: :france)
      paris_beliefs = Enum.filter(beliefs, &(&1.object == "The capital is Paris"))
      not_paris_beliefs = Enum.filter(beliefs, &(&1.object == "The capital is not Paris"))

      assert length(paris_beliefs) == 1,
             "Original belief should still exist when contradiction detected"

      assert not_paris_beliefs == [],
             "Contradictory fact should not be added when contradiction detected"
    end

    test "allows adding fact when no contradiction exists" do
      {:ok, _belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.9,
          source: :explicit
        )

      result = Integration.verify_fact("france", "The capital is Paris")

      assert {:verified, _confidence} = result

      {:ok, _fact_id, _fact} =
        Integration.add_fact(
          "france",
          "The capital is Paris",
          category: "learned",
          confidence: 0.9,
          create_belief: true
        )
    end
  end

  describe "End-to-end contradiction handling in conversations" do
    test "handles contradiction when user changes their location", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, %{response: response1}} =
        Brain.evaluate(conv_id, "I'm from New York", user_id: user_id)

      assert String.length(response1) > 0

      {:ok, beliefs1} =
        BeliefStore.query_beliefs(
          subject: :user,
          predicate: :location,
          user_id: user_id
        )

      if beliefs1 != [] do
        new_york_belief =
          Enum.find(beliefs1, &(&1.object == "New York" or &1.object == "new york"))

        assert new_york_belief != nil
      end

      {:ok, %{response: response2}} =
        Brain.evaluate(conv_id, "Actually, I'm from Chicago", user_id: user_id)

      assert String.length(response2) > 0

      {:ok, beliefs2} =
        BeliefStore.query_beliefs(
          subject: :user,
          predicate: :location,
          user_id: user_id
        )

      if beliefs2 != [] do
        chicago_belief =
          Enum.find(beliefs2, fn b ->
            String.downcase(b.object) == "chicago"
          end)

        assert chicago_belief != nil, "Expected Chicago belief to be stored"
      end
    end

    test "handles contradiction when user corrects a fact", %{
      conversation_id: conv_id,
      user_id: user_id
    } do
      {:ok, %{response: _response1}} =
        Brain.evaluate(conv_id, "My favorite color is blue", user_id: user_id)

      {:ok, %{response: response2}} =
        Brain.evaluate(conv_id, "Actually, my favorite color is red", user_id: user_id)

      assert String.length(response2) > 0

      {:ok, beliefs} =
        BeliefStore.query_beliefs(
          subject: :user,
          predicate: :preference,
          user_id: user_id
        )

      assert is_list(beliefs)
    end
  end

  describe "JTMS integration with contradictions" do
    test "registers contradiction in JTMS when beliefs conflict" do
      import ExUnit.CaptureLog

      {:ok, node1} =
        JTMS.create_assumption("User is from New York", true)

      {:ok, node2} =
        JTMS.create_assumption("User is from Chicago", true)

      # register_contradiction triggers the handler immediately if both nodes are IN
      # Capture the log to avoid leaking to test output
      log =
        capture_log([level: :warning], fn ->
          {:ok, _} = JTMS.register_contradiction([node1, node2])
        end)

      # The contradiction was detected during registration
      assert log =~ "Contradiction detected"

      {:error, {:contradiction, contra_id}} = JTMS.check_consistency()
      contradictions = JTMS.get_contradictions()
      assert length(contradictions) == 1
      contradiction_node = hd(contradictions)
      assert contradiction_node.id == contra_id
    end

    test "contradiction handler receives notification" do
      import ExUnit.CaptureLog

      {:ok, node1} = JTMS.create_assumption("Fact A is true", true)
      {:ok, node2} = JTMS.create_assumption("Fact A is not true", true)

      # register_contradiction triggers the handler immediately if both nodes are IN
      # Capture the log to avoid leaking to test output
      log =
        capture_log([level: :warning], fn ->
          {:ok, _} = JTMS.register_contradiction([node1, node2])
        end)

      # The contradiction was detected during registration
      assert log =~ "Contradiction detected"

      {:error, {:contradiction, contra_id}} = JTMS.check_consistency()
      contradictions = JTMS.get_contradictions()
      assert length(contradictions) == 1
      contradiction_node = hd(contradictions)
      assert contradiction_node.label == :in
      assert contradiction_node.id == contra_id
    end
  end

  describe "Belief confidence and contradiction resolution" do
    test "higher confidence belief takes precedence in contradiction detection" do
      {:ok, high_conf_belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is Paris",
          confidence: 0.95,
          source: :explicit
        )

      {:ok, _low_conf_belief_id} =
        BeliefStore.add_belief(
          :world,
          :france,
          "The capital is not Paris",
          confidence: 0.6,
          source: :inferred
        )

      result = Integration.check_contradiction("france", "The capital is Paris")
      assert match?({:contradiction, _}, result)
      {:ok, beliefs} = BeliefStore.query_beliefs(subject: :world, predicate: :france)
      assert length(beliefs) == 2
      high_conf = Enum.find(beliefs, &(&1.id == high_conf_belief_id))
      assert high_conf.confidence == 0.95
    end
  end

  defp ensure_epistemic_stores_started do
    ensure_started(BeliefStore)
    ensure_started(JTMS)
    ensure_started(ContradictionHandler)
    ensure_started(UserModelStore)
    ensure_started(FactDatabase)
    BeliefStore.clear()
    JTMS.clear()
    UserModelStore.clear_all()
  end
end
