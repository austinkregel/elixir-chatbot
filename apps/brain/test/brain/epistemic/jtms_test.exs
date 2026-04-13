defmodule Brain.Epistemic.JTMSTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog

  alias Brain.Epistemic.JTMS

  setup do
    # Ensure JTMS is started and cleared before each test
    case Process.whereis(JTMS) do
      nil ->
        {:ok, _pid} = JTMS.start_link([])
        :ok

      _pid ->
        JTMS.clear()
        :ok
    end
  end

  describe "node creation" do
    test "creates a premise node that is IN" do
      {:ok, node_id} = JTMS.create_premise("Socrates is a man")

      assert JTMS.is_in?(node_id) == true
      {:ok, node} = JTMS.get_node(node_id)
      assert node.node_type == :premise
      assert node.label == :in
    end

    test "creates an assumption node that is OUT by default" do
      {:ok, node_id} = JTMS.create_assumption("Birds fly")

      assert JTMS.is_in?(node_id) == false
      {:ok, node} = JTMS.get_node(node_id)
      assert node.node_type == :assumption
      assert node.label == :out
    end

    test "creates an enabled assumption node that is IN" do
      {:ok, node_id} = JTMS.create_assumption("Birds fly", true)

      assert JTMS.is_in?(node_id) == true
      {:ok, node} = JTMS.get_node(node_id)
      assert node.assumption_enabled == true
    end

    test "creates a contradiction node" do
      {:ok, node_id} = JTMS.create_contradiction("Inconsistent state")

      {:ok, node} = JTMS.get_node(node_id)
      assert node.node_type == :contradiction
    end
  end

  describe "justification" do
    test "creates a simple justification" do
      {:ok, premise} = JTMS.create_premise("Socrates is a man")
      {:ok, conclusion} = JTMS.create_node("Socrates is mortal")

      {:ok, _just_id} = JTMS.justify_node([premise], conclusion, "mortality_rule")

      # Conclusion should now be IN
      assert JTMS.is_in?(conclusion) == true
    end

    test "conclusion is OUT when premise is OUT" do
      {:ok, assumption} = JTMS.create_assumption("Tweety is a bird")
      {:ok, conclusion} = JTMS.create_node("Tweety flies")

      {:ok, _just_id} = JTMS.justify_node([assumption], conclusion, "bird_flight_rule")

      # Assumption is OUT, so conclusion should be OUT
      assert JTMS.is_in?(conclusion) == false
    end

    test "enables assumption propagates to conclusion" do
      {:ok, assumption} = JTMS.create_assumption("Tweety is a bird")
      {:ok, conclusion} = JTMS.create_node("Tweety flies")

      {:ok, _just_id} = JTMS.justify_node([assumption], conclusion, "bird_flight_rule")

      assert JTMS.is_in?(conclusion) == false

      # Enable the assumption
      :ok = JTMS.enable_assumption(assumption)

      # Now conclusion should be IN
      assert JTMS.is_in?(conclusion) == true
    end

    test "retracting assumption propagates to conclusion" do
      {:ok, assumption} = JTMS.create_assumption("Tweety is a bird", true)
      {:ok, conclusion} = JTMS.create_node("Tweety flies")

      {:ok, _just_id} = JTMS.justify_node([assumption], conclusion, "bird_flight_rule")

      assert JTMS.is_in?(conclusion) == true

      # Retract the assumption
      :ok = JTMS.retract_assumption(assumption)

      # Now conclusion should be OUT
      assert JTMS.is_in?(conclusion) == false
    end
  end

  describe "justification with out_list" do
    test "justification with out_list is valid when out nodes are OUT" do
      {:ok, bird} = JTMS.create_assumption("Tweety is a bird", true)
      {:ok, penguin} = JTMS.create_assumption("Tweety is a penguin")
      {:ok, flies} = JTMS.create_node("Tweety flies")

      # Tweety flies if bird AND NOT penguin
      {:ok, _just_id} = JTMS.justify_node([bird], [penguin], flies, "default_flight")

      # penguin is OUT, so justification is valid
      assert JTMS.is_in?(flies) == true
    end

    test "justification with out_list is invalid when out node becomes IN" do
      {:ok, bird} = JTMS.create_assumption("Tweety is a bird", true)
      {:ok, penguin} = JTMS.create_assumption("Tweety is a penguin")
      {:ok, flies} = JTMS.create_node("Tweety flies")

      {:ok, _just_id} = JTMS.justify_node([bird], [penguin], flies, "default_flight")

      assert JTMS.is_in?(flies) == true

      # Enable penguin - now Tweety doesn't fly
      :ok = JTMS.enable_assumption(penguin)

      assert JTMS.is_in?(flies) == false
    end
  end

  describe "contradiction detection" do
    test "registers and detects contradiction" do
      {:ok, a} = JTMS.create_assumption("A is guilty", true)
      {:ok, not_a} = JTMS.create_assumption("A is not guilty", true)

      # register_contradiction triggers the handler immediately if both nodes are IN
      # Capture the log to avoid leaking to test output
      log =
        capture_log([level: :warning], fn ->
          {:ok, _contra_id} = JTMS.register_contradiction([a, not_a])
        end)

      # The contradiction handler logs a warning during registration
      assert log =~ "Contradiction detected"

      # check_consistency confirms the contradiction
      {:error, {:contradiction, _}} = JTMS.check_consistency()
    end

    test "no contradiction when assumptions don't conflict" do
      {:ok, a} = JTMS.create_assumption("A is guilty", true)
      {:ok, not_a} = JTMS.create_assumption("A is not guilty")

      {:ok, _contra_id} = JTMS.register_contradiction([a, not_a])

      # not_a is OUT, so no contradiction
      {:ok, :consistent} = JTMS.check_consistency()
    end
  end

  describe "why_node" do
    test "returns supporting justifications" do
      {:ok, premise1} = JTMS.create_premise("Socrates is a man")
      {:ok, premise2} = JTMS.create_premise("All men are mortal")
      {:ok, conclusion} = JTMS.create_node("Socrates is mortal")

      {:ok, just_id} = JTMS.justify_node([premise1, premise2], conclusion, "syllogism")

      {:ok, result} = JTMS.why_node(conclusion)

      assert result.node.id == conclusion
      assert length(result.supporting_justifications) == 1
      assert hd(result.supporting_justifications).id == just_id
    end
  end

  describe "consequences_of" do
    test "returns nodes that depend on the given node" do
      {:ok, premise} = JTMS.create_premise("Socrates is a man")
      {:ok, conclusion1} = JTMS.create_node("Socrates is mortal")
      {:ok, conclusion2} = JTMS.create_node("Socrates will die")

      {:ok, _} = JTMS.justify_node([premise], conclusion1, "rule1")
      {:ok, _} = JTMS.justify_node([conclusion1], conclusion2, "rule2")

      {:ok, consequences} = JTMS.consequences_of(premise)

      consequence_ids = Enum.map(consequences, & &1.id)
      assert conclusion1 in consequence_ids
    end
  end

  describe "stats" do
    test "returns network statistics" do
      {:ok, _} = JTMS.create_premise("P1")
      {:ok, _} = JTMS.create_assumption("A1", true)
      {:ok, _} = JTMS.create_node("N1")

      stats = JTMS.stats()

      assert stats.total_nodes == 3
      assert stats.in_nodes == 2
      assert stats.out_nodes == 1
    end
  end
end
