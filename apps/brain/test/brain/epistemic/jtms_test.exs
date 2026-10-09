defmodule Brain.Epistemic.JTMSTest do
  use ExUnit.Case, async: false
  import ExUnit.CaptureLog

  alias Brain.Epistemic.JTMS

  # Every test works in a network of its own world, `w`, and empties it
  # before and after: JTMS holds its networks in GenServer state, which the
  # Sandbox does not roll back. An empty network is the boot state —
  # `JTMS.init/1` builds the state literally and loads nothing.
  setup do
    unless Process.whereis(JTMS), do: {:ok, _pid} = JTMS.start_link([])

    world = "test_jtms_#{System.unique_integer([:positive])}"
    Brain.Test.Singletons.reset_epistemic_stores!([:jtms], world_id: world)
    %{w: world}
  end

  describe "the world" do
    test "every network function raises without a world" do
      for call <- [
            fn -> JTMS.create_node(nil, "x") end,
            fn -> JTMS.create_premise("", "x") end,
            fn -> JTMS.create_assumption(:default, "x") end,
            fn -> JTMS.create_contradiction(nil, "x") end,
            fn -> JTMS.justify_node(nil, [], "c", "i") end,
            fn -> JTMS.justify_node(nil, [], [], "c", "i") end,
            fn -> JTMS.enable_assumption(nil, "n") end,
            fn -> JTMS.retract_assumption(nil, "n") end,
            fn -> JTMS.is_in?(nil, "n") end,
            fn -> JTMS.get_label(nil, "n") end,
            fn -> JTMS.get_node(nil, "n") end,
            fn -> JTMS.why_node(nil, "n") end,
            fn -> JTMS.consequences_of(nil, "n") end,
            fn -> JTMS.antecedents_of(nil, "n") end,
            fn -> JTMS.register_contradiction(nil, ["a", "b"]) end,
            fn -> JTMS.check_consistency(nil) end,
            fn -> JTMS.get_contradictions(nil) end,
            fn -> JTMS.stats(nil) end,
            fn -> JTMS.clear(nil) end
          ] do
        assert_raise ArgumentError, ~r/world/, call
      end
    end

    test "two worlds keep separate networks", %{w: w} do
      other = "test_jtms_other_#{System.unique_integer([:positive])}"
      Brain.Test.Singletons.reset_epistemic_stores!([:jtms], world_id: other)

      {:ok, premise} = JTMS.create_premise(w, "Socrates is a man")
      {:ok, _} = JTMS.create_assumption(w, "Birds fly", true)

      # A node of one world is not found in another.
      assert {:ok, _} = JTMS.get_node(w, premise)
      assert {:error, :not_found} = JTMS.get_node(other, premise)
      assert {:error, :not_found} = JTMS.is_in?(other, premise)

      assert JTMS.stats(w).total_nodes == 2
      assert JTMS.stats(other).total_nodes == 0

      # A justification cannot reach into another world's network.
      {:ok, conclusion} = JTMS.create_node(other, "Socrates is mortal")
      assert {:error, {:nodes_not_found, [^premise]}} = JTMS.justify_node(other, [premise], conclusion, "rule")
      assert JTMS.is_in?(other, conclusion) == false

      # Clearing one world leaves the other's network as it was.
      :ok = JTMS.clear(other)
      assert JTMS.stats(w).total_nodes == 2
      assert JTMS.stats(other).total_nodes == 0
    end

    test "a contradiction in one world leaves another consistent and names its world", %{w: w} do
      other = "test_jtms_calm_#{System.unique_integer([:positive])}"
      Brain.Test.Singletons.reset_epistemic_stores!([:jtms], world_id: other)

      {:ok, a} = JTMS.create_assumption(w, "A is guilty", true)
      {:ok, not_a} = JTMS.create_assumption(w, "A is not guilty", true)

      log =
        capture_log([level: :warning], fn ->
          {:ok, _} = JTMS.register_contradiction(w, [a, not_a])
        end)

      assert log =~ "Contradiction detected"
      assert log =~ w

      assert {:error, {:contradiction, _}} = JTMS.check_consistency(w)
      assert [_] = JTMS.get_contradictions(w)
      assert {:ok, :consistent} = JTMS.check_consistency(other)
      assert JTMS.get_contradictions(other) == []
    end
  end

  describe "node creation" do
    test "creates a premise node that is IN", %{w: w} do
      {:ok, node_id} = JTMS.create_premise(w, "Socrates is a man")

      assert JTMS.is_in?(w, node_id) == true
      {:ok, node} = JTMS.get_node(w, node_id)
      assert node.node_type == :premise
      assert node.label == :in
    end

    test "creates an assumption node that is OUT by default", %{w: w} do
      {:ok, node_id} = JTMS.create_assumption(w, "Birds fly")

      assert JTMS.is_in?(w, node_id) == false
      {:ok, node} = JTMS.get_node(w, node_id)
      assert node.node_type == :assumption
      assert node.label == :out
    end

    test "creates an enabled assumption node that is IN", %{w: w} do
      {:ok, node_id} = JTMS.create_assumption(w, "Birds fly", true)

      assert JTMS.is_in?(w, node_id) == true
      {:ok, node} = JTMS.get_node(w, node_id)
      assert node.assumption_enabled == true
    end

    test "creates a contradiction node", %{w: w} do
      {:ok, node_id} = JTMS.create_contradiction(w, "Inconsistent state")

      {:ok, node} = JTMS.get_node(w, node_id)
      assert node.node_type == :contradiction
    end
  end

  describe "justification" do
    test "creates a simple justification", %{w: w} do
      {:ok, premise} = JTMS.create_premise(w, "Socrates is a man")
      {:ok, conclusion} = JTMS.create_node(w, "Socrates is mortal")

      {:ok, _just_id} = JTMS.justify_node(w, [premise], conclusion, "mortality_rule")

      # Conclusion should now be IN
      assert JTMS.is_in?(w, conclusion) == true
    end

    test "conclusion is OUT when premise is OUT", %{w: w} do
      {:ok, assumption} = JTMS.create_assumption(w, "Tweety is a bird")
      {:ok, conclusion} = JTMS.create_node(w, "Tweety flies")

      {:ok, _just_id} = JTMS.justify_node(w, [assumption], conclusion, "bird_flight_rule")

      # Assumption is OUT, so conclusion should be OUT
      assert JTMS.is_in?(w, conclusion) == false
    end

    test "enables assumption propagates to conclusion", %{w: w} do
      {:ok, assumption} = JTMS.create_assumption(w, "Tweety is a bird")
      {:ok, conclusion} = JTMS.create_node(w, "Tweety flies")

      {:ok, _just_id} = JTMS.justify_node(w, [assumption], conclusion, "bird_flight_rule")

      assert JTMS.is_in?(w, conclusion) == false

      # Enable the assumption
      :ok = JTMS.enable_assumption(w, assumption)

      # Now conclusion should be IN
      assert JTMS.is_in?(w, conclusion) == true
    end

    test "retracting assumption propagates to conclusion", %{w: w} do
      {:ok, assumption} = JTMS.create_assumption(w, "Tweety is a bird", true)
      {:ok, conclusion} = JTMS.create_node(w, "Tweety flies")

      {:ok, _just_id} = JTMS.justify_node(w, [assumption], conclusion, "bird_flight_rule")

      assert JTMS.is_in?(w, conclusion) == true

      # Retract the assumption
      :ok = JTMS.retract_assumption(w, assumption)

      # Now conclusion should be OUT
      assert JTMS.is_in?(w, conclusion) == false
    end
  end

  describe "justification with out_list" do
    test "justification with out_list is valid when out nodes are OUT", %{w: w} do
      {:ok, bird} = JTMS.create_assumption(w, "Tweety is a bird", true)
      {:ok, penguin} = JTMS.create_assumption(w, "Tweety is a penguin")
      {:ok, flies} = JTMS.create_node(w, "Tweety flies")

      # Tweety flies if bird AND NOT penguin
      {:ok, _just_id} = JTMS.justify_node(w, [bird], [penguin], flies, "default_flight")

      # penguin is OUT, so justification is valid
      assert JTMS.is_in?(w, flies) == true
    end

    test "justification with out_list is invalid when out node becomes IN", %{w: w} do
      {:ok, bird} = JTMS.create_assumption(w, "Tweety is a bird", true)
      {:ok, penguin} = JTMS.create_assumption(w, "Tweety is a penguin")
      {:ok, flies} = JTMS.create_node(w, "Tweety flies")

      {:ok, _just_id} = JTMS.justify_node(w, [bird], [penguin], flies, "default_flight")

      assert JTMS.is_in?(w, flies) == true

      # Enable penguin - now Tweety doesn't fly
      :ok = JTMS.enable_assumption(w, penguin)

      assert JTMS.is_in?(w, flies) == false
    end
  end

  describe "contradiction detection" do
    test "registers and detects contradiction", %{w: w} do
      {:ok, a} = JTMS.create_assumption(w, "A is guilty", true)
      {:ok, not_a} = JTMS.create_assumption(w, "A is not guilty", true)

      # register_contradiction triggers the handler immediately if both nodes are IN
      # Capture the log to avoid leaking to test output
      log =
        capture_log([level: :warning], fn ->
          {:ok, _contra_id} = JTMS.register_contradiction(w, [a, not_a])
        end)

      # The contradiction handler logs a warning during registration
      assert log =~ "Contradiction detected"

      # check_consistency confirms the contradiction
      {:error, {:contradiction, _}} = JTMS.check_consistency(w)
    end

    test "no contradiction when assumptions don't conflict", %{w: w} do
      {:ok, a} = JTMS.create_assumption(w, "A is guilty", true)
      {:ok, not_a} = JTMS.create_assumption(w, "A is not guilty")

      {:ok, _contra_id} = JTMS.register_contradiction(w, [a, not_a])

      # not_a is OUT, so no contradiction
      {:ok, :consistent} = JTMS.check_consistency(w)
    end
  end

  describe "why_node" do
    test "returns supporting justifications", %{w: w} do
      {:ok, premise1} = JTMS.create_premise(w, "Socrates is a man")
      {:ok, premise2} = JTMS.create_premise(w, "All men are mortal")
      {:ok, conclusion} = JTMS.create_node(w, "Socrates is mortal")

      {:ok, just_id} = JTMS.justify_node(w, [premise1, premise2], conclusion, "syllogism")

      {:ok, result} = JTMS.why_node(w, conclusion)

      assert result.node.id == conclusion
      assert length(result.supporting_justifications) == 1
      assert hd(result.supporting_justifications).id == just_id
    end
  end

  describe "consequences_of" do
    test "returns nodes that depend on the given node", %{w: w} do
      {:ok, premise} = JTMS.create_premise(w, "Socrates is a man")
      {:ok, conclusion1} = JTMS.create_node(w, "Socrates is mortal")
      {:ok, conclusion2} = JTMS.create_node(w, "Socrates will die")

      {:ok, _} = JTMS.justify_node(w, [premise], conclusion1, "rule1")
      {:ok, _} = JTMS.justify_node(w, [conclusion1], conclusion2, "rule2")

      {:ok, consequences} = JTMS.consequences_of(w, premise)

      consequence_ids = Enum.map(consequences, & &1.id)
      assert conclusion1 in consequence_ids
    end
  end

  describe "stats" do
    test "returns network statistics", %{w: w} do
      {:ok, _} = JTMS.create_premise(w, "P1")
      {:ok, _} = JTMS.create_assumption(w, "A1", true)
      {:ok, _} = JTMS.create_node(w, "N1")

      stats = JTMS.stats(w)

      assert stats.total_nodes == 3
      assert stats.in_nodes == 2
      assert stats.out_nodes == 1
    end
  end
end
