defmodule Mix.Tasks.TrainKgLstmTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.Types.Belief
  alias Mix.Tasks.TrainKgLstm

  # The BeliefStore keeps beliefs in GenServer state, which the Sandbox does
  # not roll back, so the test world's beliefs are emptied before each test.
  setup %{graph_world_id: world} do
    ensure_pubsub_started()
    ensure_started(BeliefStore)
    Brain.Test.Singletons.reset_epistemic_stores!([:beliefs, :jtms], world_id: world)
    %{w: world}
  end

  describe "the ConceptNet source" do
    test "a ConceptNet relation trains as its canonical name, never the normalizer's result" do
      assert TrainKgLstm.conceptnet_relation("IsA", %{}) == "IS_A"
      assert TrainKgLstm.conceptnet_relation("PartOf", %{}) == "PART_OF"
      assert TrainKgLstm.conceptnet_relation("IS_A", %{}) == "IS_A"
    end

    test "a relation the scorer does not train on gives none" do
      assert TrainKgLstm.conceptnet_relation("NotARelationAnyoneKnows", %{}) == nil
    end
  end

  describe "the belief source" do
    test "turns a world's beliefs into training triples", %{w: w} do
      {:ok, _} = BeliefStore.add_belief(w, "Paris", :located_in, "France", source: :explicit)
      {:ok, _} = BeliefStore.add_belief(w, :user, :likes, "coffee", source: :explicit)

      triples = TrainKgLstm.load_belief_triples(w)

      assert {"Paris", "located_in", "France"} in triples
      assert {"user", "likes", "coffee"} in triples
      assert length(triples) == 2
    end

    test "trains on the named world's beliefs only", %{w: w} do
      other = "test_kg_lstm_other_#{System.unique_integer([:positive])}"
      Atlas.Graph.ensure_world_graphs!(other)
      Brain.Test.Singletons.reset_epistemic_stores!([:beliefs, :jtms], world_id: other)

      {:ok, _} = BeliefStore.add_belief(w, "Paris", :located_in, "France")
      {:ok, _} = BeliefStore.add_belief(other, "Rome", :located_in, "Italy")

      assert TrainKgLstm.load_belief_triples(w) == [{"Paris", "located_in", "France"}]
      assert TrainKgLstm.load_belief_triples(other) == [{"Rome", "located_in", "Italy"}]
    end

    test "consolidation records and empty parts give no triple" do
      beliefs = [
        Belief.new(:system, :consolidated_knowledge, "summary"),
        Belief.new("Paris", :consolidated_knowledge, "summary"),
        Belief.new(:system, :likes, "x"),
        Belief.new("", :likes, "x"),
        Belief.new("dog", :is_a, ""),
        Belief.new("dog", "IS_A", "animal")
      ]

      assert TrainKgLstm.belief_triples(beliefs) == [{"dog", "IS_A", "animal"}]
    end

    test "including beliefs without --world raises naming the switch" do
      for args <- [[], ["--include-beliefs"]] do
        error = assert_raise Mix.Error, fn -> TrainKgLstm.run(args) end
        assert error.message =~ "--world"
      end
    end
  end
end
