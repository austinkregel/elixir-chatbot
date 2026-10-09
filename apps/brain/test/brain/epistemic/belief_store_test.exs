defmodule Brain.Epistemic.BeliefStoreTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.{BeliefStore, JTMS}
  alias Brain.Epistemic.Types.Belief

  # Every test holds its beliefs in its own world, `w`: the one whose graphs
  # `GraphCase` created. The store is emptied for that world before and after
  # each test, because the BeliefStore keeps beliefs in GenServer state, which
  # the Sandbox does not roll back. Empty is the boot state (0 rows in
  # atlas_beliefs).
  setup %{graph_world_id: world} do
    # Ensure PubSub is started (required for some GenServers)
    ensure_pubsub_started()

    # Start BeliefStore under ExUnit supervision
    ensure_started(BeliefStore)

    Brain.Test.Singletons.reset_epistemic_stores!([:beliefs, :jtms], world_id: world)

    %{w: world}
  end

  defp other_world do
    world = "test_beliefs_other_#{System.unique_integer([:positive])}"
    Atlas.Graph.ensure_world_graphs!(world)
    Brain.Test.Singletons.reset_epistemic_stores!([:beliefs, :jtms], world_id: world)
    world
  end

  describe "the world" do
    test "every belief function raises without a world" do
      belief = Belief.new(:user, :likes, "coffee")

      for call <- [
            fn -> BeliefStore.add_belief(nil, belief) end,
            fn -> BeliefStore.add_belief("", :user, :likes, "coffee") end,
            fn -> BeliefStore.add_belief(:user, :likes, "coffee", []) end,
            fn -> BeliefStore.retract_belief(nil, "id") end,
            fn -> BeliefStore.get_belief(nil, "id") end,
            fn -> BeliefStore.query_beliefs(subject: :user) end,
            fn -> BeliefStore.query_beliefs(nil, []) end,
            fn -> BeliefStore.get_beliefs_for_user(nil, "u") end,
            fn -> BeliefStore.extract_beliefs_from_event(nil, %{confidence: 0.9}) end,
            fn -> BeliefStore.extract_beliefs_from_events(nil, []) end,
            fn -> BeliefStore.update_confidence(nil, "id", 0.5) end,
            fn -> BeliefStore.confirm_belief(nil, "id") end,
            fn -> BeliefStore.add_belief_with_authority(nil, :user, :likes, "x", :mentor) end,
            fn -> BeliefStore.link_to_node(nil, "id", "node") end,
            fn -> BeliefStore.stats(nil) end,
            fn -> BeliefStore.clear(nil) end
          ] do
        assert_raise ArgumentError, ~r/world/, call
      end
    end

    test "two worlds keep separate beliefs", %{w: w} do
      other = other_world()

      {:ok, here} = BeliefStore.add_belief(w, :user, :likes, "coffee", user_id: "u1", source: :explicit)
      {:ok, there} = BeliefStore.add_belief(other, :user, :likes, "tea", user_id: "u1", source: :explicit)

      assert {:ok, %{object: "coffee"}} = BeliefStore.get_belief(w, here)
      assert {:error, :not_found} = BeliefStore.get_belief(other, here)
      assert {:error, :not_found} = BeliefStore.get_belief(w, there)

      assert {:ok, [%{object: "coffee"}]} = BeliefStore.query_beliefs(w, predicate: :likes)
      assert {:ok, [%{object: "tea"}]} = BeliefStore.query_beliefs(other, predicate: :likes)
      assert {:ok, [%{object: "coffee"}]} = BeliefStore.get_beliefs_for_user(w, "u1")

      assert BeliefStore.stats(w).total_beliefs == 1
      assert BeliefStore.stats(other).total_beliefs == 1

      # A belief cannot be confirmed, retracted or changed from another world.
      assert {:error, :not_found} = BeliefStore.confirm_belief(other, here)
      assert {:error, :not_found} = BeliefStore.retract_belief(other, here)
      assert {:error, :not_found} = BeliefStore.update_confidence(other, here, 0.1)
      assert {:ok, %{object: "coffee"}} = BeliefStore.get_belief(w, here)

      # Clearing one world leaves the other's beliefs.
      :ok = BeliefStore.clear(other)
      assert BeliefStore.stats(other).total_beliefs == 0
      assert BeliefStore.stats(w).total_beliefs == 1
    end

    test "a belief's JTMS node lives in the same world's network", %{w: w} do
      other = other_world()

      {:ok, id} = BeliefStore.add_belief(w, :user, :likes, "jazz", source: :explicit)
      {:ok, %{node_id: node_id}} = BeliefStore.get_belief(w, id)

      assert is_binary(node_id)
      assert {:ok, %{node_type: :premise}} = JTMS.get_node(w, node_id)
      assert {:error, :not_found} = JTMS.get_node(other, node_id)
    end

    test "a belief's Atlas row records its world", %{w: w} do
      {:ok, id} = BeliefStore.add_belief(w, :user, :likes, "cocoa", source: :explicit)
      Brain.AtlasIntegration.drain()

      assert %Atlas.Schemas.Belief{world_id: ^w, object: "cocoa"} = Atlas.Repo.get(Atlas.Schemas.Belief, id)
    end
  end

  describe "add_belief/2" do
    test "adds a belief and returns its ID", %{w: w} do
      belief =
        Belief.new(:user, :likes, "coffee",
          confidence: 0.8,
          source: :explicit,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)

      assert is_binary(belief_id)
      assert String.length(belief_id) > 0
    end

    test "adds belief using shorthand", %{w: w} do
      {:ok, belief_id} =
        BeliefStore.add_belief(w, :user, :location, "Seattle",
          confidence: 0.9,
          source: :explicit,
          user_id: "user123"
        )

      assert is_binary(belief_id)
    end
  end

  describe "get_belief/2" do
    test "retrieves an existing belief", %{w: w} do
      belief =
        Belief.new(:user, :name, "Alice",
          confidence: 0.95,
          source: :explicit,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)
      {:ok, retrieved} = BeliefStore.get_belief(w, belief_id)

      assert retrieved.subject == :user
      assert retrieved.predicate == :name
      assert retrieved.object == "Alice"
      assert retrieved.confidence == 0.95
    end

    test "returns error for non-existent belief", %{w: w} do
      {:error, :not_found} = BeliefStore.get_belief(w, "nonexistent")
    end
  end

  describe "retract_belief/2" do
    test "retracts an existing belief", %{w: w} do
      belief = Belief.new(:user, :hobby, "gaming", user_id: "user123")
      {:ok, belief_id} = BeliefStore.add_belief(w, belief)

      :ok = BeliefStore.retract_belief(w, belief_id)

      {:error, :retracted} = BeliefStore.get_belief(w, belief_id)
    end

    test "returns error for non-existent belief", %{w: w} do
      {:error, :not_found} = BeliefStore.retract_belief(w, "nonexistent")
    end
  end

  describe "query_beliefs/2" do
    setup %{w: w} do
      # Add some test beliefs
      BeliefStore.add_belief(w, :user, :location, "Seattle",
        confidence: 0.9,
        source: :explicit,
        user_id: "user1"
      )

      BeliefStore.add_belief(w, :user, :occupation, "Engineer",
        confidence: 0.7,
        source: :inferred,
        user_id: "user1"
      )

      BeliefStore.add_belief(w, :user, :location, "Portland",
        confidence: 0.85,
        source: :explicit,
        user_id: "user2"
      )

      BeliefStore.add_belief(w, :world, :weather, "sunny",
        confidence: 0.6,
        source: :inferred
      )

      :ok
    end

    test "queries by subject", %{w: w} do
      {:ok, beliefs} = BeliefStore.query_beliefs(w, subject: :user)

      assert length(beliefs) == 3
      assert Enum.all?(beliefs, &(&1.subject == :user))
    end

    test "queries by predicate", %{w: w} do
      {:ok, beliefs} = BeliefStore.query_beliefs(w, predicate: :location)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.predicate == :location))
    end

    test "queries by user_id", %{w: w} do
      {:ok, beliefs} = BeliefStore.query_beliefs(w, user_id: "user1")

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.user_id == "user1"))
    end

    test "queries by min_confidence", %{w: w} do
      {:ok, beliefs} = BeliefStore.query_beliefs(w, min_confidence: 0.8)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.confidence >= 0.8))
    end

    test "queries by source", %{w: w} do
      {:ok, beliefs} = BeliefStore.query_beliefs(w, source: :explicit)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.source == :explicit))
    end

    test "combines multiple filters", %{w: w} do
      {:ok, beliefs} =
        BeliefStore.query_beliefs(w,
          subject: :user,
          source: :explicit,
          min_confidence: 0.85
        )

      assert length(beliefs) == 2
    end
  end

  describe "update_confidence/4" do
    test "updates belief confidence", %{w: w} do
      belief =
        Belief.new(:user, :skill, "cooking",
          confidence: 0.5,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)
      {:ok, updated} = BeliefStore.update_confidence(w, belief_id, 0.8)

      assert updated.confidence == 0.8
    end

    test "confirms belief when option is set", %{w: w} do
      belief =
        Belief.new(:user, :skill, "coding",
          confidence: 0.6,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)
      {:ok, updated} = BeliefStore.update_confidence(w, belief_id, 0.9, confirm: true)

      assert updated.confidence == 0.9
      assert updated.last_confirmed != nil
    end

    test "clamps confidence to valid range", %{w: w} do
      belief = Belief.new(:user, :skill, "flying", confidence: 0.5)
      {:ok, belief_id} = BeliefStore.add_belief(w, belief)

      {:ok, updated} = BeliefStore.update_confidence(w, belief_id, 1.5)
      assert updated.confidence == 1.0

      {:ok, updated2} = BeliefStore.update_confidence(w, belief_id, -0.5)
      assert updated2.confidence == 0.0
    end
  end

  describe "confirm_belief/3" do
    test "confirms and boosts confidence", %{w: w} do
      belief =
        Belief.new(:user, :preference, "dark mode",
          confidence: 0.7,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)
      {:ok, confirmed} = BeliefStore.confirm_belief(w, belief_id, 0.1)

      # Use approximate comparison for floating point
      assert_in_delta confirmed.confidence, 0.8, 0.001
      assert confirmed.last_confirmed != nil
    end

    test "caps confidence at 1.0", %{w: w} do
      belief =
        Belief.new(:user, :preference, "cats",
          confidence: 0.95,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(w, belief)
      {:ok, confirmed} = BeliefStore.confirm_belief(w, belief_id, 0.2)

      assert confirmed.confidence == 1.0
    end
  end

  describe "get_beliefs_for_user/2" do
    test "returns all beliefs for a user", %{w: w} do
      BeliefStore.add_belief(w, :user, :name, "Bob", user_id: "bob")
      BeliefStore.add_belief(w, :user, :age, "30", user_id: "bob")
      BeliefStore.add_belief(w, :user, :name, "Alice", user_id: "alice")

      {:ok, bob_beliefs} = BeliefStore.get_beliefs_for_user(w, "bob")

      assert length(bob_beliefs) == 2
      assert Enum.all?(bob_beliefs, &(&1.user_id == "bob"))
    end
  end

  describe "stats/1" do
    test "returns store statistics", %{w: w} do
      BeliefStore.add_belief(w, :user, :a, "1", user_id: "u1")
      BeliefStore.add_belief(w, :user, :b, "2", user_id: "u1")
      BeliefStore.add_belief(w, :world, :c, "3")

      stats = BeliefStore.stats(w)

      assert stats.total_beliefs == 3
      assert stats.active_beliefs == 3
      assert stats.unique_subjects == 2
    end
  end

  describe "graph integration" do
    test "beliefs extracted from an event are written to the world's knowledge_graph", %{w: w} do
      event = %{
        action: %{lemma: "want", tense: :present},
        actor: %{text: "I", type: "pronoun"},
        object: %{text: "coffee", type: "noun"},
        confidence: 0.9
      }

      {:ok, [_id]} = BeliefStore.extract_beliefs_from_event(w, event, "u1")
      Brain.AtlasIntegration.drain()

      assert {:ok, 1} = count_nodes(Atlas.Graph.world_graph("knowledge_graph", w), "Belief")
    end
  end
end
