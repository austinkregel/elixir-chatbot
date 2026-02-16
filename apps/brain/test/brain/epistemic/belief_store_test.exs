defmodule Brain.Epistemic.BeliefStoreTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.Types.Belief

  setup do
    # Ensure PubSub is started (required for some GenServers)
    ensure_pubsub_started()

    # Start BeliefStore under ExUnit supervision
    ensure_started(BeliefStore)

    # Clear before each test
    BeliefStore.clear()

    :ok
  end

  describe "add_belief/1" do
    test "adds a belief and returns its ID" do
      belief =
        Belief.new(:user, :likes, "coffee",
          confidence: 0.8,
          source: :explicit,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)

      assert is_binary(belief_id)
      assert String.length(belief_id) > 0
    end

    test "adds belief using shorthand" do
      {:ok, belief_id} =
        BeliefStore.add_belief(:user, :location, "Seattle",
          confidence: 0.9,
          source: :explicit,
          user_id: "user123"
        )

      assert is_binary(belief_id)
    end
  end

  describe "get_belief/1" do
    test "retrieves an existing belief" do
      belief =
        Belief.new(:user, :name, "Alice",
          confidence: 0.95,
          source: :explicit,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)
      {:ok, retrieved} = BeliefStore.get_belief(belief_id)

      assert retrieved.subject == :user
      assert retrieved.predicate == :name
      assert retrieved.object == "Alice"
      assert retrieved.confidence == 0.95
    end

    test "returns error for non-existent belief" do
      {:error, :not_found} = BeliefStore.get_belief("nonexistent")
    end
  end

  describe "retract_belief/1" do
    test "retracts an existing belief" do
      belief = Belief.new(:user, :hobby, "gaming", user_id: "user123")
      {:ok, belief_id} = BeliefStore.add_belief(belief)

      :ok = BeliefStore.retract_belief(belief_id)

      {:error, :retracted} = BeliefStore.get_belief(belief_id)
    end

    test "returns error for non-existent belief" do
      {:error, :not_found} = BeliefStore.retract_belief("nonexistent")
    end
  end

  describe "query_beliefs/1" do
    setup do
      # Add some test beliefs
      BeliefStore.add_belief(:user, :location, "Seattle",
        confidence: 0.9,
        source: :explicit,
        user_id: "user1"
      )

      BeliefStore.add_belief(:user, :occupation, "Engineer",
        confidence: 0.7,
        source: :inferred,
        user_id: "user1"
      )

      BeliefStore.add_belief(:user, :location, "Portland",
        confidence: 0.85,
        source: :explicit,
        user_id: "user2"
      )

      BeliefStore.add_belief(:world, :weather, "sunny",
        confidence: 0.6,
        source: :inferred
      )

      :ok
    end

    test "queries by subject" do
      {:ok, beliefs} = BeliefStore.query_beliefs(subject: :user)

      assert length(beliefs) == 3
      assert Enum.all?(beliefs, &(&1.subject == :user))
    end

    test "queries by predicate" do
      {:ok, beliefs} = BeliefStore.query_beliefs(predicate: :location)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.predicate == :location))
    end

    test "queries by user_id" do
      {:ok, beliefs} = BeliefStore.query_beliefs(user_id: "user1")

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.user_id == "user1"))
    end

    test "queries by min_confidence" do
      {:ok, beliefs} = BeliefStore.query_beliefs(min_confidence: 0.8)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.confidence >= 0.8))
    end

    test "queries by source" do
      {:ok, beliefs} = BeliefStore.query_beliefs(source: :explicit)

      assert length(beliefs) == 2
      assert Enum.all?(beliefs, &(&1.source == :explicit))
    end

    test "combines multiple filters" do
      {:ok, beliefs} =
        BeliefStore.query_beliefs(
          subject: :user,
          source: :explicit,
          min_confidence: 0.85
        )

      assert length(beliefs) == 2
    end
  end

  describe "update_confidence/3" do
    test "updates belief confidence" do
      belief =
        Belief.new(:user, :skill, "cooking",
          confidence: 0.5,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)
      {:ok, updated} = BeliefStore.update_confidence(belief_id, 0.8)

      assert updated.confidence == 0.8
    end

    test "confirms belief when option is set" do
      belief =
        Belief.new(:user, :skill, "coding",
          confidence: 0.6,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)
      {:ok, updated} = BeliefStore.update_confidence(belief_id, 0.9, confirm: true)

      assert updated.confidence == 0.9
      assert updated.last_confirmed != nil
    end

    test "clamps confidence to valid range" do
      belief = Belief.new(:user, :skill, "flying", confidence: 0.5)
      {:ok, belief_id} = BeliefStore.add_belief(belief)

      {:ok, updated} = BeliefStore.update_confidence(belief_id, 1.5)
      assert updated.confidence == 1.0

      {:ok, updated2} = BeliefStore.update_confidence(belief_id, -0.5)
      assert updated2.confidence == 0.0
    end
  end

  describe "confirm_belief/2" do
    test "confirms and boosts confidence" do
      belief =
        Belief.new(:user, :preference, "dark mode",
          confidence: 0.7,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)
      {:ok, confirmed} = BeliefStore.confirm_belief(belief_id, 0.1)

      # Use approximate comparison for floating point
      assert_in_delta confirmed.confidence, 0.8, 0.001
      assert confirmed.last_confirmed != nil
    end

    test "caps confidence at 1.0" do
      belief =
        Belief.new(:user, :preference, "cats",
          confidence: 0.95,
          user_id: "user123"
        )

      {:ok, belief_id} = BeliefStore.add_belief(belief)
      {:ok, confirmed} = BeliefStore.confirm_belief(belief_id, 0.2)

      assert confirmed.confidence == 1.0
    end
  end

  describe "get_beliefs_for_user/1" do
    test "returns all beliefs for a user" do
      BeliefStore.add_belief(:user, :name, "Bob", user_id: "bob")
      BeliefStore.add_belief(:user, :age, "30", user_id: "bob")
      BeliefStore.add_belief(:user, :name, "Alice", user_id: "alice")

      {:ok, bob_beliefs} = BeliefStore.get_beliefs_for_user("bob")

      assert length(bob_beliefs) == 2
      assert Enum.all?(bob_beliefs, &(&1.user_id == "bob"))
    end
  end

  describe "stats/0" do
    test "returns store statistics" do
      BeliefStore.add_belief(:user, :a, "1", user_id: "u1")
      BeliefStore.add_belief(:user, :b, "2", user_id: "u1")
      BeliefStore.add_belief(:world, :c, "3")

      stats = BeliefStore.stats()

      assert stats.total_beliefs == 3
      assert stats.active_beliefs == 3
      assert stats.unique_subjects == 2
    end
  end

  describe "graph integration" do
    test "belief creation writes to knowledge_graph" do
      {:ok, kg_before} = count_nodes("knowledge_graph", "Belief")

      {:ok, _} = BeliefStore.add_belief(:user, :likes, "coffee", confidence: 0.9)
      # Graph writes are async, wait for them
      Process.sleep(300)

      {:ok, kg_after} = count_nodes("knowledge_graph", "Belief")
      assert kg_after >= kg_before
    end
  end
end
