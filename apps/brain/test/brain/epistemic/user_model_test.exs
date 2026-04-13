defmodule Brain.Epistemic.UserModelStoreTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.UserModelStore

  setup _context do
    ensure_pubsub_started()
    ensure_started(UserModelStore)
    UserModelStore.clear_all()

    :ok
  end

  describe "get_or_create/1" do
    test "creates a new user model" do
      {:ok, model} = UserModelStore.get_or_create("user123")

      assert model.user_id == "user123"
      assert model.facts == %{}
      assert model.created_at != nil
    end

    test "returns existing model on second call" do
      {:ok, _model1} = UserModelStore.get_or_create("user123")
      UserModelStore.update_fact("user123", :name, "Alice", :explicit, 0.9)
      {:ok, model2} = UserModelStore.get_or_create("user123")

      assert model2.facts[:name] == "Alice"
    end
  end

  describe "update_fact/5" do
    test "adds a new fact to user model" do
      {:ok, _} = UserModelStore.update_fact("user1", :location, "Seattle", :explicit, 0.9)

      model = UserModelStore.get("user1")

      assert model.facts[:location] == "Seattle"
      assert model.epistemic_bounds[:location] == 0.9
      assert model.provenance_map[:location] == :explicit
    end

    test "updates an existing fact" do
      UserModelStore.update_fact("user1", :location, "Seattle", :explicit, 0.9)
      UserModelStore.update_fact("user1", :location, "Portland", :explicit, 0.95)

      model = UserModelStore.get("user1")

      assert model.facts[:location] == "Portland"
      assert model.epistemic_bounds[:location] == 0.95
    end

    test "creates user model if it doesn't exist" do
      UserModelStore.update_fact("newuser", :name, "Bob", :explicit, 0.85)

      model = UserModelStore.get("newuser")

      assert model != nil
      assert model.facts[:name] == "Bob"
    end
  end

  describe "get_fact/2" do
    test "returns fact with confidence and provenance" do
      UserModelStore.update_fact("user1", :occupation, "Engineer", :inferred, 0.7)

      fact = UserModelStore.get_fact("user1", :occupation)

      assert fact.value == "Engineer"
      assert fact.confidence == 0.7
      assert fact.provenance == :inferred
    end

    test "returns nil for non-existent fact" do
      UserModelStore.get_or_create("user1")

      fact = UserModelStore.get_fact("user1", :nonexistent)

      assert fact == nil
    end

    test "returns nil for non-existent user" do
      fact = UserModelStore.get_fact("nonexistent", :anything)

      assert fact == nil
    end
  end

  describe "get_facts_with_confidence/2" do
    setup do
      UserModelStore.update_fact("user1", :name, "Alice", :explicit, 0.95)
      UserModelStore.update_fact("user1", :location, "Seattle", :explicit, 0.85)
      UserModelStore.update_fact("user1", :occupation, "Developer", :inferred, 0.6)
      UserModelStore.update_fact("user1", :hobby, "Hiking", :assumed, 0.3)

      :ok
    end

    test "returns all facts above threshold" do
      facts = UserModelStore.get_facts_with_confidence("user1", 0.7)

      assert length(facts) == 2

      keys = Enum.map(facts, & &1.key)
      assert :name in keys
      assert :location in keys
    end

    test "returns all facts with 0.0 threshold" do
      facts = UserModelStore.get_facts_with_confidence("user1", 0.0)

      assert length(facts) == 4
    end

    test "returns empty list for high threshold" do
      facts = UserModelStore.get_facts_with_confidence("user1", 0.99)

      assert facts == []
    end
  end

  describe "get_epistemic_bounds/1" do
    test "returns confidence map for all facts" do
      UserModelStore.update_fact("user1", :a, "1", :explicit, 0.9)
      UserModelStore.update_fact("user1", :b, "2", :inferred, 0.7)

      bounds = UserModelStore.get_epistemic_bounds("user1")

      assert bounds[:a] == 0.9
      assert bounds[:b] == 0.7
    end

    test "returns empty map for non-existent user" do
      bounds = UserModelStore.get_epistemic_bounds("nonexistent")

      assert bounds == %{}
    end
  end

  describe "record_interaction_pattern/3" do
    test "records interaction patterns" do
      UserModelStore.get_or_create("user1")
      UserModelStore.record_interaction_pattern("user1", :question_style, %{type: :exploratory})
      UserModelStore.record_interaction_pattern("user1", :question_style, %{type: :direct})

      model = UserModelStore.get("user1")

      assert length(model.interaction_patterns[:question_style]) == 2
    end
  end

  describe "record_disclosure/3" do
    test "records disclosure history" do
      UserModelStore.get_or_create("user1")
      UserModelStore.record_disclosure("user1", [:name, :location], %{query: "what do you know"})

      history = UserModelStore.get_disclosure_history("user1")

      assert length(history) == 1
      assert hd(history).keys == [:name, :location]
    end

    test "limits history size" do
      UserModelStore.get_or_create("user1")

      # Record many disclosures
      for i <- 1..60 do
        UserModelStore.record_disclosure("user1", ["fact_#{i}"], %{})
      end

      history = UserModelStore.get_disclosure_history("user1", 100)

      # Should be capped at 50
      assert length(history) == 50
    end
  end

  describe "stats/0" do
    test "returns store statistics" do
      UserModelStore.update_fact("user1", :a, "1", :explicit, 0.9)
      UserModelStore.update_fact("user1", :b, "2", :explicit, 0.8)
      UserModelStore.update_fact("user2", :c, "3", :explicit, 0.7)

      stats = UserModelStore.stats()

      assert stats.total_users == 2
      assert stats.total_facts == 3
    end
  end
end
