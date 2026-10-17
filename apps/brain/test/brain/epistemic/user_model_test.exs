defmodule Brain.Epistemic.UserModelStoreTest do
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.Epistemic.{BeliefStore, UserModelStore}

  setup %{graph_world_id: world} do
    ensure_pubsub_started()
    ensure_started(UserModelStore)
    ensure_started(BeliefStore)

    # Empty before each test and empty again after: user models live in
    # GenServer state, which the Sandbox does not roll back. Empty is the boot
    # state (0 rows in atlas_user_models).
    Brain.Test.Singletons.reset_epistemic_stores!([:user_models, :beliefs], world_id: world)

    %{user: web_user!(world), other_user: web_user!(world), world: world}
  end

  # A user model is a web user's, keyed by the web user's id. These users exist
  # only inside the test's sandbox transaction.
  defp web_user!(world), do: Atlas.WebUsers.create!(world).id

  # A second world, with the same lifetime as the test's own.
  defp other_world! do
    world = "test_user_models_other_#{System.unique_integer([:positive])}"
    Atlas.Graph.ensure_world_graphs!(world)
    Brain.Test.Singletons.reset_epistemic_stores!([:beliefs], world_id: world)
    world
  end

  # Stops the store and starts it again, so what it holds afterwards is only
  # what it loaded from Atlas.
  defp restart_store! do
    :ok = Supervisor.terminate_child(Brain.Supervisor, UserModelStore)
    {:ok, _pid} = Supervisor.restart_child(Brain.Supervisor, UserModelStore)
    :ok
  end

  defp stored_row(user_id) do
    Atlas.Schemas.UserModel |> Atlas.Schemas.UserModel.for_user(user_id) |> Atlas.Repo.one()
  end

  describe "get_or_create/1" do
    test "creates a new user model", %{user: user} do
      {:ok, model} = UserModelStore.get_or_create(user)

      assert model.user_id == user
      assert model.facts == %{}
      assert model.created_at != nil
    end

    test "writes the created model to Atlas", %{user: user} do
      {:ok, _model} = UserModelStore.get_or_create(user)

      assert %{facts: %{}} = stored_row(user)
    end

    test "returns existing model on second call", %{user: user} do
      {:ok, _model1} = UserModelStore.get_or_create(user)
      UserModelStore.update_fact(user, :name, "Alice", :explicit, 0.9)
      {:ok, model2} = UserModelStore.get_or_create(user)

      assert model2.facts[:name] == "Alice"
    end
  end

  describe "update_fact/5" do
    test "adds a new fact to user model", %{user: user} do
      {:ok, _} = UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.9)

      model = UserModelStore.get(user)

      assert model.facts[:location] == "Seattle"
      assert model.epistemic_bounds[:location] == 0.9
      assert model.provenance_map[:location] == :explicit
    end

    test "updates an existing fact", %{user: user} do
      UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.9)
      UserModelStore.update_fact(user, :location, "Portland", :explicit, 0.95)

      model = UserModelStore.get(user)

      assert model.facts[:location] == "Portland"
      assert model.epistemic_bounds[:location] == 0.95
    end

    test "creates user model if it doesn't exist", %{user: user} do
      UserModelStore.update_fact(user, :name, "Bob", :explicit, 0.85)

      model = UserModelStore.get(user)

      assert model != nil
      assert model.facts[:name] == "Bob"
    end
  end

  describe "get_fact/2" do
    test "returns fact with confidence and provenance", %{user: user} do
      UserModelStore.update_fact(user, :occupation, "Engineer", :inferred, 0.7)

      fact = UserModelStore.get_fact(user, :occupation)

      assert fact.value == "Engineer"
      assert fact.confidence == 0.7
      assert fact.provenance == :inferred
    end

    test "returns nil for non-existent fact", %{user: user} do
      UserModelStore.get_or_create(user)

      fact = UserModelStore.get_fact(user, :nonexistent)

      assert fact == nil
    end

    test "returns nil for non-existent user" do
      fact = UserModelStore.get_fact(Ecto.UUID.generate(), :anything)

      assert fact == nil
    end
  end

  describe "get_facts_with_confidence/2" do
    setup %{user: user} do
      UserModelStore.update_fact(user, :name, "Alice", :explicit, 0.95)
      UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.85)
      UserModelStore.update_fact(user, :occupation, "Developer", :inferred, 0.6)
      UserModelStore.update_fact(user, :hobby, "Hiking", :assumed, 0.3)

      :ok
    end

    test "returns all facts above threshold", %{user: user} do
      facts = UserModelStore.get_facts_with_confidence(user, 0.7)

      assert length(facts) == 2

      keys = Enum.map(facts, & &1.key)
      assert :name in keys
      assert :location in keys
    end

    test "returns all facts with 0.0 threshold", %{user: user} do
      facts = UserModelStore.get_facts_with_confidence(user, 0.0)

      assert length(facts) == 4
    end

    test "returns empty list for high threshold", %{user: user} do
      facts = UserModelStore.get_facts_with_confidence(user, 0.99)

      assert facts == []
    end
  end

  describe "get_epistemic_bounds/1" do
    test "returns confidence map for all facts", %{user: user} do
      UserModelStore.update_fact(user, :a, "1", :explicit, 0.9)
      UserModelStore.update_fact(user, :b, "2", :inferred, 0.7)

      bounds = UserModelStore.get_epistemic_bounds(user)

      assert bounds[:a] == 0.9
      assert bounds[:b] == 0.7
    end

    test "returns empty map for non-existent user" do
      bounds = UserModelStore.get_epistemic_bounds(Ecto.UUID.generate())

      assert bounds == %{}
    end
  end

  describe "record_interaction_pattern/3" do
    test "records interaction patterns", %{user: user} do
      UserModelStore.get_or_create(user)
      UserModelStore.record_interaction_pattern(user, :question_style, %{type: :exploratory})
      UserModelStore.record_interaction_pattern(user, :question_style, %{type: :direct})

      model = UserModelStore.get(user)

      assert length(model.interaction_patterns[:question_style]) == 2
    end
  end

  describe "record_disclosure/3" do
    test "records disclosure history", %{user: user} do
      UserModelStore.get_or_create(user)
      UserModelStore.record_disclosure(user, [:name, :location], %{query: "what do you know"})

      history = UserModelStore.get_disclosure_history(user)

      assert length(history) == 1
      assert hd(history).keys == [:name, :location]
    end

    test "limits history size", %{user: user} do
      UserModelStore.get_or_create(user)

      # Record many disclosures
      for i <- 1..60 do
        UserModelStore.record_disclosure(user, ["fact_#{i}"], %{})
      end

      history = UserModelStore.get_disclosure_history(user, 100)

      # Should be capped at 50
      assert length(history) == 50
    end
  end

  describe "stats/0" do
    test "returns store statistics", %{user: user, other_user: other_user} do
      UserModelStore.update_fact(user, :a, "1", :explicit, 0.9)
      UserModelStore.update_fact(user, :b, "2", :explicit, 0.8)
      UserModelStore.update_fact(other_user, :c, "3", :explicit, 0.7)

      stats = UserModelStore.stats()

      assert stats.total_users == 2
      assert stats.total_facts == 3
    end
  end

  describe "persistence in Atlas" do
    test "a fact is in Atlas when update_fact returns", %{user: user} do
      {:ok, _} = UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.9)

      row = stored_row(user)
      assert row.facts == %{"location" => "Seattle"}
      assert row.epistemic_bounds == %{"location" => 0.9}
      assert row.provenance_map == %{"location" => "explicit"}
    end

    test "the store's models survive a restart", %{user: user, other_user: other_user} do
      UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.9)
      UserModelStore.update_fact(user, :occupation, "Engineer", :inferred, 0.6)
      UserModelStore.update_fact(other_user, :name, "Bob", :explicit, 0.85)
      UserModelStore.record_disclosure(user, [:location], %{query: "what do you know"})

      restart_store!()

      assert %{value: "Seattle", confidence: 0.9, provenance: :explicit} =
               UserModelStore.get_fact(user, :location)

      assert %{value: "Engineer", confidence: 0.6, provenance: :inferred} =
               UserModelStore.get_fact(user, :occupation)

      assert %{value: "Bob"} = UserModelStore.get_fact(other_user, :name)
      assert [_disclosure] = UserModelStore.get_disclosure_history(user)
    end

    test "a user id that is no web user's raises and is not kept" do
      not_a_user = Ecto.UUID.generate()

      assert_raise RuntimeError, ~r/not written to Atlas/, fn ->
        UserModelStore.update_fact(not_a_user, :location, "Seattle", :explicit, 0.9)
      end

      assert UserModelStore.get(not_a_user) == nil

      assert_raise RuntimeError, ~r/not written to Atlas/, fn ->
        UserModelStore.update_fact("anonymous", :location, "Seattle", :explicit, 0.9)
      end

      assert UserModelStore.get("anonymous") == nil
    end

    test "a call with no user raises" do
      for user_id <- [nil, ""] do
        assert_raise ArgumentError, ~r/web user's id/, fn ->
          UserModelStore.update_fact(user_id, :location, "Seattle", :explicit, 0.9)
        end

        assert_raise ArgumentError, ~r/web user's id/, fn -> UserModelStore.get(user_id) end
      end
    end
  end

  # What a turn tells about its user goes into the user's model and the
  # world's beliefs (`Brain.extract_and_store_beliefs/5`); there is no
  # stand-in user for a turn that has none.
  describe "a turn's self-statements" do
    test "are stored for the turn's web user", %{user: user, world: world} do
      assert :ok = Brain.extract_and_store_beliefs(world, "my name is Zed", [], user, "conv_test")

      assert %{value: name, provenance: :explicit} = UserModelStore.get_fact(user, :name)
      assert String.downcase(name) == "zed"
      assert {:ok, [%{predicate: :name, user_id: ^user}]} = BeliefStore.get_beliefs_for_user(world, user)
    end

    test "a turn with no user stores nothing, about no one", %{world: world} do
      beliefs_before = BeliefStore.stats(world).total_beliefs

      assert :ok = Brain.extract_and_store_beliefs(world, "my name is Zed", [], nil, "conv_test")

      assert BeliefStore.stats(world).total_beliefs == beliefs_before
      assert UserModelStore.list_all_users() == {:ok, []}
      assert Atlas.Repo.aggregate(Atlas.Schemas.UserModel, :count) == 0
    end

    test "a user id that is no web user's raises at the user-model write, before any belief",
         %{world: world} do
      not_a_user = "cli_#{System.unique_integer([:positive])}"

      assert_raise RuntimeError, ~r/not written to Atlas/, fn ->
        Brain.extract_and_store_beliefs(world, "my name is Zed", [], not_a_user, "conv_test")
      end

      assert UserModelStore.get(not_a_user) == nil
      assert {:ok, []} = BeliefStore.get_beliefs_for_user(world, not_a_user)
    end

    test "an empty user id raises", %{world: world} do
      assert_raise ArgumentError, ~r/user id or nil/, fn ->
        Brain.extract_and_store_beliefs(world, "my name is Zed", [], "", "conv_test")
      end
    end
  end

  describe "worlds" do
    test "the same user model is used in two worlds", %{user: user, world: world} do
      other = other_world!()

      {:ok, _} = UserModelStore.update_fact(user, :location, "Seattle", :explicit, 0.9)

      # Syncing into one world puts the fact into that world's beliefs only.
      assert {:ok, 1} = UserModelStore.sync_to_beliefs(world, user)
      assert {:ok, [%{predicate: :location, object: "Seattle"}]} = BeliefStore.get_beliefs_for_user(world, user)
      assert {:ok, []} = BeliefStore.get_beliefs_for_user(other, user)

      # The other world reads the same model.
      assert {:ok, 1} = UserModelStore.sync_to_beliefs(other, user)
      assert {:ok, [%{predicate: :location, object: "Seattle"}]} = BeliefStore.get_beliefs_for_user(other, user)
    end

    test "sync_from_beliefs reads the named world's beliefs only", %{user: user, world: world} do
      other = other_world!()
      {:ok, _} = BeliefStore.add_belief(other, :user, :likes, "tea", user_id: user, source: :explicit, confidence: 0.8)

      assert {:ok, 0} = UserModelStore.sync_from_beliefs(world, user)
      assert UserModelStore.get_fact(user, :likes) == nil

      assert {:ok, 1} = UserModelStore.sync_from_beliefs(other, user)
      assert %{value: "tea", provenance: :explicit} = UserModelStore.get_fact(user, :likes)

      # What it learned from that world is the user's in every world, and in Atlas.
      assert stored_row(user).facts == %{"likes" => "tea"}
    end

    test "syncing raises without a world", %{user: user} do
      for world <- [nil, ""] do
        assert_raise ArgumentError, ~r/world/, fn -> UserModelStore.sync_to_beliefs(world, user) end
        assert_raise ArgumentError, ~r/world/, fn -> UserModelStore.sync_from_beliefs(world, user) end
      end
    end
  end
end
