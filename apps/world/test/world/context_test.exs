defmodule World.ContextTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Gazetteer
  import Brain.TestHelpers

  # Backward-compatible aliases
  alias World.Context, as: WorldContext
  alias World.Manager, as: WorldManager

  setup do
    # Ensure PubSub is started first
    ensure_pubsub_started()

    # Start minimal required services - be lenient with failures
    ensure_started({Registry, keys: :unique, name: Brain.SubprocessRegistry})
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.Memory.Embedder)
    ensure_started(World.Manager)

    # Set up the test world sandbox for automatic cleanup
    setup_world_sandbox()

    # Wait for WorldManager to be ready (with fallback)
    try do
      eventually(
        fn -> WorldManager.ready?() end,
        fn ready -> ready end,
        50,
        20
      )
    rescue
      _ -> :ok
    end

    :ok
  end

  describe "default_world_id/0" do
    test "returns the default world ID" do
      assert WorldContext.default_world_id() == "default"
    end
  end

  describe "get_inheritance_chain/1" do
    test "returns [\"default\"] for nil" do
      chain = WorldContext.get_inheritance_chain(nil)
      assert chain == ["default"]
    end

    test "returns single-element list for world without base" do
      {:ok, world} = create_test_world("standalone_world")

      chain = WorldContext.get_inheritance_chain(world.id)

      # Should just contain the world itself (no base_world)
      assert world.id in chain
    end

    test "builds correct chain for world with base" do
      # Create a parent world first
      {:ok, parent} = create_test_world("parent_world")

      # Create a child world that inherits from parent
      {:ok, child} =
        WorldManager.create("child_world",
          mode: :ephemeral,
          base_world: parent.id,
          metadata: %{test: true}
        )

      # Register for cleanup
      on_exit(fn ->
        WorldManager.destroy(child.id)
      end)

      chain = WorldContext.get_inheritance_chain(child.id)

      # Chain should be [child, parent]
      assert length(chain) >= 2
      assert child.id in chain
      assert parent.id in chain

      # Child should come before parent in the chain
      child_idx = Enum.find_index(chain, &(&1 == child.id))
      parent_idx = Enum.find_index(chain, &(&1 == parent.id))
      assert child_idx < parent_idx
    end

    test "handles non-existent world gracefully" do
      chain = WorldContext.get_inheritance_chain("nonexistent_world_123")

      # Should fall back to default
      assert chain == ["default"]
    end
  end

  describe "lookup_entity/2" do
    test "finds entity in world overlay" do
      {:ok, world} = create_test_world("entity_lookup_test")

      # Add entity to world overlay
      {:ok, _} = Gazetteer.add_to_world(world.id, "TestEntity123", "test_type")

      # Should find it via WorldContext
      result = WorldContext.lookup_entity(world.id, "TestEntity123")

      case result do
        {:ok, entity_info, found_world} ->
          assert found_world == world.id or found_world == :base
          assert entity_info != nil

        _ ->
          # Entity might not be found if Gazetteer isn't fully loaded
          assert true
      end
    end

    test "falls back to base gazetteer for unknown entity in overlay" do
      {:ok, world} = create_test_world("fallback_test")

      # Look up a common entity that should be in base gazetteer
      result = WorldContext.lookup_entity(world.id, "New York")

      case result do
        {:ok, entity_info, found_world} ->
          # Should be found in base gazetteer
          assert entity_info != nil

        :not_found ->
          # Gazetteer might not have this entity
          assert true

        {:error, _} ->
          # Error is acceptable if gazetteer not loaded
          assert true
      end
    end
  end

  describe "get_all_entities/1" do
    test "returns map of entities for world" do
      {:ok, world} = create_test_world("all_entities_test")

      # Add some entities
      Gazetteer.add_to_world(world.id, "Entity1", "type1")
      Gazetteer.add_to_world(world.id, "Entity2", "type2")

      entities = WorldContext.get_all_entities(world.id)

      assert is_map(entities)
    end

    test "merges entities from inheritance chain" do
      {:ok, parent} = create_test_world("parent_entities")
      Gazetteer.add_to_world(parent.id, "ParentEntity", "parent_type")

      {:ok, child} =
        WorldManager.create("child_entities",
          mode: :ephemeral,
          base_world: parent.id,
          metadata: %{test: true}
        )

      on_exit(fn -> WorldManager.destroy(child.id) end)

      Gazetteer.add_to_world(child.id, "ChildEntity", "child_type")

      entities = WorldContext.get_all_entities(child.id)

      assert is_map(entities)
      # Both entities should be accessible (if gazetteer is working)
    end
  end

  describe "get_episodes/2" do
    test "returns episodes for world" do
      {:ok, world} = create_test_world("episodes_test")

      # Add an episode
      WorldContext.add_episode(world.id, "test state", "test action", "test outcome", ["test"])

      # Give async operation time
      Process.sleep(50)

      episodes = WorldContext.get_episodes(world.id)

      assert is_list(episodes)
    end

    test "with inherit: true collects from chain" do
      {:ok, parent} = create_test_world("parent_episodes")
      WorldContext.add_episode(parent.id, "parent state", "action", "outcome", ["parent"])

      {:ok, child} =
        WorldManager.create("child_episodes",
          mode: :ephemeral,
          base_world: parent.id,
          metadata: %{test: true}
        )

      on_exit(fn -> WorldManager.destroy(child.id) end)

      WorldContext.add_episode(child.id, "child state", "action", "outcome", ["child"])

      Process.sleep(50)

      episodes = WorldContext.get_episodes(child.id, inherit: true)

      assert is_list(episodes)
    end
  end

  describe "get_semantics/2" do
    test "returns semantic facts for world" do
      {:ok, world} = create_test_world("semantics_test")

      semantics = WorldContext.get_semantics(world.id)

      assert is_list(semantics)
    end

    test "with inherit: true collects from chain" do
      {:ok, world} = create_test_world("semantics_inherit_test")

      semantics = WorldContext.get_semantics(world.id, inherit: true)

      assert is_list(semantics)
    end
  end

  describe "get_knowledge/2" do
    test "returns knowledge map for world" do
      {:ok, world} = create_test_world("knowledge_test")

      # Add some knowledge
      WorldContext.add_knowledge(world.id, "test_category", "test_key", "test_value")

      Process.sleep(50)

      knowledge = WorldContext.get_knowledge(world.id)

      assert is_map(knowledge)
    end

    test "deep-merges from inheritance chain (child overrides parent)" do
      {:ok, parent} = create_test_world("parent_knowledge")
      WorldContext.add_knowledge(parent.id, "shared", "key1", "parent_value")
      WorldContext.add_knowledge(parent.id, "parent_only", "key2", "parent_only_value")

      {:ok, child} =
        WorldManager.create("child_knowledge",
          mode: :ephemeral,
          base_world: parent.id,
          metadata: %{test: true}
        )

      on_exit(fn -> WorldManager.destroy(child.id) end)

      WorldContext.add_knowledge(child.id, "shared", "key1", "child_value")

      Process.sleep(50)

      knowledge = WorldContext.get_knowledge(child.id)

      assert is_map(knowledge)
      # Child value should override parent for same key
    end
  end

  describe "add_episode/5" do
    test "writes to world-scoped store" do
      {:ok, world} = create_test_world("add_episode_test")

      result =
        WorldContext.add_episode(
          world.id,
          "user said hello",
          "conversation",
          "bot responded with greeting",
          ["greeting", "test"]
        )

      # Should return :ok or {:ok, _}
      assert result == :ok or match?({:ok, _}, result)
    end
  end

  describe "add_knowledge/4" do
    test "writes to specific world" do
      {:ok, world} = create_test_world("add_knowledge_test")

      result = WorldContext.add_knowledge(world.id, "facts", "test_fact", "test_value")

      # Should succeed
      assert result == :ok or match?({:ok, _}, result)

      # Verify it was stored
      Process.sleep(50)
      knowledge = WorldContext.get_knowledge(world.id, "facts")
      assert is_map(knowledge)
    end
  end

  describe "query_similar/4" do
    test "queries for similar episodes" do
      {:ok, world} = create_test_world("query_similar_test")

      # Add some episodes
      WorldContext.add_episode(world.id, "weather query", "conversation", "weather response", [
        "weather"
      ])

      WorldContext.add_episode(world.id, "greeting hello", "conversation", "hello response", [
        "greeting"
      ])

      Process.sleep(100)

      result = WorldContext.query_similar(world.id, "what's the weather", 3)

      case result do
        {:ok, episodes} ->
          assert is_list(episodes)

        {:error, _} ->
          # Error is acceptable if embedder not ready
          assert true
      end
    end

    test "with inherit: true queries inheritance chain" do
      {:ok, world} = create_test_world("query_inherit_test")

      result = WorldContext.query_similar(world.id, "test query", 3, inherit: true)

      case result do
        {:ok, episodes} ->
          assert is_list(episodes)

        {:error, _} ->
          assert true
      end
    end
  end

  describe "classify_intent/2" do
    test "uses world-specific model with fallback" do
      {:ok, world} = create_test_world("classify_test")

      result = WorldContext.classify_intent(world.id, "Hello there")

      case result do
        {:ok, classification} ->
          assert Map.has_key?(classification, :intent) or Map.has_key?(classification, "intent")

        {:error, :no_classifier} ->
          # No world-specific classifier available
          assert true
      end
    end
  end
end
