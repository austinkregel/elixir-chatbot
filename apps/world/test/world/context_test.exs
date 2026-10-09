defmodule World.ContextTest do
  # These tests write episodes through `Brain.Memory.Store`, which needs a
  # checked-out sandbox connection to reach Atlas (`World.AtlasCase`). Without
  # one the write fails; it used to be invisible because the store replied
  # `{:ok, id}` for a write it knew had failed, so "writes to world-scoped
  # store" passed while writing nothing.
  use World.AtlasCase, async: false

  alias Brain.ML.Gazetteer
  import Brain.TestHelpers

  # Backward-compatible aliases
  alias World.Context, as: WorldContext
  alias World.Manager, as: WorldManager

  setup do
    # Ensure PubSub is started first
    ensure_pubsub_started()

    ensure_started({Registry, keys: :unique, name: Brain.SubprocessRegistry})
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.Memory.Embedder)
    ensure_started(World.Manager)

    # Set up the test world sandbox for automatic cleanup
    setup_world_sandbox()

    eventually(
      fn -> WorldManager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    :ok
  end

  defp unknown_world_id, do: "test_unknown_world_#{System.unique_integer([:positive])}"

  defp unique_name(prefix), do: "#{prefix}#{System.unique_integer([:positive])}"

  # Every read names its world: there is no implicit one, and a world the
  # manager does not know is an error, not an empty result.
  @reads [
    {"resolve/3", &__MODULE__.read_resolve/1},
    {"lookup_entity/2", &__MODULE__.read_lookup_entity/1},
    {"get_all_entities/1", &WorldContext.get_all_entities/1},
    {"get_episodes/1", &WorldContext.get_episodes/1},
    {"get_semantics/1", &WorldContext.get_semantics/1},
    {"get_knowledge/2", &WorldContext.get_knowledge/1},
    {"classify_intent/2", &__MODULE__.read_classify_intent/1},
    {"query_similar/3", &__MODULE__.read_query_similar/1}
  ]

  def read_resolve(world_id), do: WorldContext.resolve(world_id, :entity, fn _ -> {:ok, :found} end)
  def read_lookup_entity(world_id), do: WorldContext.lookup_entity(world_id, "anything")
  def read_classify_intent(world_id), do: WorldContext.classify_intent(world_id, "Hello there")
  def read_query_similar(world_id), do: WorldContext.query_similar(world_id, "anything", 3)

  describe "a read with no world" do
    for {label, read} <- @reads do
      @read read
      test "#{label} raises for a nil world" do
        assert_raise ArgumentError, ~r/needs a world id, got nil/, fn -> @read.(nil) end
      end
    end
  end

  describe "a read in a world the manager does not know" do
    for {label, read} <- @reads do
      @read read
      test "#{label} raises naming the world" do
        world_id = unknown_world_id()

        assert_raise ArgumentError, ~r/world "#{world_id}" is not known/, fn ->
          @read.(world_id)
        end
      end
    end

    test "is not read as the world's base world or as \"default\"" do
      {:ok, parent} = create_test_world("unknown_base")
      world_id = unknown_world_id()

      assert_raise ArgumentError, ~r/is not known/, fn ->
        WorldContext.resolve(world_id, :entity, fn wid ->
          send(self(), {:looked_up, wid})
          {:ok, :found}
        end)
      end

      refute_received {:looked_up, _}
      assert {:ok, _} = WorldManager.get(parent.id)
    end
  end

  describe "resolve/3" do
    test "looks only in the world it is given" do
      {:ok, parent} = create_test_world("resolve_parent")
      {:ok, child} = create_test_world("resolve_child", base_world: parent.id)

      assert {:error, :not_found} =
               WorldContext.resolve(child.id, :knowledge, fn wid ->
                 send(self(), {:looked_up, wid})
                 {:error, :not_found}
               end)

      child_id = child.id
      assert_received {:looked_up, ^child_id}
      refute_received {:looked_up, _}
    end

    test "returns the result with the world it was found in" do
      {:ok, world} = create_test_world("resolve_found")

      assert {:ok, :found, world.id} ==
               WorldContext.resolve(world.id, :entity, fn _ -> {:ok, :found} end)
    end
  end

  describe "lookup_entity/2" do
    test "finds entity in world overlay" do
      {:ok, world} = create_test_world("entity_lookup_test")
      name = unique_name("TestEntity")

      {:ok, _} = Gazetteer.add_to_world(world.id, name, "test_type")

      world_id = world.id
      assert {:ok, [info], ^world_id} = WorldContext.lookup_entity(world.id, name)
      assert info.entity_type == "test_type"
      assert info.world_id == world.id
    end

    test "does not find an entity that only the base world holds" do
      {:ok, parent} = create_test_world("lookup_parent")
      name = unique_name("ParentOnlyEntity")
      {:ok, _} = Gazetteer.add_to_world(parent.id, name, "parent_type")

      {:ok, child} = create_test_world("lookup_child", base_world: parent.id)

      assert {:ok, _, _} = WorldContext.lookup_entity(parent.id, name)
      assert WorldContext.lookup_entity(child.id, name) == :not_found
    end
  end

  describe "get_all_entities/1" do
    test "returns the entities in the world's overlay" do
      {:ok, world} = create_test_world("all_entities_test")

      {:ok, key1} = Gazetteer.add_to_world(world.id, unique_name("Entity"), "type1")
      {:ok, key2} = Gazetteer.add_to_world(world.id, unique_name("Entity"), "type2")

      entities = WorldContext.get_all_entities(world.id)

      assert [%{entity_type: "type1"}] = Map.fetch!(entities, key1)
      assert [%{entity_type: "type2"}] = Map.fetch!(entities, key2)
    end

    test "does not include the base world's entities" do
      {:ok, parent} = create_test_world("parent_entities")
      {:ok, parent_key} = Gazetteer.add_to_world(parent.id, unique_name("ParentEntity"), "parent_type")

      {:ok, child} = create_test_world("child_entities", base_world: parent.id)

      {:ok, child_key} = Gazetteer.add_to_world(child.id, unique_name("ChildEntity"), "child_type")

      entities = WorldContext.get_all_entities(child.id)

      assert Map.has_key?(entities, child_key)
      refute Map.has_key?(entities, parent_key)
    end
  end

  describe "get_episodes/1" do
    test "returns episodes for world" do
      {:ok, world} = create_test_world("episodes_test")

      {:ok, id} =
        WorldContext.add_episode(world.id, "test state", "test action", "test outcome", ["test"])

      episodes = WorldContext.get_episodes(world.id)

      assert [%{id: ^id, state: "test state"}] = episodes
    end

    test "does not include the base world's episodes" do
      {:ok, parent} = create_test_world("parent_episodes")
      {:ok, _} = WorldContext.add_episode(parent.id, "parent state", "action", "outcome", ["parent"])

      {:ok, child} = create_test_world("child_episodes", base_world: parent.id)

      {:ok, _} = WorldContext.add_episode(child.id, "child state", "action", "outcome", ["child"])

      states = child.id |> WorldContext.get_episodes() |> Enum.map(& &1.state)

      assert states == ["child state"]
    end
  end

  describe "get_semantics/1" do
    test "returns no semantic facts for a new world" do
      {:ok, world} = create_test_world("semantics_test")

      assert WorldContext.get_semantics(world.id) == []
    end
  end

  describe "get_knowledge/2" do
    test "returns knowledge map for world" do
      {:ok, world} = create_test_world("knowledge_test")

      :ok = WorldContext.add_knowledge(world.id, "test_category", "test_key", "test_value")

      assert WorldContext.get_knowledge(world.id) == %{
               "test_category" => %{"test_key" => "test_value"}
             }
    end

    test "does not include the base world's knowledge" do
      {:ok, parent} = create_test_world("parent_knowledge")
      :ok = WorldContext.add_knowledge(parent.id, "shared", "key1", "parent_value")
      :ok = WorldContext.add_knowledge(parent.id, "parent_only", "key2", "parent_only_value")

      {:ok, child} = create_test_world("child_knowledge", base_world: parent.id)

      :ok = WorldContext.add_knowledge(child.id, "shared", "key1", "child_value")

      assert WorldContext.get_knowledge(child.id) == %{"shared" => %{"key1" => "child_value"}}
    end
  end

  describe "add_episode/5" do
    test "writes to world-scoped store" do
      {:ok, world} = create_test_world("add_episode_test")

      assert {:ok, id} =
               WorldContext.add_episode(
                 world.id,
                 "user said hello",
                 "conversation",
                 "bot responded with greeting",
                 ["greeting", "test"]
               )

      assert [%{id: ^id}] = WorldContext.get_episodes(world.id)
    end
  end

  describe "add_knowledge/4" do
    test "writes to specific world" do
      {:ok, world} = create_test_world("add_knowledge_test")

      assert :ok = WorldContext.add_knowledge(world.id, "facts", "test_fact", "test_value")

      assert WorldContext.get_knowledge(world.id, "facts") == %{"test_fact" => "test_value"}
    end
  end

  describe "query_similar/3" do
    test "returns only the world's own episodes" do
      {:ok, parent} = create_test_world("query_parent")

      {:ok, _} =
        WorldContext.add_episode(parent.id, "parent weather query", "conversation", "weather response", [
          "weather"
        ])

      {:ok, child} = create_test_world("query_child", base_world: parent.id)

      {:ok, _} =
        WorldContext.add_episode(child.id, "child weather query", "conversation", "weather response", [
          "weather"
        ])

      {:ok, _} =
        WorldContext.add_episode(child.id, "child greeting hello", "conversation", "hello response", [
          "greeting"
        ])

      assert {:ok, results} = WorldContext.query_similar(child.id, "what's the weather", 3)

      states = Enum.map(results, fn {episode, _score} -> episode.state end)
      assert Enum.all?(states, &(&1 in ["child weather query", "child greeting hello"]))
      refute "parent weather query" in states
    end
  end

  describe "classify_intent/2" do
    test "classifies in the world it is given" do
      {:ok, world} = create_test_world("classify_test")

      assert {:ok, %{intent: intent, confidence: confidence}} =
               WorldContext.classify_intent(world.id, "Hello there")

      assert is_binary(intent)
      assert is_number(confidence)
    end
  end
end
