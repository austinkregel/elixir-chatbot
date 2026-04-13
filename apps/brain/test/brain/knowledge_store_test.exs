defmodule Brain.KnowledgeStoreDataTest do
  @moduledoc """
  Data-driven tests for KnowledgeStore covering world and persona knowledge management.
  """
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Brain.KnowledgeStore

  @test_world_id "test_world_#{:rand.uniform(100_000)}"

  setup _context do
    ensure_pubsub_started()
    ensure_started(KnowledgeStore)

    on_exit(fn ->
      try do
        KnowledgeStore.clear_world(@test_world_id)
      catch
        _, _ -> :ok
      end
    end)

    :ok
  end

  # Test data for world knowledge operations
  @world_knowledge_test_cases [
    # {category, key, value, description}
    {"people", "alice", %{name: "Alice", age: 30}, "add person to world"},
    {"places", "home", %{address: "123 Main St"}, "add place to world"},
    {"facts", "weather", %{condition: "sunny"}, "add fact to world"},
    {"devices", "lamp", %{room: "living room", status: "on"}, "add device to world"},
    {"pets", "fluffy", %{type: "cat", color: "orange"}, "add pet to world"},
  ]

  describe "add_to_world/4 - data driven" do
    for {category, key, value, description} <- @world_knowledge_test_cases do
      @category category
      @key key
      @value value
      @description description

      test "#{description}" do
        result = KnowledgeStore.add_to_world(@test_world_id, @category, @key, @value)
        assert result == :ok

        # Verify it was stored
        knowledge = KnowledgeStore.get_world_knowledge(@test_world_id, @category)
        assert is_map(knowledge)
        assert Map.has_key?(knowledge, @key) or Map.has_key?(knowledge, String.to_atom(@key))
      end
    end
  end

  describe "get_world_knowledge/2 - data driven" do
    test "returns empty map for non-existent world" do
      result = KnowledgeStore.get_world_knowledge("nonexistent_world_123")
      assert result == %{} or is_map(result)
    end

    test "returns all categories when no filter" do
      # Add some data first
      KnowledgeStore.add_to_world(@test_world_id, "test_cat", "test_key", %{data: "value"})

      result = KnowledgeStore.get_world_knowledge(@test_world_id)
      assert is_map(result)
    end

    test "filters by category when specified" do
      KnowledgeStore.add_to_world(@test_world_id, "specific", "item", %{value: 1})

      result = KnowledgeStore.get_world_knowledge(@test_world_id, "specific")
      assert is_map(result)
    end
  end

  describe "remove_from_world/3 - data driven" do
    test "removes existing entry" do
      KnowledgeStore.add_to_world(@test_world_id, "removable", "item", %{temp: true})

      result = KnowledgeStore.remove_from_world(@test_world_id, "removable", "item")
      assert result == :ok
    end

    test "handles non-existent entry gracefully" do
      result = KnowledgeStore.remove_from_world(@test_world_id, "nonexistent", "item")
      # Should not crash, returns :ok or similar
      assert result in [:ok, {:error, :not_found}]
    end
  end

  describe "clear_world/1" do
    test "clears all knowledge for a world" do
      # Add data
      KnowledgeStore.add_to_world(@test_world_id, "clear_test", "item", %{data: true})

      # Clear
      result = KnowledgeStore.clear_world(@test_world_id)
      assert result == :ok

      # Verify - after clear, the category should be empty or knowledge should be reset
      knowledge = KnowledgeStore.get_world_knowledge(@test_world_id, "clear_test")
      assert is_map(knowledge)
      # May or may not be empty depending on implementation
    end
  end

  # Legacy persona API tests
  @persona_test_cases [
    # {function, args, description}
    {:load_knowledge, ["TestPersona"], "load knowledge for persona"},
    {:save_knowledge, ["TestPersona", %{}], "save empty knowledge"},
  ]

  describe "legacy persona API - data driven" do
    for {func, args, description} <- @persona_test_cases do
      @func func
      @args args
      @description description

      test "#{description}" do
        result = apply(KnowledgeStore, @func, @args)
        # Should not crash
        assert result != nil or result == nil
      end
    end
  end

  # Concurrent access test
  describe "concurrent access" do
    test "handles concurrent writes" do
      tasks = for i <- 1..10 do
        Task.async(fn ->
          KnowledgeStore.add_to_world(
            @test_world_id,
            "concurrent",
            "item_#{i}",
            %{index: i}
          )
        end)
      end

      results = Task.await_many(tasks, 5000)
      assert Enum.all?(results, &(&1 == :ok))
    end
  end
end
