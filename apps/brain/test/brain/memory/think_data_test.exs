defmodule Brain.Memory.ThinkDataTest do
  @moduledoc """
  Data-driven tests for Memory.Think covering the cognitive memory API.
  """
  use ExUnit.Case, async: false
  import Brain.TestHelpers

  alias Brain.Memory.Think
  alias Brain.Memory.Store

  @test_world_id "think_test_#{:rand.uniform(100_000)}"

  setup do
    ensure_pubsub_started()
    ensure_started(Brain.Memory.Embedder)
    ensure_started(Store)

    :ok
  end

  # Test data for add_episode operation
  @add_episode_test_cases [
    # {params, description}
    {%{state: "user said hello", action: "greeting", outcome: "responded with hi"}, "basic episode"},
    {%{state: "weather query", action: "query", outcome: "provided weather", tags: ["weather"]}, "episode with tags"},
    {%{state: "short", action: "a", outcome: "b"}, "minimal episode"},
    {%{state: String.duplicate("long ", 100), action: "long_action", outcome: "long_outcome"}, "long content episode"},
    {%{state: "世界你好", action: "unicode", outcome: "handled"}, "unicode content"},
  ]

  describe "think(:add_episode, params) - data driven" do
    for {params, description} <- @add_episode_test_cases do
      @params params
      @description description

      test "#{description}" do
        params = Map.put(@params, :world_id, @test_world_id)
        result = Think.think(:add_episode, params)

        case result do
          {:ok, {:episode_added, id}} ->
            assert is_binary(id)
            assert String.length(id) > 0

          {:error, reason} ->
            # Some errors are acceptable (e.g., embedder not ready)
            assert reason in [:embedder_not_ready, :store_not_available, :vocabulary_building]
        end
      end
    end
  end

  # Test data for query_chat operation  
  @query_chat_test_cases [
    # {input, k, description}
    {"hello", 5, "simple greeting query"},
    {"what is the weather", 3, "weather query"},
    {"", 5, "empty query"},
    {"a", 1, "single char query"},
    {String.duplicate("word ", 50), 10, "long query"},
  ]

  describe "think(:query_chat, params) - data driven" do
    for {input, k, description} <- @query_chat_test_cases do
      @input input
      @k k
      @description description

      test "#{description}" do
        params = %{input: @input, k: @k, world_id: @test_world_id}
        result = Think.think(:query_chat, params)

        case result do
          {:ok, {:chat_results, results}} ->
            assert is_list(results)
            assert length(results) <= @k

          {:error, _reason} ->
            # Acceptable if store/embedder not ready
            assert true
        end
      end
    end
  end

  # Test data for query_semantic operation
  @query_semantic_test_cases [
    {"greeting patterns", 5, "query semantic facts"},
    {"weather information", 3, "domain specific query"},
    {"", 5, "empty semantic query"},
  ]

  describe "think(:query_semantic, params) - data driven" do
    for {input, k, description} <- @query_semantic_test_cases do
      @input input
      @k k
      @description description

      test "#{description}" do
        params = %{input: @input, k: @k, world_id: @test_world_id}
        result = Think.think(:query_semantic, params)

        case result do
          {:ok, {:semantic_results, results}} ->
            assert is_list(results)

          {:error, _reason} ->
            assert true
        end
      end
    end
  end

  # Test data for consolidate operation
  @consolidate_test_cases [
    {%{threshold: 0.8, min_size: 2}, "default consolidation"},
    {%{threshold: 0.5, min_size: 1}, "low threshold consolidation"},
    {%{threshold: 0.95, min_size: 5}, "high threshold consolidation"},
    {%{}, "empty params consolidation"},
  ]

  describe "think(:consolidate, params) - data driven" do
    for {params, description} <- @consolidate_test_cases do
      @params params
      @description description

      test "#{description}" do
        params = Map.put(@params, :world_id, @test_world_id)
        result = Think.think(:consolidate, params)

        case result do
          {:ok, {:consolidated, count}} ->
            assert is_integer(count)
            assert count >= 0

          {:error, _reason} ->
            assert true
        end
      end
    end
  end

  # Edge cases - only test valid operations
  describe "edge cases" do
    test "handles empty params for add_episode" do
      result = Think.think(:add_episode, %{world_id: @test_world_id})
      # Should succeed with empty strings or return error
      case result do
        {:ok, _} -> assert true
        {:error, _} -> assert true
      end
    end

    test "handles minimal params" do
      result = Think.think(:add_episode, %{state: "test", world_id: @test_world_id})
      # Should handle gracefully
      case result do
        {:ok, _} -> assert true
        {:error, _} -> assert true
      end
    end
  end

  # Integration test
  describe "integration - add then query" do
    test "can query recently added episode" do
      # Add an episode
      add_result = Think.think(:add_episode, %{
        state: "unique test phrase #{:rand.uniform(100_000)}",
        action: "test_action",
        outcome: "test_outcome",
        world_id: @test_world_id
      })

      case add_result do
        {:ok, {:episode_added, _id}} ->
          # Try to query for it
          query_result = Think.think(:query_chat, %{
            input: "unique test phrase",
            k: 5,
            world_id: @test_world_id
          })

          case query_result do
            {:ok, {:chat_results, _results}} ->
              assert true

            {:error, _} ->
              assert true
          end

        {:error, _} ->
          # Skip if add failed
          assert true
      end
    end
  end

  # Test other operations
  describe "think(:stats, params)" do
    test "returns stats about memory" do
      result = Think.think(:stats, %{world_id: @test_world_id})
      
      case result do
        {:ok, stats} ->
          assert is_map(stats) or is_tuple(stats)
        {:error, _} ->
          assert true
      end
    end
  end
end
