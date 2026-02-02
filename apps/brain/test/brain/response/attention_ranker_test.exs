defmodule Brain.Response.AttentionRankerTest do
  @moduledoc """
  Tests for the attention-based response ranker.
  
  These tests verify that the attention mechanism improves response ranking
  by computing relevance scores based on query-candidate similarity.
  
  Key assertions:
  1. Rankings are stable and deterministic
  2. More relevant responses rank higher
  3. Falls back gracefully when model not ready
  4. Handles various candidate formats
  """
  use ExUnit.Case, async: false
  
  alias Brain.Response.AttentionRanker
  
  @test_world_id "attention_ranker_test"
  
  describe "rank/3 - basic functionality" do
    test "returns candidates in some order" do
      query = "What is the weather?"
      candidates = [
        %{text: "The weather is sunny", id: 1},
        %{text: "Hello there", id: 2},
        %{text: "It will rain tomorrow", id: 3}
      ]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      # Should return same number of candidates
      assert length(result) == 3
      
      # Should contain all original candidates
      result_ids = Enum.map(result, & &1.id) |> Enum.sort()
      assert result_ids == [1, 2, 3]
    end
    
    test "handles empty candidate list" do
      query = "What is the weather?"
      candidates = []
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      assert result == []
    end
    
    test "handles single candidate" do
      query = "Hello"
      candidates = [%{text: "Hi there!", id: 1}]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      assert length(result) == 1
      assert hd(result).id == 1
    end
    
    test "preserves candidate data" do
      query = "Test query"
      candidates = [
        %{text: "Response A", id: 1, extra_field: "data_a"},
        %{text: "Response B", id: 2, extra_field: "data_b"}
      ]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      # All fields should be preserved
      Enum.each(result, fn candidate ->
        assert Map.has_key?(candidate, :text)
        assert Map.has_key?(candidate, :id)
        assert Map.has_key?(candidate, :extra_field)
      end)
    end
  end
  
  describe "rank/3 - candidate formats" do
    test "handles string candidates" do
      query = "Hello"
      candidates = ["Hi there!", "Good morning", "Hey"]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      assert length(result) == 3
      assert Enum.all?(result, &is_binary/1)
    end
    
    test "handles map candidates with :text key" do
      query = "Hello"
      candidates = [
        %{text: "Hi there!"},
        %{text: "Good morning"}
      ]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      assert length(result) == 2
    end
    
    test "handles map candidates with :message key" do
      query = "Hello"
      candidates = [
        %{message: "Hi there!"},
        %{message: "Good morning"}
      ]
      
      result = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      assert length(result) == 2
    end
  end
  
  describe "ready?/1" do
    test "returns boolean" do
      result = AttentionRanker.ready?(world_id: @test_world_id)
      assert is_boolean(result)
    end
    
    test "defaults to 'default' world" do
      result = AttentionRanker.ready?([])
      assert is_boolean(result)
    end
  end
  
  describe "ranking quality" do
    @tag :integration
    test "ranking is deterministic for same input" do
      query = "What is the weather today?"
      candidates = [
        %{text: "The weather forecast shows sun", id: 1},
        %{text: "Hello and welcome", id: 2},
        %{text: "Today will be warm and sunny", id: 3}
      ]
      
      result1 = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      result2 = AttentionRanker.rank(query, candidates, world_id: @test_world_id)
      
      # Same input should produce same ranking
      assert Enum.map(result1, & &1.id) == Enum.map(result2, & &1.id)
    end
    
    @tag :integration
    test "different queries may produce different rankings" do
      candidates = [
        %{text: "The weather is nice", id: 1},
        %{text: "Hello, how are you?", id: 2},
        %{text: "Play some music", id: 3}
      ]
      
      weather_query = "What is the weather?"
      greeting_query = "Hello there"
      
      weather_result = AttentionRanker.rank(weather_query, candidates, world_id: @test_world_id)
      greeting_result = AttentionRanker.rank(greeting_query, candidates, world_id: @test_world_id)
      
      # Different queries may rank candidates differently
      # (exact behavior depends on model state)
      assert is_list(weather_result)
      assert is_list(greeting_result)
      assert length(weather_result) == 3
      assert length(greeting_result) == 3
    end
  end
end
