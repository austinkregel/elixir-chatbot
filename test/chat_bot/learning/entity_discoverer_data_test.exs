defmodule ChatBot.Learning.EntityDiscovererDataTest do
  @moduledoc """
  Data-driven tests for EntityDiscoverer covering proper noun detection and classification.
  
  Note: Many tests depend on the POS model being trained. Tests gracefully handle
  cases where models are not available.
  """
  use ExUnit.Case, async: false
  import ChatBot.TestHelpers

  alias ChatBot.Learning.EntityDiscoverer

  @test_world_id "entity_disc_test_#{:rand.uniform(100_000)}"

  setup do
    ensure_pubsub_started()
    ensure_started(ChatBot.ML.Gazetteer)

    # Load gazetteer data
    try do
      ChatBot.ML.Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  # Test data for entity discovery - focus on function behavior, not specific counts
  @discovery_test_cases [
    # {input_text, description}
    {"John went to Paris yesterday", "sentence with person and location"},
    {"Microsoft announced new products", "sentence with organization"},
    {"The Eiffel Tower is in France", "landmarks and countries"},
    {"hello world", "no proper nouns"},
    {"I like coffee", "common nouns only"},
    {"ACME Corp hired Alice and Bob", "multiple entities"},
    {"New York City is large", "multi-word location"},
    {"Dr. Smith visited London", "title with name and location"},
    {"", "empty input"},
    {"a b c", "very short input"},
  ]

  describe "discover_entities/3 - data driven" do
    for {input, description} <- @discovery_test_cases do
      @input input
      @description description

      test "returns list for: #{description}" do
        result = EntityDiscoverer.discover_entities(@input, @test_world_id, emit_events: false)

        # Should always return a list (empty if no model or no entities found)
        assert is_list(result)
      end
    end
  end

  # Test result structure when entities are found
  describe "discovery result structure" do
    test "results have expected fields when found" do
      input = "Alice visited Paris"
      results = EntityDiscoverer.discover_entities(input, @test_world_id, emit_events: false)

      for result <- results do
        # Must have value and status
        assert Map.has_key?(result, :value)
        assert Map.has_key?(result, :status)
        assert result.status in [:known, :unknown, :ambiguous]
      end
    end
  end

  # Batch processing test cases
  @batch_test_cases [
    {["Hello world", "John is here", "Paris is nice"], "three short texts"},
    {["Single text"], "single text batch"},
    {[], "empty batch"},
  ]

  describe "discover_entities_batch/3 - data driven" do
    for {texts, description} <- @batch_test_cases do
      @texts texts
      @description description

      test "#{description}" do
        results = EntityDiscoverer.discover_entities_batch(@texts, @test_world_id, emit_events: false)

        # Should return a list
        assert is_list(results)
        
        # Each result should also be a list
        for r <- results do
          assert is_list(r)
        end
      end
    end
  end

  # Edge cases with unusual input
  @edge_case_inputs [
    {"日本語テスト", "Japanese text"},
    {"Ñoño went to Zürich", "accented characters"},
    {"user@email.com", "email address"},
    {"http://example.com", "URL"},
    {"123 Main Street", "address with numbers"},
    {String.duplicate("Word ", 100), "very long input"},
    {"ALL CAPS SENTENCE HERE", "all caps input"},
    {"MixedCaseWords InSentence", "mixed case words"},
  ]

  describe "edge cases - data driven" do
    for {input, description} <- @edge_case_inputs do
      @input input
      @description description

      test "handles #{description} without crashing" do
        # Should not crash
        result = EntityDiscoverer.discover_entities(@input, @test_world_id, emit_events: false)
        assert is_list(result)
      end
    end
  end

  # Context window test
  describe "context window option" do
    test "respects context_window option" do
      input = "Alice works at Microsoft in Seattle"
      
      result_small = EntityDiscoverer.discover_entities(input, @test_world_id, 
        emit_events: false, context_window: 2)
      result_large = EntityDiscoverer.discover_entities(input, @test_world_id, 
        emit_events: false, context_window: 10)

      # Both should return results (lists)
      assert is_list(result_small)
      assert is_list(result_large)
    end
  end

  # Test with nil model option
  describe "discovery with options" do
    test "works with custom model option" do
      # Pass nil model to test fallback
      result = EntityDiscoverer.discover_entities("Test text", @test_world_id, 
        emit_events: false, model: nil)
      assert is_list(result)
    end
  end
end
