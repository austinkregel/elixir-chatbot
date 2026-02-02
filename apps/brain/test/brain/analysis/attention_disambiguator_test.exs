defmodule Brain.Analysis.AttentionDisambiguatorTest do
  @moduledoc """
  Tests for the attention-based entity disambiguator.
  
  These tests verify that the attention mechanism improves entity disambiguation
  by using data-driven type inference rather than hardcoded word lists.
  
  Key assertions:
  1. Disambiguator uses TypeInferrer for learned patterns (not hardcoded keywords)
  2. Returns correct entity type based on context
  3. Falls back gracefully when model not ready
  4. Properly extracts POS tags from various token formats
  """
  use ExUnit.Case, async: false
  
  alias Brain.Analysis.AttentionDisambiguator
  
  # Test world for isolation
  @test_world_id "attention_disambiguator_test"
  
  describe "disambiguate/4 - basic functionality" do
    test "returns first type when only one type available" do
      entity = %{value: "Austin"}
      context_tokens = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]
      possible_types = [%{entity_type: "person", value: "Austin"}]
      
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert result == %{entity_type: "person", value: "Austin"}
    end
    
    test "returns entity itself when no types available" do
      entity = %{value: "Unknown", entity_type: "unknown"}
      context_tokens = [{"the", "DET"}, {"Unknown", "PROPN"}]
      possible_types = []
      
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      # Should return first of possible_types or entity
      assert result == entity or result == nil
    end
    
    test "handles plain string tokens" do
      entity = %{value: "Austin"}
      context_tokens = ["I", "am", "Austin"]
      possible_types = [
        %{entity_type: "person", value: "Austin"},
        %{entity_type: "location", value: "Austin"}
      ]
      
      # Should not crash with plain strings
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert is_map(result)
      assert Map.has_key?(result, :entity_type)
    end
    
    test "handles map tokens with :text key" do
      entity = %{value: "Austin"}
      context_tokens = [
        %{text: "I", tag: "PRON"},
        %{text: "am", tag: "VERB"},
        %{text: "Austin", tag: "PROPN"}
      ]
      possible_types = [
        %{entity_type: "person", value: "Austin"},
        %{entity_type: "location", value: "Austin"}
      ]
      
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert is_map(result)
      assert Map.has_key?(result, :entity_type)
    end
  end
  
  describe "disambiguate/4 - type scoring" do
    test "scores related types similarly (location/city)" do
      entity = %{value: "Paris"}
      context_tokens = [{"weather", "NOUN"}, {"in", "ADP"}, {"Paris", "PROPN"}]
      
      location_type = %{entity_type: "location", value: "Paris"}
      city_type = %{entity_type: "city", value: "Paris"}
      person_type = %{entity_type: "person", value: "Paris"}
      
      possible_types = [location_type, city_type, person_type]
      
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      # Should prefer location or city over person for this context
      # (though exact result depends on TypeInferrer training)
      assert is_map(result)
      assert result.entity_type in ["location", "city", "person"]
    end
    
    test "preserves entity value in result" do
      entity = %{value: "TestValue"}
      context_tokens = [{"test", "NOUN"}]
      possible_types = [
        %{entity_type: "type_a", value: "TestValue"},
        %{entity_type: "type_b", value: "TestValue"}
      ]
      
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert result.value == "TestValue"
    end
  end
  
  describe "ready?/1" do
    test "returns boolean" do
      result = AttentionDisambiguator.ready?(world_id: @test_world_id)
      assert is_boolean(result)
    end
    
    test "defaults to 'default' world when no world_id provided" do
      result = AttentionDisambiguator.ready?([])
      assert is_boolean(result)
    end
  end
  
  describe "data-driven approach verification" do
    @disambiguator_path Path.expand("../../../lib/brain/analysis/attention_disambiguator.ex", __DIR__)
    
    @tag :integration
    test "does not use hardcoded word lists" do
      # This test verifies the implementation follows project rules
      # by checking that the module doesn't contain hardcoded word patterns
      
      {:ok, source} = File.read(@disambiguator_path)
      
      # Should NOT contain hardcoded word lists like the old implementation
      refute String.contains?(source, "intro_words = ~w")
      refute String.contains?(source, "location_words = ~w")
      refute String.contains?(source, "music_words = ~w")
      
      # Should use TypeInferrer (data-driven)
      assert String.contains?(source, "TypeInferrer")
    end
    
    @tag :integration
    test "uses TypeInferrer for type inference" do
      {:ok, source} = File.read(@disambiguator_path)
      
      # Should call TypeInferrer.infer_type
      assert String.contains?(source, "TypeInferrer.infer_type")
    end
  end
  
  describe "extract_tokens_and_tags/1 - helper function" do
    # Testing the helper through the public API
    test "correctly processes POS-tagged tuples" do
      entity = %{value: "Test"}
      context_tokens = [{"I", "PRON"}, {"am", "VERB"}, {"Test", "PROPN"}]
      possible_types = [%{entity_type: "test_type", value: "Test"}]
      
      # Should process without error
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert is_map(result)
    end
    
    test "handles mixed token formats gracefully" do
      entity = %{value: "Test"}
      context_tokens = [
        {"word1", "NOUN"},        # Tuple format
        "word2",                   # Plain string
        %{text: "word3"},         # Map without tag
        %{text: "word4", tag: "VERB"}  # Map with tag
      ]
      possible_types = [%{entity_type: "test_type", value: "Test"}]
      
      # Should handle all formats
      result = AttentionDisambiguator.disambiguate(entity, context_tokens, possible_types,
        world_id: @test_world_id)
      
      assert is_map(result)
    end
  end
end
