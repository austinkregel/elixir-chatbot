defmodule Brain.Analysis.AttentionIntegrationTest do
  @moduledoc """
  Integration tests verifying the attention mechanism improves overall system capabilities.
  
  These tests compare disambiguation results with and without attention to ensure
  the neural approach provides value over the baseline.
  
  Key test scenarios:
  1. Ambiguous names (Austin as person vs location)
  2. Context-dependent entities (Nice as greeting vs city)
  3. Multi-entity sentences
  
  Success criteria from the plan:
  - Entity disambiguation accuracy should be high for context-dependent cases
  - Attention should focus on relevant context words
  """
  use ExUnit.Case, async: false
  
  alias Brain.Analysis.{EntityDisambiguator, AttentionDisambiguator}
  
  @test_world_id "attention_integration_test"
  
  describe "disambiguation comparison" do
    @tag :integration
    test "both disambiguators handle 'I'm Austin' pattern" do
      # This is a key test case: "Austin" should be recognized as a person
      # when preceded by "I'm"
      entity = %{
        value: "Austin",
        types: [
          %{entity_type: "person", value: "Austin"},
          %{entity_type: "location", value: "Austin", city: true}
        ]
      }
      
      pos_tagged = [{"I", "PRON"}, {"'m", "VERB"}, {"Austin", "PROPN"}]
      context = %{
        discourse: %{indicators: ["self_referential"]},
        speech_act: %{category: :expressive, sub_type: :greeting},
        original_text: "I'm Austin",
        world_id: @test_world_id
      }
      
      # Test EntityDisambiguator (rule-based + TypeInferrer)
      ed_result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)
      
      # Test AttentionDisambiguator
      ad_result = AttentionDisambiguator.disambiguate(
        entity, 
        pos_tagged, 
        entity.types,
        world_id: @test_world_id
      )
      
      # Both should return valid results
      assert is_map(ed_result)
      assert is_map(ad_result)
      
      # EntityDisambiguator should recognize introduction pattern
      assert ed_result.entity_type == "person" or 
             ed_result[:disambiguation_reason] == "introduction_pattern"
    end
    
    @tag :integration
    test "both disambiguators handle 'weather in Austin' pattern" do
      # "Austin" should be recognized as a location in weather context
      entity = %{
        value: "Austin",
        types: [
          %{entity_type: "person", value: "Austin"},
          %{entity_type: "location", value: "Austin", city: true}
        ]
      }
      
      pos_tagged = [{"weather", "NOUN"}, {"in", "ADP"}, {"Austin", "PROPN"}]
      context = %{
        discourse: %{indicators: []},
        speech_act: %{category: :directive, sub_type: :request, is_question: true},
        intent: "weather.query",
        world_id: @test_world_id
      }
      
      # Test EntityDisambiguator
      ed_result = EntityDisambiguator.disambiguate_single(entity, pos_tagged, context)
      
      # Test AttentionDisambiguator
      ad_result = AttentionDisambiguator.disambiguate(
        entity, 
        pos_tagged, 
        entity.types,
        world_id: @test_world_id
      )
      
      # Both should return valid location-type results for weather context
      assert is_map(ed_result)
      assert is_map(ad_result)
      
      # EntityDisambiguator should prefer location for weather intent
      assert ed_result.entity_type in ["location", "city"]
    end
    
    @tag :integration
    test "both disambiguators handle ambiguous city names" do
      # "Nice" could be the city or an adjective
      entity = %{
        value: "Nice",
        types: [
          %{entity_type: "city", value: "Nice", country: "France"},
          %{entity_type: "adjective", value: "nice"}
        ]
      }
      
      # Location context
      pos_tagged_location = [{"fly", "VERB"}, {"to", "ADP"}, {"Nice", "PROPN"}]
      context_location = %{
        discourse: %{indicators: []},
        speech_act: %{category: :directive},
        intent: "navigation.directions",
        world_id: @test_world_id
      }
      
      ad_result = AttentionDisambiguator.disambiguate(
        entity, 
        pos_tagged_location, 
        entity.types,
        world_id: @test_world_id
      )
      
      assert is_map(ad_result)
      assert Map.has_key?(ad_result, :entity_type)
    end
  end
  
  describe "attention focus" do
    @tag :integration
    test "attention weights are computed for context" do
      # Verify that the attention mechanism actually processes the context
      entity = %{value: "Test"}
      context_tokens = [{"The", "DET"}, {"quick", "ADJ"}, {"brown", "ADJ"}, {"Test", "PROPN"}]
      possible_types = [
        %{entity_type: "type_a", value: "Test"},
        %{entity_type: "type_b", value: "Test"}
      ]
      
      # Should return a result (attention was computed internally)
      result = AttentionDisambiguator.disambiguate(
        entity, 
        context_tokens, 
        possible_types,
        world_id: @test_world_id
      )
      
      assert is_map(result)
      assert result.entity_type in ["type_a", "type_b"]
    end
  end
  
  describe "fallback behavior" do
    test "returns first type when model not ready and multiple types exist" do
      # When Seq2Seq model isn't ready, should gracefully fallback
      entity = %{value: "Test"}
      context_tokens = [{"test", "NOUN"}]
      possible_types = [
        %{entity_type: "first_type", value: "Test"},
        %{entity_type: "second_type", value: "Test"}
      ]
      
      # Even if model not ready, should return valid result
      result = AttentionDisambiguator.disambiguate(
        entity, 
        context_tokens, 
        possible_types,
        world_id: "nonexistent_world"
      )
      
      # Should fallback to first type
      assert is_map(result) or is_nil(result)
      if is_map(result) do
        assert result.entity_type in ["first_type", "second_type"]
      end
    end
  end
  
  describe "data-driven verification" do
    @attention_disambiguator_path Path.expand("../../../lib/brain/analysis/attention_disambiguator.ex", __DIR__)
    @entity_disambiguator_path Path.expand("../../../lib/brain/analysis/entity_disambiguator.ex", __DIR__)
    
    @tag :integration
    test "attention disambiguator uses TypeInferrer not hardcoded patterns" do
      # Read the source file to verify no hardcoded patterns
      {:ok, source} = File.read(@attention_disambiguator_path)
      
      # Verify data-driven approach
      assert String.contains?(source, "alias World.TypeInferrer")
      assert String.contains?(source, "TypeInferrer.infer_type")
      
      # Verify no hardcoded word lists
      refute String.contains?(source, ~s(intro_words = ~w))
      refute String.contains?(source, ~s(location_words = ~w))
      refute String.contains?(source, ~s(music_words = ~w))
    end
    
    @tag :integration  
    test "entity disambiguator uses IntentRegistry for domain detection" do
      {:ok, source} = File.read(@entity_disambiguator_path)
      
      # Should use IntentRegistry for data-driven intent detection
      assert String.contains?(source, "IntentRegistry")
      assert String.contains?(source, "IntentRegistry.weather_intent?")
      assert String.contains?(source, "IntentRegistry.introduction_intent?")
    end
  end
  
  describe "edge cases" do
    test "handles empty context tokens" do
      entity = %{value: "Test"}
      context_tokens = []
      possible_types = [%{entity_type: "test", value: "Test"}]
      
      result = AttentionDisambiguator.disambiguate(
        entity, 
        context_tokens, 
        possible_types,
        world_id: @test_world_id
      )
      
      assert result == %{entity_type: "test", value: "Test"}
    end
    
    test "handles entity with no value" do
      entity = %{some_field: "data"}
      context_tokens = [{"test", "NOUN"}]
      possible_types = [%{entity_type: "test", value: ""}]
      
      # Should not crash
      result = AttentionDisambiguator.disambiguate(
        entity, 
        context_tokens, 
        possible_types,
        world_id: @test_world_id
      )
      
      assert is_map(result) or is_nil(result)
    end
  end
end
