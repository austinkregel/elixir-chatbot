defmodule Brain.ML.LSTM.NegativeTrainingTest do
  use ExUnit.Case, async: true
  
  alias Brain.ML.DataLoaders
  
  @moduletag :lstm
  
  describe "load_negative_examples/0" do
    test "loads negative examples from file" do
      {:ok, examples} = DataLoaders.load_negative_examples()
      
      # Should have loaded the meta.self_knowledge negatives
      assert length(examples) > 0
      
      # Check structure
      first = hd(examples)
      assert Map.has_key?(first, :text)
      assert Map.has_key?(first, :intent)
      assert Map.has_key?(first, :negative_for)
      
      assert first.negative_for == "meta.self_knowledge"
    end
    
    test "negative examples have correct_intent field" do
      {:ok, examples} = DataLoaders.load_negative_examples()
      
      # All examples should have a correct intent
      Enum.each(examples, fn ex ->
        assert is_binary(ex.intent)
        assert ex.intent != ""
      end)
    end
  end
  
  describe "load_intent_training_data_for_lstm/1 with negatives" do
    test "includes negative examples when include_negative: true" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm(include_negative: true)
      
      # Should have some examples with negative_for set
      negatives = Enum.filter(examples, fn ex -> ex.negative_for != nil end)
      assert length(negatives) > 0
      
      # Negative examples should have all required fields
      Enum.each(negatives, fn ex ->
        assert is_list(ex.tokens)
        assert is_binary(ex.intent)
        assert is_list(ex.bio_tags)
        assert is_binary(ex.negative_for)
      end)
    end
    
    test "excludes negative examples when include_negative: false" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm(include_negative: false)
      
      # Should have no examples with negative_for set
      negatives = Enum.filter(examples, fn ex -> ex.negative_for != nil end)
      assert length(negatives) == 0
    end
    
    test "negative examples for meta.self_knowledge have weather-related phrases" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm(include_negative: true)
      
      # Find examples negative for meta.self_knowledge
      meta_negatives = Enum.filter(examples, fn ex -> 
        ex.negative_for == "meta.self_knowledge"
      end)
      
      assert length(meta_negatives) > 0
      
      # At least some should contain "weather"
      weather_negatives = Enum.filter(meta_negatives, fn ex ->
        Enum.any?(ex.tokens, fn token -> 
          # Token might be a string or a map with :text field
          token_text = cond do
            is_binary(token) -> token
            is_map(token) -> token[:text] || token["text"]
            true -> nil
          end
          
          is_binary(token_text) and String.downcase(token_text) == "weather"
        end)
      end)
      
      assert length(weather_negatives) > 0,
        "Expected some weather-related negative examples for meta.self_knowledge"
    end
  end
  
  describe "negative examples in training" do
    test "negative examples are properly tokenized" do
      {:ok, examples} = DataLoaders.load_intent_training_data_for_lstm(include_negative: true)
      
      negatives = Enum.filter(examples, fn ex -> ex.negative_for != nil end)
      
      Enum.each(negatives, fn ex ->
        # Tokens should not be empty
        assert length(ex.tokens) > 0
        
        # BIO tags should match token count
        assert length(ex.bio_tags) == length(ex.tokens)
        
        # All BIO tags should be "O" (negatives shouldn't have entities)
        Enum.each(ex.bio_tags, fn tag ->
          assert tag == "O", 
            "Expected 'O' tag for negative example, got: #{tag}"
        end)
      end)
    end
  end
end
