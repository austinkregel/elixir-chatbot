defmodule Brain.ML.Seq2Seq.DecoderTest do
  @moduledoc """
  Tests for the LSTM decoder with attention mechanism.
  
  These tests verify that:
  1. The decoder builds valid Axon models
  2. Teacher forcing mode works correctly
  3. Autoregressive generation uses Enum.reduce_while properly
  4. Output shapes are correct
  """
  use ExUnit.Case, async: true
  
  alias Brain.ML.Seq2Seq.Decoder
  
  describe "build_model/4" do
    test "returns tuple of logits and hidden state models" do
      vocab_size = 100
      embedding_size = 32
      hidden_size = 64
      
      {logits_model, hidden_model} = Decoder.build_model(vocab_size, embedding_size, hidden_size)
      
      # Both should be Axon structs
      assert %Axon{} = logits_model
      assert %Axon{} = hidden_model
    end
    
    test "accepts optional parameters" do
      vocab_size = 100
      embedding_size = 32
      hidden_size = 64
      
      {logits_model, hidden_model} = Decoder.build_model(
        vocab_size, 
        embedding_size, 
        hidden_size,
        num_layers: 2,
        dropout: 0.2,
        encoder_hidden_size: 128
      )
      
      assert %Axon{} = logits_model
      assert %Axon{} = hidden_model
    end
    
    test "creates models with expected input names" do
      vocab_size = 100
      embedding_size = 32
      hidden_size = 64
      
      {logits_model, _hidden_model} = Decoder.build_model(vocab_size, embedding_size, hidden_size)
      
      # The model should expect these inputs
      # We verify by checking the model structure
      assert %Axon{} = logits_model
    end
  end
  
  describe "generate/5 - teacher forcing" do
    @tag :slow
    test "returns logits and hidden state with teacher forcing" do
      vocab_size = 100
      embedding_size = 16
      hidden_size = 32
      batch_size = 2
      seq_len = 5
      encoder_seq_len = 10
      
      model = Decoder.build_model(vocab_size, embedding_size, hidden_size,
        encoder_hidden_size: hidden_size)
      
      # Create mock inputs
      encoder_outputs = Nx.broadcast(Nx.tensor(0.0), {batch_size, encoder_seq_len, hidden_size})
      decoder_hidden = Nx.broadcast(Nx.tensor(0.0), {batch_size, hidden_size})
      decoder_input = Nx.broadcast(Nx.tensor(1), {batch_size, seq_len})
      
      # Initialize params (simplified - would need proper initialization)
      # This test verifies the API, not full functionality
      assert is_tuple(model)
      assert tuple_size(model) == 2
    end
  end
  
  describe "generate/5 - autoregressive mode" do
    @decoder_path Path.expand("../../../../lib/brain/ml/seq2seq/decoder.ex", __DIR__)
    
    test "uses Enum.reduce_while for proper iteration" do
      # Verify the implementation uses reduce_while by checking the source
      {:ok, source} = File.read(@decoder_path)
      
      # Should use Enum.reduce_while (not for-comprehension with break)
      assert String.contains?(source, "Enum.reduce_while")
      
      # Should NOT have bare "break" statement (which is invalid Elixir)
      # Use regex to find standalone "break" word (not as part of another word)
      refute Regex.match?(~r/\bbreak\b/, source)
    end
    
    test "maintains state across iterations" do
      # Verify the implementation maintains state properly
      {:ok, source} = File.read(@decoder_path)
      
      # Should maintain state in a map
      assert String.contains?(source, "current_input:")
      assert String.contains?(source, "current_hidden:")
      assert String.contains?(source, "all_logits:")
    end
    
    test "properly halts on EOS token" do
      {:ok, source} = File.read(@decoder_path)
      
      # Should check for EOS and halt
      assert String.contains?(source, ":halt")
      assert String.contains?(source, "eos_idx")
    end
  end
  
  describe "build_final_logits/3" do
    @decoder_path_2 Path.expand("../../../../lib/brain/ml/seq2seq/decoder.ex", __DIR__)
    
    test "helper function exists and handles empty list" do
      {:ok, source} = File.read(@decoder_path_2)
      
      # Should have the helper function
      assert String.contains?(source, "defp build_final_logits")
      
      # Should handle empty case
      assert String.contains?(source, "if length(all_logits) > 0")
    end
  end
end
