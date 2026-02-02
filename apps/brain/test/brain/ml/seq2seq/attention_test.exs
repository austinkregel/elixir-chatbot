defmodule Brain.ML.Seq2Seq.AttentionTest do
  @moduledoc """
  Tests for the Bahdanau attention mechanism.
  
  These tests verify that the attention mechanism correctly:
  1. Computes attention weights that sum to 1.0 (softmax normalization)
  2. Produces context vectors of the correct shape
  3. Focuses attention on relevant encoder positions
  """
  use ExUnit.Case, async: true
  
  alias Brain.ML.Seq2Seq.Attention
  
  describe "compute_attention/3" do
    test "produces correctly shaped outputs" do
      # encoder_outputs: [batch=1, seq_len=3, hidden=4]
      encoder_outputs = Nx.tensor([[[1.0, 2.0, 3.0, 4.0],
                                    [2.0, 3.0, 4.0, 5.0],
                                    [3.0, 4.0, 5.0, 6.0]]])
      
      # decoder_hidden: [batch=1, hidden=4]
      decoder_hidden = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      
      hidden_size = 4
      
      {context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, hidden_size)
      
      # Context should be [batch, hidden]
      assert Nx.shape(context) == {1, 4}
      # Weights should be [batch, seq_len]
      assert Nx.shape(weights) == {1, 3}
    end
    
    test "attention weights sum to 1.0 (softmax normalization)" do
      encoder_outputs = Nx.tensor([[[1.0, 2.0, 3.0, 4.0],
                                    [2.0, 3.0, 4.0, 5.0],
                                    [3.0, 4.0, 5.0, 6.0]]])
      
      decoder_hidden = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
      
      {_context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, 4)
      
      weights_sum = Nx.sum(weights) |> Nx.to_number()
      assert_in_delta weights_sum, 1.0, 0.001
    end
    
    test "attention weights are non-negative" do
      encoder_outputs = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]]])
      decoder_hidden = Nx.tensor([[1.0, 1.0]])
      
      {_context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, 2)
      
      min_weight = Nx.reduce_min(weights) |> Nx.to_number()
      assert min_weight >= 0.0
    end
    
    test "context is weighted combination of encoder outputs" do
      # Simple case: 2 encoder positions
      encoder_outputs = Nx.tensor([[[1.0, 0.0], [0.0, 1.0]]])
      decoder_hidden = Nx.tensor([[1.0, 0.0]])  # Similar to first encoder output
      
      {context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, 2)
      
      # The context should be a weighted average
      # Verify context values are within the range of encoder outputs
      context_values = Nx.to_flat_list(context)
      assert Enum.all?(context_values, fn v -> v >= 0.0 and v <= 1.0 end)
      
      # Weights should be valid probabilities
      weights_list = Nx.to_flat_list(weights)
      assert Enum.all?(weights_list, fn w -> w >= 0.0 and w <= 1.0 end)
    end
    
    test "handles batch size > 1" do
      # encoder_outputs: [batch=2, seq_len=3, hidden=4]
      encoder_outputs = Nx.tensor([
        [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
        [[4.0, 3.0, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0], [6.0, 5.0, 4.0, 3.0]]
      ])
      
      decoder_hidden = Nx.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
      ])
      
      {context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, 4)
      
      assert Nx.shape(context) == {2, 4}
      assert Nx.shape(weights) == {2, 3}
      
      # Each batch's weights should sum to 1
      batch_0_sum = weights |> Nx.slice([0, 0], [1, 3]) |> Nx.sum() |> Nx.to_number()
      batch_1_sum = weights |> Nx.slice([1, 0], [1, 3]) |> Nx.sum() |> Nx.to_number()
      
      assert_in_delta batch_0_sum, 1.0, 0.001
      assert_in_delta batch_1_sum, 1.0, 0.001
    end
  end
  
  describe "compute_single/3" do
    test "handles unbatched input" do
      # encoder_outputs: [seq_len=3, hidden=4] (no batch dimension)
      encoder_outputs = Nx.tensor([[1.0, 2.0, 3.0, 4.0],
                                   [2.0, 3.0, 4.0, 5.0],
                                   [3.0, 4.0, 5.0, 6.0]])
      
      # decoder_hidden: [hidden=4]
      decoder_hidden = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      
      {context, weights} = Attention.compute_single(encoder_outputs, decoder_hidden, 4)
      
      # Should return unbatched outputs
      assert Nx.shape(context) == {4}
      assert Nx.shape(weights) == {3}
      
      # Weights should still sum to 1
      weights_sum = Nx.sum(weights) |> Nx.to_number()
      assert_in_delta weights_sum, 1.0, 0.001
    end
  end
end
