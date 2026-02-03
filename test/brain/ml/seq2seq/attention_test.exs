defmodule Brain.ML.Seq2Seq.AttentionTest do
  use ExUnit.Case
  
  alias Brain.ML.Seq2Seq.Attention
  
  test "computes attention for simple case" do
    # Create simple test tensors
    # encoder_outputs: [batch=1, seq_len=3, hidden=4]
    encoder_outputs = Nx.tensor([[[1.0, 2.0, 3.0, 4.0],
                                  [2.0, 3.0, 4.0, 5.0],
                                  [3.0, 4.0, 5.0, 6.0]]])
    
    # decoder_hidden: [batch=1, hidden=4]
    decoder_hidden = Nx.tensor([[1.0, 1.0, 1.0, 1.0]])
    
    hidden_size = 4
    
    {context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, hidden_size)
    
    # Check output shapes
    assert Nx.shape(context) == {1, 4}  # [batch, hidden]
    assert Nx.shape(weights) == {1, 3}  # [batch, seq_len]
    
    # Check attention weights sum to ~1.0 (softmax)
    weights_sum = Nx.sum(weights) |> Nx.to_number()
    assert abs(weights_sum - 1.0) < 0.01
  end
  
  test "attention weights are non-negative" do
    encoder_outputs = Nx.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    decoder_hidden = Nx.tensor([[1.0, 1.0]])
    
    {_context, weights} = Attention.compute_attention(encoder_outputs, decoder_hidden, 2)
    
    # All weights should be >= 0
    min_weight = Nx.reduce_min(weights) |> Nx.to_number()
    assert min_weight >= 0.0
  end
end
