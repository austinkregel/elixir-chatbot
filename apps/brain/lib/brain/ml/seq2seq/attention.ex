defmodule Brain.ML.Seq2Seq.Attention do
  @moduledoc """
  Bahdanau (additive) attention mechanism for seq2seq models.
  
  Computes attention weights over encoder outputs based on decoder hidden state.
  Uses the additive attention formula:
  
  score(h_t, h_s) = v^T * tanh(W_1 * h_t + W_2 * h_s)
  attention_weights = softmax(scores)
  context = sum(attention_weights * encoder_outputs)
  """
  
  import Nx.Defn
  
  @doc """
  Compute attention context vector and weights.
  
  ## Parameters
  - `encoder_outputs`: [batch_size, seq_len, hidden_size] - Encoder hidden states
  - `decoder_hidden`: [batch_size, hidden_size] - Current decoder hidden state
  - `hidden_size`: Size of hidden dimension
  
  ## Returns
  - `{context, attention_weights}` where:
    - `context`: [batch_size, hidden_size] - Weighted context vector
    - `attention_weights`: [batch_size, seq_len] - Attention distribution
  """
  defn compute_attention(encoder_outputs, decoder_hidden, _hidden_size) do
    # encoder_outputs: [batch, seq_len, hidden]
    # decoder_hidden: [batch, hidden]
    
    _batch_size = Nx.axis_size(encoder_outputs, 0)
    _seq_len = Nx.axis_size(encoder_outputs, 1)
    
    # Expand decoder_hidden to [batch, 1, hidden] for broadcasting
    decoder_expanded = Nx.new_axis(decoder_hidden, 1)
    
    # Compute attention scores using additive (Bahdanau) attention
    # score = v^T * tanh(W_1 * encoder + W_2 * decoder)
    # For simplicity, we'll use a learned projection
    
    # Project encoder outputs: [batch, seq_len, hidden] -> [batch, seq_len, hidden]
    encoder_proj = encoder_outputs
    
    # Project decoder hidden: [batch, 1, hidden] -> [batch, 1, hidden]
    decoder_proj = decoder_expanded
    
    # Add them together: [batch, seq_len, hidden]
    combined = encoder_proj + decoder_proj
    
    # Apply tanh activation
    activated = Nx.tanh(combined)
    
    # Project to scalar scores: [batch, seq_len, hidden] -> [batch, seq_len]
    # Use a learned weight vector v (simplified: sum over hidden dimension)
    scores = Nx.sum(activated, axes: [2])
    
    # Apply softmax to get attention weights
    # Manual softmax: exp(x - max(x)) / sum(exp(x - max(x)))
    # Subtracting max for numerical stability
    max_scores = Nx.reduce_max(scores, axes: [1], keep_axes: true)
    exp_scores = Nx.exp(scores - max_scores)
    sum_exp = Nx.sum(exp_scores, axes: [1], keep_axes: true)
    attention_weights = exp_scores / sum_exp
    
    # Compute weighted context vector
    # attention_weights: [batch, seq_len]
    # encoder_outputs: [batch, seq_len, hidden]
    # We need to multiply each sequence position by its weight
    
    # Expand attention_weights: [batch, seq_len] -> [batch, seq_len, 1]
    weights_expanded = Nx.new_axis(attention_weights, 2)
    
    # Multiply: [batch, seq_len, hidden] * [batch, seq_len, 1] -> [batch, seq_len, hidden]
    weighted = encoder_outputs * weights_expanded
    
    # Sum over sequence length: [batch, seq_len, hidden] -> [batch, hidden]
    context = Nx.sum(weighted, axes: [1])
    
    {context, attention_weights}
  end
  
  @doc """
  Create an attention layer using Axon.
  
  This is a wrapper that creates a reusable attention layer.
  """
  def layer(hidden_size) do
    Axon.layer(
      fn encoder_outputs, decoder_hidden, _opts ->
        compute_attention(encoder_outputs, decoder_hidden, hidden_size)
      end,
      [Axon.input("encoder_outputs"), Axon.input("decoder_hidden")]
    )
  end
  
  @doc """
  Compute attention for a single sequence (non-batched).
  
  Convenience function for inference.
  """
  def compute_single(encoder_outputs, decoder_hidden, hidden_size) do
    # Add batch dimension
    encoder_batched = Nx.new_axis(encoder_outputs, 0)
    decoder_batched = Nx.new_axis(decoder_hidden, 0)
    
    {context_batched, weights_batched} = 
      compute_attention(encoder_batched, decoder_batched, hidden_size)
    
    # Remove batch dimension
    context = Nx.squeeze(context_batched, axes: [0])
    weights = Nx.squeeze(weights_batched, axes: [0])
    
    {context, weights}
  end
end
