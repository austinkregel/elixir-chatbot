defmodule Brain.ML.Seq2Seq.Encoder do
  @moduledoc """
  Bidirectional LSTM encoder for seq2seq models.
  
  Encodes input sequences into hidden states that can be used
  by the attention mechanism and decoder.
  """
  
  require Logger
  
  @doc """
  Build encoder model using Axon.
  
  ## Parameters
  - `vocab_size`: Size of vocabulary
  - `embedding_size`: Dimension of word embeddings
  - `hidden_size`: Dimension of LSTM hidden state
  - `num_layers`: Number of LSTM layers (default: 1)
  - `dropout`: Dropout rate (default: 0.1)
  
  ## Returns
  Axon model for encoding sequences.
  """
  def build_model(vocab_size, embedding_size, hidden_size, opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, 1)
    dropout = Keyword.get(opts, :dropout, 0.1)
    
    # Input: [batch, seq_len] (token indices)
    input = Axon.input("input", shape: {nil, nil})
    
    # Embedding layer
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size, name: "embedding")
    
    # Bidirectional LSTM
    # Forward LSTM
    forward_lstm = 
      embedded
      |> build_lstm_stack(hidden_size, num_layers, dropout, "forward")
    
    # Backward LSTM (reverse sequence)
    backward_lstm = 
      embedded
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse")
      |> build_lstm_stack(hidden_size, num_layers, dropout, "backward")
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse_back")
    
    # Concatenate forward and backward outputs
    # Each LSTM outputs [batch, seq_len, hidden_size]
    # Concatenated: [batch, seq_len, hidden_size * 2]
    outputs = 
      Axon.concatenate([forward_lstm, backward_lstm], axis: 2, name: "concat")
    
    # Also return final hidden states for decoder initialization
    # For simplicity, we'll use the last output of each direction
    forward_final = Axon.nx(forward_lstm, fn x -> 
      Nx.take(x, [Nx.axis_size(x, 1) - 1], axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "forward_final")
    
    backward_final = Axon.nx(backward_lstm, fn x ->
      Nx.take(x, [0], axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "backward_final")
    
    # Concatenate final states: [batch, hidden_size * 2]
    final_hidden = Axon.concatenate([forward_final, backward_final], axis: 1, name: "final_hidden")
    
    # Return tuple of models (outputs model and hidden model)
    # These are separate Axon models that share the same input
    {outputs, final_hidden}
  end
  
  defp build_lstm_stack(input, hidden_size, num_layers, dropout, prefix) do
    Enum.reduce(1..num_layers, input, fn layer_idx, x ->
      layer_name = "#{prefix}_lstm_#{layer_idx}"
      
      # Axon.lstm returns {output_sequence, {c_hidden, h_hidden}}
      # We only need the output sequence for stacking
      {lstm_output, _hidden_states} = Axon.lstm(x, hidden_size, name: layer_name)
      
      # Apply dropout to the output sequence if not last layer
      if layer_idx < num_layers and dropout > 0.0 do
        Axon.dropout(lstm_output, rate: dropout, name: "#{layer_name}_dropout")
      else
        lstm_output
      end
    end)
  end
  
  @doc """
  Encode a sequence using a trained model.
  
  ## Parameters
  - `model`: Trained Axon model
  - `input`: [batch, seq_len] tensor of token indices
  - `params`: Model parameters
  
  ## Returns
  - `{outputs, final_hidden}` where:
    - `outputs`: [batch, seq_len, hidden_size * 2] - Encoder outputs for attention
    - `final_hidden`: [batch, hidden_size * 2] - Final hidden state for decoder init
  """
  def encode(model, input, params) do
    {outputs_model, hidden_model} = model
    
    outputs = Axon.predict(outputs_model, params, %{"input" => input})
    final_hidden = Axon.predict(hidden_model, params, %{"input" => input})
    
    {outputs, final_hidden}
  end
  
  @doc """
  Encode a single sequence (non-batched).
  
  Convenience function for inference.
  """
  def encode_single(model, input, params) do
    # Add batch dimension
    batched = Nx.new_axis(input, 0)
    
    {outputs, final_hidden} = encode(model, batched, params)
    
    # Remove batch dimension
    outputs_single = Nx.squeeze(outputs, axes: [0])
    hidden_single = Nx.squeeze(final_hidden, axes: [0])
    
    {outputs_single, hidden_single}
  end
end
