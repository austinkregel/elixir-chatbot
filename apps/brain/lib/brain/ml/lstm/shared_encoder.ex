defmodule Brain.ML.LSTM.SharedEncoder do
  @moduledoc """
  Shared Bidirectional LSTM encoder for multi-task NLP models.
  
  This encoder produces both:
  - Token-level outputs: [batch, seq_len, hidden*2] for sequence tagging (NER, POS)
  - Sentence-level vector: [batch, hidden*2] for classification (intent)
  
  The encoder weights are shared across all downstream tasks, allowing
  the model to learn rich contextual representations that benefit
  intent classification, entity extraction, and POS tagging simultaneously.
  """
  
  require Logger

  @doc """
  Build shared encoder model using Axon.
  
  ## Parameters
  - `vocab_size`: Size of vocabulary
  - `embedding_size`: Dimension of word embeddings (default: 128)
  - `hidden_size`: Dimension of LSTM hidden state (default: 128)
  - `opts`: Additional options
    - `:num_layers` - Number of LSTM layers (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
  
  ## Returns
  A map containing:
  - `:token_outputs_model` - Axon model for token-level outputs [batch, seq_len, hidden*2]
  - `:sentence_vector_model` - Axon model for sentence-level vector [batch, hidden*2]
  - `:config` - Configuration used to build the model
  """
  def build_model(vocab_size, embedding_size \\ 128, hidden_size \\ 128, opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, 1)
    dropout = Keyword.get(opts, :dropout, 0.1)
    
    # Input: [batch, seq_len] (token indices)
    input = Axon.input("input", shape: {nil, nil})
    
    # Embedding layer - shared across all tasks
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size, name: "shared_embedding")
    
    # Bidirectional LSTM
    # Forward LSTM
    forward_lstm = 
      embedded
      |> build_lstm_stack(hidden_size, num_layers, dropout, "forward")
    
    # Backward LSTM (reverse sequence, process, reverse back)
    backward_lstm = 
      embedded
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse_input")
      |> build_lstm_stack(hidden_size, num_layers, dropout, "backward")
      |> Axon.nx(fn x -> Nx.reverse(x, axes: [1]) end, name: "reverse_output")
    
    # Token outputs: concatenate forward and backward for each position
    # Shape: [batch, seq_len, hidden_size * 2]
    token_outputs = 
      Axon.concatenate([forward_lstm, backward_lstm], axis: 2, name: "token_outputs")
    
    # Sentence vector: concatenate final states from both directions
    # Forward final: last timestep of forward LSTM
    forward_final = Axon.nx(forward_lstm, fn x -> 
      seq_len = Nx.axis_size(x, 1)
      Nx.slice_along_axis(x, seq_len - 1, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "forward_final")
    
    # Backward final: first timestep of backward LSTM (which processed reversed input)
    backward_final = Axon.nx(backward_lstm, fn x ->
      Nx.slice_along_axis(x, 0, 1, axis: 1)
      |> Nx.squeeze(axes: [1])
    end, name: "backward_final")
    
    # Concatenate final states: [batch, hidden_size * 2]
    sentence_vector = 
      Axon.concatenate([forward_final, backward_final], axis: 1, name: "sentence_vector")
    
    config = %{
      vocab_size: vocab_size,
      embedding_size: embedding_size,
      hidden_size: hidden_size,
      num_layers: num_layers,
      dropout: dropout,
      output_size: hidden_size * 2
    }
    
    %{
      token_outputs_model: token_outputs,
      sentence_vector_model: sentence_vector,
      config: config
    }
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
  Initialize encoder parameters for training.
  
  ## Parameters
  - `encoder`: Encoder map from `build_model/4`
  - `opts`: Options passed to Axon.build
  
  ## Returns
  Initialized parameters map
  """
  def init_params(encoder, opts \\ []) do
    # Use token_outputs_model since it contains all layers
    # (sentence_vector_model shares the same layers)
    {init_fn, _predict_fn} = Axon.build(encoder.token_outputs_model, opts)
    
    # Initialize with template and empty state
    template = %{"input" => Nx.template({1, 10}, :s64)}
    
    params = init_fn.(template, Axon.ModelState.empty())
    Axon.ModelState.new(params)
  end
  
  @doc """
  Encode input sequences to get both token outputs and sentence vector.
  
  ## Parameters
  - `encoder`: Encoder map from `build_model/4`
  - `input`: [batch, seq_len] tensor of token indices
  - `params`: Model parameters
  
  ## Returns
  `{token_outputs, sentence_vector}` where:
  - `token_outputs`: [batch, seq_len, hidden*2] - For NER and POS tagging
  - `sentence_vector`: [batch, hidden*2] - For intent classification
  """
  def encode(encoder, input, params) do
    token_outputs = Axon.predict(encoder.token_outputs_model, params, %{"input" => input})
    sentence_vector = Axon.predict(encoder.sentence_vector_model, params, %{"input" => input})
    
    {token_outputs, sentence_vector}
  end
  
  @doc """
  Encode a single sequence (non-batched).
  
  Convenience function for inference.
  
  ## Parameters
  - `encoder`: Encoder map from `build_model/4`
  - `input`: [seq_len] tensor of token indices
  - `params`: Model parameters
  
  ## Returns
  `{token_outputs, sentence_vector}` where:
  - `token_outputs`: [seq_len, hidden*2]
  - `sentence_vector`: [hidden*2]
  """
  def encode_single(encoder, input, params) do
    # Add batch dimension
    batched = Nx.new_axis(input, 0)
    
    {token_outputs, sentence_vector} = encode(encoder, batched, params)
    
    # Remove batch dimension
    {Nx.squeeze(token_outputs, axes: [0]), Nx.squeeze(sentence_vector, axes: [0])}
  end
  
  @doc """
  Get the output dimension of the encoder.
  
  This is hidden_size * 2 due to bidirectional concatenation.
  """
  def output_size(%{config: config}), do: config.output_size
  def output_size(hidden_size) when is_integer(hidden_size), do: hidden_size * 2
end
