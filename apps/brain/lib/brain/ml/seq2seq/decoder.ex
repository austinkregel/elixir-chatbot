defmodule Brain.ML.Seq2Seq.Decoder do
  @moduledoc """
  LSTM decoder with attention mechanism for seq2seq models.
  
  Generates output sequences by attending to encoder outputs.
  """
  
  require Logger
  
  @doc """
  Build decoder model using Axon.
  
  ## Parameters
  - `vocab_size`: Size of vocabulary
  - `embedding_size`: Dimension of word embeddings
  - `hidden_size`: Dimension of LSTM hidden state (should match encoder hidden_size * 2)
  - `num_layers`: Number of LSTM layers (default: 1)
  - `dropout`: Dropout rate (default: 0.1)
  
  ## Returns
  Axon model for decoding sequences with attention.
  """
  def build_model(vocab_size, embedding_size, hidden_size, opts \\ []) do
    num_layers = Keyword.get(opts, :num_layers, 1)
    dropout = Keyword.get(opts, :dropout, 0.1)
    encoder_hidden_size = Keyword.get(opts, :encoder_hidden_size, hidden_size)
    
    # Input: [batch, seq_len] (token indices)
    input = Axon.input("decoder_input", shape: {nil, nil})
    
    # Encoder outputs for attention: [batch, encoder_seq_len, encoder_hidden]
    encoder_outputs = Axon.input("encoder_outputs", shape: {nil, nil, encoder_hidden_size})
    
    # Previous decoder hidden state: [batch, hidden_size]
    decoder_hidden = Axon.input("decoder_hidden", shape: {nil, hidden_size})
    
    # Embedding layer
    embedded = 
      input
      |> Axon.embedding(vocab_size, embedding_size, name: "decoder_embedding")
    
    # Compute attention context
    # attention_context: [batch, seq_len, encoder_hidden]
    attention_context = 
      Axon.layer(
        fn embedded_seq, encoder_outs, hidden_state, _opts ->
          compute_attention_for_sequence(embedded_seq, encoder_outs, hidden_state, encoder_hidden_size)
        end,
        [embedded, encoder_outputs, decoder_hidden],
        name: "attention"
      )
    
    # Concatenate embedded input with attention context
    # embedded: [batch, seq_len, embedding_size]
    # attention_context: [batch, seq_len, encoder_hidden]
    # combined: [batch, seq_len, embedding_size + encoder_hidden]
    combined = 
      Axon.concatenate([embedded, attention_context], axis: 2, name: "concat_input_context")
    
    # Project to decoder hidden size
    projected = 
      combined
      |> Axon.dense(hidden_size, name: "input_projection")
    
    # LSTM layers
    lstm_output = 
      projected
      |> build_lstm_stack(hidden_size, num_layers, dropout)
    
    # Output projection to vocabulary
    logits = 
      lstm_output
      |> Axon.dense(vocab_size, name: "output_projection")
    
    # Also return final hidden state for next step
    final_hidden = 
      Axon.nx(lstm_output, fn x ->
        # Take last timestep: [batch, seq_len, hidden] -> [batch, hidden]
        Nx.take(x, [Nx.axis_size(x, 1) - 1], axis: 1)
        |> Nx.squeeze(axes: [1])
      end, name: "final_hidden")
    
    {logits, final_hidden}
  end
  
  defp build_lstm_stack(input, hidden_size, num_layers, dropout) do
    Enum.reduce(1..num_layers, input, fn layer_idx, x ->
      layer_name = "decoder_lstm_#{layer_idx}"
      
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
  
  # Compute attention for each timestep in the decoder sequence
  defp compute_attention_for_sequence(embedded_seq, encoder_outputs, decoder_hidden, encoder_hidden_size) do
    # embedded_seq: [batch, seq_len, embedding_size]
    # encoder_outputs: [batch, encoder_seq_len, encoder_hidden]
    # decoder_hidden: [batch, hidden_size]
    
    batch_size = Nx.axis_size(embedded_seq, 0)
    decoder_seq_len = Nx.axis_size(embedded_seq, 1)
    
    # For each timestep, compute attention
    # We'll use the decoder_hidden as the query for all timesteps
    # (in practice, you'd use the previous decoder hidden state)
    
    # Project decoder_hidden to match encoder_hidden_size for attention
    # (Currently unused as we use simplified attention)
    _decoder_hidden_proj = 
      decoder_hidden
      |> Nx.new_axis(1)  # [batch, 1, hidden]
      |> Nx.broadcast({batch_size, decoder_seq_len, Nx.axis_size(decoder_hidden, 1)})
    
    # Compute attention for each position
    # Simplified: use same attention for all positions
    # In full implementation, would use previous decoder hidden state
    
    # For now, compute attention once and broadcast
    {context, _weights} = 
      Brain.ML.Seq2Seq.Attention.compute_attention(
        encoder_outputs,
        decoder_hidden,
        encoder_hidden_size
      )
    
    # Broadcast context to all decoder timesteps
    context_expanded = 
      context
      |> Nx.new_axis(1)  # [batch, 1, encoder_hidden]
      |> Nx.broadcast({batch_size, decoder_seq_len, encoder_hidden_size})
    
    context_expanded
  end
  
  @doc """
  Generate a sequence using teacher forcing (training) or autoregressive (inference).
  
  ## Parameters
  - `model`: Trained Axon model
  - `params`: Model parameters
  - `encoder_outputs`: [batch, encoder_seq_len, encoder_hidden] - Encoder outputs
  - `decoder_hidden`: [batch, hidden_size] - Initial decoder hidden state
  - `decoder_input`: [batch, seq_len] - Decoder input tokens (for teacher forcing)
  - `max_length`: Maximum generation length (for inference)
  - `teacher_forcing`: Whether to use teacher forcing (default: false for inference)
  
  ## Returns
  - `{logits, final_hidden}` where:
    - `logits`: [batch, seq_len, vocab_size] - Output logits
    - `final_hidden`: [batch, hidden_size] - Final decoder hidden state
  """
  def generate(model, params, encoder_outputs, decoder_hidden, opts \\ []) do
    teacher_forcing = Keyword.get(opts, :teacher_forcing, false)
    decoder_input = Keyword.get(opts, :decoder_input)
    max_length = Keyword.get(opts, :max_length, 50)
    
    if teacher_forcing and decoder_input do
      # Training: use teacher forcing
      inputs = %{
        "decoder_input" => decoder_input,
        "encoder_outputs" => encoder_outputs,
        "decoder_hidden" => decoder_hidden
      }
      
      {logits_model, hidden_model} = model
      
      logits = Axon.predict(logits_model, params, inputs)
      final_hidden = Axon.predict(hidden_model, params, inputs)
      
      {logits, final_hidden}
    else
      # Inference: autoregressive generation
      generate_autoregressive(model, params, encoder_outputs, decoder_hidden, max_length)
    end
  end
  
  defp generate_autoregressive(model, params, encoder_outputs, decoder_hidden, max_length) do
    {logits_model, hidden_model} = model
    
    # Get vocabulary for special tokens
    {:ok, sos_idx} = Brain.ML.Seq2Seq.Vocabulary.get_special_token(:sos)
    {:ok, eos_idx} = Brain.ML.Seq2Seq.Vocabulary.get_special_token(:eos)
    
    batch_size = Nx.axis_size(decoder_hidden, 0)
    vocab_size = Nx.axis_size(encoder_outputs, 2)  # Approximate vocab size
    
    # Start with SOS token
    initial_input = Nx.broadcast(Nx.tensor(sos_idx), {batch_size, 1})
    
    # Initial state for reduction
    initial_state = %{
      current_input: initial_input,
      current_hidden: decoder_hidden,
      all_logits: []
    }
    
    # Generate step by step using Enum.reduce_while for proper state management
    final_state = 
      Enum.reduce_while(1..max_length, initial_state, fn _step, state ->
        inputs = %{
          "decoder_input" => state.current_input,
          "encoder_outputs" => encoder_outputs,
          "decoder_hidden" => state.current_hidden
        }
        
        step_logits = Axon.predict(logits_model, params, inputs)
        step_hidden = Axon.predict(hidden_model, params, inputs)
        
        # Get predicted token (greedy)
        predicted = Nx.argmax(step_logits, axis: 2)
        
        # Update state
        new_state = %{
          current_input: predicted,
          current_hidden: step_hidden,
          all_logits: [step_logits | state.all_logits]
        }
        
        # Check for EOS - halt if all sequences have predicted EOS
        is_all_eos = 
          Nx.all(Nx.equal(predicted, eos_idx))
          |> Nx.to_number()
          |> Kernel.==(1)
        
        if is_all_eos do
          {:halt, new_state}
        else
          {:cont, new_state}
        end
      end)
    
    # Concatenate all logits
    final_logits = build_final_logits(final_state.all_logits, batch_size, vocab_size)
    
    {final_logits, final_state.current_hidden}
  end
  
  # Build final logits tensor from accumulated list
  defp build_final_logits(all_logits, batch_size, vocab_size) do
    if length(all_logits) > 0 do
      Nx.concatenate(Enum.reverse(all_logits), axis: 1)
    else
      # Fallback: return empty logits
      Nx.broadcast(Nx.tensor(0.0), {batch_size, 0, vocab_size})
    end
  end
end
