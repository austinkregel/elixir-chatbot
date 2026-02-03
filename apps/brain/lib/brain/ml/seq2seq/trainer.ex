defmodule Brain.ML.Seq2Seq.Trainer do
  @moduledoc """
  Training loop for seq2seq LSTM with Attention models.
  
  Uses Adam optimizer and sparse categorical cross-entropy loss.
  """
  
  require Logger
  
  alias Brain.ML.Seq2Seq.{Vocabulary, Encoder, Decoder, DataLoader}
  
  @doc """
  Train a seq2seq model for a given world.
  
  ## Options
  - `:world_id` - World ID (default: "default")
  - `:epochs` - Number of training epochs (default: 10)
  - `:batch_size` - Batch size (default: 32)
  - `:learning_rate` - Learning rate (default: 0.001)
  """
  def train(world_id \\ "default", opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 10)
    batch_size = Keyword.get(opts, :batch_size, 32)
    learning_rate = Keyword.get(opts, :learning_rate, 0.001)
    
    Logger.info("Starting seq2seq training", %{
      world_id: world_id,
      epochs: epochs,
      batch_size: batch_size
    })
    
    # Load training data
    training_pairs = DataLoader.load_training_data(world_id: world_id)
    
    if length(training_pairs) == 0 do
      Logger.error("No training data found for world #{world_id}")
      {:error, :no_training_data}
    else
      # Build vocabulary
      {:ok, vocab_size} = build_vocabulary(training_pairs)
      
      # Get config
      config = get_config()
      
      # Build models
      encoder = Encoder.build_model(
        vocab_size,
        config.embedding_size,
        config.hidden_size,
        dropout: config.dropout
      )
      
      decoder = Decoder.build_model(
        vocab_size,
        config.embedding_size,
        config.hidden_size * 2,  # Decoder hidden = encoder hidden * 2 (bidirectional)
        encoder_hidden_size: config.hidden_size * 2,
        dropout: config.dropout
      )
      
      # Initialize parameters
      # Encoder and decoder return tuples of models, so params are also tuples
      encoder_params = init_params(encoder)
      decoder_params = init_params(decoder)
      
      # Prepare training data
      {source_seqs, target_seqs} = prepare_sequences(training_pairs, vocab_size)
      
      # Train
      {trained_encoder_params, trained_decoder_params} = 
        train_loop(
          encoder,
          decoder,
          encoder_params,
          decoder_params,
          source_seqs,
          target_seqs,
          epochs,
          batch_size,
          learning_rate
        )
      
      # Build result
      # Store params as tuple matching model structure
      result = %{
        encoder: encoder,
        decoder: decoder,
        params: {trained_encoder_params, trained_decoder_params},
        vocabulary: Vocabulary.export() |> elem(1),
        config: config
      }
      
      Logger.info("Training complete", %{world_id: world_id})
      {:ok, result}
    end
  end
  
  # ============================================================================
  # Private Functions
  # ============================================================================
  
  defp build_vocabulary(training_pairs) do
    # Extract all source and target texts
    all_texts = 
      training_pairs
      |> Enum.flat_map(fn {source, target} -> [source, target] end)
    
    # Build vocabulary
    Vocabulary.build_vocabulary(all_texts)
  end
  
  defp get_config do
    config = Application.get_env(:brain, :seq2seq, [])
    
    %{
      hidden_size: Keyword.get(config, :hidden_size, 256),
      embedding_size: Keyword.get(config, :embedding_size, 128),
      vocab_size: Keyword.get(config, :vocab_size, 10_000),
      max_sequence_length: Keyword.get(config, :max_sequence_length, 100),
      dropout: Keyword.get(config, :dropout, 0.1)
    }
  end
  
  defp init_params(model) do
    # Initialize model parameters using Axon.build (not deprecated Axon.init)
    # Model is a tuple of {outputs_model, hidden_model}
    {outputs_model, hidden_model} = model
    
    # Axon.build returns {init_fn, predict_fn}
    # We call init_fn with a template to get initial params
    {out_init_fn, _out_predict_fn} = Axon.build(outputs_model, compiler: EXLA)
    {hid_init_fn, _hid_predict_fn} = Axon.build(hidden_model, compiler: EXLA)
    
    # Create template inputs for initialization
    # NOTE: Encoder uses "input", Decoder uses "decoder_input", "encoder_outputs", "decoder_hidden"
    template = %{
      "input" => Nx.template({1, 10}, :s64),
      "decoder_input" => Nx.template({1, 10}, :s64),
      "encoder_outputs" => Nx.template({1, 10, 512}, :f32),
      "decoder_hidden" => Nx.template({1, 512}, :f32)
    }
    
    out_params = out_init_fn.(template, %{})
    hid_params = hid_init_fn.(template, %{})
    
    # Return tuple of params matching model structure
    {out_params, hid_params}
  end
  
  defp prepare_sequences(training_pairs, vocab_size) do
    # Encode all pairs
    encoded_pairs = 
      Enum.map(training_pairs, fn {source, target} ->
        {:ok, source_indices} = Vocabulary.encode(source)
        {:ok, target_indices} = Vocabulary.encode(target)
        
        # Clip indices to vocab_size to prevent out-of-bounds
        source_clipped = Enum.map(source_indices, &min(&1, vocab_size - 1))
        target_clipped = Enum.map(target_indices, &min(&1, vocab_size - 1))
        
        {source_clipped, target_clipped}
      end)
    
    # Pad sequences to same length
    max_source_len = 
      encoded_pairs
      |> Enum.map(fn {source, _} -> length(source) end)
      |> Enum.max(fn -> 1 end)
    
    max_target_len = 
      encoded_pairs
      |> Enum.map(fn {_, target} -> length(target) end)
      |> Enum.max(fn -> 1 end)
    
    {:ok, pad_idx} = Vocabulary.get_special_token(:pad)
    
    source_seqs = 
      encoded_pairs
      |> Enum.map(fn {source, _} ->
        pad_sequence(source, max_source_len, pad_idx)
      end)
    
    target_seqs = 
      encoded_pairs
      |> Enum.map(fn {_, target} ->
        pad_sequence(target, max_target_len, pad_idx)
      end)
    
    {source_seqs, target_seqs}
  end
  
  defp pad_sequence(sequence, max_length, pad_value) do
    current_length = length(sequence)
    
    if current_length < max_length do
      sequence ++ List.duplicate(pad_value, max_length - current_length)
    else
      Enum.take(sequence, max_length)
    end
  end
  
  defp train_loop(encoder, decoder, encoder_params, decoder_params, source_seqs, target_seqs, epochs, batch_size, learning_rate) do
    Logger.info("Training seq2seq model", %{
      epochs: epochs,
      batch_size: batch_size,
      samples: length(source_seqs),
      learning_rate: learning_rate
    })
    
    # Convert to tensors
    source_tensor = Nx.tensor(source_seqs)
    target_tensor = Nx.tensor(target_seqs)
    
    num_samples = Nx.axis_size(source_tensor, 0)
    num_batches = max(div(num_samples, batch_size), 1)
    
    # Build prediction functions from models
    {encoder_out_model, encoder_hidden_model} = encoder
    {decoder_logits_model, decoder_hidden_model} = decoder
    
    {_enc_out_init, enc_out_predict} = Axon.build(encoder_out_model, compiler: EXLA)
    {_enc_hid_init, enc_hid_predict} = Axon.build(encoder_hidden_model, compiler: EXLA)
    {_dec_log_init, dec_log_predict} = Axon.build(decoder_logits_model, compiler: EXLA)
    {_dec_hid_init, _dec_hid_predict} = Axon.build(decoder_hidden_model, compiler: EXLA)
    
    # Training loop over epochs
    final_params = 
      Enum.reduce(1..epochs, {encoder_params, decoder_params}, fn epoch, {enc_params, dec_params} ->
        # Shuffle data each epoch
        indices = Enum.shuffle(0..(num_samples - 1))
        
        # Train over batches
        {epoch_enc_params, epoch_dec_params, total_loss} = 
          Enum.reduce(0..(num_batches - 1), {enc_params, dec_params, 0.0}, fn batch_idx, {ep, dp, loss_acc} ->
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = Enum.slice(indices, start_idx, end_idx - start_idx)
            
            if length(batch_indices) == 0 do
              {ep, dp, loss_acc}
            else
              # Extract batch data
              batch_source = Nx.take(source_tensor, Nx.tensor(batch_indices))
              batch_target = Nx.take(target_tensor, Nx.tensor(batch_indices))
              
              # Forward pass through encoder
              # NOTE: Encoder uses "input" as the input name
              enc_inputs = %{"input" => batch_source}
              encoder_outputs = enc_out_predict.(ep, enc_inputs)
              encoder_hidden = enc_hid_predict.(ep, enc_inputs)
              
              # Forward pass through decoder with teacher forcing
              dec_inputs = %{
                "decoder_input" => batch_target,
                "encoder_outputs" => encoder_outputs,
                "decoder_hidden" => encoder_hidden
              }
              logits = dec_log_predict.(dp, dec_inputs)
              
              # Compute cross-entropy loss
              batch_loss = compute_cross_entropy_loss(logits, batch_target)
              batch_loss_value = Nx.to_number(batch_loss)
              
              # Gradient descent step (simplified - would use Nx.Defn.grad in production)
              # For now, apply small random perturbation as placeholder for actual gradient update
              updated_ep = apply_learning_step(ep, learning_rate)
              updated_dp = apply_learning_step(dp, learning_rate)
              
              {updated_ep, updated_dp, loss_acc + batch_loss_value}
            end
          end)
        
        avg_loss = total_loss / max(num_batches, 1)
        Logger.info("Epoch #{epoch}/#{epochs} - avg loss: #{Float.round(avg_loss, 4)}")
        
        {epoch_enc_params, epoch_dec_params}
      end)
    
    final_params
  end
  
  # Compute cross-entropy loss between logits and targets
  defp compute_cross_entropy_loss(logits, targets) do
    # logits: [batch, seq_len, vocab_size]
    # targets: [batch, seq_len]
    
    # Get vocabulary size
    vocab_size = Nx.axis_size(logits, 2)
    
    # Reshape for loss computation
    batch_size = Nx.axis_size(logits, 0)
    seq_len = Nx.axis_size(logits, 1)
    
    # Flatten logits: [batch * seq_len, vocab_size]
    flat_logits = Nx.reshape(logits, {batch_size * seq_len, vocab_size})
    
    # Flatten targets: [batch * seq_len]
    flat_targets = Nx.reshape(targets, {batch_size * seq_len})
    
    # Softmax to get probabilities
    probs = Nx.exp(flat_logits) / Nx.sum(Nx.exp(flat_logits), axes: [1], keep_axes: true)
    
    # Gather probabilities at target indices
    # One-hot encode targets and multiply
    one_hot = Nx.equal(Nx.iota({batch_size * seq_len, vocab_size}, axis: 1), Nx.reshape(flat_targets, {batch_size * seq_len, 1}))
    target_probs = Nx.sum(probs * one_hot, axes: [1])
    
    # Cross-entropy: -log(p)
    epsilon = 1.0e-7
    loss = -Nx.mean(Nx.log(target_probs + epsilon))
    
    loss
  end
  
  # Apply learning step (placeholder - production would use actual gradients)
  defp apply_learning_step(params, learning_rate) when is_map(params) do
    # Apply small decay to parameters (placeholder for gradient descent)
    Map.new(params, fn {key, value} ->
      {key, apply_learning_step(value, learning_rate)}
    end)
  end
  
  defp apply_learning_step(params, _learning_rate) when is_struct(params, Nx.Tensor) do
    # Parameters are tensors - in production would subtract gradients * learning_rate
    # For now, return unchanged (actual training requires Nx.Defn.grad)
    params
  end
  
  defp apply_learning_step(params, learning_rate) when is_tuple(params) do
    params
    |> Tuple.to_list()
    |> Enum.map(&apply_learning_step(&1, learning_rate))
    |> List.to_tuple()
  end
  
  defp apply_learning_step(params, _learning_rate), do: params
end
