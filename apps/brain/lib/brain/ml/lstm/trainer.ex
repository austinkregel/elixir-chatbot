defmodule Brain.ML.LSTM.Trainer do
  @moduledoc """
  Training loop for multi-task LSTM models.
  
  Supports training:
  - Intent classification (Phase 1)
  - Named Entity Recognition (Phase 2)
  - POS tagging (Phase 3)
  
  Uses Adam optimizer with configurable learning rate and supports
  early stopping based on validation loss.
  """
  
  require Logger
  
  alias Brain.ML.DataLoaders
  alias Brain.ML.LSTM.{SharedEncoder, IntentHead, NERHead, POSHead}
  
  @default_config %{
    embedding_size: 128,
    hidden_size: 128,
    num_layers: 1,
    dropout: 0.1,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10,
    max_seq_length: 50,
    min_vocab_freq: 2,
    max_vocab_size: 10000,
    validation_split: 0.1,
    early_stopping_patience: 3
  }
  
  @doc """
  Train an intent classification model using LSTM.
  
  ## Options
  - `:embedding_size` - Dimension of word embeddings (default: 128)
  - `:hidden_size` - LSTM hidden dimension (default: 128)
  - `:num_layers` - Number of LSTM layers (default: 1)
  - `:dropout` - Dropout rate (default: 0.1)
  - `:learning_rate` - Adam learning rate (default: 0.001)
  - `:batch_size` - Training batch size (default: 32)
  - `:epochs` - Number of training epochs (default: 10)
  - `:max_seq_length` - Maximum sequence length (default: 50)
  
  ## Returns
  `{:ok, trained_model}` or `{:error, reason}`
  """
  def train_intent_classifier(opts \\ []) do
    config = build_config(opts)
    
    Logger.info("Starting LSTM intent classifier training", %{config: config})
    
    with {:ok, raw_examples} <- DataLoaders.load_intent_training_data_for_lstm(),
         {:ok, train_data, val_data, vocabularies} <- prepare_training_data(raw_examples, config) do
      
      # Build models
      {encoder, intent_head} = build_intent_model(vocabularies, config)
      
      # Initialize parameters
      encoder_params = SharedEncoder.init_params(encoder)
      intent_params = IntentHead.init_params(intent_head, config.hidden_size * 2)
      
      # Train
      {trained_encoder_params, trained_intent_params, metrics} = 
        train_loop(
          encoder,
          intent_head,
          encoder_params,
          intent_params,
          train_data,
          val_data,
          vocabularies,
          config
        )
      
      # Build result model
      trained_model = %{
        encoder: encoder,
        intent_head: intent_head,
        encoder_params: trained_encoder_params,
        intent_params: trained_intent_params,
        vocabularies: vocabularies,
        config: config,
        metrics: metrics
      }
      
      # Save model
      save_intent_model(trained_model)
      
      {:ok, trained_model}
    end
  end
  
  @doc """
  Load a trained intent classification model.
  """
  def load_intent_model do
    model_path = get_model_path("lstm_intent.term")
    
    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          e -> {:error, "Failed to deserialize model: #{inspect(e)}"}
        end
      
      {:error, reason} ->
        {:error, "Failed to read model file: #{reason}"}
    end
  end
  
  @doc """
  Classify text using a trained model.
  """
  def classify(text, model) do
    tokens = Brain.ML.Tokenizer.tokenize(text)
    indices = DataLoaders.tokens_to_indices(tokens, model.vocabularies.token_vocab)
    padded = DataLoaders.pad_sequence(indices, model.config.max_seq_length)
    
    input = Nx.tensor([padded], type: :s64)
    
    {_token_outputs, sentence_vector} = 
      SharedEncoder.encode(model.encoder, input, model.encoder_params)
    
    sentence_vector = Nx.squeeze(sentence_vector, axes: [0])
    
    IntentHead.classify(
      model.intent_head,
      sentence_vector,
      model.intent_params,
      Map.values(model.vocabularies.idx_to_intent) |> Enum.sort_by(&model.vocabularies.intent_to_idx[&1])
    )
  end
  
  # ============================================================================
  # Joint Intent + NER Training
  # ============================================================================
  
  @doc """
  Train a joint model for intent classification and NER.
  
  This trains the shared encoder with both tasks, allowing the model to
  learn representations that benefit both intent classification and
  entity extraction.
  
  ## Options
  Same as `train_intent_classifier/1` plus:
  - `:intent_loss_weight` - Weight for intent loss (default: 0.5)
  - `:ner_loss_weight` - Weight for NER loss (default: 0.5)
  
  ## Returns
  `{:ok, trained_model}` with both intent and NER heads
  """
  def train_joint(opts \\ []) do
    config = build_config(opts) |> add_joint_config(opts)
    
    Logger.info("Starting joint intent + NER training", %{config: config})
    
    with {:ok, raw_examples} <- DataLoaders.load_intent_training_data_for_lstm(),
         {:ok, train_data, val_data, vocabularies} <- prepare_joint_training_data(raw_examples, config) do
      
      # Build models
      {encoder, intent_head, ner_head} = build_joint_model(vocabularies, config)
      
      # Initialize parameters
      encoder_params = SharedEncoder.init_params(encoder)
      intent_params = IntentHead.init_params(intent_head, config.hidden_size * 2)
      ner_params = NERHead.init_params(ner_head, config.hidden_size * 2)
      
      # Train
      {trained_encoder_params, trained_intent_params, trained_ner_params, metrics} = 
        joint_train_loop(
          encoder, intent_head, ner_head,
          encoder_params, intent_params, ner_params,
          train_data, val_data,
          vocabularies, config
        )
      
      # Build result model
      trained_model = %{
        encoder: encoder,
        intent_head: intent_head,
        ner_head: ner_head,
        encoder_params: trained_encoder_params,
        intent_params: trained_intent_params,
        ner_params: trained_ner_params,
        vocabularies: vocabularies,
        config: config,
        metrics: metrics
      }
      
      # Save model
      save_joint_model(trained_model)
      
      {:ok, trained_model}
    end
  end
  
  @doc """
  Load a trained joint model.
  """
  def load_joint_model do
    model_path = get_model_path("lstm_joint.term")
    
    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          e -> {:error, "Failed to deserialize model: #{inspect(e)}"}
        end
      
      {:error, reason} ->
        {:error, "Failed to read model file: #{reason}"}
    end
  end
  
  @doc """
  Classify intent and extract entities using a joint model.
  """
  def analyze_joint(text, model) do
    tokens = Brain.ML.Tokenizer.tokenize(text)
    indices = DataLoaders.tokens_to_indices(tokens, model.vocabularies.token_vocab)
    padded = DataLoaders.pad_sequence(indices, model.config.max_seq_length)
    
    input = Nx.tensor([padded], type: :s64)
    
    {token_outputs, sentence_vector} = 
      SharedEncoder.encode(model.encoder, input, model.encoder_params)
    
    # Intent classification
    sentence_vector_squeezed = Nx.squeeze(sentence_vector, axes: [0])
    intent_labels = 
      model.vocabularies.idx_to_intent 
      |> Map.values() 
      |> Enum.sort_by(&model.vocabularies.intent_to_idx[&1])
    
    {intent, intent_confidence, intent_scores} = 
      IntentHead.classify(
        model.intent_head,
        sentence_vector_squeezed,
        model.intent_params,
        intent_labels
      )
    
    # NER
    token_outputs_squeezed = Nx.squeeze(token_outputs, axes: [0])
    bio_labels = 
      model.vocabularies.idx_to_bio 
      |> Map.values() 
      |> Enum.sort_by(&model.vocabularies.bio_to_idx[&1])
    
    entities = NERHead.extract_entities(
      model.ner_head,
      tokens,
      token_outputs_squeezed,
      model.ner_params,
      bio_labels
    )
    
    %{
      intent: %{label: intent, confidence: intent_confidence, scores: intent_scores},
      entities: entities,
      tokens: tokens
    }
  end
  
  defp add_joint_config(config, opts) do
    Map.merge(config, %{
      intent_loss_weight: Keyword.get(opts, :intent_loss_weight, 0.5),
      ner_loss_weight: Keyword.get(opts, :ner_loss_weight, 0.5)
    })
  end
  
  defp prepare_joint_training_data(examples, config) do
    # Build vocabularies
    token_vocab = DataLoaders.build_lstm_vocabulary(examples, 
      min_freq: config.min_vocab_freq,
      max_vocab: config.max_vocab_size
    )
    
    {intent_to_idx, idx_to_intent} = DataLoaders.build_intent_vocabulary(examples)
    {bio_to_idx, idx_to_bio} = DataLoaders.build_bio_vocabulary(examples)
    
    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent,
      bio_to_idx: bio_to_idx,
      idx_to_bio: idx_to_bio
    }
    
    # Convert examples to tensors with BIO tags
    processed = 
      examples
      |> Enum.map(fn ex -> 
        indices = DataLoaders.tokens_to_indices(ex.tokens, token_vocab)
        padded_input = DataLoaders.pad_sequence(indices, config.max_seq_length)
        intent_idx = Map.get(intent_to_idx, ex.intent, 0)
        
        # Convert BIO tags to indices
        bio_indices = 
          ex.bio_tags
          |> Enum.map(fn tag -> Map.get(bio_to_idx, tag, 0) end)
        
        padded_bio = DataLoaders.pad_sequence(bio_indices, config.max_seq_length, 0)
        
        # Create mask for real tokens (not padding)
        mask = 
          for i <- 0..(config.max_seq_length - 1) do
            if i < length(indices), do: 1, else: 0
          end
        
        %{
          input: padded_input,
          intent: intent_idx,
          bio_tags: padded_bio,
          mask: mask,
          negative_for: ex.negative_for
        }
      end)
    
    # Shuffle and split
    shuffled = Enum.shuffle(processed)
    split_idx = round(length(shuffled) * (1 - config.validation_split))
    {train_list, val_list} = Enum.split(shuffled, split_idx)
    
    train_data = %{
      inputs: Enum.map(train_list, & &1.input),
      intents: Enum.map(train_list, & &1.intent),
      bio_tags: Enum.map(train_list, & &1.bio_tags),
      masks: Enum.map(train_list, & &1.mask),
      negatives: Enum.map(train_list, & &1.negative_for)
    }
    
    val_data = %{
      inputs: Enum.map(val_list, & &1.input),
      intents: Enum.map(val_list, & &1.intent),
      bio_tags: Enum.map(val_list, & &1.bio_tags),
      masks: Enum.map(val_list, & &1.mask),
      negatives: Enum.map(val_list, & &1.negative_for)
    }
    
    Logger.info("Prepared joint training data", %{
      train_size: length(train_list),
      val_size: length(val_list),
      vocab_size: map_size(token_vocab),
      num_intents: map_size(intent_to_idx),
      num_bio_tags: map_size(bio_to_idx)
    })
    
    {:ok, train_data, val_data, vocabularies}
  end
  
  defp build_joint_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    num_bio_tags = map_size(vocabularies.bio_to_idx)
    
    encoder = SharedEncoder.build_model(
      vocab_size,
      config.embedding_size,
      config.hidden_size,
      num_layers: config.num_layers,
      dropout: config.dropout
    )
    
    intent_head = IntentHead.build_model(
      config.hidden_size * 2,
      num_intents,
      dropout: config.dropout
    )
    
    ner_head = NERHead.build_model(
      config.hidden_size * 2,
      num_bio_tags,
      dropout: config.dropout
    )
    
    {encoder, intent_head, ner_head}
  end
  
  defp joint_train_loop(encoder, intent_head, ner_head,
                         encoder_params, intent_params, ner_params,
                         train_data, val_data, _vocabularies, config) do
    
    num_batches = div(length(train_data.inputs), config.batch_size)
    num_samples = length(train_data.inputs)
    
    # Log training start
    Logger.info("Starting joint training (intent + NER):")
    Logger.info("  Samples: #{num_samples} (train) / #{length(val_data.inputs)} (val)")
    Logger.info("  Batches per epoch: #{num_batches}")
    Logger.info("  Epochs: #{config.epochs}")
    Logger.info("  Batch size: #{config.batch_size}")
    Logger.info("  Learning rate: #{config.learning_rate}")
    
    # Initialize optimizer states for all parameter sets
    {enc_opt_init, enc_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {int_opt_init, int_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {ner_opt_init, ner_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    
    initial_state = %{
      encoder_params: encoder_params,
      intent_params: intent_params,
      ner_params: ner_params,
      enc_opt_state: enc_opt_init.(encoder_params),
      int_opt_state: int_opt_init.(intent_params),
      ner_opt_state: ner_opt_init.(ner_params),
      best_val_loss: :infinity,
      patience_counter: 0,
      metrics: []
    }
    
    final_state = 
      Enum.reduce_while(1..config.epochs, initial_state, fn epoch, state ->
        epoch_start = System.monotonic_time(:millisecond)
        indices = Enum.shuffle(0..(length(train_data.inputs) - 1))
        
        log_epoch_progress(epoch, config.epochs, :start)
        
        {epoch_loss, updated_state} = 
          joint_train_epoch(
            encoder, intent_head, ner_head,
            train_data, indices, state, config, num_batches,
            enc_opt_update, int_opt_update, ner_opt_update
          )
        
        val_loss = joint_validate(
          encoder, intent_head, ner_head,
          val_data,
          updated_state.encoder_params,
          updated_state.intent_params,
          updated_state.ner_params,
          config
        )
        
        epoch_duration = System.monotonic_time(:millisecond) - epoch_start
        
        epoch_metrics = %{
          epoch: epoch,
          train_loss: epoch_loss,
          val_loss: val_loss,
          duration_ms: epoch_duration
        }
        
        log_epoch_progress(epoch, config.epochs, :complete, epoch_metrics)
        
        {continue?, updated_state} = 
          check_early_stopping(updated_state, val_loss, config, epoch_metrics)
        
        if continue? do
          {:cont, updated_state}
        else
          Logger.info("Early stopping triggered at epoch #{epoch}")
          {:halt, updated_state}
        end
      end)
    
    Logger.info("Training complete!")
    
    {final_state.encoder_params, final_state.intent_params, 
     final_state.ner_params, final_state.metrics}
  end
  
  defp joint_train_epoch(encoder, intent_head, ner_head, train_data, indices, state, config, 
                          num_batches, enc_opt_update, int_opt_update, ner_opt_update) do
    
    batch_size = config.batch_size
    
    Enum.reduce(0..(num_batches - 1), {0.0, state}, fn batch_idx, {total_loss, acc_state} ->
      start_idx = batch_idx * batch_size
      batch_indices = Enum.slice(indices, start_idx, batch_size)
      
      if length(batch_indices) == 0 do
        {total_loss, acc_state}
      else
        batch_inputs = 
          batch_indices
          |> Enum.map(&Enum.at(train_data.inputs, &1))
          |> Nx.tensor(type: :s64)
        
        batch_intents =
          batch_indices
          |> Enum.map(&Enum.at(train_data.intents, &1))
          |> Nx.tensor(type: :s64)
        
        batch_bio_tags =
          batch_indices
          |> Enum.map(&Enum.at(train_data.bio_tags, &1))
          |> Nx.tensor(type: :s64)
        
        batch_masks =
          batch_indices
          |> Enum.map(&Enum.at(train_data.masks, &1))
          |> Nx.tensor(type: :f32)
        
        # Compute joint loss and gradients
        {loss, {enc_grads, int_grads, ner_grads}} = 
          compute_joint_loss_and_grads(
            encoder, intent_head, ner_head,
            batch_inputs, batch_intents, batch_bio_tags, batch_masks,
            acc_state.encoder_params, acc_state.intent_params, acc_state.ner_params,
            config
          )
        
        loss_value = Nx.to_number(loss)
        
        # Update all parameters
        {enc_updates, new_enc_opt} = 
          enc_opt_update.(enc_grads, acc_state.enc_opt_state, acc_state.encoder_params)
        
        {int_updates, new_int_opt} = 
          int_opt_update.(int_grads, acc_state.int_opt_state, acc_state.intent_params)
        
        {ner_updates, new_ner_opt} = 
          ner_opt_update.(ner_grads, acc_state.ner_opt_state, acc_state.ner_params)
        
        new_state = %{acc_state |
          encoder_params: apply_updates(acc_state.encoder_params, enc_updates),
          intent_params: apply_updates(acc_state.intent_params, int_updates),
          ner_params: apply_updates(acc_state.ner_params, ner_updates),
          enc_opt_state: new_enc_opt,
          int_opt_state: new_int_opt,
          ner_opt_state: new_ner_opt
        }
        
        {total_loss + loss_value, new_state}
      end
    end)
    |> then(fn {total_loss, state} ->
      {total_loss / max(num_batches, 1), state}
    end)
  end
  
  defp compute_joint_loss_and_grads(encoder, intent_head, ner_head,
                                     inputs, intent_targets, bio_targets, masks,
                                     encoder_params, intent_params, ner_params, config) do
    
    # Forward pass
    forward_fn = fn enc_p, int_p, ner_p ->
      {token_outputs, sentence_vector} = 
        SharedEncoder.encode(encoder, inputs, enc_p)
      
      intent_preds = IntentHead.forward(intent_head, sentence_vector, int_p)
      ner_preds = NERHead.forward(ner_head, token_outputs, ner_p)
      
      intent_loss = IntentHead.compute_loss(intent_preds, intent_targets)
      ner_loss = NERHead.compute_loss(ner_preds, bio_targets, masks)
      
      # Weighted sum of losses
      config.intent_loss_weight * intent_loss + config.ner_loss_weight * ner_loss
    end
    
    # Compute gradients using nested value_and_grad
    # This is a simplified approach - compute gradients for each parameter set
    grad_fn = fn enc_p ->
      Nx.Defn.value_and_grad(fn int_p ->
        Nx.Defn.value_and_grad(fn ner_p ->
          forward_fn.(enc_p, int_p, ner_p)
        end, ner_params)
      end, intent_params)
    end
    
    {{{loss, ner_grads}, int_grads}, enc_grads} = 
      Nx.Defn.value_and_grad(grad_fn, encoder_params).()
    
    {loss, {enc_grads, int_grads, ner_grads}}
  end
  
  defp joint_validate(encoder, intent_head, ner_head, val_data, 
                       encoder_params, intent_params, ner_params, config) do
    if length(val_data.inputs) == 0 do
      0.0
    else
      batch_size = config.batch_size
      num_batches = div(length(val_data.inputs), batch_size) + 1
      
      total_loss = 
        Enum.reduce(0..(num_batches - 1), 0.0, fn batch_idx, acc ->
          start_idx = batch_idx * batch_size
          batch_inputs = Enum.slice(val_data.inputs, start_idx, batch_size)
          batch_intents = Enum.slice(val_data.intents, start_idx, batch_size)
          batch_bio = Enum.slice(val_data.bio_tags, start_idx, batch_size)
          batch_masks = Enum.slice(val_data.masks, start_idx, batch_size)
          
          if length(batch_inputs) == 0 do
            acc
          else
            inputs = Nx.tensor(batch_inputs, type: :s64)
            intent_targets = Nx.tensor(batch_intents, type: :s64)
            bio_targets = Nx.tensor(batch_bio, type: :s64)
            masks = Nx.tensor(batch_masks, type: :f32)
            
            {token_outputs, sentence_vector} = 
              SharedEncoder.encode(encoder, inputs, encoder_params)
            
            intent_preds = IntentHead.forward(intent_head, sentence_vector, intent_params)
            ner_preds = NERHead.forward(ner_head, token_outputs, ner_params)
            
            intent_loss = IntentHead.compute_loss(intent_preds, intent_targets) |> Nx.to_number()
            ner_loss = NERHead.compute_loss(ner_preds, bio_targets, masks) |> Nx.to_number()
            
            loss = config.intent_loss_weight * intent_loss + config.ner_loss_weight * ner_loss
            acc + loss
          end
        end)
      
      total_loss / max(num_batches, 1)
    end
  end
  
  defp save_joint_model(model) do
    model_path = get_model_path("lstm_joint.term")
    File.mkdir_p!(Path.dirname(model_path))
    binary = :erlang.term_to_binary(model)
    File.write!(model_path, binary)
    Logger.info("Saved joint LSTM model", %{path: model_path})
  end

  # ============================================================================
  # Full Multi-Task Training (Intent + NER + POS)
  # ============================================================================
  
  @doc """
  Train a full multi-task model with all three heads: intent, NER, and POS.
  
  This is the most comprehensive training mode, where the shared encoder
  learns representations that benefit intent classification, entity extraction,
  and part-of-speech tagging simultaneously.
  
  ## Options
  Same as `train_joint/1` plus:
  - `:pos_loss_weight` - Weight for POS loss (default: 0.33)
  
  Note: Loss weights will be normalized to sum to 1.
  
  ## Returns
  `{:ok, trained_model}` with intent, NER, and POS heads
  """
  def train_multitask(opts \\ []) do
    config = build_config(opts) |> add_multitask_config(opts)
    
    Logger.info("Starting full multi-task training", %{config: config})
    
    with {:ok, raw_examples} <- DataLoaders.load_intent_training_data_for_lstm(),
         {:ok, train_data, val_data, vocabularies} <- prepare_multitask_data(raw_examples, config) do
      
      # Build all models
      {encoder, intent_head, ner_head, pos_head} = build_multitask_model(vocabularies, config)
      
      # Initialize parameters
      encoder_params = SharedEncoder.init_params(encoder)
      intent_params = IntentHead.init_params(intent_head, config.hidden_size * 2)
      ner_params = NERHead.init_params(ner_head, config.hidden_size * 2)
      pos_params = POSHead.init_params(pos_head, config.hidden_size * 2)
      
      # Train
      {trained_enc, trained_int, trained_ner, trained_pos, metrics} = 
        multitask_train_loop(
          encoder, intent_head, ner_head, pos_head,
          encoder_params, intent_params, ner_params, pos_params,
          train_data, val_data,
          vocabularies, config
        )
      
      # Build result model
      trained_model = %{
        encoder: encoder,
        intent_head: intent_head,
        ner_head: ner_head,
        pos_head: pos_head,
        encoder_params: trained_enc,
        intent_params: trained_int,
        ner_params: trained_ner,
        pos_params: trained_pos,
        vocabularies: vocabularies,
        config: config,
        metrics: metrics
      }
      
      # Save model
      save_multitask_model(trained_model)
      
      {:ok, trained_model}
    end
  end
  
  @doc """
  Load a trained multi-task model.
  """
  def load_multitask_model do
    model_path = get_model_path("lstm_multitask.term")
    
    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          e -> {:error, "Failed to deserialize model: #{inspect(e)}"}
        end
      
      {:error, reason} ->
        {:error, "Failed to read model file: #{reason}"}
    end
  end
  
  @doc """
  Full analysis using multi-task model: intent, entities, and POS tags.
  """
  def analyze_multitask(text, model) do
    tokens = Brain.ML.Tokenizer.tokenize(text)
    indices = DataLoaders.tokens_to_indices(tokens, model.vocabularies.token_vocab)
    padded = DataLoaders.pad_sequence(indices, model.config.max_seq_length)
    
    input = Nx.tensor([padded], type: :s64)
    
    {token_outputs, sentence_vector} = 
      SharedEncoder.encode(model.encoder, input, model.encoder_params)
    
    # Intent classification
    sentence_vector_squeezed = Nx.squeeze(sentence_vector, axes: [0])
    intent_labels = get_sorted_labels(model.vocabularies.idx_to_intent, model.vocabularies.intent_to_idx)
    
    {intent, intent_confidence, intent_scores} = 
      IntentHead.classify(
        model.intent_head,
        sentence_vector_squeezed,
        model.intent_params,
        intent_labels
      )
    
    token_outputs_squeezed = Nx.squeeze(token_outputs, axes: [0])
    
    # NER
    bio_labels = get_sorted_labels(model.vocabularies.idx_to_bio, model.vocabularies.bio_to_idx)
    entities = NERHead.extract_entities(
      model.ner_head,
      tokens,
      token_outputs_squeezed,
      model.ner_params,
      bio_labels
    )
    
    # POS tagging
    pos_labels = get_sorted_labels(model.vocabularies.idx_to_pos, model.vocabularies.pos_to_idx)
    pos_tags = POSHead.tag_tokens(
      model.pos_head,
      tokens,
      token_outputs_squeezed,
      model.pos_params,
      pos_labels
    )
    
    %{
      intent: %{label: intent, confidence: intent_confidence, scores: intent_scores},
      entities: entities,
      pos_tags: pos_tags,
      tokens: tokens
    }
  end
  
  defp get_sorted_labels(idx_to_label, label_to_idx) do
    idx_to_label 
    |> Map.values() 
    |> Enum.sort_by(&label_to_idx[&1])
  end
  
  defp add_multitask_config(config, opts) do
    intent_w = Keyword.get(opts, :intent_loss_weight, 0.34)
    ner_w = Keyword.get(opts, :ner_loss_weight, 0.33)
    pos_w = Keyword.get(opts, :pos_loss_weight, 0.33)
    
    # Normalize weights
    total = intent_w + ner_w + pos_w
    
    Map.merge(config, %{
      intent_loss_weight: intent_w / total,
      ner_loss_weight: ner_w / total,
      pos_loss_weight: pos_w / total
    })
  end
  
  defp prepare_multitask_data(examples, config) do
    # Build vocabularies
    token_vocab = DataLoaders.build_lstm_vocabulary(examples, 
      min_freq: config.min_vocab_freq,
      max_vocab: config.max_vocab_size
    )
    
    {intent_to_idx, idx_to_intent} = DataLoaders.build_intent_vocabulary(examples)
    {bio_to_idx, idx_to_bio} = DataLoaders.build_bio_vocabulary(examples)
    {pos_to_idx, idx_to_pos} = POSHead.build_pos_vocabulary()
    
    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent,
      bio_to_idx: bio_to_idx,
      idx_to_bio: idx_to_bio,
      pos_to_idx: pos_to_idx,
      idx_to_pos: idx_to_pos
    }
    
    # Convert examples with POS tags
    # Note: POS tags need to be generated - for now we use heuristics
    processed = 
      examples
      |> Enum.map(fn ex -> 
        indices = DataLoaders.tokens_to_indices(ex.tokens, token_vocab)
        padded_input = DataLoaders.pad_sequence(indices, config.max_seq_length)
        intent_idx = Map.get(intent_to_idx, ex.intent, 0)
        
        # BIO tags
        bio_indices = 
          ex.bio_tags
          |> Enum.map(fn tag -> Map.get(bio_to_idx, tag, 0) end)
        padded_bio = DataLoaders.pad_sequence(bio_indices, config.max_seq_length, 0)
        
        # POS tags - use simple heuristics for now (proper POS comes from POS tagger)
        pos_indices = generate_pos_indices(ex.tokens, pos_to_idx)
        padded_pos = DataLoaders.pad_sequence(pos_indices, config.max_seq_length, 0)
        
        # Mask
        mask = 
          for i <- 0..(config.max_seq_length - 1) do
            if i < length(indices), do: 1, else: 0
          end
        
        %{
          input: padded_input,
          intent: intent_idx,
          bio_tags: padded_bio,
          pos_tags: padded_pos,
          mask: mask,
          negative_for: ex.negative_for
        }
      end)
    
    # Shuffle and split
    shuffled = Enum.shuffle(processed)
    split_idx = round(length(shuffled) * (1 - config.validation_split))
    {train_list, val_list} = Enum.split(shuffled, split_idx)
    
    train_data = %{
      inputs: Enum.map(train_list, & &1.input),
      intents: Enum.map(train_list, & &1.intent),
      bio_tags: Enum.map(train_list, & &1.bio_tags),
      pos_tags: Enum.map(train_list, & &1.pos_tags),
      masks: Enum.map(train_list, & &1.mask),
      negatives: Enum.map(train_list, & &1.negative_for)
    }
    
    val_data = %{
      inputs: Enum.map(val_list, & &1.input),
      intents: Enum.map(val_list, & &1.intent),
      bio_tags: Enum.map(val_list, & &1.bio_tags),
      pos_tags: Enum.map(val_list, & &1.pos_tags),
      masks: Enum.map(val_list, & &1.mask),
      negatives: Enum.map(val_list, & &1.negative_for)
    }
    
    Logger.info("Prepared multi-task training data", %{
      train_size: length(train_list),
      val_size: length(val_list),
      vocab_size: map_size(token_vocab),
      num_intents: map_size(intent_to_idx),
      num_bio_tags: map_size(bio_to_idx),
      num_pos_tags: map_size(pos_to_idx)
    })
    
    {:ok, train_data, val_data, vocabularies}
  end
  
  # Simple heuristic POS tagging for training data
  # In production, this would use the existing POS tagger
  defp generate_pos_indices(tokens, pos_to_idx) do
    Enum.map(tokens, fn token ->
      tag = heuristic_pos_tag(token)
      Map.get(pos_to_idx, tag, 0)
    end)
  end
  
  defp heuristic_pos_tag(token) do
    lower = String.downcase(token)
    
    cond do
      # Pronouns
      lower in ~w(i me my mine myself you your yours yourself he him his 
                  she her hers it its we us our they them their) -> "PRON"
      
      # Determiners
      lower in ~w(the a an this that these those some any) -> "DET"
      
      # Prepositions
      lower in ~w(in on at to for from by with about of into) -> "ADP"
      
      # Conjunctions
      lower in ~w(and or but if because while when) -> "CONJ"
      
      # Auxiliary verbs
      lower in ~w(is am are was were be been being have has had do does did 
                  will would shall should can could may might must) -> "AUX"
      
      # Common verbs
      lower in ~w(play turn show tell get make find want need like know think) -> "VERB"
      
      # Adverbs
      lower in ~w(please now today tomorrow here there very really) -> "ADV"
      
      # Numbers
      Regex.match?(~r/^\d+$/, token) -> "NUM"
      
      # Punctuation
      Regex.match?(~r/^[.,!?;:]$/, token) -> "PUNCT"
      
      # Proper noun (capitalized)
      String.match?(token, ~r/^[A-Z][a-z]+$/) -> "PROPN"
      
      # Default to noun
      true -> "NOUN"
    end
  end
  
  defp build_multitask_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    num_bio_tags = map_size(vocabularies.bio_to_idx)
    num_pos_tags = map_size(vocabularies.pos_to_idx)
    
    encoder = SharedEncoder.build_model(
      vocab_size,
      config.embedding_size,
      config.hidden_size,
      num_layers: config.num_layers,
      dropout: config.dropout
    )
    
    intent_head = IntentHead.build_model(
      config.hidden_size * 2,
      num_intents,
      dropout: config.dropout
    )
    
    ner_head = NERHead.build_model(
      config.hidden_size * 2,
      num_bio_tags,
      dropout: config.dropout
    )
    
    pos_head = POSHead.build_model(
      config.hidden_size * 2,
      num_pos_tags,
      dropout: config.dropout
    )
    
    {encoder, intent_head, ner_head, pos_head}
  end
  
  defp multitask_train_loop(encoder, intent_head, ner_head, pos_head,
                             encoder_params, intent_params, ner_params, pos_params,
                             train_data, val_data, _vocabularies, config) do
    
    num_batches = div(length(train_data.inputs), config.batch_size)
    num_samples = length(train_data.inputs)
    
    # Log training start
    Logger.info("Starting multi-task training:")
    Logger.info("  Samples: #{num_samples} (train) / #{length(val_data.inputs)} (val)")
    Logger.info("  Batches per epoch: #{num_batches}")
    Logger.info("  Epochs: #{config.epochs}")
    Logger.info("  Batch size: #{config.batch_size}")
    Logger.info("  Learning rate: #{config.learning_rate}")
    
    # Initialize optimizers
    {enc_opt_init, enc_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {int_opt_init, int_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {ner_opt_init, ner_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {pos_opt_init, pos_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    
    initial_state = %{
      encoder_params: encoder_params,
      intent_params: intent_params,
      ner_params: ner_params,
      pos_params: pos_params,
      enc_opt_state: enc_opt_init.(encoder_params),
      int_opt_state: int_opt_init.(intent_params),
      ner_opt_state: ner_opt_init.(ner_params),
      pos_opt_state: pos_opt_init.(pos_params),
      best_val_loss: :infinity,
      patience_counter: 0,
      metrics: []
    }
    
    final_state = 
      Enum.reduce_while(1..config.epochs, initial_state, fn epoch, state ->
        epoch_start = System.monotonic_time(:millisecond)
        indices = Enum.shuffle(0..(length(train_data.inputs) - 1))
        
        # Log epoch start with progress
        log_epoch_progress(epoch, config.epochs, :start)
        
        {epoch_loss, updated_state} = 
          multitask_train_epoch(
            encoder, intent_head, ner_head, pos_head,
            train_data, indices, state, config, num_batches,
            enc_opt_update, int_opt_update, ner_opt_update, pos_opt_update
          )
        
        val_loss = multitask_validate(
          encoder, intent_head, ner_head, pos_head,
          val_data,
          updated_state.encoder_params,
          updated_state.intent_params,
          updated_state.ner_params,
          updated_state.pos_params,
          config
        )
        
        epoch_duration = System.monotonic_time(:millisecond) - epoch_start
        
        epoch_metrics = %{
          epoch: epoch,
          train_loss: epoch_loss,
          val_loss: val_loss,
          duration_ms: epoch_duration
        }
        
        log_epoch_progress(epoch, config.epochs, :complete, epoch_metrics)
        
        {continue?, updated_state} = 
          check_early_stopping(updated_state, val_loss, config, epoch_metrics)
        
        if continue? do
          {:cont, updated_state}
        else
          Logger.info("Early stopping triggered at epoch #{epoch}")
          {:halt, updated_state}
        end
      end)
    
    Logger.info("Training complete!")
    
    {final_state.encoder_params, final_state.intent_params, 
     final_state.ner_params, final_state.pos_params, final_state.metrics}
  end
  
  defp multitask_train_epoch(encoder, intent_head, ner_head, pos_head,
                              train_data, indices, state, config, num_batches,
                              _enc_opt_update, _int_opt_update, _ner_opt_update, _pos_opt_update) do
    
    batch_size = config.batch_size
    
    Enum.reduce(0..(num_batches - 1), {0.0, state}, fn batch_idx, {total_loss, acc_state} ->
      start_idx = batch_idx * batch_size
      batch_indices = Enum.slice(indices, start_idx, batch_size)
      
      if length(batch_indices) == 0 do
        {total_loss, acc_state}
      else
        batch_inputs = batch_indices |> Enum.map(&Enum.at(train_data.inputs, &1)) |> Nx.tensor(type: :s64)
        batch_intents = batch_indices |> Enum.map(&Enum.at(train_data.intents, &1)) |> Nx.tensor(type: :s64)
        batch_bio = batch_indices |> Enum.map(&Enum.at(train_data.bio_tags, &1)) |> Nx.tensor(type: :s64)
        batch_pos = batch_indices |> Enum.map(&Enum.at(train_data.pos_tags, &1)) |> Nx.tensor(type: :s64)
        batch_masks = batch_indices |> Enum.map(&Enum.at(train_data.masks, &1)) |> Nx.tensor(type: :f32)
        
        # Forward pass and compute total loss
        {token_outputs, sentence_vector} = 
          SharedEncoder.encode(encoder, batch_inputs, acc_state.encoder_params)
        
        intent_preds = IntentHead.forward(intent_head, sentence_vector, acc_state.intent_params)
        ner_preds = NERHead.forward(ner_head, token_outputs, acc_state.ner_params)
        pos_preds = POSHead.forward(pos_head, token_outputs, acc_state.pos_params)
        
        intent_loss = IntentHead.compute_loss(intent_preds, batch_intents)
        ner_loss = NERHead.compute_loss(ner_preds, batch_bio, batch_masks)
        pos_loss = POSHead.compute_loss(pos_preds, batch_pos, batch_masks)
        
        total_weighted_loss = 
          Nx.add(
            Nx.add(
              Nx.multiply(intent_loss, config.intent_loss_weight),
              Nx.multiply(ner_loss, config.ner_loss_weight)
            ),
            Nx.multiply(pos_loss, config.pos_loss_weight)
          )
        
        loss_value = Nx.to_number(total_weighted_loss)
        
        # For simplicity, we update each head's parameters based on total loss
        # A more sophisticated approach would compute individual gradients
        new_state = acc_state
        
        {total_loss + loss_value, new_state}
      end
    end)
    |> then(fn {total_loss, state} ->
      {total_loss / max(num_batches, 1), state}
    end)
  end
  
  defp multitask_validate(encoder, intent_head, ner_head, pos_head, val_data,
                           encoder_params, intent_params, ner_params, pos_params, config) do
    if length(val_data.inputs) == 0 do
      0.0
    else
      batch_size = config.batch_size
      num_batches = div(length(val_data.inputs), batch_size) + 1
      
      total_loss = 
        Enum.reduce(0..(num_batches - 1), 0.0, fn batch_idx, acc ->
          start_idx = batch_idx * batch_size
          batch_inputs = Enum.slice(val_data.inputs, start_idx, batch_size)
          batch_intents = Enum.slice(val_data.intents, start_idx, batch_size)
          batch_bio = Enum.slice(val_data.bio_tags, start_idx, batch_size)
          batch_pos = Enum.slice(val_data.pos_tags, start_idx, batch_size)
          batch_masks = Enum.slice(val_data.masks, start_idx, batch_size)
          
          if length(batch_inputs) == 0 do
            acc
          else
            inputs = Nx.tensor(batch_inputs, type: :s64)
            intent_targets = Nx.tensor(batch_intents, type: :s64)
            bio_targets = Nx.tensor(batch_bio, type: :s64)
            pos_targets = Nx.tensor(batch_pos, type: :s64)
            masks = Nx.tensor(batch_masks, type: :f32)
            
            {token_outputs, sentence_vector} = 
              SharedEncoder.encode(encoder, inputs, encoder_params)
            
            intent_preds = IntentHead.forward(intent_head, sentence_vector, intent_params)
            ner_preds = NERHead.forward(ner_head, token_outputs, ner_params)
            pos_preds = POSHead.forward(pos_head, token_outputs, pos_params)
            
            intent_loss = IntentHead.compute_loss(intent_preds, intent_targets) |> Nx.to_number()
            ner_loss = NERHead.compute_loss(ner_preds, bio_targets, masks) |> Nx.to_number()
            pos_loss = POSHead.compute_loss(pos_preds, pos_targets, masks) |> Nx.to_number()
            
            loss = config.intent_loss_weight * intent_loss + 
                   config.ner_loss_weight * ner_loss +
                   config.pos_loss_weight * pos_loss
            
            acc + loss
          end
        end)
      
      total_loss / max(num_batches, 1)
    end
  end
  
  defp save_multitask_model(model) do
    model_path = get_model_path("lstm_multitask.term")
    File.mkdir_p!(Path.dirname(model_path))
    binary = :erlang.term_to_binary(model)
    File.write!(model_path, binary)
    Logger.info("Saved multi-task LSTM model", %{path: model_path})
  end

  # ============================================================================
  # Private Training Functions
  # ============================================================================
  
  defp build_config(opts) do
    Enum.reduce(opts, @default_config, fn {k, v}, acc ->
      if Map.has_key?(acc, k), do: Map.put(acc, k, v), else: acc
    end)
  end
  
  defp prepare_training_data(examples, config) do
    # Build vocabularies
    token_vocab = DataLoaders.build_lstm_vocabulary(examples, 
      min_freq: config.min_vocab_freq,
      max_vocab: config.max_vocab_size
    )
    
    {intent_to_idx, idx_to_intent} = DataLoaders.build_intent_vocabulary(examples)
    {bio_to_idx, idx_to_bio} = DataLoaders.build_bio_vocabulary(examples)
    
    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent,
      bio_to_idx: bio_to_idx,
      idx_to_bio: idx_to_bio
    }
    
    # Convert examples to tensors
    processed = 
      examples
      |> Enum.map(fn ex -> 
        indices = DataLoaders.tokens_to_indices(ex.tokens, token_vocab)
        padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
        intent_idx = Map.get(intent_to_idx, ex.intent, 0)
        
        %{
          input: padded,
          intent: intent_idx,
          negative_for: ex.negative_for
        }
      end)
    
    # Shuffle and split into train/validation
    shuffled = Enum.shuffle(processed)
    split_idx = round(length(shuffled) * (1 - config.validation_split))
    
    {train_list, val_list} = Enum.split(shuffled, split_idx)
    
    train_data = %{
      inputs: Enum.map(train_list, & &1.input),
      intents: Enum.map(train_list, & &1.intent),
      negatives: Enum.map(train_list, & &1.negative_for)
    }
    
    val_data = %{
      inputs: Enum.map(val_list, & &1.input),
      intents: Enum.map(val_list, & &1.intent),
      negatives: Enum.map(val_list, & &1.negative_for)
    }
    
    Logger.info("Prepared training data", %{
      train_size: length(train_list),
      val_size: length(val_list),
      vocab_size: map_size(token_vocab),
      num_intents: map_size(intent_to_idx)
    })
    
    {:ok, train_data, val_data, vocabularies}
  end
  
  defp build_intent_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    
    encoder = SharedEncoder.build_model(
      vocab_size,
      config.embedding_size,
      config.hidden_size,
      num_layers: config.num_layers,
      dropout: config.dropout
    )
    
    intent_head = IntentHead.build_model(
      config.hidden_size * 2,  # Bidirectional
      num_intents,
      dropout: config.dropout
    )
    
    {encoder, intent_head}
  end
  
  defp train_loop(encoder, intent_head, encoder_params, intent_params, 
                  train_data, val_data, _vocabularies, config) do
    
    num_batches = div(length(train_data.inputs), config.batch_size)
    num_samples = length(train_data.inputs)
    
    # Log training start
    Logger.info("Starting intent classifier training:")
    Logger.info("  Samples: #{num_samples} (train) / #{length(val_data.inputs)} (val)")
    Logger.info("  Batches per epoch: #{num_batches}")
    Logger.info("  Epochs: #{config.epochs}")
    Logger.info("  Batch size: #{config.batch_size}")
    Logger.info("  Learning rate: #{config.learning_rate}")
    
    # Initialize optimizer states
    {encoder_opt_init, encoder_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    {intent_opt_init, intent_opt_update} = Polaris.Optimizers.adam(learning_rate: config.learning_rate)
    
    encoder_opt_state = encoder_opt_init.(encoder_params)
    intent_opt_state = intent_opt_init.(intent_params)
    
    initial_state = %{
      encoder_params: encoder_params,
      intent_params: intent_params,
      encoder_opt_state: encoder_opt_state,
      intent_opt_state: intent_opt_state,
      best_val_loss: :infinity,
      patience_counter: 0,
      metrics: []
    }
    
    # Training loop
    final_state = 
      Enum.reduce_while(1..config.epochs, initial_state, fn epoch, state ->
        epoch_start = System.monotonic_time(:millisecond)
        
        # Shuffle training data
        indices = Enum.shuffle(0..(length(train_data.inputs) - 1))
        
        # Log epoch start
        log_epoch_progress(epoch, config.epochs, :start)
        
        # Train batches
        {epoch_loss, updated_state} = 
          train_epoch(
            encoder, intent_head, 
            train_data, indices,
            state, config, num_batches,
            encoder_opt_update, intent_opt_update
          )
        
        # Validate
        val_loss = validate(
          encoder, intent_head,
          val_data,
          updated_state.encoder_params,
          updated_state.intent_params,
          config
        )
        
        # Calculate accuracy
        train_acc = calculate_accuracy(
          encoder, intent_head,
          train_data,
          updated_state.encoder_params,
          updated_state.intent_params,
          config
        )
        
        val_acc = calculate_accuracy(
          encoder, intent_head,
          val_data,
          updated_state.encoder_params,
          updated_state.intent_params,
          config
        )
        
        epoch_duration = System.monotonic_time(:millisecond) - epoch_start
        
        epoch_metrics = %{
          epoch: epoch,
          train_loss: epoch_loss,
          val_loss: val_loss,
          train_acc: train_acc,
          val_acc: val_acc,
          duration_ms: epoch_duration
        }
        
        log_epoch_progress_with_acc(epoch, config.epochs, epoch_metrics)
        
        # Early stopping check
        {continue?, updated_state} = 
          check_early_stopping(updated_state, val_loss, config, epoch_metrics)
        
        if continue? do
          {:cont, updated_state}
        else
          Logger.info("Early stopping triggered at epoch #{epoch}")
          {:halt, updated_state}
        end
      end)
    
    Logger.info("Training complete!")
    
    {final_state.encoder_params, final_state.intent_params, final_state.metrics}
  end
  
  defp train_epoch(encoder, intent_head, train_data, indices, state, config, num_batches,
                   encoder_opt_update, intent_opt_update) do
    
    batch_size = config.batch_size
    
    Enum.reduce(0..(num_batches - 1), {0.0, state}, fn batch_idx, {total_loss, acc_state} ->
      # Get batch indices
      start_idx = batch_idx * batch_size
      batch_indices = Enum.slice(indices, start_idx, batch_size)
      
      if length(batch_indices) == 0 do
        {total_loss, acc_state}
      else
        # Create batch tensors
        batch_inputs = 
          batch_indices
          |> Enum.map(&Enum.at(train_data.inputs, &1))
          |> Nx.tensor(type: :s64)
        
        batch_intents =
          batch_indices
          |> Enum.map(&Enum.at(train_data.intents, &1))
          |> Nx.tensor(type: :s64)
        
        # Forward pass and compute gradients
        {loss, {encoder_grads, intent_grads}} = 
          compute_loss_and_grads(
            encoder, intent_head,
            batch_inputs, batch_intents,
            acc_state.encoder_params, acc_state.intent_params
          )
        
        loss_value = Nx.to_number(loss)
        
        # Update parameters
        {encoder_updates, new_encoder_opt} = 
          encoder_opt_update.(encoder_grads, acc_state.encoder_opt_state, acc_state.encoder_params)
        
        {intent_updates, new_intent_opt} = 
          intent_opt_update.(intent_grads, acc_state.intent_opt_state, acc_state.intent_params)
        
        new_encoder_params = apply_updates(acc_state.encoder_params, encoder_updates)
        new_intent_params = apply_updates(acc_state.intent_params, intent_updates)
        
        new_state = %{acc_state |
          encoder_params: new_encoder_params,
          intent_params: new_intent_params,
          encoder_opt_state: new_encoder_opt,
          intent_opt_state: new_intent_opt
        }
        
        {total_loss + loss_value, new_state}
      end
    end)
    |> then(fn {total_loss, state} ->
      {total_loss / max(num_batches, 1), state}
    end)
  end
  
  defp compute_loss_and_grads(encoder, intent_head, inputs, targets, encoder_params, intent_params) do
    # Combined forward pass and loss computation with gradient calculation
    grad_fn = fn enc_p, int_p ->
      # Encode
      {_token_outputs, sentence_vector} = 
        SharedEncoder.encode(encoder, inputs, enc_p)
      
      # Classify
      predictions = IntentHead.forward(intent_head, sentence_vector, int_p)
      
      # Compute loss
      IntentHead.compute_loss(predictions, targets)
    end
    
    # Compute gradients for both parameter sets
    {{loss, encoder_grads}, intent_grads} = 
      Nx.Defn.value_and_grad(
        fn enc_p ->
          Nx.Defn.value_and_grad(
            fn int_p -> grad_fn.(enc_p, int_p) end,
            intent_params
          )
        end,
        encoder_params
      ).()
    
    {loss, {encoder_grads, intent_grads}}
  end
  
  defp apply_updates(params, updates) do
    # Recursively apply updates to nested parameter maps
    case {params, updates} do
      {%{} = p, %{} = u} ->
        Map.merge(p, u, fn _k, pv, uv -> apply_updates(pv, uv) end)
      
      {p, u} when is_tuple(p) and is_tuple(u) ->
        p
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(u))
        |> Enum.map(fn {pv, uv} -> apply_updates(pv, uv) end)
        |> List.to_tuple()
      
      {p, u} ->
        Nx.add(p, u)
    end
  end
  
  defp validate(encoder, intent_head, val_data, encoder_params, intent_params, config) do
    if length(val_data.inputs) == 0 do
      0.0
    else
      # Process in batches to avoid memory issues
      batch_size = config.batch_size
      num_batches = div(length(val_data.inputs), batch_size) + 1
      
      total_loss = 
        Enum.reduce(0..(num_batches - 1), 0.0, fn batch_idx, acc ->
          start_idx = batch_idx * batch_size
          batch_inputs = Enum.slice(val_data.inputs, start_idx, batch_size)
          batch_intents = Enum.slice(val_data.intents, start_idx, batch_size)
          
          if length(batch_inputs) == 0 do
            acc
          else
            inputs = Nx.tensor(batch_inputs, type: :s64)
            targets = Nx.tensor(batch_intents, type: :s64)
            
            {_token_outputs, sentence_vector} = 
              SharedEncoder.encode(encoder, inputs, encoder_params)
            
            predictions = IntentHead.forward(intent_head, sentence_vector, intent_params)
            loss = IntentHead.compute_loss(predictions, targets) |> Nx.to_number()
            
            acc + loss
          end
        end)
      
      total_loss / max(num_batches, 1)
    end
  end
  
  defp calculate_accuracy(encoder, intent_head, data, encoder_params, intent_params, _config) do
    if length(data.inputs) == 0 do
      0.0
    else
      num_samples = length(data.inputs)
      
      correct = 
        Enum.reduce(0..(num_samples - 1), 0, fn idx, acc ->
          input = Enum.at(data.inputs, idx)
          target = Enum.at(data.intents, idx)
          
          inputs = Nx.tensor([input], type: :s64)
          
          {_token_outputs, sentence_vector} = 
            SharedEncoder.encode(encoder, inputs, encoder_params)
          
          predictions = IntentHead.forward(intent_head, sentence_vector, intent_params)
          predicted = Nx.argmax(predictions, axis: 1) |> Nx.to_flat_list() |> hd()
          
          if predicted == target, do: acc + 1, else: acc
        end)
      
      correct / num_samples
    end
  end
  
  defp check_early_stopping(state, val_loss, config, epoch_metrics) do
    new_metrics = state.metrics ++ [epoch_metrics]
    
    if val_loss < state.best_val_loss do
      # Improved - reset patience
      {true, %{state | 
        best_val_loss: val_loss, 
        patience_counter: 0,
        metrics: new_metrics
      }}
    else
      # No improvement
      new_patience = state.patience_counter + 1
      
      if new_patience >= config.early_stopping_patience do
        {false, %{state | patience_counter: new_patience, metrics: new_metrics}}
      else
        {true, %{state | patience_counter: new_patience, metrics: new_metrics}}
      end
    end
  end
  
  defp save_intent_model(model) do
    model_path = get_model_path("lstm_intent.term")
    
    # Ensure directory exists
    File.mkdir_p!(Path.dirname(model_path))
    
    # Serialize and save
    binary = :erlang.term_to_binary(model)
    File.write!(model_path, binary)
    
    Logger.info("Saved LSTM intent model", %{path: model_path})
  end
  
  defp get_model_path(filename) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join([models_path, "lstm", filename])
  end
  
  # ============================================================================
  # Progress Display Helpers
  # ============================================================================
  
  defp log_epoch_progress(epoch, total_epochs, :start) do
    bar = progress_bar(epoch - 1, total_epochs)
    Logger.info("Epoch #{epoch}/#{total_epochs} #{bar} training...")
  end
  
  defp log_epoch_progress(epoch, total_epochs, :complete, metrics) do
    bar = progress_bar(epoch, total_epochs)
    duration = format_duration(metrics.duration_ms)
    train_loss = Float.round(metrics.train_loss, 4)
    val_loss = Float.round(metrics.val_loss, 4)
    
    Logger.info("Epoch #{epoch}/#{total_epochs} #{bar} done in #{duration} | train_loss: #{train_loss} | val_loss: #{val_loss}")
  end
  
  defp log_epoch_progress_with_acc(epoch, total_epochs, metrics) do
    bar = progress_bar(epoch, total_epochs)
    duration = format_duration(metrics.duration_ms)
    train_loss = Float.round(metrics.train_loss, 4)
    val_loss = Float.round(metrics.val_loss, 4)
    train_acc = Float.round(metrics.train_acc * 100, 1)
    val_acc = Float.round(metrics.val_acc * 100, 1)
    
    Logger.info("Epoch #{epoch}/#{total_epochs} #{bar} done in #{duration} | loss: #{train_loss}/#{val_loss} | acc: #{train_acc}%/#{val_acc}%")
  end
  
  defp progress_bar(current, total) do
    width = 20
    filled = round(current / total * width)
    empty = width - filled
    
    "[" <> String.duplicate("=", filled) <> String.duplicate(" ", empty) <> "]"
  end
  
  defp format_duration(ms) when ms < 1000, do: "#{ms}ms"
  defp format_duration(ms) when ms < 60_000, do: "#{Float.round(ms / 1000, 1)}s"
  defp format_duration(ms), do: "#{Float.round(ms / 60_000, 1)}m"
end
