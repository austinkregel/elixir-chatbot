defmodule Brain.ML.LSTM.AxonTrainer do
  @moduledoc """
  Simplified LSTM trainer using Axon's Loop API for proper training.
  
  This trainer uses Axon's built-in training loop which:
  - Properly computes gradients and updates parameters
  - Uses JIT compilation via EXLA for speed
  - Handles batching efficiently
  
  ## Usage
  
      {:ok, model} = AxonTrainer.train_intent_classifier(epochs: 5)
  """
  
  require Logger
  
  alias Brain.ML.DataLoaders
  
  @default_config %{
    embedding_size: 64,
    hidden_size: 64,
    dropout: 0.1,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 5,
    max_seq_length: 50,
    min_vocab_freq: 2,
    max_vocab_size: 8000
  }
  
  @doc """
  Train an intent classification model using LSTM with Axon's Loop API.
  
  This is the recommended training method as it properly handles:
  - Gradient computation and backpropagation
  - JIT compilation for speed
  - Batching and shuffling
  - Metrics tracking
  
  ## Options
  - `:embedding_size` - Dimension of word embeddings (default: 64)
  - `:hidden_size` - LSTM hidden dimension (default: 64)
  - `:learning_rate` - Adam learning rate (default: 0.001)
  - `:batch_size` - Training batch size (default: 32)
  - `:epochs` - Number of training epochs (default: 5)
  - `:name` - Experiment name for tracking (optional)
  """
  def train_intent_classifier(opts \\ []) do
    config = Map.merge(@default_config, Map.new(opts))
    experiment_name = Keyword.get(opts, :name)
    
    Logger.info("Starting Axon LSTM intent classifier training")
    Logger.info("Config: #{inspect(config)}")
    
    with {:ok, raw_examples} <- DataLoaders.load_intent_training_data_for_lstm(),
         {:ok, train_data, val_data, vocabularies} <- prepare_data(raw_examples, config) do
      
      # Build the model
      model = build_intent_model(vocabularies, config)
      
      Logger.info("Training on #{length(train_data.inputs)} samples")
      Logger.info("Validation on #{length(val_data.inputs)} samples")
      Logger.info("Vocabulary size: #{map_size(vocabularies.token_vocab)}")
      Logger.info("Intent classes: #{map_size(vocabularies.intent_to_idx)}")
      
      # Create batched training data
      train_batches = create_batches(train_data, config.batch_size)
      val_batches = create_batches(val_data, config.batch_size)
      
      Logger.info("Training batches: #{length(train_batches)}")
      Logger.info("Validation batches: #{length(val_batches)}")
      
      # Track metrics during training
      metrics_state = %{
        best_val_accuracy: 0.0,
        best_val_loss: :infinity,
        final_train_accuracy: 0.0,
        final_train_loss: 0.0,
        final_val_accuracy: 0.0,
        final_val_loss: 0.0,
        epochs_completed: 0
      }
      metrics_ref = :ets.new(:training_metrics, [:set, :public])
      :ets.insert(metrics_ref, {:metrics, metrics_state})
      
      # Build the training loop with metric tracking
      loop = 
        model
        |> Axon.Loop.trainer(:categorical_cross_entropy, Polaris.Optimizers.adam(learning_rate: config.learning_rate))
        |> Axon.Loop.metric(:accuracy)
        |> Axon.Loop.validate(model, val_batches)
        |> Axon.Loop.early_stop("validation_loss", mode: :min, patience: 3)
        |> Axon.Loop.handle_event(:epoch_completed, fn state ->
          # Extract metrics from state - handle both tensor and scalar values
          epoch = state.epoch
          train_acc = extract_metric(state.metrics, "accuracy")
          train_loss = extract_metric(state.metrics, "loss")
          val_acc = extract_metric(state.metrics, "validation_accuracy")
          val_loss = extract_metric(state.metrics, "validation_loss")
          
          # Update tracked metrics
          [{:metrics, current}] = :ets.lookup(metrics_ref, :metrics)
          updated = %{current |
            final_train_accuracy: train_acc,
            final_train_loss: train_loss,
            final_val_accuracy: val_acc,
            final_val_loss: val_loss,
            best_val_accuracy: safe_max(current.best_val_accuracy, val_acc),
            best_val_loss: min_loss(current.best_val_loss, val_loss),
            epochs_completed: epoch + 1
          }
          :ets.insert(metrics_ref, {:metrics, updated})
          
          # Log epoch summary
          Logger.info("Epoch #{epoch + 1}: train_acc=#{format_pct(train_acc)}%, val_acc=#{format_pct(val_acc)}%, val_loss=#{format_num(val_loss)}")
          
          {:continue, state}
        end)
      
      # Run training
      Logger.info("Starting training loop with EXLA backend...")
      start_time = System.monotonic_time(:second)
      
      trained_state = Axon.Loop.run(loop, train_batches, %{}, 
        epochs: config.epochs,
        compiler: EXLA,
        strict?: false
      )
      
      duration = System.monotonic_time(:second) - start_time
      Logger.info("Training completed in #{duration} seconds")
      
      # Get final metrics
      [{:metrics, final_metrics}] = :ets.lookup(metrics_ref, :metrics)
      :ets.delete(metrics_ref)
      
      # Record experiment if name provided
      if experiment_name do
        alias Brain.ML.LSTM.ExperimentTracker
        ExperimentTracker.record(%{
          name: experiment_name,
          config: Map.take(config, [:hidden_size, :embedding_size, :batch_size, :learning_rate, :dropout, :epochs]),
          final_train_accuracy: final_metrics.final_train_accuracy,
          final_train_loss: final_metrics.final_train_loss,
          final_val_accuracy: final_metrics.final_val_accuracy,
          final_val_loss: final_metrics.final_val_loss,
          best_val_accuracy: final_metrics.best_val_accuracy,
          best_val_loss: if(final_metrics.best_val_loss == :infinity, do: nil, else: final_metrics.best_val_loss),
          epochs_completed: final_metrics.epochs_completed,
          training_time_seconds: duration
        })
        Logger.info("Experiment '#{experiment_name}' recorded")
      end
      
      # Build result
      result = %{
        model: model,
        params: trained_state,
        vocabularies: vocabularies,
        config: config,
        training_time: duration,
        metrics: final_metrics
      }
      
      # Save the model
      save_model(result)
      
      {:ok, result}
    end
  end
  
  @doc """
  Classify text using a trained model.
  """
  def classify(text, model_state) do
    tokens = Brain.ML.Tokenizer.tokenize(text)
    indices = DataLoaders.tokens_to_indices(tokens, model_state.vocabularies.token_vocab)
    padded = DataLoaders.pad_sequence(indices, model_state.config.max_seq_length)
    
    input = Nx.tensor([padded], type: :s64)
    
    # Run prediction
    output = Axon.predict(model_state.model, model_state.params, %{"input" => input})
    
    # Get predicted class
    pred_idx = output |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
    confidence = output |> Nx.to_flat_list() |> Enum.at(pred_idx)
    
    intent = Map.get(model_state.vocabularies.idx_to_intent, pred_idx, "unknown")
    
    {intent, confidence}
  end
  
  # ============================================================================
  # Private Functions
  # ============================================================================
  
  defp min_loss(:infinity, val) when is_number(val), do: val
  defp min_loss(current, val) when is_number(val) and is_number(current), do: min(current, val)
  defp min_loss(current, _), do: current
  
  defp safe_max(a, b) when is_number(a) and is_number(b), do: max(a, b)
  defp safe_max(a, _) when is_number(a), do: a
  defp safe_max(_, b) when is_number(b), do: b
  defp safe_max(_, _), do: 0.0
  
  defp extract_metric(metrics, key) do
    case get_in(metrics, [key]) do
      nil -> 0.0
      %Nx.Tensor{} = t -> Nx.to_number(t)
      val when is_number(val) -> val
      _ -> 0.0
    end
  end
  
  defp format_pct(val) when is_number(val), do: Float.round(val * 100, 1)
  defp format_pct(_), do: 0.0
  
  defp format_num(val) when is_number(val), do: Float.round(val, 3)
  defp format_num(_), do: 0.0
  
  defp build_intent_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    
    # Simple but effective architecture
    Axon.input("input", shape: {nil, config.max_seq_length})
    |> Axon.embedding(vocab_size, config.embedding_size)
    |> Axon.lstm(config.hidden_size, name: "lstm")
    |> then(fn {seq, _} -> seq end)  # Get sequence output
    |> Axon.nx(fn x -> 
      # Global average pooling over sequence
      Nx.mean(x, axes: [1])
    end)
    |> Axon.dropout(rate: config.dropout)
    |> Axon.dense(num_intents, activation: :softmax)
  end
  
  defp prepare_data(examples, config) do
    # Build vocabulary
    token_vocab = DataLoaders.build_lstm_vocabulary(examples,
      min_freq: config.min_vocab_freq,
      max_vocab: config.max_vocab_size
    )
    
    # Build intent vocabulary
    intents = examples |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort()
    intent_to_idx = intents |> Enum.with_index() |> Map.new()
    idx_to_intent = intent_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()
    
    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent
    }
    
    # Convert examples to tensors
    processed = Enum.map(examples, fn ex ->
      tokens = ex.tokens || Brain.ML.Tokenizer.tokenize(ex.text)
      indices = DataLoaders.tokens_to_indices(tokens, token_vocab)
      padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
      intent_idx = Map.get(intent_to_idx, ex.intent, 0)
      
      %{input: padded, intent: intent_idx}
    end)
    
    # Split into train/val
    shuffled = Enum.shuffle(processed)
    split_idx = floor(length(shuffled) * 0.9)
    {train_list, val_list} = Enum.split(shuffled, split_idx)
    
    train_data = %{
      inputs: Enum.map(train_list, & &1.input),
      intents: Enum.map(train_list, & &1.intent)
    }
    
    val_data = %{
      inputs: Enum.map(val_list, & &1.input),
      intents: Enum.map(val_list, & &1.intent)
    }
    
    Logger.info("Prepared data: #{length(train_list)} train, #{length(val_list)} val")
    
    {:ok, train_data, val_data, vocabularies}
  end
  
  defp create_batches(data, batch_size) do
    num_intents = data.intents |> Enum.max() |> Kernel.+(1)
    
    data.inputs
    |> Enum.zip(data.intents)
    |> Enum.chunk_every(batch_size)
    # Drop incomplete last batch to avoid dimension mismatch
    |> Enum.filter(fn batch -> length(batch) == batch_size end)
    |> Enum.map(fn batch ->
      {inputs, intents} = Enum.unzip(batch)
      
      input_tensor = Nx.tensor(inputs, type: :s64)
      # One-hot encode intents for cross-entropy loss
      intent_tensor = Nx.tensor(intents, type: :s64) |> Nx.new_axis(1)
      target_tensor = Nx.equal(
        Nx.iota({length(intents), num_intents}, axis: 1),
        intent_tensor
      ) |> Nx.as_type(:f32)
      
      {%{"input" => input_tensor}, target_tensor}
    end)
  end
  
  defp save_model(result) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)
    
    save_path = Path.join(lstm_path, "axon_intent.term")
    
    # Serialize model config and params
    serializable = %{
      params: result.params,
      vocabularies: result.vocabularies,
      config: result.config
    }
    
    binary = :erlang.term_to_binary(serializable)
    File.write!(save_path, binary)
    
    Logger.info("Model saved to #{save_path}")
  end
  
  @doc """
  Load a trained model.
  """
  def load_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    save_path = Path.join([models_path, "lstm", "axon_intent.term"])
    
    case File.read(save_path) do
      {:ok, binary} ->
        data = :erlang.binary_to_term(binary)
        
        # Rebuild model from config
        token_vocab = data.vocabularies.token_vocab
        intent_to_idx = data.vocabularies.intent_to_idx
        
        vocab_size = map_size(token_vocab)
        num_intents = map_size(intent_to_idx)
        
        model = 
          Axon.input("input", shape: {nil, data.config.max_seq_length})
          |> Axon.embedding(vocab_size, data.config.embedding_size)
          |> Axon.lstm(data.config.hidden_size, name: "lstm")
          |> then(fn {seq, _} -> seq end)
          |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
          |> Axon.dropout(rate: data.config.dropout)
          |> Axon.dense(num_intents, activation: :softmax)
        
        {:ok, %{
          model: model,
          params: data.params,
          vocabularies: data.vocabularies,
          config: data.config
        }}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
end
