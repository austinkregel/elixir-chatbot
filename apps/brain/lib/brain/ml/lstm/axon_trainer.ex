defmodule Brain.ML.LSTM.AxonTrainer do
  @moduledoc "Simplified LSTM trainer using Axon's Loop API for proper training.\n\nThis trainer uses Axon's built-in training loop which:\n- Properly computes gradients and updates parameters\n- Uses JIT compilation via EXLA for speed\n- Handles batching efficiently\n\n## Usage\n\n    {:ok, model} = AxonTrainer.train_intent_classifier(epochs: 5)\n"

  alias Brain.ML.Tokenizer
  alias Axon.Loop
  alias Polaris.Optimizers
  require Logger

  alias Brain.ML.DataLoaders

  @default_config %{
    embedding_size: 64,
    hidden_size: 64,
    dropout: 0.2,
    learning_rate: 3.0e-4,
    batch_size: 16,
    epochs: 15,
    max_seq_length: 50,
    min_vocab_freq: 2,
    max_vocab_size: 8000,
    max_intents: nil
  }

  @doc "Train an intent classification model using LSTM with Axon's Loop API.\n\nThis is the recommended training method as it properly handles:\n- Gradient computation and backpropagation\n- JIT compilation for speed\n- Batching and shuffling\n- Metrics tracking\n\n## Options\n- `:embedding_size` - Dimension of word embeddings (default: 64)\n- `:hidden_size` - LSTM hidden dimension (default: 64)\n- `:learning_rate` - Adam learning rate (default: 0.001)\n- `:batch_size` - Training batch size (default: 32)\n- `:epochs` - Number of training epochs (default: 5)\n- `:name` - Experiment name for tracking (optional)\n"
  def train_intent_classifier(opts \\ []) do
    config = Map.merge(@default_config, Map.new(opts))
    experiment_name = Keyword.get(opts, :name)

    Logger.info("Starting Axon LSTM intent classifier training")
    Logger.info("Config: #{inspect(config)}")

    with {:ok, raw_examples} <- DataLoaders.load_intent_training_data_for_lstm(),
         filtered_examples <- maybe_filter_top_intents(raw_examples, config),
         {:ok, train_data, val_data, vocabularies} <- prepare_data(filtered_examples, config) do
      model = build_intent_model(vocabularies, config)

      Logger.info("Training on #{length(train_data.inputs)} samples")
      Logger.info("Validation on #{length(val_data.inputs)} samples")
      Logger.info("Vocabulary size: #{map_size(vocabularies.token_vocab)}")
      Logger.info("Intent classes: #{map_size(vocabularies.intent_to_idx)}")
      num_intents = map_size(vocabularies.intent_to_idx)
      train_batches = create_batches(train_data, config.batch_size, num_intents)
      val_batches = create_batches(val_data, config.batch_size, num_intents)

      Logger.info("Training batches: #{length(train_batches)}")
      Logger.info("Validation batches: #{length(val_batches)}")

      # Weighted cross-entropy loss to handle class imbalance.
      # backend_copy converts from EXLA.Backend to Nx.BinaryBackend so the
      # tensor can be inlined into the EXLA-compiled defn expression.
      weights = Nx.backend_copy(vocabularies.class_weights)

      weighted_loss = fn y_true, logits ->
        log_probs = stable_log_softmax(logits)
        per_class_loss = Nx.negate(Nx.multiply(y_true, log_probs))
        weighted = Nx.multiply(per_class_loss, Nx.reshape(weights, {1, num_intents}))
        Nx.mean(Nx.sum(weighted, axes: [1]))
      end

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

      loop =
        model
        |> Loop.trainer(
          weighted_loss,
          Optimizers.adam(learning_rate: config.learning_rate),
          log: 0
        )
        |> Loop.metric(:accuracy)
        |> Loop.validate(model, val_batches)
        |> Loop.early_stop("validation_loss", mode: :min, patience: 3)
        |> Loop.handle_event(:epoch_completed, fn state ->
          epoch = state.epoch
          train_acc = extract_metric(state.metrics, "accuracy")
          train_loss = extract_metric(state.metrics, "loss")
          val_acc = extract_metric(state.metrics, "validation_accuracy")
          val_loss = extract_metric(state.metrics, "validation_loss")
          [metrics: current] = :ets.lookup(metrics_ref, :metrics)

          updated = %{
            current
            | final_train_accuracy: train_acc,
              final_train_loss: train_loss,
              final_val_accuracy: val_acc,
              final_val_loss: val_loss,
              best_val_accuracy: safe_max(current.best_val_accuracy, val_acc),
              best_val_loss: min_loss(current.best_val_loss, val_loss),
              epochs_completed: epoch + 1
          }

          :ets.insert(metrics_ref, {:metrics, updated})

          Logger.info(
            "Epoch #{epoch + 1}: train_acc=#{format_pct(train_acc)}%, val_acc=#{format_pct(val_acc)}%, val_loss=#{format_num(val_loss)}"
          )

          {:continue, state}
        end)

      Logger.info("Starting training loop with EXLA backend...")
      start_time = System.monotonic_time(:second)

      trained_state =
        Loop.run(loop, train_batches, Axon.ModelState.empty(), epochs: config.epochs, compiler: EXLA)

      duration = System.monotonic_time(:second) - start_time
      Logger.info("Training completed in #{duration} seconds")
      [metrics: final_metrics] = :ets.lookup(metrics_ref, :metrics)
      :ets.delete(metrics_ref)

      if experiment_name do
        alias Brain.ML.LSTM.ExperimentTracker

        ExperimentTracker.record(%{
          name: experiment_name,
          config:
            Map.take(config, [
              :hidden_size,
              :embedding_size,
              :batch_size,
              :learning_rate,
              :dropout,
              :epochs
            ]),
          final_train_accuracy: final_metrics.final_train_accuracy,
          final_train_loss: final_metrics.final_train_loss,
          final_val_accuracy: final_metrics.final_val_accuracy,
          final_val_loss: final_metrics.final_val_loss,
          best_val_accuracy: final_metrics.best_val_accuracy,
          best_val_loss:
            if(final_metrics.best_val_loss == :infinity) do
              nil
            else
              final_metrics.best_val_loss
            end,
          epochs_completed: final_metrics.epochs_completed,
          training_time_seconds: duration
        })

        Logger.info("Experiment '#{experiment_name}' recorded")
      end

      result = %{
        model: model,
        params: trained_state,
        vocabularies: vocabularies,
        config: config,
        training_time: duration,
        metrics: final_metrics
      }

      save_model(result)

      {:ok, result}
    end
  end

  @doc "Classify text using a trained model.\n"
  def classify(text, model_state) do
    tokens = Tokenizer.tokenize(text) |> Enum.map(& &1.text)
    indices = DataLoaders.tokens_to_indices(tokens, model_state.vocabularies.token_vocab)
    padded = DataLoaders.pad_sequence(indices, model_state.config.max_seq_length)

    input = Nx.tensor([padded], type: :s64)
    logits = Axon.predict(model_state.model, model_state.params, %{"input" => input})
    pred_idx = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
    probs = Nx.exp(stable_log_softmax(logits))
    confidence = probs |> Nx.to_flat_list() |> Enum.at(pred_idx)

    intent = Map.get(model_state.vocabularies.idx_to_intent, pred_idx, "unknown")

    {intent, confidence}
  end

  defp ensure_model_state(%Axon.ModelState{} = state), do: state
  defp ensure_model_state(params) when is_map(params), do: Axon.ModelState.new(params)

  defp min_loss(:infinity, val) when is_number(val) do
    val
  end

  defp min_loss(current, val) when is_number(val) and is_number(current) do
    min(current, val)
  end

  defp min_loss(current, _) do
    current
  end

  defp safe_max(a, b) when is_number(a) and is_number(b) do
    max(a, b)
  end

  defp safe_max(a, _) when is_number(a) do
    a
  end

  defp safe_max(_, b) when is_number(b) do
    b
  end

  defp safe_max(_, _) do
    0.0
  end

  defp extract_metric(metrics, key) do
    case get_in(metrics, [key]) do
      nil -> 0.0
      %Nx.Tensor{} = t -> Nx.to_number(t)
      val when is_number(val) -> val
      _ -> 0.0
    end
  end

  defp format_pct(val) when is_number(val) do
    Float.round(val * 100, 1)
  end

  defp format_pct(_) do
    0.0
  end

  defp format_num(val) when is_number(val) do
    Float.round(val, 3)
  end

  defp format_num(_) do
    0.0
  end

  defp build_intent_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)

    Axon.input("input", shape: {nil, config.max_seq_length})
    |> Axon.embedding(vocab_size, config.embedding_size)
    |> Axon.lstm(config.hidden_size, name: "lstm")
    |> then(fn {seq, _} -> seq end)
    |> Axon.nx(fn x ->
      Nx.mean(x, axes: [1])
    end)
    |> Axon.dropout(rate: config.dropout)
    |> Axon.dense(num_intents)
  end

  defp maybe_filter_top_intents(examples, %{max_intents: n}) when is_integer(n) and n > 0 do
    top_intents =
      examples
      |> Enum.frequencies_by(& &1.intent)
      |> Enum.sort_by(fn {_, count} -> count end, :desc)
      |> Enum.take(n)
      |> Enum.map(fn {intent, _} -> intent end)
      |> MapSet.new()

    filtered = Enum.filter(examples, fn ex -> ex.intent in top_intents end)

    Logger.info(
      "Filtered to top #{n} intents (#{length(filtered)} examples from #{length(examples)} total)"
    )

    filtered
  end

  defp maybe_filter_top_intents(examples, _config), do: examples

  defp prepare_data(examples, config) do
    token_vocab =
      DataLoaders.build_lstm_vocabulary(examples,
        min_freq: config.min_vocab_freq,
        max_vocab: config.max_vocab_size
      )

    intents = examples |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort()
    intent_to_idx = intents |> Enum.with_index() |> Map.new()
    idx_to_intent = intent_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()
    num_intents = length(intents)

    # Compute inverse-frequency class weights to handle class imbalance.
    # Weight = total_samples / (num_classes * class_count)
    intent_counts = Enum.frequencies_by(examples, & &1.intent)
    total = length(examples)

    # Use square root of inverse frequency for gentler class balancing.
    # Raw inverse frequency (total / count) can produce 300x ratios between
    # rare and common classes, causing gradient collapse. Square root compresses
    # the range (e.g., 300x → ~17x), then normalizing to mean=1.0 keeps the
    # overall loss magnitude stable.
    raw_weights =
      0..(num_intents - 1)
      |> Enum.map(fn idx ->
        intent_name = Map.get(idx_to_intent, idx, "unknown")
        count = Map.get(intent_counts, intent_name, 1)
        :math.sqrt(total / max(count, 1))
      end)

    mean_weight = Enum.sum(raw_weights) / length(raw_weights)
    class_weights = Enum.map(raw_weights, fn w -> w / mean_weight end)

    class_weights_tensor = Nx.tensor(class_weights, type: :f32)

    vocabularies = %{
      token_vocab: token_vocab,
      intent_to_idx: intent_to_idx,
      idx_to_intent: idx_to_intent,
      class_weights: class_weights_tensor
    }

    processed =
      Enum.map(examples, fn ex ->
        tokens = ex.tokens || Tokenizer.tokenize_words(ex.text)
        indices = DataLoaders.tokens_to_indices(tokens, token_vocab)
        padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
        intent_idx = Map.get(intent_to_idx, ex.intent, 0)

        %{input: padded, intent: intent_idx}
      end)

    # Stratified split: ensure every intent appears in both train and val
    grouped = Enum.group_by(processed, & &1.intent)

    {train_list, val_list} =
      Enum.reduce(grouped, {[], []}, fn {_intent, group_examples}, {train_acc, val_acc} ->
        shuffled_group = Enum.shuffle(group_examples)
        split_at = max(1, floor(length(shuffled_group) * 0.9))
        {train_part, val_part} = Enum.split(shuffled_group, split_at)

        # Ensure at least 1 example in val for each intent
        {train_part, val_part} =
          if val_part == [] and length(train_part) > 1 do
            {Enum.drop(train_part, -1), [List.last(train_part)]}
          else
            {train_part, val_part}
          end

        {train_acc ++ train_part, val_acc ++ val_part}
      end)

    # Shuffle again so batches aren't grouped by intent
    train_list = Enum.shuffle(train_list)
    val_list = Enum.shuffle(val_list)

    train_data = %{
      inputs: Enum.map(train_list, & &1.input),
      intents: Enum.map(train_list, & &1.intent)
    }

    val_data = %{
      inputs: Enum.map(val_list, & &1.input),
      intents: Enum.map(val_list, & &1.intent)
    }

    Logger.info("Prepared data: #{length(train_list)} train, #{length(val_list)} val, #{num_intents} intents")

    {:ok, train_data, val_data, vocabularies}
  end

  defp create_batches(data, batch_size, num_intents) do
    data.inputs
    |> Enum.zip(data.intents)
    |> Enum.chunk_every(batch_size)
    |> Enum.filter(fn batch -> length(batch) == batch_size end)
    |> Enum.map(fn batch ->
      {inputs, intents} = Enum.unzip(batch)

      input_tensor = Nx.tensor(inputs, type: :s64)
      intent_tensor = Nx.tensor(intents, type: :s64) |> Nx.new_axis(1)

      target_tensor =
        Nx.equal(
          Nx.iota({batch_size, num_intents}, axis: 1),
          intent_tensor
        )
        |> Nx.as_type(:f32)

      {%{"input" => input_tensor}, target_tensor}
    end)
  end

  defp stable_log_softmax(logits) do
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logit)
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  defp save_model(result) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)

    save_path = Path.join(lstm_path, "axon_intent.term")

    serializable = %{
      params: result.params,
      vocabularies: result.vocabularies,
      config: result.config
    }

    binary = :erlang.term_to_binary(serializable)
    File.write!(save_path, binary)

    Logger.info("Model saved to #{save_path}")
  end

  @doc "Load a trained model.\n"
  def load_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    save_path = Path.join([models_path, "lstm", "axon_intent.term"])

    case File.read(save_path) do
      {:ok, binary} ->
        data = :erlang.binary_to_term(binary)
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
          |> Axon.dense(num_intents)

        {:ok,
         %{
           model: model,
           params: ensure_model_state(data.params),
           vocabularies: data.vocabularies,
           config: data.config
         }}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
