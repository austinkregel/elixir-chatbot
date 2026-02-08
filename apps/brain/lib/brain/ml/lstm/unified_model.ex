defmodule Brain.ML.LSTM.UnifiedModel do
  @moduledoc """
  Unified LSTM model serving multiple NLP tasks.
  
  This module provides a single LSTM encoder that powers:
  1. **Intent Classification** - What does the user want?
  2. **Named Entity Recognition (NER)** - Extract people, places, things
  3. **Sentiment Analysis** - Positive, negative, neutral
  4. **Speech Act Classification** - Question, command, statement, greeting
  
  ## Architecture
  
  All tasks share a bidirectional LSTM encoder:
  
      Input Text → Tokenize → Embedding → BiLSTM → Task-Specific Heads
                                              ↓
                              ┌───────────────┼───────────────┐
                              ↓               ↓               ↓
                          Intent Head     NER Head      Sentiment Head
                              ↓               ↓               ↓
                          "weather"     [LOC:NYC]       "neutral"
  
  ## Benefits
  
  - **Shared representations** - One encoder learns from all tasks
  - **Faster inference** - Single forward pass for multiple predictions
  - **Better accuracy** - Multi-task learning improves generalization
  
  ## Usage
  
      # Start the model server
      UnifiedModel.start_link()
      
      # Get all predictions at once
      UnifiedModel.analyze("What's the weather in NYC?")
      # => %{
      #   intent: {"weather.query", 0.92},
      #   entities: [%{type: "location", value: "NYC", confidence: 0.88}],
      #   sentiment: {:neutral, 0.95},
      #   speech_act: {:directive, :request_information}
      # }
  """
  
  use GenServer
  require Logger
  
  alias Brain.ML.DataLoaders
  alias Brain.ML.Tokenizer
  
  @default_config %{
    embedding_size: 128,
    hidden_size: 128,
    dropout: 0.1,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 20,
    max_seq_length: 50
  }
  
  # Sentiment labels
  @sentiment_labels ["negative", "neutral", "positive"]
  
  # Speech act categories (Searle's taxonomy)
  @speech_act_labels [
    "assertive",   # statements, claims
    "directive",   # questions, commands, requests
    "commissive",  # promises, offers
    "expressive",  # thanks, greetings, apologies
    "declarative"  # performatives
  ]
  
  defstruct [
    :encoder,
    :intent_head,
    :ner_head,
    :sentiment_head,
    :speech_act_head,
    :params,
    :vocabularies,
    :config,
    :ready
  ]
  
  # ============================================================================
  # Client API
  # ============================================================================
  
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end
  
  @doc """
  Check if the model is loaded and ready.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end
  
  @doc """
  Analyze text and return all predictions.
  
  Returns a map with intent, entities, sentiment, and speech act.
  """
  def analyze(text, name \\ __MODULE__) do
    GenServer.call(name, {:analyze, text}, 10_000)
  end
  
  @doc """
  Get just the intent classification.
  """
  def classify_intent(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_intent, text}, 5_000)
  end
  
  @doc """
  Get just the entities.
  """
  def extract_entities(text, name \\ __MODULE__) do
    GenServer.call(name, {:extract_entities, text}, 5_000)
  end
  
  @doc """
  Get just the sentiment.
  """
  def classify_sentiment(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_sentiment, text}, 5_000)
  end
  
  @doc """
  Get just the speech act.
  """
  def classify_speech_act(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_speech_act, text}, 5_000)
  end

  @doc """
  Reload the model from disk without restarting the GenServer.
  """
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end
  
  # ============================================================================
  # Server Callbacks
  # ============================================================================
  
  @impl true
  def init(_opts) do
    state = %__MODULE__{ready: false}
    
    # Try to load existing model
    send(self(), :load_model)
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end
  
  @impl true
  def handle_call({:analyze, text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end
  
  @impl true
  def handle_call({:analyze, text}, _from, state) do
    try do
      result = do_analyze(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end
  
  @impl true
  def handle_call({:classify_intent, text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end
  
  @impl true
  def handle_call({:classify_intent, text}, _from, state) do
    try do
      result = do_classify_intent(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end
  
  @impl true
  def handle_call({:classify_sentiment, text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end
  
  @impl true
  def handle_call({:classify_sentiment, text}, _from, state) do
    try do
      result = do_classify_sentiment(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end
  
  @impl true
  def handle_call({:classify_speech_act, text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end
  
  @impl true
  def handle_call({:classify_speech_act, text}, _from, state) do
    try do
      result = do_classify_speech_act(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end
  
  @impl true
  def handle_call({:extract_entities, text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end
  
  @impl true
  def handle_call({:extract_entities, text}, _from, state) do
    try do
      result = do_extract_entities(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed, disabling model")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end
  
  @impl true
  def handle_call(:reload, _from, state) do
    case load_saved_model() do
      {:ok, model_data} ->
        Logger.info("UnifiedModel: Reloaded model from disk")
        new_state = %{state |
          encoder: model_data.encoder,
          intent_head: model_data.intent_head,
          ner_head: model_data.ner_head,
          sentiment_head: model_data.sentiment_head,
          speech_act_head: model_data.speech_act_head,
          params: model_data.params,
          vocabularies: model_data.vocabularies,
          config: model_data.config,
          ready: true
        }
        {:reply, :ok, new_state}

      {:error, reason} ->
        Logger.warning("UnifiedModel: Reload failed (#{inspect(reason)})")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_info(:load_model, state) do
    case load_saved_model() do
      {:ok, model_data} ->
        Logger.info("UnifiedModel: Loaded saved model")
        new_state = %{state |
          encoder: model_data.encoder,
          intent_head: model_data.intent_head,
          ner_head: model_data.ner_head,
          sentiment_head: model_data.sentiment_head,
          speech_act_head: model_data.speech_act_head,
          params: model_data.params,
          vocabularies: model_data.vocabularies,
          config: model_data.config,
          ready: true
        }
        {:noreply, new_state}
        
      {:error, reason} ->
        Logger.info("UnifiedModel: No saved model found (#{inspect(reason)}), model not ready")
        {:noreply, state}
    end
  end
  
  # ============================================================================
  # Training
  # ============================================================================
  
  @doc """
  Train the unified model on all tasks.
  
  This trains a shared encoder with task-specific heads for:
  - Intent classification
  - NER (BIO tagging)
  - Sentiment analysis
  - Speech act classification
  """
  def train(opts \\ []) do
    config = Map.merge(@default_config, Map.new(opts))
    experiment_name = Keyword.get(opts, :name)
    
    Logger.info("Training UnifiedModel with config: #{inspect(config)}")
    
    with {:ok, training_data} <- prepare_all_training_data(config) do
      # Build the unified model
      model = build_unified_model(training_data.vocabularies, config)
      
      # Train each task
      trained_params = train_all_tasks(model, training_data, config)
      
      # Save the model
      save_unified_model(model, trained_params, training_data.vocabularies, config)
      
      # Record experiment if name provided
      if experiment_name do
        alias Brain.ML.LSTM.ExperimentTracker
        ExperimentTracker.record(%{
          name: experiment_name,
          config: config,
          notes: "Unified multi-task model"
        })
      end
      
      {:ok, %{model: model, params: trained_params, vocabularies: training_data.vocabularies}}
    end
  end
  
  # ============================================================================
  # Private: Model Architecture
  # ============================================================================
  
  defp build_unified_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    num_bio_tags = map_size(vocabularies.bio_to_idx)
    num_sentiments = length(@sentiment_labels)
    num_speech_acts = length(@speech_act_labels)
    
    # Shared encoder
    encoder = build_encoder(vocab_size, config)
    
    # Task-specific heads
    intent_head = build_classification_head(config.hidden_size * 2, num_intents, "intent")
    ner_head = build_sequence_head(config.hidden_size * 2, num_bio_tags, "ner")
    sentiment_head = build_classification_head(config.hidden_size * 2, num_sentiments, "sentiment")
    speech_act_head = build_classification_head(config.hidden_size * 2, num_speech_acts, "speech_act")
    
    %{
      encoder: encoder,
      intent_head: intent_head,
      ner_head: ner_head,
      sentiment_head: sentiment_head,
      speech_act_head: speech_act_head
    }
  end
  
  defp build_encoder(vocab_size, config) do
    Axon.input("input", shape: {nil, config.max_seq_length})
    |> Axon.embedding(vocab_size, config.embedding_size)
    |> Axon.lstm(config.hidden_size, name: "encoder_lstm")
    |> then(fn {seq, _state} -> seq end)
  end
  
  defp build_classification_head(input_size, num_classes, name) do
    # Takes pooled sequence representation
    Axon.input("#{name}_input", shape: {nil, input_size})
    |> Axon.dense(64, activation: :relu, name: "#{name}_dense")
    |> Axon.dropout(rate: 0.1)
    |> Axon.dense(num_classes, activation: :softmax, name: "#{name}_output")
  end
  
  defp build_sequence_head(input_size, num_classes, name) do
    # Takes full sequence for per-token classification
    Axon.input("#{name}_input", shape: {nil, nil, input_size})
    |> Axon.dense(num_classes, activation: :softmax, name: "#{name}_output")
  end
  
  # ============================================================================
  # Private: Inference
  # ============================================================================
  
  defp do_analyze(text, state) do
    tokens = Tokenizer.tokenize(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    
    # Run encoder once
    encoder_output = Axon.predict(state.encoder, state.params.encoder, %{"input" => input})
    
    # Pool for classification heads (mean pooling)
    pooled = Nx.mean(encoder_output, axes: [1])
    
    %{
      intent: run_intent_head(pooled, state),
      entities: run_ner_head(encoder_output, tokens, state),
      sentiment: run_sentiment_head(pooled, state),
      speech_act: run_speech_act_head(pooled, state)
    }
  end
  
  # Note: These functions intentionally do NOT have try/rescue.
  # Errors propagate to handle_call where the model is disabled on failure.
  
  defp do_classify_intent(text, state) do
    tokens = Tokenizer.tokenize(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = Axon.predict(state.encoder, state.params.encoder, %{"input" => input})
    pooled = Nx.mean(encoder_output, axes: [1])
    run_intent_head(pooled, state)
  end
  
  defp do_classify_sentiment(text, state) do
    tokens = Tokenizer.tokenize(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = Axon.predict(state.encoder, state.params.encoder, %{"input" => input})
    pooled = Nx.mean(encoder_output, axes: [1])
    run_sentiment_head(pooled, state)
  end
  
  defp do_classify_speech_act(text, state) do
    tokens = Tokenizer.tokenize(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = Axon.predict(state.encoder, state.params.encoder, %{"input" => input})
    pooled = Nx.mean(encoder_output, axes: [1])
    run_speech_act_head(pooled, state)
  end
  
  defp do_extract_entities(text, state) do
    tokens = Tokenizer.tokenize(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = Axon.predict(state.encoder, state.params.encoder, %{"input" => input})
    run_ner_head(encoder_output, tokens, state)
  end
  
  defp run_intent_head(pooled, state) do
    output = Axon.predict(state.intent_head, state.params.intent, %{"intent_input" => pooled})
    
    pred_idx = output |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
    confidence = output |> Nx.to_flat_list() |> Enum.at(pred_idx)
    intent = Map.get(state.vocabularies.idx_to_intent, pred_idx, "unknown")
    
    {intent, confidence}
  end
  
  defp run_sentiment_head(pooled, state) do
    case state.params[:sentiment] do
      nil -> {:neutral, 0.5}  # Fallback if not trained
      params ->
        output = Axon.predict(state.sentiment_head, params, %{"sentiment_input" => pooled})
        
        pred_idx = output |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
        confidence = output |> Nx.to_flat_list() |> Enum.at(pred_idx)
        label = Enum.at(@sentiment_labels, pred_idx, "neutral")
        
        {String.to_atom(label), confidence}
    end
  end
  
  defp run_speech_act_head(pooled, state) do
    case state.params[:speech_act] do
      nil -> {:assertive, 0.5}  # Fallback if not trained
      params ->
        output = Axon.predict(state.speech_act_head, params, %{"speech_act_input" => pooled})
        
        pred_idx = output |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
        confidence = output |> Nx.to_flat_list() |> Enum.at(pred_idx)
        label = Enum.at(@speech_act_labels, pred_idx, "assertive")
        
        {String.to_atom(label), confidence}
    end
  end
  
  defp run_ner_head(encoder_output, tokens, state) do
    case state.params[:ner] do
      nil -> []  # Fallback if not trained
      params ->
        output = Axon.predict(state.ner_head, params, %{"ner_input" => encoder_output})
        
        # Get predictions for each token
        predictions = output 
          |> Nx.argmax(axis: 2) 
          |> Nx.to_flat_list()
          |> Enum.take(length(tokens))
        
        # Convert BIO tags to entities
        predictions
        |> Enum.zip(tokens)
        |> extract_entities_from_bio(state.vocabularies.idx_to_bio)
    end
  end
  
  defp extract_entities_from_bio(token_predictions, idx_to_bio) do
    token_predictions
    |> Enum.reduce({[], nil}, fn {idx, token}, {entities, current} ->
      tag = Map.get(idx_to_bio, idx, "O")
      
      case {tag, current} do
        {"O", nil} -> 
          {entities, nil}
          
        {"O", entity} -> 
          {[entity | entities], nil}
          
        {"B-" <> type, nil} -> 
          {entities, %{type: type, value: token, tokens: [token]}}
          
        {"B-" <> type, entity} -> 
          {[entity | entities], %{type: type, value: token, tokens: [token]}}
          
        {"I-" <> _type, nil} -> 
          {entities, nil}
          
        {"I-" <> type, %{type: type} = entity} -> 
          updated = %{entity | 
            value: entity.value <> " " <> token,
            tokens: entity.tokens ++ [token]
          }
          {entities, updated}
          
        {"I-" <> _new_type, entity} -> 
          {[entity | entities], nil}
          
        _ ->
          {entities, current}
      end
    end)
    |> then(fn {entities, current} ->
      case current do
        nil -> entities
        entity -> [entity | entities]
      end
    end)
    |> Enum.reverse()
  end
  
  defp prepare_input(tokens, vocab, config) do
    indices = DataLoaders.tokens_to_indices(tokens, vocab)
    padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
    Nx.tensor([padded], type: :s64)
  end
  
  # ============================================================================
  # Private: Training Data Preparation
  # ============================================================================
  
  defp prepare_all_training_data(config) do
    with {:ok, intent_examples} <- DataLoaders.load_intent_training_data_for_lstm() do
      # Build vocabularies
      token_vocab = DataLoaders.build_lstm_vocabulary(intent_examples, max_vocab: 8000)
      
      intents = intent_examples |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort()
      intent_to_idx = intents |> Enum.with_index() |> Map.new()
      idx_to_intent = intent_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()
      
      # BIO tags vocabulary
      bio_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", 
                  "B-MISC", "I-MISC", "B-DATE", "I-DATE", "B-TIME", "I-TIME"]
      bio_to_idx = bio_tags |> Enum.with_index() |> Map.new()
      idx_to_bio = bio_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()
      
      vocabularies = %{
        token_vocab: token_vocab,
        intent_to_idx: intent_to_idx,
        idx_to_intent: idx_to_intent,
        bio_to_idx: bio_to_idx,
        idx_to_bio: idx_to_bio
      }
      
      # Prepare intent training data
      intent_data = prepare_intent_data(intent_examples, vocabularies, config)
      
      {:ok, %{
        intent: intent_data,
        vocabularies: vocabularies
      }}
    end
  end
  
  defp prepare_intent_data(examples, vocabularies, config) do
    examples
    |> Enum.map(fn ex ->
      tokens = ex.tokens || Tokenizer.tokenize(ex.text)
      indices = DataLoaders.tokens_to_indices(tokens, vocabularies.token_vocab)
      padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
      intent_idx = Map.get(vocabularies.intent_to_idx, ex.intent, 0)
      
      %{input: padded, intent: intent_idx}
    end)
  end
  
  # ============================================================================
  # Private: Training Loop
  # ============================================================================
  
  defp train_all_tasks(model, training_data, config) do
    Logger.info("Training unified model...")
    
    # For now, just train intent classification
    # Other tasks can be added incrementally
    encoder_params = train_encoder_and_intent(
      model.encoder, 
      model.intent_head, 
      training_data.intent, 
      training_data.vocabularies,
      config
    )
    
    %{
      encoder: encoder_params.encoder,
      intent: encoder_params.intent
    }
  end
  
  defp train_encoder_and_intent(encoder, intent_head, data, vocabularies, config) do
    num_intents = map_size(vocabularies.intent_to_idx)
    
    # Split data
    shuffled = Enum.shuffle(data)
    split_idx = floor(length(shuffled) * 0.9)
    {train_list, _val_list} = Enum.split(shuffled, split_idx)
    
    # Create batches
    batches = train_list
      |> Enum.chunk_every(config.batch_size)
      |> Enum.filter(fn batch -> length(batch) == config.batch_size end)
      |> Enum.map(fn batch ->
        inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
        intents = batch |> Enum.map(& &1.intent) |> Nx.tensor(type: :s64) |> Nx.new_axis(1)
        targets = Nx.equal(
          Nx.iota({config.batch_size, num_intents}, axis: 1),
          intents
        ) |> Nx.as_type(:f32)
        
        {inputs, targets}
      end)
    
    # Build combined model for training
    combined_model = 
      encoder
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)
      |> Axon.dense(64, activation: :relu, name: "intent_dense")
      |> Axon.dropout(rate: 0.1)
      |> Axon.dense(num_intents, activation: :softmax, name: "intent_output")
    
    # Train with Axon Loop
    loop = 
      combined_model
      |> Axon.Loop.trainer(:categorical_cross_entropy, Polaris.Optimizers.adam(learning_rate: config.learning_rate))
      |> Axon.Loop.metric(:accuracy)
    
    train_data = batches |> Enum.map(fn {inputs, targets} ->
      {%{"input" => inputs}, targets}
    end)
    
    Logger.info("Training on #{length(train_data)} batches for #{config.epochs} epochs")
    
    trained_state = Axon.Loop.run(loop, train_data, %{},
      epochs: config.epochs,
      compiler: EXLA,
      strict?: false
    )
    
    # Extract params for encoder and intent head separately
    %{
      encoder: trained_state,
      intent: trained_state
    }
  end
  
  # ============================================================================
  # Private: Model Persistence
  # ============================================================================
  
  defp save_unified_model(model, params, vocabularies, config) do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)
    
    save_path = Path.join(lstm_path, "unified_model.term")
    
    data = %{
      params: params,
      vocabularies: vocabularies,
      config: config
    }
    
    binary = :erlang.term_to_binary(data)
    File.write!(save_path, binary)
    
    Logger.info("Unified model saved to #{save_path}")
  end
  
  defp load_saved_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    save_path = Path.join([models_path, "lstm", "unified_model.term"])
    
    case File.read(save_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          
          # Validate required keys exist
          unless Map.has_key?(data, :vocabularies) and Map.has_key?(data, :config) and Map.has_key?(data, :params) do
            raise "Invalid model format: missing required keys"
          end
          
          # Rebuild model architecture
          model = build_unified_model(data.vocabularies, data.config)
          
          {:ok, %{
            encoder: model.encoder,
            intent_head: model.intent_head,
            ner_head: model.ner_head,
            sentiment_head: model.sentiment_head,
            speech_act_head: model.speech_act_head,
            params: data.params,
            vocabularies: data.vocabularies,
            config: data.config
          }}
        rescue
          e ->
            Logger.error("Failed to load unified model: #{inspect(e)}")
            Logger.warning("The saved model may be corrupted or incompatible. Delete #{save_path} and retrain.")
            {:error, :corrupted_model}
        end
        
      {:error, :enoent} ->
        {:error, :not_found}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
end
