defmodule Brain.ML.LSTM.UnifiedModel do
  @moduledoc "Unified LSTM model serving multiple NLP tasks.\n\nThis module provides a single LSTM encoder that powers:\n1. **Intent Classification** - What does the user want?\n2. **Named Entity Recognition (NER)** - Extract people, places, things\n3. **Sentiment Analysis** - Positive, negative, neutral\n4. **Speech Act Classification** - Question, command, statement, greeting\n\n## Architecture\n\nAll tasks share a bidirectional LSTM encoder:\n\n    Input Text → Tokenize → Embedding → BiLSTM → Task-Specific Heads\n                                            ↓\n                            ┌───────────────┼───────────────┐\n                            ↓               ↓               ↓\n                        Intent Head     NER Head      Sentiment Head\n                            ↓               ↓               ↓\n                        \"weather\"     [LOC:NYC]       \"neutral\"\n\n## Benefits\n\n- **Shared representations** - One encoder learns from all tasks\n- **Faster inference** - Single forward pass for multiple predictions\n- **Better accuracy** - Multi-task learning improves generalization\n\n## Usage\n\n    # Start the model server\n    UnifiedModel.start_link()\n\n    # Get all predictions at once\n    UnifiedModel.analyze(\"What's the weather in NYC?\")\n    # => %{\n    #   intent: {\"weather.query\", 0.92},\n    #   entities: [%{type: \"location\", value: \"NYC\", confidence: 0.88}],\n    #   sentiment: {:neutral, 0.95},\n    #   speech_act: {:directive, :request_information}\n    # }\n"

  alias Axon.Loop
  alias Polaris.Optimizers
  use GenServer
  require Logger

  alias Brain.ML.DataLoaders
  alias Brain.ML.Tokenizer

  @default_config %{
    embedding_size: 128,
    hidden_size: 128,
    dropout: 0.2,
    learning_rate: 3.0e-4,
    batch_size: 8,
    epochs: 50,
    head_epochs: 30,
    max_seq_length: 50,
    min_examples_per_intent: 10,
    sentiment_lr_scale: 0.5,
    sentiment_epochs: 50,
    speech_act_epochs: 100,
    speech_act_batch_size: 8
  }
  @sentiment_labels ["negative", "neutral", "positive"]
  @speech_act_labels ["assertive", "directive", "commissive", "expressive", "declarative"]

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

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Check if the model is loaded and ready.\n"
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Analyze text and return all predictions.\n\nReturns a map with intent, entities, sentiment, and speech act.\n"
  def analyze(text, name \\ __MODULE__) do
    GenServer.call(name, {:analyze, text}, 10_000)
  end

  @doc "Get just the intent classification.\n"
  def classify_intent(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_intent, text}, 30_000)
  end

  @doc "Batch intent classification. Processes texts in GPU-friendly chunks\nto avoid spawning one EXLA computation per text.\n"
  def batch_classify_intents(texts, name \\ __MODULE__) when is_list(texts) do
    GenServer.call(name, {:batch_classify_intents, texts}, 120_000)
  end

  @doc "Get just the entities.\n"
  def extract_entities(text, name \\ __MODULE__) do
    GenServer.call(name, {:extract_entities, text}, 30_000)
  end

  @doc "Get just the sentiment.\n"
  def classify_sentiment(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_sentiment, text}, 30_000)
  end

  @doc "Get just the speech act.\n"
  def classify_speech_act(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify_speech_act, text}, 30_000)
  end

  @doc "Reload the model from disk without restarting the GenServer.\n"
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  @impl true
  def init(_opts) do
    state = %__MODULE__{ready: false}
    send(self(), :load_model)

    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  @impl true
  def handle_call({:analyze, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:analyze, text}, _from, state) do
    try do
      result = do_analyze(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (analyze), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:classify_intent, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:classify_intent, text}, _from, state) do
    try do
      result = do_classify_intent(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (classify_intent), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:batch_classify_intents, _texts}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:batch_classify_intents, texts}, _from, state) do
    try do
      results = do_batch_classify_intents(texts, state)
      {:reply, {:ok, results}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (batch_classify_intents), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:classify_sentiment, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:classify_sentiment, text}, _from, state) do
    try do
      result = do_classify_sentiment(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (classify_sentiment), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:classify_speech_act, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:classify_speech_act, text}, _from, state) do
    try do
      result = do_classify_speech_act(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (classify_speech_act), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:extract_entities, _text}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  @impl true
  def handle_call({:extract_entities, text}, _from, state) do
    try do
      result = do_extract_entities(text, state)
      {:reply, {:ok, result}, state}
    rescue
      e in ArgumentError ->
        Logger.warning("UnifiedModel: EXLA decode failed (extract_entities), disabling model: #{Exception.message(e)}")
        {:reply, {:error, :model_incompatible}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call(:reload, _from, state) do
    case load_saved_model() do
      {:ok, model_data} ->
        Logger.info("UnifiedModel: Reloaded model from disk")

        new_state = %{
          state
          | encoder: model_data.encoder,
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

        new_state = %{
          state
          | encoder: model_data.encoder,
            intent_head: model_data.intent_head,
            ner_head: model_data.ner_head,
            sentiment_head: model_data.sentiment_head,
            speech_act_head: model_data.speech_act_head,
            params: model_data.params,
            vocabularies: model_data.vocabularies,
            config: model_data.config,
            ready: true
        }

        warmup_jit(new_state)
        {:noreply, new_state}

      {:error, reason} ->
        Logger.warning("UnifiedModel: LSTM model not available (#{inspect(reason)}). System running in degraded mode — using TF-IDF only for intent classification.")
        {:noreply, state}
    end
  end

  @doc "Train the unified model on all tasks.\n\nThis trains a shared encoder with task-specific heads for:\n- Intent classification\n- NER (BIO tagging)\n- Sentiment analysis\n- Speech act classification\n"
  def train(opts \\ []) do
    app_min = Application.get_env(:brain, :min_examples_per_intent)
    app_overrides = if app_min, do: %{min_examples_per_intent: app_min}, else: %{}

    config =
      @default_config
      |> Map.merge(app_overrides)
      |> Map.merge(Map.new(opts))

    experiment_name = Keyword.get(opts, :name)

    Logger.info("Training UnifiedModel with config: #{inspect(config)}")

    with {:ok, training_data} <- prepare_all_training_data(config) do
      model = build_unified_model(training_data.vocabularies, config)
      {trained_params, intent_metrics} = train_all_tasks(model, training_data, config)
      save_unified_model(model, trained_params, training_data.vocabularies, config)

      if experiment_name do
        alias Brain.ML.LSTM.ExperimentTracker

        ExperimentTracker.record(%{
          name: experiment_name,
          config: config,
          intent_metrics: intent_metrics,
          notes: "Unified multi-task model"
        })
      end

      {:ok, %{model: model, params: trained_params, vocabularies: training_data.vocabularies, intent_metrics: intent_metrics}}
    end
  end

  defp build_unified_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    num_intents = map_size(vocabularies.intent_to_idx)
    num_bio_tags = map_size(vocabularies.bio_to_idx)
    num_sentiments = length(@sentiment_labels)
    num_speech_acts = length(@speech_act_labels)
    encoder = build_encoder(vocab_size, config)
    intent_head = build_classification_head(config.hidden_size, num_intents, "intent")
    ner_head = build_sequence_head(config.hidden_size, num_bio_tags, "ner")

    sentiment_head =
      build_classification_head(config.hidden_size, num_sentiments, "sentiment")

    speech_act_head =
      build_classification_head(config.hidden_size, num_speech_acts, "speech_act")

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
    hidden = min(256, max(64, num_classes * 2))

    Axon.input("#{name}_input", shape: {nil, input_size})
    |> Axon.dense(hidden, activation: :relu, name: "#{name}_dense")
    |> Axon.dropout(rate: 0.3)
    |> Axon.dense(num_classes, name: "#{name}_output")
  end

  defp build_sequence_head(input_size, num_classes, name) do
    Axon.input("#{name}_input", shape: {nil, nil, input_size})
    |> Axon.dense(num_classes, activation: :softmax, name: "#{name}_output")
  end

  defp run_encoder(input, state) do
    Axon.predict(state.encoder, state.params.encoder, %{"input" => input}, compiler: EXLA)
  end

  defp do_analyze(text, state) do
    token_maps = Tokenizer.tokenize(text)
    tokens = Enum.map(token_maps, & &1.text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = run_encoder(input, state)
    pooled = masked_mean_pool(encoder_output, input)

    %{
      intent: run_intent_head(pooled, state),
      entities: run_ner_head(encoder_output, tokens, state),
      sentiment: run_sentiment_head(pooled, state),
      speech_act: run_speech_act_head(pooled, state)
    }
  end

  defp do_classify_intent(text, state) do
    tokens = tokenize_for_model(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = run_encoder(input, state)
    pooled = masked_mean_pool(encoder_output, input)
    run_intent_head(pooled, state)
  end

  @batch_chunk_size 256

  defp do_batch_classify_intents(texts, state) do
    texts
    |> Enum.chunk_every(@batch_chunk_size)
    |> Enum.flat_map(fn chunk ->
      results = classify_intent_chunk(chunk, state)
      :erlang.garbage_collect()
      results
    end)
  end

  defp classify_intent_chunk(texts, state) do
    all_padded =
      Enum.map(texts, fn text ->
        tokens = tokenize_for_model(text)
        indices = DataLoaders.tokens_to_indices(tokens, state.vocabularies.token_vocab)
        DataLoaders.pad_sequence(indices, state.config.max_seq_length)
      end)

    batch_input = Nx.tensor(all_padded, type: :s64)

    encoder_output =
      Axon.predict(state.encoder, state.params.encoder, %{"input" => batch_input}, compiler: EXLA)

    pooled = masked_mean_pool(encoder_output, batch_input)

    logits =
      Axon.predict(state.intent_head, state.params.intent, %{"intent_input" => pooled}, compiler: EXLA)

    probs_batch = Nx.exp(stable_log_softmax(logits))
    pred_indices = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list()
    all_probs = probs_batch |> Nx.to_list()

    Enum.zip(pred_indices, all_probs)
    |> Enum.map(fn {pred_idx, probs} ->
      confidence = Enum.at(probs, pred_idx)
      intent = Map.get(state.vocabularies.idx_to_intent, pred_idx, "unknown")

      scores =
        probs
        |> Enum.with_index()
        |> Enum.sort_by(fn {prob, _} -> prob end, :desc)
        |> Enum.take(5)
        |> Enum.map(fn {prob, idx} ->
          {Map.get(state.vocabularies.idx_to_intent, idx, "unknown"), prob}
        end)

      %{label: intent, confidence: confidence, scores: scores}
    end)
  end

  defp do_classify_sentiment(text, state) do
    tokens = tokenize_for_model(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = run_encoder(input, state)
    pooled = masked_mean_pool(encoder_output, input)
    {label, confidence} = run_sentiment_head(pooled, state)
    %{label: label, confidence: confidence}
  end

  defp do_classify_speech_act(text, state) do
    tokens = tokenize_for_model(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = run_encoder(input, state)
    pooled = masked_mean_pool(encoder_output, input)
    {label, confidence} = run_speech_act_head(pooled, state)
    %{label: label, confidence: confidence}
  end

  defp do_extract_entities(text, state) do
    tokens = tokenize_for_model(text)
    input = prepare_input(tokens, state.vocabularies.token_vocab, state.config)
    encoder_output = run_encoder(input, state)
    run_ner_head(encoder_output, tokens, state)
  end

  defp run_intent_head(pooled, state) do
    logits = Axon.predict(state.intent_head, state.params.intent, %{"intent_input" => pooled}, compiler: EXLA)

    pred_idx = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
    probs = Nx.exp(stable_log_softmax(logits))
    all_probs = Nx.to_flat_list(probs)
    confidence = Enum.at(all_probs, pred_idx)
    intent = Map.get(state.vocabularies.idx_to_intent, pred_idx, "unknown")

    scores =
      all_probs
      |> Enum.with_index()
      |> Enum.sort_by(fn {prob, _} -> prob end, :desc)
      |> Enum.take(5)
      |> Enum.map(fn {prob, idx} ->
        {Map.get(state.vocabularies.idx_to_intent, idx, "unknown"), prob}
      end)

    %{label: intent, confidence: confidence, scores: scores}
  end

  defp run_sentiment_head(pooled, state) do
    case state.params[:sentiment] do
      nil ->
        {:neutral, 0.5}

      params ->
        logits = Axon.predict(state.sentiment_head, params, %{"sentiment_input" => pooled}, compiler: EXLA)

        pred_idx = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
        probs = Nx.exp(stable_log_softmax(logits))
        confidence = probs |> Nx.to_flat_list() |> Enum.at(pred_idx)
        label = Enum.at(@sentiment_labels, pred_idx, "neutral")

        {String.to_atom(label), confidence}
    end
  end

  defp run_speech_act_head(pooled, state) do
    case state.params[:speech_act] do
      nil ->
        {:assertive, 0.5}

      params ->
        logits = Axon.predict(state.speech_act_head, params, %{"speech_act_input" => pooled}, compiler: EXLA)

        pred_idx = logits |> Nx.argmax(axis: 1) |> Nx.to_flat_list() |> hd()
        probs = Nx.exp(stable_log_softmax(logits))
        confidence = probs |> Nx.to_flat_list() |> Enum.at(pred_idx)
        label = Enum.at(@speech_act_labels, pred_idx, "assertive")

        {String.to_atom(label), confidence}
    end
  end

  defp run_ner_head(encoder_output, tokens, state) do
    case state.params[:ner] do
      nil ->
        []

      params ->
        output = Axon.predict(state.ner_head, params, %{"ner_input" => encoder_output}, compiler: EXLA)

        predictions =
          output
          |> Nx.argmax(axis: 2)
          |> Nx.to_flat_list()
          |> Enum.take(length(tokens))

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
          updated = %{
            entity
            | value: entity.value <> " " <> token,
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

  defp masked_mean_pool(encoder_output, input) do
    mask = Nx.not_equal(input, 0) |> Nx.as_type(:f32) |> Nx.new_axis(-1)
    masked = Nx.multiply(encoder_output, mask)
    sum = Nx.sum(masked, axes: [1])
    count = Nx.sum(mask, axes: [1]) |> Nx.max(1)
    Nx.divide(sum, count)
  end

  defp safe_tensor_to_number(%Nx.Tensor{} = t), do: Nx.to_number(t)
  defp safe_tensor_to_number(n) when is_number(n), do: n
  defp safe_tensor_to_number(_), do: nil

  defp stable_log_softmax(logits) do
    max_logit = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_logit)
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  defp warmup_jit(state) do
    Logger.info("UnifiedModel: Warming up JIT compilation...")
    t0 = System.monotonic_time(:millisecond)

    try do
      dummy_input = Nx.tensor([List.duplicate(0, state.config.max_seq_length)], type: :s64)
      encoder_output = run_encoder(dummy_input, state)
      pooled = masked_mean_pool(encoder_output, dummy_input)

      run_intent_head(pooled, state)

      if state.params[:sentiment] do
        run_sentiment_head(pooled, state)
      end

      if state.params[:speech_act] do
        run_speech_act_head(pooled, state)
      end

      elapsed = System.monotonic_time(:millisecond) - t0
      Logger.info("UnifiedModel: JIT warmup complete (#{elapsed}ms)")
    rescue
      e ->
        Logger.warning("UnifiedModel: JIT warmup failed: #{Exception.message(e)}")
    end
  end

  defp tokenize_for_model(text) do
    Tokenizer.tokenize(text) |> Enum.map(& &1.text)
  end

  defp prepare_input(tokens, vocab, config) do
    indices = DataLoaders.tokens_to_indices(tokens, vocab)
    padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
    Nx.tensor([padded], type: :s64)
  end

  defp prepare_all_training_data(config) do
    with {:ok, intent_examples} <- DataLoaders.load_intent_training_data_for_lstm() do
      min_examples = Map.get(config, :min_examples_per_intent, 10)

      intent_examples =
        if min_examples > 0 do
          intent_counts = Enum.frequencies_by(intent_examples, & &1.intent)

          valid_intents =
            intent_counts
            |> Enum.filter(fn {_intent, count} -> count >= min_examples end)
            |> Enum.map(fn {intent, _count} -> intent end)
            |> MapSet.new()

          filtered = Enum.filter(intent_examples, &MapSet.member?(valid_intents, &1.intent))
          removed = map_size(intent_counts) - MapSet.size(valid_intents)

          Logger.info(
            "Filtered intents with <#{min_examples} examples: removed #{removed} classes, " <>
              "#{length(filtered)}/#{length(intent_examples)} examples retained"
          )

          filtered
        else
          intent_examples
        end

      sentiment_gold = Brain.ML.EvaluationStore.load_gold_standard("sentiment")
        |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["sentiment"]) end)

      speech_act_gold = Brain.ML.EvaluationStore.load_gold_standard("speech_act")
        |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["speech_act"]) end)

      extra_vocab_examples =
        (sentiment_gold ++ speech_act_gold)
        |> Enum.map(fn ex ->
          %{tokens: Tokenizer.tokenize_words(ex["text"])}
        end)

      token_vocab =
        DataLoaders.build_lstm_vocabulary(
          intent_examples ++ extra_vocab_examples,
          max_vocab: 8000
        )

      intents = intent_examples |> Enum.map(& &1.intent) |> Enum.uniq() |> Enum.sort()
      intent_to_idx = intents |> Enum.with_index() |> Map.new()
      idx_to_intent = intent_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()

      sentiment_to_idx = @sentiment_labels |> Enum.with_index() |> Map.new()

      bio_tags = [
        "O",
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-MISC",
        "I-MISC",
        "B-DATE",
        "I-DATE",
        "B-TIME",
        "I-TIME"
      ]

      bio_to_idx = bio_tags |> Enum.with_index() |> Map.new()
      idx_to_bio = bio_to_idx |> Enum.map(fn {k, v} -> {v, k} end) |> Map.new()

      vocabularies = %{
        token_vocab: token_vocab,
        intent_to_idx: intent_to_idx,
        idx_to_intent: idx_to_intent,
        bio_to_idx: bio_to_idx,
        idx_to_bio: idx_to_bio,
        sentiment_to_idx: sentiment_to_idx
      }

      intent_data = prepare_intent_data(intent_examples, vocabularies, config)
      sentiment_data = prepare_sentiment_data(sentiment_gold, vocabularies, config)
      {speech_act_data, speech_act_to_idx} = prepare_speech_act_data(speech_act_gold, vocabularies, config)

      vocabularies = Map.put(vocabularies, :speech_act_to_idx, speech_act_to_idx)

      {:ok,
       %{
         intent: intent_data,
         sentiment: sentiment_data,
         speech_act: speech_act_data,
         vocabularies: vocabularies
       }}
    end
  end

  defp prepare_intent_data(examples, vocabularies, config) do
    examples
    |> Enum.map(fn ex ->
      tokens = ex.tokens || Tokenizer.tokenize_words(ex.text)
      indices = DataLoaders.tokens_to_indices(tokens, vocabularies.token_vocab)
      padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
      intent_idx = Map.get(vocabularies.intent_to_idx, ex.intent, 0)

      %{input: padded, intent: intent_idx}
    end)
  end

  defp prepare_sentiment_data(gold, vocabularies, config) do
    if gold == [] do
      Logger.warning("No sentiment gold standard data found. Sentiment head will not be trained.")
      []
    else
      gold
      |> Enum.map(fn ex ->
        tokens = Tokenizer.tokenize_words(ex["text"])
        indices = DataLoaders.tokens_to_indices(tokens, vocabularies.token_vocab)
        padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
        sentiment_idx = Map.get(vocabularies.sentiment_to_idx, ex["sentiment"], 1)

        %{input: padded, sentiment: sentiment_idx}
      end)
    end
  end

  defp prepare_speech_act_data(gold, vocabularies, config) do
    speech_act_to_idx = @speech_act_labels |> Enum.with_index() |> Map.new()

    if gold == [] do
      Logger.warning("No speech act gold standard data found. Speech act head will not be trained.")
      {[], speech_act_to_idx}
    else
      data =
        gold
        |> Enum.map(fn ex ->
          tokens = Tokenizer.tokenize_words(ex["text"])
          indices = DataLoaders.tokens_to_indices(tokens, vocabularies.token_vocab)
          padded = DataLoaders.pad_sequence(indices, config.max_seq_length)
          speech_act_idx = Map.get(speech_act_to_idx, ex["speech_act"], 0)

          %{input: padded, speech_act: speech_act_idx}
        end)

      {data, speech_act_to_idx}
    end
  end

  defp train_all_tasks(model, training_data, config) do
    Logger.info("Training unified model...")

    {encoder_params, intent_metrics} =
      train_encoder_and_intent(
        model.encoder,
        model.intent_head,
        training_data.intent,
        training_data.vocabularies,
        config
      )

    frozen_encoder = encoder_params.encoder

    sentiment_params =
      if training_data.sentiment != [] do
        Logger.info("Training sentiment head on #{length(training_data.sentiment)} examples (frozen encoder)...")

        train_head_with_frozen_encoder(
          model.encoder,
          training_data.sentiment,
          :sentiment,
          length(@sentiment_labels),
          frozen_encoder,
          config
        )
      else
        Logger.warning("No sentiment training data. Sentiment head will not be trained.")
        nil
      end

    speech_act_params =
      if training_data.speech_act != [] do
        Logger.info("Training speech act head on #{length(training_data.speech_act)} examples (frozen encoder)...")

        train_head_with_frozen_encoder(
          model.encoder,
          training_data.speech_act,
          :speech_act,
          length(@speech_act_labels),
          frozen_encoder,
          config
        )
      else
        Logger.warning("No speech act training data. Speech act head will not be trained.")
        nil
      end

    params = %{
      encoder: frozen_encoder,
      intent: encoder_params.intent
    }

    params = if sentiment_params, do: Map.put(params, :sentiment, sentiment_params), else: params
    params = if speech_act_params, do: Map.put(params, :speech_act, speech_act_params), else: params
    {params, intent_metrics}
  end

  # Multi-task training pattern: We build an end-to-end combined model for training
  # that matches the structure of intent_head. During inference, intent_head is used
  # separately with the trained params (see run_intent_head/2).
  #
  # The intent_head has its own input layer for inference flexibility, but training
  # requires an end-to-end model. Layer names (intent_dense, intent_output) match
  # between combined_model and intent_head so trained params transfer correctly.
  defp train_encoder_and_intent(encoder, intent_head, data, vocabularies, config) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      num_intents = map_size(vocabularies.intent_to_idx)
      label_smoothing = Map.get(config, :label_smoothing, 0.1)

      # Stratified train/val split: ensure every intent appears in both sets
      grouped = Enum.group_by(data, & &1.intent)

      {train_list, val_list} =
        Enum.reduce(grouped, {[], []}, fn {_intent, group_examples}, {train_acc, val_acc} ->
          shuffled_group = Enum.shuffle(group_examples)
          split_at = max(1, floor(length(shuffled_group) * 0.9))
          {train_part, val_part} = Enum.split(shuffled_group, split_at)

          {train_part, val_part} =
            if val_part == [] and length(train_part) > 1 do
              {Enum.drop(train_part, -1), [List.last(train_part)]}
            else
              {train_part, val_part}
            end

          {train_acc ++ train_part, val_acc ++ val_part}
        end)

      train_list = Enum.shuffle(train_list)
      val_list = Enum.shuffle(val_list)

      class_weights =
        train_list
        |> Enum.frequencies_by(& &1.intent)
        |> then(fn freqs ->
          total = length(train_list)

          raw_weights =
            0..(num_intents - 1)
            |> Enum.map(fn idx ->
              count = Map.get(freqs, idx, 1)
              :math.sqrt(total / max(count, 1))
            end)

          mean_w = Enum.sum(raw_weights) / length(raw_weights)
          Enum.map(raw_weights, fn w -> w / mean_w end) |> Nx.tensor(type: :f32)
        end)

      make_batches = fn examples ->
        examples
        |> Enum.chunk_every(config.batch_size)
        |> Enum.filter(fn batch -> length(batch) == config.batch_size end)
        |> Enum.map(fn batch ->
          batch = pad_batch(batch, config.batch_size)
          inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
          mask = Nx.not_equal(inputs, 0) |> Nx.as_type(:f32) |> Nx.new_axis(-1)
          intents = batch |> Enum.map(& &1.intent) |> Nx.tensor(type: :s64) |> Nx.new_axis(1)

          one_hot =
            Nx.equal(
              Nx.iota({config.batch_size, num_intents}, axis: 1),
              intents
            )
            |> Nx.as_type(:f32)

          targets =
            Nx.multiply(one_hot, 1.0 - label_smoothing)
            |> Nx.add(label_smoothing / num_intents)

          {%{"input" => inputs, "mask" => mask}, targets}
        end)
      end

      train_data = make_batches.(train_list)
      val_data = make_batches.(val_list)

      Logger.debug("Training with intent_head layers: #{inspect(Axon.get_output_shape(intent_head, %{"intent_input" => {1, config.hidden_size}}))}")

      head_hidden = min(256, max(64, num_intents * 2))

      mask_input = Axon.input("mask", shape: {nil, config.max_seq_length, 1})

      masked_pool = Axon.layer(
        fn encoder_out, mask, _opts ->
          masked = Nx.multiply(encoder_out, mask)
          sum = Nx.sum(masked, axes: [1])
          count = Nx.sum(mask, axes: [1]) |> Nx.max(1)
          Nx.divide(sum, count)
        end,
        [encoder, mask_input],
        name: "masked_mean_pool"
      )

      combined_model =
        masked_pool
        |> Axon.dense(head_hidden, activation: :relu, name: "intent_dense")
        |> Axon.dropout(rate: config.dropout)
        |> Axon.dense(num_intents, name: "intent_output")

      weighted_loss = fn y_true, logits ->
        log_probs = stable_log_softmax(logits)
        per_class_loss = Nx.negate(Nx.multiply(y_true, log_probs))
        weighted = Nx.multiply(per_class_loss, Nx.reshape(class_weights, {1, num_intents}))
        Nx.mean(Nx.sum(weighted, axes: [1]))
      end

      metrics_ref = :ets.new(:encoder_train_metrics, [:set, :public])
      :ets.insert(metrics_ref, {:metrics, %{}})

      loop =
        combined_model
        |> Loop.trainer(
          weighted_loss,
          Optimizers.adam(learning_rate: config.learning_rate),
          log: 0
        )
        |> Loop.metric(:accuracy)
        |> exla_validate(combined_model, val_data)
        |> Loop.early_stop("validation_loss", mode: :min, patience: 5)
        |> Loop.handle_event(:epoch_completed, fn state ->
          epoch_metrics = %{
            epoch: state.epoch,
            accuracy: safe_tensor_to_number(state.metrics["accuracy"]),
            loss: safe_tensor_to_number(state.metrics["loss"]),
            validation_accuracy: safe_tensor_to_number(state.metrics["validation_accuracy"]),
            validation_loss: safe_tensor_to_number(state.metrics["validation_loss"])
          }

          :ets.insert(metrics_ref, {:metrics, epoch_metrics})
          {:continue, state}
        end)

      Logger.info("Training on #{length(train_data)} train / #{length(val_data)} val batches for up to #{config.epochs} epochs")

      trained_state =
        Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: config.epochs, compiler: EXLA)

      [{:metrics, final_metrics}] = :ets.lookup(metrics_ref, :metrics)
      :ets.delete(metrics_ref)

      {%{encoder: trained_state, intent: trained_state}, final_metrics}
    end)
  end

  # Frozen-encoder head training: pre-computes encoder outputs once, then trains
  # only the classification head on those features. The encoder is never modified.
  # Returns only the head's trained params (not combined encoder+head state).
  #
  # Per-head config overrides:
  #   {head_name}_lr_scale, {head_name}_epochs, {head_name}_batch_size
  defp train_head_with_frozen_encoder(
         encoder_model,
         data,
         head_name,
         num_classes,
         frozen_encoder_params,
         config
       ) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      name = to_string(head_name)
      label_smoothing = Map.get(config, :label_smoothing, 0.1)

      lr_scale_key = :"#{head_name}_lr_scale"
      lr_scale = Map.get(config, lr_scale_key, 1 / 3)
      finetune_lr = config.learning_rate * lr_scale

      batch_size_key = :"#{head_name}_batch_size"
      head_batch_size = Map.get(config, batch_size_key, config.batch_size)

      grouped = Enum.group_by(data, &Map.get(&1, head_name))

      {train_list, val_list} =
        Enum.reduce(grouped, {[], []}, fn {_class, group_examples}, {train_acc, val_acc} ->
          shuffled_group = Enum.shuffle(group_examples)
          split_at = max(1, floor(length(shuffled_group) * 0.9))
          {train_part, val_part} = Enum.split(shuffled_group, split_at)

          {train_part, val_part} =
            if val_part == [] and length(train_part) > 1 do
              {Enum.drop(train_part, -1), [List.last(train_part)]}
            else
              {train_part, val_part}
            end

          {train_acc ++ train_part, val_acc ++ val_part}
        end)

      train_list = Enum.shuffle(train_list)
      val_list = Enum.shuffle(val_list)

      class_weights =
        train_list
        |> Enum.frequencies_by(&Map.get(&1, head_name))
        |> then(fn freqs ->
          total = length(train_list)

          raw_weights =
            0..(num_classes - 1)
            |> Enum.map(fn idx ->
              count = Map.get(freqs, idx, 1)
              :math.sqrt(total / max(count, 1))
            end)

          mean_w = Enum.sum(raw_weights) / length(raw_weights)
          Enum.map(raw_weights, fn w -> w / mean_w end) |> Nx.tensor(type: :f32)
        end)

      pre_compute_batches = fn examples ->
        examples
        |> Enum.chunk_every(head_batch_size)
        |> Enum.filter(fn batch -> length(batch) == head_batch_size end)
        |> Enum.map(fn batch ->
          batch = pad_batch(batch, head_batch_size)
          inputs = batch |> Enum.map(& &1.input) |> Nx.tensor(type: :s64)
          mask = Nx.not_equal(inputs, 0) |> Nx.as_type(:f32) |> Nx.new_axis(-1)

          labels =
            batch
            |> Enum.map(&Map.get(&1, head_name))
            |> Nx.tensor(type: :s64)
            |> Nx.new_axis(1)

          one_hot =
            Nx.equal(
              Nx.iota({head_batch_size, num_classes}, axis: 1),
              labels
            )
            |> Nx.as_type(:f32)

          targets =
            Nx.multiply(one_hot, 1.0 - label_smoothing)
            |> Nx.add(label_smoothing / num_classes)

          encoder_output =
            Axon.predict(encoder_model, frozen_encoder_params, %{"input" => inputs}, compiler: EXLA)

          masked = Nx.multiply(encoder_output, mask)
          sum = Nx.sum(masked, axes: [1])
          count = Nx.sum(mask, axes: [1]) |> Nx.max(1)
          pooled = Nx.divide(sum, count)

          pooled = Nx.backend_transfer(pooled, Nx.BinaryBackend)
          targets = Nx.backend_transfer(targets, Nx.BinaryBackend)

          {%{"#{name}_input" => pooled}, targets}
        end)
      end

      Logger.info("Pre-computing encoder outputs for #{name} head...")
      train_data = pre_compute_batches.(train_list)
      val_data = pre_compute_batches.(val_list)

      head_hidden = min(256, max(64, num_classes * 2))

      head_model =
        Axon.input("#{name}_input", shape: {nil, config.hidden_size})
        |> Axon.dense(head_hidden, activation: :relu, name: "#{name}_dense")
        |> Axon.dropout(rate: config.dropout)
        |> Axon.dense(num_classes, name: "#{name}_output")

      weighted_loss = fn y_true, logits ->
        log_probs = stable_log_softmax(logits)
        per_class_loss = Nx.negate(Nx.multiply(y_true, log_probs))
        weighted = Nx.multiply(per_class_loss, Nx.reshape(class_weights, {1, num_classes}))
        Nx.mean(Nx.sum(weighted, axes: [1]))
      end

      loop =
        head_model
        |> Loop.trainer(
          weighted_loss,
          Optimizers.adam(learning_rate: finetune_lr),
          log: 0
        )
        |> Loop.metric(:accuracy)
        |> exla_validate(head_model, val_data)
        |> Loop.early_stop("validation_loss", mode: :min, patience: 3)

      epochs_key = :"#{head_name}_epochs"

      head_epochs =
        Map.get(config, epochs_key) ||
          Map.get(config, :head_epochs, config.epochs)

      Logger.info(
        "Training #{name} head (frozen encoder) on #{length(train_data)} train / #{length(val_data)} val batches " <>
          "for up to #{head_epochs} epochs (lr=#{finetune_lr}, batch_size=#{head_batch_size})"
      )

      Loop.run(loop, train_data, Axon.ModelState.empty(), epochs: head_epochs, compiler: EXLA)
    end)
  end

  # Like Loop.validate/4 but passes compiler: EXLA to the internal evaluator
  # run. The built-in Loop.validate calls Loop.run without a compiler, so the
  # evaluator falls back to Nx.Defn.Evaluator. When model params live on EXLA
  # (from training), the evaluator's BinaryBackend accumulation crashes on the
  # EXLA output tensors.
  defp exla_validate(%Loop{metrics: metric_fns} = loop, model, validation_data) do
    evaluator = Loop.evaluator(model)

    validation_handler = fn %Axon.Loop.State{metrics: metrics, step_state: step_state} = state ->
      %{model_state: model_state} = step_state

      val_metrics =
        Enum.reduce(metric_fns, evaluator, fn {k, {_, v}}, acc -> Loop.metric(acc, v, k) end)
        |> Loop.run(validation_data, model_state, compiler: EXLA)
        |> Access.get(0)
        |> Map.new(fn {k, v} -> {"validation_#{k}", v} end)
        |> Map.merge(metrics, fn _, _, v -> v end)

      {:continue, %{state | metrics: val_metrics}}
    end

    Loop.handle_event(loop, :epoch_completed, validation_handler)
  end

  # Pad undersized last batch by repeating the last sample so EXLA
  # compiled shapes remain consistent across all batches
  defp pad_batch(batch, target_size) when length(batch) >= target_size, do: batch

  defp pad_batch(batch, target_size) do
    last = List.last(batch)
    padding = List.duplicate(last, target_size - length(batch))
    batch ++ padding
  end

  # Model architecture is reconstructed from config during load since Axon models
  # are built dynamically from hyperparameters (vocab_size, hidden_size, etc.)
  defp save_unified_model(_model, params, vocabularies, config) do
    models_path =
      Map.get(config, :models_path) ||
        Application.get_env(:brain, :ml)[:models_path] ||
        Brain.priv_path("ml_models")

    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)

    save_path = Path.join(lstm_path, "unified_model.term")

    portable_params = transfer_params_to_binary_backend(params)

    data = %{
      params: portable_params,
      vocabularies: vocabularies,
      config: config
    }

    binary = :erlang.term_to_binary(data)
    File.write!(save_path, binary)

    Logger.info("Unified model saved to #{save_path}")
  end

  defp ensure_model_state(%Axon.ModelState{} = state), do: state
  defp ensure_model_state(params) when is_map(params), do: Axon.ModelState.new(params)

  defp transfer_params_to_binary_backend(%Axon.ModelState{} = state) do
    Axon.ModelState.new(transfer_params_to_binary_backend(state.data))
  end

  defp transfer_params_to_binary_backend(%Nx.Tensor{} = tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defp transfer_params_to_binary_backend(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {k, transfer_params_to_binary_backend(v)} end)
  end

  defp transfer_params_to_binary_backend(other), do: other

  defp load_saved_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    save_path = Path.join([models_path, "lstm", "unified_model.term"])
    Brain.ML.ModelStore.ensure_local("lstm/unified_model.term", save_path)

    case File.read(save_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)

          unless Map.has_key?(data, :vocabularies) and Map.has_key?(data, :config) and
                   Map.has_key?(data, :params) do
            raise "Invalid model format: missing required keys"
          end

          model = build_unified_model(data.vocabularies, data.config)

          params =
            Map.new(data.params, fn
              {k, nil} -> {k, nil}
              {k, v} -> {k, ensure_model_state(v)}
            end)

          {:ok,
           %{
             encoder: model.encoder,
             intent_head: model.intent_head,
             ner_head: model.ner_head,
             sentiment_head: model.sentiment_head,
             speech_act_head: model.speech_act_head,
             params: params,
             vocabularies: data.vocabularies,
             config: data.config
           }}
        rescue
          e ->
            Logger.error("Failed to load unified model: #{inspect(e)}")

            Logger.warning(
              "The saved model may be corrupted or incompatible. Delete #{save_path} and retrain."
            )

            {:error, :corrupted_model}
        end

      {:error, :enoent} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end
end
