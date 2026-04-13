defmodule Brain.Response.LSTMResponse do
  @moduledoc "LSTM-enhanced response generation for more coherent, contextual responses.\n\nThis module augments the existing template-based response system with:\n\n1. **Response Scoring** - Evaluates candidate responses for:\n   - Relevance to the query\n   - Coherence and fluency\n   - Appropriateness to intent\n\n2. **Response Selection** - Picks the best from multiple candidates\n\n3. **Response Refinement** - Improves responses by:\n   - Selecting better word choices\n   - Ensuring proper transitions\n   - Matching the query's tone\n\n## Architecture\n\n    User Query → Generate Candidates → LSTM Scorer → Best Response\n                      ↓\n                 [Template 1]\n                 [Template 2]    →  Score each  →  Pick highest\n                 [Blended]\n                 [Memory-based]\n\n## Training Data\n\nThe model learns from query-response pairs in the intent files,\nlearning what makes a good response for each type of query.\n"

  alias Axon.Loop
  alias Polaris.Optimizers
  alias Brain.Response.MemoryAugmented
  alias Brain.Response
  use GenServer
  require Logger

  alias Brain.ML.Tokenizer
  alias Brain.ML.DataLoaders
  alias Response.{Generator, TemplateStore, TemplateBlender}

  @default_config %{
    embedding_size: 128,
    hidden_size: 128,
    dropout: 0.1,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 15,
    max_query_length: 50,
    max_response_length: 100,
    num_candidates: 5,
    max_vocab: 8000
  }

  defstruct [
    :scorer_model,
    :scorer_params,
    :generator_model,
    :generator_params,
    :vocabularies,
    :config,
    :ready
  ]

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Check if the LSTM response system is ready.\n"
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Generate the best response for a query using LSTM scoring.\n\nGenerates multiple candidate responses, scores each, and returns the best.\n"
  def generate(query, intent, entities, opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.call(name, {:generate, query, intent, entities, opts}, 10_000)
  catch
    :exit, _ -> fallback_generate(query, intent, entities)
  end

  @doc "Score a candidate response for a given query.\n\nReturns a score between 0 and 1 indicating response quality.\n"
  def score_response(query, response, name \\ __MODULE__) do
    GenServer.call(name, {:score, query, response}, 5000)
  catch
    :exit, _ -> {:ok, 0.5}
  end

  @doc "Score multiple candidate responses and return them ranked.\n"
  def rank_responses(query, responses, name \\ __MODULE__) do
    GenServer.call(name, {:rank, query, responses}, 10_000)
  catch
    :exit, _ -> {:ok, Enum.with_index(responses, fn r, i -> {r, 1.0 - i * 0.1} end)}
  end

  @doc "Reload the model from disk without restarting the GenServer.\n"
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  @doc "Select the best response from candidates using LSTM scoring.\n"
  def select_best(query, candidates, name \\ __MODULE__) when is_list(candidates) do
    case rank_responses(query, candidates, name) do
      {:ok, ranked} ->
        {best, _score} = List.first(ranked)
        {:ok, best}

      error ->
        error
    end
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
  def handle_call({:generate, query, intent, entities, _opts}, _from, %{ready: false} = state) do
    result = fallback_generate(query, intent, entities)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:generate, query, intent, entities, opts}, _from, state) do
    num_candidates = Keyword.get(opts, :num_candidates, state.config.num_candidates)

    try do
      candidates = generate_candidates(query, intent, entities, num_candidates)
      scored = score_candidates(query, candidates, state)

      case scored do
        [{best, score} | _] ->
          Logger.debug("LSTM selected response with score #{Float.round(score, 3)}")
          {:reply, {:ok, best, score}, state}

        [] ->
          {:reply, fallback_generate(query, intent, entities), state}
      end
    rescue
      e in ArgumentError ->
        Logger.warning(
          "LSTMResponse model inference failed (generate), disabling: #{Exception.message(e)}"
        )

        {:reply, fallback_generate(query, intent, entities), %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:score, _query, _response}, _from, %{ready: false} = state) do
    {:reply, {:ok, 0.5}, state}
  end

  @impl true
  def handle_call({:score, query, response}, _from, state) do
    try do
      score = compute_response_score(query, response, state)
      {:reply, {:ok, score}, state}
    rescue
      e in ArgumentError ->
        Logger.warning(
          "LSTMResponse model inference failed (score), disabling: #{Exception.message(e)}"
        )

        {:reply, {:ok, 0.5}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call({:rank, _query, responses}, _from, %{ready: false} = state) do
    ranked = Enum.map(responses, &{&1, 0.5})
    {:reply, {:ok, ranked}, state}
  end

  @impl true
  def handle_call({:rank, query, responses}, _from, state) do
    try do
      scored =
        responses
        |> Enum.map(fn response ->
          score = compute_response_score(query, response, state)
          {response, score}
        end)
        |> Enum.sort_by(fn {_, score} -> score end, :desc)

      {:reply, {:ok, scored}, state}
    rescue
      e in ArgumentError ->
        Logger.warning(
          "LSTMResponse model inference failed (rank), disabling: #{Exception.message(e)}"
        )

        ranked = Enum.map(responses, &{&1, 0.5})
        {:reply, {:ok, ranked}, %{state | ready: false}}
    end
  end

  @impl true
  def handle_call(:reload, _from, state) do
    case load_saved_model() do
      {:ok, model_data} ->
        Logger.info("LSTMResponse: Reloaded model from disk")

        new_state = %{
          state
          | scorer_model: model_data.scorer_model,
            scorer_params: model_data.scorer_params,
            vocabularies: model_data.vocabularies,
            config: model_data.config,
            ready: true
        }

        {:reply, :ok, new_state}

      {:error, reason} ->
        Logger.warning("LSTMResponse: Reload failed (#{inspect(reason)})")
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_info(:load_model, state) do
    case load_saved_model() do
      {:ok, model_data} ->
        Logger.info("LSTMResponse: Loaded response scorer model")

        new_state = %{
          state
          | scorer_model: model_data.scorer_model,
            scorer_params: model_data.scorer_params,
            vocabularies: model_data.vocabularies,
            config: model_data.config,
            ready: true
        }

        {:noreply, new_state}

      {:error, reason} ->
        Logger.debug("LSTMResponse: No saved model (#{inspect(reason)})")
        {:noreply, state}
    end
  end

  @doc "Train the response scoring model.\n\nThe model learns to score query-response pairs based on:\n- Positive examples: actual responses from training data\n- Negative examples: mismatched query-response pairs\n"
  def train(opts \\ []) do
    config = Map.merge(@default_config, Map.new(opts))

    Logger.info("Training LSTM Response Scorer")
    Logger.info("Config: #{inspect(config)}")

    with {:ok, training_data} <- prepare_training_data(config) do
      model = build_scorer_model(training_data.vocabularies, config)
      trained_params = train_scorer(model, training_data, config)
      save_model(model, trained_params, training_data.vocabularies, config)

      {:ok, %{model: model, params: trained_params}}
    end
  end

  defp build_scorer_model(vocabularies, config) do
    vocab_size = map_size(vocabularies.token_vocab)
    query_input = Axon.input("query", shape: {nil, config.max_query_length})

    query_encoded =
      query_input
      |> Axon.embedding(vocab_size, config.embedding_size, name: "query_embedding")
      |> Axon.lstm(config.hidden_size, name: "query_lstm")
      |> then(fn {seq, _} -> seq end)
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)

    response_input = Axon.input("response", shape: {nil, config.max_response_length})

    response_encoded =
      response_input
      |> Axon.embedding(vocab_size, config.embedding_size, name: "response_embedding")
      |> Axon.lstm(config.hidden_size, name: "response_lstm")
      |> then(fn {seq, _} -> seq end)
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)

    Axon.concatenate([query_encoded, response_encoded], axis: 1)
    |> Axon.dense(64, activation: :relu, name: "scorer_hidden")
    |> Axon.dropout(rate: config.dropout)
    |> Axon.dense(1, activation: :sigmoid, name: "scorer_output")
  end

  defp generate_candidates(query, intent, entities, num_candidates) do
    template_response =
      case TemplateStore.get_random_template(intent) do
        {:ok, response} when is_binary(response) -> [response]
        {:ok, %{text: text}} -> [text]
        _ -> []
      end

    blended_response =
      if TemplateBlender.ready?() do
        context = %{
          intent: intent,
          entities: entities,
          speech_acts: infer_speech_acts(query, entities)
        }

        case TemplateBlender.blend(query, context) do
          {:ok, response} -> [response]
          _ -> []
        end
      else
        []
      end

    template_variations =
      case TemplateStore.get_templates(intent) do
        templates when is_list(templates) ->
          templates
          |> Enum.map(fn t ->
            cond do
              is_binary(t) -> t
              is_map(t) -> Map.get(t, :text) || Map.get(t, "text") || ""
              true -> ""
            end
          end)
          |> Enum.filter(&(String.length(&1) > 0))
          |> Enum.take(3)

        _ ->
          []
      end

    memory_response =
      case MemoryAugmented.generate(intent, entities) do
        {:ok, response, _meta} -> [response]
        _ -> []
      end

    all_candidates =
      (template_response ++ blended_response ++ template_variations ++ memory_response)
      |> Enum.uniq()
      |> Enum.filter(&is_binary/1)
      |> Enum.filter(&(String.length(&1) > 0))

    Enum.take(all_candidates, num_candidates)
  end

  defp infer_speech_acts(query, entities) do
    result = Brain.Analysis.SpeechActClassifier.classify(query, entities: entities)
    acts = [result.category]

    if result.sub_type != :unknown and result.sub_type != nil do
      acts ++ [result.sub_type]
    else
      acts
    end
  end

  defp score_candidates(query, candidates, state) do
    candidates
    |> Enum.map(fn candidate ->
      score = compute_response_score(query, candidate, state)
      {candidate, score}
    end)
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
  end

  defp compute_response_score(query, response, state) do
    query_tokens = Tokenizer.tokenize_normalized(query)
    response_tokens = Tokenizer.tokenize_normalized(response)

    query_indices = DataLoaders.tokens_to_indices(query_tokens, state.vocabularies.token_vocab)

    response_indices =
      DataLoaders.tokens_to_indices(response_tokens, state.vocabularies.token_vocab)

    query_padded = DataLoaders.pad_sequence(query_indices, state.config.max_query_length)
    response_padded = DataLoaders.pad_sequence(response_indices, state.config.max_response_length)

    query_tensor = Nx.tensor([query_padded], type: :s64)
    response_tensor = Nx.tensor([response_padded], type: :s64)

    output =
      Axon.predict(state.scorer_model, state.scorer_params, %{
        "query" => query_tensor,
        "response" => response_tensor
      }, compiler: EXLA)

    output |> Nx.squeeze() |> Nx.to_number()
  end

  defp prepare_training_data(config) do
    case DataLoaders.load_intent_training_data_for_lstm() do
      {:ok, examples} ->
        all_texts =
          examples
          |> Enum.flat_map(fn ex ->
            responses = get_responses_for_intent(ex.intent)
            text = Map.get(ex, :text) || Enum.join(Map.get(ex, :tokens, []), " ")
            [text | responses]
          end)

        token_vocab =
          DataLoaders.build_lstm_vocabulary(
            Enum.map(all_texts, &%{text: &1, tokens: Tokenizer.tokenize_normalized(&1)}),
            max_vocab: config.max_vocab
          )

        positive_pairs =
          examples
          |> Enum.flat_map(fn ex ->
            responses = get_responses_for_intent(ex.intent)
            text = Map.get(ex, :text) || Enum.join(Map.get(ex, :tokens, []), " ")

            Enum.map(responses, fn response ->
              %{query: text, response: response, label: 1.0}
            end)
          end)
          |> Enum.filter(fn pair ->
            String.length(pair.query) > 0 and String.length(pair.response) > 0
          end)
          |> Enum.take(5000)

        negative_pairs = create_negative_pairs(examples, 2000)

        all_pairs = Enum.shuffle(positive_pairs ++ negative_pairs)
        split_idx = floor(length(all_pairs) * 0.9)
        {train_pairs, val_pairs} = Enum.split(all_pairs, split_idx)

        Logger.info("Prepared #{length(train_pairs)} train, #{length(val_pairs)} val pairs")

        {:ok,
         %{
           train: train_pairs,
           val: val_pairs,
           vocabularies: %{token_vocab: token_vocab}
         }}

      error ->
        error
    end
  end

  defp get_responses_for_intent(intent) do
    case TemplateStore.get_templates(intent) do
      templates when is_list(templates) and templates != [] ->
        templates
        |> Enum.map(fn t ->
          cond do
            is_binary(t) -> t
            is_map(t) -> Map.get(t, :text) || Map.get(t, "text") || ""
            true -> ""
          end
        end)
        |> Enum.filter(&(String.length(&1) > 0))
        |> Enum.take(3)

      _ ->
        []
    end
  end

  defp create_negative_pairs(examples, count) do
    all_responses =
      examples
      |> Enum.flat_map(fn ex -> get_responses_for_intent(ex.intent) end)
      |> Enum.uniq()
      |> Enum.filter(&(String.length(&1) > 0))

    if length(all_responses) < 10 do
      []
    else
      examples
      |> Enum.take(count)
      |> Enum.map(fn ex ->
        text = Map.get(ex, :text) || Enum.join(Map.get(ex, :tokens, []), " ")
        wrong_response = Enum.random(all_responses)
        %{query: text, response: wrong_response, label: 0.0}
      end)
      |> Enum.filter(fn pair -> String.length(pair.query) > 0 end)
    end
  end

  defp train_scorer(model, training_data, config) do
    train_batches =
      training_data.train
      |> Enum.chunk_every(config.batch_size)
      |> Enum.filter(fn batch -> length(batch) == config.batch_size end)
      |> Enum.map(fn batch ->
        queries =
          batch
          |> Enum.map(fn pair ->
            tokens = Tokenizer.tokenize_normalized(pair.query)

            indices =
              DataLoaders.tokens_to_indices(tokens, training_data.vocabularies.token_vocab)

            DataLoaders.pad_sequence(indices, config.max_query_length)
          end)
          |> Nx.tensor(type: :s64)

        responses =
          batch
          |> Enum.map(fn pair ->
            tokens = Tokenizer.tokenize_normalized(pair.response)

            indices =
              DataLoaders.tokens_to_indices(tokens, training_data.vocabularies.token_vocab)

            DataLoaders.pad_sequence(indices, config.max_response_length)
          end)
          |> Nx.tensor(type: :s64)

        labels =
          batch
          |> Enum.map(& &1.label)
          |> Nx.tensor(type: :f32)
          |> Nx.reshape({config.batch_size, 1})

        {%{"query" => queries, "response" => responses}, labels}
      end)

    Logger.info("Training on #{length(train_batches)} batches for #{config.epochs} epochs")

    loop =
      model
      |> Loop.trainer(
        :binary_cross_entropy,
        Optimizers.adam(learning_rate: config.learning_rate),
        log: 0
      )
      |> Loop.metric(:accuracy)

    Loop.run(loop, train_batches, Axon.ModelState.empty(), epochs: config.epochs, compiler: EXLA)
  end

  # Model architecture is reconstructed from config during load since Axon models
  # are built dynamically from hyperparameters (vocab_size, hidden_size, etc.)
  defp save_model(_model, params, vocabularies, config) do
    models_path =
      Map.get(config, :models_path) ||
        Application.get_env(:brain, :ml)[:models_path] ||
        Brain.priv_path("ml_models")

    lstm_path = Path.join(models_path, "lstm")
    File.mkdir_p!(lstm_path)

    save_path = Path.join(lstm_path, "response_scorer.term")

    portable_params = transfer_params_to_binary_backend(params)

    data = %{
      scorer_params: portable_params,
      vocabularies: vocabularies,
      config: config
    }

    binary = :erlang.term_to_binary(data)
    File.write!(save_path, binary)

    Logger.info("Response scorer saved to #{save_path}")
  end

  defp load_saved_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    save_path = Path.join([models_path, "lstm", "response_scorer.term"])
    Brain.ML.ModelStore.ensure_local("lstm/response_scorer.term", save_path)

    case File.read(save_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          model = build_scorer_model(data.vocabularies, data.config)
          portable_params = transfer_params_to_binary_backend(data.scorer_params)

          {:ok,
           %{
             scorer_model: model,
             scorer_params: portable_params,
             vocabularies: data.vocabularies,
             config: data.config
           }}
        rescue
          e ->
            Logger.warning(
              "LSTMResponse: Failed to deserialize model: #{Exception.message(e)}"
            )

            {:error, :corrupted_model}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

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

  defp fallback_generate(_query, intent, entities) do
    case Generator.generate(intent, entities) do
      {:ok, response, _type} -> {:ok, response, 0.5}
      error -> error
    end
  end
end
