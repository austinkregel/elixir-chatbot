defmodule Brain.ML.IntentArbitrator do
  @moduledoc """
  Stacked meta-learner that arbitrates between LSTM and TF-IDF intent
  classifiers. Uses a small Axon model trained on features extracted from
  all available subsystems to decide which base classifier to trust.

  Follows the stacked generalization pattern: Level-0 classifiers (LSTM,
  TF-IDF) produce predictions, then this Level-1 meta-learner decides
  which prediction to use based on confidence profiles, structural signals,
  and other subsystem outputs.

  The model must be trained via `mix train_arbitrator` (or `mix train`).
  If the model file is missing or incompatible, the GenServer stays in
  degraded mode and returns `{:error, :not_ready}` for arbitration requests.
  """

  use GenServer
  require Logger

  alias Axon.Loop
  alias Brain.ML.SimpleClassifier
  alias Brain.Analysis.{IntentRegistry, SlotDetector}
  alias Polaris.Optimizers

  @feature_count 32
  @model_filename "intent_arbitrator.term"

  @doc "Returns the expected feature vector length for the arbitrator model."
  def feature_count, do: @feature_count

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  def arbitrate(features, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:arbitrate, features}, 5_000)
    catch
      :exit, _ -> {:error, :not_available}
    end
  end

  def stats(name \\ __MODULE__) do
    try do
      GenServer.call(name, :stats, 1_000)
    catch
      :exit, _ -> %{}
    end
  end

  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 30_000)
  end

  # --- Feature Extraction (stateless, callable without GenServer) ---

  @doc """
  Extracts a fixed-size feature vector from classifier outputs and
  subsystem signals. Missing inputs default to 0.0.

  Expected input map keys:
    - :lstm -- %{intent, confidence, scores} (required for arbitration)
    - :tfidf -- %{intent, confidence, scores} or nil
    - :text -- the raw input text (optional)
    - :structural -- structural analysis result (optional)
    - :keyword -- keyword analysis result (optional)
    - :pragmatic -- pragmatic analysis result (optional)
    - :memory -- memory analysis result (optional)
    - :sentiment -- sentiment result (optional)
    - :discourse -- discourse result (optional)
    - :entities -- list of detected entities (optional)
  """
  def extract_features(inputs) do
    lstm = inputs[:lstm] || %{}
    tfidf = inputs[:tfidf] || %{}
    text = inputs[:text] || ""
    structural = inputs[:structural] || %{}
    keyword = inputs[:keyword] || %{}
    _pragmatic = inputs[:pragmatic] || %{}
    memory = inputs[:memory] || %{}
    sentiment = inputs[:sentiment] || %{}
    discourse = inputs[:discourse] || %{}
    entities = inputs[:entities] || []

    lstm_conf = to_float(lstm[:confidence])
    tfidf_conf = to_float(tfidf[:confidence])
    lstm_intent = to_string(lstm[:intent] || "")
    tfidf_intent = to_string(tfidf[:intent] || "")

    lstm_scores = lstm[:scores] || []
    tfidf_scores = tfidf[:scores] || []

    lstm_margin = compute_margin(lstm_scores, lstm_conf)
    tfidf_margin = compute_margin(tfidf_scores, tfidf_conf)

    agree = if lstm_intent != "" and lstm_intent == tfidf_intent, do: 1.0, else: 0.0

    lstm_top_labels = top_labels(lstm_scores, 5)
    tfidf_top_labels = top_labels(tfidf_scores, 5)

    lstm_t2_in_tfidf =
      if length(lstm_top_labels) >= 2 do
        if Enum.at(lstm_top_labels, 1) in tfidf_top_labels, do: 1.0, else: 0.0
      else
        0.0
      end

    tfidf_t2_in_lstm =
      if length(tfidf_top_labels) >= 2 do
        if Enum.at(tfidf_top_labels, 1) in lstm_top_labels, do: 1.0, else: 0.0
      else
        0.0
      end

    words = String.split(text)
    word_count = length(words)

    entity_types = extract_entity_types(entities)

    [
      lstm_conf,
      tfidf_conf,
      lstm_margin,
      tfidf_margin,
      agree,
      lstm_conf - tfidf_conf,
      lstm_t2_in_tfidf,
      tfidf_t2_in_lstm,
      bool_to_float(structural[:is_question]),
      bool_to_float(structural[:is_imperative]),
      bool_to_float(structural[:has_modal]),
      bool_to_float(structural[:is_declarative]),
      to_float(keyword[:confidence]),
      if(word_count <= 3, do: 1.0, else: 0.0),
      min(word_count / 50.0, 1.0),
      to_float(memory[:confidence]),
      min(to_float(memory[:similar_count]) / 5.0, 1.0),
      to_float(sentiment[:confidence]),
      if(sentiment[:label] in [:neutral, "neutral"], do: 1.0, else: 0.0),
      to_float(discourse[:confidence]),
      domain_depth(lstm_intent),
      domain_depth(tfidf_intent),
      if(top_domain(lstm_intent) != "" and top_domain(lstm_intent) == top_domain(tfidf_intent),
        do: 1.0,
        else: 0.0
      ),
      # Entity/slot coherence features (9 new)
      min(length(entities) / 5.0, 1.0),
      entity_coherence(lstm_intent, entity_types),
      entity_coherence(tfidf_intent, entity_types),
      slot_fill_ratio(lstm_intent, entities),
      slot_fill_ratio(tfidf_intent, entities),
      bool_to_float(has_expected_entities?(lstm_intent)),
      bool_to_float(has_expected_entities?(tfidf_intent)),
      entity_domain_alignment(lstm_intent, entity_types),
      entity_domain_alignment(tfidf_intent, entity_types)
    ]
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(_opts) do
    state = %{
      model: nil,
      params: nil,
      ready: false,
      status: :initializing,
      arbitration_count: 0,
      lstm_wins: 0,
      tfidf_wins: 0
    }

    send(self(), :load_model)
    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  def handle_call({:arbitrate, _features}, _from, %{ready: false} = state) do
    {:reply, {:error, :not_ready}, state}
  end

  def handle_call({:arbitrate, features}, _from, state) do
    result = do_arbitrate(features, state)

    new_state =
      case result do
        {:lstm, _} ->
          %{state | arbitration_count: state.arbitration_count + 1, lstm_wins: state.lstm_wins + 1}

        {:tfidf, _} ->
          %{state | arbitration_count: state.arbitration_count + 1, tfidf_wins: state.tfidf_wins + 1}

        _ ->
          %{state | arbitration_count: state.arbitration_count + 1}
      end

    {:reply, result, new_state}
  end

  def handle_call(:stats, _from, state) do
    stats = %{
      ready: state.ready,
      status: state.status,
      arbitration_count: state.arbitration_count,
      lstm_wins: state.lstm_wins,
      tfidf_wins: state.tfidf_wins,
      agreement_rate:
        if state.arbitration_count > 0 do
          (state.lstm_wins + state.tfidf_wins) / state.arbitration_count
        else
          0.0
        end
    }

    {:reply, stats, state}
  end

  def handle_call(:reload, _from, state) do
    case load_model_file() do
      {:ok, model, params} ->
        {:reply, :ok, %{state | model: model, params: params, ready: true, status: :ready}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_info(:load_model, state) do
    case load_model_file() do
      {:ok, model, params} ->
        Logger.info("IntentArbitrator: Loaded trained model")
        {:noreply, %{state | model: model, params: params, ready: true, status: :ready}}

      {:error, :not_found} ->
        Logger.error("IntentArbitrator: No trained model found. Run `mix train_arbitrator` to train.")
        {:noreply, %{state | status: :unavailable}}

      {:error, reason} ->
        Logger.error("IntentArbitrator: Failed to load model (#{inspect(reason)}). Run `mix train_arbitrator` to retrain.")
        {:noreply, %{state | status: :unavailable}}
    end
  end

  # --- Internal ---

  defp do_arbitrate(features, state) do
    input_tensor = Nx.tensor([features], type: :f32)

    try do
      output = Axon.predict(state.model, state.params, %{"features" => input_tensor}, compiler: EXLA)
      probs = output |> Nx.to_flat_list()

      case probs do
        [lstm_prob, tfidf_prob] ->
          if lstm_prob >= tfidf_prob do
            {:lstm, lstm_prob}
          else
            {:tfidf, tfidf_prob}
          end

        _ ->
          {:lstm, 0.5}
      end
    rescue
      e ->
        Logger.warning("IntentArbitrator: Inference failed (#{Exception.message(e)}), defaulting to TF-IDF")
        {:tfidf, 0.5}
    end
  end

  def build_model do
    Axon.input("features", shape: {nil, @feature_count})
    |> Axon.dense(32, activation: :relu, name: "arb_hidden1")
    |> Axon.dropout(rate: 0.2)
    |> Axon.dense(16, activation: :relu, name: "arb_hidden2")
    |> Axon.dense(2, activation: :softmax, name: "arb_output")
  end

  def train(training_data, opts \\ []) do
    epochs = Keyword.get(opts, :epochs, 50)
    batch_size = Keyword.get(opts, :batch_size, 16)
    learning_rate = Keyword.get(opts, :learning_rate, 1.0e-3)

    model = build_model()

    batches =
      training_data
      |> Enum.shuffle()
      |> Enum.chunk_every(batch_size)
      |> Enum.filter(fn batch -> length(batch) == batch_size end)
      |> Enum.map(fn batch ->
        features =
          batch
          |> Enum.map(fn %{features: f} -> f end)
          |> Nx.tensor(type: :f32)

        labels =
          batch
          |> Enum.map(fn %{label: l} ->
            case l do
              :lstm -> [1.0, 0.0]
              :tfidf -> [0.0, 1.0]
              _ -> [0.5, 0.5]
            end
          end)
          |> Nx.tensor(type: :f32)

        {%{"features" => features}, labels}
      end)

    if batches == [] do
      {:error, :insufficient_training_data}
    else
      loop =
        model
        |> Loop.trainer(
          :categorical_cross_entropy,
          Optimizers.adam(learning_rate: learning_rate),
          log: 1
        )
        |> Loop.metric(:accuracy)

      params = Loop.run(loop, batches, Axon.ModelState.empty(), epochs: epochs, compiler: EXLA)
      {:ok, model, params}
    end
  end

  def save_model(params) do
    path = model_path()
    File.mkdir_p!(Path.dirname(path))

    portable =
      if is_struct(params, Axon.ModelState) do
        Axon.ModelState.new(transfer_to_binary(params.data))
      else
        transfer_to_binary(params)
      end

    data = %{params: portable, feature_count: @feature_count}
    binary = :erlang.term_to_binary(data)
    File.write!(path, binary)
    Logger.info("IntentArbitrator: Model saved to #{path}")
    :ok
  end

  defp load_model_file do
    path = model_path()

    case File.read(path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          stored_fc = Map.get(data, :feature_count, 0)

          if stored_fc != @feature_count do
            Logger.warning(
              "IntentArbitrator: Model feature count mismatch " <>
                "(stored=#{stored_fc}, expected=#{@feature_count}). Needs retraining."
            )

            {:error, :feature_count_mismatch}
          else
            model = build_model()
            {:ok, model, ensure_model_state(data.params)}
          end
        rescue
          e ->
            Logger.error("IntentArbitrator: Failed to deserialize model: #{inspect(e)}")
            {:error, :corrupted_model}
        end

      {:error, :enoent} ->
        {:error, :not_found}

      {:error, reason} ->
        {:error, reason}
    end
  end

  def generate_training_data(gold_standard) do
    all_texts = Enum.map(gold_standard, & &1["text"])
    lstm_lookup = batch_classify_lstm(all_texts)
    entity_lookup = batch_extract_entities(all_texts)

    all_examples =
      gold_standard
      |> Enum.map(fn example ->
        text = example["text"]
        expected = example["intent"]

        lstm_result = Map.get(lstm_lookup, text, %{intent: nil, confidence: 0.0, scores: []})
        tfidf_result = safe_classify_tfidf(text)
        entities = Map.get(entity_lookup, text, [])

        lstm_correct = to_string(lstm_result[:intent]) == to_string(expected)
        tfidf_correct = to_string(tfidf_result[:intent]) == to_string(expected)

        label =
          cond do
            lstm_correct and not tfidf_correct -> :lstm
            tfidf_correct and not lstm_correct -> :tfidf
            lstm_correct and tfidf_correct ->
              if (lstm_result[:confidence] || 0) >= (tfidf_result[:confidence] || 0),
                do: :lstm,
                else: :tfidf
            true -> :skip
          end

        category =
          cond do
            lstm_correct and tfidf_correct -> :both_correct
            lstm_correct and not tfidf_correct -> :only_lstm
            tfidf_correct and not lstm_correct -> :only_tfidf
            true -> :both_wrong
          end

        features =
          extract_features(%{
            lstm: lstm_result,
            tfidf: tfidf_result,
            text: text,
            entities: entities
          })

        %{
          features: features,
          label: label,
          category: category,
          text: text,
          expected: expected
        }
      end)

    diagnostics = training_data_diagnostics(all_examples)
    log_training_diagnostics(diagnostics)

    all_examples
    |> Enum.filter(fn ex -> ex.label in [:lstm, :tfidf] end)
  end

  @doc "Returns diagnostic counts from labeled training data (before filtering)."
  def training_data_diagnostics(all_examples) do
    freqs = Enum.frequencies_by(all_examples, & &1.category)

    both_correct = Map.get(freqs, :both_correct, 0)
    only_lstm = Map.get(freqs, :only_lstm, 0)
    only_tfidf = Map.get(freqs, :only_tfidf, 0)
    both_wrong = Map.get(freqs, :both_wrong, 0)

    both_correct_lstm_wins =
      Enum.count(all_examples, &(&1.category == :both_correct and &1.label == :lstm))

    both_correct_tfidf_wins =
      Enum.count(all_examples, &(&1.category == :both_correct and &1.label == :tfidf))

    %{
      total: length(all_examples),
      both_correct: both_correct,
      both_correct_lstm_wins: both_correct_lstm_wins,
      both_correct_tfidf_wins: both_correct_tfidf_wins,
      only_lstm_correct: only_lstm,
      only_tfidf_correct: only_tfidf,
      both_wrong: both_wrong,
      usable: both_correct + only_lstm + only_tfidf
    }
  end

  @doc """
  Cross-validated training data generation. Splits gold standard into k folds,
  trains a fresh TF-IDF model on k-1 folds, and evaluates on the held-out fold.
  This eliminates data leakage where TF-IDF memorizes its training data.
  The LSTM model is kept as-is (too expensive to retrain per fold).
  """
  def generate_training_data_cv(gold_standard, opts \\ []) do
    k = Keyword.get(opts, :folds, 5)

    folds = stratified_kfold_split(gold_standard, k)

    all_texts = Enum.map(gold_standard, & &1["text"])
    lstm_lookup = batch_classify_lstm(all_texts)
    entity_lookup = batch_extract_entities(all_texts)

    all_examples =
      folds
      |> Enum.with_index()
      |> Enum.flat_map(fn {held_out, fold_idx} ->
        train_set =
          folds
          |> Enum.with_index()
          |> Enum.reject(fn {_, idx} -> idx == fold_idx end)
          |> Enum.flat_map(fn {fold, _} -> fold end)

        tfidf_training_data =
          Enum.map(train_set, fn ex -> {ex["text"], ex["intent"]} end)

        fold_model = SimpleClassifier.train(tfidf_training_data)

        Enum.map(held_out, fn example ->
          text = example["text"]
          expected = example["intent"]

          lstm_result = Map.get(lstm_lookup, text, %{intent: nil, confidence: 0.0, scores: []})
          tfidf_result = classify_with_fold_model(text, fold_model)
          entities = Map.get(entity_lookup, text, [])

          lstm_correct = to_string(lstm_result[:intent]) == to_string(expected)
          tfidf_correct = to_string(tfidf_result[:intent]) == to_string(expected)

          label =
            cond do
              lstm_correct and not tfidf_correct -> :lstm
              tfidf_correct and not lstm_correct -> :tfidf
              lstm_correct and tfidf_correct ->
                if (lstm_result[:confidence] || 0) >= (tfidf_result[:confidence] || 0),
                  do: :lstm,
                  else: :tfidf
              true -> :skip
            end

          category =
            cond do
              lstm_correct and tfidf_correct -> :both_correct
              lstm_correct and not tfidf_correct -> :only_lstm
              tfidf_correct and not lstm_correct -> :only_tfidf
              true -> :both_wrong
            end

          features =
            extract_features(%{
              lstm: lstm_result,
              tfidf: tfidf_result,
              text: text,
              entities: entities
            })

          %{
            features: features,
            label: label,
            category: category,
            text: text,
            expected: expected
          }
        end)
      end)

    diagnostics = training_data_diagnostics(all_examples)
    log_training_diagnostics(diagnostics)

    all_examples
    |> Enum.filter(fn ex -> ex.label in [:lstm, :tfidf] end)
  end

  defp classify_with_fold_model(text, model) do
    case SimpleClassifier.classify_with_details(text, model, top_k: 5) do
      {:ok, label, score, details} ->
        scores =
          Enum.map(details.top_k, fn
            {l, s} -> {l, s}
            _ -> {"unknown", 0.0}
          end)

        %{intent: label, confidence: score, scores: scores}

      _ ->
        %{intent: nil, confidence: 0.0, scores: []}
    end
  rescue
    _ -> %{intent: nil, confidence: 0.0, scores: []}
  end

  defp stratified_kfold_split(examples, k) do
    by_intent = Enum.group_by(examples, fn ex -> ex["intent"] end)

    fold_lists = List.duplicate([], k)

    Enum.reduce(by_intent, fold_lists, fn {_intent, group}, folds ->
      shuffled = Enum.shuffle(group)

      shuffled
      |> Enum.with_index()
      |> Enum.reduce(folds, fn {example, idx}, acc ->
        fold_idx = rem(idx, k)
        List.update_at(acc, fold_idx, fn fold -> [example | fold] end)
      end)
    end)
  end

  defp batch_classify_lstm(texts) do
    case Brain.ML.LSTM.UnifiedModel.batch_classify_intents(texts) do
      {:ok, results} ->
        Enum.zip(texts, results)
        |> Map.new(fn {text, result} ->
          {text, %{intent: result[:label], confidence: result[:confidence], scores: result[:scores] || []}}
        end)

      _ ->
        Map.new(texts, fn text -> {text, %{intent: nil, confidence: 0.0, scores: []}} end)
    end
  rescue
    _ -> Map.new(texts, fn text -> {text, %{intent: nil, confidence: 0.0, scores: []}} end)
  catch
    _, _ -> Map.new(texts, fn text -> {text, %{intent: nil, confidence: 0.0, scores: []}} end)
  end

  defp batch_extract_entities(texts) do
    Map.new(texts, fn text -> {text, safe_extract_entities(text)} end)
  end

  defp safe_extract_entities(text) do
    if Code.ensure_loaded?(Brain.ML.EntityExtractor) and
         function_exported?(Brain.ML.EntityExtractor, :extract_entities, 1) do
      Brain.ML.EntityExtractor.extract_entities(text)
    else
      []
    end
  rescue
    _ -> []
  catch
    _, _ -> []
  end

  defp safe_classify_tfidf(text) do
    case Brain.ML.IntentClassifierSimple.classify(text, with_details: true, top_k: 5) do
      {:ok, %{intent: intent, confidence: conf} = result} ->
        top_k = Map.get(result, :top_k, [])

        scores =
          Enum.map(top_k, fn
            {label, score} -> {label, score}
            %{intent: i, score: s} -> {i, s}
            _ -> {"unknown", 0.0}
          end)

        %{intent: intent, confidence: conf, scores: scores}

      {:ok, {intent, conf}} ->
        %{intent: intent, confidence: conf, scores: []}

      _ ->
        %{intent: nil, confidence: 0.0, scores: []}
    end
  rescue
    _ -> %{intent: nil, confidence: 0.0, scores: []}
  catch
    _, _ -> %{intent: nil, confidence: 0.0, scores: []}
  end

  defp log_training_diagnostics(d) do
    Logger.info("""
    IntentArbitrator training data diagnostics:
      Total examples:          #{d.total}
      Both correct:            #{d.both_correct} (LSTM wins: #{d.both_correct_lstm_wins}, TF-IDF wins: #{d.both_correct_tfidf_wins})
      Only LSTM correct:       #{d.only_lstm_correct}
      Only TF-IDF correct:     #{d.only_tfidf_correct}
      Both wrong (excluded):   #{d.both_wrong}
      Usable training examples: #{d.usable}
    """)
  end

  defp model_path do
    models_path =
      Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")

    Path.join(models_path, @model_filename)
  end

  defp ensure_model_state(%Axon.ModelState{} = state), do: state
  defp ensure_model_state(params) when is_map(params), do: Axon.ModelState.new(params)

  defp transfer_to_binary(%Nx.Tensor{} = tensor) do
    Nx.backend_copy(tensor, Nx.BinaryBackend)
  end

  defp transfer_to_binary(%Axon.ModelState{} = state) do
    Axon.ModelState.new(transfer_to_binary(state.data))
  end

  defp transfer_to_binary(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {k, transfer_to_binary(v)} end)
  end

  defp transfer_to_binary(other), do: other

  # --- Entity/Slot Coherence Helpers ---

  defp extract_entity_types(entities) when is_list(entities) do
    entities
    |> Enum.map(fn e ->
      e[:entity_type] || e["entity_type"] || e["type"] || e[:type] || "unknown"
    end)
    |> Enum.map(&to_string/1)
    |> Enum.map(&String.downcase/1)
    |> MapSet.new()
  end

  defp extract_entity_types(_), do: MapSet.new()

  defp entity_coherence("", _entity_types), do: 0.0
  defp entity_coherence(intent, entity_types) when is_binary(intent) do
    expected = safe_expected_entity_types(intent)

    if expected == [] or MapSet.size(entity_types) == 0 do
      0.0
    else
      expected_set = expected |> Enum.map(&String.downcase/1) |> MapSet.new()
      matched = MapSet.intersection(expected_set, entity_types) |> MapSet.size()
      matched / max(MapSet.size(expected_set), 1)
    end
  end

  defp slot_fill_ratio("", _entities), do: 0.0
  defp slot_fill_ratio(intent, entities) when is_binary(intent) do
    schema = safe_get_slot_schema(intent)

    case schema do
      nil -> 0.0
      %{required: required} when is_list(required) and required != [] ->
        entity_types = extract_entity_types(entities)
        filled = Enum.count(required, fn slot ->
          slot_str = to_string(slot) |> String.downcase()
          MapSet.member?(entity_types, slot_str)
        end)
        filled / length(required)
      _ -> 0.0
    end
  end

  defp has_expected_entities?(""), do: false
  defp has_expected_entities?(intent) when is_binary(intent) do
    safe_expected_entity_types(intent) != []
  end

  defp entity_domain_alignment("", _entity_types), do: 0.0
  defp entity_domain_alignment(intent, entity_types) when is_binary(intent) do
    domain = top_domain(intent)

    if domain == "" or MapSet.size(entity_types) == 0 do
      0.0
    else
      matching = Enum.count(entity_types, fn et ->
        String.contains?(et, domain) or String.contains?(domain, et)
      end)
      min(matching / max(MapSet.size(entity_types), 1), 1.0)
    end
  end

  defp safe_expected_entity_types(intent) do
    if Code.ensure_loaded?(IntentRegistry) and
         function_exported?(IntentRegistry, :expected_entity_types, 1) do
      IntentRegistry.expected_entity_types(intent) || []
    else
      []
    end
  rescue
    _ -> []
  end

  defp safe_get_slot_schema(intent) do
    if Code.ensure_loaded?(SlotDetector) and
         function_exported?(SlotDetector, :get_schema, 1) do
      SlotDetector.get_schema(intent)
    else
      nil
    end
  rescue
    _ -> nil
  end

  defp to_float(nil), do: 0.0
  defp to_float(val) when is_float(val), do: val
  defp to_float(val) when is_integer(val), do: val / 1.0
  defp to_float(%Nx.Tensor{} = t), do: Nx.to_number(t)
  defp to_float(_), do: 0.0

  defp bool_to_float(nil), do: 0.0
  defp bool_to_float(true), do: 1.0
  defp bool_to_float(false), do: 0.0
  defp bool_to_float(_), do: 0.0

  defp compute_margin(scores, top_confidence) when is_list(scores) and length(scores) >= 2 do
    sorted =
      scores
      |> Enum.map(fn
        {_label, score} -> score
        %{score: score} -> score
        _ -> 0.0
      end)
      |> Enum.sort(:desc)

    case sorted do
      [first, second | _] -> first - second
      _ -> top_confidence
    end
  end

  defp compute_margin(_, top_confidence), do: top_confidence

  defp top_labels(scores, n) when is_list(scores) do
    scores
    |> Enum.map(fn
      {label, _score} -> to_string(label)
      %{intent: label} -> to_string(label)
      _ -> ""
    end)
    |> Enum.reject(&(&1 == ""))
    |> Enum.take(n)
  end

  defp top_labels(_, _), do: []

  defp domain_depth(intent) when is_binary(intent) and intent != "" do
    parts = String.split(intent, ".")
    min((length(parts) - 1) / 4.0, 1.0)
  end

  defp domain_depth(_), do: 0.0

  defp top_domain(intent) when is_binary(intent) and intent != "" do
    intent |> String.split(".") |> hd()
  end

  defp top_domain(_), do: ""
end
