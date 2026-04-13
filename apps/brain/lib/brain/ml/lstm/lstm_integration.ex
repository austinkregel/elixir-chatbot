defmodule Brain.ML.LSTM.Integration do
  @moduledoc """
  Integration layer for using LSTM models alongside existing TF-IDF classifiers.

  This module is called from the main analysis pipeline as an **ensemble
  fallback**.  When `Brain.Analysis.SpeechActClassifier` obtains a low-confidence
  result from the LSTM `UnifiedModel` (below the ensemble threshold), it
  delegates to `Integration.classify_intent/1` which combines TF-IDF and LSTM
  predictions via ensemble voting to produce a more robust classification.

  The pipeline still invokes LSTM models directly via
  `Brain.ML.LSTM.UnifiedModel` for high-confidence predictions; this module
  is only consulted when the primary LSTM confidence is insufficient.

  ## Provides

  1. Get hybrid predictions combining TF-IDF and LSTM
  2. Ensemble voting for improved accuracy
  3. Sentiment classification via trained TF-IDF or LSTM models
  4. Speech act classification via the SpeechActClassifier pipeline

  ## Hybrid Approach

  The system uses both TF-IDF (fast, interpretable) and LSTM (accurate, contextual):

      User Input
          │
          ├──> TF-IDF Classifier (fast, always available)
          │         │
          │         ▼
          │    {intent, confidence}
          │
          └──> LSTM Classifier (if available)
                    │
                    ▼
               {intent, confidence}

          │         │
          ▼         ▼
       Ensemble Combiner
               │
               ▼
       Final Prediction

  ## Usage

      # Get best prediction using all available models
      Integration.classify_intent("What's the weather?")
      # => {:ok, {"weather.query", 0.87, :ensemble}}

      # Get full analysis with all models
      Integration.analyze("Play some music")
      # => {:ok, %{
      #   intent: {"music.play", 0.92},
      #   entities: [...],
      #   source: :lstm
      # }}
  """

  alias Brain.ML.EntityExtractor
  alias Brain.ML.LSTM
  alias Brain.ML.GCN
  alias Brain.Graph.Training, as: GraphTraining
  require Logger

  alias Brain.ML.IntentClassifierSimple
  alias Brain.ML.SentimentClassifierSimple
  alias LSTM.{AxonTrainer, UnifiedModel}

  @doc """
  Classify intent using the best available method.

  Returns `{:ok, {intent, confidence, source}}` where source is:
  - `:lstm` - LSTM model prediction
  - `:tfidf` - TF-IDF model prediction
  - `:ensemble` - Combined prediction from both
  """
  def classify_intent(text, opts \\ []) do
    use_ensemble = Keyword.get(opts, :ensemble, true)
    previous_intent = Keyword.get(opts, :previous_intent)
    tfidf_result = get_tfidf_prediction(text)
    lstm_result = get_lstm_prediction(text)
    gcn_result = get_gcn_prediction(text)

    case {tfidf_result, lstm_result, use_ensemble} do
      {{:ok, tfidf}, {:ok, lstm}, true} ->
        {:ok, ensemble_intent(tfidf, lstm, previous_intent, gcn_result)}

      {_, {:ok, {intent, conf}}, _} when conf > 0.7 ->
        {:ok, {intent, conf, :lstm}}

      {{:ok, {intent, conf}}, _, _} ->
        {:ok, {intent, conf, :tfidf}}

      _ ->
        {:error, :no_classifier_available}
    end
  end

  @doc """
  Get full NLP analysis using LSTM if available, with TF-IDF components.
  """
  def analyze(text, opts \\ []) do
    case UnifiedModel.ready?() && UnifiedModel.analyze(text) do
      {:ok, result} ->
        {:ok, Map.put(result, :source, :lstm)}

      _ ->
        fallback_analysis(text, opts)
    end
  end

  @doc """
  Classify sentiment using ensemble of available models (LSTM + TF-IDF).

  When both LSTM and TF-IDF are available, uses confidence-weighted voting.
  When only one is available, uses that model directly.
  When neither is available, returns `{:error, :no_sentiment_classifier}`.

  ## Options

  - `:atlas_context` - Optional map with Atlas-derived context for disambiguation.
    Keys: `:topic` (current conversation topic), `:entity_sentiments` (list of
    `%{entity: String.t(), sentiment: atom}` from prior conversation).
    Used when LSTM and TF-IDF disagree.
  """
  def classify_sentiment(text, opts \\ []) do
    atlas_context = Keyword.get(opts, :atlas_context)

    lstm_result =
      if UnifiedModel.ready?() do
        case UnifiedModel.classify_sentiment(text) do
          {:ok, result} -> {:ok, result}
          _ -> :unavailable
        end
      else
        :unavailable
      end

    tfidf_result =
      if SentimentClassifierSimple.ready?() do
        case SentimentClassifierSimple.classify(text) do
          {:ok, result} -> {:ok, result}
          _ -> :unavailable
        end
      else
        :unavailable
      end

    case {lstm_result, tfidf_result} do
      {{:ok, lstm}, {:ok, tfidf}} ->
        {:ok, ensemble_sentiment(lstm, tfidf, atlas_context)}

      {{:ok, lstm}, :unavailable} ->
        {:ok, lstm}

      {:unavailable, {:ok, tfidf}} ->
        {:ok, tfidf}

      {:unavailable, :unavailable} ->
        {:error, :no_sentiment_classifier}
    end
  end

  defp ensemble_sentiment(lstm, tfidf, atlas_context) do
    if lstm.label == tfidf.label do
      combined = 1.0 - (1.0 - lstm.confidence) * (1.0 - tfidf.confidence)
      %{label: lstm.label, confidence: combined}
    else
      atlas_bias = atlas_sentiment_bias(atlas_context)

      {lstm_adj, tfidf_adj} =
        case atlas_bias do
          nil ->
            {lstm.confidence, tfidf.confidence}

          bias_label ->
            lstm_boost = if lstm.label == bias_label, do: 0.1, else: 0.0
            tfidf_boost = if tfidf.label == bias_label, do: 0.1, else: 0.0
            {lstm.confidence + lstm_boost, tfidf.confidence + tfidf_boost}
        end

      if lstm_adj >= tfidf_adj do
        %{label: lstm.label, confidence: lstm.confidence}
      else
        %{label: tfidf.label, confidence: tfidf.confidence}
      end
    end
  end

  defp atlas_sentiment_bias(nil), do: nil

  defp atlas_sentiment_bias(context) when is_map(context) do
    entity_sentiments = Map.get(context, :entity_sentiments, [])

    if entity_sentiments == [] do
      nil
    else
      frequencies =
        Enum.frequencies_by(entity_sentiments, fn es ->
          Map.get(es, :sentiment) || Map.get(es, "sentiment")
        end)

      case Enum.max_by(frequencies, fn {_label, count} -> count end, fn -> nil end) do
        {label, count} when count >= 2 -> label
        _ -> nil
      end
    end
  end

  defp atlas_sentiment_bias(_), do: nil

  @doc """
  Get speech act using LSTM if available, otherwise SpeechActClassifier pipeline.

  Returns `{:ok, %{label: atom, confidence: float}}` or `{:error, reason}`.
  """
  def classify_speech_act(text) do
    case UnifiedModel.ready?() && UnifiedModel.classify_speech_act(text) do
      {:ok, result} ->
        {:ok, result}

      _ ->
        result = Brain.Analysis.SpeechActClassifier.classify(text)
        {:ok, %{label: result.category, confidence: result.confidence}}
    end
  end

  @doc """
  Extract entities using LSTM NER if available, otherwise entity extractor.
  """
  def extract_entities(text, opts \\ []) do
    case UnifiedModel.ready?() && UnifiedModel.extract_entities(text) do
      {:ok, entities} when is_list(entities) and entities != [] ->
        {:ok, entities}

      _ ->
        case EntityExtractor.extract_entities(text, opts) do
          entities when is_list(entities) -> {:ok, entities}
          {:ok, entities} -> {:ok, entities}
          _other -> {:ok, []}
        end
    end
  end

  @doc "Check if LSTM models are available and ready."
  def lstm_available? do
    UnifiedModel.ready?()
  end

  @doc "Get status of all available models."
  def model_status do
    %{
      tfidf: IntentClassifierSimple.ready?(),
      tfidf_sentiment: SentimentClassifierSimple.ready?(),
      lstm_unified: UnifiedModel.ready?(),
      lstm_intent: check_axon_model(),
      gcn: gcn_available?()
    }
  end

  @doc "Check if GCN model is available."
  def gcn_available? do
    GCN.Model.ready?()
  end

  defp get_tfidf_prediction(text) do
    case IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: confidence}} -> {:ok, {intent, confidence}}
      {:ok, {intent, confidence}} -> {:ok, {intent, confidence}}
      {:ok, intent, confidence} -> {:ok, {intent, confidence}}
      other -> {:error, other}
    end
  end

  defp get_lstm_prediction(text) do
    cond do
      UnifiedModel.ready?() ->
        case UnifiedModel.classify_intent(text) do
          {:ok, %{label: intent, confidence: confidence}} -> {:ok, {intent, confidence}}
          {:ok, {intent, confidence}} -> {:ok, {intent, confidence}}
          _ -> {:error, :lstm_failed}
        end

      check_axon_model() ->
        case AxonTrainer.load_model() do
          {:ok, model} ->
            {intent, confidence} = AxonTrainer.classify(text, model)
            {:ok, {intent, confidence}}

          _ ->
            {:error, :no_model}
        end

      true ->
        {:error, :no_lstm}
    end
  end

  defp get_gcn_prediction(text) do
    if GCN.Model.ready?() do
      case GCN.Model.classify(text) do
        {:ok, {intent, confidence}} -> {:ok, {intent, confidence}}
        _ -> {:error, :gcn_failed}
      end
    else
      log_once(:gcn_not_ready, "GCN model not ready, skipping GCN vote")
      :telemetry.execute([:brain, :model, :unavailable], %{}, %{model: :gcn_intent, reason: :not_ready})
      {:error, :gcn_not_ready}
    end
  end

  defp check_axon_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    path = Path.join([models_path, "lstm", "axon_intent.term"])
    File.exists?(path)
  end

  defp ensemble_intent({tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}, previous_intent, gcn_result) do
    {tfidf_conf, lstm_conf} =
      apply_graph_prior_boost({tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}, previous_intent)

    candidates = [
      {tfidf_intent, tfidf_conf, :tfidf},
      {lstm_intent, lstm_conf, :lstm}
    ]

    candidates = case gcn_result do
      {:ok, {gcn_intent, gcn_conf}} ->
        [{gcn_intent, gcn_conf, :gcn} | candidates]

      _ ->
        candidates
    end

    intent_votes = Enum.group_by(candidates, fn {intent, _conf, _src} -> intent end)

    best = intent_votes
    |> Enum.map(fn {intent, votes} ->
      if length(votes) >= 2 do
        combined = Enum.reduce(votes, 1.0, fn {_, conf, _}, acc ->
          acc * (1.0 - conf)
        end)
        {intent, 1.0 - combined, :ensemble}
      else
        [{_, conf, src}] = votes
        {intent, conf, src}
      end
    end)
    |> Enum.max_by(fn {_, conf, _} -> conf end)

    best
  end

  defp apply_graph_prior_boost({_ti, tfidf_conf}, {_li, lstm_conf}, nil), do: {tfidf_conf, lstm_conf}

  defp apply_graph_prior_boost({tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}, previous_intent)
       when is_binary(previous_intent) do
    priors = GraphTraining.extract_intent_priors()

    scores = [{tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}]
    boosted = GraphTraining.apply_intent_priors(scores, previous_intent, priors)

    boosted_map = Map.new(boosted)
    {Map.get(boosted_map, tfidf_intent, tfidf_conf), Map.get(boosted_map, lstm_intent, lstm_conf)}
  end

  defp apply_graph_prior_boost({_tfidf_intent, tfidf_conf}, {_lstm_intent, lstm_conf}, _) do
    {tfidf_conf, lstm_conf}
  end

  defp fallback_analysis(text, _opts) do
    case get_tfidf_prediction(text) do
      {:ok, {intent, confidence}} ->
        entities =
          case EntityExtractor.extract_entities(text) do
            entities when is_list(entities) -> entities
            _ -> []
          end

        sentiment =
          case classify_sentiment(text) do
            {:ok, result} ->
              result

            {:error, reason} ->
              raise "Sentiment classification failed: #{inspect(reason)}. " <>
                      "Run `mix train` to train the sentiment classifier."
          end

        {:ok, speech_act} = classify_speech_act(text)

        {:ok,
         %{
           intent: {intent, confidence},
           entities: entities,
           sentiment: sentiment,
           speech_act: speech_act,
           source: :tfidf
         }}

      error ->
        error
    end
  end

  defp log_once(key, message) do
    pt_key = {__MODULE__, :logged, key}
    unless :persistent_term.get(pt_key, false) do
      Logger.warning(message)
      :persistent_term.put(pt_key, true)
    end
  end
end
