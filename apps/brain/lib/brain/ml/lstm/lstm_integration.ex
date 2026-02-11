defmodule Brain.ML.LSTM.Integration do
  @moduledoc """
  Integration layer for using LSTM models alongside existing TF-IDF classifiers.

  This module is called from the main analysis pipeline as an **ensemble
  fallback**.  When `Brain.Analysis.SpeechActClassifier` obtains a low-confidence
  result from the LSTM `MultiTaskModel` (below the ensemble threshold), it
  delegates to `Integration.classify_intent/1` which combines TF-IDF and LSTM
  predictions via ensemble voting to produce a more robust classification.

  The pipeline still invokes LSTM models directly via
  `Brain.ML.LSTM.MultiTaskModel` and `Brain.ML.LSTM.UnifiedModel` for
  high-confidence predictions; this module is only consulted when the primary
  LSTM confidence is insufficient.

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
    tfidf_result = get_tfidf_prediction(text)
    lstm_result = get_lstm_prediction(text)

    case {tfidf_result, lstm_result, use_ensemble} do
      {{:ok, tfidf}, {:ok, lstm}, true} ->
        {:ok, ensemble_intent(tfidf, lstm)}

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
  """
  def classify_sentiment(text) do
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
        {:ok, ensemble_sentiment(lstm, tfidf)}

      {{:ok, lstm}, :unavailable} ->
        {:ok, lstm}

      {:unavailable, {:ok, tfidf}} ->
        {:ok, tfidf}

      {:unavailable, :unavailable} ->
        {:error, :no_sentiment_classifier}
    end
  end

  defp ensemble_sentiment(lstm, tfidf) do
    if lstm.label == tfidf.label do
      # Both agree -- boost confidence
      combined = 1.0 - (1.0 - lstm.confidence) * (1.0 - tfidf.confidence)
      %{label: lstm.label, confidence: combined}
    else
      # Disagree -- pick the higher confidence prediction
      if lstm.confidence >= tfidf.confidence do
        lstm
      else
        tfidf
      end
    end
  end

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
      lstm_intent: check_axon_model()
    }
  end

  defp get_tfidf_prediction(text) do
    case IntentClassifierSimple.classify(text) do
      {:ok, {intent, confidence}} -> {:ok, {intent, confidence}}
      {:ok, intent, confidence} -> {:ok, {intent, confidence}}
      other -> {:error, other}
    end
  end

  defp get_lstm_prediction(text) do
    cond do
      UnifiedModel.ready?() ->
        case UnifiedModel.classify_intent(text) do
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

  defp check_axon_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    path = Path.join([models_path, "lstm", "axon_intent.term"])
    File.exists?(path)
  end

  defp ensemble_intent({tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}) do
    cond do
      tfidf_intent == lstm_intent ->
        combined_conf = 1.0 - (1.0 - tfidf_conf) * (1.0 - lstm_conf)
        {tfidf_intent, combined_conf, :ensemble}

      lstm_conf > tfidf_conf + 0.2 ->
        {lstm_intent, lstm_conf, :lstm}

      tfidf_conf > lstm_conf + 0.2 ->
        {tfidf_intent, tfidf_conf, :tfidf}

      lstm_conf > 0.5 ->
        {lstm_intent, lstm_conf, :lstm}

      true ->
        {tfidf_intent, tfidf_conf, :tfidf}
    end
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
            {:ok, result} -> result
            {:error, _} -> %{label: :unknown, confidence: 0.0}
          end

        speech_act =
          case classify_speech_act(text) do
            {:ok, result} -> result
            {:error, _} -> %{label: :unknown, confidence: 0.0}
          end

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
end
