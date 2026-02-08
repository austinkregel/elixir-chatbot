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
  2. Fallback gracefully when LSTM is not available
  3. Ensemble voting for improved accuracy
  
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
  
  require Logger
  
  alias Brain.ML.IntentClassifierSimple
  alias Brain.ML.LSTM.{AxonTrainer, UnifiedModel}
  
  @doc """
  Classify intent using the best available method.
  
  Returns `{:ok, {intent, confidence, source}}` where source is:
  - `:lstm` - LSTM model prediction
  - `:tfidf` - TF-IDF model prediction
  - `:ensemble` - Combined prediction from both
  """
  def classify_intent(text, opts \\ []) do
    use_ensemble = Keyword.get(opts, :ensemble, true)
    
    # Get TF-IDF prediction (always available)
    tfidf_result = get_tfidf_prediction(text)
    
    # Try LSTM prediction
    lstm_result = get_lstm_prediction(text)
    
    case {tfidf_result, lstm_result, use_ensemble} do
      # Both available - use ensemble
      {{:ok, tfidf}, {:ok, lstm}, true} ->
        {:ok, ensemble_intent(tfidf, lstm)}
        
      # Only LSTM available and high confidence - use it
      {_, {:ok, {intent, conf}}, _} when conf > 0.7 ->
        {:ok, {intent, conf, :lstm}}
        
      # Only TF-IDF available or LSTM low confidence
      {{:ok, {intent, conf}}, _, _} ->
        {:ok, {intent, conf, :tfidf}}
        
      # Neither available
      _ ->
        {:error, :no_classifier_available}
    end
  end
  
  @doc """
  Get full NLP analysis using LSTM if available, falling back to TF-IDF.
  """
  def analyze(text, opts \\ []) do
    # Try unified model first
    case UnifiedModel.ready?() && UnifiedModel.analyze(text) do
      {:ok, result} ->
        {:ok, Map.put(result, :source, :lstm)}
        
      _ ->
        # Fall back to TF-IDF-based analysis
        fallback_analysis(text, opts)
    end
  end
  
  @doc """
  Get sentiment using LSTM if available, falling back to keyword heuristics.
  """
  def classify_sentiment(text) do
    case UnifiedModel.ready?() && UnifiedModel.classify_sentiment(text) do
      {:ok, result} -> 
        {:ok, result}
        
      _ -> 
        # Fallback to keyword-based sentiment
        {:ok, keyword_sentiment(text)}
    end
  end
  
  @doc """
  Get speech act using LSTM if available, falling back to structural analysis.
  """
  def classify_speech_act(text) do
    case UnifiedModel.ready?() && UnifiedModel.classify_speech_act(text) do
      {:ok, result} -> 
        {:ok, result}
        
      _ -> 
        # Fallback to structural analysis
        {:ok, structural_speech_act(text)}
    end
  end
  
  @doc """
  Extract entities using LSTM NER if available, falling back to gazetteer.
  """
  def extract_entities(text, opts \\ []) do
    case UnifiedModel.ready?() && UnifiedModel.extract_entities(text) do
      {:ok, entities} when is_list(entities) and length(entities) > 0 -> 
        {:ok, entities}
        
      _ -> 
        # Fallback to gazetteer-based extraction
        case Brain.ML.EntityExtractor.extract_entities(text, opts) do
          entities when is_list(entities) -> {:ok, entities}
          {:ok, entities} -> {:ok, entities}
          other -> {:ok, []}
        end
    end
  end
  
  @doc """
  Check if LSTM models are available and ready.
  """
  def lstm_available? do
    UnifiedModel.ready?()
  end
  
  @doc """
  Get status of all available models.
  """
  def model_status do
    %{
      tfidf: IntentClassifierSimple.ready?(),
      lstm_unified: UnifiedModel.ready?(),
      lstm_intent: check_axon_model()
    }
  end
  
  # ============================================================================
  # Private: Predictions
  # ============================================================================
  
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
  
  # ============================================================================
  # Private: Ensemble
  # ============================================================================
  
  defp ensemble_intent({tfidf_intent, tfidf_conf}, {lstm_intent, lstm_conf}) do
    cond do
      # Same prediction - boost confidence
      tfidf_intent == lstm_intent ->
        combined_conf = 1.0 - (1.0 - tfidf_conf) * (1.0 - lstm_conf)
        {tfidf_intent, combined_conf, :ensemble}
        
      # LSTM much more confident
      lstm_conf > tfidf_conf + 0.2 ->
        {lstm_intent, lstm_conf, :lstm}
        
      # TF-IDF much more confident
      tfidf_conf > lstm_conf + 0.2 ->
        {tfidf_intent, tfidf_conf, :tfidf}
        
      # Close confidence - prefer LSTM for contextual understanding
      lstm_conf > 0.5 ->
        {lstm_intent, lstm_conf, :lstm}
        
      # Fallback to TF-IDF
      true ->
        {tfidf_intent, tfidf_conf, :tfidf}
    end
  end
  
  # ============================================================================
  # Private: Fallbacks
  # ============================================================================
  
  defp fallback_analysis(text, _opts) do
    case get_tfidf_prediction(text) do
      {:ok, {intent, confidence}} ->
        entities = case Brain.ML.EntityExtractor.extract_entities(text) do
          entities when is_list(entities) -> entities
          _ -> []
        end
        
        {:ok, %{
          intent: {intent, confidence},
          entities: entities,
          sentiment: keyword_sentiment(text),
          speech_act: structural_speech_act(text),
          source: :tfidf
        }}
        
      error ->
        error
    end
  end
  
  defp keyword_sentiment(text) do
    lower = String.downcase(text)

    positive_words = Brain.LinguisticData.positive_words()
    negative_words = Brain.LinguisticData.negative_words()

    pos_count = Enum.count(positive_words, &String.contains?(lower, &1))
    neg_count = Enum.count(negative_words, &String.contains?(lower, &1))

    cond do
      pos_count > neg_count -> {:positive, 0.6 + 0.1 * pos_count}
      neg_count > pos_count -> {:negative, 0.6 + 0.1 * neg_count}
      true -> {:neutral, 0.8}
    end
  end
  
  defp structural_speech_act(text) do
    cond do
      String.ends_with?(text, "?") -> {:directive, 0.9}
      String.starts_with?(String.downcase(text), "please") -> {:directive, 0.8}
      String.starts_with?(String.downcase(text), "can you") -> {:directive, 0.85}
      String.starts_with?(String.downcase(text), "hi") -> {:expressive, 0.9}
      String.starts_with?(String.downcase(text), "hello") -> {:expressive, 0.9}
      String.starts_with?(String.downcase(text), "thanks") -> {:expressive, 0.9}
      true -> {:assertive, 0.7}
    end
  end
end
