defmodule Brain.ML.LSTM.Integration do
  @moduledoc "Integration layer for using LSTM models alongside existing TF-IDF classifiers.\n\nThis module is called from the main analysis pipeline as an **ensemble\nfallback**.  When `Brain.Analysis.SpeechActClassifier` obtains a low-confidence\nresult from the LSTM `MultiTaskModel` (below the ensemble threshold), it\ndelegates to `Integration.classify_intent/1` which combines TF-IDF and LSTM\npredictions via ensemble voting to produce a more robust classification.\n\nThe pipeline still invokes LSTM models directly via\n`Brain.ML.LSTM.MultiTaskModel` and `Brain.ML.LSTM.UnifiedModel` for\nhigh-confidence predictions; this module is only consulted when the primary\nLSTM confidence is insufficient.\n\n## Provides\n\n1. Get hybrid predictions combining TF-IDF and LSTM\n2. Fallback gracefully when LSTM is not available\n3. Ensemble voting for improved accuracy\n\n## Hybrid Approach\n\nThe system uses both TF-IDF (fast, interpretable) and LSTM (accurate, contextual):\n\n    User Input\n        │\n        ├──> TF-IDF Classifier (fast, always available)\n        │         │\n        │         ▼\n        │    {intent, confidence}\n        │\n        └──> LSTM Classifier (if available)\n                  │\n                  ▼\n             {intent, confidence}\n\n        │         │\n        ▼         ▼\n     Ensemble Combiner\n             │\n             ▼\n     Final Prediction\n\n## Usage\n\n    # Get best prediction using all available models\n    Integration.classify_intent(\"What's the weather?\")\n    # => {:ok, {\"weather.query\", 0.87, :ensemble}}\n\n    # Get full analysis with all models\n    Integration.analyze(\"Play some music\")\n    # => {:ok, %{\n    #   intent: {\"music.play\", 0.92},\n    #   entities: [...],\n    #   source: :lstm\n    # }}\n"

  alias Brain.LinguisticData
  alias Brain.ML.EntityExtractor
  alias Brain.ML.LSTM
  require Logger

  alias Brain.ML.IntentClassifierSimple
  alias LSTM.{AxonTrainer, UnifiedModel}

  @doc "Classify intent using the best available method.\n\nReturns `{:ok, {intent, confidence, source}}` where source is:\n- `:lstm` - LSTM model prediction\n- `:tfidf` - TF-IDF model prediction\n- `:ensemble` - Combined prediction from both\n"
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

  @doc "Get full NLP analysis using LSTM if available, falling back to TF-IDF.\n"
  def analyze(text, opts \\ []) do
    case UnifiedModel.ready?() && UnifiedModel.analyze(text) do
      {:ok, result} ->
        {:ok, Map.put(result, :source, :lstm)}

      _ ->
        fallback_analysis(text, opts)
    end
  end

  @doc "Get sentiment using LSTM if available, falling back to keyword heuristics.\n"
  def classify_sentiment(text) do
    case UnifiedModel.ready?() && UnifiedModel.classify_sentiment(text) do
      {:ok, result} ->
        {:ok, result}

      _ ->
        {:ok, keyword_sentiment(text)}
    end
  end

  @doc "Get speech act using LSTM if available, falling back to structural analysis.\n"
  def classify_speech_act(text) do
    case UnifiedModel.ready?() && UnifiedModel.classify_speech_act(text) do
      {:ok, result} ->
        {:ok, result}

      _ ->
        {:ok, structural_speech_act(text)}
    end
  end

  @doc "Extract entities using LSTM NER if available, falling back to gazetteer.\n"
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

  @doc "Check if LSTM models are available and ready.\n"
  def lstm_available? do
    UnifiedModel.ready?()
  end

  @doc "Get status of all available models.\n"
  def model_status do
    %{
      tfidf: IntentClassifierSimple.ready?(),
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

        {:ok,
         %{
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

    positive_words = LinguisticData.positive_words()
    negative_words = LinguisticData.negative_words()

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