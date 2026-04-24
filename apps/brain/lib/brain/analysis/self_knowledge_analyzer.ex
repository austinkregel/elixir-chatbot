defmodule Brain.Analysis.SelfKnowledgeAnalyzer do
  @moduledoc "Analyzes meta-cognitive queries about what the system knows.\n\nThis analyzer uses the trained intent classifier to detect queries like:\n- \"What do you know about me?\" (meta.self_knowledge)\n- \"Do you remember me?\" (meta.memory_check)\n- \"Are you tracking me?\" (meta.privacy_probe)\n\nTraining data for these intents lives in:\n- data/intents/meta.self_knowledge_usersays_en.json\n- data/intents/meta.memory_check_usersays_en.json\n- data/intents/meta.privacy_probe_usersays_en.json\n\nFor such queries, it builds a SelfKnowledgeAssessment that categorizes\nthe system's knowledge into:\n- Discloseable (safe to share confidently)\n- Inferred but uncertain (share with hedging)\n- Should avoid (too personal, uncertain, or inappropriate)\n"

  alias Brain.Analysis.{AnalyzerResult, ChunkProfile}
  alias Brain.Epistemic.Types.{SelfKnowledgeAssessment, Config}
  alias Brain.Epistemic.UserModelStore

  alias Brain.ML.Tokenizer
  require Logger
  @min_confidence 0.5

  @doc "Analyzes text to detect meta-cognitive queries.\n\nReturns an AnalyzerResult if a meta-cognitive query is detected,\notherwise returns a low-confidence result.\n"
  def analyze(text, opts \\ []) do
    user_id = Keyword.get(opts, :user_id)

    case detect_meta_intent(text) do
      {:ok, intent, confidence} when confidence >= @min_confidence ->
        query_type = intent_to_query_type(intent)

        assessment =
          if user_id && Config.enabled?() do
            build_self_knowledge_assessment(user_id)
          else
            SelfKnowledgeAssessment.new(user_id || "unknown")
          end

        AnalyzerResult.new(:self_knowledge, intent, confidence,
          confidence_estimate: confidence,
          indicators: ["meta_cognitive_query", to_string(query_type)],
          metadata: %{
            query_type: query_type,
            assessment: assessment,
            user_id: user_id
          }
        )

      _ ->
        AnalyzerResult.new(:self_knowledge, nil, 0.0)
    end
  end

  @doc "Checks if the text contains a meta-cognitive query.\nUses the intent classifier trained on meta.* intents.\n"
  def is_self_knowledge_query?(text) do
    case detect_meta_intent(text) do
      {:ok, _intent, confidence} when confidence >= @min_confidence -> true
      _ -> false
    end
  end

  @doc "Builds a SelfKnowledgeAssessment for the given user.\n"
  def build_self_knowledge_assessment(nil) do
    SelfKnowledgeAssessment.new("unknown")
  end

  def build_self_knowledge_assessment(user_id) do
    if Process.whereis(UserModelStore) == nil do
      SelfKnowledgeAssessment.new(user_id || "unknown")
    else
      config = Config.get()

      case UserModelStore.get(user_id) do
        nil ->
          SelfKnowledgeAssessment.new(user_id)

        model ->
          SelfKnowledgeAssessment.from_user_model(model,
            high_confidence: config.high_confidence_threshold,
            low_confidence: config.low_confidence_threshold,
            sensitive_keys: sensitive_keys()
          )
      end
    end
  end

  @doc "Detects if the text matches a meta-cognitive intent.\n\nUses the intent classifier and filters for meta.* intents.\nFalls back to keyword-based detection when classifier is unavailable.\nReturns {:ok, intent, confidence} or :no_match\n"
  def detect_meta_intent(text) do
    case classify_meta(text) do
      {:ok, %{intent: intent, confidence: confidence}} when is_binary(intent) ->
        if is_meta_intent?(intent) do
          {:ok, intent, confidence}
        else
          keyword_fallback_detection(text)
        end

      _ ->
        keyword_fallback_detection(text)
    end
  rescue
    _ ->
      keyword_fallback_detection(text)
  end

  defp keyword_fallback_detection(text) do
    tokens = Tokenizer.tokenize_normalized(text)
    token_set = MapSet.new(tokens)

    cond do
      is_self_knowledge_pattern?(tokens, token_set) ->
        {:ok, "meta.self_knowledge", 0.75}

      is_memory_check_pattern?(tokens, token_set) ->
        {:ok, "meta.memory_check", 0.75}

      is_privacy_probe_pattern?(tokens, token_set) ->
        {:ok, "meta.privacy_probe", 0.75}

      true ->
        :no_match
    end
  end

  defp is_self_knowledge_pattern?(_tokens, token_set) do
    has_know = MapSet.member?(token_set, "know") or MapSet.member?(token_set, "learned")
    has_about_me = MapSet.member?(token_set, "about") and MapSet.member?(token_set, "me")
    has_you = MapSet.member?(token_set, "you")
    has_question = MapSet.member?(token_set, "what") or MapSet.member?(token_set, "how")

    (has_know and has_about_me and has_you) or
      (has_question and has_know and MapSet.member?(token_set, "me"))
  end

  defp is_memory_check_pattern?(_tokens, token_set) do
    has_remember = MapSet.member?(token_set, "remember") or MapSet.member?(token_set, "recall")
    has_you = MapSet.member?(token_set, "you")
    has_me = MapSet.member?(token_set, "me") or MapSet.member?(token_set, "anything")

    has_remember and has_you and has_me
  end

  defp is_privacy_probe_pattern?(_tokens, token_set) do
    tracking_words = ~w(tracking watching monitoring spying collecting)
    has_tracking = Enum.any?(tracking_words, &MapSet.member?(token_set, &1))
    has_you = MapSet.member?(token_set, "you")
    has_me = MapSet.member?(token_set, "me")

    has_tracking and has_you and has_me
  end

  @doc "Determines the type of meta-cognitive query from the intent.\n"
  def detect_query_type(text) do
    case detect_meta_intent(text) do
      {:ok, intent, confidence} ->
        {:ok, intent_to_query_type(intent), confidence}

      :no_match ->
        :no_match
    end
  end

  @doc "Returns the list of meta-cognitive intents.\n\nWith profiles, returns an empty list as graceful degradation since\nthe registry is no longer the source of truth.\n"
  def meta_intents do
    []
  end

  defp classify_meta(text) do
    alias Brain.Analysis.{FeatureExtractor, Pipeline}
    alias Brain.ML.MicroClassifiers

    if MicroClassifiers.ready?() do
      analysis = Pipeline.analyze_chunk(text)
      {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)

      case MicroClassifiers.classify_vector(:intent_full, feature_vector) do
        {:ok, intent, confidence} ->
          {:ok, %{intent: intent, confidence: confidence}}

        _ ->
          {:error, :classification_failed}
      end
    else
      {:error, :no_classifier}
    end
  rescue
    _ -> {:error, :classification_failed}
  end

  defp is_meta_intent?(intent, profile \\ nil) do
    case profile do
      %ChunkProfile{domain: :meta} -> true
      _ -> String.starts_with?(to_string(intent || ""), "meta.")
    end
  end

  defp intent_to_query_type(intent, profile \\ nil) do
    case profile do
      %ChunkProfile{target: target, speech_act_category: cat, modality: modality} when target != :ambiguous ->
        derive_query_type_from_profile(target, cat, modality)

      _ ->
        cond do
            String.contains?(to_string(intent), "privacy") -> :privacy_probe
            String.contains?(to_string(intent), "memory") -> :memory_check
            true -> :self_query
          end
    end
  end

  defp derive_query_type_from_profile(target, category, modality) do
    cond do
      target == :self and category == :directive -> :self_query
      target == :agent and modality == :interrogative -> :capability_query
      category == :directive and modality == :interrogative -> :self_query
      true -> :self_query
    end
  end

  defp sensitive_keys do
    [
      :password,
      :ssn,
      :social_security,
      :credit_card,
      :bank_account,
      :medical,
      :health,
      :salary,
      :income,
      :political,
      :religion,
      :sexual_orientation
    ]
  end
end