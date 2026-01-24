defmodule ChatBot.Analysis.SelfKnowledgeAnalyzer do
  @moduledoc """
  Analyzes meta-cognitive queries about what the system knows.

  This analyzer uses the trained intent classifier to detect queries like:
  - "What do you know about me?" (meta.self_knowledge)
  - "Do you remember me?" (meta.memory_check)
  - "Are you tracking me?" (meta.privacy_probe)

  Training data for these intents lives in:
  - data/intents/meta.self_knowledge_usersays_en.json
  - data/intents/meta.memory_check_usersays_en.json
  - data/intents/meta.privacy_probe_usersays_en.json

  For such queries, it builds a SelfKnowledgeAssessment that categorizes
  the system's knowledge into:
  - Discloseable (safe to share confidently)
  - Inferred but uncertain (share with hedging)
  - Should avoid (too personal, uncertain, or inappropriate)
  """

  alias ChatBot.Analysis.AnalyzerResult
  alias ChatBot.Epistemic.Types.{SelfKnowledgeAssessment, Config}
  alias ChatBot.Epistemic.UserModelStore
  alias ChatBot.ML.IntentClassifierSimple

  require Logger

  # Meta-cognitive intent prefixes we recognize
  @meta_intent_prefixes [
    "meta.self_knowledge",
    "meta.memory_check",
    "meta.privacy_probe",
    "meta.trust_check"
  ]

  # Minimum confidence to consider a meta-cognitive intent
  @min_confidence 0.5

  @doc """
  Analyzes text to detect meta-cognitive queries.

  Returns an AnalyzerResult if a meta-cognitive query is detected,
  otherwise returns a low-confidence result.
  """
  def analyze(text, opts \\ []) do
    user_id = Keyword.get(opts, :user_id)

    case detect_meta_intent(text) do
      {:ok, intent, confidence} when confidence >= @min_confidence ->
        query_type = intent_to_query_type(intent)

        # Build assessment if we have a user_id
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

  @doc """
  Checks if the text contains a meta-cognitive query.
  Uses the intent classifier trained on meta.* intents.
  """
  def is_self_knowledge_query?(text) do
    case detect_meta_intent(text) do
      {:ok, _intent, confidence} when confidence >= @min_confidence -> true
      _ -> false
    end
  end

  @doc """
  Builds a SelfKnowledgeAssessment for the given user.
  """
  def build_self_knowledge_assessment(nil) do
    # No user_id - return empty assessment
    SelfKnowledgeAssessment.new("unknown")
  end

  def build_self_knowledge_assessment(user_id) do
    # Check if UserModelStore is available
    if Process.whereis(UserModelStore) == nil do
      SelfKnowledgeAssessment.new(user_id || "unknown")
    else
      config = Config.get()

      case UserModelStore.get(user_id) do
        nil ->
          # No user model - return empty assessment
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

  @doc """
  Detects if the text matches a meta-cognitive intent.

  Uses the intent classifier and filters for meta.* intents.
  Falls back to keyword-based detection when classifier is unavailable.
  Returns {:ok, intent, confidence} or :no_match
  """
  def detect_meta_intent(text) do
    # Try the classifier first
    case IntentClassifierSimple.classify(text) do
      {:ok, %{intent: intent, confidence: confidence}} when is_binary(intent) ->
        if is_meta_intent?(intent) do
          {:ok, intent, confidence}
        else
          # Classifier returned non-meta intent, try keyword fallback
          keyword_fallback_detection(text)
        end

      _ ->
        # Classifier unavailable, use keyword fallback
        keyword_fallback_detection(text)
    end
  rescue
    _ ->
      # Error in classifier, use keyword fallback
      keyword_fallback_detection(text)
  end

  # Keyword-based fallback detection using tokenizer (no regex)
  # Used when the classifier isn't trained or available
  defp keyword_fallback_detection(text) do
    tokens = ChatBot.ML.Tokenizer.tokenize_normalized(text)
    token_set = MapSet.new(tokens)

    cond do
      # Self-knowledge patterns
      is_self_knowledge_pattern?(tokens, token_set) ->
        {:ok, "meta.self_knowledge", 0.75}

      # Memory check patterns
      is_memory_check_pattern?(tokens, token_set) ->
        {:ok, "meta.memory_check", 0.75}

      # Privacy probe patterns
      is_privacy_probe_pattern?(tokens, token_set) ->
        {:ok, "meta.privacy_probe", 0.75}

      true ->
        :no_match
    end
  end

  defp is_self_knowledge_pattern?(_tokens, token_set) do
    # "what do you know about me"
    has_know = MapSet.member?(token_set, "know") or MapSet.member?(token_set, "learned")
    has_about_me = MapSet.member?(token_set, "about") and MapSet.member?(token_set, "me")
    has_you = MapSet.member?(token_set, "you")
    has_question = MapSet.member?(token_set, "what") or MapSet.member?(token_set, "how")

    (has_know and has_about_me and has_you) or
      (has_question and has_know and MapSet.member?(token_set, "me"))
  end

  defp is_memory_check_pattern?(_tokens, token_set) do
    # "do you remember me"
    has_remember = MapSet.member?(token_set, "remember") or MapSet.member?(token_set, "recall")
    has_you = MapSet.member?(token_set, "you")
    has_me = MapSet.member?(token_set, "me") or MapSet.member?(token_set, "anything")

    has_remember and has_you and has_me
  end

  defp is_privacy_probe_pattern?(_tokens, token_set) do
    # "are you tracking me"
    tracking_words = ~w(tracking watching monitoring spying collecting)
    has_tracking = Enum.any?(tracking_words, &MapSet.member?(token_set, &1))
    has_you = MapSet.member?(token_set, "you")
    has_me = MapSet.member?(token_set, "me")

    has_tracking and has_you and has_me
  end

  @doc """
  Determines the type of meta-cognitive query from the intent.
  """
  def detect_query_type(text) do
    case detect_meta_intent(text) do
      {:ok, intent, confidence} ->
        {:ok, intent_to_query_type(intent), confidence}

      :no_match ->
        :no_match
    end
  end

  @doc """
  Returns the list of meta-cognitive intents.
  """
  def meta_intents, do: @meta_intent_prefixes

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp is_meta_intent?(intent) when is_binary(intent) do
    Enum.any?(@meta_intent_prefixes, &String.starts_with?(intent, &1))
  end

  defp is_meta_intent?(_), do: false

  defp intent_to_query_type(intent) when is_binary(intent) do
    cond do
      String.starts_with?(intent, "meta.self_knowledge") -> :self_query
      String.starts_with?(intent, "meta.memory_check") -> :memory_check
      String.starts_with?(intent, "meta.privacy_probe") -> :privacy_probe
      String.starts_with?(intent, "meta.trust_check") -> :trust_check
      true -> :self_query
    end
  end

  defp intent_to_query_type(_), do: :self_query

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
