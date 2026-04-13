defmodule Brain.Analysis.ContextAccumulator do
  @moduledoc """
  Accumulates and fuses heterogeneous context signals using
  log-odds evidence combination and Dempster-Shafer conflict detection.

  Constructed fresh per-turn. Queries the existing Memory system
  (Memory.Store) for relevant episodic and semantic context rather
  than maintaining its own scope. Discarded after response generation.
  """

  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.Graph.Reader

  require Logger

  @max_confidence 0.999
  @min_confidence 0.001

  defstruct signals: [],
            combined_confidence: 0.5,
            conflict_measure: 0.0,
            dominant_signal: nil,
            entity_familiarity: 0.5,
            relevant_episodes: [],
            relevant_semantics: [],
            conversation_topics: [],
            interlocutor_adaptations: %{}

  @doc """
  Creates a fresh accumulator for a single turn.

  Queries the existing Memory system for relevant context and the
  graph system for conversation topics and interlocutor preferences.

  ## Options
    - `:user_id` - interlocutor ID for practical adaptations
    - `:conversation_id` - conversation for topic continuity
    - `:world_id` - world scope for memory queries (default: "default")
    - `:memory_k` - number of similar memories to retrieve (default: 5)
  """
  def new(text, opts \\ []) when is_binary(text) do
    user_id = Keyword.get(opts, :user_id)
    conversation_id = Keyword.get(opts, :conversation_id)
    world_id = Keyword.get(opts, :world_id, "default")
    memory_k = Keyword.get(opts, :memory_k, 5)

    %__MODULE__{
      relevant_episodes: query_episodes(text, memory_k, world_id),
      relevant_semantics: query_semantics(text, memory_k, world_id),
      conversation_topics: fetch_conversation_topics(conversation_id),
      interlocutor_adaptations: fetch_interlocutor_adaptations(user_id)
    }
  end

  @doc """
  Adds a signal to the accumulator.

  Each signal is a tuple of {source, value, confidence} where:
  - source: atom identifying the signal origin (e.g. :discourse, :speech_act)
  - value: the signal's categorical or structured value (any term)
  - confidence: float 0.0-1.0 representing certainty
  """
  def add_signal(%__MODULE__{} = acc, source, value, confidence) when is_atom(source) do
    confidence = clamp(confidence, @min_confidence, @max_confidence)
    %{acc | signals: [{source, value, confidence} | acc.signals]}
  end

  @doc """
  Computes the accumulated context by fusing all signals.

  Uses log-odds evidence accumulation for combining confidences,
  entropy-weighted relevance for signal importance, and
  Dempster-Shafer K for conflict detection.
  """
  def accumulate(%__MODULE__{signals: []} = acc), do: acc

  def accumulate(%__MODULE__{signals: signals} = acc) do
    weighted = compute_relevance_weights(signals)

    %{acc |
      combined_confidence: log_odds_combine(weighted),
      conflict_measure: compute_conflict(signals),
      dominant_signal: find_dominant(weighted),
      entity_familiarity: extract_entity_familiarity(signals)
    }
  end

  @doc """
  Returns the effective confidence after accounting for conflict.

  High conflict pulls confidence toward 0.5 (maximum uncertainty),
  preventing overconfident output when signals disagree.
  """
  def effective_confidence(%__MODULE__{combined_confidence: cc, conflict_measure: k}) do
    cc * (1.0 - k) + 0.5 * k
  end

  @doc """
  Returns true when the system should hedge its response.

  Hedging is appropriate when conflict is high, entity familiarity
  is low, or effective confidence is too weak.
  """
  def should_hedge?(%__MODULE__{} = acc) do
    acc.conflict_measure > 0.3 or
      acc.entity_familiarity < 0.3 or
      effective_confidence(acc) < 0.4
  end

  @doc """
  Returns the source atom of the strongest signal after accumulation.
  """
  def dominant_strategy(%__MODULE__{dominant_signal: nil}), do: :unknown
  def dominant_strategy(%__MODULE__{dominant_signal: {source, _value, _conf}}), do: source

  @doc """
  Retrieves a practical adaptation for the interlocutor.

  Only returns practical preferences (units, formats), never personality data.
  """
  def interlocutor_adaptation(%__MODULE__{interlocutor_adaptations: adaptations}, key) do
    Map.get(adaptations, key)
  end

  @doc """
  Returns memory context for response generation.

  Provides relevant episodes and semantic facts that the response
  generator can reference.
  """
  def memory_context(%__MODULE__{} = acc) do
    %{
      episodes: acc.relevant_episodes,
      semantics: acc.relevant_semantics,
      conversation_topics: acc.conversation_topics
    }
  end

  # ===========================================================================
  # Log-Odds Evidence Combination
  # ===========================================================================

  @doc false
  def log_odds_combine(weighted_signals) do
    llr_combined =
      Enum.reduce(weighted_signals, 0.0, fn {_source, _value, confidence, relevance}, sum ->
        clamped = clamp(confidence, @min_confidence, @max_confidence)
        llr = :math.log(clamped / (1.0 - clamped))
        sum + relevance * llr
      end)

    sigmoid(llr_combined)
  end

  # ===========================================================================
  # Dempster-Shafer Conflict Detection
  # ===========================================================================

  @doc false
  def compute_conflict(signals) when length(signals) < 2, do: 0.0

  def compute_conflict(signals) do
    pairs =
      for {s1, _v1, c1} <- signals,
          {s2, _v2, c2} <- signals,
          s1 != s2,
          do: {c1, c2}

    if pairs == [] do
      0.0
    else
      total_conflict =
        Enum.reduce(pairs, 0.0, fn {c1, c2}, sum ->
          sum + c1 * (1.0 - c2) + (1.0 - c1) * c2
        end)

      clamp(total_conflict / max(length(pairs), 1), 0.0, 1.0)
    end
  end

  # ===========================================================================
  # Entropy-Weighted Signal Relevance
  # ===========================================================================

  @doc false
  def compute_relevance_weights(signals) do
    Enum.map(signals, fn {source, value, confidence} ->
      clamped = clamp(confidence, @min_confidence, @max_confidence)
      entropy = binary_entropy(clamped)
      gate = :math.sqrt(max(1.0 - entropy, 0.0))
      {source, value, confidence, gate}
    end)
  end

  # ===========================================================================
  # Private Helpers
  # ===========================================================================

  defp binary_entropy(p) do
    q = 1.0 - clamp(p, @min_confidence, @max_confidence)
    p_c = clamp(p, @min_confidence, @max_confidence)
    -(p_c * :math.log2(p_c) + q * :math.log2(q))
  end

  defp sigmoid(x), do: 1.0 / (1.0 + :math.exp(-x))

  defp clamp(value, lo, hi), do: value |> max(lo) |> min(hi)

  defp find_dominant([]), do: nil

  defp find_dominant(weighted) do
    {source, value, confidence, _relevance} =
      Enum.max_by(weighted, fn {_s, _v, c, r} -> c * r end)

    {source, value, confidence}
  end

  defp extract_entity_familiarity(signals) do
    case Enum.find(signals, fn {source, _, _} -> source == :entity_familiarity end) do
      {_, _, confidence} -> confidence
      nil -> 0.5
    end
  end

  defp query_episodes(text, k, world_id) do
    if MemoryStore.ready?() do
      case MemoryStore.query_similar(text, k, world_id: world_id) do
        {:ok, results} -> results
        _ -> []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  defp query_semantics(text, k, world_id) do
    if MemoryStore.ready?() do
      case MemoryStore.query_semantic(text, k, world_id: world_id) do
        {:ok, results} -> results
        _ -> []
      end
    else
      []
    end
  rescue
    _ -> []
  end

  defp fetch_conversation_topics(nil), do: []

  defp fetch_conversation_topics(conversation_id) do
    Reader.conversation_topics(to_string(conversation_id))
  rescue
    _ -> []
  end

  defp fetch_interlocutor_adaptations(nil), do: %{}

  defp fetch_interlocutor_adaptations(user_id) do
    user_id
    |> Reader.user_preferences()
    |> Enum.reduce(%{}, fn pref, acc ->
      case pref do
        %{rel_type: "WANTS", topic: topic, properties: props} ->
          extract_practical_adaptation(acc, topic, props)

        %{rel_type: "NEEDS", topic: topic, properties: props} ->
          extract_practical_adaptation(acc, topic, props)

        _ ->
          acc
      end
    end)
  rescue
    _ -> %{}
  end

  @practical_topics MapSet.new([
    "fahrenheit", "celsius", "metric", "imperial",
    "24h", "12h", "date_format", "timezone"
  ])

  defp extract_practical_adaptation(acc, topic, props) do
    normalized = String.downcase(topic)

    if MapSet.member?(@practical_topics, normalized) do
      Map.put(acc, String.to_atom(normalized), Map.get(props, "value", topic))
    else
      acc
    end
  end
end
