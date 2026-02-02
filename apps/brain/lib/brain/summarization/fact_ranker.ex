defmodule Brain.Summarization.FactRanker do
  @moduledoc """
  Ranks extracted events by newsworthiness/importance.

  Scoring factors:
  - Event type (life events > plans > states > greetings)
  - Specificity (concrete details score higher)
  - Uniqueness (novel information scores higher)
  - Recency (later events may be more relevant)

  Filters out phatic/low-value content and returns ranked Facts.
  """

  alias Brain.Summarization.Types.{Event, Fact}

  # Importance weights by event type
  @type_weights %{
    announcement: 1.0,
    plan: 0.8,
    statement: 0.5,
    state: 0.4,
    question: 0.3,
    greeting: 0.1,
    farewell: 0.1,
    thanks: 0.1,
    apology: 0.1,
    other: 0.2
  }

  # Keywords that boost importance
  @high_value_keywords ~w(
    promoted married graduated engaged hired pregnant
    won bought sold moved died born
    meeting tonight tomorrow today
  )

  @medium_value_keywords ~w(
    will going need want plan visit
    love hate excited worried happy sad
  )

  @doc """
  Rank events and convert to Facts.

  Returns facts sorted by importance score (highest first).
  """
  @spec rank([Event.t()]) :: [Fact.t()]
  def rank(events) when is_list(events) do
    events
    |> Enum.map(&event_to_fact/1)
    |> Enum.reject(&should_skip?/1)
    |> Enum.sort_by(& &1.score, :desc)
  end

  @doc """
  Get top N facts by importance.
  """
  @spec top_facts([Event.t()], non_neg_integer()) :: [Fact.t()]
  def top_facts(events, n \\ 3) do
    events
    |> rank()
    |> Enum.take(n)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp event_to_fact(%Event{} = event) do
    score = calculate_score(event)
    importance = score_to_importance(score)

    Fact.new(
      event.subject || event.speaker,
      event.predicate,
      importance,
      object: event.object,
      participants: [event.speaker],
      tense: event.tense,
      score: score,
      event_type: event.subtype,
      template_key: subtype_to_template(event.subtype)
    )
  end

  defp calculate_score(%Event{} = event) do
    base_score = Map.get(@type_weights, event.subtype, 0.2)

    # Boost for high-value keywords
    keyword_boost = calculate_keyword_boost(event.predicate)

    # Boost for specificity (longer predicates often more specific)
    specificity_boost = calculate_specificity_boost(event.predicate)

    # Penalty for negation (negative statements often less summary-worthy)
    negation_penalty = if event.negated, do: -0.1, else: 0.0

    # Combine scores
    base_score + keyword_boost + specificity_boost + negation_penalty
  end

  defp calculate_keyword_boost(nil), do: 0.0
  defp calculate_keyword_boost(predicate) do
    predicate_lower = String.downcase(predicate)

    high_count = Enum.count(@high_value_keywords, &String.contains?(predicate_lower, &1))
    medium_count = Enum.count(@medium_value_keywords, &String.contains?(predicate_lower, &1))

    high_count * 0.3 + medium_count * 0.1
  end

  defp calculate_specificity_boost(nil), do: 0.0
  defp calculate_specificity_boost(predicate) do
    word_count = predicate |> String.split() |> length()

    # Sweet spot is 5-15 words
    cond do
      word_count < 3 -> 0.0
      word_count <= 15 -> 0.1
      true -> 0.05  # Too long may be rambling
    end
  end

  defp score_to_importance(score) do
    cond do
      score >= 0.8 -> :high
      score >= 0.5 -> :medium
      score >= 0.2 -> :low
      true -> :skip
    end
  end

  defp should_skip?(%Fact{importance: :skip}), do: true
  defp should_skip?(%Fact{predicate: nil}), do: true
  defp should_skip?(%Fact{predicate: ""}), do: true
  defp should_skip?(%Fact{event_type: event_type}) when event_type in [:greeting, :farewell, :thanks, :apology, :other], do: true
  defp should_skip?(%Fact{} = fact) do
    # Skip if predicate is too short or just punctuation
    cleaned = String.replace(fact.predicate, ~r/[^a-zA-Z\s]/, "")
    String.length(String.trim(cleaned)) < 5
  end

  defp subtype_to_template(subtype) do
    case subtype do
      :announcement -> :life_event
      :plan -> :plan
      :statement -> :statement
      :state -> :state
      :question -> :question
      _ -> :generic
    end
  end
end
