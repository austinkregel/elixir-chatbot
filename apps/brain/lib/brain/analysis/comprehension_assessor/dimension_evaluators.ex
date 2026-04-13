defmodule Brain.Analysis.ComprehensionAssessor.DimensionEvaluators do
  @moduledoc """
  Eight dimension evaluator functions for comprehension assessment.

  Each evaluator reads from existing ChunkAnalysis fields — no new extraction passes needed.
  Returns `{score :: float(), evidence :: map()}` for each dimension.
  """

  alias Brain.Analysis.ChunkAnalysis

  @doc """
  Evaluates all dimensions for a single ChunkAnalysis.

  Returns a map of `%{dimension => {score, evidence}}`.
  """
  def evaluate_all(%ChunkAnalysis{} = analysis) do
    %{
      referential_clarity: referential_clarity(analysis),
      actor_identification: actor_identification(analysis),
      propositional_content: propositional_content(analysis),
      temporal_grounding: temporal_grounding(analysis),
      contextual_sufficiency: contextual_sufficiency(analysis),
      epistemic_grounding: epistemic_grounding(analysis),
      structural_coherence: structural_coherence(analysis),
      illocutionary_clarity: illocutionary_clarity(analysis)
    }
  end

  @doc """
  "Do I know WHAT this text is about?"

  Scores based on entity count and event object presence.
  """
  def referential_clarity(%ChunkAnalysis{} = analysis) do
    entities = analysis.entities || []
    events = analysis.events || []

    entity_count = length(entities)
    has_high_conf_entity = Enum.any?(entities, fn e -> Map.get(e, :confidence, 0) >= 0.6 end)
    objects_identified = Enum.count(events, fn e -> e.object != nil end)

    score =
      cond do
        entity_count >= 2 and has_high_conf_entity -> 1.0
        entity_count >= 1 and has_high_conf_entity -> 0.8
        entity_count >= 1 -> 0.6
        objects_identified >= 1 -> 0.4
        true -> 0.1
      end

    evidence = %{
      entity_count: entity_count,
      has_high_confidence_entity: has_high_conf_entity,
      objects_identified: objects_identified
    }

    {score, evidence}
  end

  @doc """
  "Do I know WHO is involved?"

  Scores based on discourse addressee resolution and event actor presence.
  """
  def actor_identification(%ChunkAnalysis{} = analysis) do
    discourse = analysis.discourse
    events = analysis.events || []

    addressee_known =
      case discourse do
        %{addressee: addr, confidence: conf} when addr != :unknown and addr != :ambiguous ->
          conf

        _ ->
          0.0
      end

    actors_identified = Enum.count(events, fn e -> e.actor != nil end)
    total_events = length(events)

    actor_ratio =
      if total_events > 0, do: actors_identified / total_events, else: 0.0

    score =
      cond do
        addressee_known >= 0.7 and actor_ratio >= 0.5 -> 1.0
        addressee_known >= 0.5 or actor_ratio >= 0.5 -> 0.7
        addressee_known > 0.0 or actors_identified > 0 -> 0.4
        true -> 0.1
      end

    evidence = %{
      addressee_confidence: addressee_known,
      actors_identified: actors_identified,
      total_events: total_events
    }

    {score, evidence}
  end

  @doc """
  "Can I state WHAT is being claimed?"

  Scores based on assertive speech act presence and event action+object completeness.
  """
  def propositional_content(%ChunkAnalysis{} = analysis) do
    speech_act = analysis.speech_act
    events = analysis.events || []

    is_assertive =
      case speech_act do
        %{category: :assertive} -> true
        _ -> false
      end

    complete_events = Enum.count(events, fn e -> e.action != nil and e.object != nil end)
    has_action = Enum.any?(events, fn e -> e.action != nil end)

    score =
      cond do
        is_assertive and complete_events >= 1 -> 1.0
        is_assertive and has_action -> 0.7
        complete_events >= 1 -> 0.6
        has_action -> 0.4
        is_assertive -> 0.3
        true -> 0.1
      end

    evidence = %{
      is_assertive: is_assertive,
      complete_events: complete_events,
      has_action: has_action
    }

    {score, evidence}
  end

  @doc """
  "Do I know WHEN this applies?"

  Scores based on temporal information in events (tense, temporal modifiers, temporal entities).
  """
  def temporal_grounding(%ChunkAnalysis{} = analysis) do
    events = analysis.events || []
    entities = analysis.entities || []

    has_explicit_tense =
      Enum.any?(events, fn e ->
        case e.action do
          %{tense: tense} when tense in [:present, :past, :future] -> true
          _ -> false
        end
      end)

    temporal_modifiers =
      events
      |> Enum.flat_map(fn e -> Map.get(e, :modifiers, []) end)
      |> Enum.count(fn m -> Map.get(m, :type) == :temporal end)

    temporal_entities =
      Enum.count(entities, fn e ->
        entity_type = Map.get(e, :entity_type) || Map.get(e, "type") || ""
        entity_type in ["date", "time", "duration", "temporal", "DATE", "TIME"]
      end)

    score =
      cond do
        temporal_entities >= 1 and has_explicit_tense -> 1.0
        temporal_entities >= 1 or temporal_modifiers >= 1 -> 0.7
        has_explicit_tense -> 0.5
        true -> 0.1
      end

    evidence = %{
      has_explicit_tense: has_explicit_tense,
      temporal_modifiers: temporal_modifiers,
      temporal_entities: temporal_entities
    }

    {score, evidence}
  end

  @doc """
  "Do I have enough CONTEXT?"

  Scores based on missing required slots and context resolution.
  """
  def contextual_sufficiency(%ChunkAnalysis{} = analysis) do
    missing_context = analysis.missing_context || []
    slots = analysis.slots

    missing_required =
      case slots do
        %{missing_required: mr} when is_list(mr) -> length(mr)
        _ -> 0
      end

    all_required_filled =
      case slots do
        %{all_required_filled: true} -> true
        _ -> missing_required == 0
      end

    missing_count = length(missing_context) + missing_required

    score =
      cond do
        all_required_filled and missing_count == 0 -> 1.0
        missing_count <= 1 -> 0.7
        missing_count <= 2 -> 0.5
        missing_count <= 3 -> 0.3
        true -> 0.1
      end

    evidence = %{
      missing_context_count: length(missing_context),
      missing_required_slots: missing_required,
      all_required_filled: all_required_filled
    }

    {score, evidence}
  end

  @doc """
  "Does this relate to things I already KNOW?"

  Scores based on fact verification status and related beliefs.
  """
  def epistemic_grounding(%ChunkAnalysis{} = analysis) do
    fact_verification = analysis.fact_verification
    related_beliefs = analysis.related_beliefs || []

    verification_score =
      case fact_verification do
        {:verified, confidence} when is_number(confidence) -> confidence
        {:contradicted, _} -> 0.3
        {:uncertain, _} -> 0.2
        _ -> 0.0
      end

    belief_count = length(related_beliefs)

    score =
      cond do
        verification_score >= 0.7 and belief_count >= 1 -> 1.0
        verification_score >= 0.5 or belief_count >= 2 -> 0.7
        verification_score > 0.0 or belief_count >= 1 -> 0.5
        true -> 0.1
      end

    evidence = %{
      fact_verification: fact_verification,
      related_belief_count: belief_count,
      verification_score: verification_score
    }

    {score, evidence}
  end

  @doc """
  "Does this MAKE SENSE as language?"

  Hard gate: score < 0.2 results in :garbled verdict regardless of other dimensions.
  Scores based on speech act confidence and text quality heuristics.
  """
  def structural_coherence(%ChunkAnalysis{} = analysis) do
    speech_act = analysis.speech_act
    text = analysis.text || ""

    speech_act_confidence =
      case speech_act do
        %{confidence: conf} when is_number(conf) -> conf
        _ -> 0.0
      end

    # Text quality checks (using basic properties, not regex)
    text_length = String.length(text)
    word_count = text |> String.split() |> length()
    has_reasonable_length = text_length >= 5 and text_length <= 10_000
    has_words = word_count >= 2

    # Check for very high non-alpha ratio (HTML/code/binary)
    graphemes = String.graphemes(text)
    total_graphemes = length(graphemes)

    alpha_count =
      if total_graphemes > 0 do
        Enum.count(graphemes, fn g ->
          byte_size(g) == 1 and is_alpha_or_space?(g)
        end)
      else
        0
      end

    alpha_ratio = if total_graphemes > 0, do: alpha_count / total_graphemes, else: 0.0

    lexical_coverage = compute_lexical_coverage(text)

    score =
      cond do
        not has_reasonable_length or not has_words -> 0.1
        alpha_ratio < 0.3 -> 0.1
        speech_act_confidence < 0.2 and alpha_ratio < 0.7 -> 0.15
        speech_act_confidence >= 0.7 and alpha_ratio >= 0.6 and lexical_coverage >= 0.5 -> 1.0
        speech_act_confidence >= 0.5 and alpha_ratio >= 0.5 -> 0.7
        lexical_coverage >= 0.8 and alpha_ratio >= 0.5 -> 0.65
        speech_act_confidence >= 0.3 -> 0.5
        alpha_ratio >= 0.5 -> 0.3
        true -> 0.15
      end

    evidence = %{
      speech_act_confidence: speech_act_confidence,
      text_length: text_length,
      word_count: word_count,
      alpha_ratio: Float.round(alpha_ratio, 3),
      lexical_coverage: Float.round(lexical_coverage, 3)
    }

    {score, evidence}
  end

  defp compute_lexical_coverage(text) do
    if Process.whereis(Brain.ML.Lexicon) do
      words =
        text
        |> Brain.ML.Tokenizer.tokenize_normalized(min_length: 2)
        |> Enum.reject(&(String.length(&1) < 2))

      if words == [] do
        0.0
      else
        known = Enum.count(words, &Brain.ML.Lexicon.known_word?/1)
        known / length(words)
      end
    else
      0.5
    end
  end

  @doc """
  "Do I know what KIND of communication this is?"

  Scores based on speech act category classification and intent confidence.
  """
  def illocutionary_clarity(%ChunkAnalysis{} = analysis) do
    speech_act = analysis.speech_act
    intent = analysis.intent

    category_known =
      case speech_act do
        %{category: cat} when cat != :unknown -> true
        _ -> false
      end

    speech_confidence =
      case speech_act do
        %{confidence: conf} when is_number(conf) -> conf
        _ -> 0.0
      end

    has_intent = intent != nil and intent != ""

    score =
      cond do
        category_known and speech_confidence >= 0.7 and has_intent -> 1.0
        category_known and speech_confidence >= 0.5 -> 0.8
        category_known -> 0.6
        has_intent -> 0.4
        true -> 0.1
      end

    evidence = %{
      category_known: category_known,
      speech_confidence: speech_confidence,
      has_intent: has_intent
    }

    {score, evidence}
  end

  defp is_alpha_or_space?(<<c>>) when (c >= ?a and c <= ?z) or (c >= ?A and c <= ?Z) or c == ?\s,
    do: true

  defp is_alpha_or_space?(_), do: false
end
