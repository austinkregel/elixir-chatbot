defmodule Brain.Response.ContextBuilder do
  @moduledoc """
  Assembles a single unified context map from all available sources.

  Merges analysis pipeline signals, graph database context, enrichment
  data, and memory signals into one rich map that flows through the
  entire synthesis pipeline (RefinementLoop -> ContentSpecifier ->
  SurfaceRealizer -> OuroRealizer -> RealizationPacket -> ConstraintEnforcer).
  """

  alias Brain.Analysis.{InternalModel, ChunkAnalysis, ContextAccumulator, EntityGraphEnricher}
  alias Brain.Graph.Reader
  alias Brain.Response.Enricher

  require Logger

  @doc """
  Builds a unified context map from the analysis model and opts.

  Pulls together:
  - All ChunkAnalysis fields (speech_act, discourse, entities, slots,
    events, sentiment, epistemic_status, etc.)
  - Graph database context (justification chains, evidence chains,
    user preferences, conversation topics, recent context)
  - Enrichment data from external services
  - Memory signals (similar episodes, entity familiarity)
  - ContextAccumulator signals (combined confidence, conflict, hedging)
  """
  def build(%InternalModel{} = model, opts \\ []) do
    analyses = model.analyses || []
    primary = select_primary_analysis(analyses)
    user_id = Keyword.get(opts, :user_id)
    conversation_id = Keyword.get(opts, :conversation_id)

    analysis_context = extract_analysis_context(primary, analyses, model)
    graph_context = extract_graph_context(analyses, user_id, conversation_id)
    enrichment_context = extract_enrichment_context(primary, opts)
    memory_context = extract_memory_context(primary, analyses)
    accumulator_context = extract_accumulator_context(primary)

    %{
      analysis: analysis_context,
      graph: graph_context,
      enrichment: enrichment_context,
      memory: memory_context,
      accumulator: accumulator_context,
      raw_input: model.raw_input,
      overall_strategy: model.overall_strategy,
      user_id: user_id,
      conversation_id: conversation_id,
      primary_analysis: primary
    }
  end

  defp select_primary_analysis([]), do: %ChunkAnalysis{}

  defp select_primary_analysis(analyses) do
    respondable =
      Enum.filter(analyses, fn a ->
        a.response_strategy in [:can_respond, :hedged_response]
      end)

    case respondable do
      [primary | _] -> primary
      [] -> List.first(analyses) || %ChunkAnalysis{}
    end
  end

  defp extract_analysis_context(primary, analyses, model) do
    %{
      text: primary.text,
      intent: primary.intent,
      confidence: primary.confidence,
      response_strategy: primary.response_strategy,
      speech_act: serialize_speech_act(primary.speech_act),
      discourse: serialize_discourse(primary.discourse),
      sentiment: primary.sentiment,
      entities: primary.entities || [],
      slots: serialize_slots(primary.slots),
      epistemic_status: primary.epistemic_status,
      related_beliefs: primary.related_beliefs || [],
      events: primary.events || [],
      event_frames: Map.get(primary, :event_frames, []),
      srl_frames: Map.get(primary, :srl_frames, []),
      pos_tags: Map.get(primary, :pos_tags, []),
      fact_verification: primary.fact_verification,
      clarification_prompts: primary.clarification_prompts || [],
      chunk_count: length(analyses),
      all_strategies: Enum.map(analyses, & &1.response_strategy),
      raw_input: model.raw_input
    }
  end

  defp serialize_speech_act(nil), do: %{}

  defp serialize_speech_act(sa) when is_struct(sa) do
    %{
      category: Map.get(sa, :category),
      sub_type: Map.get(sa, :sub_type),
      confidence: Map.get(sa, :confidence),
      is_question: Map.get(sa, :is_question, false),
      is_imperative: Map.get(sa, :is_imperative, false),
      intent_confidence: Map.get(sa, :intent_confidence)
    }
  end

  defp serialize_speech_act(sa) when is_map(sa), do: sa
  defp serialize_speech_act(_), do: %{}

  defp serialize_discourse(nil), do: %{}

  defp serialize_discourse(d) when is_struct(d) do
    %{
      addressee: Map.get(d, :addressee),
      confidence: Map.get(d, :confidence),
      direct_address_detected: Map.get(d, :direct_address_detected, false),
      participants: Map.get(d, :participants, [])
    }
  end

  defp serialize_discourse(d) when is_map(d), do: d
  defp serialize_discourse(_), do: %{}

  defp serialize_slots(nil), do: %{filled: %{}, missing_required: [], missing_optional: []}

  defp serialize_slots(s) when is_struct(s) do
    %{
      filled: Map.get(s, :filled_slots, %{}),
      missing_required: Map.get(s, :missing_required, []),
      missing_optional: Map.get(s, :missing_optional, [])
    }
  end

  defp serialize_slots(s) when is_map(s), do: s
  defp serialize_slots(_), do: %{filled: %{}, missing_required: [], missing_optional: []}

  defp extract_graph_context(analyses, user_id, conversation_id) do
    beliefs = analyses |> Enum.flat_map(&(Map.get(&1, :related_beliefs) || [])) |> Enum.uniq()

    justification_chains = fetch_justification_chains(beliefs)
    evidence_chains = fetch_evidence_chains(beliefs)
    assumption_consequences = fetch_assumption_consequences(analyses, beliefs)
    user_preferences = if user_id, do: Reader.user_preferences(user_id), else: []

    conversation_topics =
      if conversation_id,
        do: safe_call(fn -> Reader.conversation_topics(to_string(conversation_id)) end),
        else: []

    recent_context =
      if conversation_id,
        do: safe_call(fn -> Reader.recent_context(to_string(conversation_id)) end),
        else: []

    %{
      justification_chains: justification_chains,
      evidence_chains: evidence_chains,
      assumption_consequences: assumption_consequences,
      user_preferences: user_preferences,
      conversation_topics: conversation_topics,
      recent_context: recent_context
    }
  end

  defp fetch_justification_chains(beliefs) do
    beliefs
    |> Enum.flat_map(fn belief ->
      node_id = Map.get(belief, :node_id) || Map.get(belief, "node_id")
      if node_id, do: safe_call(fn -> Reader.belief_justification_chain(to_string(node_id)) end), else: []
    end)
  end

  defp fetch_evidence_chains(beliefs) do
    beliefs
    |> Enum.flat_map(fn belief ->
      subject = Map.get(belief, :subject) || Map.get(belief, "subject")
      if subject, do: safe_call(fn -> Reader.evidence_chain(to_string(subject)) end), else: []
    end)
  end

  defp fetch_assumption_consequences(analyses, beliefs) do
    has_contradiction =
      Enum.any?(analyses, fn a -> Map.get(a, :epistemic_status) == :contradicted end)

    if has_contradiction do
      beliefs
      |> Enum.flat_map(fn belief ->
        node_id = Map.get(belief, :node_id) || Map.get(belief, "node_id")
        if node_id, do: safe_call(fn -> Reader.assumption_consequences(to_string(node_id)) end), else: []
      end)
    else
      []
    end
  end

  defp extract_enrichment_context(primary, _opts) do
    entities = primary.entities || []
    intent = primary.intent
    slots = build_slot_map(entities)

    base_context = %{
      entities: entities,
      intent: intent
    }

    enriched = Enricher.prepare_context(intent, slots, base_context)
    entity_familiarity = EntityGraphEnricher.familiarity_score(entities)

    %{
      enriched_data: Map.get(enriched, :enriched_data, %{}),
      enrichment_status: Map.get(enriched, :enrichment_status, :not_configured),
      entity_familiarity: entity_familiarity,
      graph_enrichment: Map.get(enriched, :graph_enrichment, %{})
    }
  end

  defp build_slot_map(entities) when is_list(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      type = Map.get(entity, :entity_type) || Map.get(entity, "entity_type")
      value = Map.get(entity, :value) || Map.get(entity, "value") || Map.get(entity, :text)

      if type && value do
        slot_name =
          try do
            type |> to_string() |> String.downcase() |> String.to_existing_atom()
          rescue
            ArgumentError -> :entity
          end

        Map.put(acc, slot_name, value)
      else
        acc
      end
    end)
  end

  defp build_slot_map(_), do: %{}

  defp extract_memory_context(primary, analyses) do
    accumulated_contexts =
      analyses
      |> Enum.map(&Map.get(&1, :accumulated_context))
      |> Enum.filter(&(not is_nil(&1)))

    episodes =
      accumulated_contexts
      |> Enum.flat_map(&(Map.get(&1, :relevant_episodes) || []))
      |> Enum.uniq()

    semantics =
      accumulated_contexts
      |> Enum.flat_map(&(Map.get(&1, :relevant_semantics) || []))
      |> Enum.uniq()

    query =
      [
        primary.intent || "",
        primary.text || "",
        (primary.entities || []) |> Enum.map_join(" ", fn e -> e[:value] || e["value"] || "" end)
      ]
      |> Enum.filter(&(&1 != ""))
      |> Enum.join(" ")

    fallback_episodes =
      if episodes == [] and query != "" do
        safe_call(fn ->
          if Process.whereis(Brain.Memory.Store) do
            case Brain.Memory.Store.query_similar(query, 5) do
              {:ok, results} -> results
              _ -> []
            end
          else
            []
          end
        end)
      else
        episodes
      end

    %{
      similar_episodes: fallback_episodes,
      semantic_facts: semantics
    }
  end

  defp extract_accumulator_context(primary) do
    case Map.get(primary, :accumulated_context) do
      %ContextAccumulator{} = acc ->
        %{
          combined_confidence: acc.combined_confidence,
          conflict_measure: acc.conflict_measure,
          effective_confidence: ContextAccumulator.effective_confidence(acc),
          should_hedge: ContextAccumulator.should_hedge?(acc),
          dominant_signal: acc.dominant_signal,
          entity_familiarity: acc.entity_familiarity,
          conversation_topics: acc.conversation_topics,
          interlocutor_adaptations: acc.interlocutor_adaptations
        }

      _ ->
        %{
          combined_confidence: 0.5,
          conflict_measure: 0.0,
          effective_confidence: 0.5,
          should_hedge: false,
          dominant_signal: nil,
          entity_familiarity: 0.5,
          conversation_topics: [],
          interlocutor_adaptations: %{}
        }
    end
  end

  defp safe_call(fun) do
    fun.()
  rescue
    _ -> []
  catch
    :exit, _ -> []
  end
end
