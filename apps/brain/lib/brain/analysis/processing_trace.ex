defmodule Brain.Analysis.ProcessingTrace do
  @moduledoc "Captures and formats the cognitive processing trace for UI visualization.\n\nThis module provides a structured view of what the system considered\nwhen processing a user message, including:\n- Semantic chunking (splitting multi-part messages)\n- Racing analyzer results per chunk\n- Fast-path triggers\n- Backtracking decisions\n- Slot detection\n- Final interpretation with alternatives\n"

  alias Brain.Analysis.{
    Interpretation,
    ChunkProfile,
    RacingAnalyzer,
    BacktrackController,
    HeuristicStore,
    SemanticChunker,
    DiscourseAnalyzer,
    SpeechActClassifier,
    SlotDetector
  }
  alias Brain.ML.EntityExtractor

  defstruct [
    :input_text,
    :timestamp,
    :chunk_count,
    :chunks,
    :total_processing_time_ms,
    :overall_strategy,
    :fast_path_triggered,
    :triggering_heuristic,
    :analyzer_results,
    :racing_time_ms,
    :primary_intent,
    :primary_activation,
    :primary_source,
    :confidence_level,
    :alternatives,
    :entities_found,
    :slots_filled,
    :slots_missing,
    :needs_clarification,
    :backtrack_count,
    :backtrack_reason,
    :response_strategy,
    :clarification_prompt,
    :total_activation,
    :activation_normalized
  ]

  defmodule ChunkTrace do
    @moduledoc false
    defstruct [
      :index,
      :text,
      :fast_path_triggered,
      :triggering_heuristic,
      :analyzer_results,
      :racing_time_ms,
      :primary_intent,
      :primary_activation,
      :primary_source,
      :confidence_level,
      :alternatives,
      :entities_found,
      :slots_filled,
      :slots_missing,
      :needs_clarification,
      :backtrack_count,
      :backtrack_reason,
      :response_strategy,
      :clarification_prompt,
      :total_activation,
      :activation_normalized
    ]
  end

  @type t :: %__MODULE__{}

  @doc "Creates a processing trace from an interpretation and related context.\n"
  def from_interpretation(%Interpretation{} = interp, opts \\ []) do
    backtrack_state = Keyword.get(opts, :backtrack_state)
    clarification = Keyword.get(opts, :clarification)
    racing_time = Keyword.get(opts, :racing_time_ms, 0)
    all_activations = [interp.activation | Enum.map(interp.alternatives, & &1.activation)]
    total = Enum.sum(all_activations)

    %__MODULE__{
      input_text: interp.text,
      timestamp: System.system_time(:millisecond),
      fast_path_triggered: Interpretation.from_heuristic?(interp),
      triggering_heuristic: get_heuristic_info(interp.triggering_heuristic_id),
      analyzer_results: format_analyzer_results(interp.analyzer_results),
      racing_time_ms: racing_time,
      primary_intent: interp.intent,
      primary_activation: Float.round(interp.activation, 3),
      primary_source: interp.source,
      confidence_level: Interpretation.confidence_level(interp),
      alternatives: format_alternatives(interp.alternatives),
      entities_found: format_entities(interp.entities),
      slots_filled: get_filled_slots(interp.slots),
      slots_missing: Interpretation.missing_required(interp),
      needs_clarification: Interpretation.has_missing_required?(interp),
      backtrack_count:
        if(backtrack_state) do
          backtrack_state.backtrack_count
        else
          0
        end,
      backtrack_reason: get_backtrack_reason(backtrack_state),
      response_strategy: determine_strategy(interp, clarification),
      clarification_prompt: get_clarification_prompt(clarification),
      total_activation: Float.round(total, 3),
      activation_normalized: total > 1.0
    }
  end

  @doc "Creates a simplified trace for display in the UI.\n"
  def to_display_map(%__MODULE__{} = trace) do
    chunks_display =
      (trace.chunks || [])
      |> Enum.map(fn chunk ->
        %{
          index: chunk.index,
          text: chunk.text,
          intent: chunk.primary_intent,
          confidence: format_confidence(chunk.primary_activation),
          confidence_level: chunk.confidence_level,
          source: format_source(chunk.primary_source),
          fast_path: chunk.fast_path_triggered,
          heuristic_name: chunk.triggering_heuristic && chunk.triggering_heuristic[:id],
          analyzers: chunk.analyzer_results,
          racing_ms: chunk.racing_time_ms,
          alternatives: chunk.alternatives,
          entities: chunk.entities_found,
          slots_filled: chunk.slots_filled,
          slots_missing: chunk.slots_missing,
          needs_clarification: chunk.needs_clarification,
          clarification: chunk.clarification_prompt,
          backtrack_count: chunk.backtrack_count,
          backtrack_reason: chunk.backtrack_reason,
          total_activation: chunk.total_activation,
          was_normalized: chunk.activation_normalized
        }
      end)

    %{
      chunk_count: trace.chunk_count || 1,
      chunks: chunks_display,
      total_processing_ms: trace.total_processing_time_ms,
      overall_strategy: trace.overall_strategy,
      intent: trace.primary_intent,
      confidence: format_confidence(trace.primary_activation),
      confidence_level: trace.confidence_level,
      source: format_source(trace.primary_source),
      fast_path: trace.fast_path_triggered,
      heuristic_name: trace.triggering_heuristic && trace.triggering_heuristic[:id],
      analyzers: trace.analyzer_results,
      racing_ms: trace.racing_time_ms,
      alternatives: trace.alternatives,
      entities: trace.entities_found,
      slots_filled: trace.slots_filled,
      slots_missing: trace.slots_missing,
      needs_clarification: trace.needs_clarification,
      clarification: trace.clarification_prompt,
      backtrack_count: trace.backtrack_count,
      backtrack_reason: trace.backtrack_reason,
      total_activation: trace.total_activation,
      was_normalized: trace.activation_normalized
    }
  end

  @doc "Runs a full trace of processing for a given input.\n\nThis is the main entry point for the UI to get processing details.\nUses semantic chunking to split multi-part messages.\n"
  def trace_processing(text, opts \\ []) do
    start_time = System.monotonic_time(:millisecond)
    chunks = SemanticChunker.chunk(text)

    chunk_traces =
      chunks
      |> Enum.with_index()
      |> Enum.map(fn {chunk, idx} ->
        trace_single_chunk(chunk.text, idx, opts)
      end)

    total_time = System.monotonic_time(:millisecond) - start_time
    overall_strategy = determine_overall_strategy(chunk_traces)
    primary_chunk = find_primary_chunk(chunk_traces)

    trace = %__MODULE__{
      input_text: text,
      timestamp: System.system_time(:millisecond),
      chunk_count: length(chunks),
      chunks: chunk_traces,
      total_processing_time_ms: total_time,
      overall_strategy: overall_strategy,
      fast_path_triggered: primary_chunk.fast_path_triggered,
      triggering_heuristic: primary_chunk.triggering_heuristic,
      analyzer_results: primary_chunk.analyzer_results,
      racing_time_ms: primary_chunk.racing_time_ms,
      primary_intent: primary_chunk.primary_intent,
      primary_activation: primary_chunk.primary_activation,
      primary_source: primary_chunk.primary_source,
      confidence_level: primary_chunk.confidence_level,
      alternatives: primary_chunk.alternatives,
      entities_found: primary_chunk.entities_found,
      slots_filled: primary_chunk.slots_filled,
      slots_missing: primary_chunk.slots_missing,
      needs_clarification: primary_chunk.needs_clarification,
      backtrack_count: primary_chunk.backtrack_count,
      backtrack_reason: primary_chunk.backtrack_reason,
      response_strategy: primary_chunk.response_strategy,
      clarification_prompt: primary_chunk.clarification_prompt,
      total_activation: primary_chunk.total_activation,
      activation_normalized: primary_chunk.activation_normalized
    }

    primary_interp = build_interpretation_from_chunk(primary_chunk, text)

    {trace, primary_interp}
  end

  defp trace_single_chunk(chunk_text, index, opts) do
    start_time = System.monotonic_time(:millisecond)
    interpretation = RacingAnalyzer.race(chunk_text, opts)

    racing_time = System.monotonic_time(:millisecond) - start_time

    discourse_result =
      try do
        DiscourseAnalyzer.analyze(chunk_text, [])
      rescue
        _ -> nil
      catch
        _ -> nil
      end

    speech_act_result =
      try do
        SpeechActClassifier.classify(chunk_text)
      rescue
        _ -> nil
      catch
        _ -> nil
      end

    entity_opts =
      opts ++
        [discourse: discourse_result, speech_act: speech_act_result]

    entities =
      try do
        EntityExtractor.extract_entities(chunk_text, entity_opts)
      rescue
        _ -> []
      catch
        _ -> []
      end

    interpretation = Interpretation.with_entities(interpretation, entities)

    slots =
      if interpretation.intent do
        SlotDetector.detect(interpretation.intent, entities)
      else
        nil
      end

    interpretation = Interpretation.with_slots(interpretation, slots)
    backtrack_state = BacktrackController.new(chunk_text)

    {final_interp, final_backtrack, clarification} =
      case BacktrackController.check_for_contradictions(interpretation) do
        :ok ->
          {interpretation, backtrack_state, nil}

        {:needs_backtrack, reason} ->
          handle_backtrack(interpretation, backtrack_state, reason)
      end

    all_activations = [
      final_interp.activation | Enum.map(final_interp.alternatives, & &1.activation)
    ]

    total = Enum.sum(all_activations)

    %ChunkTrace{
      index: index,
      text: chunk_text,
      fast_path_triggered: Interpretation.from_heuristic?(final_interp),
      triggering_heuristic: get_heuristic_info(final_interp.triggering_heuristic_id),
      analyzer_results: format_analyzer_results(final_interp.analyzer_results),
      racing_time_ms: racing_time,
      primary_intent: final_interp.intent,
      primary_activation: Float.round(final_interp.activation, 3),
      primary_source: final_interp.source,
      confidence_level: Interpretation.confidence_level(final_interp),
      alternatives: format_alternatives(final_interp.alternatives),
      entities_found: format_entities(final_interp.entities),
      slots_filled: get_filled_slots(final_interp.slots),
      slots_missing: Interpretation.missing_required(final_interp),
      needs_clarification: Interpretation.has_missing_required?(final_interp),
      backtrack_count: final_backtrack.backtrack_count,
      backtrack_reason: get_backtrack_reason(final_backtrack),
      response_strategy: determine_chunk_strategy(final_interp, clarification),
      clarification_prompt: get_clarification_prompt(clarification),
      total_activation: Float.round(total, 3),
      activation_normalized: total > 1.0
    }
  end

  defp determine_overall_strategy(chunk_traces) do
    strategies = Enum.map(chunk_traces, & &1.response_strategy)

    cond do
      Enum.all?(strategies, &(&1 == :can_respond)) -> :can_respond
      Enum.any?(strategies, &(&1 == :force_clarification)) -> :needs_clarification
      Enum.any?(strategies, &(&1 == :needs_clarification)) -> :partial_response_with_clarification
      Enum.any?(strategies, &(&1 == :can_respond)) -> :partial_response_with_clarification
      true -> :low_confidence
    end
  end

  defp find_primary_chunk([]) do
    %ChunkTrace{}
  end

  defp find_primary_chunk(chunk_traces) do
    priority_domains = [
      :question,
      :weather,
      :device,
      :music,
      :reminder,
      :action,
      :information,
      :search
    ]

    Enum.find(chunk_traces, List.first(chunk_traces), fn trace ->
      intent = trace.primary_intent
      profile = Map.get(trace, :profile)

      domain =
        case profile do
          %ChunkProfile{domain: d} when d != :unknown -> d
          _ ->
            case String.split(to_string(intent || ""), ".", parts: 2) do
              [d, _] -> String.to_atom(d)
              _ -> nil
            end
        end

      domain in priority_domains
    end)
  end

  defp build_interpretation_from_chunk(chunk_trace, original_text) do
    %Interpretation{
      intent: chunk_trace.primary_intent,
      text: original_text,
      raw_activation: chunk_trace.primary_activation,
      activation: chunk_trace.primary_activation,
      calibrated_activation: chunk_trace.primary_activation,
      source: chunk_trace.primary_source,
      alternatives: [],
      entities: [],
      slots: nil
    }
  end

  defp determine_chunk_strategy(interp, clarification) do
    cond do
      clarification != nil -> :force_clarification
      Interpretation.has_missing_required?(interp) -> :needs_clarification
      interp.activation >= 0.6 -> :can_respond
      true -> :low_confidence
    end
  end

  defp handle_backtrack(interp, state, reason) do
    case BacktrackController.attempt_backtrack(state, interp, reason) do
      {:ok, new_state, new_interp, _cost} ->
        case BacktrackController.check_for_contradictions(new_interp) do
          :ok ->
            {new_interp, new_state, nil}

          {:needs_backtrack, new_reason} ->
            handle_backtrack(new_interp, new_state, new_reason)
        end

      {:force_clarification, clarification} ->
        {interp, state, clarification}

      {:error, :no_alternatives} ->
        {interp, state, nil}
    end
  end

  defp get_heuristic_info(nil) do
    nil
  end

  defp get_heuristic_info(heuristic_id) do
    case HeuristicStore.get(heuristic_id) do
      nil -> nil
      h -> %{id: h.id, scope: h.scope, pattern: h.pattern}
    end
  end

  defp format_analyzer_results(results) when is_list(results) do
    results
    |> Enum.map(fn r ->
      %{
        analyzer: format_source(r.analyzer),
        intent: r.intent,
        raw_score: Float.round(r.raw_score, 3),
        calibrated: Float.round(r.calibrated_activation, 3),
        indicators: r.indicators
      }
    end)
    |> Enum.sort_by(& &1.calibrated, :desc)
    |> Enum.take(5)
  end

  defp format_analyzer_results(_) do
    []
  end

  defp format_alternatives(alternatives) when is_list(alternatives) do
    alternatives
    |> Enum.take(3)
    |> Enum.map(fn alt ->
      %{
        intent: alt.intent,
        activation: Float.round(alt.activation, 3),
        source: format_source(alt.source)
      }
    end)
  end

  defp format_alternatives(_) do
    []
  end

  defp format_entities(entities) when is_list(entities) do
    Enum.map(entities, fn e ->
      %{
        type: e[:entity_type] || e[:type] || "unknown",
        value: e[:value] || "unknown",
        confidence: Float.round((e[:confidence] || 0.8) * 1.0, 2)
      }
    end)
  end

  defp format_entities(_) do
    []
  end

  defp get_filled_slots(nil) do
    %{}
  end

  defp get_filled_slots(%{filled_slots: slots}) when is_map(slots) do
    Map.new(slots, fn {k, v} ->
      value =
        case v do
          %{value: val} -> val
          val -> val
        end

      {k, value}
    end)
  end

  defp get_filled_slots(_) do
    %{}
  end

  defp get_backtrack_reason(nil) do
    nil
  end

  defp get_backtrack_reason(%{demoted_interpretations: [%{reason: reason} | _]}) do
    format_backtrack_reason(reason)
  end

  defp get_backtrack_reason(_) do
    nil
  end

  defp format_backtrack_reason({:missing_required, slots}) do
    "Missing: #{inspect(slots)}"
  end

  defp format_backtrack_reason({:entity_mismatch, msg}) do
    "Entity mismatch: #{msg}"
  end

  defp format_backtrack_reason({:low_confidence, val}) do
    "Low confidence: #{val}"
  end

  defp format_backtrack_reason(other) do
    inspect(other)
  end

  defp determine_strategy(interp, clarification) do
    cond do
      clarification != nil -> :force_clarification
      Interpretation.has_missing_required?(interp) -> :needs_clarification
      interp.activation >= 0.6 -> :can_respond
      true -> :low_confidence
    end
  end

  defp get_clarification_prompt(nil) do
    nil
  end

  defp get_clarification_prompt(%{prompt: prompt}) do
    prompt
  end

  defp get_clarification_prompt(_) do
    nil
  end

  defp format_confidence(activation) when is_float(activation) do
    "#{round(activation * 100)}%"
  end

  defp format_confidence(_) do
    "0%"
  end

  defp format_source(source) when is_atom(source) do
    source
    |> Atom.to_string()
    |> String.replace("_", " ")
    |> String.split()
    |> Enum.map_join(
      " ",
      &String.capitalize/1
    )
  end

  defp format_source(_) do
    "Unknown"
  end
end