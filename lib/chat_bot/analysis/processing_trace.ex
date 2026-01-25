defmodule ChatBot.Analysis.ProcessingTrace do
  @moduledoc """
  Captures and formats the cognitive processing trace for UI visualization.

  This module provides a structured view of what the system considered
  when processing a user message, including:
  - Semantic chunking (splitting multi-part messages)
  - Racing analyzer results per chunk
  - Fast-path triggers
  - Backtracking decisions
  - Slot detection
  - Final interpretation with alternatives
  """

  alias ChatBot.Analysis.{
    Interpretation,
    RacingAnalyzer,
    BacktrackController,
    HeuristicStore,
    SemanticChunker,
    DiscourseAnalyzer,
    SpeechActClassifier,
    IntentRegistry
  }

  alias ChatBot.Analysis.SlotDetector
  alias ChatBot.ML.EntityExtractor

  # Trace for the full message
  defstruct [
    # Input
    :input_text,
    :timestamp,

    # Chunking
    :chunk_count,
    :chunks,

    # Overall metrics
    :total_processing_time_ms,
    :overall_strategy,

    # Legacy fields for single-chunk display
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

  # Trace for an individual chunk
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

  @doc """
  Creates a processing trace from an interpretation and related context.
  """
  def from_interpretation(%Interpretation{} = interp, opts \\ []) do
    backtrack_state = Keyword.get(opts, :backtrack_state)
    clarification = Keyword.get(opts, :clarification)
    racing_time = Keyword.get(opts, :racing_time_ms, 0)

    # Calculate total activation
    all_activations = [interp.activation | Enum.map(interp.alternatives, & &1.activation)]
    total = Enum.sum(all_activations)

    %__MODULE__{
      input_text: interp.text,
      timestamp: System.system_time(:millisecond),

      # Fast path
      fast_path_triggered: Interpretation.from_heuristic?(interp),
      triggering_heuristic: get_heuristic_info(interp.triggering_heuristic_id),

      # Racing
      analyzer_results: format_analyzer_results(interp.analyzer_results),
      racing_time_ms: racing_time,

      # Primary
      primary_intent: interp.intent,
      primary_activation: Float.round(interp.activation, 3),
      primary_source: interp.source,
      confidence_level: Interpretation.confidence_level(interp),

      # Alternatives
      alternatives: format_alternatives(interp.alternatives),

      # Slots
      entities_found: format_entities(interp.entities),
      slots_filled: get_filled_slots(interp.slots),
      slots_missing: Interpretation.missing_required(interp),
      needs_clarification: Interpretation.has_missing_required?(interp),

      # Backtracking
      backtrack_count: if(backtrack_state, do: backtrack_state.backtrack_count, else: 0),
      backtrack_reason: get_backtrack_reason(backtrack_state),

      # Decision
      response_strategy: determine_strategy(interp, clarification),
      clarification_prompt: get_clarification_prompt(clarification),

      # Stability
      total_activation: Float.round(total, 3),
      activation_normalized: total > 1.0
    }
  end

  @doc """
  Creates a simplified trace for display in the UI.
  """
  def to_display_map(%__MODULE__{} = trace) do
    # Convert chunk traces for display
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
      # Multi-chunk info
      chunk_count: trace.chunk_count || 1,
      chunks: chunks_display,
      total_processing_ms: trace.total_processing_time_ms,
      overall_strategy: trace.overall_strategy,

      # Header info (from primary chunk for backwards compatibility)
      intent: trace.primary_intent,
      confidence: format_confidence(trace.primary_activation),
      confidence_level: trace.confidence_level,
      source: format_source(trace.primary_source),

      # What happened
      fast_path: trace.fast_path_triggered,
      heuristic_name: trace.triggering_heuristic && trace.triggering_heuristic[:id],

      # Racing summary
      analyzers: trace.analyzer_results,
      racing_ms: trace.racing_time_ms,

      # Alternatives
      alternatives: trace.alternatives,

      # Context
      entities: trace.entities_found,
      slots_filled: trace.slots_filled,
      slots_missing: trace.slots_missing,

      # Decision
      needs_clarification: trace.needs_clarification,
      clarification: trace.clarification_prompt,
      backtrack_count: trace.backtrack_count,
      backtrack_reason: trace.backtrack_reason,

      # Health
      total_activation: trace.total_activation,
      was_normalized: trace.activation_normalized
    }
  end

  @doc """
  Runs a full trace of processing for a given input.

  This is the main entry point for the UI to get processing details.
  Uses semantic chunking to split multi-part messages.
  """
  def trace_processing(text, opts \\ []) do
    start_time = System.monotonic_time(:millisecond)

    # Step 1: Chunk the input into semantic units
    chunks = SemanticChunker.chunk(text)

    # Step 2: Trace each chunk
    chunk_traces =
      chunks
      |> Enum.with_index()
      |> Enum.map(fn {chunk, idx} ->
        trace_single_chunk(chunk.text, idx, opts)
      end)

    total_time = System.monotonic_time(:millisecond) - start_time

    # Step 3: Determine overall strategy
    overall_strategy = determine_overall_strategy(chunk_traces)

    # Step 4: Build the combined trace
    # For backwards compatibility, also populate the legacy single-interpretation fields
    # using the most "important" chunk (prioritize questions/commands over greetings)
    primary_chunk = find_primary_chunk(chunk_traces)

    trace = %__MODULE__{
      input_text: text,
      timestamp: System.system_time(:millisecond),
      chunk_count: length(chunks),
      chunks: chunk_traces,
      total_processing_time_ms: total_time,
      overall_strategy: overall_strategy,

      # Legacy fields from primary chunk
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

    # Return the primary interpretation for compatibility
    primary_interp = build_interpretation_from_chunk(primary_chunk, text)

    {trace, primary_interp}
  end

  defp trace_single_chunk(chunk_text, index, opts) do
    start_time = System.monotonic_time(:millisecond)

    # Run racing analyzers
    interpretation = RacingAnalyzer.race(chunk_text, opts)

    racing_time = System.monotonic_time(:millisecond) - start_time

    # Extract discourse and speech_act context for proper entity disambiguation
    # This ensures entities are disambiguated with the same context as the main pipeline
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

    # Extract entities with proper disambiguation context
    entity_opts =
      opts ++
        [
          discourse: discourse_result,
          speech_act: speech_act_result
        ]

    entities =
      try do
        EntityExtractor.extract_entities(chunk_text, entity_opts)
      rescue
        _ -> []
      catch
        _ -> []
      end

    interpretation = Interpretation.with_entities(interpretation, entities)

    # Detect slots
    slots =
      if interpretation.intent do
        SlotDetector.detect(interpretation.intent, entities)
      else
        nil
      end

    interpretation = Interpretation.with_slots(interpretation, slots)

    # Check for contradictions
    backtrack_state = BacktrackController.new(chunk_text)

    {final_interp, final_backtrack, clarification} =
      case BacktrackController.check_for_contradictions(interpretation) do
        :ok ->
          {interpretation, backtrack_state, nil}

        {:needs_backtrack, reason} ->
          handle_backtrack(interpretation, backtrack_state, reason)
      end

    # Calculate total activation
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

  defp find_primary_chunk([]), do: %ChunkTrace{}

  defp find_primary_chunk(chunk_traces) do
    # Priority: questions/commands > weather > other substantive > greetings
    # Using IntentRegistry domain checks instead of string prefix matching
    priority_domains = [:question, :weather, :device, :music, :reminder, :action, :information, :search]

    Enum.find(chunk_traces, List.first(chunk_traces), fn trace ->
      intent = trace.primary_intent
      domain = IntentRegistry.domain(intent)
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

  # Private functions

  defp handle_backtrack(interp, state, reason) do
    case BacktrackController.attempt_backtrack(state, interp, reason) do
      {:ok, new_state, new_interp, _cost} ->
        # Check the new interpretation
        case BacktrackController.check_for_contradictions(new_interp) do
          :ok ->
            {new_interp, new_state, nil}

          {:needs_backtrack, new_reason} ->
            # Try again if we have budget
            handle_backtrack(new_interp, new_state, new_reason)
        end

      {:force_clarification, clarification} ->
        {interp, state, clarification}

      {:error, :no_alternatives} ->
        {interp, state, nil}
    end
  end

  defp get_heuristic_info(nil), do: nil

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

  defp format_analyzer_results(_), do: []

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

  defp format_alternatives(_), do: []

  defp format_entities(entities) when is_list(entities) do
    Enum.map(entities, fn e ->
      %{
        type: e[:entity] || e["entity"] || e[:type] || "unknown",
        value: e[:value] || e["value"] || "unknown",
        confidence: Float.round((e[:confidence] || e["confidence"] || 0.8) * 1.0, 2)
      }
    end)
  end

  defp format_entities(_), do: []

  defp get_filled_slots(nil), do: %{}

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

  defp get_filled_slots(_), do: %{}

  defp get_backtrack_reason(nil), do: nil

  defp get_backtrack_reason(%{demoted_interpretations: [%{reason: reason} | _]}) do
    format_backtrack_reason(reason)
  end

  defp get_backtrack_reason(_), do: nil

  defp format_backtrack_reason({:missing_required, slots}), do: "Missing: #{inspect(slots)}"
  defp format_backtrack_reason({:entity_mismatch, msg}), do: "Entity mismatch: #{msg}"
  defp format_backtrack_reason({:low_confidence, val}), do: "Low confidence: #{val}"
  defp format_backtrack_reason(other), do: inspect(other)

  defp determine_strategy(interp, clarification) do
    cond do
      clarification != nil -> :force_clarification
      Interpretation.has_missing_required?(interp) -> :needs_clarification
      interp.activation >= 0.6 -> :can_respond
      true -> :low_confidence
    end
  end

  defp get_clarification_prompt(nil), do: nil
  defp get_clarification_prompt(%{prompt: prompt}), do: prompt
  defp get_clarification_prompt(_), do: nil

  defp format_confidence(activation) when is_float(activation) do
    "#{round(activation * 100)}%"
  end

  defp format_confidence(_), do: "0%"

  defp format_source(source) when is_atom(source) do
    source
    |> Atom.to_string()
    |> String.replace("_", " ")
    |> String.split()
    |> Enum.map(&String.capitalize/1)
    |> Enum.join(" ")
  end

  defp format_source(_), do: "Unknown"
end
