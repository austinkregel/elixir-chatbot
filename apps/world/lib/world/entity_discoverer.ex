defmodule World.EntityDiscoverer do
  @moduledoc """
  Discovers unknown entities from text using POS tagging.

  Uses the trained POS tagger model to identify proper nouns (PROPN),
  then checks against the gazetteer to determine if they are:
  - Known (single type)
  - Ambiguous (multiple types)
  - Unknown (new discovery)

  Emits events for everything found, including ambiguities.
  This module does NOT use regex or explicit string matching.
  """

  require Logger

  alias Brain.ML.{POSTagger, Tokenizer, Gazetteer}
  alias World.Manager, as: WorldManager, as: WorldManager
  alias World.Metrics, as: WorldMetrics

  @type discovery_result :: %{
          value: String.t(),
          position: {non_neg_integer(), non_neg_integer()},
          context: String.t(),
          known_types: [map()],
          inferred_type: String.t() | nil,
          confidence: float(),
          status: :unknown | :known | :ambiguous
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Discovers entities in text for a given training world.

  Returns a list of discovery results and emits events for each.
  Uses POS tagging to identify proper nouns (PROPN), then checks
  the gazetteer for known entities.

  ## Options
    - `:model` - Pre-loaded POS model (will load if not provided)
    - `:context_window` - Number of tokens around entity for context (default: 5)
    - `:emit_events` - Whether to emit world events (default: true)
  """
  def discover_entities(text, world_id, opts \\ [])
      when is_binary(text) and is_binary(world_id) do
    model = Keyword.get_lazy(opts, :model, &load_pos_model/0)
    context_window = Keyword.get(opts, :context_window, 5)
    emit_events = Keyword.get(opts, :emit_events, true)

    case model do
      nil ->
        Logger.warning("POS model not available, entity discovery skipped")
        []

      model ->
        # Tokenize the text
        tokens = Tokenizer.tokenize(text)
        token_texts = Enum.map(tokens, & &1.text)

        # Get POS tags for all tokens
        pos_predictions = POSTagger.predict(token_texts, model)

        # Find proper nouns and their context
        proper_nouns = extract_proper_nouns(tokens, pos_predictions, context_window)

        # Analyze each proper noun
        discoveries =
          Enum.map(proper_nouns, fn pn ->
            analyze_proper_noun(pn, world_id, tokens, pos_predictions, emit_events)
          end)

        discoveries
    end
  end

  @doc """
  Processes a batch of texts for entity discovery.

  More efficient than calling discover_entities individually
  as it loads the model once.
  """
  def discover_entities_batch(texts, world_id, opts \\ []) when is_list(texts) do
    model = Keyword.get_lazy(opts, :model, &load_pos_model/0)
    opts_with_model = Keyword.put(opts, :model, model)

    Enum.flat_map(texts, fn text ->
      discover_entities(text, world_id, opts_with_model)
    end)
  end

  @doc """
  Finds entities that appear multiple times across a list of discoveries.

  Returns entities sorted by occurrence count, useful for identifying
  candidates that should be promoted to the gazetteer.
  """
  def aggregate_discoveries(discoveries) when is_list(discoveries) do
    discoveries
    |> Enum.filter(&(&1.status == :unknown))
    |> Enum.group_by(&String.downcase(&1.value))
    |> Enum.map(fn {normalized, occurrences} ->
      # Take the first occurrence for representative data
      first = hd(occurrences)

      # Collect all inferred types
      inferred_types =
        occurrences
        |> Enum.map(& &1.inferred_type)
        |> Enum.filter(&(&1 != nil))
        |> Enum.frequencies()

      # Get the most common inferred type
      most_common_type =
        case Enum.max_by(inferred_types, fn {_, count} -> count end, fn -> nil end) do
          {type, _} -> type
          nil -> "unknown"
        end

      # Average confidence
      avg_confidence =
        occurrences
        |> Enum.map(& &1.confidence)
        |> Enum.sum()
        |> Kernel./(length(occurrences))

      %{
        value: first.value,
        normalized: normalized,
        occurrences: length(occurrences),
        contexts: Enum.map(occurrences, & &1.context) |> Enum.take(10),
        inferred_type: most_common_type,
        type_distribution: inferred_types,
        confidence: avg_confidence
      }
    end)
    |> Enum.sort_by(& &1.occurrences, :desc)
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp load_pos_model do
    case POSTagger.load_model() do
      {:ok, model} -> model
      {:error, _} -> nil
    end
  end

  defp extract_proper_nouns(tokens, pos_predictions, context_window) do
    # Find sequences of PROPN tags (for multi-word names)
    pos_predictions
    |> Enum.with_index()
    |> Enum.reduce([], fn {{_token_text, tag}, idx}, acc ->
      if tag == "PROPN" do
        token = Enum.at(tokens, idx)
        context = extract_context(tokens, idx, context_window)

        pn = %{
          value: token.text,
          token_index: idx,
          start_pos: token.start_pos,
          end_pos: token.end_pos,
          context: context
        }

        [pn | acc]
      else
        acc
      end
    end)
    |> Enum.reverse()
    |> merge_consecutive_proper_nouns(tokens)
  end

  defp merge_consecutive_proper_nouns(proper_nouns, tokens) do
    # Merge consecutive proper nouns into single entities (e.g., "New York")
    proper_nouns
    |> Enum.reduce([], fn pn, acc ->
      case acc do
        [] ->
          [pn]

        [prev | rest] ->
          if pn.token_index == prev.token_index + 1 do
            # Consecutive - merge
            prev_token = Enum.at(tokens, prev.token_index)
            curr_token = Enum.at(tokens, pn.token_index)

            merged = %{
              value: prev.value <> " " <> pn.value,
              token_index: pn.token_index,
              start_pos: prev.start_pos,
              end_pos: curr_token.end_pos,
              context: pn.context,
              # Track all indices for multi-word
              token_indices: Map.get(prev, :token_indices, [prev.token_index]) ++ [pn.token_index]
            }

            _ = prev_token
            [merged | rest]
          else
            [pn | acc]
          end
      end
    end)
    |> Enum.reverse()
  end

  defp extract_context(tokens, idx, window) do
    start_idx = max(0, idx - window)
    end_idx = min(length(tokens) - 1, idx + window)

    tokens
    |> Enum.slice(start_idx..end_idx)
    |> Enum.map(& &1.text)
    |> Enum.join(" ")
  end

  defp analyze_proper_noun(pn, world_id, tokens, pos_predictions, emit_events) do
    # Check gazetteer (including world overlay)
    known_types = Gazetteer.lookup_all_types(pn.value, world_id)

    {status, inferred_type, confidence} =
      case known_types do
        [] ->
          # Unknown - try to infer type from context
          {type, conf} = infer_type_from_context(pn, tokens, pos_predictions)
          {:unknown, type, conf}

        [single] ->
          # Known with single type
          type = Map.get(single, :entity_type) || Map.get(single, :type)
          {:known, type, 1.0}

        multiple ->
          # Ambiguous - multiple possible types
          types = Enum.map(multiple, &(Map.get(&1, :entity_type) || Map.get(&1, :type)))
          {:ambiguous, Enum.join(types, "|"), 0.5}
      end

    result = %{
      value: pn.value,
      position: {pn.start_pos, pn.end_pos},
      context: pn.context,
      known_types: known_types,
      inferred_type: inferred_type,
      confidence: confidence,
      status: status
    }

    # Emit events and record candidate if enabled
    if emit_events do
      emit_discovery_event(result, world_id)
    end

    result
  end

  defp infer_type_from_context(pn, tokens, pos_predictions) do
    # Use the TypeInferrer if available, otherwise use basic heuristics
    # based on POS context (no string matching)
    context_tags = extract_context_tags(pn, pos_predictions)

    # Calculate confidence based on context clarity
    confidence = calculate_context_confidence(context_tags)

    # Infer type based on surrounding POS patterns
    inferred_type = infer_from_pos_context(context_tags, tokens, pn)

    {inferred_type, confidence}
  end

  defp extract_context_tags(pn, pos_predictions) do
    idx = pn.token_index
    window = 3

    start_idx = max(0, idx - window)
    end_idx = min(length(pos_predictions) - 1, idx + window)

    pos_predictions
    |> Enum.slice(start_idx..end_idx)
    |> Enum.map(fn {_token, tag} -> tag end)
  end

  defp calculate_context_confidence(context_tags) do
    # Higher confidence if context has clear grammatical structure
    # More context tags = more context to work with
    base_confidence = 0.3

    # Bonus for having verb context (indicates sentence structure)
    verb_bonus = if Enum.any?(context_tags, &(&1 in ["VERB", "AUX"])), do: 0.2, else: 0.0

    # Bonus for having determiners/prepositions (clear grammatical role)
    grammar_bonus = if Enum.any?(context_tags, &(&1 in ["DET", "ADP"])), do: 0.1, else: 0.0

    min(base_confidence + verb_bonus + grammar_bonus, 1.0)
  end

  defp infer_from_pos_context(context_tags, tokens, pn) do
    # Use grammatical patterns to infer entity type
    # This is based on learned patterns, not hard-coded strings

    idx = pn.token_index

    # Check for patterns like "ADP PROPN" (at [location], in [location])
    prev_tag = if idx > 0, do: Enum.at(context_tags, 0), else: nil

    # Check token before for title-like words (requires looking at actual token)
    prev_token =
      if idx > 0 do
        Enum.at(tokens, idx - 1)
      else
        nil
      end

    cond do
      # If preceded by preposition, likely a location
      prev_tag == "ADP" ->
        "location"

      # If preceded by determiner, could be organization or thing
      prev_tag == "DET" ->
        "organization"

      # If capitalized and at sentence start, might be person
      # (check if first token)
      idx == 0 ->
        # Sentence-initial proper nouns are often persons in dialogue
        "person"

      # If preceded by a proper noun (title + name pattern)
      prev_tag == "PROPN" and prev_token != nil ->
        # Could be part of a multi-word name
        "person"

      # Default to unknown/general entity
      true ->
        "entity"
    end
  end

  defp emit_discovery_event(result, world_id) do
    event_type =
      case result.status do
        :unknown -> :entity_candidate_detected
        :known -> :entity_occurrence
        :ambiguous -> :entity_ambiguity_detected
      end

    event_data = %{
      value: result.value,
      position: result.position,
      context: result.context,
      inferred_type: result.inferred_type,
      known_types:
        Enum.map(result.known_types, &(Map.get(&1, :entity_type) || Map.get(&1, :type))),
      confidence: result.confidence
    }

    # Record event
    WorldManager.record_event(world_id, event_type, event_data, confidence: result.confidence)

    # If unknown, add as candidate
    if result.status == :unknown do
      candidate = %{
        value: result.value,
        inferred_type: result.inferred_type,
        confidence: result.confidence,
        context: result.context,
        discovered_at: DateTime.utc_now(),
        occurrences: 1
      }

      WorldManager.add_candidate(world_id, candidate)
    end

    # If ambiguous, record the ambiguity in metrics
    if result.status == :ambiguous do
      ambiguity_info = %{
        value: result.value,
        types: Enum.map(result.known_types, &(Map.get(&1, :entity_type) || Map.get(&1, :type))),
        context: result.context,
        detected_at: DateTime.utc_now()
      }

      WorldManager.update_metrics(world_id, &WorldMetrics.record_ambiguity(&1, ambiguity_info))
    end
  end
end
