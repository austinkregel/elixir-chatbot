defmodule World.EntityDiscoverer do
  @moduledoc "Discovers unknown entities from text using POS tagging.\n\nUses the trained POS tagger model to identify proper nouns (PROPN),\nthen checks against the gazetteer to determine if they are:\n- Known (single type)\n- Ambiguous (multiple types)\n- Unknown (new discovery)\n\nEmits events for everything found, including ambiguities.\nThis module does NOT use regex or explicit string matching.\n"

  alias Brain.ML
  require Logger

  alias ML.{POSTagger, Tokenizer, Gazetteer}
  alias World.Manager, as: WorldManager
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

  @doc "Discovers entities in text for a given training world.\n\nReturns a list of discovery results and emits events for each.\nUses POS tagging to identify proper nouns (PROPN), then checks\nthe gazetteer for known entities.\n\n## Options\n  - `:model` - Pre-loaded POS model (will load if not provided)\n  - `:context_window` - Number of tokens around entity for context (default: 5)\n  - `:emit_events` - Whether to emit world events (default: true)\n"
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
        tokens = Tokenizer.tokenize(text)
        token_texts = Enum.map(tokens, & &1.text)
        pos_predictions = POSTagger.predict(token_texts, model)
        proper_nouns = extract_proper_nouns(tokens, pos_predictions, context_window)

        discoveries =
          Enum.map(proper_nouns, fn pn ->
            analyze_proper_noun(pn, world_id, tokens, pos_predictions, emit_events)
          end)

        discoveries
    end
  end

  @doc "Processes a batch of texts for entity discovery.\n\nMore efficient than calling discover_entities individually\nas it loads the model once.\n"
  def discover_entities_batch(texts, world_id, opts \\ []) when is_list(texts) do
    model = Keyword.get_lazy(opts, :model, &load_pos_model/0)
    opts_with_model = Keyword.put(opts, :model, model)

    Enum.flat_map(texts, fn text ->
      discover_entities(text, world_id, opts_with_model)
    end)
  end

  @doc "Finds entities that appear multiple times across a list of discoveries.\n\nReturns entities sorted by occurrence count, useful for identifying\ncandidates that should be promoted to the gazetteer.\n"
  def aggregate_discoveries(discoveries) when is_list(discoveries) do
    discoveries
    |> Enum.filter(&((Map.get(&1, :status) || :unknown) == :unknown))
    |> Enum.group_by(&String.downcase(&1.value))
    |> Enum.map(fn {normalized, occurrences} ->
      first = hd(occurrences)

      inferred_types =
        occurrences
        |> Enum.map(& &1.inferred_type)
        |> Enum.filter(&(&1 != nil))
        |> Enum.frequencies()

      most_common_type =
        case Enum.max_by(inferred_types, fn {_, count} -> count end, fn -> nil end) do
          {type, _} -> type
          nil -> "unknown"
        end

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

  defp load_pos_model do
    case POSTagger.load_model() do
      {:ok, model} -> model
      {:error, _} -> nil
    end
  end

  defp extract_proper_nouns(tokens, pos_predictions, context_window) do
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
    proper_nouns
    |> Enum.reduce([], fn pn, acc ->
      case acc do
        [] ->
          [pn]

        [prev | rest] ->
          if pn.token_index == prev.token_index + 1 do
            curr_token = Enum.at(tokens, pn.token_index)

            merged = %{
              value: prev.value <> " " <> pn.value,
              token_index: pn.token_index,
              start_pos: prev.start_pos,
              end_pos: curr_token.end_pos,
              context: pn.context,
              token_indices: Map.get(prev, :token_indices, [prev.token_index]) ++ [pn.token_index]
            }

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
    |> Enum.map_join(
      " ",
      & &1.text
    )
  end

  defp analyze_proper_noun(pn, world_id, tokens, pos_predictions, emit_events) do
    known_types = Gazetteer.lookup_all_types(pn.value, world_id)

    {status, inferred_type, confidence} =
      case known_types do
        [] ->
          {type, conf} = infer_type_from_context(pn, tokens, pos_predictions)
          {:unknown, type, conf}

        [single] ->
          type = Map.get(single, :entity_type) || Map.get(single, :type)
          {:known, type, 1.0}

        multiple ->
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

    if emit_events do
      emit_discovery_event(result, world_id)
    end

    result
  end

  defp infer_type_from_context(pn, tokens, pos_predictions) do
    context_tags = extract_context_tags(pn, pos_predictions)
    world_id = World.Context.default_world_id()

    if world_id do
      case World.TypeInferrer.infer_type(pn.value, tokens, context_tags, world_id) do
        {type, confidence} when is_binary(type) and is_float(confidence) ->
          {type, confidence}

        _ ->
          # Fallback to local context confidence
          confidence = calculate_context_confidence(context_tags)
          inferred_type = infer_from_pos_context(context_tags, tokens, pn)
          {inferred_type, confidence}
      end
    else
      confidence = calculate_context_confidence(context_tags)
      inferred_type = infer_from_pos_context(context_tags, tokens, pn)
      {inferred_type, confidence}
    end
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
    base_confidence = 0.3

    verb_bonus =
      if Enum.any?(context_tags, &(&1 in ["VERB", "AUX"])) do
        0.2
      else
        0.0
      end

    grammar_bonus =
      if Enum.any?(context_tags, &(&1 in ["DET", "ADP"])) do
        0.1
      else
        0.0
      end

    min(base_confidence + verb_bonus + grammar_bonus, 1.0)
  end

  defp infer_from_pos_context(context_tags, tokens, pn) do
    idx = pn.token_index

    prev_tag =
      if idx > 0 do
        Enum.at(context_tags, 0)
      else
        nil
      end

    prev_token =
      if idx > 0 do
        Enum.at(tokens, idx - 1)
      else
        nil
      end

    cond do
      prev_tag == "ADP" ->
        "location"

      prev_tag == "DET" ->
        "organization"

      idx == 0 ->
        "person"

      prev_tag == "PROPN" and prev_token != nil ->
        "person"

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

    WorldManager.record_event(world_id, event_type, event_data, confidence: result.confidence)

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
