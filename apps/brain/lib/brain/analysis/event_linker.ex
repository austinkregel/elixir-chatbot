defmodule Brain.Analysis.EventLinker do
  @moduledoc """
  Links detected event triggers to their arguments using trained classifiers.

  After event triggers are identified (by the rule-based EventExtractor),
  this module:

  1. Assigns argument roles (ARG0, ARG1, ARGM-TMP, ARGM-LOC) using the
     trained `:event_argument_role` MicroClassifier
  2. Detects temporal relations between events using EntityExtractor's
     temporal entity detection
  3. Creates sub-event links when one event's temporal span contains another's

  All classification is data-driven via MicroClassifiers -- no string matching
  or positional heuristics for NLP decisions.
  """

  alias Brain.ML.MicroClassifiers
  alias Brain.Analysis.TemporalResolver
  require Logger

  # Entity types (strings, as EntityExtractor emits them) that denote a time.
  @temporal_types ~w(sys-date date datetime relative_date day weekday month day_name month_name time sys-time year temporal)

  @type event_frame :: %{
    trigger: String.t(),
    trigger_index: non_neg_integer(),
    tense: atom(),
    arguments: [argument()],
    temporal_relations: [temporal_relation()],
    sub_events: [non_neg_integer()]
  }

  @type argument :: %{
    text: String.t(),
    role: atom(),
    entity_type: atom() | nil,
    confidence: float()
  }

  @type temporal_relation :: %{
    target_event_index: non_neg_integer(),
    relation: :before | :after | :during
  }

  @doc """
  Link event triggers to their arguments and temporal relations.

  ## Parameters
    - `events` - List of detected events with trigger info
    - `entities` - List of extracted entities from the same text
    - `tokens` - List of token structs from Tokenizer
    - `pos_tags` - List of POS tags corresponding to tokens

  ## Returns
    List of enriched event frames with argument roles and temporal relations.
  """
  def link(events, entities, tokens, pos_tags, opts \\ []) do
    ref_date = Keyword.get(opts, :reference_date)

    # Associate temporal entities to events by document-order rank: both events
    # and temporal entities occur left-to-right, so the k-th date belongs to the
    # k-th event. This avoids the old all-to-all assignment (which gave every
    # event every date) without mixing char offsets and token indices.
    temporal_sorted = Enum.sort_by(Enum.filter(entities, &temporal_entity?/1), &entity_position/1)

    events
    |> Enum.with_index()
    |> Enum.map(fn {event, idx} ->
      arguments = assign_argument_roles(event, entities, tokens, pos_tags)
      temporal_args = temporal_arg_for_index(temporal_sorted, idx)

      %{
        trigger: get_trigger_text(event),
        trigger_index: idx,
        tense: get_tense(event),
        arguments: arguments ++ temporal_args,
        temporal_relations: [],
        sub_events: []
      }
    end)
    |> detect_temporal_relations(ref_date)
    |> detect_sub_events(ref_date)
  end

  @doc """
  Assign semantic roles to entities near an event trigger using the
  trained `:event_argument_role` MicroClassifier.

  Falls back to entity type heuristics if the classifier is not loaded.
  """
  def assign_argument_roles(event, entities, tokens, pos_tags) do
    trigger_text = get_trigger_text(event)
    trigger_idx = get_trigger_index(event)

    entities
    |> Enum.reject(&temporal_entity?/1)
    |> Enum.map(fn entity ->
      role = classify_argument_role(entity, trigger_text, trigger_idx, tokens, pos_tags)
      %{
        text: entity_text(entity),
        role: role.label,
        entity_type: entity_type(entity),
        confidence: role.confidence
      }
    end)
  end

  defp classify_argument_role(entity, trigger_text, trigger_idx, tokens, pos_tags) do
    entity_text_str = entity_text(entity)
    entity_pos = entity_position(entity)
    entity_type_str = entity_type(entity) |> to_string()

    relative_pos = cond do
      entity_pos < trigger_idx -> "before"
      entity_pos > trigger_idx -> "after"
      true -> "at"
    end

    pos_context = pos_tags
    |> Enum.at(min(entity_pos, length(pos_tags) - 1), "UNK")
    |> to_string()

    token_count = length(tokens)
    window_start = max(entity_pos - 2, 0) |> min(max(token_count - 1, 0))
    window_end = min(entity_pos + 2, max(token_count - 1, 0))
    surrounding = if window_start <= window_end and token_count > 0 do
      tokens
      |> Enum.slice(window_start..window_end)
      |> Enum.map(fn
        %{text: t} -> t
        t when is_binary(t) -> t
        other -> to_string(other)
      end)
      |> Enum.join(" ")
    else
      ""
    end

    input_text = "#{entity_text_str} #{entity_type_str} #{relative_pos} #{trigger_text} #{pos_context} #{surrounding}"

    case MicroClassifiers.classify(:event_argument_role, input_text) do
      {:ok, label, confidence} ->
        %{label: normalize_role(label), confidence: confidence}

      {:error, _} ->
        log_once(:event_arg_role_not_ready, "event_argument_role classifier not ready, skipping role assignment")
        :telemetry.execute([:brain, :model, :unavailable], %{}, %{
          model: :event_argument_role,
          reason: :not_ready
        })
        %{label: :arg1, confidence: 0.3}
    end
  end

  # The temporal entity for this event's document-order rank (or none). Real
  # entities carry string types ("date", "time", …); the old code matched the
  # atom :temporal, which never matched, so temporal args were never extracted.
  defp temporal_arg_for_index(temporal_sorted, idx) do
    case Enum.at(temporal_sorted, idx) do
      nil ->
        []

      entity ->
        [%{text: entity_text(entity), role: :argm_tmp, entity_type: :temporal, confidence: 0.8}]
    end
  end

  defp temporal_entity?(entity) do
    type = entity |> entity_type() |> to_string() |> String.downcase()
    type in @temporal_types
  end

  # Only frames that carry a temporal argument get relations. Ordering prefers
  # real resolved dates (TemporalResolver), then grammatical tense, then trigger
  # position — never bare array index as the primary signal.
  defp detect_temporal_relations(event_frames, ref_date) do
    Enum.map(event_frames, fn frame ->
      relations =
        event_frames
        |> Enum.reject(&(&1.trigger_index == frame.trigger_index))
        |> Enum.flat_map(fn other ->
          case infer_temporal_order(frame, other, ref_date) do
            nil -> []
            rel -> [rel]
          end
        end)

      %{frame | temporal_relations: relations}
    end)
  end

  defp infer_temporal_order(frame, other, ref_date) do
    frame_temp = temporal_text(frame)

    if is_nil(frame_temp) do
      nil
    else
      relation =
        case resolver_order(frame_temp, temporal_text(other), ref_date) do
          order when order in [:before, :after] ->
            order

          # Dates equal or unresolvable (incl. the all-to-all argm_tmp case where
          # both frames share the same temporal text): let grammatical tense —
          # then position — decide, rather than collapsing everything to :during.
          _ ->
            tense_or_position(frame, other)
        end

      %{target_event_index: other.trigger_index, relation: relation}
    end
  end

  defp resolver_order(_frame_temp, nil, _ref_date), do: :unknown

  defp resolver_order(frame_temp, other_temp, ref_date),
    do: TemporalResolver.order(frame_temp, other_temp, ref_date)

  # Grammatical fallback: past < present/imperative < future; then token position.
  defp tense_or_position(frame, other) do
    fr = tense_rank(Map.get(frame, :tense))
    ot = tense_rank(Map.get(other, :tense))

    cond do
      is_integer(fr) and is_integer(ot) and fr < ot -> :before
      is_integer(fr) and is_integer(ot) and fr > ot -> :after
      frame.trigger_index < other.trigger_index -> :before
      frame.trigger_index > other.trigger_index -> :after
      true -> :during
    end
  end

  defp tense_rank(:past), do: 0
  defp tense_rank(:present), do: 1
  defp tense_rank(:imperative), do: 1
  defp tense_rank(:infinitive), do: 1
  defp tense_rank(:future), do: 2
  defp tense_rank(_), do: nil

  # A frame is a sub-event of another when its temporal span is contained by the
  # other's (real interval containment via the resolver, e.g. "on Jan 3 2024"
  # inside "in 2024").
  defp detect_sub_events(event_frames, ref_date) do
    Enum.map(event_frames, fn frame ->
      outer = temporal_text(frame)

      sub =
        event_frames
        |> Enum.reject(&(&1.trigger_index == frame.trigger_index))
        |> Enum.filter(fn other ->
          inner = temporal_text(other)
          is_binary(outer) and is_binary(inner) and
            TemporalResolver.contains?(outer, inner, ref_date)
        end)
        |> Enum.map(& &1.trigger_index)

      %{frame | sub_events: sub}
    end)
  end

  # The first temporal (argm_tmp) argument's text, or nil.
  defp temporal_text(frame) do
    case Enum.find(frame.arguments, &(&1.role == :argm_tmp)) do
      %{text: t} when is_binary(t) and t != "" -> t
      _ -> nil
    end
  end

  defp get_tense(%{action: %{tense: tense}}) when is_atom(tense), do: tense
  defp get_tense(%{"action" => %{"tense" => tense}}) when is_binary(tense), do: String.to_atom(tense)
  defp get_tense(_), do: :unknown

  # --- Entity accessor helpers (handle both struct and map formats) ---

  defp entity_text(%{text: text}), do: text
  defp entity_text(%{value: value}), do: value
  defp entity_text(%{"text" => text}), do: text
  defp entity_text(_), do: ""

  defp entity_type(%{type: type}), do: type
  defp entity_type(%{entity_type: type}), do: type
  defp entity_type(%{"type" => type}), do: String.to_atom(type)
  defp entity_type(_), do: :unknown

  defp entity_position(%{start_pos: pos}), do: pos
  defp entity_position(%{token_index: idx}), do: idx
  defp entity_position(%{"start_pos" => pos}), do: pos
  defp entity_position(_), do: 0

  defp get_trigger_text(%{action: %{verb: verb}}), do: verb
  defp get_trigger_text(%{trigger: trigger}) when is_binary(trigger), do: trigger
  defp get_trigger_text(%{"trigger" => trigger}), do: trigger
  defp get_trigger_text(%{"action" => %{"verb" => verb}}), do: verb
  defp get_trigger_text(_), do: ""

  defp get_trigger_index(%{source_tokens: [_ | [idx | _]]}), do: idx
  defp get_trigger_index(%{trigger_index: idx}), do: idx
  defp get_trigger_index(_), do: 0

  defp normalize_role("arg0"), do: :arg0
  defp normalize_role("ARG0"), do: :arg0
  defp normalize_role("arg1"), do: :arg1
  defp normalize_role("ARG1"), do: :arg1
  defp normalize_role("argm_tmp"), do: :argm_tmp
  defp normalize_role("ARGM-TMP"), do: :argm_tmp
  defp normalize_role("argm_loc"), do: :argm_loc
  defp normalize_role("ARGM-LOC"), do: :argm_loc
  defp normalize_role("argm_mnr"), do: :argm_mnr
  defp normalize_role("ARGM-MNR"), do: :argm_mnr
  defp normalize_role(other) when is_binary(other), do: String.to_atom(String.downcase(other))
  defp normalize_role(other) when is_atom(other), do: other

  defp log_once(key, message) do
    pt_key = {__MODULE__, :logged, key}
    unless :persistent_term.get(pt_key, false) do
      Logger.warning(message)
      :persistent_term.put(pt_key, true)
    end
  end
end
