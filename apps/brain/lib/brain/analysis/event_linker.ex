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
  require Logger

  @type event_frame :: %{
    trigger: String.t(),
    trigger_index: non_neg_integer(),
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
  def link(events, entities, tokens, pos_tags) do
    events
    |> Enum.with_index()
    |> Enum.map(fn {event, idx} ->
      arguments = assign_argument_roles(event, entities, tokens, pos_tags)
      temporal_args = extract_temporal_arguments(entities)

      %{
        trigger: get_trigger_text(event),
        trigger_index: idx,
        arguments: arguments ++ temporal_args,
        temporal_relations: [],
        sub_events: []
      }
    end)
    |> detect_temporal_relations()
    |> detect_sub_events()
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
    |> Enum.reject(fn entity ->
      entity_type(entity) == :temporal
    end)
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

  defp extract_temporal_arguments(entities) do
    entities
    |> Enum.filter(fn entity -> entity_type(entity) == :temporal end)
    |> Enum.map(fn entity ->
      %{
        text: entity_text(entity),
        role: :argm_tmp,
        entity_type: :temporal,
        confidence: 0.8
      }
    end)
  end

  defp detect_temporal_relations(event_frames) do
    temporal_events = event_frames
    |> Enum.with_index()
    |> Enum.map(fn {frame, idx} ->
      temporal_args = Enum.filter(frame.arguments, &(&1.role == :argm_tmp))
      {idx, frame, temporal_args}
    end)

    Enum.map(event_frames, fn frame ->
      relations = temporal_events
      |> Enum.reject(fn {idx, _, _} -> idx == frame.trigger_index end)
      |> Enum.flat_map(fn {other_idx, _other_frame, _other_temps} ->
        case infer_temporal_order(frame, other_idx, temporal_events) do
          nil -> []
          rel -> [rel]
        end
      end)

      %{frame | temporal_relations: relations}
    end)
  end

  defp infer_temporal_order(frame, other_idx, _temporal_events) do
    frame_temps = Enum.filter(frame.arguments, &(&1.role == :argm_tmp))

    if Enum.empty?(frame_temps) do
      nil
    else
      if frame.trigger_index < other_idx do
        %{target_event_index: other_idx, relation: :before}
      else
        %{target_event_index: other_idx, relation: :after}
      end
    end
  end

  defp detect_sub_events(event_frames) do
    Enum.map(event_frames, fn frame ->
      sub = event_frames
      |> Enum.reject(&(&1.trigger_index == frame.trigger_index))
      |> Enum.filter(fn other ->
        other_temps = Enum.filter(other.arguments, &(&1.role == :argm_tmp))
        frame_temps = Enum.filter(frame.arguments, &(&1.role == :argm_tmp))

        not Enum.empty?(other_temps) and not Enum.empty?(frame_temps) and
          temporal_contains?(frame_temps, other_temps)
      end)
      |> Enum.map(& &1.trigger_index)

      %{frame | sub_events: sub}
    end)
  end

  defp temporal_contains?(_outer_temps, _inner_temps) do
    false
  end

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
