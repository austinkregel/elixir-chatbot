defmodule Brain.Analysis.FollowupDetector do
  @moduledoc "Detects when a user message is providing follow-up context\nfor a previous intent rather than starting a new conversation.\n\nThis module helps handle multi-turn conversations where users provide\nadditional information (like location) in response to clarifying questions.\n\nUses POS tagging for grammatical detection rather than keyword lists.\n"

  alias Brain.ML.POSTagger
  alias Brain.Analysis.ChunkProfile
  @max_followup_words 5
  @context_timeout_ms 5 * 60 * 1000

  @doc "Determines if the given text is a follow-up to a previous intent.\n\nReturns true if:\n- Previous context exists and has missing slots\n- The message looks like it's providing slot values\n- The message is short and starts with a preposition\n- The message looks like a bare location name\n"
  def is_followup?(text, previous_context) do
    cond do
      is_nil(previous_context) ->
        false

      context_expired?(previous_context) ->
        false

      answers_pending_clarification?(text, previous_context) ->
        true

      has_missing_slots?(previous_context) and looks_like_slot_filler?(text, previous_context) ->
        true

      is_short_prepositional_phrase?(text) and has_recent_intent?(previous_context) ->
        true

      is_bare_location?(text) and needs_location?(previous_context) ->
        true

      true ->
        false
    end
  end

  @doc """
  What the conversation waits for after a turn whose service returned an
  error, read from the turn's enrichment (`Brain.Response.ContextBuilder`):
  the context fields to carry into the turn's snapshot, or `%{}` when the
  reply asked nothing.

  The reply planned for the error
  (`Brain.Response.DiscoursePlanner.enrichment_error_primitive/2`) decides:

  - a clarification listing candidates (`:ambiguous_item`) waits for the user
    to pick one: `:pending_clarification` holds the item kind and the
    candidates, with the intent and slots the service was asked with;
  - a clarification naming the slots that answer it (`missing_slots`) waits
    for those slots, as a turn missing required slots does: the intent, the
    slots asked with less those, and `:missing_slots`;
  - any other reply waits for nothing.
  """
  @spec awaiting_answer(map()) :: map()
  def awaiting_answer(%{enrichment_status: :failed, enrichment_error: reason} = enrichment) do
    intent = Map.get(enrichment, :intent)
    slots = Map.new(Map.get(enrichment, :slots) || %{}, fn {slot, value} -> {to_string(slot), value} end)

    planned =
      Brain.Response.DiscoursePlanner.enrichment_error_primitive(reason, %{
        service: Map.get(enrichment, :enrichment_service),
        intent: intent
      })

    case planned do
      %{type: :follow_up, content: %{reply: :ambiguous_item, item_kind: kind, candidates: candidates}} ->
        %{
          intent: intent,
          slots: slots,
          missing_slots: [],
          pending_clarification: %{item_kind: kind, candidates: candidates}
        }

      %{type: :follow_up, content: %{missing_slots: [_ | _] = missing}} ->
        %{intent: intent, slots: Map.drop(slots, missing), missing_slots: missing}

      _ ->
        %{}
    end
  end

  def awaiting_answer(_enrichment), do: %{}

  @doc """
  Reads a reply against the candidates a clarification listed
  (`awaiting_answer/1`'s `:pending_clarification`).

  A candidate is described by its fields (`Brain.Response.Synthesizer.service_item_slots/2`);
  a field counts when every token of its value is among the reply's tokens
  (`Brain.ML.Tokenizer.tokenize_normalized/1`), and the candidates with the
  most fields counted are the ones the reply names. One is
  `{:resolved, candidate}`; fewer than were listed is `{:narrowed, candidates}`;
  none, or all of them, is `:unrelated`: the reply does not tell them apart.
  Only whole field values are recognized; a reply naming a candidate by part
  of its name, by position ("the first one") or by anything but its stored
  values is `:unrelated`.
  """
  @spec resolve_pending(String.t(), map()) :: {:resolved, map()} | {:narrowed, [map()]} | :unrelated
  def resolve_pending(text, %{item_kind: kind, candidates: candidates}) when is_binary(text) do
    reply_tokens = text |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()

    scored =
      Enum.map(candidates, fn candidate ->
        named =
          kind
          |> Brain.Response.Synthesizer.service_item_slots(candidate)
          |> Enum.count(fn {_slot, value} ->
            value_tokens = value |> Brain.ML.Tokenizer.tokenize_normalized() |> MapSet.new()
            MapSet.size(value_tokens) > 0 and MapSet.subset?(value_tokens, reply_tokens)
          end)

        {candidate, named}
      end)

    best = scored |> Enum.map(&elem(&1, 1)) |> Enum.max(fn -> 0 end)
    named = for {candidate, ^best} <- scored, best > 0, do: candidate

    cond do
      named == [] -> :unrelated
      length(named) == 1 -> {:resolved, hd(named)}
      length(named) < length(candidates) -> {:narrowed, named}
      true -> :unrelated
    end
  end

  @doc """
  The slots that complete the pending request with the candidate the reply
  picked: the slots it was asked with, and the candidate's own
  (`Brain.Response.Synthesizer.service_item_slots/2`) over them.
  """
  @spec complete_pending(map(), map()) :: map()
  def complete_pending(previous_context, candidate) do
    %{item_kind: kind} = Map.fetch!(previous_context, :pending_clarification)

    Map.merge(
      Map.get(previous_context, :slots) || %{},
      Brain.Response.Synthesizer.service_item_slots(kind, candidate)
    )
  end

  defp answers_pending_clarification?(text, previous_context) do
    case Map.get(previous_context, :pending_clarification) do
      %{} = pending -> resolve_pending(text, pending) != :unrelated
      nil -> false
    end
  end

  @doc "Returns the previous context with information about what to carry forward.\n"
  def get_carried_context(text, previous_context) do
    %{
      intent: previous_context.intent,
      carry_forward: true,
      new_input: text,
      missing_slots: previous_context.missing_slots || [],
      previous_entities: previous_context.entities || [],
      previous_slots: previous_context.slots || %{},
      timestamp: previous_context.timestamp
    }
  end

  @doc "Merges new entities with previous context to fill missing slots.\n"
  def merge_with_previous(previous_context, new_entities) do
    merged_entities = (previous_context.previous_entities || []) ++ new_entities

    intent = previous_context.intent

    {filled_slots, remaining_missing} =
      fill_slots_from_entities(
        previous_context.missing_slots || [],
        new_entities,
        previous_context.previous_slots || %{},
        intent
      )

    %{
      intent: previous_context.intent,
      entities: merged_entities,
      slots: filled_slots,
      missing_slots: remaining_missing,
      all_required_filled: remaining_missing == []
    }
  end

  defp is_short_prepositional_phrase?(text) do
    words = String.split(String.trim(text))
    length(words) <= @max_followup_words and starts_with_preposition?(words)
  end

  defp starts_with_preposition?(words) do
    case words do
      [] ->
        false

      _ ->
        case POSTagger.load_model() do
          {:ok, model} ->
            predictions = POSTagger.predict(Enum.take(words, 2), model)

            case predictions do
              [{_word, "ADP"} | _] -> true
              _ -> false
            end

          {:error, _} ->
            false
        end
    end
  end

  defp is_bare_location?(text) do
    words = String.split(String.trim(text))

    length(words) in 1..3 and
      Enum.all?(words, &starts_with_capital?/1) and
      not contains_verb?(words)
  end

  defp starts_with_capital?(word) do
    first_char = String.first(word)

    first_char != nil and
      first_char == String.upcase(first_char) and
      first_char != String.downcase(first_char)
  end

  defp contains_verb?(words) do
    case POSTagger.load_model() do
      {:ok, model} ->
        predictions = POSTagger.predict(words, model)
        Enum.any?(predictions, fn {_word, tag} -> tag in ["VERB", "AUX"] end)

      {:error, _} ->
        false
    end
  end

  defp needs_location?(context) do
    missing = context[:missing_slots] || context.missing_slots || []
    "location" in missing
  end

  defp has_missing_slots?(context) do
    missing = context[:missing_slots] || context.missing_slots || []
    missing != []
  end

  defp has_recent_intent?(context) do
    intent = context[:intent] || context.intent
    intent != nil and intent != "" and intent != "unknown"
  end

  defp looks_like_slot_filler?(text, context) do
    missing = context[:missing_slots] || context.missing_slots || []

    cond do
      "location" in missing and (is_bare_location?(text) or is_short_prepositional_phrase?(text)) ->
        true

      length(String.split(String.trim(text))) <= @max_followup_words ->
        true

      true ->
        false
    end
  end

  defp context_expired?(context) do
    now = System.system_time(:millisecond)
    timestamp = context[:timestamp] || context.timestamp || 0
    now - timestamp > @context_timeout_ms
  end

  defp fill_slots_from_entities(missing_slots, entities, existing_slots, intent, profile \\ nil) do
    slot_mappings =
      case profile do
        %ChunkProfile{domain: domain} when domain != :unknown ->
          entity_mappings_from_domain(domain)

        _ ->
          entity_mappings_from_intent(intent)
      end

    Enum.reduce(missing_slots, {existing_slots, []}, fn slot_name, {filled, still_missing} ->
      matching_entity_types = Map.get(slot_mappings, slot_name, [slot_name])

      matching_entity =
        Enum.find(entities, fn entity ->
          entity[:entity_type] in matching_entity_types
        end)

      case matching_entity do
        nil ->
          {filled, [slot_name | still_missing]}

        entity ->
          value = entity[:value] || entity["value"]
          {Map.put(filled, slot_name, value), still_missing}
      end
    end)
    |> then(fn {filled, still_missing} -> {filled, Enum.reverse(still_missing)} end)
  end

  defp entity_mappings_from_intent(intent) when is_binary(intent) do
    domain = case String.split(intent, ".", parts: 2) do
      [d, _] -> String.to_atom(d)
      _ -> :unknown
    end
    entity_mappings_from_domain(domain)
  end
  defp entity_mappings_from_intent(_), do: %{}

  defp entity_mappings_from_domain(domain) do
    case domain do
      :weather -> %{"location" => ["location"], "date" => ["date"], "time" => ["time"]}
      :music -> %{"song" => ["song"], "artist" => ["artist", "person"], "genre" => ["genre"]}
      :smarthome -> %{"device" => ["device"], "room" => ["room"], "setting" => ["setting"]}
      :reminder -> %{"date" => ["date"], "time" => ["time"], "description" => ["description"]}
      :navigation -> %{"destination" => ["location"], "origin" => ["location"]}
      _ -> %{}
    end
  end
end
