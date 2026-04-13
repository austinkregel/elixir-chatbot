defmodule Brain.Analysis.FollowupDetector do
  @moduledoc "Detects when a user message is providing follow-up context\nfor a previous intent rather than starting a new conversation.\n\nThis module helps handle multi-turn conversations where users provide\nadditional information (like location) in response to clarifying questions.\n\nUses POS tagging for grammatical detection rather than keyword lists.\n"

  alias Brain.ML.POSTagger
  alias Brain.Analysis.IntentRegistry
  @max_followup_words 5
  @context_timeout_ms 5 * 60 * 1000

  @doc "Determines if the given text is a follow-up to a previous intent.\n\nReturns true if:\n- Previous context exists and has missing slots\n- The message looks like it's providing slot values\n- The message is short and starts with a preposition\n- The message looks like a bare location name\n"
  def is_followup?(text, previous_context) do
    cond do
      is_nil(previous_context) ->
        false

      context_expired?(previous_context) ->
        false

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

  defp fill_slots_from_entities(missing_slots, entities, existing_slots, intent) do
    slot_mappings = IntentRegistry.entity_mappings(intent)

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
end
