defmodule ChatBot.Analysis.FollowupDetector do
  @moduledoc """
  Detects when a user message is providing follow-up context
  for a previous intent rather than starting a new conversation.

  This module helps handle multi-turn conversations where users provide
  additional information (like location) in response to clarifying questions.
  """

  # Prepositions that commonly introduce slot-filling responses
  @location_prepositions ~w(in at for near around)

  # Maximum word count for a likely follow-up message
  @max_followup_words 5

  # Time window for considering previous context (5 minutes)
  @context_timeout_ms 5 * 60 * 1000

  @doc """
  Determines if the given text is a follow-up to a previous intent.

  Returns true if:
  - Previous context exists and has missing slots
  - The message looks like it's providing slot values
  - The message is short and starts with a preposition
  - The message looks like a bare location name
  """
  def is_followup?(text, previous_context) do
    cond do
      # No previous context - can't be a follow-up
      is_nil(previous_context) ->
        false

      # Context too old
      context_expired?(previous_context) ->
        false

      # Previous intent had missing required slots and this looks like a slot filler
      has_missing_slots?(previous_context) and looks_like_slot_filler?(text, previous_context) ->
        true

      # Short message that starts with a preposition (likely providing context)
      is_short_prepositional_phrase?(text) and has_recent_intent?(previous_context) ->
        true

      # Looks like a bare location (capitalized words, no verb)
      is_bare_location?(text) and needs_location?(previous_context) ->
        true

      true ->
        false
    end
  end

  @doc """
  Returns the previous context with information about what to carry forward.
  """
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

  @doc """
  Merges new entities with previous context to fill missing slots.
  """
  def merge_with_previous(previous_context, new_entities) do
    # Combine previous entities with new ones
    merged_entities = (previous_context.previous_entities || []) ++ new_entities

    # Try to fill missing slots with new entities
    {filled_slots, remaining_missing} =
      fill_slots_from_entities(
        previous_context.missing_slots || [],
        new_entities,
        previous_context.previous_slots || %{}
      )

    %{
      intent: previous_context.intent,
      entities: merged_entities,
      slots: filled_slots,
      missing_slots: remaining_missing,
      all_required_filled: remaining_missing == []
    }
  end

  # Private functions

  defp is_short_prepositional_phrase?(text) do
    words = String.split(String.trim(text))
    first_word = words |> List.first() |> to_string() |> String.downcase()

    length(words) <= @max_followup_words and first_word in @location_prepositions
  end

  defp is_bare_location?(text) do
    words = String.split(String.trim(text))

    # 1-3 words, all starting with capital letters, no common verbs
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
    common_verbs =
      ~w(is are was were do does did can could will would should have has had am be been)

    Enum.any?(words, fn w -> String.downcase(w) in common_verbs end)
  end

  defp needs_location?(context) do
    missing = context[:missing_slots] || context.missing_slots || []
    "location" in missing
  end

  defp has_missing_slots?(context) do
    missing = context[:missing_slots] || context.missing_slots || []
    length(missing) > 0
  end

  defp has_recent_intent?(context) do
    intent = context[:intent] || context.intent
    intent != nil and intent != "" and intent != "unknown"
  end

  defp looks_like_slot_filler?(text, context) do
    # Check if this text could fill one of the missing slots
    missing = context[:missing_slots] || context.missing_slots || []

    cond do
      # If location is missing and text looks like a location
      "location" in missing and (is_bare_location?(text) or is_short_prepositional_phrase?(text)) ->
        true

      # If the text is short (likely an answer to a question)
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

  defp fill_slots_from_entities(missing_slots, entities, existing_slots) do
    # Entity type to slot name mappings
    slot_mappings = %{
      "location" => ["location", "room", "city", "place-name", "geo-location"],
      "date" => ["date", "relative_date", "sys-date"],
      "time" => ["time", "sys-time"],
      "device" => ["device", "lights", "heating"],
      "number" => ["number", "value", "temperature"]
    }

    Enum.reduce(missing_slots, {existing_slots, []}, fn slot_name, {filled, still_missing} ->
      # Find entity types that can fill this slot
      matching_entity_types = Map.get(slot_mappings, slot_name, [slot_name])

      # Look for an entity that matches
      matching_entity =
        Enum.find(entities, fn entity ->
          entity[:entity_type] in matching_entity_types
        end)

      case matching_entity do
        nil ->
          # No matching entity, slot still missing
          {filled, [slot_name | still_missing]}

        entity ->
          # Found matching entity, fill the slot
          value = entity[:value] || entity["value"]
          {Map.put(filled, slot_name, value), still_missing}
      end
    end)
    |> then(fn {filled, still_missing} -> {filled, Enum.reverse(still_missing)} end)
  end
end
