defmodule Brain.Analysis.FollowupDetector do
  @moduledoc "Detects when a user message is providing follow-up context\nfor a previous intent rather than starting a new conversation.\n\nThis module helps handle multi-turn conversations where users provide\nadditional information (like location) in response to clarifying questions.\n\nUses POS tagging for grammatical detection rather than keyword lists.\n"

  alias Brain.ML.POSTagger
  alias Brain.Analysis.ChunkProfile
  @max_followup_words 5
  @context_timeout_ms 5 * 60 * 1000

  @doc """
  Determines if the given text is a follow-up to a previous intent.

  Returns true, when the previous context has not expired, if:
  - the previous reply asked the user to pick one of several candidates
    (`awaiting_answer/1`'s `:pending_clarification`): the message is read
    against them first, with the entities extracted from it
    (`resolve_pending/2`), and taken as a new message when it names none
  - the previous context has missing slots and the message looks like it
    provides slot values
  - the message is short and starts with a preposition
  - the message looks like a bare location name
  """
  def is_followup?(text, previous_context) do
    cond do
      is_nil(previous_context) ->
        false

      context_expired?(previous_context) ->
        false

      Map.get(previous_context, :pending_clarification) != nil ->
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
    to pick one: `:pending_clarification` holds the item kind, the
    candidates, the service that named them and the intent it was asked
    for, and the context keeps the intent and slots the service was asked
    with;
  - a clarification naming the slots that answer it (`missing_slots`) waits
    for those slots, as a turn missing required slots does: the intent, the
    slots asked with less those, and `:missing_slots`;
  - any other reply waits for nothing.
  """
  @spec awaiting_answer(map()) :: map()
  def awaiting_answer(%{enrichment_status: :failed, enrichment_error: reason} = enrichment) do
    intent = Map.get(enrichment, :intent)
    service = Map.get(enrichment, :enrichment_service)
    slots = Map.new(Map.get(enrichment, :slots) || %{}, fn {slot, value} -> {to_string(slot), value} end)

    planned = Brain.Response.DiscoursePlanner.enrichment_error_primitive(reason, %{service: service, intent: intent})

    case planned do
      %{type: :follow_up, content: %{reply: :ambiguous_item, item_kind: kind, candidates: candidates}} ->
        %{
          intent: intent,
          slots: slots,
          missing_slots: [],
          pending_clarification: %{item_kind: kind, candidates: candidates, service: service, intent: intent}
        }

      %{type: :follow_up, content: %{missing_slots: [_ | _] = missing}} ->
        %{intent: intent, slots: Map.drop(slots, missing), missing_slots: missing}

      _ ->
        %{}
    end
  end

  def awaiting_answer(_enrichment), do: %{}

  @doc """
  Reads a reply, its `text` and the `entities` extracted from it, against
  the candidates a clarification listed (`awaiting_answer/1`'s
  `:pending_clarification`).

  A candidate is told apart by the fields its driver identifies an item of
  its kind by (`Brain.Services.SchedulingDriver.identifying_fields!/2`). The
  reply names a field by:

  - its words: the reply's tokens (`Brain.ML.Tokenizer.tokenize_normalized/1`)
    that are among the field value's, so part of a value names it
    ("dentist" names "Dentist appointment"). Closed-class words
    (`Brain.Lexicon.ClosedClass`) and tokens every candidate's fields hold
    tell no candidate apart, and are not counted;
  - an entity: an entity of the reply filling the field's slot for the
    pending intent (`Brain.Analysis.SlotDetector.detect/3`) with the field's
    value, compared with case and whitespace ignored, so a time entity
    "3pm" names the time "3 pm".

  A field scores the more of the two; a candidate scores the sum over its
  fields. The candidates with the top score above zero are the ones the reply
  names: one is `{:resolved, candidate}`; fewer than were listed is
  `{:narrowed, candidates}`, to ask again among them; none, or all of them,
  is `:unrelated`, since the reply tells none apart.

  A reply naming a candidate by position ("the first one", "the last one")
  is `:unrelated`: no owned lexicon records which words are ordinals or what
  position each names.

  Raises when the pending clarification's service is not a scheduling driver.
  """
  @spec resolve_pending(%{text: String.t(), entities: [map()]}, map()) ::
          {:resolved, map()} | {:narrowed, [map()]} | :unrelated
  def resolve_pending(%{text: text, entities: entities}, %{
        item_kind: kind,
        candidates: [_ | _] = candidates,
        service: service,
        intent: intent
      })
      when is_binary(text) and is_list(entities) and is_binary(intent) do
    fields = Brain.Services.SchedulingDriver.identifying_fields!(driver!(service), kind)
    described = Enum.map(candidates, &described_fields(&1, fields))

    shared =
      case Enum.map(described, fn fields -> fields |> Enum.flat_map(&elem(&1, 2)) |> MapSet.new() end) do
        [_only] -> MapSet.new()
        token_sets -> Enum.reduce(token_sets, &MapSet.intersection/2)
      end

    reply_tokens =
      text
      |> Brain.ML.Tokenizer.tokenize_normalized()
      |> Enum.reject(&Brain.Lexicon.ClosedClass.member?/1)
      |> MapSet.new()
      |> MapSet.difference(shared)

    reply_slots = reply_slots(intent, entities)

    scored =
      candidates
      |> Enum.zip(described)
      |> Enum.map(fn {candidate, fields} ->
        score =
          fields
          |> Enum.map(fn {slot, value, tokens} ->
            by_words = tokens |> MapSet.new() |> MapSet.intersection(reply_tokens) |> MapSet.size()
            by_entity = if same_value?(Map.get(reply_slots, slot), value), do: 1, else: 0
            max(by_words, by_entity)
          end)
          |> Enum.sum()

        {candidate, score}
      end)

    best = scored |> Enum.map(&elem(&1, 1)) |> Enum.max()
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
  (`Brain.Services.SchedulingDriver.item_slots/3`) over them.
  """
  @spec complete_pending(map(), map()) :: map()
  def complete_pending(previous_context, candidate) do
    %{item_kind: kind, service: service} = Map.fetch!(previous_context, :pending_clarification)

    Map.merge(
      Map.get(previous_context, :slots) || %{},
      Brain.Services.SchedulingDriver.item_slots(driver!(service), kind, candidate)
    )
  end

  # The driver a pending clarification's candidates came from.
  defp driver!(service) do
    case is_atom(service) and not is_nil(service) and Brain.Services.Dispatcher.find_service_by_name(service) do
      module when is_atom(module) and module not in [nil, false] ->
        module

      _not_found ->
        raise ArgumentError,
              "FollowupDetector: the candidates a clarification listed came from #{inspect(service)}, " <>
                "which is no registered service"
    end
  end

  # Each identifying field the candidate gives a non-blank value, as the slot
  # naming it, the value and the value's tokens.
  defp described_fields(candidate, fields) do
    Enum.flat_map(fields, fn {field, slot} ->
      case Map.get(candidate, field) do
        value when is_binary(value) ->
          if String.trim(value) == "",
            do: [],
            else: [{Atom.to_string(slot), value, Brain.ML.Tokenizer.tokenize_normalized(value)}]

        nil ->
          []

        other ->
          raise ArgumentError,
                "FollowupDetector: a candidate's #{field} must be a string to be matched, got: #{inspect(other)}"
      end
    end)
  end

  # The slots the reply's entities fill for the pending intent, each with the
  # value an entity gave it; a slot filled by a default is not the reply's.
  defp reply_slots(_intent, []), do: %{}

  defp reply_slots(intent, entities) do
    intent
    |> Brain.Analysis.SlotDetector.detect(entities, side_effects: false)
    |> Map.fetch!(:filled_slots)
    |> Enum.flat_map(fn
      {slot, %{source: source, value: value}} when source in [:explicit, :type_compatible] and is_binary(value) ->
        [{slot, value}]

      _other ->
        []
    end)
    |> Map.new()
  end

  defp same_value?(nil, _value), do: false
  defp same_value?(given, value), do: compact(given) == compact(value)

  defp compact(text), do: text |> String.split() |> Enum.join() |> String.downcase()

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
