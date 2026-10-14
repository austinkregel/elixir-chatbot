defmodule Brain.Analysis.SlotDetector do
  @moduledoc "Detects required and optional slots for a given intent and fills them from entities.\n\nThis module:\n- Loads slot schemas from JSON configuration\n- Maps extracted entities to slots\n- Identifies missing required slots\n- Applies default values where configured\n- Provides clarification prompts for missing slots\n"

  alias Brain.Analysis.{FreeTextCapture, SlotResult, TypeHierarchy}

  require Logger

  @schemas_path "analysis/intent_registry.json"

  @doc """
  Detects slots for the given intent and fills them from entities, and its
  free-text slots from the utterance.

  Returns a SlotResult struct indicating which slots are filled and which are missing.

  A free-text slot (a reminder's text) is one `Brain.Analysis.FreeTextCapture`
  captures: no entity type names its value, so it is a span of the utterance.
  Each slot of the intent's schema that the capture model captures and no
  entity filled is filled with the span it captures, its source `:captured`
  and its confidence the capture's. The span holds no token of an entity that
  filled another slot, nor a date, time or length. When the model abstains the
  slot stays unfilled, among the missing slots.

  Options:
  - `:side_effects` - when a slot is filled through a compatible (not exact)
    entity type, the match is recorded as a Gazetteer type preference. `false`
    skips that write. Defaults to `true`.
  - `:text` - the utterance the entities' `:start_pos` and `:end_pos` index.
    Free-text slots are captured only from it: without it, none is filled.
  - `:language` - the language of `:text`, whose loaded capture model
    (`Brain.Analysis.FreeTextCapture.loaded!/1`) is used. Required with `:text`
    unless `:free_text_capture` is given.
  - `:free_text_capture` - a `Brain.Analysis.FreeTextCapture` model used in
    place of the loaded one.

  With `:text`, every entity that fills a slot must carry its `:start_pos` and
  `:end_pos`, or this raises: a captured span could not be kept off it.
  """
  def detect(intent, entities, opts \\ []) when is_binary(intent) and is_list(entities) do
    schemas = load_schemas()
    record? = Keyword.get(opts, :side_effects, true)
    entities = slot_fillers!(entities)
    utterance = utterance!(opts)

    case Map.get(schemas, intent) do
      nil ->
        parent_intent = get_parent_intent(intent)

        case Map.get(schemas, parent_intent) do
          nil ->
            case find_domain_prefix_schema(intent, entities, schemas) do
              nil -> build_unknown_result(entities)
              {schema, matched_intent} -> process_schema(schema, matched_intent, entities, record?, utterance)
            end

          schema ->
            process_schema(schema, intent, entities, record?, utterance)
        end

      schema ->
        process_schema(schema, intent, entities, record?, utterance)
    end
  end

  # The utterance free-text slots are captured from, with the capture model,
  # or nil when no `:text` was given.
  defp utterance!(opts) do
    case Keyword.fetch(opts, :text) do
      :error ->
        nil

      {:ok, text} when is_binary(text) ->
        model =
          case {Keyword.fetch(opts, :free_text_capture), Keyword.fetch(opts, :language)} do
            {{:ok, %FreeTextCapture{} = model}, _} ->
              model

            {{:ok, other}, _} ->
              raise ArgumentError, "SlotDetector: :free_text_capture must be a FreeTextCapture model, got #{inspect(other, limit: 3)}"

            {:error, {:ok, language}} ->
              FreeTextCapture.loaded!(language)

            {:error, :error} ->
              raise ArgumentError, "SlotDetector: :text needs :language, whose free-text capture model reads it"
          end

        %{text: text, model: model}

      {:ok, other} ->
        raise ArgumentError, "SlotDetector: :text must be the utterance's text, got #{inspect(other, limit: 3)}"
    end
  end

  @doc "Returns the slot schema for a given intent.\n"
  def get_schema(intent) do
    schemas = load_schemas()
    Map.get(schemas, intent) || Map.get(schemas, get_parent_intent(intent))
  end

  @doc "The intent registry as stored, without the service schemas `detect/3` merges in. Raises when it cannot be read.\n"
  def registry, do: load_registry()

  @doc "Returns the set of entity types that can fill slots for a given intent.\n\nThis is useful for filtering entities to only those relevant to the intent.\n"
  def get_entity_types_for_intent(intent) do
    case get_schema(intent) do
      nil ->
        MapSet.new()

      schema ->
        entity_mappings = Map.get(schema, "entity_mappings", %{})

        entity_mappings
        |> Map.values()
        |> List.flatten()
        |> MapSet.new()
    end
  end

  @doc """
  The entity types that can fill a slot of any registry intent in `domain`, a
  consolidated domain as `:intent_domain` and `ChunkProfile.domain` give it
  (`:music`, `:reminder`, ...).

  Intents belong to a domain through `Brain.Lexicon.IntentDomains.consolidate/1`,
  so `alarm.*` and `timer.*` both count toward `:reminder`. Only the registry is
  read, not the service schemas merged into it, because a service's bare-domain
  entries (Home Assistant declares `"music"`) are not intents of that domain.
  """
  def entity_types_for_domain(domain) when is_atom(domain) do
    domain_name = Atom.to_string(domain)

    load_registry()
    |> Enum.filter(fn {intent, _schema} -> Brain.Lexicon.IntentDomains.consolidate(intent) == domain_name end)
    |> Enum.flat_map(fn {_intent, schema} -> schema |> Map.get("entity_mappings", %{}) |> Map.values() end)
    |> List.flatten()
    |> MapSet.new()
  end

  @doc "Lists all available slot schemas.\n"
  def list_schemas do
    load_schemas()
  end

  @doc "Suggests an intent based on entities present.\n\nThis can be used when intent classification has low confidence.\n\nScoring algorithm:\n1. Primary: Count unique entity types that can fill ANY slot (not slots per type)\n2. Tiebreaker: Ratio of required slots that can be filled\n\nThis prevents intents with multiple slots accepting the same type from\nscoring higher (e.g., navigation.directions with destination+origin both\naccepting location should not beat weather.query for a single location entity).\n\nOnly the intents of the registry are suggested. A service's schema answers for\na domain (\"device\", \"smarthome\") through `get_parent_intent/1` and merges into\nthe registry's intents; it is no intent of its own.\n"
  def suggest_intent_from_entities(entities) when is_list(entities) do
    schemas = Map.take(load_schemas(), Map.keys(load_registry()))
    entity_types = Enum.map(entities, fn e -> e[:entity_type] end) |> Enum.uniq()

    scored_schemas =
      schemas
      |> Enum.map(fn {intent, schema} ->
        mappings = Map.get(schema, "entity_mappings", %{})
        required = Map.get(schema, "required", [])

        matched_types =
          entity_types
          |> Enum.count(fn etype ->
            Enum.any?(mappings, fn {_slot, mapped} ->
              etype in mapped or narrowing_match?(etype, mapped)
            end)
          end)

        filled_required =
          Enum.count(required, fn slot ->
            mapped = Map.get(mappings, slot, [])
            Enum.any?(entity_types, fn etype ->
              etype in mapped or narrowing_match?(etype, mapped)
            end)
          end)

        fill_ratio =
          if required != [] do
            filled_required / length(required)
          else
            1.0
          end

        expected = expected_entity_types_from_schema(intent, schema)

        domain_priority =
          if expected != [] do
            Enum.count(entity_types, fn etype ->
              etype in expected or narrowing_match?(etype, expected)
            end)
          else
            0
          end

        {intent, matched_types, fill_ratio, domain_priority}
      end)
      |> Enum.filter(fn {_, score, _, _} -> score > 0 end)
      |> Enum.sort_by(fn {intent, score, ratio, priority} ->
        {-score, -ratio, -priority, intent}
      end)

    case scored_schemas do
      [{intent, score, _, _} | _] when score > 0 -> {:ok, intent, score}
      _ -> {:error, :no_match}
    end
  end

  @doc "Gets clarification prompt for a missing slot from intent_registry.json.\nFalls back to a generic prompt if not defined.\n\n## Examples\n\n    iex> SlotDetector.get_clarification_prompt(\"location\", \"weather.query\")\n    \"What location would you like the weather for?\"\n\n    iex> SlotDetector.get_clarification_prompt(\"unknown_slot\", \"some.intent\")\n    \"Could you please specify the unknown slot?\"\n"
  def get_clarification_prompt(slot_name, intent) when is_binary(slot_name) do
    schemas = load_schemas()
    schema = Map.get(schemas, intent) || Map.get(schemas, get_parent_intent(intent), %{})
    templates = Map.get(schema, "clarification_templates", %{})

    case Map.get(templates, slot_name) do
      nil -> generate_generic_prompt(slot_name)
      prompt -> prompt
    end
  end

  def get_clarification_prompt(slot_name, intent) when is_atom(slot_name) do
    get_clarification_prompt(Atom.to_string(slot_name), intent)
  end

  def get_clarification_prompt(_, _) do
    "Could you please provide more information?"
  end

  @doc "Gets all clarification prompts for a list of missing slots.\n"
  def get_clarification_prompts(missing_slots, intent) when is_list(missing_slots) do
    Enum.map(missing_slots, fn slot ->
      slot_name =
        if is_atom(slot) do
          Atom.to_string(slot)
        else
          slot
        end

      get_clarification_prompt(slot_name, intent)
    end)
  end

  defp generate_generic_prompt(slot_name) do
    readable =
      slot_name
      |> String.replace("-", " ")
      |> String.replace("_", " ")

    "Could you please specify the #{readable}?"
  end

  # The registry is resolved from the app's priv directory, not the working
  # directory, and is required: without it every intent would get only the
  # service schemas, which is how music.play lost its artist slot whenever the
  # app ran from the umbrella root.
  defp load_schemas, do: merge_service_schemas(load_registry())

  defp load_registry do
    path = Brain.priv_path(@schemas_path)

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, schemas} -> schemas
          {:error, reason} -> raise "SlotDetector: cannot parse #{path}: #{inspect(reason)}"
        end

      {:error, reason} ->
        raise "SlotDetector: cannot read intent registry at #{path}: #{inspect(reason)}"
    end
  end

  defp merge_service_schemas(base) do
    service_schemas =
      try do
        Brain.Services.Dispatcher.service_schemas()
      rescue
        _ -> %{}
      end

    Map.merge(base, service_schemas, fn _intent, json_schema, service_schema ->
      deep_merge_schemas(json_schema, service_schema)
    end)
  end

  defp deep_merge_schemas(json, service) when is_map(json) and is_map(service) do
    json
    |> merge_list_field(service, "required")
    |> merge_list_field(service, "optional")
    |> merge_mappings(service)
    |> merge_templates(service)
    |> merge_defaults(service)
  end

  defp deep_merge_schemas(_json, service), do: service

  defp merge_list_field(base, overlay, key) do
    base_list = Map.get(base, key, [])
    overlay_list = Map.get(overlay, key, [])
    Map.put(base, key, Enum.uniq(base_list ++ overlay_list))
  end

  defp merge_mappings(base, overlay) do
    base_mappings = Map.get(base, "entity_mappings", %{})
    overlay_mappings = Map.get(overlay, "entity_mappings", %{})

    merged =
      Map.merge(base_mappings, overlay_mappings, fn _slot, base_types, overlay_types ->
        Enum.uniq(base_types ++ overlay_types)
      end)

    Map.put(base, "entity_mappings", merged)
  end

  defp merge_templates(base, overlay) do
    base_templates = Map.get(base, "clarification_templates", %{})
    overlay_templates = Map.get(overlay, "clarification_templates", %{})
    Map.put(base, "clarification_templates", Map.merge(base_templates, overlay_templates))
  end

  defp merge_defaults(base, overlay) do
    base_defaults = Map.get(base, "defaults", %{})
    overlay_defaults = Map.get(overlay, "defaults", %{})
    Map.put(base, "defaults", Map.merge(base_defaults, overlay_defaults))
  end

  defp expected_entity_types_from_schema(_intent, schema) do
    mappings = Map.get(schema, "entity_mappings", %{})
    mappings |> Map.values() |> List.flatten() |> Enum.uniq()
  rescue
    _ -> []
  end

  defp get_parent_intent(intent) do
    case String.split(intent, ".") do
      [parent | _] -> parent
      _ -> intent
    end
  end

  # The entities that can fill a slot. Entities are decided spans only: the
  # spans entity extraction could not decide travel apart from them
  # (`Brain.ML.EntityExtractor.extract/2`), so none reaches a slot. Every
  # entity must carry a type: one without raises rather than filling a slot
  # named nil or matching as "".
  defp slot_fillers!(entities) do
    Enum.map(entities, fn
      %{entity_type: type} = entity when is_binary(type) or (is_atom(type) and not is_nil(type)) ->
        entity

      entity ->
        raise ArgumentError, "SlotDetector: an entity with no type cannot fill a slot: #{inspect(entity, limit: 8)}"
    end)
  end

  defp process_schema(schema, intent, entities, record?, utterance) do
    required = Map.get(schema, "required", [])
    optional = Map.get(schema, "optional", [])
    defaults = Map.get(schema, "defaults", %{})
    mappings = Map.get(schema, "entity_mappings", %{})
    result = SlotResult.new(intent)
    {result, fillers} = fill_slots_from_entities(result, required ++ optional, entities, mappings, record?)
    result = capture_free_text(result, required ++ optional, fillers, utterance)
    result = apply_defaults(result, defaults)
    filled_slot_names = Map.keys(result.filled_slots)
    missing_required = Enum.reject(required, &(&1 in filled_slot_names))
    missing_optional = Enum.reject(optional, &(&1 in filled_slot_names))

    %{
      result
      | missing_required: missing_required,
        missing_optional: missing_optional,
        all_required_filled: missing_required == []
    }
  end

  # The result with each slot an entity fills filled, and the entities that
  # filled one.
  defp fill_slots_from_entities(result, slots, entities, mappings, record?) do
    Enum.map_reduce(slots, result, fn slot_name, acc ->
      mapped_entity_types = Map.get(mappings, slot_name, [slot_name])

      exact_match =
        Enum.find(entities, fn entity ->
          entity_type = entity[:entity_type]
          entity_type in mapped_entity_types
        end)

      {matching_entity, via_compatibility} =
        case exact_match do
          nil ->
            compat = find_compatible_entity(entities, mapped_entity_types)
            {compat, compat != nil}

          entity ->
            {entity, false}
        end

      case matching_entity do
        nil ->
          {nil, acc}

        entity ->
          if via_compatibility and record? do
            record_learned_type_preference(entity, slot_name, result.schema_name)
          end

          value = entity[:value] || entity["value"]
          confidence = entity[:confidence] || entity["confidence"] || 1.0

          # A fill through type compatibility (a "person" in a music_artist
          # slot) is recorded as such, so a narrowing step can still refine
          # the entity's type to what the slot asks for.
          source = if via_compatibility, do: :type_compatible, else: :explicit
          {entity, SlotResult.fill_slot(acc, slot_name, value, source, confidence)}
      end
    end)
    |> then(fn {fillers, filled} -> {filled, Enum.reject(fillers, &is_nil/1)} end)
  end

  # Each slot of `slots` the capture model captures and no entity filled,
  # filled with the span it captures from the utterance, kept off the
  # entities that filled the other slots. Nothing is captured without an
  # utterance.
  defp capture_free_text(result, _slots, _fillers, nil), do: result

  defp capture_free_text(result, slots, fillers, %{text: text, model: model}) do
    captured = FreeTextCapture.slots(model)

    case Enum.filter(slots, &(&1 in captured and not Map.has_key?(result.filled_slots, &1))) do
      [] ->
        result

      free ->
        excluded = Enum.map(fillers, &filler_range!/1)

        Enum.reduce(free, result, fn slot, acc ->
          case FreeTextCapture.capture(model, slot, text, excluded) do
            {:ok, %{value: value, confidence: confidence}} -> SlotResult.fill_slot(acc, slot, value, :captured, confidence)
            :abstain -> acc
          end
        end)
    end
  end

  defp filler_range!(%{start_pos: start_pos, end_pos: end_pos}) when is_integer(start_pos) and is_integer(end_pos),
    do: {start_pos, end_pos}

  defp filler_range!(entity) do
    raise ArgumentError,
          "SlotDetector: an entity that fills a slot carries no :start_pos and :end_pos, so a captured free-text " <>
            "span cannot be kept off it: #{inspect(entity, limit: 8)}"
  end

  defp find_compatible_entity(entities, mapped_entity_types) do
    Enum.find(entities, fn entity ->
      entity_type = to_string(entity.entity_type)

      Enum.any?(mapped_entity_types, fn mapped ->
        TypeHierarchy.compatible?(entity_type, mapped)
      end)
    end)
  end

  defp record_learned_type_preference(entity, slot_name, schema_name) do
    entity_value = entity[:value] || entity[:match] || ""
    domain =
      case String.split(to_string(schema_name), ".", parts: 2) do
        [d, _] -> String.to_atom(d)
        _ -> nil
      end

    if entity_value != "" and domain != nil do
      try do
        Brain.ML.Gazetteer.record_type_preference(entity_value, slot_name, domain)
      rescue
        _ -> :ok
      end
    end
  end

  defp apply_defaults(result, defaults) do
    Enum.reduce(defaults, result, fn {slot_name, default_value}, acc ->
      if Map.has_key?(acc.filled_slots, slot_name) do
        acc
      else
        SlotResult.fill_slot(acc, slot_name, default_value, :default, 1.0)
      end
    end)
  end

  defp find_domain_prefix_schema(intent, entities, schemas) do
    domain =
      case String.split(intent, ".", parts: 2) do
        [d, _] -> d
        _ -> nil
      end

    if domain do
      entity_types = Enum.map(entities, fn e -> e[:entity_type] end) |> Enum.uniq()

      candidates =
        schemas
        |> Enum.filter(fn {schema_intent, _schema} ->
          String.starts_with?(schema_intent, domain <> ".")
        end)
        |> Enum.map(fn {schema_intent, schema} ->
          mappings = Map.get(schema, "entity_mappings", %{})

          match_count =
            Enum.count(entity_types, fn etype ->
              Enum.any?(mappings, fn {_slot, mapped_types} ->
                is_list(mapped_types) and
                  (etype in mapped_types or
                     Enum.any?(mapped_types, &TypeHierarchy.compatible?(etype, &1)))
              end)
            end)

          {schema_intent, schema, match_count}
        end)
        |> Enum.filter(fn {_, _, count} -> count > 0 end)
        |> Enum.sort_by(fn {_, _, count} -> -count end)

      case candidates do
        [{matched_intent, schema, _} | _] -> {schema, matched_intent}
        [] -> nil
      end
    else
      nil
    end
  end

  defp narrowing_match?(etype, mapped_types) when is_list(mapped_types) do
    if TypeHierarchy.ready?() do
      Enum.any?(mapped_types, &TypeHierarchy.is_a?(etype, &1))
    else
      false
    end
  rescue
    _ -> false
  end

  defp build_unknown_result(entities) do
    result = SlotResult.new("unknown")

    Enum.reduce(entities, result, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value]
      confidence = entity[:confidence] || 1.0
      SlotResult.fill_slot(acc, entity_type, value, :explicit, confidence)
    end)
  end
end
