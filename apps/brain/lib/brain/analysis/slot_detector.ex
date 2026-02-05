defmodule Brain.Analysis.SlotDetector do
  @moduledoc """
  Detects required and optional slots for a given intent and fills them from entities.

  This module:
  - Loads slot schemas from JSON configuration
  - Maps extracted entities to slots
  - Identifies missing required slots
  - Applies default values where configured
  - Provides clarification prompts for missing slots
  """

  alias Brain.Analysis.{SlotResult, IntentRegistry, EntityTypes}

  require Logger

  @schemas_path "priv/analysis/intent_registry.json"

  @doc """
  Detects slots for the given intent and fills them from entities.

  Returns a SlotResult struct indicating which slots are filled and which are missing.
  """
  def detect(intent, entities) when is_binary(intent) and is_list(entities) do
    schemas = load_schemas()

    case Map.get(schemas, intent) do
      nil ->
        # Try to match a parent intent (e.g., "weather" from "weather.query")
        parent_intent = get_parent_intent(intent)

        case Map.get(schemas, parent_intent) do
          nil -> build_unknown_result(entities)
          schema -> process_schema(schema, intent, entities)
        end

      schema ->
        process_schema(schema, intent, entities)
    end
  end

  @doc """
  Returns the slot schema for a given intent.
  """
  def get_schema(intent) do
    schemas = load_schemas()
    Map.get(schemas, intent) || Map.get(schemas, get_parent_intent(intent))
  end

  @doc """
  Returns the set of entity types that can fill slots for a given intent.

  This is useful for filtering entities to only those relevant to the intent.
  """
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
  Lists all available slot schemas.
  """
  def list_schemas do
    load_schemas()
  end

  @doc """
  Suggests an intent based on entities present.

  This can be used when intent classification has low confidence.

  Scoring algorithm:
  1. Primary: Count unique entity types that can fill ANY slot (not slots per type)
  2. Tiebreaker: Ratio of required slots that can be filled

  This prevents intents with multiple slots accepting the same type from
  scoring higher (e.g., navigation.directions with destination+origin both
  accepting location should not beat weather.query for a single location entity).
  """
  def suggest_intent_from_entities(entities) when is_list(entities) do
    schemas = load_schemas()
    entity_types = Enum.map(entities, fn e -> e[:entity_type] end) |> Enum.uniq()

    # Score each schema by unique entity type matches, with tiebreakers
    scored_schemas =
      schemas
      |> Enum.map(fn {intent, schema} ->
        mappings = Map.get(schema, "entity_mappings", %{})
        required = Map.get(schema, "required", [])
        domain = Map.get(schema, "domain", "unknown")

        # Count entity types that can fill ANY slot (not slots per type)
        # This prevents double-counting when multiple slots accept the same type
        matched_types =
          entity_types
          |> Enum.count(fn etype ->
            Enum.any?(mappings, fn {_slot, mapped} -> etype in mapped end)
          end)

        # Tiebreaker 1: ratio of required slots that can be filled
        filled_required =
          Enum.count(required, fn slot ->
            mapped = Map.get(mappings, slot, [])
            Enum.any?(entity_types, &(&1 in mapped))
          end)

        fill_ratio =
          if length(required) > 0, do: filled_required / length(required), else: 1.0

        # Tiebreaker 2: domain priority based on entity types present
        # When location entities are present, prefer weather over navigation
        # (navigation typically needs both origin AND destination to be useful)
        domain_priority = domain_priority_for_entities(domain, entity_types)

        {intent, matched_types, fill_ratio, domain_priority}
      end)
      |> Enum.filter(fn {_, score, _, _} -> score > 0 end)
      # Sort by: most matches, highest fill ratio, highest domain priority, alphabetical
      |> Enum.sort_by(fn {intent, score, ratio, priority} ->
        {-score, -ratio, -priority, intent}
      end)

    case scored_schemas do
      [{intent, score, _, _} | _] when score > 0 -> {:ok, intent, score}
      _ -> {:error, :no_match}
    end
  end

  # Calculate domain priority based on entity types present
  # Higher priority = more likely to be the intended domain
  defp domain_priority_for_entities(domain, entity_types) do
    has_location = EntityTypes.has_location_type?(entity_types)
    has_device = EntityTypes.has_device_type?(entity_types)
    has_music = EntityTypes.has_music_type?(entity_types)

    cond do
      # Weather queries with location are common - prioritize weather for location entities
      domain == "weather" and has_location -> 10
      # Device control with device entities
      domain == "device" and has_device -> 10
      # Music with music entities
      domain == "music" and has_music -> 10
      # Navigation requires destination, but location alone is ambiguous
      # (could be weather, could be navigation) - slightly lower priority
      domain == "navigation" and has_location -> 5
      # Default priority
      true -> 0
    end
  end

  @doc """
  Gets clarification prompt for a missing slot from intent_registry.json.
  Falls back to a generic prompt if not defined.

  ## Examples

      iex> SlotDetector.get_clarification_prompt("location", "weather.query")
      "What location would you like the weather for?"

      iex> SlotDetector.get_clarification_prompt("unknown_slot", "some.intent")
      "Could you please specify the unknown slot?"
  """
  def get_clarification_prompt(slot_name, intent) when is_binary(slot_name) do
    templates = IntentRegistry.clarification_templates(intent)

    case Map.get(templates, slot_name) do
      nil -> generate_generic_prompt(slot_name)
      prompt -> prompt
    end
  end

  def get_clarification_prompt(slot_name, intent) when is_atom(slot_name) do
    get_clarification_prompt(Atom.to_string(slot_name), intent)
  end

  def get_clarification_prompt(_, _), do: "Could you please provide more information?"

  @doc """
  Gets all clarification prompts for a list of missing slots.
  """
  def get_clarification_prompts(missing_slots, intent) when is_list(missing_slots) do
    Enum.map(missing_slots, fn slot ->
      slot_name = if is_atom(slot), do: Atom.to_string(slot), else: slot
      get_clarification_prompt(slot_name, intent)
    end)
  end

  # Private functions

  defp generate_generic_prompt(slot_name) do
    readable =
      slot_name
      |> String.replace("-", " ")
      |> String.replace("_", " ")

    "Could you please specify the #{readable}?"
  end

  defp load_schemas do
    case Application.get_env(:brain, :analysis_schemas_path, @schemas_path) do
      path when is_binary(path) ->
        case File.read(path) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, schemas} -> schemas
              {:error, _} -> default_schemas()
            end

          {:error, _} ->
            default_schemas()
        end

      _ ->
        default_schemas()
    end
  end

  defp default_schemas do
    %{
      "unknown" => %{
        "required" => [],
        "optional" => [],
        "defaults" => %{},
        "entity_mappings" => %{},
        "clarification_templates" => %{}
      }
    }
  end

  defp get_parent_intent(intent) do
    case String.split(intent, ".") do
      [parent | _] -> parent
      _ -> intent
    end
  end

  defp process_schema(schema, intent, entities) do
    required = Map.get(schema, "required", [])
    optional = Map.get(schema, "optional", [])
    defaults = Map.get(schema, "defaults", %{})
    mappings = Map.get(schema, "entity_mappings", %{})

    # Initialize result
    result = SlotResult.new(intent)

    # Fill slots from entities
    result = fill_slots_from_entities(result, required ++ optional, entities, mappings)

    # Apply defaults for unfilled slots
    result = apply_defaults(result, defaults)

    # Set missing required slots
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

  defp fill_slots_from_entities(result, slots, entities, mappings) do
    Enum.reduce(slots, result, fn slot_name, acc ->
      mapped_entity_types = Map.get(mappings, slot_name, [slot_name])

      # Find an entity that matches one of the mapped types
      matching_entity =
        Enum.find(entities, fn entity ->
          entity_type = entity[:entity_type]
          entity_type in mapped_entity_types
        end)

      case matching_entity do
        nil ->
          acc

        entity ->
          value = entity[:value] || entity["value"]
          confidence = entity[:confidence] || entity["confidence"] || 1.0
          SlotResult.fill_slot(acc, slot_name, value, :explicit, confidence)
      end
    end)
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

  defp build_unknown_result(entities) do
    result = SlotResult.new("unknown")

    # Still fill any entities we have
    Enum.reduce(entities, result, fn entity, acc ->
      entity_type = entity[:entity_type]
      value = entity[:value]
      confidence = entity[:confidence] || 1.0
      SlotResult.fill_slot(acc, entity_type, value, :explicit, confidence)
    end)
  end
end
