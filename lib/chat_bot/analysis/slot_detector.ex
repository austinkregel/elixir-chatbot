defmodule ChatBot.Analysis.SlotDetector do
  @moduledoc """
  Detects required and optional slots for a given intent and fills them from entities.

  This module:
  - Loads slot schemas from JSON configuration
  - Maps extracted entities to slots
  - Identifies missing required slots
  - Applies default values where configured
  """

  alias ChatBot.Analysis.SlotResult

  require Logger

  @schemas_path "priv/analysis/slot_schemas.json"

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
  Lists all available slot schemas.
  """
  def list_schemas do
    load_schemas()
  end

  @doc """
  Suggests an intent based on entities present.

  This can be used when intent classification has low confidence.
  """
  def suggest_intent_from_entities(entities) when is_list(entities) do
    schemas = load_schemas()
    entity_types = Enum.map(entities, fn e -> e[:entity] || e["entity"] end) |> Enum.uniq()

    # Score each schema by how many entity types match
    scored_schemas =
      schemas
      |> Enum.map(fn {intent, schema} ->
        mappings = Map.get(schema, "entity_mappings", %{})

        matching_slots =
          mappings
          |> Enum.filter(fn {_slot, mapped_entities} ->
            Enum.any?(mapped_entities, &(&1 in entity_types))
          end)
          |> length()

        {intent, matching_slots}
      end)
      |> Enum.filter(fn {_, score} -> score > 0 end)
      |> Enum.sort_by(fn {_, score} -> -score end)

    case scored_schemas do
      [{intent, score} | _] when score > 0 -> {:ok, intent, score}
      _ -> {:error, :no_match}
    end
  end

  @doc """
  Suggests an intent based on keywords in the text.

  This is a heuristic fallback when intent classification has low confidence.
  Returns {:ok, intent, confidence} or {:error, :no_match}.
  """
  def suggest_intent_from_keywords(text) when is_binary(text) do
    lower_text = String.downcase(text)

    # Keyword patterns for different intents (ordered by specificity)
    keyword_patterns = [
      {"weather.query", ~w(weather forecast temperature rain sunny cloudy), 0.7},
      {"music.play", ~w(play music song playlist album artist), 0.7},
      {"reminder.create", ~w(remind reminder remember), 0.7},
      {"device.control", ~w(turn on off lights switch dim brightness), 0.6},
      {"news.query", ~w(news headlines), 0.6},
      {"search.web", ~w(search google look up find information), 0.5}
    ]

    # Score each pattern
    scored =
      keyword_patterns
      |> Enum.map(fn {intent, keywords, base_confidence} ->
        matches = Enum.count(keywords, &String.contains?(lower_text, &1))

        if matches > 0 do
          # Boost confidence based on number of keyword matches
          confidence = min(base_confidence + matches * 0.1, 0.9)
          {intent, confidence, matches}
        else
          {intent, 0.0, 0}
        end
      end)
      |> Enum.filter(fn {_, conf, _} -> conf > 0 end)
      |> Enum.sort_by(fn {_, conf, matches} -> {-conf, -matches} end)

    case scored do
      [{intent, confidence, _} | _] -> {:ok, intent, confidence}
      [] -> {:error, :no_match}
    end
  end

  # Private functions

  defp load_schemas do
    case Application.get_env(:chat_bot, :analysis_schemas_path, @schemas_path) do
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
          entity_type = entity[:entity] || entity["entity"]
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
      entity_type = entity[:entity] || entity["entity"]
      value = entity[:value] || entity["value"]
      confidence = entity[:confidence] || entity["confidence"] || 1.0
      SlotResult.fill_slot(acc, entity_type, value, :explicit, confidence)
    end)
  end
end
