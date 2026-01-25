defmodule ChatBot.Analysis.EntityDisambiguator do
  @moduledoc """
  Disambiguates entities when multiple types are possible,
  using speech act, discourse, and POS-tagged syntactic context.

  When the gazetteer returns multiple possible entity types for the same
  text (e.g., "Austin" could be a person or a location), this module
  uses contextual features to determine the most likely interpretation.

  Also handles cases where a single-type entity (e.g., "Nice" as location)
  is being used as a proper noun/name in context (e.g., "I'm Nice"),
  recognizing proper noun usage and mapping it to the appropriate entity type.

  ## Features Used

  - **POS tags**: What part of speech precedes/follows the entity
  - **Discourse indicators**: Is this a self-referential statement?
  - **Speech act context**: Is this a greeting, question, command?
  - **Intent hints**: What domain does the intent belong to?

  Uses IntentRegistry for intent domain lookups instead of keyword matching.

  ## Usage

      # With POS-tagged tokens
      pos_tagged = [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]
      entities = [%{value: "Austin", types: [person_info, location_info], ...}]
      context = %{discourse: discourse_result, speech_act: speech_act_result}

      disambiguated = EntityDisambiguator.disambiguate(entities, pos_tagged, context)

  """

  require Logger

  alias ChatBot.Analysis.IntentRegistry

  # Entity type preferences for different contexts
  # Higher score = more preferred in that context
  @context_preferences %{
    # Introduction context: prefer person over location
    introduction: %{
      "person" => 1.0,
      "location" => 0.2,
      "city" => 0.2,
      "music-artist" => 0.3
    },
    # Weather/location context: prefer location over person
    location_query: %{
      "location" => 1.0,
      "city" => 1.0,
      "person" => 0.1,
      "music-artist" => 0.1
    },
    # Music context: prefer artist
    music: %{
      "music-artist" => 1.0,
      "person" => 0.3,
      "location" => 0.1,
      "city" => 0.1
    },
    # Default: slight preference for more specific types
    default: %{
      "person" => 0.5,
      "location" => 0.5,
      "city" => 0.5,
      "music-artist" => 0.5
    }
  }

  @type entity_candidate :: %{
          value: String.t(),
          types: list(map()),
          start_pos: integer(),
          end_pos: integer()
        }

  @type context :: %{
          discourse: map() | nil,
          speech_act: map() | nil,
          intent: String.t() | nil
        }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Disambiguate entities that have multiple possible types.

  Takes a list of entities (where each entity may have multiple type interpretations)
  and returns a list with the most likely type selected for each.

  ## Parameters

  - `entities`: List of entity maps, each potentially having multiple `types`
  - `pos_tagged`: List of {token, pos_tag} tuples (or just tokens)
  - `context`: Map with :discourse, :speech_act, and optionally :intent

  ## Returns

  List of entities with a single type selected for each.
  """
  def disambiguate(entities, pos_tagged, context) when is_list(entities) do
    entities
    |> Enum.map(fn entity ->
      disambiguate_single(entity, pos_tagged, context)
    end)
  end

  @doc """
  Disambiguate a single entity with multiple possible types.
  """
  def disambiguate_single(entity, pos_tagged, context) do
    types = get_entity_types(entity)
    entity_position = get_entity_position(entity, pos_tagged)

    cond do
      # No types to disambiguate
      length(types) == 0 ->
        entity

      # Only one type - but check if context suggests proper noun/name usage
      length(types) == 1 ->
        single_type = hd(types)
        single_type_name = get_type_name(single_type)

        # If context strongly suggests proper noun/name usage (introduction pattern)
        # but entity is location/city, recognize it as a proper noun/name
        intro_confidence = introduction_confidence(pos_tagged, entity_position, context)

        if intro_confidence >= 0.7 and single_type_name in ["location", "city", "place-name"] do
          # Strong introduction context - this is being used as a proper noun/name,
          # not as a location reference. In our entity system, proper names map to "person" type.
          name_type = %{
            entity_type: "person",
            entity: "person",
            value: Map.get(entity, :value) || Map.get(entity, "value"),
            confidence: intro_confidence,
            # Metadata indicating this was recognized as proper noun usage, not from person database
            disambiguation_reason: "proper_noun_usage"
          }

          select_type(entity, name_type)
        else
          # Normal case - use the single type
          select_type(entity, single_type)
        end

      # Multiple types - need to disambiguate
      true ->
        # Extract features from context
        features = extract_features(entity, pos_tagged, context)

        # Score each type based on features
        scored_types =
          Enum.map(types, fn type_info ->
            score = score_type(type_info, features, context)
            {type_info, score}
          end)

        # Select highest scoring type
        {best_type, _score} = Enum.max_by(scored_types, fn {_, score} -> score end)

        select_type(entity, best_type)
    end
  end

  @doc """
  Detect if the context suggests an introduction pattern.

  Returns a confidence score (0.0 to 1.0) indicating how likely
  this is an introduction context.
  """
  def introduction_confidence(pos_tagged, entity_position, context) do
    # Extract features
    features = %{
      pron_verb_pattern: has_pron_verb_before?(pos_tagged, entity_position),
      self_referential: self_referential?(context),
      greeting_context: greeting_context?(context)
    }

    # Combine features with weights
    pron_verb_score = if features.pron_verb_pattern, do: 0.5, else: 0.0
    self_ref_score = if features.self_referential, do: 0.3, else: 0.0
    greeting_score = if features.greeting_context, do: 0.2, else: 0.0

    score = pron_verb_score + self_ref_score + greeting_score

    min(score, 1.0)
  end

  # ============================================================================
  # Feature Extraction
  # ============================================================================

  defp extract_features(entity, pos_tagged, context) do
    entity_pos = get_entity_position(entity)

    %{
      # POS-based features
      preceding_pron: count_preceding_tag(pos_tagged, entity_pos, "PRON"),
      preceding_verb: count_preceding_tag(pos_tagged, entity_pos, "VERB"),
      pron_verb_adjacent: has_pron_verb_before?(pos_tagged, entity_pos),

      # Discourse features
      self_referential: self_referential?(context),

      # Speech act features
      greeting_context: greeting_context?(context),
      question_context: question_context?(context),
      command_context: command_context?(context),

      # Intent-based features
      weather_intent: weather_intent?(context),
      music_intent: music_intent?(context),
      navigation_intent: navigation_intent?(context),

      # Detected context type (for preference lookup)
      context_type: detect_context_type(context, pos_tagged, entity_pos)
    }
  end

  defp get_entity_position(entity) do
    Map.get(entity, :start_pos) || Map.get(entity, "start_pos") || 0
  end

  # Find entity position in POS-tagged tokens by matching entity value
  defp get_entity_position(entity, pos_tagged) when is_list(pos_tagged) do
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    entity_value_lower = String.downcase(entity_value)

    # Find the token that matches the entity value
    pos_tagged
    |> Enum.with_index()
    |> Enum.find_value(fn {{token, _tag}, idx} ->
      if String.downcase(token) == entity_value_lower, do: idx
    end) || 0
  end

  defp get_entity_position(_entity, _pos_tagged), do: 0

  defp get_entity_types(entity) do
    cond do
      is_list(Map.get(entity, :types)) ->
        entity.types

      is_list(Map.get(entity, "types")) ->
        entity["types"]

      # Single type in entity
      is_map(entity) and (Map.has_key?(entity, :entity) or Map.has_key?(entity, "entity")) ->
        [entity]

      true ->
        []
    end
  end

  # ============================================================================
  # POS-based Feature Detection
  # ============================================================================

  defp count_preceding_tag(pos_tagged, entity_pos, target_tag) when is_list(pos_tagged) do
    # Get tokens before entity position
    preceding =
      pos_tagged
      |> Enum.take(entity_pos)
      |> Enum.take(-3)

    # Count occurrences of target tag
    Enum.count(preceding, fn
      {_token, tag} -> String.upcase(to_string(tag)) == target_tag
      _ -> false
    end)
  end

  defp has_pron_verb_before?(pos_tagged, entity_pos) when is_list(pos_tagged) do
    # Check for PRON + VERB pattern in the 2-3 tokens before entity
    preceding =
      pos_tagged
      |> Enum.take(entity_pos)
      |> Enum.take(-3)
      |> Enum.map(fn
        {_token, tag} -> String.upcase(to_string(tag))
        tag when is_binary(tag) -> String.upcase(tag)
        _ -> "X"
      end)

    # Check for patterns like [PRON, VERB] or [PRON, AUX]
    case preceding do
      ["PRON", "VERB"] -> true
      ["PRON", "VERB" | _] -> true
      ["PRON", "AUX"] -> true
      ["PRON", "AUX" | _] -> true
      [_, "PRON", "VERB"] -> true
      [_, "PRON", "AUX"] -> true
      _ -> false
    end
  end

  defp has_pron_verb_before?(_, _), do: false

  # ============================================================================
  # Context Detection
  # ============================================================================

  defp self_referential?(context) do
    discourse = Map.get(context, :discourse) || %{}
    indicators = Map.get(discourse, :indicators) || []

    "self_referential" in indicators or
      Enum.any?(indicators, &String.contains?(to_string(&1), "first_person"))
  end

  defp greeting_context?(context) do
    speech_act = Map.get(context, :speech_act) || %{}

    Map.get(speech_act, :category) == :expressive and
      Map.get(speech_act, :sub_type) in [:greeting, :nice_to_meet]
  end

  defp question_context?(context) do
    speech_act = Map.get(context, :speech_act) || %{}
    Map.get(speech_act, :is_question, false)
  end

  defp command_context?(context) do
    speech_act = Map.get(context, :speech_act) || %{}

    Map.get(speech_act, :category) == :directive and
      Map.get(speech_act, :sub_type) == :command
  end

  defp weather_intent?(context) do
    intent = Map.get(context, :intent, "")
    IntentRegistry.weather_intent?(intent)
  end

  defp music_intent?(context) do
    intent = Map.get(context, :intent, "")
    IntentRegistry.music_intent?(intent)
  end

  defp navigation_intent?(context) do
    intent = Map.get(context, :intent, "")
    IntentRegistry.navigation_intent?(intent)
  end

  defp detect_context_type(context, pos_tagged, entity_pos) do
    cond do
      # Introduction: PRON+VERB before entity + self-referential discourse
      has_pron_verb_before?(pos_tagged, entity_pos) and self_referential?(context) ->
        :introduction

      # Music context
      music_intent?(context) ->
        :music

      # Weather/location query
      weather_intent?(context) or navigation_intent?(context) ->
        :location_query

      # Default
      true ->
        :default
    end
  end

  # ============================================================================
  # Scoring
  # ============================================================================

  defp score_type(type_info, features, _context) do
    entity_type = get_type_name(type_info)
    context_type = features.context_type

    # Get base preference for this entity type in this context
    preferences = Map.get(@context_preferences, context_type, @context_preferences.default)
    base_score = Map.get(preferences, entity_type, 0.3)

    # Boost score based on specific features
    boost =
      cond do
        # Strong introduction signal + person type
        features.pron_verb_adjacent and features.self_referential and entity_type == "person" ->
          0.5

        # Weather intent + location type
        features.weather_intent and entity_type in ["location", "city"] ->
          0.4

        # Music intent + artist type
        features.music_intent and entity_type == "music-artist" ->
          0.4

        # Greeting context + person type
        features.greeting_context and entity_type == "person" ->
          0.3

        true ->
          0.0
      end

    base_score + boost
  end

  defp get_type_name(type_info) when is_map(type_info) do
    Map.get(type_info, :entity_type) ||
      Map.get(type_info, :entity) ||
      Map.get(type_info, :type) ||
      Map.get(type_info, "entity_type") ||
      Map.get(type_info, "entity") ||
      Map.get(type_info, "type") ||
      "unknown"
  end

  defp get_type_name(_), do: "unknown"

  # ============================================================================
  # Entity Selection
  # ============================================================================

  defp select_type(entity, selected_type) when is_map(selected_type) do
    # Merge the selected type info into the entity
    base_merge = %{
      entity: get_type_name(selected_type),
      entity_type: get_type_name(selected_type),
      value: Map.get(selected_type, :value) || Map.get(entity, :value) || Map.get(entity, "value"),
      disambiguation_source: :context_analysis
    }

    # Preserve disambiguation_reason if provided (e.g., "proper_noun_usage")
    final_merge =
      if Map.has_key?(selected_type, :disambiguation_reason) do
        Map.put(base_merge, :disambiguation_reason, selected_type.disambiguation_reason)
      else
        base_merge
      end

    entity
    |> Map.delete(:types)
    |> Map.delete("types")
    |> Map.merge(final_merge)
  end

  defp select_type(entity, _), do: entity
end
