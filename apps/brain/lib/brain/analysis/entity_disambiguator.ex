defmodule Brain.Analysis.EntityDisambiguator do
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

  alias Brain.Analysis.IntentRegistry
  alias World.TypeInferrer

  # Entity type preferences for different contexts
  # Higher score = more preferred in that context
  # NOTE: These are fallback preferences. The primary scoring now uses
  # IntentRegistry.expected_entity_types/1 for dynamic context-aware scoring.
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
    # Device/smarthome context: prefer device entities over music/person
    device: %{
      "device" => 1.0,
      "lights" => 1.0,
      "heating" => 1.0,
      "room" => 0.8,
      "music-artist" => 0.1,
      "person" => 0.1,
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
  
  Context map can include:
  - `:discourse` - Discourse analysis result
  - `:speech_act` - Speech act classification
  - `:intent` - Classified intent string
  - `:original_text` - The original text being processed (for text-based pattern matching)
  """
  def disambiguate_single(entity, pos_tagged, context) do
    types = get_entity_types(entity)
    entity_position = get_entity_position(entity, pos_tagged)
    entity_type = get_type_name(entity)
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    
    # Enrich context with entity value for introduction pattern detection
    enriched_context = Map.put(context, :entity_value, entity_value)

    cond do
      # Single type that requires inference (e.g., ambiguous_name_location)
      # Use TypeInferrer to dynamically determine the actual type from context
      length(types) <= 1 and requires_inference?(entity_type) ->
        # First check if this is an introduction pattern - override type inference
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)
        
        if intro_confidence >= 0.7 do
          # Strong introduction context - this is a person's name
          # Either keep existing person type or convert location to person
          if entity_type == "person" do
            # Already marked as person, just add disambiguation metadata
            entity
            |> Map.put(:disambiguation_reason, "introduction_pattern")
            |> Map.put(:disambiguation_source, :context_analysis)
          else
            # Convert location/city to person
            create_person_type_from_intro(entity, intro_confidence)
          end
        else
          infer_type_with_type_inferrer(entity, pos_tagged, context)
        end

      # No types to disambiguate
      length(types) == 0 ->
        entity

      # Only one type - but check if context suggests proper noun/name usage
      length(types) == 1 ->
        single_type = hd(types)
        single_type_name = get_type_name(single_type)

        # If context strongly suggests proper noun/name usage (introduction pattern)
        # but entity is location/city, recognize it as a proper noun/name
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 and single_type_name in ["location", "city", "place-name"] do
          # Strong introduction context - this is being used as a proper noun/name,
          # not as a location reference. In our entity system, proper names map to "person" type.
          create_person_type_from_intro(entity, intro_confidence)
        else
          # Normal case - use the single type
          select_type(entity, single_type)
        end

      # Multiple types - need to disambiguate
      true ->
        # First check for introduction pattern - this takes precedence
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)
        
        if intro_confidence >= 0.7 do
          # Strong introduction pattern - prefer person type
          person_type = Enum.find(types, fn t -> get_type_name(t) == "person" end)
          
          if person_type do
            select_type(entity, person_type)
          else
            # No person type available, create one
            create_person_type_from_intro(entity, intro_confidence)
          end
        else
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
  end
  
  # Create a person type entity when introduction pattern is detected
  defp create_person_type_from_intro(entity, intro_confidence) do
    name_type = %{
      entity_type: "person",
      entity: "person",
      value: Map.get(entity, :value) || Map.get(entity, "value"),
      confidence: intro_confidence,
      # Metadata indicating this was recognized as proper noun usage, not from person database
      disambiguation_reason: "introduction_pattern"
    }
    
    select_type(entity, name_type)
  end

  @doc """
  Check if an entity type requires context-based disambiguation.

  Types that require inference include:
  - Types with "ambiguous_" prefix
  - "person" and "location" types, which are often ambiguous (e.g., Austin)
  - These require context (intent, discourse) to determine the correct type
  """
  def requires_inference?(entity_type) when is_binary(entity_type) do
    # ambiguous_* types always need inference
    String.starts_with?(entity_type, "ambiguous_") or
      # person/location are contextually ambiguous - "Austin" could be either
      entity_type in ["person", "location"]
  end

  def requires_inference?(_), do: false

  @doc """
  Infer the entity type using intent context and TypeInferrer.

  Primary: Uses IntentRegistry expected_entity_types to match the current
  intent's requirements.
  
  Fallback: TypeInferrer's learned patterns when no intent context or
  when TypeInferrer returns a type that matches expected types.

  Requires world_id in context for proper data isolation.
  """
  def infer_type_with_type_inferrer(entity, pos_tagged, context) do
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    original_type = Map.get(entity, :entity_type) || ""
    intent = Map.get(context, :intent, "")
    # Use "default" world_id if none is provided for backward compatibility
    world_id = Map.get(context, :world_id) || "default"

    # Get expected entity types from IntentRegistry
    expected_types = IntentRegistry.expected_entity_types(intent)

    # Extract tokens and tags from POS-tagged list
    {context_tokens, context_tags} =
      case pos_tagged do
        [{_, _} | _] ->
          tokens = Enum.map(pos_tagged, fn {token, _tag} -> token end)
          tags = Enum.map(pos_tagged, fn {_token, tag} -> tag end)
          {tokens, tags}

        _ ->
          {[], []}
      end

    # Use TypeInferrer to infer the type from context patterns (world-scoped)
    {inferred_type, type_confidence} =
      TypeInferrer.infer_type(entity_value, context_tokens, context_tags, world_id)

    # Determine final type based on intent context
    # Intent context takes priority over TypeInferrer for ambiguous cases
    final_type =
      cond do
        # If inferred type is in expected types, use it
        inferred_type in expected_types ->
          inferred_type

        # If original type is in expected types, keep it
        original_type in expected_types ->
          original_type

        # Weather/navigation intents expect location types
        # Choose location if it's in expected types
        IntentRegistry.weather_intent?(intent) or IntentRegistry.navigation_intent?(intent) ->
          Enum.find(expected_types, "location", &(&1 in ["location", "city"]))

        # Introduction intents expect person types
        IntentRegistry.introduction_intent?(intent) ->
          Enum.find(expected_types, "person", &(&1 in ["person", "name"]))

        # Music intents expect artist types
        IntentRegistry.music_intent?(intent) ->
          Enum.find(expected_types, inferred_type, &(&1 in ["music-artist", "song", "album"]))

        # If expected types exist but inferred doesn't match, use first expected type
        length(expected_types) > 0 ->
          hd(expected_types)

        # Default: use inferred type
        true ->
          inferred_type
      end

    # Return entity with final type
    entity
    |> Map.put(:entity, final_type)
    |> Map.put(:entity_type, final_type)
    |> Map.put(:disambiguation_source, :type_inferrer)
    |> Map.put(:disambiguation_confidence, type_confidence)
  end

  @doc """
  Detect if the context suggests an introduction pattern.

  Returns a confidence score (0.0 to 1.0) indicating how likely
  this is an introduction context.
  
  Uses multiple signals:
  - POS tag patterns (PRON + VERB before entity)
  - Text-based patterns ("I'm [Name]", "My name is [Name]", etc.)
  - Discourse indicators (self-referential)
  - Speech act context (greeting)
  """
  def introduction_confidence(pos_tagged, entity_position, context) do
    entity_value = extract_entity_value_from_context(context)
    original_text = Map.get(context, :original_text, "")
    
    # Extract features
    features = %{
      pron_verb_pattern: has_pron_verb_before?(pos_tagged, entity_position),
      text_intro_pattern: text_has_introduction_pattern?(original_text, entity_value),
      self_referential: self_referential?(context),
      greeting_context: greeting_context?(context)
    }

    # Combine features with weights
    # Text-based pattern is more reliable than POS tags for contractions
    text_intro_score = if features.text_intro_pattern, do: 0.7, else: 0.0
    pron_verb_score = if features.pron_verb_pattern, do: 0.5, else: 0.0
    self_ref_score = if features.self_referential, do: 0.2, else: 0.0
    greeting_score = if features.greeting_context, do: 0.1, else: 0.0

    # Take the max of text pattern or POS pattern (don't double-count)
    pattern_score = max(text_intro_score, pron_verb_score)
    score = pattern_score + self_ref_score + greeting_score

    # Round to 4 decimal places to avoid floating-point precision issues
    Float.round(min(score, 1.0), 4)
  end
  
  # Extract entity value from context if available
  defp extract_entity_value_from_context(context) do
    Map.get(context, :entity_value, "")
  end
  
  @doc """
  Check if text contains an introduction pattern with the given entity value.
  
  Detects patterns like:
  - "I'm [Name]" / "I am [Name]"
  - "My name is [Name]"
  - "This is [Name]" (when self-referential)
  - "Call me [Name]"
  - "I go by [Name]"
  - "[Name] here" (at start)
  """
  def text_has_introduction_pattern?(text, entity_value) when is_binary(text) and is_binary(entity_value) do
    return_false_if_empty = entity_value == "" or String.length(entity_value) < 2
    if return_false_if_empty do
      false
    else
      lower_text = String.downcase(text)
      lower_entity = String.downcase(entity_value)
      
      # Build introduction patterns with the entity value
      # These are common ways people introduce themselves
      introduction_patterns = [
        # "I'm Austin" / "I am Austin"
        "i'm #{lower_entity}",
        "i am #{lower_entity}",
        "im #{lower_entity}",
        # "My name is Austin" / "My name's Austin"
        "my name is #{lower_entity}",
        "my name's #{lower_entity}",
        "name is #{lower_entity}",
        "name's #{lower_entity}",
        # "Call me Austin" / "They call me Austin"
        "call me #{lower_entity}",
        "called #{lower_entity}",
        # "I go by Austin"
        "i go by #{lower_entity}",
        "go by #{lower_entity}",
        # "It's Austin" (when context is greeting)
        "it's #{lower_entity}",
        "this is #{lower_entity}",
        # "[Name] here" at start
        "#{lower_entity} here"
      ]
      
      # Check if any pattern matches
      # We use String.contains? which is character-based, not regex
      Enum.any?(introduction_patterns, fn pattern ->
        String.contains?(lower_text, pattern)
      end)
    end
  end
  
  def text_has_introduction_pattern?(_, _), do: false

  # ============================================================================
  # Feature Extraction
  # ============================================================================

  defp extract_features(entity, pos_tagged, context) do
    entity_pos = get_entity_position(entity)
    intent = Map.get(context, :intent, "")

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
      device_intent: device_intent?(context),

      # Dynamic entity expectations from IntentRegistry
      expected_entity_types: IntentRegistry.expected_entity_types(intent),
      intent: intent,

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

      # Single type in entity - check for entity_type key
      is_map(entity) and Map.has_key?(entity, :entity_type) ->
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

  defp device_intent?(context) do
    intent = Map.get(context, :intent, "")
    # Use IntentRegistry for device intent detection (data-driven via intent_registry.json)
    IntentRegistry.device_intent?(intent)
  end

  defp detect_context_type(context, pos_tagged, entity_pos) do
    intent = Map.get(context, :intent, "")
    domain = IntentRegistry.domain(intent)
    
    cond do
      # Introduction: Use IntentRegistry to detect introduction intents
      # These prefer person over location for entity disambiguation
      IntentRegistry.introduction_intent?(intent) ->
        :introduction

      # Introduction: PRON+VERB before entity + self-referential discourse
      # Structural pattern detection as backup
      has_pron_verb_before?(pos_tagged, entity_pos) and self_referential?(context) ->
        :introduction

      # Use intent domain directly for dynamic context detection
      # This automatically handles any domain defined in intent_registry.json
      domain == :device ->
        :device

      domain == :music ->
        :music

      domain in [:weather, :navigation] ->
        :location_query

      # Device intents via IntentRegistry
      IntentRegistry.device_intent?(intent) ->
        :device

      # Default for unknown/unhandled domains
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
    expected_types = Map.get(features, :expected_entity_types, [])

    # PRIMARY: Dynamic scoring based on IntentRegistry entity_mappings
    # If the entity type matches what the intent expects, give it a high score
    dynamic_score = if entity_type in expected_types, do: 0.8, else: 0.0

    # FALLBACK: Static context preferences for edge cases
    preferences = Map.get(@context_preferences, context_type, @context_preferences.default)
    static_score = Map.get(preferences, entity_type, 0.3)

    # Use dynamic score if we have expected types, otherwise use static
    base_score = if length(expected_types) > 0 and dynamic_score > 0 do
      dynamic_score
    else
      static_score
    end

    # Additional boost based on specific POS/discourse features
    boost = calculate_feature_boost(entity_type, features)

    base_score + boost
  end

  # Calculate boost from POS patterns and discourse features
  defp calculate_feature_boost(entity_type, features) do
    cond do
      # Strong introduction signal + person type
      features.pron_verb_adjacent and features.self_referential and entity_type == "person" ->
        0.5

      # Device intent + device/lights type
      features.device_intent and entity_type in ["device", "lights", "heating", "room"] ->
        0.4

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
  end

  defp get_type_name(type_info) when is_map(type_info) do
    Map.get(type_info, :entity_type, "unknown")
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
      value:
        Map.get(selected_type, :value) || Map.get(entity, :value) || Map.get(entity, "value"),
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
