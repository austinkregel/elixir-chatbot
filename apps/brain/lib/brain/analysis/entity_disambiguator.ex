defmodule Brain.Analysis.EntityDisambiguator do
  @moduledoc "Disambiguates entities when multiple types are possible,\nusing speech act, discourse, and POS-tagged syntactic context.\n\nWhen the gazetteer returns multiple possible entity types for the same\ntext (e.g., \"Austin\" could be a person or a location), this module\nuses contextual features to determine the most likely interpretation.\n\nAlso handles cases where a single-type entity (e.g., \"Nice\" as location)\nis being used as a proper noun/name in context (e.g., \"I'm Nice\"),\nrecognizing proper noun usage and mapping it to the appropriate entity type.\n\n## Features Used\n\n- **POS tags**: What part of speech precedes/follows the entity\n- **Discourse indicators**: Is this a self-referential statement?\n- **Speech act context**: Is this a greeting, question, command?\n- **Intent hints**: What domain does the intent belong to?\n\nUses IntentRegistry for intent domain lookups instead of keyword matching.\n\n## Usage\n\n    # With POS-tagged tokens\n    pos_tagged = [{\"I\", \"PRON\"}, {\"am\", \"VERB\"}, {\"Austin\", \"PROPN\"}]\n    entities = [%{value: \"Austin\", types: [person_info, location_info], ...}]\n    context = %{discourse: discourse_result, speech_act: speech_act_result}\n\n    disambiguated = EntityDisambiguator.disambiguate(entities, pos_tagged, context)\n\n"

  # World.TypeInferrer is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.TypeInferrer}

  alias Brain.Analysis
  alias Brain.ML.Tokenizer
  require Logger

  alias Analysis.{IntentRegistry, EntityTypes}
  alias World.TypeInferrer

  @external_resource Path.join(:code.priv_dir(:brain), "analysis/context_preferences.json")
  @context_preferences Path.join(:code.priv_dir(:brain), "analysis/context_preferences.json")
                       |> File.read!()
                       |> Jason.decode!()
                       |> Enum.into(%{}, fn {k, v} -> {String.to_atom(k), v} end)

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

  @doc "Disambiguate entities that have multiple possible types.\n\nTakes a list of entities (where each entity may have multiple type interpretations)\nand returns a list with the most likely type selected for each.\n\n## Parameters\n\n- `entities`: List of entity maps, each potentially having multiple `types`\n- `pos_tagged`: List of {token, pos_tag} tuples (or just tokens)\n- `context`: Map with :discourse, :speech_act, and optionally :intent\n\n## Returns\n\nList of entities with a single type selected for each.\n"
  def disambiguate(entities, pos_tagged, context) when is_list(entities) do
    entities
    |> Enum.map(fn entity ->
      disambiguate_single(entity, pos_tagged, context)
    end)
  end

  @doc "Disambiguate a single entity with multiple possible types.\n\nContext map can include:\n- `:discourse` - Discourse analysis result\n- `:speech_act` - Speech act classification\n- `:intent` - Classified intent string\n- `:original_text` - The original text being processed (for text-based pattern matching)\n"
  def disambiguate_single(entity, pos_tagged, context) do
    types = get_entity_types(entity)
    entity_position = get_entity_position(entity, pos_tagged)
    entity_type = get_type_name(entity)
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    enriched_context = Map.put(context, :entity_value, entity_value)

    cond do
      length(types) <= 1 and requires_inference?(entity_type) ->
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 do
          if entity_type == "person" do
            entity
            |> Map.put(:disambiguation_reason, "introduction_pattern")
            |> Map.put(:disambiguation_source, :context_analysis)
          else
            create_person_type_from_intro(entity, intro_confidence)
          end
        else
          infer_type_with_type_inferrer(entity, pos_tagged, context)
        end

      types == [] ->
        entity

      length(types) == 1 ->
        single_type = hd(types)
        single_type_name = get_type_name(single_type)
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 and single_type_name in ["location", "city", "place-name"] do
          create_person_type_from_intro(entity, intro_confidence)
        else
          select_type(entity, single_type)
        end

      true ->
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 do
          person_type = Enum.find(types, fn t -> get_type_name(t) == "person" end)

          if person_type do
            select_type(entity, person_type)
          else
            create_person_type_from_intro(entity, intro_confidence)
          end
        else
          features = extract_features(entity, pos_tagged, context)

          scored_types =
            Enum.map(types, fn type_info ->
              score = score_type(type_info, features, context)
              {type_info, score}
            end)

          {best_type, _score} = Enum.max_by(scored_types, fn {_, score} -> score end)

          select_type(entity, best_type)
        end
    end
  end

  defp create_person_type_from_intro(entity, intro_confidence) do
    name_type = %{
      entity_type: "person",
      entity: "person",
      value: Map.get(entity, :value) || Map.get(entity, "value"),
      confidence: intro_confidence,
      disambiguation_reason: "introduction_pattern"
    }

    select_type(entity, name_type)
  end

  @doc "Check if an entity type requires context-based disambiguation.\n\nTypes that require inference include:\n- Types with \"ambiguous_\" prefix\n- \"person\" and \"location\" types, which are often ambiguous (e.g., Austin)\n- These require context (intent, discourse) to determine the correct type\n"
  def requires_inference?(entity_type) when is_binary(entity_type) do
    String.starts_with?(entity_type, "ambiguous_") or
      EntityTypes.is_person_type?(entity_type) or entity_type == "location"
  end

  def requires_inference?(_) do
    false
  end

  @doc "Infer the entity type using intent context and TypeInferrer.\n\nPrimary: Uses IntentRegistry expected_entity_types to match the current\nintent's requirements.\n\nFallback: TypeInferrer's learned patterns when no intent context or\nwhen TypeInferrer returns a type that matches expected types.\n\nRequires world_id in context for proper data isolation.\n"
  def infer_type_with_type_inferrer(entity, pos_tagged, context) do
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    original_type = Map.get(entity, :entity_type) || ""
    intent = Map.get(context, :intent, "")
    world_id = Map.get(context, :world_id) || "default"
    expected_types = IntentRegistry.expected_entity_types(intent)

    {context_tokens, context_tags} =
      case pos_tagged do
        [{_, _} | _] ->
          tokens = Enum.map(pos_tagged, fn {token, _tag} -> token end)
          tags = Enum.map(pos_tagged, fn {_token, tag} -> tag end)
          {tokens, tags}

        _ ->
          {[], []}
      end

    {inferred_type, type_confidence} =
      TypeInferrer.infer_type(entity_value, context_tokens, context_tags, world_id)

    final_type =
      cond do
        inferred_type in expected_types ->
          inferred_type

        original_type in expected_types ->
          original_type

        IntentRegistry.weather_intent?(intent) or IntentRegistry.navigation_intent?(intent) ->
          Enum.find(expected_types, "location", &(&1 in ["location", "city"]))

        IntentRegistry.introduction_intent?(intent) ->
          Enum.find(expected_types, "person", &(&1 in ["person", "name"]))

        IntentRegistry.music_intent?(intent) ->
          Enum.find(expected_types, inferred_type, &(&1 in ["music-artist", "song", "album"]))

        expected_types != [] ->
          hd(expected_types)

        true ->
          inferred_type
      end

    entity
    |> Map.put(:entity, final_type)
    |> Map.put(:entity_type, final_type)
    |> Map.put(:disambiguation_source, :type_inferrer)
    |> Map.put(:disambiguation_confidence, type_confidence)
  end

  @doc "Detect if the context suggests an introduction pattern.\n\nReturns a confidence score (0.0 to 1.0) indicating how likely\nthis is an introduction context.\n\nUses multiple signals:\n- POS tag patterns (PRON + VERB before entity)\n- Text-based patterns (\"I'm [Name]\", \"My name is [Name]\", etc.)\n- Discourse indicators (self-referential)\n- Speech act context (greeting)\n"
  def introduction_confidence(pos_tagged, entity_position, context) do
    entity_value = extract_entity_value_from_context(context)
    original_text = Map.get(context, :original_text, "")

    features = %{
      pron_verb_pattern: has_pron_verb_before?(pos_tagged, entity_position),
      text_intro_pattern: text_has_introduction_pattern?(original_text, entity_value),
      self_referential: self_referential?(context),
      greeting_context: greeting_context?(context)
    }

    text_intro_score =
      if features.text_intro_pattern do
        0.7
      else
        0.0
      end

    pron_verb_score =
      if features.pron_verb_pattern do
        0.5
      else
        0.0
      end

    self_ref_score =
      if features.self_referential do
        0.2
      else
        0.0
      end

    greeting_score =
      if features.greeting_context do
        0.1
      else
        0.0
      end

    pattern_score = max(text_intro_score, pron_verb_score)
    score = pattern_score + self_ref_score + greeting_score
    Float.round(min(score, 1.0), 4)
  end

  defp extract_entity_value_from_context(context) do
    Map.get(context, :entity_value, "")
  end

  # Introduction pattern prefixes as token sequences (entity token follows these)
  @introduction_prefixes [
    ~w(i am),
    ~w(im),
    ~w(my name is),
    ~w(name is),
    ~w(call me),
    ~w(called),
    ~w(i go by),
    ~w(go by),
    ~w(this is)
  ]

  @doc "Check if text contains an introduction pattern with the given entity value.\n\nDetects patterns like:\n- \"I'm [Name]\" / \"I am [Name]\"\n- \"My name is [Name]\"\n- \"This is [Name]\" (when self-referential)\n- \"Call me [Name]\"\n- \"I go by [Name]\"\n- \"[Name] here\" (at start)\n"
  def text_has_introduction_pattern?(text, entity_value)
      when is_binary(text) and is_binary(entity_value) do
    if entity_value == "" or String.length(entity_value) < 2 do
      false
    else
      tokens = Tokenizer.tokenize_normalized(text)
      entity_tokens = Tokenizer.tokenize_normalized(entity_value)

      # Check "prefix + entity" patterns
      prefix_match =
        Enum.any?(@introduction_prefixes, fn prefix ->
          pattern = prefix ++ entity_tokens
          contains_subsequence?(tokens, pattern)
        end)

      # Check "entity + here" pattern (e.g. "Austin here")
      suffix_match = contains_subsequence?(tokens, entity_tokens ++ ~w(here))

      prefix_match or suffix_match
    end
  end

  def text_has_introduction_pattern?(_, _) do
    false
  end

  defp contains_subsequence?(tokens, pattern) when is_list(tokens) and is_list(pattern) do
    pattern_len = length(pattern)

    if pattern_len == 0 or length(tokens) < pattern_len do
      false
    else
      tokens
      |> Enum.chunk_every(pattern_len, 1, :discard)
      |> Enum.any?(fn window -> window == pattern end)
    end
  end

  defp extract_features(entity, pos_tagged, context) do
    entity_pos = get_entity_position(entity)
    intent = Map.get(context, :intent, "")

    %{
      preceding_pron: count_preceding_tag(pos_tagged, entity_pos, "PRON"),
      preceding_verb: count_preceding_tag(pos_tagged, entity_pos, "VERB"),
      pron_verb_adjacent: has_pron_verb_before?(pos_tagged, entity_pos),
      self_referential: self_referential?(context),
      greeting_context: greeting_context?(context),
      question_context: question_context?(context),
      command_context: command_context?(context),
      weather_intent: weather_intent?(context),
      music_intent: music_intent?(context),
      navigation_intent: navigation_intent?(context),
      device_intent: device_intent?(context),
      expected_entity_types: IntentRegistry.expected_entity_types(intent),
      intent: intent,
      context_type: detect_context_type(context, pos_tagged, entity_pos)
    }
  end

  defp get_entity_position(entity) do
    Map.get(entity, :start_pos) || Map.get(entity, "start_pos") || 0
  end

  defp get_entity_position(entity, pos_tagged) when is_list(pos_tagged) do
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    entity_value_lower = String.downcase(entity_value)

    pos_tagged
    |> Enum.with_index()
    |> Enum.find_value(fn {{token, _tag}, idx} ->
      if String.downcase(token) == entity_value_lower do
        idx
      end
    end) || 0
  end

  defp get_entity_position(_entity, _pos_tagged) do
    0
  end

  defp get_entity_types(entity) do
    cond do
      is_list(Map.get(entity, :types)) ->
        entity.types

      is_map(entity) and Map.has_key?(entity, :entity_type) ->
        [entity]

      true ->
        []
    end
  end

  defp count_preceding_tag(pos_tagged, entity_pos, target_tag) when is_list(pos_tagged) do
    preceding =
      pos_tagged
      |> Enum.take(entity_pos)
      |> Enum.take(-3)

    Enum.count(preceding, fn
      {_token, tag} -> String.upcase(to_string(tag)) == target_tag
      _ -> false
    end)
  end

  defp has_pron_verb_before?(pos_tagged, entity_pos) when is_list(pos_tagged) do
    preceding =
      pos_tagged
      |> Enum.take(entity_pos)
      |> Enum.take(-3)
      |> Enum.map(fn
        {_token, tag} -> String.upcase(to_string(tag))
        tag when is_binary(tag) -> String.upcase(tag)
        _ -> "X"
      end)

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

  defp has_pron_verb_before?(_, _) do
    false
  end

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
    IntentRegistry.device_intent?(intent)
  end

  defp detect_context_type(context, pos_tagged, entity_pos) do
    intent = Map.get(context, :intent, "")
    domain = IntentRegistry.domain(intent)

    cond do
      IntentRegistry.introduction_intent?(intent) ->
        :introduction

      has_pron_verb_before?(pos_tagged, entity_pos) and self_referential?(context) ->
        :introduction

      domain == :device ->
        :device

      domain == :music ->
        :music

      domain in [:weather, :navigation] ->
        :location_query

      IntentRegistry.device_intent?(intent) ->
        :device

      true ->
        :default
    end
  end

  defp score_type(type_info, features, _context) do
    entity_type = get_type_name(type_info)
    context_type = features.context_type
    expected_types = Map.get(features, :expected_entity_types, [])

    dynamic_score =
      if entity_type in expected_types do
        0.8
      else
        0.0
      end

    preferences = Map.get(@context_preferences, context_type, @context_preferences.default)
    static_score = Map.get(preferences, entity_type, 0.3)

    base_score =
      if expected_types != [] and dynamic_score > 0 do
        dynamic_score
      else
        static_score
      end

    boost = calculate_feature_boost(entity_type, features)

    base_score + boost
  end

  defp calculate_feature_boost(entity_type, features) do
    cond do
      features.pron_verb_adjacent and features.self_referential and
          EntityTypes.is_person_type?(entity_type) ->
        0.5

      features.device_intent and EntityTypes.is_device_type?(entity_type) ->
        0.4

      features.weather_intent and EntityTypes.is_location_type?(entity_type) ->
        0.4

      features.music_intent and EntityTypes.is_music_type?(entity_type) ->
        0.4

      features.greeting_context and EntityTypes.is_person_type?(entity_type) ->
        0.3

      true ->
        0.0
    end
  end

  defp get_type_name(type_info) when is_map(type_info) do
    Map.get(type_info, :entity_type, "unknown")
  end

  defp get_type_name(_) do
    "unknown"
  end

  defp select_type(entity, selected_type) when is_map(selected_type) do
    base_merge = %{
      entity: get_type_name(selected_type),
      entity_type: get_type_name(selected_type),
      value:
        Map.get(selected_type, :value) || Map.get(entity, :value) || Map.get(entity, "value"),
      disambiguation_source: :context_analysis
    }

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

  defp select_type(entity, _) do
    entity
  end
end
