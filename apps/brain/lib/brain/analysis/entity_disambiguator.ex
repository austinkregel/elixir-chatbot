defmodule Brain.Analysis.EntityDisambiguator do
  @moduledoc "Disambiguates entities when multiple types are possible,\nusing speech act, discourse, and POS-tagged syntactic context.\n\nWhen the gazetteer returns multiple possible entity types for the same\ntext (e.g., \"Austin\" could be a person or a location), this module\nuses contextual features to determine the most likely interpretation.\n\nAlso handles cases where a single-type entity (e.g., \"Nice\" as location)\nis being used as a proper noun/name in context (e.g., \"I'm Nice\"),\nrecognizing proper noun usage and mapping it to the appropriate entity type.\n\n## Features Used\n\n- **POS tags**: What part of speech precedes/follows the entity\n- **Discourse indicators**: Is this a self-referential statement?\n- **Speech act context**: Is this a greeting, question, command?\n- **Intent hints**: What domain does the intent belong to?\n- **ChunkProfile**: Domain from feature-vector classification for entity type expectations\n\n## Usage\n\n    # With POS-tagged tokens\n    pos_tagged = [{\"I\", \"PRON\"}, {\"am\", \"VERB\"}, {\"Austin\", \"PROPN\"}]\n    entities = [%{value: \"Austin\", types: [person_info, location_info], ...}]\n    context = %{discourse: discourse_result, speech_act: speech_act_result}\n\n    disambiguated = EntityDisambiguator.disambiguate(entities, pos_tagged, context)\n\n"

  # World.TypeInferrer is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.TypeInferrer}

  alias Brain.Analysis
  alias Brain.Graph.Reader
  require Logger

  alias Analysis.{ChunkProfile, TypeHierarchy}
  alias Brain.Analysis.Pipeline
  alias World.TypeInferrer

  @external_resource Path.join(:code.priv_dir(:brain), "analysis/entity_types.json")
  @ambiguous_types (
    Path.join(:code.priv_dir(:brain), "analysis/entity_types.json")
    |> File.read!()
    |> Jason.decode!()
    |> Map.get("disambiguation_groups", %{})
    |> Map.values()
    |> List.flatten()
    |> MapSet.new()
  )

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

    default_propn_type = TypeHierarchy.config("default_propn_type", "person")

    cond do
      length(types) <= 1 and requires_inference?(entity_type) ->
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 do
          if TypeHierarchy.is_a?(entity_type, default_propn_type) or entity_type == default_propn_type do
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

        is_not_person_type = not TypeHierarchy.compatible?(single_type_name, default_propn_type)

        if intro_confidence >= 0.7 and is_not_person_type do
          create_person_type_from_intro(entity, intro_confidence)
        else
          select_type(entity, single_type)
        end

      true ->
        intro_confidence = introduction_confidence(pos_tagged, entity_position, enriched_context)

        if intro_confidence >= 0.7 do
          person_type = Enum.find(types, fn t ->
            tn = get_type_name(t)
            tn == default_propn_type or TypeHierarchy.is_a?(tn, default_propn_type)
          end)

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
    default_type = TypeHierarchy.config("default_propn_type", "person")

    name_type = %{
      entity_type: default_type,
      entity: default_type,
      value: Map.get(entity, :value) || Map.get(entity, "value"),
      confidence: intro_confidence,
      disambiguation_reason: "introduction_pattern"
    }

    select_type(entity, name_type)
  end

  @doc """
  Check if an entity type requires context-based disambiguation.

  Types that require inference are those listed in the `disambiguation_groups`
  from `entity_types.json`, plus any type with the "ambiguous_" prefix.
  """
  def requires_inference?(entity_type) when is_binary(entity_type) do
    String.starts_with?(entity_type, "ambiguous_") or
      MapSet.member?(@ambiguous_types, entity_type)
  end

  def requires_inference?(_) do
    false
  end

  @doc "Infer the entity type using intent context and TypeInferrer.\n\nPrimary: Uses ChunkProfile domain to derive expected entity types for\nthe current intent's requirements.\n\nFallback: TypeInferrer's learned patterns when no intent context or\nwhen TypeInferrer returns a type that matches expected types.\n\nRequires world_id in context for proper data isolation.\n"
  def infer_type_with_type_inferrer(entity, pos_tagged, context) do
    entity_value = Map.get(entity, :value) || Map.get(entity, "value") || ""
    original_type = Map.get(entity, :entity_type) || ""
    intent = Map.get(context, :intent, "")
    world_id = Map.get(context, :world_id) || "default"
    profile = Map.get(context, :profile)

    expected_types =
      case profile do
        %ChunkProfile{domain: domain} when domain != :unknown ->
          Pipeline.expected_entity_types_from_domain(domain)

        _ ->
          Pipeline.expected_entity_types_from_domain(domain_from_intent(intent))
      end

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

        expected_types != [] ->
          compatible = Enum.find(expected_types, fn et ->
            TypeHierarchy.compatible?(inferred_type, et) or
              TypeHierarchy.compatible?(original_type, et)
          end)

          compatible || hd(expected_types)

        true ->
          inferred_type
      end

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

  All signals come from trained classifiers -- no hardcoded token matching:
  - Intent classifier: classified as introduction domain
  - POS tagger output: PRON + VERB preceding entity (trained model tags)
  - Discourse analyzer: self-referential indicator
  - Speech act classifier: greeting/expressive category
  """
  def introduction_confidence(pos_tagged, entity_position, context) do
    intent = Map.get(context, :intent, "")
    profile = Map.get(context, :profile)

    intent_score =
      case profile do
        %ChunkProfile{domain: :smalltalk, speech_act_subtype: sub}
        when sub in [:statement, :greeting] ->
          0.7

        _ ->
          if String.starts_with?(to_string(intent || ""), "greeting"), do: 0.7, else: 0.0
      end

    # Signal 2: Trained POS tagger output shows PRON+VERB before the entity
    pron_verb_score =
      if has_pron_verb_before?(pos_tagged, entity_position) do
        0.5
      else
        0.0
      end

    # Signal 3: Discourse analyzer flagged self-referential (trained model output)
    self_ref_score =
      if self_referential?(context) do
        0.2
      else
        0.0
      end

    # Signal 4: Speech act classifier says greeting (trained model output)
    greeting_score =
      if greeting_context?(context) do
        0.1
      else
        0.0
      end

    primary_score = max(intent_score, pron_verb_score)
    score = primary_score + self_ref_score + greeting_score
    Float.round(min(score, 1.0), 4)
  end

  defp extract_features(_entity, _pos_tagged, context) do
    intent = Map.get(context, :intent, "")
    profile = Map.get(context, :profile)

    {domain, expected_types} =
      case profile do
        %ChunkProfile{domain: d} when d != :unknown ->
          {d, Pipeline.expected_entity_types_from_domain(d)}

        _ ->
          d = domain_from_intent(intent)
          {d, Pipeline.expected_entity_types_from_domain(d)}
      end

    %{
      expected_entity_types: expected_types,
      intent: intent,
      domain: domain
    }
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

    :self_referential in indicators or
      "self_referential" in indicators or
      :first_person in indicators or
      "first_person" in indicators
  end

  defp greeting_context?(context) do
    speech_act = Map.get(context, :speech_act) || %{}

    Map.get(speech_act, :category) == :expressive and
      Map.get(speech_act, :sub_type) in [:greeting, :nice_to_meet]
  end


  defp score_type(type_info, features, context) do
    entity_type = get_type_name(type_info)
    domain = features.domain
    expected_types = Map.get(features, :expected_entity_types, [])

    dynamic_score =
      if entity_type in expected_types do
        0.8
      else
        0.0
      end

    preferences = Map.get(@context_preferences, domain, @context_preferences.default)
    static_score = Map.get(preferences, entity_type, 0.3)

    base_score =
      if expected_types != [] and dynamic_score > 0 do
        dynamic_score
      else
        static_score
      end

    atlas_boost = atlas_co_occurrence_boost(type_info, context)
    lexicon_boost = lexicon_sense_boost(type_info, context)

    base_score + atlas_boost + lexicon_boost
  end

  defp lexicon_sense_boost(type_info, context) do
    if Process.whereis(Brain.ML.Lexicon) do
      entity_value = Map.get(type_info, :value) || ""
      entity_type = get_type_name(type_info)
      original_text = Map.get(context, :original_text, "")

      if entity_value == "" or original_text == "" do
        0.0
      else
        senses = Brain.ML.Lexicon.senses(entity_value)

        if senses == [] do
          0.0
        else
          context_words =
            original_text
            |> Brain.ML.Tokenizer.tokenize_normalized(min_length: 3)
            |> MapSet.new()

          best_match =
            senses
            |> Enum.map(fn sense ->
              defn_words =
                sense.definition
                |> Brain.ML.Tokenizer.tokenize_normalized(min_length: 3)
                |> MapSet.new()

              overlap = MapSet.intersection(context_words, defn_words) |> MapSet.size()

              chain = Brain.ML.Lexicon.hypernym_chain(entity_value, sense.pos, max_depth: 5)
              type_match = if entity_type in chain, do: 1, else: 0

              overlap + type_match * 2
            end)
            |> Enum.max(fn -> 0 end)

          cond do
            best_match >= 3 -> 0.2
            best_match >= 1 -> 0.1
            true -> 0.0
          end
        end
      end
    else
      0.0
    end
  end

  defp atlas_co_occurrence_boost(type_info, context) do
    entity_value = Map.get(type_info, :value) || ""
    entity_type = get_type_name(type_info)

    if entity_value == "" do
      0.0
    else
      entity = %{entity_type: entity_type, value: entity_value}

      case Reader.entity_context([entity]) do
        [%{node: node, neighbors: neighbors}] when node != nil ->
          intent = Map.get(context, :intent, "")
          profile = Map.get(context, :profile)

          expected_types =
            case profile do
              %ChunkProfile{domain: domain} when domain != :unknown ->
                Pipeline.expected_entity_types_from_domain(domain)

              _ ->
                Pipeline.expected_entity_types_from_domain(domain_from_intent(intent))
            end
          neighbor_count = length(neighbors)

          # Check if neighbor labels/types align with what the intent expects.
          # e.g., for weather.query expecting ["location"], if "Austin" as location
          # has neighbors whose labels include Location-related types, that's a signal.
          intent_aligned =
            if expected_types != [] do
              Enum.count(neighbors, fn neighbor ->
                neighbor_labels = Map.get(neighbor, :labels, [])
                neighbor_props = Map.get(neighbor, :properties, %{})
                neighbor_type = Map.get(neighbor_props, "type", "")

                neighbor_type_indicators =
                  [neighbor_type | neighbor_labels]
                  |> Enum.map(&String.downcase(to_string(&1)))

                Enum.any?(expected_types, fn et ->
                  et_lower = String.downcase(et)
                  Enum.any?(neighbor_type_indicators, fn ind ->
                    ind == et_lower or String.contains?(ind, et_lower)
                  end)
                end)
              end)
            else
              0
            end

          # Intent-aligned neighbors are a strong disambiguation signal:
          # Atlas confirms this entity type co-occurs with the intent's expected types
          intent_boost =
            cond do
              intent_aligned >= 3 -> 0.35
              intent_aligned >= 1 -> 0.25
              true -> 0.0
            end

          # General existence in Atlas is a weaker but still useful signal
          existence_boost =
            cond do
              neighbor_count >= 5 -> 0.15
              neighbor_count >= 2 -> 0.1
              neighbor_count >= 1 -> 0.05
              true -> 0.0
            end

          max(intent_boost, existence_boost)

        _ ->
          0.0
      end
    end
  rescue
    _ -> 0.0
  catch
    :exit, _ -> 0.0
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

  defp domain_from_intent(nil), do: nil
  defp domain_from_intent(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [d, _] -> String.to_atom(d)
      _ -> nil
    end
  end
  defp domain_from_intent(_), do: nil
end
