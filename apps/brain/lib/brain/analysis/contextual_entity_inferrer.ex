defmodule Brain.Analysis.ContextualEntityInferrer do
  @moduledoc """
  Multi-pass contextual entity type narrowing.

  Uses intent hypotheses to narrow generic entity types along the IS-A
  hierarchy defined in `TypeHierarchy`. For example, a "person" entity
  in a music.play context can be narrowed to "artist" because
  artist IS-A person.

  ## Algorithm

  1. Find entities whose types are parent types in the hierarchy
     (e.g., "person" has children like "artist", "name")
  2. For each intent hypothesis (from top_k candidates):
     - Determine which unfilled slots could be filled by narrowing
     - Score each (entity, narrowed_type) pair using four signals
     - Compute an overall hypothesis score
  3. Apply the best hypothesis: narrow entity types and update intent if needed

  ## Signals

  The four scoring signals are all data-driven (no string matching):

  - **TypeInferrer**: Learned context patterns from World.TypeInferrer
  - **POS context**: POS tag sequence around the entity
  - **Syntactic role**: Position relative to the verb (object of imperative verb)
  - **Atlas context**: Entity neighborhood in knowledge_graph
  """

  alias Brain.Analysis.{TypeHierarchy, IntentRegistry, SlotDetector}
  alias Brain.ML.{POSTagger, Tokenizer}
  require Logger

  @compile {:no_warn_undefined, [World.TypeInferrer, World.Manager]}

  @doc """
  Run contextual type narrowing on entities using intent context.

  Returns `{updated_entities, intent, intent_details}` where entities
  may have narrowed types and intent may be revised if a different
  hypothesis scored higher.

  ## Options

  - `:world_id` -- World scope for TypeInferrer (default: "default")
  """
  def infer(text, entities, intent, intent_details, opts \\ []) do
    narrowable = find_narrowable_entities(entities, intent, intent_details)

    if narrowable == [] do
      {entities, intent, intent_details}
    else
      pos_tags = get_pos_tags(text)
      top_k = extract_intent_candidates(intent, intent_details)

      hypotheses =
        Enum.map(top_k, fn {candidate_intent, candidate_score} ->
          evaluate_hypothesis(
            candidate_intent, candidate_score,
            entities, narrowable, text, pos_tags, opts
          )
        end)
        |> Enum.reject(&is_nil/1)

      case select_best_hypothesis(hypotheses) do
        nil ->
          {entities, intent, intent_details}

        best ->
          apply_hypothesis(best, entities, intent, intent_details, text, opts)
      end
    end
  end

  # ============================================================================
  # Pass 1: Find Narrowable Entities
  # ============================================================================

  defp find_narrowable_entities(entities, intent, intent_details) do
    top_k = extract_intent_candidates(intent, intent_details)
    all_expected = collect_all_expected_types(top_k)

    Enum.filter(entities, fn entity ->
      entity_type = entity[:entity_type]

      entity_type != nil and
        TypeHierarchy.parent_type?(entity_type) and
        TypeHierarchy.narrowing_candidates(entity_type, all_expected) != []
    end)
  end

  defp collect_all_expected_types(top_k) do
    top_k
    |> Enum.flat_map(fn {intent, _score} ->
      IntentRegistry.expected_entity_types(intent)
    end)
    |> Enum.uniq()
  end

  # ============================================================================
  # Pass 2: Evaluate Hypotheses
  # ============================================================================

  defp evaluate_hypothesis(candidate_intent, candidate_score, entities, narrowable, text, pos_tags, opts) do
    entity_mappings = IntentRegistry.entity_mappings(candidate_intent)

    if entity_mappings == %{} do
      nil
    else
      current_slot_result = SlotDetector.detect(candidate_intent, entities)

      unfilled_slots =
        if current_slot_result.schema_name == "unknown" do
          filled_types = entities |> Enum.map(& &1[:entity_type]) |> MapSet.new()
          entity_mappings
          |> Map.keys()
          |> Enum.reject(fn slot ->
            slot_types = Map.get(entity_mappings, slot, [])
            Enum.any?(slot_types, &MapSet.member?(filled_types, &1))
          end)
        else
          (current_slot_result.missing_required ++ current_slot_result.missing_optional)
          |> Enum.uniq()
        end

      narrowings =
        Enum.flat_map(unfilled_slots, fn slot_name ->
          acceptable_types = Map.get(entity_mappings, slot_name, [])

          Enum.flat_map(narrowable, fn entity ->
            candidates = TypeHierarchy.narrowing_candidates(entity[:entity_type], acceptable_types)

            Enum.map(candidates, fn narrowed_type ->
              score = score_narrowing(entity, narrowed_type, text, pos_tags, opts)

              %{
                entity: entity,
                slot: slot_name,
                narrowed_type: narrowed_type,
                original_type: entity[:entity_type],
                score: score
              }
            end)
          end)
        end)
        |> Enum.sort_by(& &1.score, :desc)
        |> deduplicate_assignments()

      filled_count = Enum.count(narrowings, &(&1.score > 0.0))
      slot_fill_ratio = if length(unfilled_slots) > 0, do: filled_count / length(unfilled_slots), else: 1.0

      mean_quality =
        if narrowings == [] do
          0.0
        else
          scores = Enum.map(narrowings, & &1.score)
          Enum.sum(scores) / length(scores)
        end

      %{
        intent: candidate_intent,
        intent_score: candidate_score || 0.0,
        narrowings: narrowings,
        slot_fill_ratio: slot_fill_ratio,
        score: (candidate_score || 0.0) * (0.5 + 0.5 * slot_fill_ratio) * max(mean_quality, 0.1)
      }
    end
  end

  defp deduplicate_assignments(narrowings) do
    {_used_entities, _used_slots, result} =
      Enum.reduce(narrowings, {MapSet.new(), MapSet.new(), []}, fn narrowing, {used_entities, used_slots, acc} ->
        entity_key = entity_identity(narrowing.entity)

        if MapSet.member?(used_entities, entity_key) or MapSet.member?(used_slots, narrowing.slot) do
          {used_entities, used_slots, acc}
        else
          {MapSet.put(used_entities, entity_key),
           MapSet.put(used_slots, narrowing.slot),
           [narrowing | acc]}
        end
      end)

    Enum.reverse(result)
  end

  defp entity_identity(entity) do
    {entity[:value] || entity[:match], entity[:start_pos]}
  end

  # ============================================================================
  # Scoring Signals
  # ============================================================================

  defp score_narrowing(entity, narrowed_type, text, pos_tags, opts) do
    signals = [
      type_inferrer_signal(entity, narrowed_type, text, pos_tags, opts),
      pos_context_signal(entity, narrowed_type, pos_tags),
      syntactic_role_signal(entity, narrowed_type, pos_tags),
      atlas_context_signal(entity, narrowed_type),
      hierarchy_compatibility_signal(entity, narrowed_type)
    ]

    confirming = Enum.count(signals, &(&1 > 0.0))

    signal_factor =
      case confirming do
        0 -> 0.0
        1 -> 0.6
        2 -> 0.7
        3 -> 0.85
        _ -> 0.95
      end

    signal_factor
  end

  # Signal 1: TypeInferrer learned patterns
  defp type_inferrer_signal(entity, narrowed_type, text, pos_tags, opts) do
    world_id = Keyword.get(opts, :world_id, "default")
    entity_value = entity[:value] || entity[:match] || ""

    tokens = Tokenizer.tokenize_words(text)
    tags = Enum.map(pos_tags, fn {_token, tag} -> tag end)

    if Code.ensure_loaded?(World.TypeInferrer) and function_exported?(World.TypeInferrer, :infer_type, 4) do
      try do
        case World.TypeInferrer.infer_type(entity_value, tokens, tags, world_id) do
          {inferred_type, confidence} when is_binary(inferred_type) and confidence > 0.3 ->
            if inferred_type == narrowed_type, do: confidence, else: 0.0

          _ ->
            0.0
        end
      rescue
        _ -> 0.0
      catch
        :exit, _ -> 0.0
      end
    else
      0.0
    end
  end

  # Signal 2: POS context around the entity
  defp pos_context_signal(entity, _narrowed_type, pos_tags) do
    entity_value = entity[:value] || entity[:match] || ""
    entity_tokens = Tokenizer.tokenize_words(entity_value)

    entity_start_idx = find_entity_start(pos_tags, entity_tokens)

    if entity_start_idx == nil do
      0.0
    else
      preceding_tags = get_preceding_tags(pos_tags, entity_start_idx, 3)

      cond do
        has_imperative_verb?(preceding_tags) -> 0.6
        has_preposition?(preceding_tags) -> 0.5
        has_determiner?(preceding_tags) -> 0.3
        true -> 0.0
      end
    end
  end

  # Signal 3: Syntactic role relative to verb
  defp syntactic_role_signal(entity, _narrowed_type, pos_tags) do
    entity_value = entity[:value] || entity[:match] || ""
    entity_tokens = Tokenizer.tokenize_words(entity_value)
    entity_start_idx = find_entity_start(pos_tags, entity_tokens)

    if entity_start_idx == nil do
      0.0
    else
      verb_idx = find_nearest_verb(pos_tags, entity_start_idx)

      cond do
        verb_idx == nil -> 0.0
        verb_idx < entity_start_idx -> 0.7
        true -> 0.3
      end
    end
  end

  # Signal 4: Hierarchy compatibility — if the narrowed type IS-A the entity's current type,
  # that's structural confirmation that narrowing is valid
  defp hierarchy_compatibility_signal(entity, narrowed_type) do
    current_type = entity[:entity_type]

    if current_type && TypeHierarchy.is_a?(narrowed_type, current_type) do
      0.5
    else
      0.0
    end
  end

  # Signal 5: Atlas graph neighborhood
  defp atlas_context_signal(entity, narrowed_type) do
    graph_neighbors = Map.get(entity, :graph_neighbors, [])

    if graph_neighbors != [] do
      type_matches =
        Enum.count(graph_neighbors, fn neighbor ->
          n_type = Map.get(neighbor, :type, "")
          n_name = Map.get(neighbor, :name, "")

          String.downcase(n_type) == narrowed_type or
            String.downcase(n_name) == narrowed_type or
            TypeHierarchy.is_a?(String.downcase(n_type), narrowed_type)
        end)

      if type_matches > 0, do: min(type_matches * 0.3, 0.8), else: 0.0
    else
      if Map.get(entity, :graph_known, false) do
        0.2
      else
        0.0
      end
    end
  end

  # ============================================================================
  # POS Tag Helpers
  # ============================================================================

  defp get_pos_tags(text) do
    tokens = Tokenizer.tokenize_words(text)

    case POSTagger.get_model() do
      {:ok, model} -> POSTagger.predict(tokens, model)
      {:error, _} -> Enum.map(tokens, &{&1, "X"})
    end
  end

  defp find_entity_start(pos_tags, entity_tokens) do
    if entity_tokens == [] do
      nil
    else
      first_token = hd(entity_tokens) |> String.downcase()

      pos_tags
      |> Enum.with_index()
      |> Enum.find_value(fn {{token, _tag}, idx} ->
        if String.downcase(token) == first_token, do: idx
      end)
    end
  end

  defp get_preceding_tags(pos_tags, entity_start_idx, count) do
    start = max(entity_start_idx - count, 0)

    pos_tags
    |> Enum.slice(start, entity_start_idx - start)
    |> Enum.map(fn {_token, tag} -> tag end)
  end

  defp has_imperative_verb?(tags) do
    verb_tags = TypeHierarchy.config(["pos_tag_roles", "verb_tags"], [])
    Enum.any?(tags, &(&1 in verb_tags))
  end

  defp has_preposition?(tags) do
    prep_tag = TypeHierarchy.config(["pos_tag_roles", "preposition_tag"], "ADP")
    prep_tag in tags
  end

  defp has_determiner?(tags) do
    det_tag = TypeHierarchy.config(["pos_tag_roles", "determiner_tag"], "DET")
    det_tag in tags
  end

  defp find_nearest_verb(pos_tags, entity_start_idx) do
    verb_tags = TypeHierarchy.config(["pos_tag_roles", "verb_tags"], [])

    pos_tags
    |> Enum.with_index()
    |> Enum.filter(fn {{_token, tag}, _idx} -> tag in verb_tags end)
    |> Enum.map(fn {_, idx} -> idx end)
    |> Enum.min_by(fn idx -> abs(idx - entity_start_idx) end, fn -> nil end)
  end

  # ============================================================================
  # Pass 3: Select and Apply
  # ============================================================================

  defp select_best_hypothesis(hypotheses) do
    hypotheses
    |> Enum.filter(fn h -> h.score > 0.0 and h.narrowings != [] end)
    |> Enum.max_by(& &1.score, fn -> nil end)
  end

  defp apply_hypothesis(hypothesis, entities, current_intent, current_details, text, opts) do
    narrowing_map =
      hypothesis.narrowings
      |> Enum.filter(&(&1.score > 0.0))
      |> Map.new(fn n -> {entity_identity(n.entity), n} end)

    updated_entities =
      Enum.map(entities, fn entity ->
        key = entity_identity(entity)

        case Map.get(narrowing_map, key) do
          nil ->
            entity

          narrowing ->
            original_confidence = entity[:confidence] || 0.5

            signal_factor =
              case narrowing.score do
                s when s >= 0.95 -> 0.95
                s when s >= 0.85 -> 0.85
                s when s >= 0.7 -> 0.7
                _ -> 0.6
              end

            entity
            |> Map.put(:entity_type, narrowing.narrowed_type)
            |> Map.put(:original_type, narrowing.original_type)
            |> Map.put(:source, :type_narrowing)
            |> Map.put(:confidence, original_confidence * signal_factor)
            |> Map.put(:narrowing_signals, narrowing.score)
        end
      end)

    updated_intent =
      if hypothesis.intent != current_intent and hypothesis.intent_score > (Map.get(current_details, :intent_confidence) || 0.0) do
        hypothesis.intent
      else
        current_intent
      end

    updated_details =
      if updated_intent != current_intent do
        Map.merge(current_details, %{
          original_intent: current_intent,
          narrowing_revised: true,
          narrowing_score: hypothesis.score
        })
      else
        Map.put(current_details, :narrowing_applied, true)
      end

    narrowed_count = map_size(narrowing_map)

    if narrowed_count > 0 do
      Logger.debug(
        "ContextualEntityInferrer: narrowed #{narrowed_count} entities for intent #{updated_intent}"
      )

      report_narrowings_for_learning(hypothesis.narrowings, text, Keyword.get(opts, :world_id, "default"))
    end

    {updated_entities, updated_intent, updated_details}
  end

  # ============================================================================
  # Intent Candidate Extraction
  # ============================================================================

  defp extract_intent_candidates(intent, intent_details) when is_map(intent_details) do
    top_k = Map.get(intent_details, :top_k, [])

    normalized =
      Enum.flat_map(top_k, fn
        %{intent: i, score: s} -> [{i, s}]
        {i, s} when is_binary(i) -> [{i, s}]
        _ -> []
      end)

    intent_score =
      case normalized do
        [{^intent, s} | _] -> s
        _ -> Map.get(intent_details, :intent_confidence) || 0.5
      end

    if Enum.any?(normalized, fn {i, _} -> i == intent end) do
      normalized
    else
      [{intent, intent_score} | normalized]
    end
    |> Enum.take(5)
  end

  defp extract_intent_candidates(intent, _) do
    [{intent, 0.5}]
  end

  # ============================================================================
  # Learning Feedback
  # ============================================================================

  defp report_narrowings_for_learning(narrowings, text, world_id) do
    Task.start(fn ->
      tokens = Tokenizer.tokenize_words(text)

      tags =
        case POSTagger.get_model() do
          {:ok, model} -> POSTagger.predict_tags(tokens, model)
          {:error, _} -> Enum.map(tokens, fn _ -> "X" end)
        end

      Enum.each(narrowings, fn narrowing ->
        if narrowing.score > 0.0 do
          entity_value = narrowing.entity[:value] || narrowing.entity[:match] || ""

          if Code.ensure_loaded?(World.TypeInferrer) and
               function_exported?(World.TypeInferrer, :learn_from_known_entity, 4) do
            try do
              World.TypeInferrer.learn_from_known_entity(
                narrowing.narrowed_type, tokens, tags, world_id
              )
            rescue
              _ -> :ok
            catch
              :exit, _ -> :ok
            end
          end

          if Code.ensure_loaded?(World.Manager) and
               function_exported?(World.Manager, :add_candidate, 2) and
               Process.whereis(World.Manager) != nil do
            try do
              World.Manager.add_candidate(world_id, %{
                value: entity_value,
                inferred_type: narrowing.narrowed_type,
                confidence: narrowing.score,
                source: :type_narrowing,
                original_type: narrowing.original_type
              })
            rescue
              _ -> :ok
            catch
              :exit, _ -> :ok
            end
          end
        end
      end)
    end)
  end
end
