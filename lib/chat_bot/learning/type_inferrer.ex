defmodule ChatBot.Learning.TypeInferrer do
  @moduledoc """
  Infers entity types from context using learned patterns.

  This module does NOT use regex or hard-coded strings.
  Instead, it learns type associations from:
  - Co-occurrence with known entities
  - POS tag context patterns
  - Semantic role patterns

  Over time, it builds a statistical model of what contexts
  are associated with what entity types.
  """

  require Logger

  alias ChatBot.ML.Gazetteer
  alias ChatBot.Learning.WorldManager

  @ets_patterns :type_inferrer_patterns
  @ets_cooccurrence :type_inferrer_cooccurrence

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Initializes the type inferrer ETS tables.
  Call this during application startup.
  """
  def init do
    create_tables()
    :ok
  end

  @doc """
  Infers the entity type for a value based on its context.

  Uses learned patterns from previous observations. Returns
  {inferred_type, confidence} where confidence is 0.0 to 1.0.

  Requires world_id to ensure proper data isolation.
  """
  def infer_type(value, context_tokens, context_tags, world_id) when is_binary(world_id) do
    # Build context features from POS tags
    context_features = build_context_features(context_tokens, context_tags)

    # Check learned patterns (world-scoped)
    pattern_match = match_learned_patterns(context_features, world_id)

    # Check co-occurrence with known entities (world-scoped)
    cooccurrence_match = match_cooccurrences(context_tokens, world_id)

    # Combine evidence
    combine_evidence(value, pattern_match, cooccurrence_match)
  end

  def infer_type(_value, _context_tokens, _context_tags, nil) do
    raise ArgumentError, "world_id is required for infer_type/4 to ensure data isolation"
  end

  @doc """
  Learns type patterns from a known entity occurrence.

  Call this when you see a known entity in context - it will
  learn to associate that context pattern with the entity type.

  Requires world_id to ensure proper data isolation.
  """
  def learn_from_known_entity(entity_type, context_tokens, context_tags, world_id)
      when is_binary(world_id) do
    # Extract context features
    context_features = build_context_features(context_tokens, context_tags)

    # Update pattern counts (world-scoped)
    update_pattern_counts(entity_type, context_features, world_id)

    # Learn co-occurrence with other entities in context (world-scoped)
    learn_cooccurrences(entity_type, context_tokens, world_id)

    # Emit learning event
    WorldManager.record_event(world_id, :context_pattern_learned, %{
      entity_type: entity_type,
      features: context_features
    })

    :ok
  end

  def learn_from_known_entity(_entity_type, _context_tokens, _context_tags, nil) do
    raise ArgumentError, "world_id is required for learn_from_known_entity/4 to ensure data isolation"
  end

  @doc """
  Gets the current learned patterns for an entity type.
  Useful for debugging and introspection.

  Requires world_id to ensure proper data isolation.
  """
  def get_patterns_for_type(entity_type, world_id) when is_binary(world_id) do
    key = {world_id, entity_type}

    try do
      :ets.lookup(@ets_patterns, key)
      |> case do
        [{^key, patterns}] -> patterns
        [] -> %{}
      end
    rescue
      ArgumentError -> %{}
    end
  end

  def get_patterns_for_type(_entity_type, nil) do
    raise ArgumentError, "world_id is required for get_patterns_for_type/2 to ensure data isolation"
  end

  @doc """
  Gets all learned entity types for a world.

  Requires world_id to ensure proper data isolation.
  """
  def get_learned_types(world_id) when is_binary(world_id) do
    try do
      :ets.tab2list(@ets_patterns)
      |> Enum.filter(fn {{w, _type}, _patterns} -> w == world_id end)
      |> Enum.map(fn {{_world, type}, _patterns} -> type end)
    rescue
      ArgumentError -> []
    end
  end

  def get_learned_types(nil) do
    raise ArgumentError, "world_id is required for get_learned_types/1 to ensure data isolation"
  end

  @doc """
  Gets co-occurrence statistics for an entity type.

  Requires world_id to ensure proper data isolation.
  """
  def get_cooccurrences(entity_type, world_id) when is_binary(world_id) do
    key = {world_id, entity_type}

    try do
      :ets.lookup(@ets_cooccurrence, key)
      |> case do
        [{^key, cooccurrences}] -> cooccurrences
        [] -> %{}
      end
    rescue
      ArgumentError -> %{}
    end
  end

  def get_cooccurrences(_entity_type, nil) do
    raise ArgumentError, "world_id is required for get_cooccurrences/2 to ensure data isolation"
  end

  @doc """
  Exports all learned data for persistence.
  """
  def export_learned_data do
    patterns =
      try do
        :ets.tab2list(@ets_patterns)
        |> Enum.into(%{})
      rescue
        ArgumentError -> %{}
      end

    cooccurrences =
      try do
        :ets.tab2list(@ets_cooccurrence)
        |> Enum.into(%{})
      rescue
        ArgumentError -> %{}
      end

    %{
      patterns: patterns,
      cooccurrences: cooccurrences,
      exported_at: DateTime.utc_now()
    }
  end

  @doc """
  Imports previously exported learned data.
  """
  def import_learned_data(data) when is_map(data) do
    create_tables()

    # Import patterns
    patterns = Map.get(data, :patterns, %{})

    Enum.each(patterns, fn {type, pattern_data} ->
      :ets.insert(@ets_patterns, {type, pattern_data})
    end)

    # Import co-occurrences
    cooccurrences = Map.get(data, :cooccurrences, %{})

    Enum.each(cooccurrences, fn {type, cooc_data} ->
      :ets.insert(@ets_cooccurrence, {type, cooc_data})
    end)

    :ok
  end

  @doc """
  Clears all learned data.
  """
  def clear do
    try do
      :ets.delete_all_objects(@ets_patterns)
      :ets.delete_all_objects(@ets_cooccurrence)
      :ok
    rescue
      ArgumentError -> :ok
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp create_tables do
    # Pattern table: entity_type -> %{feature -> count}
    unless :ets.whereis(@ets_patterns) != :undefined do
      :ets.new(@ets_patterns, [:set, :public, :named_table, read_concurrency: true])
    end

    # Co-occurrence table: entity_type -> %{other_type -> count}
    unless :ets.whereis(@ets_cooccurrence) != :undefined do
      :ets.new(@ets_cooccurrence, [:set, :public, :named_table, read_concurrency: true])
    end
  end

  defp build_context_features(context_tokens, context_tags) do
    # Build features from POS tag patterns
    # These are abstract features, not based on specific words

    features = []

    # Feature: preceding POS tags (up to 3)
    # Find the entity position (assumed to be marked or middle)
    entity_idx = div(length(context_tags), 2)

    preceding_tags = Enum.take(context_tags, entity_idx)
    following_tags = Enum.drop(context_tags, entity_idx + 1)

    # Preceding tag features
    features =
      preceding_tags
      |> Enum.with_index()
      |> Enum.reduce(features, fn {tag, offset}, acc ->
        [{:prev_tag, -offset - 1, tag} | acc]
      end)

    # Following tag features
    features =
      following_tags
      |> Enum.with_index()
      |> Enum.reduce(features, fn {tag, offset}, acc ->
        [{:next_tag, offset + 1, tag} | acc]
      end)

    # Bi-gram features (adjacent tag pairs)
    bigrams =
      context_tags
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] -> {:bigram, a, b} end)

    # Position-independent features
    has_verb = Enum.any?(context_tags, &(&1 in ["VERB", "AUX"]))
    has_prep = Enum.any?(context_tags, &(&1 == "ADP"))
    has_det = Enum.any?(context_tags, &(&1 == "DET"))

    position_features = [
      {:has_verb, has_verb},
      {:has_prep, has_prep},
      {:has_det, has_det}
    ]

    # Token-based features (without using specific strings)
    token_features = build_token_features(context_tokens)

    features ++ bigrams ++ position_features ++ token_features
  end

  defp build_token_features(tokens) do
    # Build features based on token properties, not content
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      text = if is_map(token), do: Map.get(token, :text, ""), else: to_string(token)

      [
        {:token_length, idx, String.length(text)},
        {:token_capitalized, idx, capitalized?(text)},
        {:token_all_caps, idx, all_caps?(text)}
      ]
    end)
  end

  defp match_learned_patterns(context_features, world_id) do
    # Score each known entity type against the features (world-scoped)
    try do
      :ets.tab2list(@ets_patterns)
      |> Enum.filter(fn {{w, _type}, _patterns} -> w == world_id end)
      |> Enum.map(fn {{_world, entity_type}, patterns} ->
        score = calculate_pattern_score(context_features, patterns)
        {entity_type, score}
      end)
      |> Enum.filter(fn {_type, score} -> score > 0 end)
      |> Enum.sort_by(fn {_type, score} -> score end, :desc)
    rescue
      ArgumentError -> []
    end
  end

  defp calculate_pattern_score(context_features, patterns) do
    # Calculate how well the context features match learned patterns
    # Uses log-likelihood style scoring

    total_pattern_count = Map.get(patterns, :_total, 1)

    context_features
    |> Enum.reduce(0.0, fn feature, score ->
      feature_key = feature_to_key(feature)
      feature_count = Map.get(patterns, feature_key, 0)

      if feature_count > 0 do
        # Add log probability (with smoothing)
        score + :math.log((feature_count + 1) / (total_pattern_count + 1))
      else
        score
      end
    end)
  end

  defp match_cooccurrences(context_tokens, world_id) do
    # Check if any tokens in context are known entities
    # and use their types to infer this entity's type

    token_texts =
      Enum.map(context_tokens, fn token ->
        if is_map(token), do: Map.get(token, :text, ""), else: to_string(token)
      end)

    # Look up each token in gazetteer (world-scoped)
    known_entities =
      token_texts
      |> Enum.flat_map(fn text ->
        types = Gazetteer.lookup_all_types(text, world_id)

        Enum.map(types, fn info ->
          Map.get(info, :entity_type) || Map.get(info, :type)
        end)
      end)
      |> Enum.filter(&(&1 != nil))
      |> Enum.frequencies()

    # Score based on co-occurrence patterns (world-scoped)
    if map_size(known_entities) == 0 do
      []
    else
      try do
        :ets.tab2list(@ets_cooccurrence)
        |> Enum.filter(fn {{w, _type}, _cooc_counts} -> w == world_id end)
        |> Enum.map(fn {{_world, entity_type}, cooc_counts} ->
          # Score based on how often this type co-occurs with found types
          score =
            Enum.reduce(known_entities, 0.0, fn {found_type, count}, acc ->
              cooc_count = Map.get(cooc_counts, found_type, 0)
              acc + cooc_count * count
            end)

          {entity_type, score}
        end)
        |> Enum.filter(fn {_type, score} -> score > 0 end)
        |> Enum.sort_by(fn {_type, score} -> score end, :desc)
      rescue
        ArgumentError -> []
      end
    end
  end

  defp combine_evidence(_value, pattern_matches, cooccurrence_matches) do
    # Combine evidence from patterns and co-occurrences

    # Merge scores for same types
    all_matches = pattern_matches ++ cooccurrence_matches

    if length(all_matches) == 0 do
      {"unknown", 0.0}
    else
      # Group by type and sum scores
      type_scores =
        all_matches
        |> Enum.group_by(fn {type, _score} -> type end)
        |> Enum.map(fn {type, matches} ->
          total_score = Enum.sum(Enum.map(matches, fn {_t, s} -> s end))
          {type, total_score}
        end)
        |> Enum.sort_by(fn {_type, score} -> score end, :desc)

      case type_scores do
        [{best_type, best_score} | rest] ->
          # Calculate confidence based on score margin
          second_score =
            case rest do
              [{_, s} | _] -> s
              [] -> 0
            end

          confidence = calculate_confidence(best_score, second_score, length(all_matches))
          {best_type, confidence}

        [] ->
          {"unknown", 0.0}
      end
    end
  end

  defp calculate_confidence(best_score, second_score, num_matches) do
    # Confidence based on:
    # 1. Absolute score (more evidence = higher confidence)
    # 2. Margin over second best (clearer winner = higher confidence)

    base_confidence = :math.tanh(abs(best_score) / 10) * 0.5

    margin =
      if second_score != 0 do
        (best_score - second_score) / abs(second_score)
      else
        1.0
      end

    margin_bonus = min(margin * 0.3, 0.3)

    # Bonus for having multiple matches
    evidence_bonus = min(num_matches * 0.05, 0.2)

    min(base_confidence + margin_bonus + evidence_bonus, 1.0)
  end

  defp update_pattern_counts(entity_type, context_features, world_id) do
    key = {world_id, entity_type}

    try do
      current =
        case :ets.lookup(@ets_patterns, key) do
          [{^key, patterns}] -> patterns
          [] -> %{_total: 0}
        end

      # Increment counts for each feature
      updated =
        Enum.reduce(context_features, current, fn feature, acc ->
          feature_key = feature_to_key(feature)
          Map.update(acc, feature_key, 1, &(&1 + 1))
        end)

      # Increment total
      updated = Map.update(updated, :_total, 1, &(&1 + 1))

      :ets.insert(@ets_patterns, {key, updated})
      :ok
    rescue
      ArgumentError ->
        create_tables()
        update_pattern_counts(entity_type, context_features, world_id)
    end
  end

  defp learn_cooccurrences(entity_type, context_tokens, world_id) do
    # Find other known entities in context (world-scoped)
    token_texts =
      Enum.map(context_tokens, fn token ->
        if is_map(token), do: Map.get(token, :text, ""), else: to_string(token)
      end)

    other_types =
      token_texts
      |> Enum.flat_map(fn text ->
        types = Gazetteer.lookup_all_types(text, world_id)

        Enum.map(types, fn info ->
          Map.get(info, :entity_type) || Map.get(info, :type)
        end)
      end)
      |> Enum.filter(&(&1 != nil and &1 != entity_type))
      |> Enum.frequencies()

    if map_size(other_types) > 0 do
      key = {world_id, entity_type}

      try do
        current =
          case :ets.lookup(@ets_cooccurrence, key) do
            [{^key, coocs}] -> coocs
            [] -> %{}
          end

        updated =
          Enum.reduce(other_types, current, fn {other_type, count}, acc ->
            Map.update(acc, other_type, count, &(&1 + count))
          end)

        :ets.insert(@ets_cooccurrence, {key, updated})
      rescue
        ArgumentError ->
          create_tables()
          learn_cooccurrences(entity_type, context_tokens, world_id)
      end
    end

    :ok
  end

  defp feature_to_key(feature) do
    # Convert feature tuple to a string key for storage
    case feature do
      {:prev_tag, offset, tag} -> "prev_tag:#{offset}:#{tag}"
      {:next_tag, offset, tag} -> "next_tag:#{offset}:#{tag}"
      {:bigram, a, b} -> "bigram:#{a}:#{b}"
      {:has_verb, val} -> "has_verb:#{val}"
      {:has_prep, val} -> "has_prep:#{val}"
      {:has_det, val} -> "has_det:#{val}"
      {:token_length, idx, len} -> "token_len:#{idx}:#{len}"
      {:token_capitalized, idx, val} -> "token_cap:#{idx}:#{val}"
      {:token_all_caps, idx, val} -> "token_allcaps:#{idx}:#{val}"
      other -> inspect(other)
    end
  end

  defp capitalized?(text) when is_binary(text) and byte_size(text) > 0 do
    first = String.first(text)
    first == String.upcase(first) and first != String.downcase(first)
  end

  defp capitalized?(_), do: false

  defp all_caps?(text) when is_binary(text) and byte_size(text) > 0 do
    text == String.upcase(text) and text != String.downcase(text)
  end

  defp all_caps?(_), do: false
end
