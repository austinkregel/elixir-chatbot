defmodule World.TypeInferrer do
  @moduledoc "Infers entity types from context using learned patterns.\n\nThis module does NOT use regex or hard-coded strings.\nInstead, it learns type associations from:\n- Co-occurrence with known entities\n- POS tag context patterns\n- Semantic role patterns\n\nOver time, it builds a statistical model of what contexts\nare associated with what entity types.\n"

  require Logger

  alias Brain.ML.Gazetteer
  alias World.Manager, as: WorldManager

  @ets_patterns :type_inferrer_patterns
  @ets_cooccurrence :type_inferrer_cooccurrence

  @doc "Initializes the type inferrer ETS tables.\nCall this during application startup.\n"
  def init do
    create_tables()
    :ok
  end

  @doc "Infers the entity type for a value based on its context.\n\nUses learned patterns from previous observations. Returns\n{inferred_type, confidence} where confidence is 0.0 to 1.0.\n\nRequires world_id to ensure proper data isolation.\n"
  def infer_type(value, context_tokens, context_tags, world_id) when is_binary(world_id) do
    context_features = build_context_features(context_tokens, context_tags)
    pattern_match = match_learned_patterns(context_features, world_id)
    cooccurrence_match = match_cooccurrences(context_tokens, world_id)
    combine_evidence(value, pattern_match, cooccurrence_match)
  end

  def infer_type(_value, _context_tokens, _context_tags, nil) do
    raise ArgumentError, "world_id is required for infer_type/4 to ensure data isolation"
  end

  @doc "Learns type patterns from a known entity occurrence.\n\nCall this when you see a known entity in context - it will\nlearn to associate that context pattern with the entity type.\n\nRequires world_id to ensure proper data isolation.\n"
  def learn_from_known_entity(entity_type, context_tokens, context_tags, world_id)
      when is_binary(world_id) do
    context_features = build_context_features(context_tokens, context_tags)
    update_pattern_counts(entity_type, context_features, world_id)
    learn_cooccurrences(entity_type, context_tokens, world_id)

    WorldManager.record_event(world_id, :context_pattern_learned, %{
      entity_type: entity_type,
      features: context_features
    })

    :ok
  end

  def learn_from_known_entity(_entity_type, _context_tokens, _context_tags, nil) do
    raise ArgumentError,
          "world_id is required for learn_from_known_entity/4 to ensure data isolation"
  end

  @doc "Gets the current learned patterns for an entity type.\nUseful for debugging and introspection.\n\nRequires world_id to ensure proper data isolation.\n"
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
    raise ArgumentError,
          "world_id is required for get_patterns_for_type/2 to ensure data isolation"
  end

  @doc "Gets all learned entity types for a world.\n\nRequires world_id to ensure proper data isolation.\n"
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

  @doc "Gets co-occurrence statistics for an entity type.\n\nRequires world_id to ensure proper data isolation.\n"
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

  @doc "Exports all learned data for persistence.\n"
  def export_learned_data do
    patterns =
      try do
        :ets.tab2list(@ets_patterns)
        |> Enum.map(&serialize_ets_entry/1)
        |> Enum.into(%{})
      rescue
        ArgumentError -> %{}
      end

    cooccurrences =
      try do
        :ets.tab2list(@ets_cooccurrence)
        |> Enum.map(&serialize_ets_entry/1)
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

  defp serialize_ets_entry({{world_id, type}, data})
       when is_binary(world_id) and is_binary(type) do
    {"#{world_id}::#{type}", data}
  end

  defp serialize_ets_entry({key, data}) when is_binary(key) do
    {key, data}
  end

  defp serialize_ets_entry({key, data}) do
    {inspect(key), data}
  end

  defp deserialize_key(key) when is_binary(key) do
    case String.split(key, "::", parts: 2) do
      [world_id, type] -> {world_id, type}
      _ -> key
    end
  end

  @doc "Imports previously exported learned data.\n"
  def import_learned_data(data) when is_map(data) do
    create_tables()
    patterns = Map.get(data, :patterns) || Map.get(data, "patterns", %{})

    Enum.each(patterns, fn {key, pattern_data} ->
      :ets.insert(@ets_patterns, {deserialize_key(key), pattern_data})
    end)

    cooccurrences = Map.get(data, :cooccurrences) || Map.get(data, "cooccurrences", %{})

    Enum.each(cooccurrences, fn {key, cooc_data} ->
      :ets.insert(@ets_cooccurrence, {deserialize_key(key), cooc_data})
    end)

    :ok
  end

  @doc "Clears all learned data.\n"
  def clear do
    try do
      :ets.delete_all_objects(@ets_patterns)
      :ets.delete_all_objects(@ets_cooccurrence)
      :ok
    rescue
      ArgumentError -> :ok
    end
  end

  defp create_tables do
    unless :ets.whereis(@ets_patterns) != :undefined do
      :ets.new(@ets_patterns, [:set, :public, :named_table, read_concurrency: true])
    end

    unless :ets.whereis(@ets_cooccurrence) != :undefined do
      :ets.new(@ets_cooccurrence, [:set, :public, :named_table, read_concurrency: true])
    end
  end

  defp build_context_features(context_tokens, context_tags) do
    features = []
    entity_idx = div(length(context_tags), 2)

    preceding_tags = Enum.take(context_tags, entity_idx)
    following_tags = Enum.drop(context_tags, entity_idx + 1)

    features =
      preceding_tags
      |> Enum.with_index()
      |> Enum.reduce(features, fn {tag, offset}, acc ->
        [{:prev_tag, -offset - 1, tag} | acc]
      end)

    features =
      following_tags
      |> Enum.with_index()
      |> Enum.reduce(features, fn {tag, offset}, acc ->
        [{:next_tag, offset + 1, tag} | acc]
      end)

    bigrams =
      context_tags
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [a, b] -> {:bigram, a, b} end)

    has_verb = Enum.any?(context_tags, &(&1 in ["VERB", "AUX"]))
    has_prep = Enum.any?(context_tags, &(&1 == "ADP"))
    has_det = Enum.any?(context_tags, &(&1 == "DET"))

    position_features = [has_verb: has_verb, has_prep: has_prep, has_det: has_det]
    token_features = build_token_features(context_tokens)

    features ++ bigrams ++ position_features ++ token_features
  end

  defp build_token_features(tokens) do
    tokens
    |> Enum.with_index()
    |> Enum.flat_map(fn {token, idx} ->
      text =
        if is_map(token) do
          Map.get(token, :text, "")
        else
          to_string(token)
        end

      [
        {:token_length, idx, String.length(text)},
        {:token_capitalized, idx, capitalized?(text)},
        {:token_all_caps, idx, all_caps?(text)}
      ]
    end)
  end

  defp match_learned_patterns(context_features, world_id) do
    try do
      :ets.tab2list(@ets_patterns)
      |> Enum.filter(fn
        {{w, _type}, _patterns} when is_binary(w) -> w == world_id
        _ -> false
      end)
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
    total_pattern_count = Map.get(patterns, :_total, 1)

    context_features
    |> Enum.reduce(0.0, fn feature, score ->
      feature_key = feature_to_key(feature)
      feature_count = Map.get(patterns, feature_key, 0)

      if feature_count > 0 do
        score + :math.log((feature_count + 1) / (total_pattern_count + 1))
      else
        score
      end
    end)
  end

  defp match_cooccurrences(context_tokens, world_id) do
    token_texts =
      Enum.map(context_tokens, fn token ->
        if is_map(token) do
          Map.get(token, :text, "")
        else
          to_string(token)
        end
      end)

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

    if map_size(known_entities) == 0 do
      []
    else
      try do
        :ets.tab2list(@ets_cooccurrence)
        |> Enum.filter(fn
          {{w, _type}, _cooc_counts} when is_binary(w) -> w == world_id
          _ -> false
        end)
        |> Enum.map(fn {{_world, entity_type}, cooc_counts} ->
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
    all_matches = pattern_matches ++ cooccurrence_matches

    if all_matches == [] do
      {"unknown", 0.0}
    else
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
    base_confidence = :math.tanh(abs(best_score) / 10) * 0.5

    margin =
      if second_score != 0 do
        (best_score - second_score) / abs(second_score)
      else
        1.0
      end

    margin_bonus = min(margin * 0.3, 0.3)
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

      updated =
        Enum.reduce(context_features, current, fn feature, acc ->
          feature_key = feature_to_key(feature)
          Map.update(acc, feature_key, 1, &(&1 + 1))
        end)

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
    token_texts =
      Enum.map(context_tokens, fn token ->
        if is_map(token) do
          Map.get(token, :text, "")
        else
          to_string(token)
        end
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

  defp capitalized?(_) do
    false
  end

  defp all_caps?(text) when is_binary(text) and byte_size(text) > 0 do
    text == String.upcase(text) and text != String.downcase(text)
  end

  defp all_caps?(_) do
    false
  end
end
