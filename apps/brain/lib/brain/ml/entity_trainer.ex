defmodule Brain.ML.EntityTrainer do
  @moduledoc "Entity recognition training using BIO tagging.\n\nConverts Dialogflow-format annotated data to BIO (Begin-Inside-Outside) format\nand trains a sequence model for entity boundary detection.\n\nBIO Tagging:\n- B-{type}: Beginning of an entity\n- I-{type}: Inside (continuation) of an entity\n- O: Outside any entity\n\nExample:\n  \"weather in New York today\"\n  [\"O\", \"O\", \"B-location\", \"I-location\", \"B-date\"]\n"

  alias Brain.ML
  require Logger

  alias ML.{DataLoaders, Tokenizer}
  @type bio_tag :: String.t()

  @type training_sequence :: %{
          tokens: [String.t()],
          tags: [bio_tag()],
          intent: String.t()
        }

  @type entity_model :: %{
          tag_vocabulary: %{bio_tag() => integer()},
          feature_weights: %{String.t() => %{bio_tag() => float()}},
          transition_weights: %{bio_tag() => %{bio_tag() => float()}},
          tag_priors: %{bio_tag() => float()}
        }

  @doc "Train entity recognition model from intent data.\nReturns {:ok, model} or {:error, reason}.\nEmits telemetry events for training metrics.\n"
  def train do
    start_time = System.monotonic_time(:millisecond)

    Logger.info("Starting entity model training...")

    :telemetry.execute(
      [:chat_bot, :ml, :train, :start],
      %{sequence_count: 0},
      %{model: :entity_trainer, started_at: DateTime.utc_now()}
    )

    {:ok, examples} = DataLoaders.load_all_intents()

    Logger.info("Converting examples to BIO format", %{count: length(examples)})
    sequences = convert_to_bio_sequences(examples)
    Logger.info("Generated BIO sequences", %{count: length(sequences)})

    result =
      if sequences != [] do
        model = train_sequence_model(sequences)

        Logger.info("Entity model trained", %{
          tag_count: map_size(model.tag_vocabulary),
          feature_count: map_size(model.feature_weights)
        })

        {:ok, model, length(sequences)}
      else
        {:error, "No valid training sequences generated"}
      end

    duration_ms = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, model, sequence_count} ->
        :telemetry.execute(
          [:chat_bot, :ml, :train, :stop],
          %{
            duration_ms: duration_ms,
            sequence_count: sequence_count,
            tag_count: map_size(model.tag_vocabulary),
            feature_count: map_size(model.feature_weights)
          },
          %{model: :entity_trainer, success: true}
        )

        {:ok, model}

      {:error, reason} ->
        :telemetry.execute(
          [:chat_bot, :ml, :train, :exception],
          %{duration_ms: duration_ms, sequence_count: 0},
          %{model: :entity_trainer, success: false, reason: reason}
        )

        {:error, reason}
    end
  end

  @doc "Train entity recognition model and save to disk.\n\n## Options\n  - models_path: Override the default models output path\n"
  def train_and_save(opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path) ||
        Application.get_env(:brain, :ml)[:models_path] ||
        Brain.priv_path("ml_models")

    case train() do
      {:ok, model} ->
        save_model(model, models_path)

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc "Load a trained entity model from disk.\n"
  def load_model do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    model_path = Path.join(models_path, "entity_model.term")

    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          e -> {:error, "Failed to deserialize model: #{inspect(e)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read model: #{reason}"}
    end
  end

  @doc "Predict entity tags for a sequence of tokens.\nReturns list of {token, predicted_tag} tuples.\n"
  def predict(tokens, model) when is_list(tokens) do
    predictions = viterbi_decode(tokens, model)
    Enum.zip(tokens, predictions)
  end

  @doc "Extract entities from predicted BIO tags.\nReturns list of entity maps.\n"
  def extract_entities_from_bio(token_tag_pairs) do
    extract_entities_from_bio_impl(token_tag_pairs, [], nil)
  end

  @doc "Convert Dialogflow-format examples to BIO-tagged sequences.\n"
  def convert_to_bio_sequences(examples) when is_list(examples) do
    examples
    |> Enum.map(&convert_example_to_bio/1)
    |> Enum.filter(fn seq -> seq.tokens != [] end)
  end

  defp train_sequence_model(sequences) do
    all_tags =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.uniq()
      |> Enum.sort()

    tag_vocabulary =
      all_tags
      |> Enum.with_index()
      |> Enum.into(%{})

    tag_counts =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.frequencies()

    total_tags = Enum.sum(Map.values(tag_counts))

    tag_priors =
      Enum.into(tag_counts, %{}, fn {tag, count} ->
        {tag, count / total_tags}
      end)

    transition_counts = calculate_transition_counts(sequences)
    transition_weights = normalize_transition_counts(transition_counts, all_tags)
    feature_weights = calculate_feature_weights(sequences)

    %{
      tag_vocabulary: tag_vocabulary,
      feature_weights: feature_weights,
      transition_weights: transition_weights,
      tag_priors: tag_priors
    }
  end

  defp calculate_transition_counts(sequences) do
    Enum.reduce(sequences, %{}, fn seq, acc ->
      tags = ["<START>" | seq.tags] ++ ["<END>"]

      tags
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.reduce(acc, fn [prev, curr], inner_acc ->
        key = {prev, curr}
        Map.update(inner_acc, key, 1, &(&1 + 1))
      end)
    end)
  end

  defp normalize_transition_counts(counts, all_tags) do
    all_tags_with_markers = ["<START>" | all_tags] ++ ["<END>"]
    grouped = Enum.group_by(counts, fn {{prev, _curr}, _count} -> prev end)

    Enum.reduce(all_tags_with_markers, %{}, fn prev_tag, acc ->
      transitions = Map.get(grouped, prev_tag, [])
      total = Enum.sum(Enum.map(transitions, fn {_, count} -> count end))

      if total > 0 do
        probs =
          Enum.into(transitions, %{}, fn {{_prev, curr}, count} ->
            {curr, count / total}
          end)

        Map.put(acc, prev_tag, probs)
      else
        uniform = 1.0 / length(all_tags_with_markers)
        probs = Enum.into(all_tags_with_markers, %{}, fn tag -> {tag, uniform} end)
        Map.put(acc, prev_tag, probs)
      end
    end)
  end

  defp calculate_feature_weights(sequences) do
    feature_tag_counts =
      Enum.reduce(sequences, %{}, fn seq, acc ->
        seq.tokens
        |> Enum.zip(seq.tags)
        |> Enum.with_index()
        |> Enum.reduce(acc, fn {{token, tag}, idx}, inner_acc ->
          features = extract_token_features(token, seq.tokens, idx)

          Enum.reduce(features, inner_acc, fn feature, feat_acc ->
            Map.update(feat_acc, feature, %{tag => 1}, fn tag_counts ->
              Map.update(tag_counts, tag, 1, &(&1 + 1))
            end)
          end)
        end)
      end)

    Enum.into(feature_tag_counts, %{}, fn {feature, tag_counts} ->
      total = Enum.sum(Map.values(tag_counts))

      probs =
        Enum.into(tag_counts, %{}, fn {tag, count} ->
          {tag, count / total}
        end)

      {feature, probs}
    end)
  end

  defp extract_token_features(token, all_tokens, idx) do
    features = [
      "token:#{String.downcase(token)}",
      "prefix2:#{String.slice(String.downcase(token), 0, 2)}",
      "prefix3:#{String.slice(String.downcase(token), 0, 3)}",
      "suffix2:#{String.slice(String.downcase(token), -2, 2)}",
      "suffix3:#{String.slice(String.downcase(token), -3, 3)}",
      if(capitalized?(token)) do
        "is_capitalized"
      else
        "not_capitalized"
      end,
      if(all_caps?(token)) do
        "is_all_caps"
      else
        "not_all_caps"
      end,
      if(has_digit?(token)) do
        "has_digit"
      else
        "no_digit"
      end,
      if(all_digits?(token)) do
        "is_number"
      else
        "not_number"
      end,
      if(idx == 0) do
        "is_first"
      else
        "not_first"
      end,
      if(idx == length(all_tokens) - 1) do
        "is_last"
      else
        "not_last"
      end
    ]

    prev_features =
      if idx > 0 do
        prev_token = Enum.at(all_tokens, idx - 1)
        ["prev_token:#{String.downcase(prev_token)}"]
      else
        ["prev_token:<START>"]
      end

    next_features =
      if idx < length(all_tokens) - 1 do
        next_token = Enum.at(all_tokens, idx + 1)
        ["next_token:#{String.downcase(next_token)}"]
      else
        ["next_token:<END>"]
      end

    Enum.filter(features ++ prev_features ++ next_features, &(&1 != nil))
  end

  defp capitalized?(token) do
    first = String.first(token) || ""
    first == String.upcase(first) and first != String.downcase(first)
  end

  defp all_caps?(token) do
    token == String.upcase(token) and token != String.downcase(token)
  end

  defp has_digit?(token) do
    Enum.any?(String.graphemes(token), fn g ->
      g >= "0" and g <= "9"
    end)
  end

  defp all_digits?(token) do
    token != "" and
      Enum.all?(String.graphemes(token), fn g ->
        g >= "0" and g <= "9"
      end)
  end

  defp viterbi_decode(tokens, model) do
    if tokens == [] do
      []
    else
      tags = Map.keys(model.tag_vocabulary) |> Enum.filter(&(&1 != "<START>" and &1 != "<END>"))

      {initial_viterbi, initial_backpointer} =
        initialize_viterbi(Enum.at(tokens, 0), tokens, 0, tags, model)

      {final_viterbi, backpointers} =
        tokens
        |> Enum.with_index()
        |> Enum.drop(1)
        |> Enum.reduce({initial_viterbi, [initial_backpointer]}, fn {token, idx},
                                                                    {prev_viterbi, bps} ->
          {new_viterbi, new_bp} = viterbi_step(token, tokens, idx, prev_viterbi, tags, model)
          {new_viterbi, [new_bp | bps]}
        end)

      backtrack(final_viterbi, Enum.reverse(backpointers), tags)
    end
  end

  defp initialize_viterbi(token, all_tokens, idx, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    viterbi =
      Enum.into(tags, %{}, fn tag ->
        emission = calculate_emission_prob(features, tag, model)
        transition = get_transition_prob("<START>", tag, model)
        score = :math.log(max(emission, 1.0e-10)) + :math.log(max(transition, 1.0e-10))
        {tag, score}
      end)

    backpointer = Enum.into(tags, %{}, fn tag -> {tag, nil} end)

    {viterbi, backpointer}
  end

  defp viterbi_step(token, all_tokens, idx, prev_viterbi, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    {viterbi, backpointer} =
      Enum.reduce(tags, {%{}, %{}}, fn curr_tag, {vit_acc, bp_acc} ->
        emission = calculate_emission_prob(features, curr_tag, model)

        {best_score, best_prev_tag} =
          Enum.reduce(tags, {nil, nil}, fn prev_tag, {best, best_tag} ->
            prev_score = Map.get(prev_viterbi, prev_tag, -1000)
            transition = get_transition_prob(prev_tag, curr_tag, model)

            score =
              prev_score + :math.log(max(transition, 1.0e-10)) + :math.log(max(emission, 1.0e-10))

            if best == nil or score > best do
              {score, prev_tag}
            else
              {best, best_tag}
            end
          end)

        {Map.put(vit_acc, curr_tag, best_score), Map.put(bp_acc, curr_tag, best_prev_tag)}
      end)

    {viterbi, backpointer}
  end

  defp calculate_emission_prob(features, tag, model) do
    probs =
      Enum.map(features, fn feature ->
        case Map.get(model.feature_weights, feature) do
          nil -> Map.get(model.tag_priors, tag, 0.01)
          tag_probs -> Map.get(tag_probs, tag, 0.01)
        end
      end)

    if probs != [] do
      Enum.sum(probs) / length(probs)
    else
      Map.get(model.tag_priors, tag, 0.01)
    end
  end

  defp get_transition_prob(prev_tag, curr_tag, model) do
    case Map.get(model.transition_weights, prev_tag) do
      nil -> 0.01
      probs -> Map.get(probs, curr_tag, 0.01)
    end
  end

  defp backtrack(final_viterbi, backpointers, tags) do
    {_best_score, best_tag} =
      Enum.reduce(tags, {nil, nil}, fn tag, {best, best_tag} ->
        score = Map.get(final_viterbi, tag, -1000)

        if best == nil or score > best do
          {score, tag}
        else
          {best, best_tag}
        end
      end)

    backtrack_impl(backpointers, best_tag, [best_tag])
  end

  defp backtrack_impl([], _current_tag, path) do
    path
  end

  defp backtrack_impl([bp | rest], current_tag, path) do
    prev_tag = Map.get(bp, current_tag)

    if prev_tag != nil do
      backtrack_impl(rest, prev_tag, [prev_tag | path])
    else
      backtrack_impl(rest, current_tag, path)
    end
  end

  defp convert_example_to_bio(example) do
    text = example.text
    entities = example.entities || []
    intent = example.intent
    tokens = Tokenizer.tokenize(text)
    tags = assign_bio_tags(tokens, entities)

    %{
      tokens: Enum.map(tokens, & &1.text),
      tags: tags,
      intent: intent
    }
  end

  defp assign_bio_tags(tokens, entities) do
    entity_positions = build_entity_position_map(entities)

    Enum.map(tokens, fn token ->
      find_bio_tag_for_token(token, entity_positions)
    end)
  end

  defp build_entity_position_map(entities) do
    Enum.reduce(entities, %{}, fn entity, acc ->
      start_pos = entity.start_pos || 0
      end_pos = entity.end_pos || start_pos + String.length(entity.text) - 1
      entity_type = entity.type || entity.alias || "unknown"

      Enum.reduce(start_pos..end_pos, acc, fn pos, inner_acc ->
        is_start = pos == start_pos
        Map.put(inner_acc, pos, {entity_type, is_start})
      end)
    end)
  end

  defp find_bio_tag_for_token(token, entity_positions) do
    token_start = token.start_pos
    _token_end = token.end_pos

    case Map.get(entity_positions, token_start) do
      nil ->
        "O"

      {entity_type, true} ->
        "B-#{entity_type}"

      {entity_type, false} ->
        "I-#{entity_type}"
    end
  end

  defp extract_entities_from_bio_impl([], acc, current_entity) do
    case current_entity do
      nil -> Enum.reverse(acc)
      entity -> Enum.reverse([finalize_entity(entity) | acc])
    end
  end

  defp extract_entities_from_bio_impl([{token, tag} | rest], acc, current_entity) do
    cond do
      String.starts_with?(tag, "B-") ->
        entity_type = String.replace_prefix(tag, "B-", "")

        new_entity = %{
          type: entity_type,
          tokens: [token],
          text: token
        }

        case current_entity do
          nil ->
            extract_entities_from_bio_impl(rest, acc, new_entity)

          prev ->
            extract_entities_from_bio_impl(rest, [finalize_entity(prev) | acc], new_entity)
        end

      String.starts_with?(tag, "I-") and current_entity != nil ->
        entity_type = String.replace_prefix(tag, "I-", "")

        if entity_type == current_entity.type do
          updated = %{
            current_entity
            | tokens: current_entity.tokens ++ [token],
              text: current_entity.text <> " " <> token
          }

          extract_entities_from_bio_impl(rest, acc, updated)
        else
          new_entity = %{type: entity_type, tokens: [token], text: token}

          extract_entities_from_bio_impl(
            rest,
            [finalize_entity(current_entity) | acc],
            new_entity
          )
        end

      tag == "O" ->
        case current_entity do
          nil ->
            extract_entities_from_bio_impl(rest, acc, nil)

          entity ->
            extract_entities_from_bio_impl(rest, [finalize_entity(entity) | acc], nil)
        end

      true ->
        case current_entity do
          nil ->
            extract_entities_from_bio_impl(rest, acc, nil)

          entity ->
            extract_entities_from_bio_impl(rest, [finalize_entity(entity) | acc], nil)
        end
    end
  end

  defp finalize_entity(entity) do
    %{
      entity_type: entity.type,
      value: entity.text,
      tokens: entity.tokens,
      confidence: 0.8
    }
  end

  defp save_model(model, models_path) do
    File.mkdir_p!(models_path)

    model_path = Path.join(models_path, "entity_model.term")
    binary = :erlang.term_to_binary(model)

    case File.write(model_path, binary) do
      :ok ->
        Logger.info("Entity model saved", %{path: model_path})
        {:ok, model}

      {:error, reason} ->
        {:error, "Failed to save model: #{reason}"}
    end
  end
end
