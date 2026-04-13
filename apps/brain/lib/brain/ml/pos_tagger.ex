defmodule Brain.ML.POSTagger do
  @moduledoc "Part-of-Speech tagging using trained sequence model.\n\nTags tokens with grammatical roles (PRON, VERB, NOUN, ADJ, etc.)\nusing the same HMM-like architecture as EntityTrainer:\n- Feature extraction (prefix, suffix, capitalization, context)\n- Transition probabilities (P(tag|prev_tag))\n- Emission probabilities (P(features|tag))\n- Viterbi decoding for optimal tag sequence\n\n## Training\n\nTraining data should be in the format:\n    %{\n      tokens: [\"I\", \"am\", \"Austin\"],\n      tags: [\"PRON\", \"VERB\", \"PROPN\"],\n      source: \"intent_name\"  # optional\n    }\n\n## Usage\n\n    # Train from data\n    {:ok, model} = POSTagger.train(training_sequences)\n\n    # Or load pre-trained model\n    {:ok, model} = POSTagger.load_model()\n\n    # Predict POS tags\n    predictions = POSTagger.predict([\"I\", \"am\", \"Austin\"], model)\n    # => [{\"I\", \"PRON\"}, {\"am\", \"VERB\"}, {\"Austin\", \"PROPN\"}]\n\n"

  require Logger
  alias Brain.Analysis.TypeHierarchy

  @pos_tags ~w(
    NOUN PROPN VERB AUX ADJ ADV PRON DET ADP
    CONJ PART NUM INTJ PUNCT SYM X
  )

  @type pos_tag :: String.t()

  @type training_sequence :: %{
          tokens: [String.t()],
          tags: [pos_tag()],
          source: String.t() | nil
        }

  @type pos_model :: %{
          tag_vocabulary: %{pos_tag() => integer()},
          feature_weights: %{String.t() => %{pos_tag() => float()}},
          transition_weights: %{pos_tag() => %{pos_tag() => float()}},
          tag_priors: %{pos_tag() => float()}
        }
  defp model_path do
    Brain.priv_path("ml_models/pos_model.term")
  end

  @doc "Train POS model from labeled training sequences.\nReturns {:ok, model} or {:error, reason}.\nEmits telemetry events for training metrics.\n"
  def train(training_sequences) when is_list(training_sequences) do
    start_time = System.monotonic_time(:millisecond)
    sequence_count = length(training_sequences)

    Logger.info("Starting POS model training...", %{sequences: sequence_count})

    :telemetry.execute(
      [:chat_bot, :ml, :train, :start],
      %{sequence_count: sequence_count},
      %{model: :pos_tagger, started_at: DateTime.utc_now()}
    )

    result =
      if sequence_count == 0 do
        {:error, "No training sequences provided"}
      else
        valid_sequences =
          training_sequences
          |> Enum.filter(fn seq ->
            tokens = Map.get(seq, :tokens) || Map.get(seq, "tokens", [])
            tags = Map.get(seq, :tags) || Map.get(seq, "tags", [])
            tokens != [] and length(tokens) == length(tags)
          end)
          |> Enum.map(&normalize_sequence/1)

        if valid_sequences == [] do
          {:error, "No valid training sequences after filtering"}
        else
          Logger.info("Training on valid sequences", %{count: length(valid_sequences)})
          model = train_sequence_model(valid_sequences)

          Logger.info("POS model trained", %{
            tag_count: map_size(model.tag_vocabulary),
            feature_count: map_size(model.feature_weights)
          })

          {:ok, model}
        end
      end

    duration_ms = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, model} ->
        :telemetry.execute(
          [:chat_bot, :ml, :train, :stop],
          %{
            duration_ms: duration_ms,
            sequence_count: sequence_count,
            tag_count: map_size(model.tag_vocabulary),
            feature_count: map_size(model.feature_weights)
          },
          %{model: :pos_tagger, success: true}
        )

      {:error, reason} ->
        :telemetry.execute(
          [:chat_bot, :ml, :train, :exception],
          %{duration_ms: duration_ms, sequence_count: sequence_count},
          %{model: :pos_tagger, success: false, reason: reason}
        )
    end

    result
  end

  @doc "Train and save POS model to disk.\n"
  def train_and_save(training_sequences) do
    case train(training_sequences) do
      {:ok, model} -> save_model(model)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc "Load training data from JSON file and train model.\n"
  def train_from_file(training_file_path \\ "data/training/pos/sequences.json") do
    case File.read(training_file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, sequences} when is_list(sequences) ->
            train(sequences)

          {:ok, %{"sequences" => sequences}} when is_list(sequences) ->
            train(sequences)

          {:error, reason} ->
            {:error, "Failed to parse training file: #{inspect(reason)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read training file: #{reason}"}
    end
  end

  @doc "Load a trained POS model from disk.\n"
  def load_model(path \\ nil) do
    model_path = path || get_model_path()

    if is_nil(path) do
      Brain.ML.ModelStore.ensure_local("pos_model.term", model_path)
    end

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

  @doc "Save trained model to disk.\n"
  def save_model(model, path \\ nil) do
    model_path = path || get_model_path()
    File.mkdir_p!(Path.dirname(model_path))

    binary = :erlang.term_to_binary(model)

    case File.write(model_path, binary) do
      :ok ->
        Logger.info("POS model saved to #{model_path}")
        {:ok, model_path}

      {:error, reason} ->
        {:error, "Failed to save model: #{reason}"}
    end
  end

  @doc """
  Get the current POS model from disk.

  Returns `{:ok, model}` or `{:error, reason}`.
  """
  def get_model do
    load_model()
  end

  @doc """
  Update model weights by blending graph-derived weights with existing model.

  The `blend` option controls how much weight the graph data gets:
  - `blend: 0.3` means 30% graph + 70% existing (default)
  - `blend: 1.0` means 100% graph (replaces existing)
  - `blend: 0.0` means 0% graph (no change)

  ## Parameters

  - `transition_weights` - Map of `%{from_tag => %{to_tag => frequency}}`
  - `tag_priors` - Map of `%{tag => prior_probability}`
  - `opts` - Options including `:blend` ratio

  ## Returns

  `:ok` on success, `{:error, reason}` on failure.
  """
  def update_weights(transition_weights, tag_priors, opts \\ []) do
    blend = Keyword.get(opts, :blend, 0.3)

    case load_model() do
      {:ok, model} ->
        blended_transitions = blend_maps(model.transition_weights, transition_weights, blend)
        blended_priors = blend_flat_map(model.tag_priors, tag_priors, blend)

        updated_model = %{
          model
          | transition_weights: blended_transitions,
            tag_priors: blended_priors
        }

        case save_model(updated_model) do
          {:ok, _path} ->
            Logger.info("POS model weights updated with graph data", blend: blend)
            :ok

          error ->
            error
        end

      {:error, reason} ->
        Logger.warning("Cannot update POS weights - no model loaded", reason: inspect(reason))
        {:error, reason}
    end
  end

  defp blend_maps(existing, new_data, blend) when is_map(existing) and is_map(new_data) do
    all_keys = MapSet.union(MapSet.new(Map.keys(existing)), MapSet.new(Map.keys(new_data)))

    Map.new(all_keys, fn key ->
      existing_inner = Map.get(existing, key, %{})
      new_inner = Map.get(new_data, key, %{})

      blended =
        cond do
          is_map(existing_inner) and is_map(new_inner) ->
            blend_flat_map(existing_inner, new_inner, blend)

          is_number(existing_inner) and is_number(new_inner) ->
            existing_inner * (1 - blend) + new_inner * blend

          is_map(existing_inner) ->
            existing_inner

          is_map(new_inner) ->
            new_inner

          true ->
            existing_inner
        end

      {key, blended}
    end)
  end

  defp blend_maps(existing, _, _), do: existing

  defp blend_flat_map(existing, new_data, blend) when is_map(existing) and is_map(new_data) do
    all_keys = MapSet.union(MapSet.new(Map.keys(existing)), MapSet.new(Map.keys(new_data)))

    Map.new(all_keys, fn key ->
      e = Map.get(existing, key, 0)
      n = Map.get(new_data, key, 0)

      if is_number(e) and is_number(n) do
        {key, e * (1 - blend) + n * blend}
      else
        {key, e}
      end
    end)
  end

  defp blend_flat_map(existing, _, _), do: existing

  @doc """
  Blends graph-derived POS transition frequencies into the model.

  Reads `tag_transitions` from the pos_graph for each known tag,
  normalizes them into transition probabilities, and blends them
  with the existing model via `update_weights/3`.
  """
  def blend_graph_transitions(opts \\ []) do
    blend = Keyword.get(opts, :blend, 0.2)

    graph_transitions =
      valid_tags()
      |> Enum.reduce(%{}, fn tag, acc ->
        case Brain.Graph.Reader.tag_transitions(tag) do
          transitions when is_list(transitions) and transitions != [] ->
            total = transitions |> Enum.map(& &1.frequency) |> Enum.sum() |> max(1)

            probs =
              Map.new(transitions, fn %{to_tag: to, frequency: freq} ->
                {to, freq / total}
              end)

            Map.put(acc, tag, probs)

          _ ->
            acc
        end
      end)

    if map_size(graph_transitions) > 0 do
      update_weights(graph_transitions, %{}, blend: blend)
    else
      :ok
    end
  rescue
    _ -> :ok
  end

  @doc "Predict POS tags for a sequence of tokens.\nReturns list of {token, predicted_tag} tuples.\n"
  def predict(tokens, model) when is_list(tokens) and is_map(model) do
    if tokens == [] do
      []
    else
      predictions = viterbi_decode(tokens, model) |> correct_propn(tokens, model)
      Enum.zip(tokens, predictions)
    end
  end

  @doc "Predict POS tags, returning just the tags.\n"
  def predict_tags(tokens, model) when is_list(tokens) do
    if tokens == [] do
      []
    else
      viterbi_decode(tokens, model) |> correct_propn(tokens, model)
    end
  end

  @doc false
  defp correct_propn(tags, tokens, model) do
    propn_tag = TypeHierarchy.config(["pos_tag_roles", "proper_noun"], "PROPN")
    cap_weights = Map.get(model.feature_weights, "cap_non_initial", %{})
    cap_propn = Map.get(cap_weights, propn_tag, 0)

    if cap_propn < 0.3 do
      tags
    else
      tokens
      |> Enum.with_index()
      |> Enum.zip(tags)
      |> Enum.map(fn {{token, idx}, tag} ->
        if idx > 0 and tag != propn_tag and capitalized?(token) do
          token_weights = Map.get(model.feature_weights, "token:#{String.downcase(token)}", %{})
          token_propn = Map.get(token_weights, propn_tag, 0)
          token_current = Map.get(token_weights, tag, 0)

          cond do
            map_size(token_weights) == 0 ->
              propn_tag

            token_propn > token_current ->
              propn_tag

            true ->
              tag
          end
        else
          tag
        end
      end)
    end
  end

  @doc "Check if a trained model exists.\n"
  def model_exists?(path \\ nil) do
    model_path = path || get_model_path()
    File.exists?(model_path)
  end

  @doc "Return list of valid POS tags.\n"
  def valid_tags do
    @pos_tags
  end

  defp normalize_sequence(seq) do
    tokens = Map.get(seq, :tokens) || Map.get(seq, "tokens", [])
    tags = Map.get(seq, :tags) || Map.get(seq, "tags", [])
    source = Map.get(seq, :source) || Map.get(seq, "source")
    normalized_tags = Enum.map(tags, &normalize_tag/1)

    %{
      tokens: tokens,
      tags: normalized_tags,
      source: source
    }
  end

  defp normalize_tag(tag) when is_atom(tag) do
    Atom.to_string(tag) |> String.upcase()
  end

  defp normalize_tag(tag) when is_binary(tag) do
    String.upcase(tag)
  end

  defp normalize_tag(_) do
    "X"
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
    lower_token = String.downcase(token)
    is_cap = capitalized?(token)

    features = [
      "token:#{lower_token}",
      "prefix2:#{String.slice(lower_token, 0, 2)}",
      "prefix3:#{String.slice(lower_token, 0, 3)}",
      "suffix2:#{String.slice(lower_token, -2, 2) || ""}",
      "suffix3:#{String.slice(lower_token, -3, 3) || ""}",
      if(is_cap, do: "is_capitalized", else: "not_capitalized"),
      if(is_cap and idx > 0, do: "cap_non_initial", else: "not_cap_non_initial"),
      if(all_caps?(token), do: "is_all_caps", else: "not_all_caps"),
      if(all_lower?(token), do: "is_all_lower", else: "not_all_lower"),
      if(has_digit?(token), do: "has_digit", else: "no_digit"),
      if(all_digits?(token), do: "is_number", else: "not_number"),
      if(is_punctuation?(token), do: "is_punct", else: "not_punct"),
      if(idx == 0, do: "is_first", else: "not_first"),
      if(idx == length(all_tokens) - 1, do: "is_last", else: "not_last"),
      "length:#{min(String.length(token), 10)}"
    ]

    prev_features =
      if idx > 0 do
        prev_token = Enum.at(all_tokens, idx - 1)
        prev_lower = String.downcase(prev_token)
        [
          "prev_token:#{prev_lower}",
          if(is_cap and capitalized?(prev_token), do: "prev_also_cap", else: nil)
        ]
      else
        ["prev_token:<START>"]
      end

    next_features =
      if idx < length(all_tokens) - 1 do
        next_token = Enum.at(all_tokens, idx + 1)
        next_lower = String.downcase(next_token)
        [
          "next_token:#{next_lower}",
          if(is_cap and capitalized?(next_token), do: "next_also_cap", else: nil)
        ]
      else
        ["next_token:<END>"]
      end

    Enum.filter(features ++ prev_features ++ next_features, &(&1 != nil))
  end

  defp viterbi_decode(tokens, model) do
    tags =
      Map.keys(model.tag_vocabulary)
      |> Enum.filter(&(&1 != "<START>" and &1 != "<END>"))

    if tags == [] do
      Enum.map(tokens, fn _ -> "X" end)
    else
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
        trans_prob = get_transition_prob("<START>", tag, model)
        emit_prob = get_emission_prob(features, tag, model)
        {tag, trans_prob * emit_prob}
      end)

    backpointer = Enum.into(tags, %{}, fn tag -> {tag, nil} end)

    {viterbi, backpointer}
  end

  defp viterbi_step(token, all_tokens, idx, prev_viterbi, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    {viterbi, backpointer} =
      Enum.reduce(tags, {%{}, %{}}, fn tag, {v_acc, bp_acc} ->
        {best_prev, best_prob} =
          Enum.reduce(tags, {nil, 0.0}, fn prev_tag, {best, best_p} ->
            prev_prob = Map.get(prev_viterbi, prev_tag, 0.0)
            trans_prob = get_transition_prob(prev_tag, tag, model)
            prob = prev_prob * trans_prob

            if prob > best_p do
              {prev_tag, prob}
            else
              {best, best_p}
            end
          end)

        emit_prob = get_emission_prob(features, tag, model)
        final_prob = best_prob * emit_prob

        {Map.put(v_acc, tag, final_prob), Map.put(bp_acc, tag, best_prev)}
      end)

    {viterbi, backpointer}
  end

  defp backtrack(final_viterbi, backpointers, tags) do
    {best_tag, _} =
      Enum.max_by(final_viterbi, fn {_tag, prob} -> prob end, fn ->
        {Enum.at(tags, 0), 0.0}
      end)

    path =
      Enum.reduce(Enum.reverse(backpointers), [best_tag], fn bp, [current | _] = path ->
        prev = Map.get(bp, current)

        if prev do
          [prev | path]
        else
          path
        end
      end)

    Enum.take(path, -length(backpointers))
    |> case do
      [] -> [best_tag]
      p -> p
    end
  end

  defp get_transition_prob(prev_tag, current_tag, model) do
    model.transition_weights
    |> Map.get(prev_tag, %{})
    |> Map.get(current_tag, 0.001)
  end

  defp get_emission_prob(features, tag, model) do
    known_probs =
      features
      |> Enum.flat_map(fn feature ->
        case Map.get(model.feature_weights, feature) do
          nil -> []
          weights -> [Map.get(weights, tag, 0.001)]
        end
      end)

    if known_probs != [] do
      Enum.sum(known_probs) / length(known_probs)
    else
      Map.get(model.tag_priors, tag, 0.001)
    end
  end

  defp get_model_path do
    case Application.get_env(:brain, :ml)[:models_path] do
      nil -> model_path()
      models_path -> Path.join(models_path, "pos_model.term")
    end
  end

  defp capitalized?(token) do
    first = String.first(token) || ""
    first == String.upcase(first) and first != String.downcase(first)
  end

  defp all_caps?(token) do
    token == String.upcase(token) and token != String.downcase(token)
  end

  defp all_lower?(token) do
    token == String.downcase(token) and token != String.upcase(token)
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

  defp is_punctuation?(token) do
    punct = [
      ".",
      ",",
      "!",
      "?",
      ";",
      ":",
      "'",
      "\"",
      "-",
      "--",
      "...",
      "(",
      ")",
      "[",
      "]",
      "{",
      "}",
      "/",
      "\\",
      "@",
      "#",
      "$",
      "%",
      "^",
      "&",
      "*",
      "+",
      "=",
      "~",
      "`"
    ]

    token in punct
  end
end
