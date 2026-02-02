defmodule Brain.ML.POSTagger do
  @moduledoc """
  Part-of-Speech tagging using trained sequence model.

  Tags tokens with grammatical roles (PRON, VERB, NOUN, ADJ, etc.)
  using the same HMM-like architecture as EntityTrainer:
  - Feature extraction (prefix, suffix, capitalization, context)
  - Transition probabilities (P(tag|prev_tag))
  - Emission probabilities (P(features|tag))
  - Viterbi decoding for optimal tag sequence

  ## Training

  Training data should be in the format:
      %{
        tokens: ["I", "am", "Austin"],
        tags: ["PRON", "VERB", "PROPN"],
        source: "intent_name"  # optional
      }

  ## Usage

      # Train from data
      {:ok, model} = POSTagger.train(training_sequences)

      # Or load pre-trained model
      {:ok, model} = POSTagger.load_model()

      # Predict POS tags
      predictions = POSTagger.predict(["I", "am", "Austin"], model)
      # => [{"I", "PRON"}, {"am", "VERB"}, {"Austin", "PROPN"}]

  """

  require Logger

  # Universal POS tags (subset based on Universal Dependencies)
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

  # Model path is resolved at runtime via Brain.priv_path/1
  defp model_path, do: Brain.priv_path("ml_models/pos_model.term")

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Train POS model from labeled training sequences.
  Returns {:ok, model} or {:error, reason}.
  Emits telemetry events for training metrics.
  """
  def train(training_sequences) when is_list(training_sequences) do
    start_time = System.monotonic_time(:millisecond)
    sequence_count = length(training_sequences)

    Logger.info("Starting POS model training...", %{sequences: sequence_count})

    # Emit training start event
    :telemetry.execute(
      [:chat_bot, :ml, :train, :start],
      %{sequence_count: sequence_count},
      %{model: :pos_tagger, started_at: DateTime.utc_now()}
    )

    result =
      if sequence_count == 0 do
        {:error, "No training sequences provided"}
      else
        # Filter valid sequences
        valid_sequences =
          training_sequences
          |> Enum.filter(fn seq ->
            tokens = Map.get(seq, :tokens) || Map.get(seq, "tokens", [])
            tags = Map.get(seq, :tags) || Map.get(seq, "tags", [])
            length(tokens) > 0 and length(tokens) == length(tags)
          end)
          |> Enum.map(&normalize_sequence/1)

        if length(valid_sequences) == 0 do
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

    # Calculate training metrics
    duration_ms = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, model} ->
        # Emit training success event
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
        # Emit training failure event
        :telemetry.execute(
          [:chat_bot, :ml, :train, :exception],
          %{duration_ms: duration_ms, sequence_count: sequence_count},
          %{model: :pos_tagger, success: false, reason: reason}
        )
    end

    result
  end

  @doc """
  Train and save POS model to disk.
  """
  def train_and_save(training_sequences) do
    case train(training_sequences) do
      {:ok, model} -> save_model(model)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Load training data from JSON file and train model.
  """
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

  @doc """
  Load a trained POS model from disk.
  """
  def load_model(path \\ nil) do
    model_path = path || get_model_path()

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

  @doc """
  Save trained model to disk.
  """
  def save_model(model, path \\ nil) do
    model_path = path || get_model_path()

    # Ensure directory exists
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
  Predict POS tags for a sequence of tokens.
  Returns list of {token, predicted_tag} tuples.
  """
  def predict(tokens, model) when is_list(tokens) and is_map(model) do
    if length(tokens) == 0 do
      []
    else
      predictions = viterbi_decode(tokens, model)
      Enum.zip(tokens, predictions)
    end
  end

  @doc """
  Predict POS tags, returning just the tags.
  """
  def predict_tags(tokens, model) when is_list(tokens) do
    if length(tokens) == 0 do
      []
    else
      viterbi_decode(tokens, model)
    end
  end

  @doc """
  Check if a trained model exists.
  """
  def model_exists?(path \\ nil) do
    model_path = path || get_model_path()
    File.exists?(model_path)
  end

  @doc """
  Return list of valid POS tags.
  """
  def valid_tags, do: @pos_tags

  # ============================================================================
  # Training Implementation
  # ============================================================================

  defp normalize_sequence(seq) do
    tokens = Map.get(seq, :tokens) || Map.get(seq, "tokens", [])
    tags = Map.get(seq, :tags) || Map.get(seq, "tags", [])
    source = Map.get(seq, :source) || Map.get(seq, "source")

    # Normalize tags to uppercase strings
    normalized_tags = Enum.map(tags, &normalize_tag/1)

    %{
      tokens: tokens,
      tags: normalized_tags,
      source: source
    }
  end

  defp normalize_tag(tag) when is_atom(tag), do: Atom.to_string(tag) |> String.upcase()
  defp normalize_tag(tag) when is_binary(tag), do: String.upcase(tag)
  defp normalize_tag(_), do: "X"

  defp train_sequence_model(sequences) do
    # Collect all tags
    all_tags =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.uniq()
      |> Enum.sort()

    # Build tag vocabulary
    tag_vocabulary =
      all_tags
      |> Enum.with_index()
      |> Enum.into(%{})

    # Calculate tag priors
    tag_counts =
      sequences
      |> Enum.flat_map(& &1.tags)
      |> Enum.frequencies()

    total_tags = Enum.sum(Map.values(tag_counts))

    tag_priors =
      Enum.into(tag_counts, %{}, fn {tag, count} ->
        {tag, count / total_tags}
      end)

    # Calculate transition probabilities
    transition_counts = calculate_transition_counts(sequences)
    transition_weights = normalize_transition_counts(transition_counts, all_tags)

    # Calculate feature weights (emission probabilities)
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

    # Group by previous tag
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
        # Default uniform distribution
        uniform = 1.0 / length(all_tags_with_markers)
        probs = Enum.into(all_tags_with_markers, %{}, fn tag -> {tag, uniform} end)
        Map.put(acc, prev_tag, probs)
      end
    end)
  end

  defp calculate_feature_weights(sequences) do
    # Count (token_feature, tag) co-occurrences
    feature_tag_counts =
      Enum.reduce(sequences, %{}, fn seq, acc ->
        seq.tokens
        |> Enum.zip(seq.tags)
        |> Enum.with_index()
        |> Enum.reduce(acc, fn {{token, tag}, idx}, inner_acc ->
          # Extract features for this token
          features = extract_token_features(token, seq.tokens, idx)

          Enum.reduce(features, inner_acc, fn feature, feat_acc ->
            Map.update(feat_acc, feature, %{tag => 1}, fn tag_counts ->
              Map.update(tag_counts, tag, 1, &(&1 + 1))
            end)
          end)
        end)
      end)

    # Normalize to probabilities
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

    features = [
      # Current token (lowercased)
      "token:#{lower_token}",
      # Token prefix
      "prefix2:#{String.slice(lower_token, 0, 2)}",
      "prefix3:#{String.slice(lower_token, 0, 3)}",
      # Token suffix
      "suffix2:#{String.slice(lower_token, -2, 2) || ""}",
      "suffix3:#{String.slice(lower_token, -3, 3) || ""}",
      # Capitalization features
      if(capitalized?(token), do: "is_capitalized", else: "not_capitalized"),
      if(all_caps?(token), do: "is_all_caps", else: "not_all_caps"),
      if(all_lower?(token), do: "is_all_lower", else: "not_all_lower"),
      # Digit features
      if(has_digit?(token), do: "has_digit", else: "no_digit"),
      if(all_digits?(token), do: "is_number", else: "not_number"),
      # Punctuation
      if(is_punctuation?(token), do: "is_punct", else: "not_punct"),
      # Position features
      if(idx == 0, do: "is_first", else: "not_first"),
      if(idx == length(all_tokens) - 1, do: "is_last", else: "not_last"),
      # Length features
      "length:#{min(String.length(token), 10)}"
    ]

    # Previous token feature
    prev_features =
      if idx > 0 do
        prev_token = Enum.at(all_tokens, idx - 1)
        ["prev_token:#{String.downcase(prev_token)}"]
      else
        ["prev_token:<START>"]
      end

    # Next token feature
    next_features =
      if idx < length(all_tokens) - 1 do
        next_token = Enum.at(all_tokens, idx + 1)
        ["next_token:#{String.downcase(next_token)}"]
      else
        ["next_token:<END>"]
      end

    Enum.filter(features ++ prev_features ++ next_features, &(&1 != nil))
  end

  # ============================================================================
  # Viterbi Decoding
  # ============================================================================

  defp viterbi_decode(tokens, model) do
    tags =
      Map.keys(model.tag_vocabulary)
      |> Enum.filter(&(&1 != "<START>" and &1 != "<END>"))

    if length(tags) == 0 do
      # Fallback if no tags in vocabulary
      Enum.map(tokens, fn _ -> "X" end)
    else
      # Initialize with start probabilities
      {initial_viterbi, initial_backpointer} =
        initialize_viterbi(Enum.at(tokens, 0), tokens, 0, tags, model)

      # Forward pass
      {final_viterbi, backpointers} =
        tokens
        |> Enum.with_index()
        |> Enum.drop(1)
        |> Enum.reduce({initial_viterbi, [initial_backpointer]}, fn {token, idx},
                                                                    {prev_viterbi, bps} ->
          {new_viterbi, new_bp} = viterbi_step(token, tokens, idx, prev_viterbi, tags, model)
          {new_viterbi, [new_bp | bps]}
        end)

      # Backtrack to find best path
      backtrack(final_viterbi, Enum.reverse(backpointers), tags)
    end
  end

  defp initialize_viterbi(token, all_tokens, idx, tags, model) do
    features = extract_token_features(token, all_tokens, idx)

    viterbi =
      Enum.into(tags, %{}, fn tag ->
        # P(tag | START) * P(features | tag)
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
        # Find best previous tag
        {best_prev, best_prob} =
          Enum.reduce(tags, {nil, 0.0}, fn prev_tag, {best, best_p} ->
            prev_prob = Map.get(prev_viterbi, prev_tag, 0.0)
            trans_prob = get_transition_prob(prev_tag, tag, model)
            prob = prev_prob * trans_prob

            if prob > best_p, do: {prev_tag, prob}, else: {best, best_p}
          end)

        emit_prob = get_emission_prob(features, tag, model)
        final_prob = best_prob * emit_prob

        {Map.put(v_acc, tag, final_prob), Map.put(bp_acc, tag, best_prev)}
      end)

    {viterbi, backpointer}
  end

  defp backtrack(final_viterbi, backpointers, tags) do
    # Find best final tag
    {best_tag, _} =
      Enum.max_by(final_viterbi, fn {_tag, prob} -> prob end, fn ->
        {Enum.at(tags, 0), 0.0}
      end)

    # Backtrack through backpointers
    path =
      Enum.reduce(Enum.reverse(backpointers), [best_tag], fn bp, [current | _] = path ->
        prev = Map.get(bp, current)
        if prev, do: [prev | path], else: path
      end)

    # Take only as many tags as we need (drop START markers)
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
    # Average probability across all features
    probs =
      Enum.map(features, fn feature ->
        model.feature_weights
        |> Map.get(feature, %{})
        |> Map.get(tag, 0.001)
      end)

    if length(probs) > 0 do
      Enum.sum(probs) / length(probs)
    else
      # Fallback to prior
      Map.get(model.tag_priors, tag, 0.001)
    end
  end

  # ============================================================================
  # Helper Functions
  # ============================================================================

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
