defmodule Brain.ML.SimpleClassifier do
  @moduledoc "A simple text classifier using TF-IDF and cosine similarity.\nThis is a lightweight alternative to the full SVM implementation.\n\nUses the Tokenizer module for unicode-aware tokenization.\n"

  require Logger

  alias Brain.ML.Tokenizer

  def train(training_data) do
    sample_count = length(training_data)
    Logger.info("[SimpleClassifier] Training on #{sample_count} samples...")
    {texts, labels} = Enum.unzip(training_data)

    doc_tokens = Enum.map(texts, &tokenize/1)

    all_words =
      doc_tokens
      |> Enum.flat_map(& &1)
      |> Enum.frequencies()
      |> Enum.filter(fn {_word, count} -> count >= 2 end)
      |> Enum.sort_by(fn {_word, count} -> -count end)
      |> Enum.take(1000)
      |> Enum.map(fn {word, _count} -> word end)

    vocabulary = all_words |> Enum.with_index() |> Enum.into(%{})
    num_docs = length(texts)
    Logger.info("[SimpleClassifier] Tokenized #{sample_count} samples, vocabulary size: #{map_size(vocabulary)}")

    doc_token_sets = Enum.map(doc_tokens, &MapSet.new/1)

    idf_weights =
      vocabulary
      |> Enum.map(fn {word, _idx} ->
        doc_freq = Enum.count(doc_token_sets, fn token_set -> MapSet.member?(token_set, word) end)
        idf = :math.log(num_docs / max(doc_freq, 1))
        {word, idf}
      end)
      |> Enum.into(%{})

    Logger.info("[SimpleClassifier] Computed IDF weights")

    vectors =
      texts
      |> Enum.map(fn text ->
        vectorize(text, vocabulary, idf_weights)
      end)

    Logger.info("[SimpleClassifier] Vectorized #{sample_count} samples")

    label_vectors =
      Enum.zip([labels, vectors])
      |> Enum.group_by(fn {label, _vector} -> label end, fn {_label, vector} -> vector end)

    label_centroids =
      label_vectors
      |> Enum.map(fn {label, vecs} ->
        Logger.debug("[SimpleClassifier] Computing centroid for label '#{label}' (#{length(vecs)} vectors)")
        centroid = calculate_centroid(vecs)
        {label, centroid}
      end)
      |> Enum.into(%{})

    label_count = map_size(label_centroids)
    Logger.info("[SimpleClassifier] Training complete: #{label_count} labels, vocab size #{map_size(vocabulary)}")

    %{
      vocabulary: vocabulary,
      idf_weights: idf_weights,
      label_centroids: label_centroids
    }
  end

  @doc """
  Incrementally updates a model with new training examples.

  `new_examples` is a list of `{text, label}` tuples.
  Updates vocabulary, IDF weights, and adjusts centroids without full retrain.

  Returns `{updated_model, incremental_update_count}`.
  """
  def update_model(model, new_examples, incremental_count \\ 0) do
    {texts, labels} = Enum.unzip(new_examples)

    # Extend vocabulary with new words
    all_words =
      texts
      |> Enum.flat_map(&Tokenizer.tokenize/1)
      |> Enum.frequencies()
      |> Enum.filter(fn {_word, count} -> count >= 1 end)

    existing_vocab = model.vocabulary
    max_index = if map_size(existing_vocab) > 0, do: Enum.max(Map.values(existing_vocab)), else: -1

    new_words =
      all_words
      |> Enum.reject(fn {word, _} -> Map.has_key?(existing_vocab, word) end)

    extended_vocab =
      new_words
      |> Enum.with_index(max_index + 1)
      |> Enum.reduce(existing_vocab, fn {{word, _count}, idx}, acc ->
        Map.put(acc, word, idx)
      end)

    # Update IDF weights (approximate with new docs)
    num_new_docs = length(texts)
    doc_word_sets = Enum.map(texts, fn text -> text |> Tokenizer.tokenize() |> MapSet.new() end)

    new_idf =
      extended_vocab
      |> Map.keys()
      |> Enum.reduce(model.idf_weights, fn word, idf_acc ->
        new_doc_freq = Enum.count(doc_word_sets, &MapSet.member?(&1, word))

        if new_doc_freq > 0 do
          old_idf = Map.get(idf_acc, word, 0.0)
          new_idf = :math.log(num_new_docs / max(new_doc_freq, 1))
          # EMA blend of old and new IDF
          blended = old_idf * 0.8 + new_idf * 0.2
          Map.put(idf_acc, word, blended)
        else
          idf_acc
        end
      end)

    # Adjust centroids with new examples
    new_vectors =
      texts
      |> Enum.map(&vectorize(&1, extended_vocab, new_idf))
      |> Enum.zip(labels)

    updated_centroids =
      Enum.reduce(new_vectors, model.label_centroids, fn {vec, label}, centroids ->
        case Map.get(centroids, label) do
          nil ->
            Map.put(centroids, label, vec)

          existing ->
            # Running average: blend new vector with existing centroid
            blended = blend_vectors(existing, vec, 0.9)
            Map.put(centroids, label, blended)
        end
      end)

    updated_model = %{
      model
      | vocabulary: extended_vocab,
        idf_weights: new_idf,
        label_centroids: updated_centroids
    }

    {updated_model, incremental_count + 1}
  end

  defp blend_vectors(v1, v2, alpha) when is_map(v1) and is_map(v2) do
    all_keys = MapSet.union(MapSet.new(Map.keys(v1)), MapSet.new(Map.keys(v2)))

    Map.new(all_keys, fn k ->
      val1 = Map.get(v1, k, 0.0)
      val2 = Map.get(v2, k, 0.0)
      {k, val1 * alpha + val2 * (1.0 - alpha)}
    end)
  end

  defp blend_vectors(v1, _v2, _alpha), do: v1

  def classify(text, model) do
    classify_with_details(text, model, top_k: 1)
  end

  @doc "Classifies text and returns detailed results including top-k scores and margin.\n\nReturns `{:ok, best_label, best_score, details}` where details contains:\n- `:second_score` - score of second-best intent\n- `:margin` - difference between best and second score\n- `:top_k` - list of {label, score} tuples for top k intents\n"
  def classify_with_details(text, model, opts \\ []) do
    vector = vectorize(text, model.vocabulary, model.idf_weights)
    top_k = Keyword.get(opts, :top_k, 5)

    scored_intents =
      model.label_centroids
      |> Enum.map(fn {label, centroid} ->
        similarity = cosine_similarity(vector, centroid)
        {label, similarity}
      end)
      |> Enum.sort_by(fn {_label, score} -> -score end)

    [{best_label, best_score} | rest] = scored_intents

    second_score =
      if rest != [] do
        elem(List.first(rest), 1)
      else
        0.0
      end

    margin = best_score - second_score
    top_k_list = Enum.take(scored_intents, top_k)

    details = %{
      second_score: second_score,
      margin: margin,
      top_k: top_k_list
    }

    {:ok, best_label, best_score, details}
  end

  defp tokenize(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2, expand_contractions: true)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    expanded = expand_tokens_with_lexicon(tokens, vocabulary)
    token_freq = Enum.frequencies(expanded)

    vector =
      vocabulary
      |> Enum.map(fn {word, _idx} ->
        tf = Map.get(token_freq, word, 0)
        idf = Map.get(idf_weights, word, 0.0)
        tf * idf
      end)

    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

  defp expand_tokens_with_lexicon(tokens, vocabulary) do
    if Process.whereis(Brain.ML.Lexicon) do
      Brain.ML.Lexicon.expand_with_synonyms(tokens, vocabulary)
    else
      tokens
    end
  end

  defp calculate_centroid([]), do: []
  defp calculate_centroid([single]), do: single

  defp calculate_centroid(vectors) do
    count = length(vectors)

    vectors
    |> Enum.reduce(fn vec, acc -> Enum.zip_with(acc, vec, &Kernel.+/2) end)
    |> Enum.map(&(&1 / count))
  end

  defp cosine_similarity(vec1, vec2) do
    if length(vec1) != length(vec2) do
      0.0
    else
      dot_product =
        Enum.zip(vec1, vec2)
        |> Enum.reduce(0, fn {a, b}, acc -> acc + a * b end)

      mag1 = :math.sqrt(Enum.reduce(vec1, 0, fn val, acc -> acc + val * val end))
      mag2 = :math.sqrt(Enum.reduce(vec2, 0, fn val, acc -> acc + val * val end))

      if mag1 > 0 and mag2 > 0 do
        dot_product / (mag1 * mag2)
      else
        0.0
      end
    end
  end

  @doc "Save trained model to disk.\n"
  def save_model(model, path \\ nil) do
    model_path = path || get_model_path()
    File.mkdir_p!(Path.dirname(model_path))
    binary = :erlang.term_to_binary(model)
    File.write!(model_path, binary)
    Logger.info("Saved SimpleClassifier model", %{path: model_path})
    :ok
  end

  @doc "Load trained model from disk.\n"
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
        {:error, "Failed to read model file: #{reason}"}
    end
  end

  defp get_model_path do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join(models_path, "simple_classifier.term")
  end
end
