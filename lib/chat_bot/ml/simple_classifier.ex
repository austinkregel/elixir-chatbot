defmodule ChatBot.ML.SimpleClassifier do
  @moduledoc """
  A simple text classifier using TF-IDF and cosine similarity.
  This is a lightweight alternative to the full SVM implementation.

  Uses the Tokenizer module for unicode-aware tokenization.
  """

  require Logger

  alias ChatBot.ML.Tokenizer

  def train(training_data) do
    Logger.info("Training simple classifier on #{length(training_data)} samples")

    # Build TF-IDF vectors for each sample
    {texts, labels} = Enum.unzip(training_data)

    # Build vocabulary using the new tokenizer
    all_words =
      texts
      |> Enum.flat_map(&tokenize/1)
      |> Enum.frequencies()
      |> Enum.filter(fn {_word, count} -> count >= 2 end)
      |> Enum.sort_by(fn {_word, count} -> -count end)
      |> Enum.take(1000)
      |> Enum.map(fn {word, _count} -> word end)

    vocabulary = all_words |> Enum.with_index() |> Enum.into(%{})

    # Calculate IDF weights
    num_docs = length(texts)

    idf_weights =
      vocabulary
      |> Enum.map(fn {word, _idx} ->
        doc_freq = Enum.count(texts, fn text -> word in tokenize(text) end)
        idf = :math.log(num_docs / max(doc_freq, 1))
        {word, idf}
      end)
      |> Enum.into(%{})

    # Vectorize all training samples
    vectors =
      texts
      |> Enum.map(fn text ->
        vectorize(text, vocabulary, idf_weights)
      end)

    # Group by label
    label_vectors =
      Enum.zip([labels, vectors])
      |> Enum.group_by(fn {label, _vector} -> label end, fn {_label, vector} -> vector end)

    # Calculate centroid for each label
    label_centroids =
      label_vectors
      |> Enum.map(fn {label, vecs} ->
        centroid = calculate_centroid(vecs)
        {label, centroid}
      end)
      |> Enum.into(%{})

    %{
      vocabulary: vocabulary,
      idf_weights: idf_weights,
      label_centroids: label_centroids
    }
  end

  def classify(text, model) do
    vector = vectorize(text, model.vocabulary, model.idf_weights)

    # Find nearest centroid
    {best_label, best_score} =
      model.label_centroids
      |> Enum.map(fn {label, centroid} ->
        similarity = cosine_similarity(vector, centroid)
        {label, similarity}
      end)
      |> Enum.max_by(fn {_label, score} -> score end)

    {:ok, best_label, best_score}
  end

  defp tokenize(text) do
    # Use the new Tokenizer module for unicode-aware tokenization
    # Expand contractions so "I'm" becomes "I am" - ensures consistent
    # tokenization between contracted and expanded forms
    Tokenizer.tokenize_normalized(text, min_length: 2, expand_contractions: true)
  end

  defp vectorize(text, vocabulary, idf_weights) do
    tokens = tokenize(text)
    token_freq = Enum.frequencies(tokens)

    # Build TF-IDF vector
    vector =
      vocabulary
      |> Enum.map(fn {word, _idx} ->
        tf = Map.get(token_freq, word, 0)
        idf = Map.get(idf_weights, word, 0.0)
        tf * idf
      end)

    # Normalize
    magnitude = :math.sqrt(Enum.reduce(vector, 0, fn val, acc -> acc + val * val end))

    if magnitude > 0 do
      Enum.map(vector, &(&1 / magnitude))
    else
      vector
    end
  end

  defp calculate_centroid(vectors) do
    if length(vectors) == 0 do
      []
    else
      vec_length = length(List.first(vectors))

      for i <- 0..(vec_length - 1) do
        sum =
          Enum.reduce(vectors, 0, fn vec, acc ->
            acc + Enum.at(vec, i)
          end)

        sum / length(vectors)
      end
    end
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
end
