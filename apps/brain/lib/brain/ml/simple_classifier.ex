defmodule Brain.ML.SimpleClassifier do
  @moduledoc """
  A simple text classifier using TF-IDF and cosine similarity.
  This is a lightweight alternative to the full SVM implementation.

  Uses the Tokenizer module for unicode-aware tokenization.
  """

  require Logger

  alias Brain.ML.Tokenizer

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
    classify_with_details(text, model, top_k: 1)
  end

  @doc """
  Classifies text and returns detailed results including top-k scores and margin.
  
  Returns `{:ok, best_label, best_score, details}` where details contains:
  - `:second_score` - score of second-best intent
  - `:margin` - difference between best and second score
  - `:top_k` - list of {label, score} tuples for top k intents
  """
  def classify_with_details(text, model, opts \\ []) do
    vector = vectorize(text, model.vocabulary, model.idf_weights)
    top_k = Keyword.get(opts, :top_k, 5)

    # Calculate similarity to all centroids
    scored_intents =
      model.label_centroids
      |> Enum.map(fn {label, centroid} ->
        similarity = cosine_similarity(vector, centroid)
        {label, similarity}
      end)
      |> Enum.sort_by(fn {_label, score} -> -score end)

    # Extract top results
    [{best_label, best_score} | rest] = scored_intents
    second_score = if length(rest) > 0, do: elem(List.first(rest), 1), else: 0.0
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
  
  @doc """
  Save trained model to disk.
  """
  def save_model(model, path \\ nil) do
    model_path = path || get_model_path()
    File.mkdir_p!(Path.dirname(model_path))
    binary = :erlang.term_to_binary(model)
    File.write!(model_path, binary)
    Logger.info("Saved SimpleClassifier model", %{path: model_path})
    :ok
  end
  
  @doc """
  Load trained model from disk.
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
        {:error, "Failed to read model file: #{reason}"}
    end
  end
  
  defp get_model_path do
    models_path = Application.get_env(:brain, :ml)[:models_path] || "priv/ml_models"
    Path.join(models_path, "simple_classifier.term")
  end
end
