defmodule Brain.ML.IntentClassifier do
  @moduledoc "**DEPRECATED** - This module is unused dead code. Use `Brain.ML.IntentClassifierSimple` instead.\n\nThis was an early intent classifier that was described as \"SVM with TF-IDF features\"\nbut actually implements TF-IDF vectorization with nearest-neighbor cosine similarity\n(not a true SVM). It is not referenced anywhere in the codebase and is not started\nin any supervision tree.\n\n`Brain.ML.IntentClassifierSimple` is the active intent classifier used throughout\nthe application.\n"

  alias Brain.ML.Tokenizer
  alias Nx.Tensor
  require Logger

  @type classification_result :: %{
          intent: String.t(),
          confidence: float(),
          probabilities: %{String.t() => float()}
        }

  @type tfidf_vectorizer :: %{
          vocabulary: %{String.t() => integer()},
          idf_weights: Tensor.t(),
          max_features: integer()
        }

  @type svm_model :: %{
          model: any(),
          label_encoder: %{
            label_to_index: %{String.t() => integer()},
            index_to_label: %{integer() => String.t()}
          }
        }

  @doc "Load pre-trained models from disk.\nReturns {:ok, models} or {:error, reason}.\n"
  def load_models do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")

    with {:ok, vectorizer} <- load_vectorizer(models_path),
         {:ok, svm_model} <- load_svm_model(models_path) do
      Logger.info("ML models loaded successfully")
      {:ok, %{vectorizer: vectorizer, svm_model: svm_model}}
    else
      {:error, reason} ->
        Logger.error("Failed to load ML models", %{reason: reason})
        {:error, reason}
    end
  end

  @doc "Classify intent from text using pre-trained models.\nReturns {:ok, result} or {:error, reason}.\n"
  def classify(text, models \\ nil) do
    models = models || get_loaded_models()

    case models do
      {:ok, %{vectorizer: vectorizer, svm_model: svm_model}} ->
        classify_with_models(text, vectorizer, svm_model)

      {:error, reason} ->
        {:error, reason}

      nil ->
        {:error, "Models not loaded"}
    end
  end

  @doc "Vectorize text using TF-IDF vectorizer.\n"
  def vectorize_text(text, vectorizer) do
    vectorize_single_text(text, vectorizer)
  end

  defp load_vectorizer(models_path) do
    vectorizer_path = Path.join(models_path, "vectorizer.term")

    case File.read(vectorizer_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)

          vectorizer = %{
            vocabulary: data.vocabulary,
            idf_weights: Nx.from_binary(data.idf_weights, :f32),
            max_features: data.max_features
          }

          {:ok, vectorizer}
        rescue
          error ->
            {:error, "Failed to deserialize vectorizer: #{inspect(error)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read vectorizer: #{reason}"}
    end
  end

  defp load_svm_model(models_path) do
    svm_path = Path.join(models_path, "svm_model.term")

    case File.read(svm_path) do
      {:ok, binary} ->
        try do
          data = :erlang.binary_to_term(binary)
          {:ok, data}
        rescue
          error ->
            {:error, "Failed to deserialize SVM model: #{inspect(error)}"}
        end

      {:error, reason} ->
        {:error, "Failed to read SVM model: #{reason}"}
    end
  end

  defp get_loaded_models do
    case Application.get_env(:brain, :ml_models) do
      nil ->
        load_models()

      models ->
        {:ok, models}
    end
  end

  defp classify_with_models(text, vectorizer, svm_model) do
    try do
      text_vector = vectorize_single_text(text, vectorizer)
      text_vector = Nx.reshape(text_vector, {1, -1})
      predicted_index = predict_nearest_neighbor(svm_model.model, text_vector)
      probabilities = get_prediction_probabilities(svm_model, text_vector, predicted_index)

      predicted_intent =
        Map.get(svm_model.label_encoder.index_to_label, predicted_index, "unknown")

      confidence = calculate_confidence(probabilities, predicted_index)

      result = %{
        intent: predicted_intent,
        confidence: confidence,
        probabilities: probabilities
      }

      {:ok, result}
    rescue
      error ->
        Logger.error("Classification failed", %{error: inspect(error), text: text})
        {:error, "Classification failed: #{inspect(error)}"}
    end
  end

  defp vectorize_single_text(text, vectorizer) do
    tokens = tokenize_text(text)
    vocab_size = vectorizer.max_features
    tf_counts = Enum.frequencies(tokens)

    tf_vector =
      Enum.reduce(vectorizer.vocabulary, Nx.broadcast(0.0, {vocab_size}), fn {term, index}, acc ->
        tf = Map.get(tf_counts, term, 0)

        if tf > 0 do
          Nx.put_slice(acc, [index], Nx.tensor([tf]))
        else
          acc
        end
      end)

    tfidf_vector = Nx.multiply(tf_vector, vectorizer.idf_weights)
    norm = Nx.reduce_max(tfidf_vector)

    if Nx.to_number(norm) > 0 do
      Nx.divide(tfidf_vector, norm)
    else
      tfidf_vector
    end
  end

  defp tokenize_text(text) do
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp predict_nearest_neighbor(model, text_vector) do
    training_vectors = model.training_vectors
    training_labels = model.training_labels
    num_samples = Nx.axis_size(training_vectors, 0)

    similarities =
      for i <- 0..(num_samples - 1) do
        training_vec = Nx.slice(training_vectors, [i], [1])
        dot_product = Nx.sum(Nx.multiply(text_vector, training_vec))
        norm_text = Nx.sqrt(Nx.sum(Nx.multiply(text_vector, text_vector)))
        norm_training = Nx.sqrt(Nx.sum(Nx.multiply(training_vec, training_vec)))

        if Nx.to_number(norm_text) > 0 and Nx.to_number(norm_training) > 0 do
          Nx.to_number(dot_product) / (Nx.to_number(norm_text) * Nx.to_number(norm_training))
        else
          0.0
        end
      end

    {_max_similarity, best_index} =
      similarities
      |> Enum.with_index()
      |> Enum.max_by(fn {sim, _idx} -> sim end)

    label_index = Nx.to_number(Nx.slice(training_labels, [best_index], [1]))
    label_index
  end

  defp get_prediction_probabilities(svm_model, text_vector, _predicted_index) do
    try do
      model = svm_model.model
      training_vectors = model.training_vectors
      training_labels = model.training_labels
      num_samples = Nx.axis_size(training_vectors, 0)

      similarities =
        for i <- 0..(num_samples - 1) do
          training_vec = Nx.slice(training_vectors, [i], [1])
          dot_product = Nx.sum(Nx.multiply(text_vector, training_vec))
          norm_text = Nx.sqrt(Nx.sum(Nx.multiply(text_vector, text_vector)))
          norm_training = Nx.sqrt(Nx.sum(Nx.multiply(training_vec, training_vec)))

          if Nx.to_number(norm_text) > 0 and Nx.to_number(norm_training) > 0 do
            Nx.to_number(dot_product) / (Nx.to_number(norm_text) * Nx.to_number(norm_training))
          else
            0.0
          end
        end

      label_similarities =
        similarities
        |> Enum.with_index()
        |> Enum.group_by(fn {_sim, idx} ->
          Nx.to_number(Nx.slice(training_labels, [idx], [1]))
        end)
        |> Enum.into(%{}, fn {label_idx, sims} ->
          max_sim = Enum.max_by(sims, fn {sim, _idx} -> sim end) |> elem(0)
          {label_idx, max_sim}
        end)

      probabilities =
        svm_model.label_encoder.index_to_label
        |> Enum.map(fn {index, label} ->
          similarity = Map.get(label_similarities, index, 0.0)
          probability = max(0.0, similarity)
          {label, probability}
        end)
        |> Enum.into(%{})

      total = Enum.sum(Map.values(probabilities))

      if total > 0 do
        Enum.into(probabilities, %{}, fn {label, prob} -> {label, prob / total} end)
      else
        num_classes = map_size(svm_model.label_encoder.index_to_label)
        uniform_prob = 1.0 / num_classes

        svm_model.label_encoder.index_to_label
        |> Enum.into(%{}, fn {_index, label} -> {label, uniform_prob} end)
      end
    rescue
      _error ->
        num_classes = map_size(svm_model.label_encoder.index_to_label)
        uniform_prob = 1.0 / num_classes

        svm_model.label_encoder.index_to_label
        |> Enum.into(%{}, fn {_index, label} -> {label, uniform_prob} end)
    end
  end

  defp calculate_confidence(probabilities, _predicted_index) do
    sorted_probs =
      probabilities
      |> Map.values()
      |> Enum.sort(:desc)

    case sorted_probs do
      [top_prob | [second_prob | _]] ->
        min(0.99, max(0.1, top_prob - second_prob + 0.5))

      [top_prob] ->
        min(0.99, max(0.1, top_prob))

      [] ->
        0.1
    end
  end
end