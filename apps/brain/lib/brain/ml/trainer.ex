defmodule Brain.ML.Trainer do
  @moduledoc """
  Training pipeline for building classical NLP models from training data.

  This module orchestrates the training process:
  - Loads intent and entity training data using DataLoaders
  - Builds gazetteers for fast entity lookup
  - Trains intent classifier using TF-IDF + centroid approach
  - Trains entity recognition model using BIO tagging
  - Serializes all models for production use
  """

  require Logger

  alias Brain.ML.{DataLoaders, Tokenizer, EntityTrainer}

  # Configure Nx backend for optimal performance
  defp configure_nx_backend do
    Nx.default_backend(Nx.BinaryBackend)
    Logger.info("Using optimized CPU backend for training")
  end

  # {text, intent_label}
  @type training_sample :: {String.t(), String.t()}
  @type tfidf_vectorizer :: %{
          vocabulary: %{String.t() => integer()},
          idf_weights: Nx.Tensor.t(),
          max_features: integer()
        }

  @type training_stats :: %{
          intent_samples: integer(),
          vocab_size: integer(),
          entity_types: integer(),
          gazetteer_entries: integer()
        }

  # Client API

  @doc """
  Main training function that loads data, trains all models, and saves them.
  Returns {:ok, stats} or {:error, reason}.

  ## Options
    - models_path: Override the default models output path
  """
  def train_and_save(opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path, Application.get_env(:brain, :ml)[:models_path])

    Logger.info("Starting ML model training pipeline", %{models_path: models_path})

    # Configure Nx backend
    configure_nx_backend()

    stats = %{
      intent_samples: 0,
      vocab_size: 0,
      entity_types: 0,
      gazetteer_entries: 0,
      entity_model_trained: false
    }

    # Step 1: Load and train intent classifier
    {stats, result} = train_intent_classifier(stats, models_path: models_path)

    case result do
      :ok ->
        # Step 2: Train entity recognition model
        stats = train_entity_model(stats, models_path: models_path)

        # Step 3: Build and save gazetteer data
        stats = build_gazetteer_data(stats, models_path: models_path)

        Logger.info("Training pipeline completed", stats)
        {:ok, stats}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Train only the intent classifier.

  ## Options
    - models_path: Override the default models output path
  """
  def train_intent_classifier(stats \\ %{}, opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path, Application.get_env(:brain, :ml)[:models_path])

    Logger.info("Training intent classifier...")

    # Load training data using new DataLoaders
    training_data = load_training_data()
    Logger.info("Loaded training data", %{samples: length(training_data)})

    if length(training_data) == 0 do
      Logger.error("No training data found")
      {stats, {:error, "No training data"}}
    else
      # Train simple classifier
      model = Brain.ML.SimpleClassifier.train(training_data)

      Logger.info("Trained classifier", %{
        vocab_size: map_size(model.vocabulary),
        num_labels: map_size(model.label_centroids)
      })

      # Save model
      File.mkdir_p!(models_path)
      model_path = Path.join(models_path, "classifier.term")
      File.write!(model_path, :erlang.term_to_binary(model))
      Logger.info("Intent classifier saved", %{path: model_path})

      # Also save embedder vocabulary for this world
      embedder_model = %{
        vocabulary: model.vocabulary,
        idf_weights: build_idf_weights_from_model(model)
      }

      embedder_path = Path.join(models_path, "embedder.term")
      File.write!(embedder_path, :erlang.term_to_binary(embedder_model))
      Logger.info("Embedder vocabulary saved", %{path: embedder_path})

      updated_stats = %{
        stats
        | intent_samples: length(training_data),
          vocab_size: map_size(model.vocabulary)
      }

      {updated_stats, :ok}
    end
  end

  # Extract IDF weights from the classifier model's vocabulary
  defp build_idf_weights_from_model(model) do
    # The classifier model has vocabulary and idf weights computed during training
    # Return a simple map with uniform IDF for now (the actual IDF calculation is done during classification)
    model.vocabulary
    |> Enum.map(fn {word, _idx} -> {word, 1.0} end)
    |> Map.new()
  end

  @doc """
  Train the entity recognition model using BIO tagging.

  ## Options
    - models_path: Override the default models output path
  """
  def train_entity_model(stats \\ %{}, opts \\ []) do
    models_path =
      Keyword.get(opts, :models_path, Application.get_env(:brain, :ml)[:models_path])

    Logger.info("Training entity recognition model...")

    case EntityTrainer.train_and_save(models_path: models_path) do
      {:ok, model} ->
        Logger.info("Entity model trained and saved", %{
          tag_count: map_size(model.tag_vocabulary)
        })

        %{stats | entity_model_trained: true}

      {:error, reason} ->
        Logger.warning("Entity model training failed", %{reason: reason})
        stats
    end
  end

  @doc """
  Build gazetteer lookup data and save for fast runtime access.

  ## Options
    - models_path: Override the default models output path
  """
  def build_gazetteer_data(stats \\ %{}, opts \\ []) do
    Logger.info("Building gazetteer data...")

    models_path =
      Keyword.get(opts, :models_path, Application.get_env(:brain, :ml)[:models_path])
    File.mkdir_p!(models_path)

    gazetteer_data = %{
      entities: %{},
      cities: %{},
      artists: %{},
      emojis: %{}
    }

    # Load and process entity definitions
    gazetteer_data =
      case DataLoaders.load_all_entities() do
        {:ok, entities} ->
          entity_lookup = DataLoaders.build_entity_lookup(entities)
          Logger.info("Built entity lookup", %{entries: map_size(entity_lookup)})
          %{gazetteer_data | entities: entity_lookup}

        {:error, _} ->
          gazetteer_data
      end

    # Load and process cities (sample for faster loading)
    gazetteer_data =
      case DataLoaders.load_cities() do
        {:ok, cities} ->
          # Take top cities by population (assuming sorted) or all if small
          sampled_cities = if length(cities) > 10000, do: Enum.take(cities, 10000), else: cities
          city_lookup = DataLoaders.build_city_lookup(sampled_cities)
          Logger.info("Built city lookup", %{entries: map_size(city_lookup)})
          %{gazetteer_data | cities: city_lookup}

        {:error, _} ->
          gazetteer_data
      end

    # Load and process artists (sample for faster loading)
    gazetteer_data =
      case DataLoaders.load_artists() do
        {:ok, artists} ->
          # Take subset of artists
          sampled_artists =
            if length(artists) > 10000, do: Enum.take(artists, 10000), else: artists

          artist_lookup = DataLoaders.build_artist_lookup(sampled_artists)
          Logger.info("Built artist lookup", %{entries: map_size(artist_lookup)})
          %{gazetteer_data | artists: artist_lookup}

        {:error, _} ->
          gazetteer_data
      end

    # Load and process emojis
    gazetteer_data =
      case DataLoaders.load_emojis() do
        {:ok, emojis} ->
          emoji_lookup = DataLoaders.build_emoji_lookup(emojis)
          Logger.info("Built emoji lookup", %{entries: map_size(emoji_lookup)})
          %{gazetteer_data | emojis: emoji_lookup}

        {:error, _} ->
          gazetteer_data
      end

    # Merge all lookups into a single combined gazetteer
    combined_lookup =
      gazetteer_data.entities
      |> Map.merge(gazetteer_data.cities)
      |> Map.merge(gazetteer_data.artists)
      |> Map.merge(gazetteer_data.emojis)

    # Save combined gazetteer
    gazetteer_path = Path.join(models_path, "gazetteer.term")
    File.write!(gazetteer_path, :erlang.term_to_binary(combined_lookup))

    Logger.info("Gazetteer saved", %{
      path: gazetteer_path,
      total_entries: map_size(combined_lookup)
    })

    # Update stats
    total_entries =
      map_size(gazetteer_data.entities) +
        map_size(gazetteer_data.cities) +
        map_size(gazetteer_data.artists) +
        map_size(gazetteer_data.emojis)

    stats
    |> Map.put(:gazetteer_entries, total_entries)
    |> Map.put(:entity_types, count_entity_types(gazetteer_data))
  end

  defp count_entity_types(gazetteer_data) do
    all_lookups = [
      gazetteer_data.entities,
      gazetteer_data.cities,
      gazetteer_data.artists,
      gazetteer_data.emojis
    ]

    all_lookups
    |> Enum.flat_map(fn lookup ->
      Map.values(lookup)
      |> Enum.flat_map(fn info ->
        # Handle both single maps and lists of maps
        case info do
          entries when is_list(entries) ->
            Enum.map(entries, fn entry -> Map.get(entry, :entity_type) end)
          entry when is_map(entry) ->
            [Map.get(entry, :entity_type)]
          _ ->
            []
        end
      end)
    end)
    |> Enum.uniq()
    |> Enum.filter(&(&1 != nil))
    |> length()
  end

  @doc """
  Load training data from intent files using DataLoaders.
  Returns a list of {text, intent_label} tuples.
  """
  def load_training_data do
    # Use the new DataLoaders module for consistent loading
    case DataLoaders.load_all_intents() do
      {:ok, examples} ->
        # Convert to {text, intent} tuple format
        examples
        |> Enum.map(fn example ->
          {example.text, example.intent}
        end)
        |> Enum.filter(fn {text, _intent} ->
          String.trim(text) != ""
        end)

      {:error, reason} ->
        Logger.error("Failed to load training data", %{reason: reason})
        # Fallback to legacy loading
        load_training_data_legacy()
    end
  end

  @doc """
  Legacy training data loading (fallback).
  """
  def load_training_data_legacy do
    intents_path = Path.join(Application.get_env(:brain, :ml)[:training_data_path], "intents")

    case File.ls(intents_path) do
      {:ok, files} ->
        json_files = Enum.filter(files, &String.ends_with?(&1, ".json"))

        usersays_files = Enum.filter(json_files, &String.contains?(&1, "_usersays_"))
        other_intent_files = json_files -- usersays_files

        Logger.info("Loading training data from #{length(json_files)} intent files", %{
          usersays: length(usersays_files),
          others: length(other_intent_files)
        })

        usersays_samples =
          Enum.flat_map(usersays_files, fn file ->
            file_path = Path.join(intents_path, file)
            intent_name = extract_intent_name(file)

            case File.read(file_path) do
              {:ok, content} ->
                parse_usersays_file(content, intent_name)

              {:error, reason} ->
                Logger.warning("Failed to read usersays file", %{file: file, reason: reason})
                []
            end
          end)

        other_samples =
          Enum.flat_map(other_intent_files, fn file ->
            file_path = Path.join(intents_path, file)
            intent_name = base_intent_name(file)

            case File.read(file_path) do
              {:ok, content} ->
                parse_general_intent_file(content, intent_name)

              {:error, reason} ->
                Logger.debug("Skipping unreadable intent file", %{file: file, reason: reason})
                []
            end
          end)

        usersays_samples ++ other_samples

      {:error, reason} ->
        Logger.error("Failed to list intents directory", %{path: intents_path, reason: reason})
        []
    end
  end

  @doc """
  Build TF-IDF vectorizer from training data.
  """
  def build_tfidf_vectorizer(training_data) do
    # Extract all texts
    texts = Enum.map(training_data, fn {text, _label} -> text end)

    # Build vocabulary
    vocabulary = build_vocabulary(texts)

    # Calculate IDF weights
    idf_weights = calculate_idf_weights(texts, vocabulary)

    %{
      vocabulary: vocabulary,
      idf_weights: idf_weights,
      max_features: map_size(vocabulary)
    }
  end

  @doc """
  Train simple classifier using TF-IDF features.
  For now, we'll use a simple nearest neighbor approach.
  """
  def train_svm_classifier(training_data, vectorizer) do
    # Prepare training data
    {texts, labels} = Enum.unzip(training_data)

    # Vectorize texts
    Logger.debug("Starting vectorization", %{num_texts: length(texts)})
    Logger.debug("First few texts: #{inspect(Enum.take(texts, 3))}")

    Logger.debug(
      "Vectorizer info: vocab_size=#{map_size(vectorizer.vocabulary)}, max_features=#{vectorizer.max_features}"
    )

    X = vectorize_texts(texts, vectorizer)
    Logger.debug("Vectorized texts", %{shape: Nx.shape(X), type: Nx.type(X)})

    # Encode labels
    {y, label_encoder} = encode_labels(labels)
    Logger.debug("Encoded labels", %{shape: Nx.shape(y)})

    # For now, store the training data for nearest neighbor classification
    # In a real implementation, you'd train an actual SVM
    classifier = %{
      training_vectors: X,
      training_labels: y,
      label_encoder: label_encoder
    }

    %{
      model: classifier,
      label_encoder: label_encoder
    }
  end

  @doc """
  Save trained models to disk.
  """
  def save_models(vectorizer, svm_model) do
    models_path = Application.get_env(:brain, :ml)[:models_path]

    # Ensure directory exists
    File.mkdir_p!(models_path)

    # Save vectorizer
    vectorizer_path = Path.join(models_path, "vectorizer.term")

    vectorizer_data = %{
      vocabulary: vectorizer.vocabulary,
      idf_weights: Nx.to_binary(vectorizer.idf_weights),
      max_features: vectorizer.max_features
    }

    File.write!(vectorizer_path, :erlang.term_to_binary(vectorizer_data))

    # Save SVM model
    svm_path = Path.join(models_path, "svm_model.term")

    svm_data = %{
      model: svm_model.model,
      label_encoder: svm_model.label_encoder
    }

    File.write!(svm_path, :erlang.term_to_binary(svm_data))

    Logger.info("Models saved", %{vectorizer_path: vectorizer_path, svm_path: svm_path})
  end

  # Private Functions

  defp extract_intent_name(filename) do
    filename
    |> String.replace("_usersays_en.json", "")
    |> String.replace("_usersays.json", "")
    |> String.replace_suffix(".json", "")
    |> String.replace("_", ".")
  end

  defp base_intent_name(filename) do
    filename
    |> String.replace_suffix(".json", "")
    |> String.replace("_", ".")
  end

  # Parse Dialogflow-like usersays entries
  defp parse_usersays_file(content, intent_name) do
    case Jason.decode(content) do
      {:ok, examples} when is_list(examples) ->
        Enum.flat_map(examples, fn example ->
          text = extract_text_from_example(example)

          if text == "" do
            []
          else
            [{text, intent_name}]
          end
        end)

      _ ->
        []
    end
  end

  # Parse general intent JSON that may contain "contexts", "responses", or plain structures
  defp parse_general_intent_file(content, intent_name) do
    case Jason.decode(content) do
      {:ok, %{"responses" => responses}} when is_list(responses) ->
        # Some files include example "messages" or "speech"
        Enum.flat_map(responses, fn resp ->
          msgs = Map.get(resp, "messages", [])

          Enum.flat_map(msgs, fn msg ->
            cond do
              is_binary(Map.get(msg, "speech")) ->
                [{Map.get(msg, "speech"), intent_name}]

              is_list(Map.get(msg, "speech")) ->
                Enum.map(Map.get(msg, "speech"), &{&1, intent_name})

              true ->
                []
            end
          end)
        end)

      {:ok, %{"userSays" => examples}} when is_list(examples) ->
        # Alternate key casing
        Enum.map(examples, fn ex -> {extract_text_from_example(ex), intent_name} end)

      {:ok, list} when is_list(list) ->
        # Some intents might be a list of simple strings
        Enum.flat_map(list, fn item ->
          cond do
            is_binary(item) ->
              [{item, intent_name}]

            is_map(item) ->
              text = Map.get(item, "text") || Map.get(item, "phrase") || ""
              if text == "", do: [], else: [{text, intent_name}]

            true ->
              []
          end
        end)

      _ ->
        []
    end
  end

  defp extract_text_from_example(example) do
    case Map.get(example, "data") do
      nil ->
        # Fallback: try to extract from other fields
        case Map.get(example, "text") do
          nil -> ""
          text -> text
        end

      data when is_list(data) ->
        # Extract text from data array
        Enum.map(data, fn item ->
          case item do
            %{"text" => text, "userDefined" => false} -> text
            %{"text" => text, "userDefined" => true} -> text
            _ -> ""
          end
        end)
        |> Enum.join("")

      _ ->
        ""
    end
  end

  defp build_vocabulary(texts) do
    # Tokenize and count words
    word_counts =
      texts
      |> Enum.flat_map(&tokenize_text/1)
      |> Enum.frequencies()

    # Filter by minimum frequency and take top N words
    min_freq = 2
    # Default to 5000 features if not configured
    max_features = Application.get_env(:brain, :ml)[:max_features] || 5000

    word_counts
    |> Enum.filter(fn {_word, count} -> count >= min_freq end)
    |> Enum.sort_by(fn {_word, count} -> count end, :desc)
    |> Enum.take(max_features)
    |> Enum.with_index()
    |> Enum.into(%{}, fn {{word, _count}, index} -> {word, index} end)
  end

  defp tokenize_text(text) do
    # Use the new Tokenizer module for unicode-aware tokenization
    Tokenizer.tokenize_normalized(text, min_length: 2)
  end

  defp calculate_idf_weights(texts, vocabulary) do
    _vocab_size = map_size(vocabulary)
    num_docs = length(texts)

    # Calculate IDF for each term
    idf_values =
      vocabulary
      |> Enum.map(fn {term, _index} ->
        # Count documents containing this term
        doc_freq =
          texts
          |> Enum.count(fn text ->
            tokens = tokenize_text(text)
            term in tokens
          end)

        # Calculate IDF
        idf = :math.log(num_docs / max(doc_freq, 1))
        {term, idf}
      end)
      |> Enum.into(%{})

    # Convert to tensor
    idf_list =
      vocabulary
      |> Enum.map(fn {term, index} ->
        {index, Map.get(idf_values, term, 0.0)}
      end)
      |> Enum.sort_by(fn {index, _} -> index end)
      |> Enum.map(fn {_index, idf} -> idf end)

    Nx.tensor(idf_list, type: :f32)
  end

  defp vectorize_texts(texts, vectorizer) do
    # Convert texts to TF-IDF vectors
    vectors =
      texts
      |> Enum.map(&vectorize_single_text(&1, vectorizer))

    # Stack vectors into a single tensor
    case vectors do
      [] ->
        Nx.broadcast(0.0, {0, vectorizer.max_features})

      [_first | _] ->
        Nx.stack(vectors)
    end
  end

  defp vectorize_single_text(text, vectorizer) do
    tokens = tokenize_text(text)
    vocab_size = vectorizer.max_features

    # Calculate term frequencies
    tf_counts = Enum.frequencies(tokens)

    # Build TF vector as a list first, then convert to tensor
    tf_list =
      for i <- 0..(vocab_size - 1) do
        # Find the term for this index
        term = Enum.find(vectorizer.vocabulary, fn {_term, idx} -> idx == i end)

        case term do
          {term_name, _} -> Map.get(tf_counts, term_name, 0)
          nil -> 0
        end
      end

    # Convert to tensor
    tf_vector = Nx.tensor(tf_list, type: :f32)

    # Apply TF-IDF weighting
    tfidf_vector = Nx.multiply(tf_vector, vectorizer.idf_weights)

    # Normalize
    norm = Nx.reduce_max(tfidf_vector)

    if Nx.to_number(norm) > 0 do
      Nx.divide(tfidf_vector, norm)
    else
      tfidf_vector
    end
  end

  defp encode_labels(labels) do
    # Create label encoder
    unique_labels = labels |> Enum.uniq() |> Enum.sort()
    label_to_index = Enum.with_index(unique_labels) |> Enum.into(%{})

    index_to_label =
      Enum.with_index(unique_labels) |> Enum.into(%{}, fn {label, index} -> {index, label} end)

    # Encode labels
    encoded_labels =
      labels
      |> Enum.map(&Map.get(label_to_index, &1))
      |> Nx.tensor()

    label_encoder = %{
      label_to_index: label_to_index,
      index_to_label: index_to_label
    }

    {encoded_labels, label_encoder}
  end
end
