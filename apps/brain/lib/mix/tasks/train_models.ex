defmodule Mix.Tasks.TrainModels do
  @moduledoc "Mix task to train ML models from training data.\n\n## Usage\n\n    mix train_models [options]\n\n## Options\n\n  --world <id>     Train models for a specific world (default: saves to priv/ml_models/)\n  --intent-only    Train only the intent classifier\n  --entity-only    Train only the entity recognition model\n  --pos-only       Train only the POS tagger model\n  --gazetteer-only Build only the gazetteer lookup tables\n  --skip-gazetteer Skip gazetteer building (faster training)\n  --skip-pos       Skip POS tagger training\n  --lstm-only      Train ONLY the LSTM multi-task model\n  --lstm-intent    Train LSTM intent classifier only\n  --lstm-joint     Train LSTM joint model (intent + NER)\n  --lstm-epochs N  Number of epochs for LSTM training (default: 10)\n  --hidden-size N  LSTM hidden dimension (default: 128)\n  --embedding-size N  LSTM embedding dimension (default: 128)\n  --batch-size N   Training batch size (default: 32)\n  --skip-lstm      Skip LSTM training (train only TF-IDF models)\n\n## World-Specific Training\n\nWhen --world is specified, models are saved to:\n  priv/training_worlds/{world_id}/models/\n\nThis allows each world to have its own isolated ML models. When no world is\nspecified, models are saved to the default location (priv/ml_models/).\n\n## Examples\n\n    # Train all models for the default location\n    mix train_models\n\n    # Train all models for the \"star_trek\" world\n    mix train_models --world star_trek\n\n    # Train only intent classifier for a world\n    mix train_models --world my_world --intent-only\n\n    # Train all models (TF-IDF + LSTM, default behavior)\n    mix train_models\n\n    # Train with more LSTM epochs\n    mix train_models --lstm-epochs 20\n\n    # Skip LSTM training (faster, TF-IDF only)\n    mix train_models --skip-lstm\n\n    # Train ONLY the LSTM model (skip TF-IDF)\n    mix train_models --lstm-only\n\nThis task will:\n- Load intent training data from data/intents/ (or data/training/intents/)\n- Load entity definitions from data/entities/\n- Load supplementary data (cities, artists, emojis) from CSVs\n- Build TF-IDF vectorizer and train intent classifier\n- Train BIO-tagged entity recognition model\n- Train POS tagger from annotated data (if available)\n- Build gazetteer lookup tables for fast entity extraction\n- Save all models to priv/ml_models/ (or world-specific path)\n- Report training statistics and model sizes\n"

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  use Mix.Task
  require Logger
  alias Brain.ML.Trainer
  alias Brain.ML.POSTagger
  alias Brain.ML.LSTM.UnifiedModel

  @shortdoc "Train ML models from training data"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          world: :string,
          intent_only: :boolean,
          entity_only: :boolean,
          pos_only: :boolean,
          gazetteer_only: :boolean,
          skip_gazetteer: :boolean,
          skip_pos: :boolean,
          lstm_only: :boolean,
          lstm_intent: :boolean,
          lstm_joint: :boolean,
          lstm_epochs: :integer,
          hidden_size: :integer,
          embedding_size: :integer,
          batch_size: :integer,
          skip_lstm: :boolean
        ]
      )

    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")
    world_id = Keyword.get(opts, :world)
    models_path = get_models_path(world_id)

    if world_id do
      Logger.info("Starting ML model training pipeline for world: #{world_id}")
      Mix.shell().info("Training models for world: #{world_id}")
      Mix.shell().info("Models will be saved to: #{models_path}")
    else
      Logger.info("Starting ML model training pipeline (default models)...")
    end

    File.mkdir_p!(models_path)
    training_data_path = Application.get_env(:brain, :ml)[:training_data_path] || "data"

    if not File.exists?(training_data_path) do
      Mix.shell().error("Training data path not found: #{training_data_path}")
      Mix.shell().error("Please ensure training data is available.")
      System.halt(1)
    end

    display_data_sources(training_data_path)
    start_time = System.monotonic_time(:millisecond)

    lstm_opts = %{
      epochs: Keyword.get(opts, :lstm_epochs, 60),
      hidden_size: Keyword.get(opts, :hidden_size, 128),
      embedding_size: Keyword.get(opts, :embedding_size, 128),
      batch_size: Keyword.get(opts, :batch_size, 32)
    }

    skip_lstm = Keyword.get(opts, :skip_lstm, false)

    result =
      cond do
        Keyword.get(opts, :lstm_only, false) ->
          run_lstm_multitask_training(lstm_opts)

        Keyword.get(opts, :lstm_intent, false) ->
          run_lstm_intent_training(lstm_opts)

        Keyword.get(opts, :lstm_joint, false) ->
          run_lstm_joint_training(lstm_opts)

        Keyword.get(opts, :intent_only, false) ->
          run_intent_training(models_path)

        Keyword.get(opts, :entity_only, false) ->
          run_entity_training(models_path)

        Keyword.get(opts, :pos_only, false) ->
          run_pos_training(models_path)

        Keyword.get(opts, :gazetteer_only, false) ->
          run_gazetteer_building(models_path)

        Keyword.get(opts, :skip_gazetteer, false) ->
          run_training_without_gazetteer(models_path)

        true ->
          skip_pos = Keyword.get(opts, :skip_pos, false)
          run_full_training(skip_pos, skip_lstm, lstm_opts, models_path)
      end

    end_time = System.monotonic_time(:millisecond)
    duration = end_time - start_time

    case result do
      {:ok, stats} ->
        display_success(stats, duration, models_path)

      {:error, reason} ->
        display_error(reason)
        System.halt(1)
    end
  end

  defp get_models_path(nil) do
    Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
  end

  defp get_models_path(world_id) do
    world_path = Persistence.world_path(world_id)
    Path.join(world_path, "models")
  end

  defp run_intent_training(models_path) do
    Mix.shell().info("Training intent classifier only...")
    {stats, result} = Trainer.train_intent_classifier(%{}, models_path: models_path)

    case result do
      :ok -> {:ok, stats}
      {:error, reason} -> {:error, reason}
    end
  end

  defp run_entity_training(models_path) do
    Mix.shell().info("Training entity recognition model only...")
    stats = Trainer.train_entity_model(%{}, models_path: models_path)
    {:ok, stats}
  end

  defp run_gazetteer_building(_models_path) do
    Mix.shell().info("Building gazetteer lookup tables only...")
    stats = Trainer.build_gazetteer_data()
    {:ok, stats}
  end

  defp run_training_without_gazetteer(models_path) do
    Mix.shell().info("Training models (skipping gazetteer)...")

    stats = %{
      intent_samples: 0,
      vocab_size: 0,
      entity_model_trained: false
    }

    {stats, result} = Trainer.train_intent_classifier(stats, models_path: models_path)

    case result do
      :ok ->
        stats = Trainer.train_entity_model(stats, models_path: models_path)
        {:ok, stats}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp run_full_training(skip_pos, skip_lstm, lstm_opts, models_path) do
    case Trainer.train_and_save(models_path: models_path) do
      {:ok, stats} ->
        stats =
          if skip_pos do
            stats
          else
            case run_pos_training_internal(models_path) do
              {:ok, pos_stats} -> Map.merge(stats, pos_stats)
              {:error, _reason} -> stats
            end
          end

        if skip_lstm do
          {:ok, stats}
        else
          Mix.shell().info("")
          Mix.shell().info("Now training LSTM models (GPU accelerated)...")

          case run_lstm_multitask_training(lstm_opts) do
            {:ok, lstm_stats} ->
              {:ok, Map.merge(stats, lstm_stats)}

            {:error, reason} ->
              {:error, {:lstm_training_failed, reason}}
          end
        end

      error ->
        error
    end
  end

  defp run_pos_training(models_path) do
    Mix.shell().info("Training POS tagger model only...")
    run_pos_training_internal(models_path)
  end

  defp run_lstm_multitask_training(lstm_opts) do
    run_lstm_unified_training(lstm_opts, :multitask)
  end

  defp run_lstm_intent_training(lstm_opts) do
    run_lstm_unified_training(lstm_opts, :intent)
  end

  defp run_lstm_joint_training(lstm_opts) do
    run_lstm_unified_training(lstm_opts, :joint)
  end

  defp run_lstm_unified_training(lstm_opts, type) do
    epochs = Map.get(lstm_opts, :epochs, 10)
    hidden_size = Map.get(lstm_opts, :hidden_size, 128)
    embedding_size = Map.get(lstm_opts, :embedding_size, 128)
    batch_size = Map.get(lstm_opts, :batch_size, 32)

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("Training LSTM Model (EXLA GPU Accelerated)")
    Mix.shell().info("=" |> String.duplicate(60))
    Mix.shell().info("")
    Mix.shell().info("This model powers:")
    Mix.shell().info("  - Intent Classification")
    Mix.shell().info("  - Named Entity Recognition")
    Mix.shell().info("  - Sentiment Analysis")
    Mix.shell().info("  - Speech Act Classification")
    Mix.shell().info("")
    Mix.shell().info("Configuration:")
    Mix.shell().info("  Epochs: #{epochs}")
    Mix.shell().info("  Hidden size: #{hidden_size}")
    Mix.shell().info("  Embedding size: #{embedding_size}")
    Mix.shell().info("  Batch size: #{batch_size}")
    Mix.shell().info("  Model type: #{type}")
    Mix.shell().info("  Backend: EXLA (GPU accelerated)")
    Mix.shell().info("")

    config = [
      epochs: epochs,
      hidden_size: hidden_size,
      embedding_size: embedding_size,
      batch_size: batch_size,
      learning_rate: 0.001
    ]

    case UnifiedModel.train(config) do
      {:ok, model} ->
        stats = build_lstm_stats(model, type)
        {:ok, stats}

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_lstm_stats(result, type) do
    vocabularies = result.vocabularies

    base_stats = %{
      lstm_model_type: type,
      lstm_vocab_size: map_size(vocabularies.token_vocab),
      lstm_num_intents: map_size(vocabularies.intent_to_idx),
      lstm_num_bio_tags: map_size(vocabularies.bio_to_idx)
    }

    metrics = Map.get(result, :metrics, [])

    final_metrics =
      if is_list(metrics) and metrics != [] do
        List.last(metrics)
      else
        %{}
      end

    Map.merge(base_stats, %{
      lstm_epochs_trained:
        if(is_list(metrics)) do
          length(metrics)
        else
          0
        end,
      lstm_final_train_loss: Map.get(final_metrics, :train_loss, 0),
      lstm_final_val_loss: Map.get(final_metrics, :val_loss, 0),
      lstm_final_train_acc: Map.get(final_metrics, :train_acc, 0),
      lstm_final_val_acc: Map.get(final_metrics, :val_acc, 0)
    })
  end

  defp run_pos_training_internal(models_path) do
    gold_standard_path = Brain.priv_path("evaluation/intent/gold_standard.json")

    Mix.shell().info("  Loading POS training data from gold standard...")
    sequences = load_pos_from_gold_standard(gold_standard_path)

    if sequences != [] do
      Mix.shell().info("  Found #{length(sequences)} POS-annotated sequences from gold standard")

      case POSTagger.train(sequences) do
        {:ok, model} ->
          save_path = Path.join(models_path, "pos_model.term")

          case POSTagger.save_model(model, save_path) do
            {:ok, path} ->
              Mix.shell().info("  POS model saved to #{path}")

              {:ok,
               %{
                 pos_model_trained: true,
                 pos_tag_count: map_size(model.tag_vocabulary),
                 pos_feature_count: map_size(model.feature_weights)
               }}

            {:error, reason} ->
              Mix.shell().error("  Failed to save POS model: #{reason}")
              {:error, reason}
          end

        {:error, reason} ->
          Mix.shell().error("  POS training failed: #{reason}")
          {:error, reason}
      end
    else
      Mix.shell().info("  No POS-annotated data in gold standard. Skipping POS training.")
      Mix.shell().info("  Run: python scripts/enrich_gold_standard_pos.py")
      {:error, :no_training_data}
    end
  end

  defp load_pos_from_gold_standard(gold_standard_path) do
    case File.read(gold_standard_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, examples} when is_list(examples) ->
            examples
            |> Enum.filter(fn ex ->
              tokens = ex["tokens"] || []
              tags = ex["pos_tags"] || []
              tokens != [] and length(tokens) == length(tags)
            end)
            |> Enum.map(fn ex ->
              %{
                tokens: ex["tokens"],
                tags: ex["pos_tags"],
                source: ex["intent"]
              }
            end)

          _ ->
            Mix.shell().info("  Warning: Could not parse #{gold_standard_path}")
            []
        end

      {:error, reason} ->
        Mix.shell().info("  Warning: Could not read #{gold_standard_path}: #{inspect(reason)}")
        []
    end
  end

  defp display_data_sources(training_data_path) do
    Mix.shell().info("")
    Mix.shell().info("Training Data Sources:")
    Mix.shell().info("=" |> String.duplicate(50))
    enriched_intents = Path.join(training_data_path, "training/intents")
    legacy_intents = Path.join(training_data_path, "intents")

    intents_path =
      if File.exists?(enriched_intents) do
        enriched_intents
      else
        legacy_intents
      end

    if File.exists?(intents_path) do
      case File.ls(intents_path) do
        {:ok, files} ->
          json_files = Enum.filter(files, &String.ends_with?(&1, ".json"))

          label =
            if intents_path == enriched_intents do
              "(enriched)"
            else
              "(legacy)"
            end

          Mix.shell().info("  Intents:    #{length(json_files)} files #{label}")

        _ ->
          Mix.shell().info("  Intents:    (unable to list)")
      end
    else
      Mix.shell().error("  Intents:    NOT FOUND at #{legacy_intents}")
    end

    pos_data_path = Path.join(training_data_path, "training/pos/sequences.json")

    if File.exists?(pos_data_path) do
      Mix.shell().info("  POS Data:   training/pos/sequences.json")
    else
      if File.exists?(enriched_intents) do
        Mix.shell().info("  POS Data:   (from enriched intents)")
      else
        Mix.shell().info("  POS Data:   NOT FOUND (run 'mix migrate_training_data')")
      end
    end

    entities_dir = Path.join(training_data_path, "entities")

    if File.exists?(entities_dir) do
      case File.ls(entities_dir) do
        {:ok, files} ->
          json_files = Enum.filter(files, &String.ends_with?(&1, ".json"))
          Mix.shell().info("  Entities:   #{length(json_files)} files in #{entities_dir}")

        _ ->
          Mix.shell().info("  Entities:   (unable to list)")
      end
    else
      Mix.shell().info("  Entities:   NOT FOUND (optional)")
    end

    cities_path = Path.join(training_data_path, "world-cities.csv")

    if File.exists?(cities_path) do
      line_count = count_lines(cities_path)
      Mix.shell().info("  Cities:     ~#{line_count} entries in world-cities.csv")
    else
      Mix.shell().info("  Cities:     NOT FOUND (optional)")
    end

    artists_path = Path.join(training_data_path, "Global Music Artists.csv")

    if File.exists?(artists_path) do
      line_count = count_lines(artists_path)
      Mix.shell().info("  Artists:    ~#{line_count} entries in Global Music Artists.csv")
    else
      Mix.shell().info("  Artists:    NOT FOUND (optional)")
    end

    emojis_path = Path.join(training_data_path, "emojis.csv")

    if File.exists?(emojis_path) do
      line_count = count_lines(emojis_path)
      Mix.shell().info("  Emojis:     ~#{line_count} entries in emojis.csv")
    else
      Mix.shell().info("  Emojis:     NOT FOUND (optional)")
    end

    smalltalk_path = Path.join(training_data_path, "customSmalltalkResponses_en.json")

    if File.exists?(smalltalk_path) do
      Mix.shell().info("  Smalltalk:  customSmalltalkResponses_en.json")
    else
      Mix.shell().info("  Smalltalk:  NOT FOUND (optional)")
    end

    Mix.shell().info("")
  end

  defp count_lines(path) do
    case File.read(path) do
      {:ok, content} ->
        content
        |> String.split("\n")
        |> length()
        |> Kernel.-(1)

      _ ->
        0
    end
  end

  defp display_success(stats, duration, models_path) do
    Mix.shell().info("")
    Mix.shell().info("Training completed successfully!")
    Mix.shell().info("")
    Mix.shell().info("Training Statistics:")
    Mix.shell().info("=" |> String.duplicate(50))

    if Map.has_key?(stats, :intent_samples) and stats.intent_samples > 0 do
      Mix.shell().info("  Intent Classifier:")
      Mix.shell().info("    - Training samples: #{stats.intent_samples}")
      Mix.shell().info("    - Vocabulary size:  #{stats.vocab_size}")
    end

    if Map.get(stats, :entity_model_trained, false) do
      Mix.shell().info("  Entity Recognition:")
      Mix.shell().info("    - BIO model trained: Yes")
    end

    if Map.get(stats, :pos_model_trained, false) do
      Mix.shell().info("  POS Tagger:")
      Mix.shell().info("    - Tag vocabulary:   #{Map.get(stats, :pos_tag_count, 0)}")
      Mix.shell().info("    - Feature count:    #{Map.get(stats, :pos_feature_count, 0)}")
    end

    if Map.has_key?(stats, :lstm_model_type) do
      Mix.shell().info("  LSTM Multi-Task Model:")
      Mix.shell().info("    - Model type:       #{stats.lstm_model_type}")
      Mix.shell().info("    - Vocabulary size:  #{Map.get(stats, :lstm_vocab_size, 0)}")
      Mix.shell().info("    - Intent classes:   #{Map.get(stats, :lstm_num_intents, 0)}")
      Mix.shell().info("    - BIO tag classes:  #{Map.get(stats, :lstm_num_bio_tags, 0)}")
      Mix.shell().info("    - Epochs trained:   #{Map.get(stats, :lstm_epochs_trained, 0)}")

      final_val_acc = Map.get(stats, :lstm_final_val_acc, 0)

      if final_val_acc > 0 do
        Mix.shell().info(
          "    - Final train acc:  #{Float.round(Map.get(stats, :lstm_final_train_acc, 0) * 100, 1)}%"
        )

        Mix.shell().info("    - Final val acc:    #{Float.round(final_val_acc * 100, 1)}%")
      end
    end

    if Map.has_key?(stats, :gazetteer_entries) and stats.gazetteer_entries > 0 do
      Mix.shell().info("  Gazetteer:")
      Mix.shell().info("    - Total entries:    #{stats.gazetteer_entries}")
      Mix.shell().info("    - Entity types:     #{stats.entity_types}")
    end

    Mix.shell().info("")
    Mix.shell().info("  Total training time: #{format_duration(duration)}")
    Mix.shell().info("")
    Mix.shell().info("Saved Models:")
    Mix.shell().info("=" |> String.duplicate(50))

    display_model_file(models_path, "classifier.term", "Intent Classifier")
    display_model_file(models_path, "entity_model.term", "Entity Model")
    display_model_file(models_path, "pos_model.term", "POS Tagger")
    display_model_file(models_path, "gazetteer.term", "Gazetteer")
    display_model_file(models_path, "vectorizer.term", "TF-IDF Vectorizer")
    display_model_file(models_path, "embedder.term", "Embedder Vocabulary")
    lstm_path = Path.join(models_path, "lstm")
    display_model_file(lstm_path, "lstm_intent.term", "LSTM Intent Classifier")
    display_model_file(lstm_path, "lstm_joint.term", "LSTM Joint Model")
    display_model_file(lstm_path, "lstm_multitask.term", "LSTM Multi-Task Model")

    Mix.shell().info("")
    Mix.shell().info("Models saved to: #{models_path}")
    Mix.shell().info("")
    Mix.shell().info("You can now start the application to use the trained models.")
  end

  defp display_model_file(models_path, filename, label) do
    path = Path.join(models_path, filename)

    if File.exists?(path) do
      size = File.stat!(path).size
      Mix.shell().info("  #{label}: #{format_file_size(size)}")
    end
  end

  defp display_error(reason) do
    Mix.shell().error("")
    Mix.shell().error("Training failed: #{inspect(reason)}")
    Mix.shell().error("")
    Mix.shell().error("Please check:")
    Mix.shell().error("  - Training data format and locations")
    Mix.shell().error("  - Dependencies (Nx for tensor operations)")
    Mix.shell().error("  - Available memory (large gazetteers need RAM)")
    Mix.shell().error("")
    Mix.shell().error("Try running with --skip-gazetteer to reduce memory usage")
  end

  defp format_duration(ms) when ms < 1000 do
    "#{ms}ms"
  end

  defp format_duration(ms) when ms < 60_000 do
    "#{Float.round(ms / 1000, 1)}s"
  end

  defp format_duration(ms) do
    "#{Float.round(ms / 60000, 1)}m"
  end

  defp format_file_size(bytes) do
    cond do
      bytes < 1024 -> "#{bytes} B"
      bytes < 1024 * 1024 -> "#{Float.round(bytes / 1024, 1)} KB"
      bytes < 1024 * 1024 * 1024 -> "#{Float.round(bytes / (1024 * 1024), 1)} MB"
      true -> "#{Float.round(bytes / (1024 * 1024 * 1024), 1)} GB"
    end
  end
end