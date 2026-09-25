defmodule Mix.Tasks.TrainModels do
  @moduledoc "Mix task to train ML models from training data.\n\n## Usage\n\n    mix train_models [options]\n\n## Options\n\n  --world <id>     Train models for a specific world (default: saves to priv/ml_models/)\n  --intent-only    Train only the intent classifier\n  --entity-only    Train only the entity recognition model\n  --pos-only       Train only the POS tagger model\n  --skip-pos       Skip POS tagger training\n\n## World-Specific Training\n\nWhen --world is specified, models are saved to:\n  priv/training_worlds/{world_id}/models/\n\nThis allows each world to have its own isolated ML models. When no world is\nspecified, models are saved to the default location (priv/ml_models/).\n\n## Examples\n\n    # Train all models for the default location\n    mix train_models\n\n    # Train all models for the \"star_trek\" world\n    mix train_models --world star_trek\n\n    # Train only intent classifier for a world\n    mix train_models --world my_world --intent-only\n\nThis task will:\n- Load intent training data from data/intents/ (or data/training/intents/)\n- Build TF-IDF vectorizer and train intent classifier\n- Train BIO-tagged entity recognition model\n- Train POS tagger from annotated data (if available)\n- Save all models to priv/ml_models/ (or world-specific path)\n- Report training statistics and model sizes\n"

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  use Mix.Task
  require Logger
  alias Brain.ML.Trainer

  @shortdoc "Train ML models from training data"

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          world: :string,
          intent_only: :boolean,
          entity_only: :boolean,
          pos_only: :boolean,
          skip_pos: :boolean
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

    result =
      cond do
        Keyword.get(opts, :intent_only, false) ->
          run_intent_training(models_path)

        Keyword.get(opts, :entity_only, false) ->
          run_entity_training(models_path)

        Keyword.get(opts, :pos_only, false) ->
          run_pos_training(models_path)

        true ->
          skip_pos = Keyword.get(opts, :skip_pos, false)
          run_full_training(skip_pos, models_path)
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

  defp run_full_training(skip_pos, models_path) do
    case Trainer.train_and_save(models_path: models_path) do
      {:ok, stats} ->
        stats =
          if skip_pos do
            stats
          else
            {:ok, pos_stats} = run_pos_training_internal(models_path)
            Map.merge(stats, pos_stats)
          end

        {:ok, stats}

      error ->
        error
    end
  end

  defp run_pos_training(models_path) do
    Mix.shell().info("Training POS tagger model only...")
    run_pos_training_internal(models_path)
  end

  # Trains on the committed EWT fixtures (Brain.Training.POS), which raises
  # rather than saving a model that cannot beat the lookup baseline.
  defp run_pos_training_internal(models_path) do
    Mix.shell().info("  Training the POS tagger on the EWT fixtures...")

    {path, model} =
      Brain.Training.POS.train_and_save!(out: Path.join(models_path, "pos_model.term"))

    Mix.shell().info("  POS model saved to #{path}")

    {:ok,
     %{
       pos_model_trained: true,
       pos_accuracy: model.evaluation.accuracy,
       pos_lookup_baseline: model.evaluation.lookup_baseline
     }}
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

    for {split, path} <- Enum.sort(Brain.Training.POS.fixture_paths()) do
      status = if File.exists?(path), do: Path.relative_to_cwd(path), else: "NOT FOUND (run mix pos.import_ud_ewt)"
      Mix.shell().info("  POS #{String.pad_trailing(split, 6)} #{status}")
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
      Mix.shell().info("    - Test accuracy:    #{Float.round(stats.pos_accuracy * 100, 2)}%")
      Mix.shell().info("    - Lookup baseline:  #{Float.round(stats.pos_lookup_baseline * 100, 2)}%")
    end

    Mix.shell().info("")
    Mix.shell().info("  Total training time: #{format_duration(duration)}")
    Mix.shell().info("")
    Mix.shell().info("Saved Models:")
    Mix.shell().info("=" |> String.duplicate(50))

    display_model_file(models_path, "classifier.term", "Intent Classifier")
    display_model_file(models_path, "entity_model.term", "Entity Model")
    display_model_file(models_path, "pos_model.term", "POS Tagger")
    display_model_file(models_path, "vectorizer.term", "TF-IDF Vectorizer")
    display_model_file(models_path, "embedder.term", "Embedder Vocabulary")

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