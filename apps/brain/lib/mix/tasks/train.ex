defmodule Mix.Tasks.Train do
  @moduledoc "Master training task that trains ALL models in the correct order.\n\n## Usage\n\n    mix train [options]\n\n## Options\n\n  --quick          Skip slow/optional models (seq2seq, response scorer)\n  --skip-tfidf     Skip TF-IDF models (intent classifier, entity model, gazetteer)\n  --skip-lstm      Skip all LSTM models\n  --skip-pos       Skip POS tagger training\n  --skip-unified   Skip unified multi-task LSTM model\n  --skip-response  Skip response scorer model\n  --skip-seq2seq   Skip seq2seq generation model\n  --epochs N       Default epochs for LSTM training (default: 20)\n  --batch-size N   Default batch size (default: 32)\n  --world ID       Train world-specific models\n  --name NAME      Experiment name for tracking (default: train_YYYYMMDD_HHMMSS)\n  --compare        Print experiment comparison table after training\n  --list           List all available training tasks\n\n## Training Order\n\nModels are trained in dependency order:\n\n1. **TF-IDF Models** (fast, ~30 seconds)\n   - Intent Classifier (classifier.term)\n   - Entity Recognition (entity_model.term)\n   - Gazetteer (gazetteer.term)\n   - TF-IDF Vectorizer (vectorizer.term)\n   - Embedder Vocabulary (embedder.term)\n\n2. **POS Tagger** (fast, ~10 seconds)\n   - Part-of-speech model (pos_model.term)\n\n3. **LSTM Unified Model** (GPU accelerated, ~2-5 minutes)\n   - Intent classification\n   - Named Entity Recognition\n   - Sentiment analysis\n   - Speech act classification\n\n4. **Response Scorer** (GPU accelerated, ~1-2 minutes)\n   - Query-response pair scoring\n\n## Examples\n\n    # Train everything\n    mix train\n\n    # Quick training (skip slow models)\n    mix train --quick\n\n    # Train only LSTM models with more epochs\n    mix train --skip-tfidf --epochs 30\n\n    # Train for a specific world\n    mix train --world star_trek\n\n    # List all training tasks\n    mix train --list\n"

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  alias Brain.Response.LSTMResponse
  alias Brain.ML.LSTM.UnifiedModel
  alias Brain.ML.POSTagger
  alias Brain.ML.Trainer
  use Mix.Task
  require Logger

  alias Brain.ML.LSTM.ExperimentTracker

  @shortdoc "Train ALL ML models (master training pipeline)"

  @training_tasks [
    %{
      name: "TF-IDF Models",
      description: "Intent classifier, entity recognition, gazetteer, embedder",
      task: :tfidf,
      duration: "~30 seconds",
      outputs: [
        "classifier.term",
        "entity_model.term",
        "gazetteer.term",
        "vectorizer.term",
        "embedder.term"
      ]
    },
    %{
      name: "POS Tagger",
      description: "Part-of-speech tagging model",
      task: :pos,
      duration: "~10 seconds",
      outputs: ["pos_model.term"]
    },
    %{
      name: "Unified LSTM",
      description: "Multi-task model: intent, NER, sentiment, speech acts",
      task: :unified,
      duration: "~2-5 minutes (GPU)",
      outputs: ["lstm/unified_model.term"]
    },
    %{
      name: "Response Scorer",
      description: "Query-response quality scoring model",
      task: :response,
      duration: "~1-2 minutes (GPU)",
      outputs: ["lstm/response_scorer.term"]
    }
  ]

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          quick: :boolean,
          skip_tfidf: :boolean,
          skip_lstm: :boolean,
          skip_pos: :boolean,
          skip_unified: :boolean,
          skip_response: :boolean,
          skip_seq2seq: :boolean,
          epochs: :integer,
          batch_size: :integer,
          hidden_size: :integer,
          world: :string,
          list: :boolean,
          name: :string,
          compare: :boolean
        ]
      )

    if opts[:list] do
      display_training_tasks()
      return_ok()
    else
      run_training(opts)
    end
  end

  defp return_ok do
    :ok
  end

  defp run_training(opts) do
    Application.put_env(:brain, :skip_ml_init, true)
    Mix.Task.run("app.start")

    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  MASTER TRAINING PIPELINE")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    skip_list = build_skip_list(opts)
    display_training_plan(skip_list)

    if length(skip_list) < length(@training_tasks) do
      Mix.shell().info("")
      Mix.shell().info("Starting training in 3 seconds... (Ctrl+C to cancel)")
      Process.sleep(3000)
    else
      Mix.shell().info("")
      Mix.shell().info("All training tasks are skipped. Nothing to do.")
      System.halt(0)
    end

    start_time = System.monotonic_time(:second)
    results = run_training_pipeline(opts, skip_list)
    total_duration = System.monotonic_time(:second) - start_time
    display_summary(results, total_duration)
    experiment_name = opts[:name] || generate_experiment_name("train")

    ExperimentTracker.record(%{
      name: experiment_name,
      config: %{
        epochs: opts[:epochs] || 20,
        batch_size: opts[:batch_size] || 32,
        hidden_size: opts[:hidden_size] || 128
      },
      epochs_completed: opts[:epochs] || 20,
      training_time_seconds: total_duration,
      notes: "Master pipeline. Tasks: #{summarize_results(results)}"
    })

    Mix.shell().info("  Experiment recorded: #{experiment_name}")

    if opts[:compare] do
      Mix.shell().info("")
      ExperimentTracker.print_comparison()
    end
  end

  defp build_skip_list(opts) do
    skip_list = []

    skip_list =
      if opts[:skip_tfidf] do
        [:tfidf | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_pos] do
        [:pos | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_unified] do
        [:unified | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_response] do
        [:response | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_seq2seq] do
        [:seq2seq | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:quick] do
        skip_list
        |> Kernel.++([:response, :seq2seq])
        |> Enum.uniq()
      else
        skip_list
      end

    skip_list =
      if opts[:skip_lstm] do
        skip_list
        |> Kernel.++([:unified, :response, :seq2seq])
        |> Enum.uniq()
      else
        skip_list
      end

    skip_list
  end

  defp display_training_tasks do
    Mix.shell().info("")
    Mix.shell().info("Available Training Tasks:")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")

    for task <- @training_tasks do
      Mix.shell().info("  #{task.name}")
      Mix.shell().info("    Description: #{task.description}")
      Mix.shell().info("    Duration:    #{task.duration}")
      Mix.shell().info("    Outputs:     #{Enum.join(task.outputs, ", ")}")
      Mix.shell().info("    Skip flag:   --skip-#{task.task}")
      Mix.shell().info("")
    end

    Mix.shell().info("Individual training tasks:")
    Mix.shell().info("  mix train_models    - TF-IDF + optional LSTM")
    Mix.shell().info("  mix train_unified   - Unified multi-task LSTM")
    Mix.shell().info("  mix train_response  - Response quality scorer")
    Mix.shell().info("  mix train_seq2seq   - Seq2seq generation model")
    Mix.shell().info("  mix train_lstm      - Standalone LSTM intent classifier")
    Mix.shell().info("")
  end

  defp display_training_plan(skip_list) do
    Mix.shell().info("Training Plan:")
    Mix.shell().info("-" |> String.duplicate(70))

    for task <- @training_tasks do
      status =
        if task.task in skip_list do
          "[SKIP]"
        else
          "[TRAIN]"
        end

      color =
        if task.task in skip_list do
          :yellow
        else
          :green
        end

      message = "  #{status} #{task.name} (#{task.duration})"

      if color == :green do
        Mix.shell().info(IO.ANSI.green() <> message <> IO.ANSI.reset())
      else
        Mix.shell().info(IO.ANSI.yellow() <> message <> IO.ANSI.reset())
      end
    end
  end

  defp run_training_pipeline(opts, skip_list) do
    world_id = opts[:world]
    epochs = opts[:epochs] || 20
    batch_size = opts[:batch_size] || 32
    hidden_size = opts[:hidden_size] || 128
    models_path = get_models_path(world_id)

    results = []

    results =
      if :tfidf in skip_list do
        [{:tfidf, :skipped, 0} | results]
      else
        result = train_tfidf_models(opts)
        [{:tfidf, result, 0} | results]
      end

    results =
      if :pos in skip_list do
        [{:pos, :skipped, 0} | results]
      else
        result = train_pos_model(opts)
        [{:pos, result, 0} | results]
      end

    results =
      if :unified in skip_list do
        [{:unified, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_unified_lstm(epochs, batch_size, hidden_size, models_path)
        duration = System.monotonic_time(:second) - start
        [{:unified, result, duration} | results]
      end

    results =
      if :response in skip_list do
        [{:response, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_response_scorer(epochs, batch_size, hidden_size, models_path)
        duration = System.monotonic_time(:second) - start
        [{:response, result, duration} | results]
      end

    Enum.reverse(results)
  end

  defp train_tfidf_models(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 1/4: TF-IDF Models")
    Mix.shell().info("=" |> String.duplicate(70))

    models_path = get_models_path(opts[:world])

    case Trainer.train_and_save(models_path: models_path) do
      {:ok, stats} ->
        Mix.shell().info("  TF-IDF training complete!")
        Mix.shell().info("    Intent samples: #{stats.intent_samples}")
        Mix.shell().info("    Vocabulary size: #{stats.vocab_size}")
        {:ok, stats}

      {:error, reason} ->
        Mix.shell().error("  TF-IDF training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp train_pos_model(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 2/4: POS Tagger")
    Mix.shell().info("=" |> String.duplicate(70))

    models_path = get_models_path(opts[:world])
    training_dir = "data/training/intents"

    if File.exists?(training_dir) do
      sequences = load_pos_from_enriched_intents(training_dir)

      if sequences != [] do
        Mix.shell().info("  Found #{length(sequences)} POS-annotated sequences")

        case POSTagger.train(sequences) do
          {:ok, model} ->
            save_path = Path.join(models_path, "pos_model.term")
            File.mkdir_p!(Path.dirname(save_path))

            case POSTagger.save_model(model, save_path) do
              {:ok, path} ->
                Mix.shell().info("  POS model saved to #{path}")
                {:ok, %{pos_trained: true, tag_count: map_size(model.tag_vocabulary)}}

              {:error, reason} ->
                {:error, reason}
            end

          {:error, reason} ->
            {:error, reason}
        end
      else
        Mix.shell().info("  No POS training data found. Skipping.")
        {:ok, %{pos_trained: false}}
      end
    else
      Mix.shell().info("  No enriched training data found. Skipping POS training.")
      {:ok, %{pos_trained: false}}
    end
  end

  defp train_unified_lstm(epochs, batch_size, hidden_size, models_path) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 3/4: Unified LSTM Model (GPU Accelerated)")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    Mix.shell().info("  Training multi-task model for:")
    Mix.shell().info("    - Intent Classification")
    Mix.shell().info("    - Named Entity Recognition")
    Mix.shell().info("    - Sentiment Analysis")
    Mix.shell().info("    - Speech Act Classification")
    Mix.shell().info("")

    config = [
      epochs: epochs,
      batch_size: batch_size,
      hidden_size: hidden_size,
      embedding_size: hidden_size,
      learning_rate: 0.001,
      models_path: models_path
    ]

    case UnifiedModel.train(config) do
      {:ok, result} ->
        Mix.shell().info("  Unified LSTM training complete!")
        Mix.shell().info("    Vocabulary size: #{map_size(result.vocabularies.token_vocab)}")
        Mix.shell().info("    Intent classes: #{map_size(result.vocabularies.intent_to_idx)}")
        {:ok, result}

      {:error, reason} ->
        Mix.shell().error("  Unified LSTM training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp train_response_scorer(epochs, batch_size, hidden_size, models_path) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 4/4: Response Scorer (GPU Accelerated)")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    Mix.shell().info("  Training query-response scoring model...")
    Mix.shell().info("")

    config = [
      epochs: min(epochs, 15),
      batch_size: batch_size,
      hidden_size: hidden_size,
      models_path: models_path
    ]

    case LSTMResponse.train(config) do
      {:ok, result} ->
        Mix.shell().info("  Response scorer training complete!")
        {:ok, result}

      {:error, reason} ->
        Mix.shell().error("  Response scorer training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp display_summary(results, total_duration) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  TRAINING SUMMARY")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")

    task_names = %{
      tfidf: "TF-IDF Models",
      pos: "POS Tagger",
      unified: "Unified LSTM",
      response: "Response Scorer"
    }

    for {task, result, duration} <- results do
      name = task_names[task]

      case result do
        :skipped ->
          Mix.shell().info("  #{name}: " <> IO.ANSI.yellow() <> "SKIPPED" <> IO.ANSI.reset())

        {:ok, _} ->
          duration_str =
            if duration > 0 do
              " (#{duration}s)"
            else
              ""
            end

          Mix.shell().info(
            "  #{name}: " <> IO.ANSI.green() <> "OK#{duration_str}" <> IO.ANSI.reset()
          )

        {:error, reason} ->
          Mix.shell().info(
            "  #{name}: " <> IO.ANSI.red() <> "FAILED - #{inspect(reason)}" <> IO.ANSI.reset()
          )
      end
    end

    success_count = Enum.count(results, fn {_, r, _} -> match?({:ok, _}, r) end)
    skip_count = Enum.count(results, fn {_, r, _} -> r == :skipped end)
    fail_count = Enum.count(results, fn {_, r, _} -> match?({:error, _}, r) end)

    Mix.shell().info("")
    Mix.shell().info("  Total time: #{format_duration(total_duration)}")

    Mix.shell().info(
      "  Results: #{success_count} succeeded, #{skip_count} skipped, #{fail_count} failed"
    )

    Mix.shell().info("")
    models_path = Brain.priv_path("ml_models")
    Mix.shell().info("  Models saved to: #{models_path}")
    Mix.shell().info("")

    if fail_count > 0 do
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

  defp load_pos_from_enriched_intents(training_dir) do
    training_dir
    |> Path.join("*.json")
    |> Path.wildcard()
    |> Enum.flat_map(fn file ->
      case File.read(file) do
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
              []
          end

        _ ->
          []
      end
    end)
  end

  defp generate_experiment_name(prefix) do
    now = NaiveDateTime.utc_now()

    ts =
      now
      |> NaiveDateTime.to_iso8601()
      |> String.slice(0, 19)
      |> String.replace("-", "")
      |> String.replace("T", "_")
      |> String.replace(":", "")

    "#{prefix}_#{ts}"
  end

  defp summarize_results(results) do
    results
    |> Enum.map_join(
      ", ",
      fn
        {task, :skipped, _} -> "#{task}:skipped"
        {task, {:ok, _}, _} -> "#{task}:ok"
        {task, {:error, _}, _} -> "#{task}:failed"
      end
    )
  end

  defp format_duration(seconds) when seconds < 60 do
    "#{seconds} seconds"
  end

  defp format_duration(seconds) when seconds < 3600 do
    minutes = div(seconds, 60)
    secs = rem(seconds, 60)
    "#{minutes}m #{secs}s"
  end

  defp format_duration(seconds) do
    hours = div(seconds, 3600)
    minutes = div(rem(seconds, 3600), 60)
    "#{hours}h #{minutes}m"
  end
end