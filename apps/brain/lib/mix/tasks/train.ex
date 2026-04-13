defmodule Mix.Tasks.Train do
  @moduledoc """
  Master training task that trains ALL models in the correct order.

  Each model has its own tuned hyperparameters (epochs, batch size, learning rate)
  set in the training functions below. Use `--list` to see all stages.

  ## Usage

      mix train [options]

  ## Options

    --quick            Skip slow/optional models (seq2seq, response scorer)
    --skip-tfidf       Skip TF-IDF models (intent classifier, entity model, gazetteer)
    --skip-lstm        Skip all LSTM models (also skips arbitrator)
    --skip-pos         Skip POS tagger training
    --skip-unified     Skip unified multi-task LSTM model
    --skip-response    Skip response scorer model
    --skip-seq2seq     Skip seq2seq generation model
    --skip-gcn         Skip GCN text classifier
    --skip-poincare    Skip Poincare embeddings
    --skip-kg-lstm     Skip KG triple scorer
    --skip-arbitrator  Skip intent arbitrator meta-learner
    --skip-micro       Skip micro-classifiers
    --include-graph    Run graph-to-training integration after training
    --world ID         Train world-specific models
    --name NAME        Experiment name for tracking (default: train_YYYYMMDD_HHMMSS)
    --compare          Print experiment comparison table after training
    --publish          Publish trained models to S3/MinIO
    --list             List all available training tasks

  ## Training Order (9 stages)

  1. TF-IDF Models + Speech Act TF-IDF (~1 minute)
  2. POS Tagger (~1 second)
  3. Unified LSTM - intent, sentiment, speech act (~17 min GPU)
  4. Response Scorer (~1-2 min GPU)
  5. GCN Text Classifier (~5 min GPU)
  6. Poincare Embeddings (~1 min)
  7. KG Triple Scorer (~1 min GPU)
  8. Intent Arbitrator (~30 sec)
  9. MicroClassifiers (~10 sec)

  ## Examples

      mix train                    # Train everything
      mix train --quick            # Skip slow models
      mix train --skip-tfidf       # Skip TF-IDF stage
      mix train --world star_trek  # Train for a specific world
      mix train --list             # List all training tasks
  """

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  alias Brain.Response.LSTMResponse
  alias Brain.ML.LSTM.UnifiedModel
  alias Brain.ML.POSTagger
  alias Brain.ML.Trainer
  alias Brain.ML.ModelStore
  use Mix.Task
  require Logger

  alias Brain.ML.LSTM.ExperimentTracker

  @shortdoc "Train ALL ML models (master training pipeline)"

  @training_tasks [
    %{
      name: "TF-IDF Models",
      description: "Intent classifier, entity recognition, gazetteer, embedder, speech act",
      task: :tfidf,
      duration: "~1 minute",
      outputs: [
        "classifier.term",
        "entity_model.term",
        "gazetteer.term",
        "vectorizer.term",
        "embedder.term",
        "speech_act_classifier.term"
      ]
    },
    %{
      name: "POS Tagger",
      description: "Part-of-speech tagging model",
      task: :pos,
      duration: "~1 second",
      outputs: ["pos_model.term"]
    },
    %{
      name: "Unified LSTM",
      description: "Multi-task model: intent, NER, sentiment, speech acts",
      task: :unified,
      duration: "~17 minutes (GPU, includes JIT compilation)",
      outputs: ["lstm/unified_model.term"]
    },
    %{
      name: "Response Scorer",
      description: "Query-response quality scoring model",
      task: :response,
      duration: "~1 minute (GPU)",
      outputs: ["lstm/response_scorer.term"]
    },
    %{
      name: "GCN Text Classifier",
      description: "Graph convolutional network for intent classification",
      task: :gcn,
      duration: "~5 minutes (GPU, includes JIT compilation)",
      outputs: ["gcn/model.term"]
    },
    %{
      name: "Poincare Embeddings",
      description: "Hyperbolic embeddings for entity type hierarchy",
      task: :poincare,
      duration: "~1 minute",
      outputs: ["poincare/embeddings.term"]
    },
    %{
      name: "KG Triple Scorer",
      description: "Knowledge graph triple validity scoring via BiLSTM",
      task: :kg_lstm,
      duration: "~1 minute (GPU)",
      outputs: ["kg_lstm/triple_scorer.term"]
    },
    %{
      name: "Intent Arbitrator",
      description: "Stacked meta-learner for LSTM vs TF-IDF intent arbitration",
      task: :arbitrator,
      duration: "~30 seconds",
      outputs: ["intent_arbitrator.term"]
    },
    %{
      name: "MicroClassifiers",
      description: "TF-IDF micro-classifiers for lightweight decisions",
      task: :micro,
      duration: "~10 seconds",
      outputs: ["micro/*.term"]
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
          skip_gcn: :boolean,
          skip_poincare: :boolean,
          skip_kg_lstm: :boolean,
          skip_arbitrator: :boolean,
          skip_micro: :boolean,
          include_graph: :boolean,
          world: :string,
          list: :boolean,
          name: :string,
          compare: :boolean,
          publish: :boolean
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
    Mix.shell().info("  MASTER TRAINING PIPELINE  [#{stage_timestamp()}]")
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

    cleaner_pid = start_exla_cleanup_loop()

    start_time = System.monotonic_time(:second)
    results = run_training_pipeline(opts, skip_list)

    stop_exla_cleanup_loop(cleaner_pid)

    if opts[:include_graph] do
      Mix.shell().info("")
      Mix.shell().info("  Running graph-to-training integration...")
      Mix.Tasks.TrainFromGraph.run([])
    end

    total_duration = System.monotonic_time(:second) - start_time

    if opts[:publish] || ModelStore.enabled?() do
      publish_models(get_models_path(opts[:world]))
    end

    display_summary(results, total_duration)
    experiment_name = opts[:name] || generate_experiment_name("train")

    ExperimentTracker.record(%{
      name: experiment_name,
      config: %{
        unified: %{epochs: 80, batch_size: 32, lr: 3.0e-4, label_smoothing: 0.1},
        response: %{epochs: 30, batch_size: 64, lr: 0.001},
        hidden_size: 128
      },
      epochs_completed: 80,
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
        |> Kernel.++([:unified, :response, :seq2seq, :arbitrator])
        |> Enum.uniq()
      else
        skip_list
      end

    skip_list =
      if opts[:skip_arbitrator] do
        [:arbitrator | skip_list] |> Enum.uniq()
      else
        skip_list
      end

    skip_list =
      if opts[:skip_gcn] do
        [:gcn | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_poincare] do
        [:poincare | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_kg_lstm] do
        [:kg_lstm | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:skip_micro] do
        [:micro | skip_list]
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
    Mix.shell().info("  mix train_lstm      - Standalone LSTM intent classifier")
    Mix.shell().info("  mix train_micro     - TF-IDF micro-classifiers")
    Mix.shell().info("  mix train_gcn       - GCN text classifier")
    Mix.shell().info("  mix train_poincare  - Poincare embeddings")
    Mix.shell().info("  mix train_kg_lstm   - KG triple scorer")
    Mix.shell().info("  mix train_arbitrator - Intent arbitrator")
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
    models_path = get_models_path(world_id)

    results = []

    results =
      if :tfidf in skip_list do
        [{:tfidf, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_tfidf_models(opts)
        duration = System.monotonic_time(:second) - start
        [{:tfidf, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :pos in skip_list do
        [{:pos, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_pos_model(opts)
        duration = System.monotonic_time(:second) - start
        [{:pos, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :unified in skip_list do
        [{:unified, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_unified_lstm(models_path)
        duration = System.monotonic_time(:second) - start
        [{:unified, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :response in skip_list do
        [{:response, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_response_scorer(models_path)
        duration = System.monotonic_time(:second) - start
        [{:response, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :gcn in skip_list do
        [{:gcn, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_gcn_model(opts)
        duration = System.monotonic_time(:second) - start
        [{:gcn, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :poincare in skip_list do
        [{:poincare, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_poincare_embeddings(opts)
        duration = System.monotonic_time(:second) - start
        [{:poincare, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :kg_lstm in skip_list do
        [{:kg_lstm, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_kg_triple_scorer(opts)
        duration = System.monotonic_time(:second) - start
        [{:kg_lstm, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :arbitrator in skip_list do
        [{:arbitrator, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_arbitrator()
        duration = System.monotonic_time(:second) - start
        [{:arbitrator, result, duration} | results]
      end

    results =
      if :micro in skip_list do
        [{:micro, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_micro_classifiers()
        duration = System.monotonic_time(:second) - start
        [{:micro, result, duration} | results]
      end

    Enum.reverse(results)
  end

  defp stage_timestamp, do: Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")

  defp train_tfidf_models(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 1/9: TF-IDF Models  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    models_path = get_models_path(opts[:world])

    case Trainer.train_and_save(models_path: models_path) do
      {:ok, stats} ->
        Mix.shell().info("  TF-IDF training complete!")
        Mix.shell().info("    Intent samples: #{stats.intent_samples}")
        Mix.shell().info("    Vocabulary size: #{stats.vocab_size}")

        train_speech_act_tfidf(models_path)

        {:ok, stats}

      {:error, reason} ->
        Mix.shell().error("  TF-IDF training failed: #{inspect(reason)}")
        {:error, reason}
    end
  end

  defp train_speech_act_tfidf(models_path) do
    gold = Brain.ML.EvaluationStore.load_gold_standard("speech_act")

    training_data =
      gold
      |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["speech_act"]) end)
      |> Enum.map(fn ex -> {ex["text"], ex["speech_act"]} end)

    if training_data != [] do
      Mix.shell().info("  Training speech act TF-IDF classifier (#{length(training_data)} examples)...")
      model = Brain.ML.SimpleClassifier.train(training_data)
      save_path = Path.join(models_path, "speech_act_classifier.term")
      File.mkdir_p!(Path.dirname(save_path))
      binary = :erlang.term_to_binary(model, [:compressed])
      File.write!(save_path, binary)
      Mix.shell().info("  Speech act TF-IDF classifier saved to #{save_path}")
    else
      Mix.shell().info("  No speech act gold standard data. Skipping speech act TF-IDF training.")
    end
  end

  defp train_pos_model(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 2/9: POS Tagger  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    models_path = get_models_path(opts[:world])
    gold_standard_path = Brain.priv_path("evaluation/intent/gold_standard.json")

    sequences = load_pos_from_gold_standard(gold_standard_path)

    if sequences != [] do
      Mix.shell().info("  Found #{length(sequences)} POS-annotated sequences from gold standard")

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
      Mix.shell().info("  No POS-annotated data in gold standard. Skipping POS training.")
      Mix.shell().info("  Run: python scripts/enrich_gold_standard_pos.py")
      {:ok, %{pos_trained: false}}
    end
  end

  defp train_unified_lstm(models_path) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 3/9: Unified LSTM Model (GPU Accelerated)  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    Mix.shell().info("  Training multi-task model for:")
    Mix.shell().info("    - Intent Classification")
    Mix.shell().info("    - Named Entity Recognition")
    Mix.shell().info("    - Sentiment Analysis")
    Mix.shell().info("    - Speech Act Classification")
    Mix.shell().info("")

    config = [
      epochs: 80,
      batch_size: 32,
      hidden_size: 128,
      embedding_size: 128,
      learning_rate: 3.0e-4,
      sentiment_epochs: 60,
      speech_act_epochs: 120,
      speech_act_batch_size: 16,
      label_smoothing: 0.1,
      dropout: 0.2,
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

  defp train_response_scorer(models_path) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 4/9: Response Scorer (GPU Accelerated)  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")
    Mix.shell().info("  Training query-response scoring model...")
    Mix.shell().info("")

    config = [
      epochs: 30,
      batch_size: 64,
      hidden_size: 128,
      embedding_size: 128,
      learning_rate: 0.001,
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

  defp train_gcn_model(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 5/9: GCN Text Classifier  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    world = if opts[:world], do: ["--world", opts[:world]], else: []
    args = world ++ ["--verbose"]

    try do
      Mix.Tasks.TrainGcn.run(args)
      {:ok, %{gcn_trained: true}}
    rescue
      e ->
        Mix.shell().error("  GCN training failed: #{Exception.message(e)}")
        {:error, Exception.message(e)}
    catch
      :exit, {:shutdown, 1} -> {:error, "no training data"}
    end
  end

  defp train_poincare_embeddings(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 6/9: Poincare Embeddings  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    world = if opts[:world], do: ["--world", opts[:world]], else: []
    args = world ++ ["--verbose"]

    try do
      Mix.Tasks.TrainPoincare.run(args)
      {:ok, %{poincare_trained: true}}
    rescue
      e -> {:error, Exception.message(e)}
    catch
      :exit, {:shutdown, 1} -> {:error, "no hierarchy data"}
    end
  end

  defp train_kg_triple_scorer(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 7/9: KG Triple Scorer  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    world = if opts[:world], do: ["--world", opts[:world]], else: []
    args = world ++ ["--verbose"]

    try do
      Mix.Tasks.TrainKgLstm.run(args)
      {:ok, %{kg_lstm_trained: true}}
    rescue
      e -> {:error, Exception.message(e)}
    catch
      :exit, {:shutdown, 1} -> {:error, "no triples found"}
    end
  end

  defp train_arbitrator do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 8/9: Intent Arbitrator  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    try do
      Mix.Tasks.TrainArbitrator.run([])
      {:ok, %{arbitrator_trained: true}}
    rescue
      e -> {:error, Exception.message(e)}
    catch
      :exit, {:shutdown, 1} -> {:error, "insufficient training data"}
    end
  end

  defp train_micro_classifiers do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 9/9: MicroClassifiers  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    try do
      Mix.Tasks.TrainMicro.run([])
      {:ok, %{micro_trained: true}}
    rescue
      e -> {:error, Exception.message(e)}
    catch
      :exit, {:shutdown, 1} -> {:error, "no classifier data"}
    end
  end

  defp display_summary(results, total_duration) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  TRAINING SUMMARY  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("")

    task_names = %{
      tfidf: "TF-IDF Models",
      pos: "POS Tagger",
      unified: "Unified LSTM",
      response: "Response Scorer",
      gcn: "GCN Text Classifier",
      poincare: "Poincare Embeddings",
      kg_lstm: "KG Triple Scorer",
      arbitrator: "Intent Arbitrator",
      micro: "MicroClassifiers"
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

  defp publish_models(models_path) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Publishing models to S3/MinIO")
    Mix.shell().info("=" |> String.duplicate(70))

    version = ModelStore.version_prefix()
    Mix.shell().info("  Version: #{version}")

    term_files =
      Path.join(models_path, "**/*.term")
      |> Path.wildcard()

    if term_files == [] do
      Mix.shell().info("  No .term files found in #{models_path}")
    else
      results =
        Enum.map(term_files, fn file ->
          relative = Path.relative_to(file, models_path)
          remote_key = version <> relative

          case ModelStore.publish(file, remote_key) do
            {:ok, _} -> {:ok, relative}
            {:error, reason} -> {:error, relative, reason}
            :disabled -> {:disabled, relative}
          end
        end)

      published = Enum.count(results, &match?({:ok, _}, &1))
      failed = Enum.count(results, &match?({:error, _, _}, &1))

      ModelStore.set_latest(version)

      Mix.shell().info("  Published #{published} models, #{failed} failed")
    end
  end

  defp get_models_path(nil) do
    Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
  end

  defp get_models_path(world_id) do
    world_path = Persistence.world_path(world_id)
    Path.join(world_path, "models")
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

  # EXLA creates a new CallbackServer process for every JIT call, even on
  # compilation cache hits. These orphan processes accumulate because they
  # only terminate when the native XLA executable is garbage collected.
  # During long training runs this exhausts the BEAM process table.

  @exla_cleanup_interval_ms 30_000

  defp start_exla_cleanup_loop do
    parent = self()

    spawn_link(fn ->
      exla_cleanup_loop(parent)
    end)
  end

  defp exla_cleanup_loop(parent) do
    Process.sleep(@exla_cleanup_interval_ms)

    if Process.alive?(parent) do
      do_cleanup_exla_callback_servers()
      exla_cleanup_loop(parent)
    end
  end

  defp stop_exla_cleanup_loop(pid) do
    Process.unlink(pid)
    Process.exit(pid, :shutdown)
  end

  defp cleanup_exla_callback_servers do
    :erlang.garbage_collect()
    do_cleanup_exla_callback_servers()
  end

  defp do_cleanup_exla_callback_servers do
    supervisor = Process.whereis(EXLA.CallbackServer.Supervisor)

    if supervisor && Process.alive?(supervisor) do
      children = DynamicSupervisor.which_children(supervisor)
      count = length(children)

      if count > 100 do
        Mix.shell().info(
          "  [cleanup] Reclaiming #{count} orphan EXLA callback servers " <>
            "(BEAM process count: #{:erlang.system_info(:process_count)})"
        )

        for {_, pid, _, _} <- children, is_pid(pid) do
          DynamicSupervisor.terminate_child(supervisor, pid)
        end
      end
    end
  rescue
    _ -> :ok
  end
end
