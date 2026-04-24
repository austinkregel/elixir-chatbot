defmodule Mix.Tasks.Train do
  @moduledoc """
  Master training task that trains models in dependency order.

  Primary runtime “understanding” for utterances uses `ChunkProfile` (engineered
  features + micro-classifiers), not the old registry/TF-IDF intent GenServer.
  This pipeline trains entity model, gazetteer, embedder vocabulary, speech-act
  TF-IDF, and **all** micro-classifiers including axis models
  (`intent_domain`, `tense_class`, …).

  Use `--list` to print stages. Hyperparameters live in the per-stage trainers below.

  ## Usage

      mix train [options]

  ## Micro-classifier data (axis models)

  The six axis micro-classifiers read training JSON from `data/classifiers/*.json`.
  To **regenerate** those JSON files from `priv/evaluation/intent/gold_standard.json`
  (after you change gold data or heuristics), run **before** `mix train` or
  `mix train_micro`:

      mix gen_micro_data

  Then train micro models (or let stage 5 of `mix train` run `mix train_micro`).

  ## Options

    --quick            Skip slow/optional models
    --skip-tfidf       Skip TF-IDF bundle (entity model, gazetteer, embedder, speech-act TF-IDF)
    --skip-pos         Skip POS tagger training
    --skip-seq2seq     Skip seq2seq generation model
    --skip-poincare    Skip Poincare embeddings
    --skip-kg-lstm     Skip KG triple scorer
    --skip-micro       Skip TF-IDF micro-classifiers (including ChunkProfile axis models)
    --skip-framing     Skip framing classifier (GVFC corpus)
    --include-graph    Run graph-to-training integration after training
    --world ID         Train world-specific models
    --publish          Publish trained models to S3/MinIO
    --list             List all available training tasks

  ## Training order (6 stages)

  1. **TF-IDF bundle** — `Trainer.train_and_save/1`: entity model, gazetteer,
     embedder vocabulary, plus speech-act TF-IDF (~1 minute).
  2. **POS tagger** — `pos_model.term` (~1 second).
  3. **Poincare embeddings** — `poincare/embeddings.term` (~1 min).
  4. **KG triple scorer** — `kg_lstm/triple_scorer.term` (~1 min GPU).
  5. **MicroClassifiers** — runs `mix train_micro`: trains every name in
     `Mix.Tasks.TrainMicro` (pragmatic classifiers + six axis models). Writes
     `priv/ml_models/micro/*.term`. Regenerate JSON with `mix gen_micro_data` when needed.
  6. **Framing Classifier** — runs `mix gen_framing_data` (if JSON missing) then
     `mix train_framing`. Requires GVFC corpus at `data/framing/`. Run
     `mix ingest_framing_corpus` if the CSV is not yet extracted.

  ## Examples

      mix train                    # Full pipeline
      mix gen_micro_data && mix train   # Refresh axis micro data, then full train
      mix train --quick            # Skip slow models
      mix train --skip-tfidf       # Skip TF-IDF bundle stage
      mix train --skip-micro       # Skip micro-classifiers only
      mix train --world star_trek  # Train for a specific world
      mix train --list             # List stages
  """

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias World.Persistence
  alias Brain.ML.POSTagger
  alias Brain.ML.Trainer
  alias Brain.ML.ModelStore
  use Mix.Task
  require Logger

  @shortdoc "Train ALL ML models (master training pipeline)"

  @training_tasks [
    %{
      name: "TF-IDF bundle",
      description:
        "Entity model, gazetteer, embedder vocabulary, speech-act TF-IDF",
      task: :tfidf,
      duration: "~1 minute",
      outputs: [
        "entity_model.term",
        "gazetteer.term",
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
      name: "MicroClassifiers",
      description:
        "Micro-classifiers (pragmatic + ChunkProfile axes: intent_domain, tense_class, etc.); run `mix gen_micro_data` to refresh JSON",
      task: :micro,
      duration: "~30-90 seconds (depends on JSON size)",
      outputs: ["micro/*.term"]
    },
    %{
      name: "Framing Classifier",
      description:
        "Document-level framing classifier (GVFC corpus); run `mix ingest_framing_corpus` then `mix gen_framing_data` to prepare data",
      task: :framing,
      duration: "~2-5 minutes",
      outputs: ["micro/framing_class.term", "micro/framing_neutral_centroid.term"]
    }
  ]

  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [
          quick: :boolean,
          skip_tfidf: :boolean,
          skip_pos: :boolean,
          skip_seq2seq: :boolean,
          skip_poincare: :boolean,
          skip_kg_lstm: :boolean,
          skip_micro: :boolean,
          skip_framing: :boolean,
          include_graph: :boolean,
          world: :string,
          list: :boolean,
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
      if opts[:skip_seq2seq] do
        [:seq2seq | skip_list]
      else
        skip_list
      end

    skip_list =
      if opts[:quick] do
        skip_list
        |> Kernel.++([:seq2seq])
        |> Enum.uniq()
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

    skip_list =
      if opts[:skip_framing] do
        [:framing | skip_list]
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
    Mix.shell().info("  mix train_models    - TF-IDF models")
    Mix.shell().info("  mix train_micro     - Micro-classifiers (feature vector + text)")
    Mix.shell().info("  mix train_poincare  - Poincare embeddings")
    Mix.shell().info("  mix train_kg_lstm   - KG triple scorer")
    Mix.shell().info("  mix train_framing   - Framing classifier (GVFC)")
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
      if :micro in skip_list do
        [{:micro, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_micro_classifiers()
        duration = System.monotonic_time(:second) - start
        [{:micro, result, duration} | results]
      end

    cleanup_exla_callback_servers()

    results =
      if :framing in skip_list do
        [{:framing, :skipped, 0} | results]
      else
        start = System.monotonic_time(:second)
        result = train_framing_classifier()
        duration = System.monotonic_time(:second) - start
        [{:framing, result, duration} | results]
      end

    Enum.reverse(results)
  end

  defp stage_timestamp, do: Calendar.strftime(DateTime.utc_now(), "%H:%M:%S")

  defp train_tfidf_models(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 1/6: TF-IDF Models  [#{stage_timestamp()}]")
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
    Mix.shell().info("  Stage 2/6: POS Tagger  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    models_path = get_models_path(opts[:world])
    gold_standard_path = Brain.priv_path("evaluation/intent/gold_standard.json")

    sequences = load_pos_from_gold_standard(gold_standard_path)

    sequences =
      if sequences != [] do
        Mix.shell().info("  Found #{length(sequences)} pre-annotated POS sequences")
        sequences
      else
        Mix.shell().info("  No POS-annotated data in gold standard. Auto-enriching with WordNet + rules...")
        auto_enrich_pos(gold_standard_path)
      end

    if sequences != [] do
      Mix.shell().info("  Training POS model on #{length(sequences)} sequences...")

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
      Mix.shell().info("  No training data available for POS model.")
      {:ok, %{pos_trained: false}}
    end
  end

  defp train_poincare_embeddings(opts) do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 3/6: Poincare Embeddings  [#{stage_timestamp()}]")
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
    Mix.shell().info("  Stage 4/6: KG Triple Scorer  [#{stage_timestamp()}]")
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

  defp train_micro_classifiers do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 5/6: MicroClassifiers  [#{stage_timestamp()}]")
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

  defp train_framing_classifier do
    Mix.shell().info("")
    Mix.shell().info("=" |> String.duplicate(70))
    Mix.shell().info("  Stage 6/6: Framing Classifier  [#{stage_timestamp()}]")
    Mix.shell().info("=" |> String.duplicate(70))

    data_path = "data/classifiers/framing_class.json"

    gen_result =
      if File.exists?(data_path) do
        :ok
      else
        Mix.shell().info("  Generating framing training data from GVFC corpus...")

        try do
          Mix.Tasks.GenFramingData.run(["--corpus", "gvfc"])
          :ok
        rescue
          e -> {:error, Exception.message(e)}
        end
      end

    case gen_result do
      :ok ->
        try do
          Mix.Tasks.TrainFraming.run([])
          {:ok, %{framing_trained: true}}
        rescue
          e -> {:error, Exception.message(e)}
        catch
          :exit, {:shutdown, 1} -> {:error, "framing training failed"}
        end

      {:error, _} = err ->
        err
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
      poincare: "Poincare Embeddings",
      kg_lstm: "KG Triple Scorer",
      micro: "MicroClassifiers",
      framing: "Framing Classifier"
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

  # Auto-enriches gold standard examples with tokens + POS tags using the
  # Elixir tokenizer and a WordNet-backed rule-based tagger. Replaces the
  # old Python/NLTK dependency (scripts/enrich_gold_standard_pos.py).
  defp auto_enrich_pos(gold_standard_path) do
    case File.read(gold_standard_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, examples} when is_list(examples) ->
            sequences =
              examples
              |> Enum.filter(fn ex -> is_binary(ex["text"]) and ex["text"] != "" end)
              |> Enum.map(fn ex ->
                tokens = Brain.ML.Tokenizer.tokenize_words(ex["text"])
                tags = Enum.map(tokens, &rule_based_pos/1)
                %{tokens: tokens, tags: tags, source: ex["intent"]}
              end)
              |> Enum.filter(fn seq -> seq.tokens != [] end)

            Mix.shell().info("  Auto-enriched #{length(sequences)} examples with rule-based POS tags")
            sequences

          _ ->
            []
        end

      {:error, _} ->
        []
    end
  end

  @determiners ~w(a an the this that these those my your his her its our their some any no every each all both few many much several)
  @prepositions ~w(in on at to for from by with of into onto upon about above below between through during before after since until)
  @conjunctions ~w(and or but nor yet so for because although though while if when unless)
  @pronouns ~w(i me my mine myself you your yours yourself he him his himself she her hers herself it its itself we us our ours ourselves they them their theirs themselves who whom whose which what)
  @auxiliaries ~w(am is are was were be been being have has had do does did will would shall should can could may might must)
  @particles ~w(not to up down out off away back)
  @interjections ~w(oh hey wow oops ah uh um hmm hello hi bye yes no ok okay please thanks)

  defp rule_based_pos(token) do
    lower = String.downcase(token)

    cond do
      String.match?(token, ~r/^\d+(\.\d+)?$/) -> "NUM"
      String.match?(token, ~r/^[[:punct:]]+$/) -> "PUNCT"
      lower in @determiners -> "DET"
      lower in @prepositions -> "ADP"
      lower in @conjunctions -> "CONJ"
      lower in @pronouns -> "PRON"
      lower in @auxiliaries -> "AUX"
      lower in @particles -> "PART"
      lower in @interjections -> "INTJ"
      true -> wordnet_pos_lookup(lower, token)
    end
  end

  defp wordnet_pos_lookup(lower, original) do
    alias Brain.ML.Lexicon, as: WordNet

    case WordNet.senses(lower) do
      [_ | _] = senses ->
        best =
          senses
          |> Enum.group_by(& &1.pos)
          |> Enum.max_by(fn {_pos, group} -> Enum.sum(Enum.map(group, & &1.tag_count)) end)
          |> elem(0)

        wordnet_to_universal(best)

      [] ->
        guess_pos_from_shape(lower, original)
    end
  rescue
    _ -> guess_pos_from_shape(lower, original)
  end

  defp wordnet_to_universal(:n), do: "NOUN"
  defp wordnet_to_universal(:v), do: "VERB"
  defp wordnet_to_universal(:a), do: "ADJ"
  defp wordnet_to_universal(:s), do: "ADJ"
  defp wordnet_to_universal(:r), do: "ADV"
  defp wordnet_to_universal(_), do: "NOUN"

  defp guess_pos_from_shape(lower, original) do
    cond do
      original == String.upcase(original) and String.length(original) > 1 -> "PROPN"
      String.match?(original, ~r/^[A-Z]/) -> "PROPN"
      String.ends_with?(lower, "ly") -> "ADV"
      String.ends_with?(lower, "ing") -> "VERB"
      String.ends_with?(lower, "ed") -> "VERB"
      String.ends_with?(lower, "tion") or String.ends_with?(lower, "ness") -> "NOUN"
      String.ends_with?(lower, "able") or String.ends_with?(lower, "ible") -> "ADJ"
      String.ends_with?(lower, "ous") or String.ends_with?(lower, "ful") -> "ADJ"
      String.ends_with?(lower, "er") or String.ends_with?(lower, "est") -> "ADJ"
      true -> "NOUN"
    end
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
