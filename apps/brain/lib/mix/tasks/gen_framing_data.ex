defmodule Mix.Tasks.GenFramingData do
  @moduledoc """
  Generate training data for the `:framing_class` document-level classifier.

  Loads a published labeled framing corpus, runs each article through the
  chunk feature-vector pipeline, aggregates via `DocumentProfile`, and writes
  `data/classifiers/framing_class.json` with records of the form:

      %{"feature_vector" => [float], "label" => string}

  ## Supported corpora

  - `--corpus gvfc` — Gun Violence Frame Corpus (Liu et al. 2019). Expects the
    CSV file at `data/framing/GVFC_extension_multimodal.csv`. Download the
    GVFC zip from <https://github.com/ganggit/GVFC-raw-corpus> (Google Drive
    link), unzip it, and place the CSV in `data/framing/`. ~1,300 articles,
    9 frame labels.

  - `--corpus mfc` — Media Frames Corpus (Card et al. 2015). Expects the
    JSON file at the path given by `--mfc-path`. ~12,000 articles, 15 frames.
    (Phase 5 — not yet implemented.)

  ## Workflow

      # 1. Place GVFC_extension_multimodal.csv in data/framing/
      #    (or run `mix ingest_framing_corpus` if GVFC.zip is in project root)
      # 2. Generate training data
      mix gen_framing_data --corpus gvfc

      # 3. Train the classifier
      mix train_framing

  ## Options

      --corpus NAME        Corpus to load: `gvfc` (default) or `mfc`
      --mfc-path PATH      Path to the MFC JSON file (only for `--corpus mfc`)
      --stats              Print label histograms without writing files
      --max N              Limit number of articles processed (for dev/test)
      --full-pipeline      Use the full Pipeline.analyze_chunk (slower, richer features)
  """

  use Mix.Task
  require Logger

  alias Brain.Analysis.{
    FeatureExtractor,
    DocumentProfile,
    ChunkProfile,
    DiscourseAnalyzer,
    SemanticRoleLabeler,
    SlotDetector
  }

  alias Brain.ML.{Tokenizer, POSTagger, SpeechActClassifierSimple, SentimentClassifierSimple, EntityExtractor}
  alias Brain.Analysis.Pipeline, as: FullPipeline

  @shortdoc "Generate training data for :framing_class from a labeled corpus"

  NimbleCSV.define(GVFCParser, separator: ",", escape: "\"")

  @gvfc_default_path "data/framing/GVFC_extension_multimodal.csv"
  @output_path "data/classifiers/framing_class.json"

  @speech_act_fallback %{
    category: :unknown, sub_type: :unknown, is_question: false,
    is_imperative: false, indicators: [], confidence: 0.0
  }

  # Q3 Theme1 numeric codes from the GVFC codebook
  @gvfc_theme_codes %{
    "1" => "economic",
    "2" => "gun_control",
    "3" => "gun_rights",
    "4" => "mental_health",
    "5" => "public_safety",
    "6" => "race_ethnicity",
    "7" => "politics",
    "8" => "public_opinion",
    "9" => "society_culture"
  }

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          corpus: :string,
          mfc_path: :string,
          stats: :boolean,
          max: :integer,
          full_pipeline: :boolean
        ]
      )

    corpus = opts[:corpus] || "gvfc"

    case corpus do
      "gvfc" -> run_gvfc(opts)
      "mfc" -> run_mfc(opts)
      other -> Mix.raise("Unknown corpus: #{other}. Use --corpus gvfc or --corpus mfc")
    end
  end

  # -- GVFC loader -------------------------------------------------------

  defp run_gvfc(opts) do
    Mix.Task.run("app.start")
    wait_for_pipeline_ready()

    path = @gvfc_default_path

    unless File.exists?(path) do
      Mix.raise("""
      GVFC CSV not found at #{path}.

      To obtain the data:
        1. Download GVFC.zip from https://github.com/ganggit/GVFC-raw-corpus
           (the repo links to a Google Drive download)
        2. Place GVFC.zip in the project root
        3. Run: mix ingest_framing_corpus

      Or manually:
        unzip GVFC.zip -d data/framing/
        mv data/framing/GVFC/GVFC_extension_multimodal.csv #{path}
      """)
    end

    mode = if opts[:full_pipeline], do: :full, else: :batch
    articles = load_gvfc_csv(path, opts[:max])
    Mix.shell().info("Loaded #{length(articles)} GVFC articles")

    if mode == :full do
      Mix.shell().info("Using FULL pipeline (--full-pipeline). Slower but includes graph enrichment, fact verification, etc.")
    end

    training_data = articles_to_training_data(articles, mode)
    Mix.shell().info("Generated #{length(training_data)} training records")

    if opts[:stats] do
      show_stats(training_data)
    else
      write_training_data(training_data)
      show_stats(training_data)
    end
  end

  defp load_gvfc_csv(path, max_articles) do
    path
    |> File.stream!()
    |> GVFCParser.parse_stream(skip_headers: true)
    |> Stream.flat_map(&parse_gvfc_row/1)
    |> maybe_limit_stream(max_articles)
    |> Enum.to_list()
  end

  defp parse_gvfc_row(row) do
    # Columns: id(0), article_url(1), headline(2), ..., lead_3_sentences(5),
    #          ..., Q1_Relevant(7), Q2_Focus(8), Q3_Theme1(9), ...
    with headline when byte_size(headline) > 0 <- Enum.at(row, 2),
         lead_text <- Enum.at(row, 5) || "",
         q1 <- String.trim(Enum.at(row, 7) || ""),
         theme_code <- String.trim(Enum.at(row, 9) || ""),
         true <- q1 == "1",
         label when is_binary(label) <- Map.get(@gvfc_theme_codes, theme_code) do
      text = if byte_size(lead_text) > 20, do: lead_text, else: headline
      [%{text: text, label: label}]
    else
      _ -> []
    end
  end

  defp maybe_limit_stream(stream, nil), do: stream
  defp maybe_limit_stream(stream, max) when is_integer(max), do: Stream.take(stream, max)

  # -- MFC loader ---------------------------------------------------------

  @mfc_frame_normalization %{
    "economic" => "economic",
    "capacity and resources" => "capacity",
    "morality" => "morality",
    "fairness and equality" => "fairness",
    "legality, constitutionality and jurisprudence" => "legality",
    "policy prescription and evaluation" => "policy",
    "crime and punishment" => "crime",
    "security and defense" => "security",
    "health and safety" => "health",
    "quality of life" => "quality_of_life",
    "cultural identity" => "cultural_identity",
    "public opinion" => "public_opinion",
    "political" => "political",
    "external regulation and reputation" => "external_regulation",
    "other" => "other"
  }

  defp run_mfc(opts) do
    Mix.Task.run("app.start")

    path =
      opts[:mfc_path] ||
        Mix.raise("""
        MFC corpus requires --mfc-path to point to the MFC JSON file.

        The Media Frames Corpus (Card et al. 2015) is not freely redistributable.
        To obtain it:
          1. Request access from the authors via https://aclanthology.org/P15-2072/
          2. Download the JSON-formatted corpus file
          3. Run: mix gen_framing_data --corpus mfc --mfc-path /path/to/mfc.json

        See data/classifiers/README_framing.md for license details.
        """)

    unless File.exists?(path) do
      Mix.raise("MFC file not found at #{path}")
    end

    articles = load_mfc_json(path, opts[:max])
    Mix.shell().info("Loaded #{length(articles)} MFC articles")

    training_data = articles_to_training_data(articles)
    Mix.shell().info("Generated #{length(training_data)} training records")

    if opts[:stats] do
      show_stats(training_data)
    else
      write_training_data(training_data)
      show_stats(training_data)
    end
  end

  defp load_mfc_json(path, max_articles) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, entries} when is_list(entries) ->
            entries
            |> Enum.flat_map(&parse_mfc_entry/1)
            |> maybe_limit(max_articles)

          {:ok, %{"articles" => entries}} when is_list(entries) ->
            entries
            |> Enum.flat_map(&parse_mfc_entry/1)
            |> maybe_limit(max_articles)

          {:error, reason} ->
            Mix.raise("Failed to parse MFC JSON: #{inspect(reason)}")
        end

      {:error, reason} ->
        Mix.raise("Failed to read MFC file: #{inspect(reason)}")
    end
  end

  defp parse_mfc_entry(%{"text" => text, "frame" => frame})
       when is_binary(text) and text != "" do
    normalized = normalize_mfc_frame(String.downcase(String.trim(frame)))
    if normalized, do: [%{text: text, label: normalized}], else: []
  end

  defp parse_mfc_entry(%{"text" => text, "primary_frame" => frame})
       when is_binary(text) and text != "" do
    normalized = normalize_mfc_frame(String.downcase(String.trim(frame)))
    if normalized, do: [%{text: text, label: normalized}], else: []
  end

  defp parse_mfc_entry(_), do: []

  defp normalize_mfc_frame(raw) do
    Map.get(@mfc_frame_normalization, raw, raw)
  end

  # -- Feature vector pipeline -------------------------------------------

  defp wait_for_pipeline_ready do
    Mix.shell().info("Waiting for NLP pipeline to fully initialize...")
    deadline = System.monotonic_time(:millisecond) + 90_000
    wait_loop(deadline)
    Mix.shell().info("Running warmup through full pipeline (60s timeout)...")
    warmup_pipeline()
  end

  defp wait_loop(deadline) do
    checks = [
      {"Gazetteer", fn -> Brain.ML.Gazetteer.loaded?() end},
      {"EntityExtractor", fn -> Brain.ML.EntityExtractor.is_loaded?() end},
      {"SpeechAct", fn -> Brain.ML.SpeechActClassifierSimple.ready?() end},
      {"Sentiment", fn -> Brain.ML.SentimentClassifierSimple.ready?() end},
      {"MicroClassifiers", fn -> Brain.ML.MicroClassifiers.ready?() end}
    ]

    results =
      Enum.map(checks, fn {name, check_fn} ->
        ready =
          try do
            check_fn.()
          catch
            :exit, _ -> false
          end

        {name, ready}
      end)

    all_ready = Enum.all?(results, fn {_, ready} -> ready end)
    not_ready = results |> Enum.reject(fn {_, r} -> r end) |> Enum.map(fn {n, _} -> n end)

    cond do
      all_ready ->
        Mix.shell().info("All subsystems ready: #{inspect(Enum.map(results, &elem(&1, 0)))}")

      System.monotonic_time(:millisecond) > deadline ->
        Mix.shell().info(
          "Pipeline init timed out after 90s. Still not ready: #{inspect(not_ready)}. " <>
            "Proceeding anyway..."
        )

      true ->
        Process.sleep(2_000)
        wait_loop(deadline)
    end
  end

  defp warmup_pipeline do
    analysis = batch_analyze("The city council voted to fund new schools.", :batch)
    {fv, _} = FeatureExtractor.extract(analysis)
    Mix.shell().info("Warmup complete. Feature vector dim: #{length(fv)}")
  rescue
    e -> Mix.shell().info("Warmup raised: #{Exception.message(e)} (continuing)")
  end

  defp articles_to_training_data(articles, mode \\ :batch) do
    total = length(articles)

    concurrency =
      case mode do
        :full -> max(div(System.schedulers_online(), 2), 1)
        _ -> max(System.schedulers_online() - 1, 1)
      end

    timeout =
      case mode do
        :full -> 60_000
        _ -> 300_000
      end

    Mix.shell().info(
      "Processing #{total} articles through feature-vector pipeline " <>
        "(max_concurrency=#{concurrency}, mode=#{mode})..."
    )

    started_at = System.monotonic_time(:millisecond)
    timeout_count = :counters.new(1, [:atomics])

    results =
      articles
      |> Enum.with_index(1)
      |> Task.async_stream(
        fn {article, idx} ->
          if rem(idx, 50) == 0 do
            elapsed = System.monotonic_time(:millisecond) - started_at
            rate = if idx > 1, do: Float.round(elapsed / idx, 0), else: 0
            Mix.shell().info("  Progress: #{idx}/#{total} (#{rate}ms/article)")
          end

          process_article(article, mode)
        end,
        max_concurrency: concurrency,
        timeout: timeout,
        on_timeout: :kill_task,
        ordered: false
      )
      |> Enum.flat_map(fn
        {:ok, nil} -> []
        {:ok, record} -> [record]

        {:exit, reason} ->
          :counters.add(timeout_count, 1, 1)
          Logger.warning("Task exited: #{inspect(reason)}")
          []
      end)

    elapsed = System.monotonic_time(:millisecond) - started_at
    timed_out = :counters.get(timeout_count, 1)

    Mix.shell().info(
      "Pipeline complete: #{length(results)}/#{total} in #{elapsed}ms" <>
        if(timed_out > 0, do: " (#{timed_out} timed out)", else: "")
    )

    results
  end

  defp process_article(%{text: text, label: label}, mode) do
    chunks = chunk_text(text)

    profiles =
      Enum.map(chunks, fn chunk_text ->
        analysis = batch_analyze(chunk_text, mode)
        {feature_vector, _word_feats} = FeatureExtractor.extract(analysis)
        ChunkProfile.materialize(analysis, feature_vector)
      end)

    valid_profiles = Enum.filter(profiles, fn p -> p.feature_vector != [] end)

    if valid_profiles == [] do
      fv_lengths = Enum.map(profiles, fn p -> length(p.feature_vector) end)
      Logger.warning("Article '#{String.slice(text, 0, 60)}' produced 0 valid profiles out of #{length(profiles)}. FV lengths: #{inspect(fv_lengths)}")
    end

    if valid_profiles != [] do
      doc_profile = DocumentProfile.aggregate(valid_profiles)

      if doc_profile.mean_vector == [] do
        Logger.warning("Article '#{String.slice(text, 0, 60)}' has empty mean_vector after aggregation")
      end

      if doc_profile.mean_vector != [] do
        %{
          "feature_vector" => doc_profile.mean_vector,
          "label" => label
        }
      end
    end
  rescue
    e ->
      Logger.warning("Failed to process article: #{Exception.message(e)}")
      nil
  catch
    :exit, _ ->
      Logger.warning("Article exited: #{String.slice(text, 0, 60)}")
      nil
  end

  # Batch-safe analysis: includes all pure-computation subsystems from the
  # full Pipeline (discourse, POS, speech act, sentiment, entities, SRL,
  # events, slots) but skips Atlas/DB-backed steps that deadlock under
  # concurrent load (EntityGraphEnricher, fact verification, belief
  # extraction, atlas-based intent disambiguation).
  #
  # When `mode: :full` is passed, delegates to `Pipeline.analyze_chunk/1`
  # with a per-chunk timeout to let you compare output quality.
  defp batch_analyze(text, :full) do
    task = Task.async(fn -> FullPipeline.analyze_chunk(text) end)

    case Task.yield(task, 30_000) || Task.shutdown(task, :brutal_kill) do
      {:ok, analysis} -> analysis
      _ ->
        Logger.warning("Full pipeline timed out for chunk, falling back to batch_analyze")
        batch_analyze(text, :batch)
    end
  end

  defp batch_analyze(text, _mode) do
    tokens = Tokenizer.tokenize_words(text)

    pos_tags =
      case POSTagger.get_model() do
        {:ok, model} -> POSTagger.predict(tokens, model)
        _ -> Enum.map(tokens, &{&1, :X})
      end

    entities = safe_call(fn -> EntityExtractor.extract_entities(text) end, [])

    discourse =
      safe_call(
        fn -> DiscourseAnalyzer.analyze(text, participants: [:user, :bot]) end,
        %{addressee: :unknown, confidence: 0.0, indicators: []}
      )

    # Use the simple GenServer classifier, NOT the full SpeechActClassifier
    # which internally calls Pipeline.analyze_chunk and would deadlock.
    speech_act =
      safe_call(
        fn -> SpeechActClassifierSimple.classify(text) end,
        @speech_act_fallback
      )

    sentiment =
      safe_call(
        fn ->
          case SentimentClassifierSimple.classify(text) do
            {:ok, result} -> result
            _ -> %{label: :neutral, score: 0.5}
          end
        end,
        %{label: :neutral, score: 0.5}
      )

    tag_strings = Enum.map(pos_tags, fn {_t, tag} -> to_string(tag) end)
    bio_tags = generate_bio_tags(tokens, tag_strings, entities)

    srl_frames =
      safe_call(
        fn -> SemanticRoleLabeler.label(tokens, bio_tags, entities) end,
        []
      )

    slot_result =
      safe_call(
        fn ->
          intent = Map.get(speech_act, :intent) || infer_intent(speech_act)
          SlotDetector.detect(intent, entities)
        end,
        nil
      )

    %{
      text: text,
      pos_tags: pos_tags,
      speech_act: speech_act,
      discourse: discourse,
      sentiment: sentiment,
      entities: entities,
      srl_frames: srl_frames,
      accumulated_context: nil,
      slots: slot_result
    }
  end

  defp safe_call(fun, fallback) do
    fun.()
  rescue
    _ -> fallback
  catch
    :exit, _ -> fallback
  end

  defp infer_intent(speech_act) do
    cond do
      Map.get(speech_act, :is_question) -> "general.question"
      speech_act.category == :directive -> "general.request"
      speech_act.category == :expressive -> "smalltalk.greeting"
      true -> "general.statement"
    end
  end

  # Minimal BIO tag generation for SRL — marks entity spans as B-ENT/I-ENT.
  defp generate_bio_tags(tokens, _tag_strings, entities) do
    entity_tokens =
      entities
      |> Enum.flat_map(fn e ->
        value = Map.get(e, :value) || Map.get(e, :text) || ""
        String.split(String.downcase(value), ~r/\s+/, trim: true)
      end)
      |> MapSet.new()

    Enum.map(tokens, fn token ->
      if MapSet.member?(entity_tokens, String.downcase(token)), do: "B-ENT", else: "O"
    end)
  end

  defp chunk_text(text) when byte_size(text) > 2000 do
    text
    |> String.split(~r/(?<=[.!?])\s+/, trim: true)
    |> Enum.chunk_every(3)
    |> Enum.map(&Enum.join(&1, " "))
    |> Enum.reject(&(&1 == ""))
  end

  defp chunk_text(text), do: [text]

  # -- Output ------------------------------------------------------------

  defp write_training_data(data) do
    dir = Path.dirname(@output_path)
    File.mkdir_p!(dir)

    json = Jason.encode!(data, pretty: true)
    File.write!(@output_path, json)
    Mix.shell().info("Wrote #{length(data)} records to #{@output_path}")
  end

  defp show_stats(data) do
    by_label =
      data
      |> Enum.group_by(& &1["label"])
      |> Enum.sort_by(fn {_label, items} -> -length(items) end)

    Mix.shell().info("\n--- Label distribution ---")

    Enum.each(by_label, fn {label, items} ->
      Mix.shell().info("  #{String.pad_trailing(label, 20)} #{length(items)}")
    end)

    Mix.shell().info("  #{String.pad_trailing("TOTAL", 20)} #{length(data)}")
  end

  # -- Helpers -----------------------------------------------------------

  defp maybe_limit(articles, nil), do: articles
  defp maybe_limit(articles, max) when is_integer(max), do: Enum.take(articles, max)
end
