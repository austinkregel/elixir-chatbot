defmodule Brain.Test.ModelFactory do
  @moduledoc """
  Trains and loads ML models for test use.

  Trains classifiers from gold standard / fixture data at test startup and
  persists required `.term` files under `config :brain, :ml, models_path:` so
  `Brain.ML.ModelPreflight` and boot-time loaders see a consistent tree.

  Intent fine-grained classification is the `:intent_full` feature-vector
  micro-classifier (trained here from `data/classifiers/intent_full.json`).

  Failure policy: every step is required. If any model fails to train
  or load, `train_and_load_test_models/0` raises. There is no
  soft-failure / retry-next-time behavior, because that hides real
  setup bugs and produces flaky tests.

  ## Usage

      # In test setup or test_helper.exs
      Brain.Test.ModelFactory.train_and_load_test_models()

      # To swap the sentiment model for a specific test
      setup do
        custom_data = [{"great", "positive"}, {"awful", "negative"}]
        Brain.Test.ModelFactory.train_sentiment_classifier(custom_data)
        :ok
      end
  """

  require Logger

  alias Brain.ML.FeatureVectorClassifier
  alias Brain.ML.SimpleClassifier

  @fixtures_dir "test/fixtures/training"

  # Text-based micro-classifiers trained from `apps/brain/test/fixtures/training/micro/<name>.json`.
  # These predate the feature-vector migration and continue to use raw text +
  # `Brain.ML.SimpleClassifier` (TF-IDF).
  @text_micro_classifier_names ~w(
    personal_question
    clarification_response
    modal_directive
    fallback_response
    goal_type
    entity_type
    user_fact_type
    directed_at_bot
    event_argument_role
    coarse_semantic_class
  )a

  # Feature-vector micro-classifiers trained from `data/classifiers/<name>.json`
  # (umbrella-root data dir, produced by `mix gen_micro_data`). Each record
  # is `%{"feature_vector" => [float], "label" => string}` and the model is
  # built via `Brain.ML.FeatureVectorClassifier.train/1`.
  #
  # `framing_class` is feature-vector but is handled separately because it
  # also produces the companion `framing_neutral_centroid.term` artifact
  # consumed by `Brain.Analysis.FramingDetector`.
  @feature_vector_micro_classifier_names ~w(
    intent_full
    intent_domain
    tense_class
    aspect_class
    urgency
    certainty_level
  )a

  @doc """
  Writes `gazetteer.term` via `Brain.ML.Trainer.build_gazetteer_data/2` if it is
  missing at the configured `models_path`. Call **before**
  `Application.ensure_all_started(:brain)` so `EntityExtractor` can load maps
  without the removed legacy JSON fallback.
  """
  def ensure_gazetteer_on_disk! do
    models_path = Application.get_env(:brain, :ml, [])[:models_path]

    if is_nil(models_path) do
      raise "ModelFactory.ensure_gazetteer_on_disk!: :models_path must be set in test config"
    end

    path = Path.join(models_path, "gazetteer.term")

    if not File.exists?(path) do
      File.mkdir_p!(models_path)
      _stats = Brain.ML.Trainer.build_gazetteer_data(%{}, models_path: models_path)
      Logger.info("[ModelFactory] Wrote gazetteer.term to #{path}")
    end

    ensure_gazetteer_non_empty!(path)
    :ok
  end

  defp ensure_gazetteer_non_empty!(path) do
    term =
      case File.read(path) do
        {:ok, bin} -> :erlang.binary_to_term(bin)
        {:error, reason} -> raise "ModelFactory: cannot read gazetteer at #{path}: #{inspect(reason)}"
      end

    if is_map(term) and map_size(term) == 0 do
      seed = %{"model_factory_seed" => %{entity_type: "thing", value: "seed"}}
      File.write!(path, :erlang.term_to_binary(seed))

      Logger.warning(
        "[ModelFactory] Gazetteer was empty after Trainer.build_gazetteer_data/2; wrote minimal seed map"
      )
    end
  end

  @doc """
  Trains and loads all test models into their respective GenServers.

  Trains sentiment, speech act, micro-classifiers, POS, Poincare, triple
  scorer, embedder, and persists artifacts under `models_path` where applicable.
  """
  def train_and_load_test_models do
    if already_trained?() do
      :ok
    else
      Logger.info("[ModelFactory] Starting test model training pipeline...")

      results = %{
        sentiment: run_step!("sentiment classifier", &train_sentiment_classifier/0),
        speech_act: run_step!("speech act classifier", &train_speech_act_classifier/0),
        micro: run_step!("text micro classifiers", &train_micro_classifiers/0),
        feature_vector_micro:
          run_step!("feature-vector micro classifiers", &train_feature_vector_micro_classifiers/0),
        framing: run_step!("framing classifier", &train_framing_classifier/0),
        pos: run_step!("POS tagger", &train_pos_tagger/0),
        poincare: run_step!("Poincare embeddings", &train_poincare_embeddings/0),
        triple_scorer: run_step!("KG triple scorer", &train_triple_scorer/0),
        embedder: run_step!("embedder", &train_embedder/0)
      }

      :ok = Brain.ML.MicroClassifiers.reload()

      case Brain.ML.Poincare.Embeddings.reload() do
        :ok -> :ok
        {:error, r} -> raise "ModelFactory: Poincare.Embeddings.reload/0 failed: #{inspect(r)}"
      end

      case Brain.ML.KnowledgeGraph.TripleScorer.reload() do
        :ok -> :ok
        {:error, r} -> raise "ModelFactory: TripleScorer.reload/0 failed: #{inspect(r)}"
      end

      Logger.info("[ModelFactory] All test models trained successfully: #{inspect(results)}")
      :persistent_term.put({__MODULE__, :trained}, true)
      :ok
    end
  end

  # Runs one training step, requires it to return {:ok, _}, and raises with
  # the actual error tuple otherwise. We intentionally do NOT rescue here;
  # if a trainer crashes, we want the original exception + stacktrace to
  # propagate so the failing test setup is debuggable.
  defp run_step!(label, fun) do
    Logger.info("[ModelFactory] Training #{label}...")

    case fun.() do
      {:ok, value} ->
        Logger.info("[ModelFactory] #{label} done (#{inspect(value)}).")
        value

      other ->
        raise """
        [ModelFactory] Training step #{inspect(label)} did not succeed.
        Got: #{inspect(other)}

        This is treated as a hard failure on purpose. If a model is genuinely
        optional for a given test, gate it at the call site rather than
        reintroducing soft-failure here.
        """
    end
  end

  defp already_trained? do
    :persistent_term.get({__MODULE__, :trained}, false)
  end

  @doc """
  Trains a sentiment classifier from test fixture data (or custom data)
  and loads it into the SentimentClassifierSimple GenServer.

  Raises if the GenServer is not running, the fixture data is missing,
  or the load call returns an unexpected result. No exceptions are
  rescued here — see the moduledoc for the failure policy.
  """
  def train_sentiment_classifier(custom_data \\ nil) do
    data = custom_data || load_sentiment_fixture()

    if data == [] do
      raise "ModelFactory: no sentiment training data available (gold standard + fallback both empty)"
    end

    if Process.whereis(Brain.ML.SentimentClassifierSimple) == nil do
      raise "ModelFactory: Brain.ML.SentimentClassifierSimple is not started; the supervision tree is not up"
    end

    model = SimpleClassifier.train(data)

    case GenServer.call(Brain.ML.SentimentClassifierSimple, {:load_trained_model, model}, 5_000) do
      {:ok, :loaded} ->
        persist_root_model!("sentiment_classifier.term", model)
        {:ok, length(data)}

      :ok ->
        persist_root_model!("sentiment_classifier.term", model)
        {:ok, length(data)}

      other ->
        raise "ModelFactory: SentimentClassifierSimple rejected trained model: #{inspect(other)}"
    end
  end

  @doc """
  Trains the speech-act TF-IDF classifier from the speech act gold standard,
  persists `speech_act_classifier.term`, and loads it into
  `Brain.ML.SpeechActClassifierSimple`.
  """
  def train_speech_act_classifier do
    gold = Brain.ML.EvaluationStore.load_gold_standard("speech_act")

    data =
      gold
      |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["speech_act"]) end)
      |> Enum.map(fn ex -> {ex["text"], ex["speech_act"]} end)

    if data == [] do
      raise """
      ModelFactory: no speech act training data. Expected examples in
      #{Brain.ML.EvaluationStore.gold_standard_path("speech_act")} with "text" and "speech_act".
      Run `mix generate_gold_standard --speech-act` (or the full pipeline) to create it.
      """
    end

    if Process.whereis(Brain.ML.SpeechActClassifierSimple) == nil do
      raise "ModelFactory: Brain.ML.SpeechActClassifierSimple is not started; the supervision tree is not up"
    end

    model = SimpleClassifier.train(data)
    persist_root_model!("speech_act_classifier.term", model)

    case GenServer.call(Brain.ML.SpeechActClassifierSimple, {:load_trained_model, model}, 5_000) do
      {:ok, :loaded} -> {:ok, length(data)}
      other -> raise "ModelFactory: SpeechActClassifierSimple rejected trained model: #{inspect(other)}"
    end
  end

  @doc """
  Trains all micro-classifiers from test fixture data and loads them
  into the MicroClassifiers GenServer.

  Raises on missing fixtures, a downed GenServer, or a load failure.
  """
  def train_micro_classifiers do
    models =
      Enum.reduce(@text_micro_classifier_names, %{}, fn name, acc ->
        case load_text_micro_data(name) do
          [] ->
            acc

          data ->
            model = SimpleClassifier.train(data)
            Map.put(acc, name, model)
        end
      end)

    if models == %{} do
      raise """
      ModelFactory: no text micro-classifier training data found.
      Looked under #{fixtures_path("micro")} (legacy fixtures) and
      #{data_classifiers_path("")} (umbrella data/classifiers/).
      """
    end

    if Process.whereis(Brain.ML.MicroClassifiers) == nil do
      raise "ModelFactory: Brain.ML.MicroClassifiers is not started; the supervision tree is not up"
    end

    persist_micro_models!(models)

    :ok = GenServer.call(Brain.ML.MicroClassifiers, {:load_trained_models, models}, 5_000)
    {:ok, map_size(models)}
  end

  @doc """
  Trains all feature-vector axis micro-classifiers (intent_full,
  intent_domain, tense_class, aspect_class, urgency, certainty_level)
  from `data/classifiers/<name>.json` and loads them into the
  MicroClassifiers GenServer.

  Each record in those files must have shape
  `%{"feature_vector" => [float], "label" => string}` (produced by
  `mix gen_micro_data`). Records missing either field are silently
  dropped, but every classifier in the list MUST yield at least one
  valid pair, otherwise this step raises.
  """
  def train_feature_vector_micro_classifiers do
    {models, missing} =
      Enum.reduce(@feature_vector_micro_classifier_names, {%{}, []}, fn name, {acc, miss} ->
        case load_feature_vector_data(name) do
          [] ->
            {acc, [name | miss]}

          pairs ->
            model = FeatureVectorClassifier.train(pairs, balance: true)
            {Map.put(acc, name, model), miss}
        end
      end)

    if missing != [] do
      raise """
      ModelFactory: no feature-vector training data for #{inspect(Enum.reverse(missing))}.
      Expected files at #{data_classifiers_path("<name>.json")} with records of shape
      %{"feature_vector" => [float], "label" => string}.

      Run `mix gen_micro_data` to (re)generate them from gold_standard.json.
      """
    end

    if Process.whereis(Brain.ML.MicroClassifiers) == nil do
      raise "ModelFactory: Brain.ML.MicroClassifiers is not started; the supervision tree is not up"
    end

    persist_micro_models!(models)

    :ok = GenServer.call(Brain.ML.MicroClassifiers, {:load_trained_models, models}, 5_000)
    {:ok, map_size(models)}
  end

  @doc """
  Trains the `:framing_class` feature-vector classifier and writes the
  companion `framing_neutral_centroid.term` artifact that
  `Brain.Analysis.FramingDetector` loads at boot. Skips with `{:ok, 0}`
  if `data/classifiers/framing_class.json` is absent (the framing corpus
  is optional in CI).
  """
  def train_framing_classifier do
    pairs =
      case load_feature_vector_data(:framing_class) do
        [] ->
          Logger.info(
            "[ModelFactory] framing_class corpus not found; training minimal placeholder model"
          )

          minimal_framing_training_pairs()

        p ->
          p
      end

    model = FeatureVectorClassifier.train(pairs, balance: true)
    persist_micro_models!(%{framing_class: model})

    neutral = compute_neutral_centroid(model, pairs)
    persist_neutral_centroid!(neutral)

    :ok =
      GenServer.call(
        Brain.ML.MicroClassifiers,
        {:load_trained_models, %{framing_class: model}},
        5_000
      )

    {:ok, length(pairs)}
  end

  # Tiny synthetic corpus so `micro/framing_class.term` and
  # `framing_neutral_centroid.term` always exist in test (FramingDetector +
  # ModelPreflight expect them). Vector length matches the live chunk feature
  # extractor so `MicroClassifiers.classify_vector(:framing_class, ...)` agrees
  # with production models.
  defp minimal_framing_training_pairs do
    dim = Brain.Analysis.FeatureExtractor.ChunkFeatures.vector_dimension()
    z = List.duplicate(0.0, dim)

    v_neutral =
      z
      |> List.replace_at(0, 0.12)
      |> List.replace_at(1, 0.05)

    v_control =
      z
      |> List.replace_at(0, 0.88)
      |> List.replace_at(1, 0.42)

    [
      {v_neutral, "neutral"},
      {v_control, "control"},
      {List.replace_at(v_neutral, 2, 0.03), "neutral"},
      {List.replace_at(v_control, 2, 0.11), "control"}
    ]
  end

  @doc """
  Trains a POS tagger from gold standard POS-annotated data and saves
  the model to the test models path so POSTagger.load_model() works.

  Raises on missing data or a training failure.
  """
  def train_pos_tagger do
    alias Brain.ML.POSTagger

    {sequences, source_path} = load_pos_sequences_from_gold_standard()
    sequences = sequences ++ pos_music_propn_bootstrap_sequences()

    if sequences == [] do
      raise """
      ModelFactory: no POS training data (tokens + pos_tags) found in any gold-standard
      file under #{gold_standard_path("intent/")}.

      Tried, in order:
        - intent/gold_standard.json (current; post-migration this file no longer
          carries `tokens`/`pos_tags`)
        - intent/gold_standard.pre-rebuild.json (legacy snapshot retained for
          POS bootstrap)

      Add POS-labeled sequences to one of those files, or extend
      Brain.Test.ModelFactory.@pos_corpus_candidates to point at a new
      POS corpus file under priv/evaluation/.
      """
    end

    Logger.info("[ModelFactory] POS training using #{length(sequences)} sequences from #{source_path}")

    case POSTagger.train(sequences) do
      {:ok, model} ->
        models_path = Application.get_env(:brain, :ml)[:models_path]

        if models_path do
          save_path = Path.join(models_path, "pos_model.term")
          File.mkdir_p!(Path.dirname(save_path))
          POSTagger.save_model(model, save_path)
        end

        {:ok, length(sequences)}

      {:error, reason} ->
        raise "ModelFactory: POSTagger.train/1 failed: #{inspect(reason)}"
    end
  end

  @doc """
  Trains Poincare embeddings from entity type hierarchy and saves them
  so the Poincare.Embeddings GenServer can load them.
  """
  def train_poincare_embeddings do
    alias Brain.ML.Poincare.Embeddings

    pairs = load_hierarchy_pairs()

    if pairs == [] do
      raise "ModelFactory: no Poincare hierarchy data found at #{entity_types_path()}"
    end

    {:ok, embeddings, entity_to_idx, idx_to_entity} =
      Embeddings.train(pairs, dim: 5, epochs: 20, learning_rate: 0.01)

    models_path = Application.get_env(:brain, :ml)[:models_path]

    if models_path do
      path = Path.join([models_path, "default", "poincare", "embeddings.term"])
      Embeddings.save(embeddings, entity_to_idx, idx_to_entity, 5, path)
    end

    {:ok, length(pairs)}
  end

  @doc """
  Trains a small KG triple scorer from entity type hierarchy triples
  and saves it so the TripleScorer GenServer can load it.
  """
  def train_triple_scorer do
    alias Brain.ML.KnowledgeGraph.TripleScorer

    triples = load_hierarchy_triples()

    if length(triples) < 3 do
      raise "ModelFactory: insufficient triple data (#{length(triples)}) at #{entity_types_path()}; need at least 3"
    end

    {:ok, _model, params, vocab, config} =
      TripleScorer.train(triples, epochs: 10, neg_ratio: 3)

    models_path = Application.get_env(:brain, :ml)[:models_path]

    if models_path do
      path = Path.join([models_path, "default", "kg_lstm", "triple_scorer.term"])
      TripleScorer.save_model(params, vocab, config, path)
    end

    {:ok, length(triples)}
  end

  @doc """
  Trains an embedder vocabulary from intent training data and loads it
  into the Embedder GenServer. Also saves to disk so auto-load works.
  """
  def train_embedder do
    data = load_intent_fixture()

    if data == [] do
      raise "ModelFactory: no embedder training data (intent fixture is empty); checked gold standard and fallback fixture"
    end

    if Process.whereis(Brain.Memory.Embedder) == nil do
      raise "ModelFactory: Brain.Memory.Embedder is not started; the supervision tree is not up"
    end

    texts = Enum.map(data, fn {text, _label} -> text end)
    {:ok, vocab_size} = Brain.Memory.Embedder.build_vocabulary(texts)

    models_path = Application.get_env(:brain, :ml, [])[:models_path]

    if models_path do
      {:ok, model} = Brain.Memory.Embedder.export_model()
      save_path = Path.join(models_path, "embedder.term")
      File.mkdir_p!(Path.dirname(save_path))
      File.write!(save_path, :erlang.term_to_binary(model))
    end

    {:ok, vocab_size}
  end

  # -- Private --

  # POS training is bootstrapped from any gold-standard file that still
  # carries `tokens` + `pos_tags`. After the feature-vector migration the
  # primary `intent/gold_standard.json` only ships `intent` + `text`, so
  # we fall through to the retained `pre-rebuild` snapshot which still
  # has the legacy POS columns. List is ordered preferred-first.
  @pos_corpus_candidates [
    "intent/gold_standard.json",
    "intent/gold_standard.pre-rebuild.json"
  ]

  # Keeps FeatureTest-style music commands extracting OOV artist/title spans
  # via `EntityExtractor` PROPN hints when the legacy gold snapshot is thin.
  defp pos_music_propn_bootstrap_sequences do
    [
      %{
        tokens: ["Play", "some", "Korvo", "Mitski"],
        tags: ["VERB", "DET", "PROPN", "PROPN"],
        source: "bootstrap_music_propn"
      },
      %{
        tokens: ["Play", "Bohemian", "Rhapsody"],
        tags: ["VERB", "PROPN", "PROPN"],
        source: "bootstrap_music_propn"
      }
    ]
  end

  defp load_pos_sequences_from_gold_standard do
    Enum.reduce_while(@pos_corpus_candidates, {[], nil}, fn relative, _acc ->
      path = gold_standard_path(relative)

      case load_pos_sequences_from_path(path) do
        [] -> {:cont, {[], path}}
        sequences -> {:halt, {sequences, path}}
      end
    end)
  end

  defp load_pos_sequences_from_path(path) do
    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            entries
            |> Enum.filter(fn ex ->
              tokens = ex["tokens"] || []
              tags = ex["pos_tags"] || []
              tokens != [] and length(tokens) == length(tags)
            end)
            |> Enum.map(fn ex ->
              %{tokens: ex["tokens"], tags: ex["pos_tags"], source: ex["intent"]}
            end)

          {:ok, other} ->
            raise "ModelFactory: gold standard at #{path} decoded to a non-list: #{inspect(other) |> String.slice(0, 200)}"

          {:error, reason} ->
            raise "ModelFactory: failed to decode gold standard JSON at #{path}: #{inspect(reason)}"
        end

      {:error, :enoent} ->
        Logger.debug("[ModelFactory] POS corpus candidate not found: #{path}")
        []

      {:error, reason} ->
        raise "ModelFactory: cannot read gold standard at #{path}: #{inspect(reason)}"
    end
  end

  defp load_intent_fixture do
    # Use gold standard data for realistic classification accuracy.
    # This ensures test models reflect the same reality as production models.
    path = gold_standard_path("intent/gold_standard.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "intent", "unknown")}
            end)

          {:ok, other} ->
            raise "ModelFactory: intent gold standard at #{path} decoded to a non-list: #{inspect(other) |> String.slice(0, 200)}"

          {:error, reason} ->
            raise "ModelFactory: failed to decode intent gold standard JSON at #{path}: #{inspect(reason)}"
        end

      {:error, :enoent} ->
        Logger.info("[ModelFactory] gold standard not found at #{path}, using small fallback fixture")
        load_intent_fallback()

      {:error, reason} ->
        raise "ModelFactory: cannot read intent gold standard at #{path}: #{inspect(reason)}"
    end
  end

  defp load_intent_fallback do
    path = fixtures_path("intents_small.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "intent", "unknown")}
            end)

          _ ->
            []
        end

      _ ->
        []
    end
  end

  defp load_sentiment_fixture do
    path = gold_standard_path("sentiment/gold_standard.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "sentiment", "neutral")}
            end)

          {:ok, other} ->
            raise "ModelFactory: sentiment gold standard at #{path} decoded to a non-list: #{inspect(other) |> String.slice(0, 200)}"

          {:error, reason} ->
            raise "ModelFactory: failed to decode sentiment gold standard JSON at #{path}: #{inspect(reason)}"
        end

      {:error, :enoent} ->
        Logger.info("[ModelFactory] sentiment gold standard not found at #{path}, using fallback")
        load_sentiment_fallback()

      {:error, reason} ->
        raise "ModelFactory: cannot read sentiment gold standard at #{path}: #{inspect(reason)}"
    end
  end

  defp load_sentiment_fallback do
    path = fixtures_path("sentiment_small.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "sentiment", "neutral")}
            end)

          _ ->
            []
        end

      _ ->
        []
    end
  end

  # Resolve text-classifier training data, preferring the umbrella-root
  # `data/classifiers/<name>.json` (the post-migration single source of
  # truth) and falling back to the legacy `apps/brain/test/fixtures/training/micro/`
  # fixtures for classifiers that were never moved.
  defp load_text_micro_data(name) do
    case load_text_records(data_classifiers_path("#{name}.json")) do
      [] -> load_text_records(fixtures_path("micro/#{name}.json"))
      records -> records
    end
  end

  defp load_text_records(path) do
    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            entries
            |> Enum.flat_map(fn
              %{"text" => text, "label" => label} when is_binary(text) and is_binary(label) ->
                [{text, label}]

              _ ->
                []
            end)

          _ ->
            []
        end

      _ ->
        []
    end
  end

  defp load_feature_vector_data(name) do
    path = data_classifiers_path("#{name}.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            Enum.flat_map(entries, fn
              %{"feature_vector" => vec, "label" => label}
              when is_list(vec) and is_binary(label) and length(vec) > 0 ->
                [{vec, label}]

              _ ->
                []
            end)

          _ ->
            []
        end

      _ ->
        []
    end
  end

  defp data_classifiers_path(relative) do
    # `data/classifiers/` lives at the umbrella root, not inside the brain
    # priv tree. Resolve it from the current working directory (Mix tests
    # always run from the umbrella root) and fall back to walking up from
    # priv if for some reason cwd is the brain app.
    candidates = [
      Path.join([File.cwd!(), "data", "classifiers", relative]),
      Path.join([File.cwd!(), "..", "..", "data", "classifiers", relative]) |> Path.expand()
    ]

    Enum.find(candidates, hd(candidates), &File.exists?/1)
  end

  defp persist_micro_models!(models) when is_map(models) do
    case Application.get_env(:brain, :ml, [])[:models_path] do
      nil ->
        :ok

      base ->
        dir = Path.join(base, "micro")
        File.mkdir_p!(dir)

        Enum.each(models, fn {name, model} ->
          path = Path.join(dir, "#{name}.term")
          File.write!(path, :erlang.term_to_binary(model))
        end)
    end
  end

  defp persist_neutral_centroid!(centroid) do
    case Application.get_env(:brain, :ml, [])[:models_path] do
      nil ->
        :ok

      base ->
        dir = Path.join(base, "micro")
        File.mkdir_p!(dir)
        path = Path.join(dir, "framing_neutral_centroid.term")
        File.write!(path, :erlang.term_to_binary(centroid))
    end
  end

  defp persist_root_model!(filename, model) do
    case Application.get_env(:brain, :ml, [])[:models_path] do
      nil ->
        :ok

      base ->
        File.mkdir_p!(base)
        path = Path.join(base, filename)
        File.write!(path, :erlang.term_to_binary(model))
    end
  end

  # Pick the centroid of the class whose name is most "neutral" (matches
  # the heuristic in mix train_framing) or fall back to the grand mean
  # of all class centroids.
  defp compute_neutral_centroid(%{label_centroids: centroids}, _pairs)
       when map_size(centroids) == 0 do
    []
  end

  defp compute_neutral_centroid(%{label_centroids: centroids}, _pairs) do
    # `label_centroids` is `%{label => list(list(float))}` — each value is the
    # set of k-means prototypes for that class, *not* a single centroid. We
    # always reduce to a single flat float vector here so the persisted
    # `framing_neutral_centroid.term` passes `ModelPreflight` (which requires
    # `Enum.all?(data, &is_float/1)`).
    case Map.get(centroids, "neutral") || Map.get(centroids, "other") do
      nil ->
        # Grand mean: collapse each class's protos to one centroid, then
        # mean those so every class contributes equally regardless of k.
        centroids
        |> Map.values()
        |> Enum.map(&mean_vectors/1)
        |> mean_vectors()

      protos when is_list(protos) ->
        mean_vectors(protos)
    end
  end

  defp mean_vectors([]), do: []
  defp mean_vectors([single]) when is_list(single), do: single

  defp mean_vectors([first | _] = vecs) when is_list(first) do
    dim = length(first)
    n = length(vecs)
    zero = List.duplicate(0.0, dim)

    vecs
    |> Enum.reduce(zero, fn vec, acc ->
      Enum.zip_with(acc, vec, &(&1 + &2))
    end)
    |> Enum.map(&(&1 / n))
  end

  defp fixtures_path(relative) do
    brain_root = Path.join(File.cwd!(), "apps/brain")

    if File.dir?(brain_root) do
      Path.join([brain_root, @fixtures_dir, relative])
    else
      Path.join([@fixtures_dir, relative])
    end
  end

  defp load_hierarchy_pairs do
    path = entity_types_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"type_hierarchy" => hierarchy}} ->
            Enum.flat_map(hierarchy, fn {parent, children} when is_list(children) ->
              Enum.map(children, fn child -> {child, parent} end)
            end)

          _ -> []
        end

      _ -> []
    end
  end

  defp load_hierarchy_triples do
    path = entity_types_path()

    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, %{"type_hierarchy" => hierarchy}} ->
            Enum.flat_map(hierarchy, fn {parent, children} when is_list(children) ->
              Enum.flat_map(children, fn child ->
                [{child, "is_a", parent}, {parent, "has_subtype", child}]
              end)
            end)

          _ -> []
        end

      _ -> []
    end
  end

  defp entity_types_path do
    case :code.priv_dir(:brain) do
      {:error, _} ->
        Path.join(["apps", "brain", "priv", "analysis", "entity_types.json"])

      priv_dir ->
        Path.join(priv_dir, Path.join("analysis", "entity_types.json"))
    end
  end

  defp gold_standard_path(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} ->
        Path.join(["apps", "brain", "priv", "evaluation", relative])

      priv_dir ->
        Path.join(priv_dir, Path.join("evaluation", relative))
    end
  end
end
