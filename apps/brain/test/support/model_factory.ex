defmodule Brain.Test.ModelFactory do
  @moduledoc """
  Trains and loads ML models for test use.

  Instead of relying on pre-committed .term files, this module trains
  classifiers from gold standard data at test startup. Intent classification
  uses the full gold standard (5000+ examples, 230+ intents) to ensure
  tests reflect production-level accuracy. Sentiment and micro-classifiers
  use smaller fixture datasets.

  ## Usage

      # In test setup or test_helper.exs
      Brain.Test.ModelFactory.train_and_load_test_models()

      # To swap a model for a specific test
      setup do
        custom_data = [{"hello", "greeting"}, {"bye", "farewell"}]
        Brain.Test.ModelFactory.train_intent_classifier(custom_data)
        :ok
      end
  """

  require Logger

  alias Brain.ML.SimpleClassifier

  @fixtures_dir "test/fixtures/training"

  @micro_classifier_names ~w(
    personal_question
    clarification_response
    modal_directive
    fallback_response
    goal_type
    entity_type
    user_fact_type
    directed_at_bot
  )a

  @doc """
  Trains and loads all test models into their respective GenServers.

  Trains intent classifier, sentiment classifier, and all micro-classifiers
  from small test fixture datasets.
  """
  def train_and_load_test_models do
    unless already_trained?() do
      Logger.info("[ModelFactory] Starting test model training pipeline...")

      Logger.info("[ModelFactory] Training intent classifier...")
      intent_result = train_intent_classifier()
      Logger.info("[ModelFactory] Intent classifier done.")

      Logger.info("[ModelFactory] Training sentiment classifier...")
      sentiment_result = train_sentiment_classifier()
      Logger.info("[ModelFactory] Sentiment classifier done.")

      Logger.info("[ModelFactory] Training micro classifiers...")
      micro_result = train_micro_classifiers()
      Logger.info("[ModelFactory] Micro classifiers done.")

      Logger.info("[ModelFactory] Training POS tagger...")
      pos_result = train_pos_tagger()
      Logger.info("[ModelFactory] POS tagger done.")

      Logger.info("[ModelFactory] Training embedder...")
      embedder_result = train_embedder()
      Logger.info("[ModelFactory] Embedder done: #{inspect(embedder_result)}")

      Logger.info("[ModelFactory] Initializing intent arbitrator...")
      arb_result = init_intent_arbitrator()
      Logger.info("[ModelFactory] Intent arbitrator done: #{inspect(arb_result)}")

      Logger.info("[ModelFactory] Initializing response scorer...")
      resp_result = init_response_scorer()
      Logger.info("[ModelFactory] Response scorer done: #{inspect(resp_result)}")

      case {intent_result, sentiment_result, pos_result} do
        {{:ok, _}, {:ok, _}, {:ok, _}} ->
          Logger.info("[ModelFactory] All test models trained successfully.")
          :persistent_term.put({__MODULE__, :trained}, true)

        _ ->
          # Micro classifiers are optional in many tests; include status for debugging.
          Logger.warning("[ModelFactory] Model training incomplete, will retry on next setup", %{
            intent_result: inspect(intent_result),
            sentiment_result: inspect(sentiment_result),
            micro_result: inspect(micro_result),
            pos_result: inspect(pos_result)
          })
      end
    end
  end

  defp already_trained? do
    :persistent_term.get({__MODULE__, :trained}, false)
  end

  @doc """
  Trains an intent classifier from test fixture data (or custom data)
  and loads it into the IntentClassifierSimple GenServer.
  """
  def train_intent_classifier(custom_data \\ nil) do
    data = custom_data || load_intent_fixture()

    cond do
      data == [] ->
        {:error, :no_intent_training_data}

      Process.whereis(Brain.ML.IntentClassifierSimple) == nil ->
        {:error, :intent_classifier_not_started}

      true ->
        model = SimpleClassifier.train(data)
        :ok = GenServer.call(Brain.ML.IntentClassifierSimple, {:load_trained_model, model}, 120_000)
        {:ok, length(data)}
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train intent classifier: #{inspect(e)}")
      {:error, :intent_training_failed}
  end

  @doc """
  Trains a sentiment classifier from test fixture data (or custom data)
  and loads it into the SentimentClassifierSimple GenServer.
  """
  def train_sentiment_classifier(custom_data \\ nil) do
    data = custom_data || load_sentiment_fixture()

    cond do
      data == [] ->
        {:error, :no_sentiment_training_data}

      Process.whereis(Brain.ML.SentimentClassifierSimple) == nil ->
        {:error, :sentiment_classifier_not_started}

      true ->
        model = SimpleClassifier.train(data)

        case GenServer.call(Brain.ML.SentimentClassifierSimple, {:load_trained_model, model}, 5_000) do
          {:ok, :loaded} -> {:ok, length(data)}
          :ok -> {:ok, length(data)}
          other -> {:error, {:unexpected_load_result, other}}
        end
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train sentiment classifier: #{inspect(e)}")
      {:error, :sentiment_training_failed}
  end

  @doc """
  Trains all micro-classifiers from test fixture data and loads them
  into the MicroClassifiers GenServer.
  """
  def train_micro_classifiers do
    models =
      Enum.reduce(@micro_classifier_names, %{}, fn name, acc ->
        case load_micro_fixture(name) do
          [] ->
            acc

          data ->
            model = SimpleClassifier.train(data)
            Map.put(acc, name, model)
        end
      end)

    cond do
      models == %{} ->
        {:error, :no_micro_training_data}

      Process.whereis(Brain.ML.MicroClassifiers) == nil ->
        {:error, :micro_classifiers_not_started}

      true ->
        :ok = GenServer.call(Brain.ML.MicroClassifiers, {:load_trained_models, models}, 5_000)
        {:ok, map_size(models)}
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train micro classifiers: #{inspect(e)}")
      {:error, :micro_training_failed}
  end

  @doc """
  Trains a POS tagger from gold standard POS-annotated data and saves
  the model to the test models path so POSTagger.load_model() works.
  """
  def train_pos_tagger do
    alias Brain.ML.POSTagger

    sequences = load_pos_sequences_from_gold_standard()

    cond do
      sequences == [] ->
        {:error, :no_pos_training_data}

      true ->
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
            Logger.warning("ModelFactory: failed to train POS tagger: #{inspect(reason)}")
            {:error, :pos_training_failed}
        end
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train POS tagger: #{inspect(e)}")
      {:error, :pos_training_failed}
  end

  @doc """
  Trains a small GCN model from a subset of gold standard data
  and saves it so the GCN.Model GenServer can load it.
  """
  def train_gcn_model do
    alias Brain.ML.GCN.{TextGraph, Model}

    data = load_intent_fixture() |> Enum.take(200)

    cond do
      length(data) < 10 ->
        {:error, :insufficient_gcn_data}

      true ->
        text_graph = TextGraph.build(data, vocab_size: 500)
        {:ok, _model, params, _adj_norm, config} = Model.train(text_graph,
          epochs: 5, hidden_dim: 64, dropout: 0.3)

        models_path = Application.get_env(:brain, :ml)[:models_path]
        if models_path do
          path = Path.join([models_path, "default", "gcn", "model.term"])
          Model.save_model(params, text_graph, config, path)
        end

        {:ok, length(data)}
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train GCN model: #{inspect(e)}")
      {:error, :gcn_training_failed}
  end

  @doc """
  Trains Poincare embeddings from entity type hierarchy and saves them
  so the Poincare.Embeddings GenServer can load them.
  """
  def train_poincare_embeddings do
    alias Brain.ML.Poincare.Embeddings

    pairs = load_hierarchy_pairs()

    cond do
      pairs == [] ->
        {:error, :no_hierarchy_data}

      true ->
        {:ok, embeddings, entity_to_idx, idx_to_entity} = Embeddings.train(pairs,
          dim: 5, epochs: 20, learning_rate: 0.01)

        models_path = Application.get_env(:brain, :ml)[:models_path]
        if models_path do
          path = Path.join([models_path, "default", "poincare", "embeddings.term"])
          Embeddings.save(embeddings, entity_to_idx, idx_to_entity, 5, path)
        end

        {:ok, length(pairs)}
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train Poincare embeddings: #{inspect(e)}")
      {:error, :poincare_training_failed}
  end

  @doc """
  Trains a small KG triple scorer from entity type hierarchy triples
  and saves it so the TripleScorer GenServer can load it.
  """
  def train_triple_scorer do
    alias Brain.ML.KnowledgeGraph.TripleScorer

    triples = load_hierarchy_triples()

    cond do
      length(triples) < 3 ->
        {:error, :insufficient_triple_data}

      true ->
        {:ok, _model, params, vocab, config} = TripleScorer.train(triples,
          epochs: 10, neg_ratio: 3)

        models_path = Application.get_env(:brain, :ml)[:models_path]
        if models_path do
          path = Path.join([models_path, "default", "kg_lstm", "triple_scorer.term"])
          TripleScorer.save_model(params, vocab, config, path)
        end

        {:ok, length(triples)}
    end
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train triple scorer: #{inspect(e)}")
      {:error, :triple_scorer_training_failed}
  end

  @doc """
  Trains an embedder vocabulary from intent training data and loads it
  into the Embedder GenServer. Also saves to disk so auto-load works.
  """
  def train_embedder do
    data = load_intent_fixture()

    cond do
      data == [] ->
        {:error, :no_training_data}

      Process.whereis(Brain.Memory.Embedder) == nil ->
        {:error, :embedder_not_started}

      true ->
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
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train embedder: #{inspect(e)}")
      {:error, :embedder_training_failed}
  end

  @doc """
  Initializes the IntentArbitrator with a randomly-initialized model so it
  loads successfully and the ensemble voting path gets exercised in tests.
  """
  def init_intent_arbitrator do
    alias Brain.ML.IntentArbitrator

    model = IntentArbitrator.build_model()
    template = %{"features" => Nx.template({1, IntentArbitrator.feature_count()}, :f32)}
    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    IntentArbitrator.save_model(params)

    if Process.whereis(Brain.ML.IntentArbitrator) do
      try do
        IntentArbitrator.reload()
      catch
        :exit, _ -> :ok
      end
    end

    {:ok, :arbitrator_initialized}
  rescue
    e ->
      Logger.warning("ModelFactory: failed to initialize intent arbitrator: #{inspect(e)}")
      {:error, :arbitrator_init_failed}
  end

  @doc """
  Initializes the LSTMResponse scorer with a randomly-initialized model so it
  loads successfully and LSTM response scoring gets exercised in tests.
  """
  def init_response_scorer do
    vocab_size = 50

    config = %{
      embedding_size: 32,
      hidden_size: 32,
      dropout: 0.1,
      max_query_length: 20,
      max_response_length: 20,
      num_candidates: 3,
      max_vocab: vocab_size,
      learning_rate: 0.001,
      batch_size: 16,
      epochs: 1
    }

    token_vocab = for i <- 0..(vocab_size - 1), into: %{}, do: {"token_#{i}", i}
    vocabularies = %{token_vocab: token_vocab}

    query_input = Axon.input("query", shape: {nil, config.max_query_length})

    query_encoded =
      query_input
      |> Axon.embedding(vocab_size, config.embedding_size, name: "query_embedding")
      |> Axon.lstm(config.hidden_size, name: "query_lstm")
      |> then(fn {seq, _} -> seq end)
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)

    response_input = Axon.input("response", shape: {nil, config.max_response_length})

    response_encoded =
      response_input
      |> Axon.embedding(vocab_size, config.embedding_size, name: "response_embedding")
      |> Axon.lstm(config.hidden_size, name: "response_lstm")
      |> then(fn {seq, _} -> seq end)
      |> Axon.nx(fn x -> Nx.mean(x, axes: [1]) end)

    model =
      Axon.concatenate([query_encoded, response_encoded], axis: 1)
      |> Axon.dense(64, activation: :relu, name: "scorer_hidden")
      |> Axon.dropout(rate: config.dropout)
      |> Axon.dense(1, activation: :sigmoid, name: "scorer_output")

    template = %{
      "query" => Nx.template({1, config.max_query_length}, :s64),
      "response" => Nx.template({1, config.max_response_length}, :s64)
    }

    {init_fn, _} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    params = if is_struct(params, Axon.ModelState), do: params.data, else: params
    portable_params = transfer_to_binary(params)

    models_path = Application.get_env(:brain, :ml)[:models_path]

    if models_path do
      lstm_path = Path.join(models_path, "lstm")
      File.mkdir_p!(lstm_path)
      save_path = Path.join(lstm_path, "response_scorer.term")

      data = %{
        scorer_params: portable_params,
        vocabularies: vocabularies,
        config: config
      }

      binary = :erlang.term_to_binary(data)
      File.write!(save_path, binary)
    end

    if Process.whereis(Brain.Response.LSTMResponse) do
      try do
        Brain.Response.LSTMResponse.reload()
      catch
        :exit, _ -> :ok
      end
    end

    {:ok, :response_scorer_initialized}
  rescue
    e ->
      Logger.warning("ModelFactory: failed to initialize response scorer: #{inspect(e)}")
      {:error, :response_scorer_failed}
  end

  # -- Private --

  defp transfer_to_binary(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp transfer_to_binary(%Axon.ModelState{} = s),
    do: Axon.ModelState.new(transfer_to_binary(s.data))

  defp transfer_to_binary(map) when is_map(map),
    do: Map.new(map, fn {k, v} -> {k, transfer_to_binary(v)} end)

  defp transfer_to_binary(other), do: other

  defp load_pos_sequences_from_gold_standard do
    gold_standard_path = gold_standard_path("intent/gold_standard.json")

    case File.read(gold_standard_path) do
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
              %{
                tokens: ex["tokens"],
                tags: ex["pos_tags"],
                source: ex["intent"]
              }
            end)

          _ ->
            Logger.warning("ModelFactory: failed to decode gold standard POS data")
            []
        end

      {:error, _} ->
        Logger.warning("ModelFactory: gold standard not found for POS training")
        []
    end
  end

  defp load_intent_fixture do
    # Use gold standard data for realistic classification accuracy.
    # This ensures test models reflect the same reality as production models.
    gold_standard_path = gold_standard_path("intent/gold_standard.json")

    case File.read(gold_standard_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "intent", "unknown")}
            end)

          _ ->
            Logger.warning("ModelFactory: failed to decode gold standard intent data")
            load_intent_fallback()
        end

      {:error, _} ->
        Logger.warning("ModelFactory: gold standard not found at #{gold_standard_path}, using fallback")
        load_intent_fallback()
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
    gold_standard_path = gold_standard_path("sentiment/gold_standard.json")

    case File.read(gold_standard_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} when is_list(entries) ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "sentiment", "neutral")}
            end)

          _ ->
            Logger.warning("ModelFactory: failed to decode gold standard sentiment data")
            load_sentiment_fallback()
        end

      {:error, _} ->
        Logger.warning("ModelFactory: gold standard sentiment not found, using fallback")
        load_sentiment_fallback()
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

  defp load_micro_fixture(name) do
    path = fixtures_path("micro/#{name}.json")

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} ->
            Enum.map(entries, fn entry ->
              {Map.get(entry, "text", ""), Map.get(entry, "label", "unknown")}
            end)

          _ ->
            []
        end

      _ ->
        []
    end
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
