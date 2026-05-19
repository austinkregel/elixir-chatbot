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

  # -- Private --

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

  defp gold_standard_path(relative) do
    case :code.priv_dir(:brain) do
      {:error, _} ->
        Path.join(["apps", "brain", "priv", "evaluation", relative])

      priv_dir ->
        Path.join(priv_dir, Path.join("evaluation", relative))
    end
  end
end
