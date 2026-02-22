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
  )a

  @doc """
  Trains and loads all test models into their respective GenServers.

  Trains intent classifier, sentiment classifier, and all micro-classifiers
  from small test fixture datasets.
  """
  def train_and_load_test_models do
    # Only train if models haven't been loaded yet (training is expensive with gold standard data)
    unless already_trained?() do
      train_intent_classifier()
      train_sentiment_classifier()
      train_micro_classifiers()
      train_pos_tagger()
      :persistent_term.put({__MODULE__, :trained}, true)
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

    if data != [] do
      model = SimpleClassifier.train(data)

      if Process.whereis(Brain.ML.IntentClassifierSimple) do
        GenServer.call(Brain.ML.IntentClassifierSimple, {:load_trained_model, model}, 120_000)
      end
    end

    :ok
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train intent classifier: #{inspect(e)}")
      :ok
  end

  @doc """
  Trains a sentiment classifier from test fixture data (or custom data)
  and loads it into the SentimentClassifierSimple GenServer.
  """
  def train_sentiment_classifier(custom_data \\ nil) do
    data = custom_data || load_sentiment_fixture()

    if data != [] do
      model = SimpleClassifier.train(data)

      if Process.whereis(Brain.ML.SentimentClassifierSimple) do
        GenServer.call(Brain.ML.SentimentClassifierSimple, {:load_trained_model, model}, 5_000)
      end
    end

    :ok
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train sentiment classifier: #{inspect(e)}")
      :ok
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

    if models != %{} and Process.whereis(Brain.ML.MicroClassifiers) do
      GenServer.call(Brain.ML.MicroClassifiers, {:load_trained_models, models}, 5_000)
    end

    :ok
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train micro classifiers: #{inspect(e)}")
      :ok
  end

  @doc """
  Trains a POS tagger from gold standard POS-annotated data and saves
  the model to the test models path so POSTagger.load_model() works.
  """
  def train_pos_tagger do
    alias Brain.ML.POSTagger

    sequences = load_pos_sequences_from_gold_standard()

    if sequences != [] do
      case POSTagger.train(sequences) do
        {:ok, model} ->
          models_path = Application.get_env(:brain, :ml)[:models_path]

          if models_path do
            save_path = Path.join(models_path, "pos_model.term")
            File.mkdir_p!(Path.dirname(save_path))
            POSTagger.save_model(model, save_path)
          end

        {:error, reason} ->
          Logger.warning("ModelFactory: failed to train POS tagger: #{inspect(reason)}")
      end
    end

    :ok
  rescue
    e ->
      Logger.warning("ModelFactory: failed to train POS tagger: #{inspect(e)}")
      :ok
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
