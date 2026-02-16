defmodule Brain.Test.ModelFactory do
  @moduledoc """
  Trains and loads minimal ML models for test use.

  Instead of relying on pre-committed .term files, this module trains
  small classifiers from test fixture data at test startup. The datasets
  are intentionally small (8-30 examples) so training completes in
  milliseconds.

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
    train_intent_classifier()
    train_sentiment_classifier()
    train_micro_classifiers()
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
        GenServer.call(Brain.ML.IntentClassifierSimple, {:load_trained_model, model}, 5_000)
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

  # -- Private --

  defp load_intent_fixture do
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
end
