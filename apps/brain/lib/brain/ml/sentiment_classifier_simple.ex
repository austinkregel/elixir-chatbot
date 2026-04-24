defmodule Brain.ML.SentimentClassifierSimple do
  @moduledoc """
  Sentiment classifier using TF-IDF and nearest centroid classification.

  Mirrors the GenServer+SimpleClassifier pattern but for sentiment labels
  (positive, negative, neutral). Trained from the sentiment gold standard
  via `Brain.ML.Trainer.train_sentiment_classifier/2`.

  ## Usage

      SentimentClassifierSimple.classify("I love this product!")
      # => {:ok, %{label: :positive, confidence: 0.87}}

      SentimentClassifierSimple.ready?()
      # => true
  """

  alias Brain.ML.SimpleClassifier
  use GenServer
  require Logger

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Returns true if the sentiment classifier model is loaded and ready.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Classifies the sentiment of the given text.

  Returns `{:ok, %{label: atom, confidence: float}}` or `{:error, reason}`.
  """
  def classify(text, name \\ __MODULE__) do
    GenServer.call(name, {:classify, text}, 5000)
  end

  @doc """
  Reloads the model from disk.
  """
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 10_000)
  end

  @doc """
  Loads (or trains on-the-fly from gold standard) the sentiment model.

  If a serialized model file exists on disk, loads it directly.
  Otherwise, trains a fresh model from the sentiment gold standard data
  and loads it into the GenServer.

  Training happens outside the GenServer process to avoid blocking it.

  Returns `{:ok, :loaded}` or `{:error, reason}`.
  """
  def load_models(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)

    # First, try telling the GenServer to reload from disk
    case GenServer.call(name, :reload, 10_000) do
      :ok ->
        {:ok, :loaded}

      {:error, _} ->
        # No model on disk — train outside the GenServer process to avoid blocking
        case train_from_gold_standard() do
          {:ok, model} ->
            GenServer.call(name, {:load_trained_model, model}, 10_000)

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  @doc false
  def train_from_gold_standard do
    gold = Brain.ML.EvaluationStore.load_gold_standard("sentiment")

    training_data =
      gold
      |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["sentiment"]) end)
      |> Enum.map(fn ex -> {ex["text"], ex["sentiment"]} end)

    if training_data == [] do
      {:error, :no_training_data}
    else
      model = SimpleClassifier.train(training_data)
      {:ok, model}
    end
  end

  # --- GenServer callbacks ---

  @impl true
  def init(_opts) do
    send(self(), :load_model)
    {:ok, %{model: nil}}
  end

  @impl true
  def handle_info(:load_model, state) do
    case do_load_model() do
      {:ok, model} ->
        Logger.info("SentimentClassifier: model loaded from disk",
          vocab_size: map_size(model.vocabulary),
          label_count: map_size(model.label_centroids)
        )

        {:noreply, %{state | model: model}}

      {:error, _disk_reason} ->
        Logger.error(
          "SentimentClassifier: no model on disk. " <>
            "Run `mix train` to train the sentiment classifier."
        )

        {:noreply, state}
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.model != nil, state}
  end

  @impl true
  def handle_call({:classify, _text}, _from, %{model: nil} = state) do
    {:reply, {:error, :not_trained}, state}
  end

  @impl true
  def handle_call({:classify, text}, _from, %{model: model} = state) do
    case SimpleClassifier.classify(text, model) do
      {:ok, label, score, _details} ->
        {:reply, {:ok, %{label: String.to_atom(label), confidence: score}}, state}

      error ->
        {:reply, {:error, error}, state}
    end
  end

  @impl true
  def handle_call(:reload, _from, state) do
    case do_load_model() do
      {:ok, model} ->
        Logger.info("SentimentClassifier: model reloaded")
        {:reply, :ok, %{state | model: model}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:load_trained_model, model}, _from, state) do
    Logger.info("SentimentClassifier: model loaded from gold standard training",
      vocab_size: map_size(model.vocabulary),
      label_count: map_size(model.label_centroids)
    )

    {:reply, {:ok, :loaded}, %{state | model: model}}
  end

  defp do_load_model do
    path = model_path()

    if File.exists?(path) do
      try do
        model = path |> File.read!() |> :erlang.binary_to_term()
        {:ok, model}
      rescue
        e ->
          Logger.warning("SentimentClassifier: failed to load model: #{inspect(e)}")
          {:error, :load_failed}
      end
    else
      {:error, :model_not_found}
    end
  end

  @doc false
  def model_path do
    models_path =
      Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")

    Path.join(models_path, "sentiment_classifier.term")
  end
end
