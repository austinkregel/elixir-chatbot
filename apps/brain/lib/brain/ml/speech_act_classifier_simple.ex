defmodule Brain.ML.SpeechActClassifierSimple do
  @moduledoc """
  Speech act classifier using TF-IDF and nearest centroid classification.

  Mirrors the pattern of `SentimentClassifierSimple` but for speech act labels
  (assertive, directive, commissive, expressive). Trained from the speech act
  gold standard via the gold standard data.

  ## Usage

      SpeechActClassifierSimple.classify("I promise to help you")
      # => {:ok, %{label: :commissive, confidence: 0.72}}

      SpeechActClassifierSimple.ready?()
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
  Returns true if the speech act classifier model is loaded and ready.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Classifies the speech act of the given text.

  Returns `{:ok, %{label: atom, confidence: float}}` or `{:error, reason}`.
  """
  def classify(text, name \\ __MODULE__) do
    try do
      GenServer.call(name, {:classify, text}, 5000)
    catch
      :exit, _ -> {:error, :not_available}
    end
  end

  @doc """
  Reloads the model from disk.
  """
  def reload(name \\ __MODULE__) do
    GenServer.call(name, :reload, 10_000)
  end

  @doc """
  Loads (or trains on-the-fly from gold standard) the speech act model.

  If a serialized model file exists on disk, loads it directly.
  Otherwise, trains a fresh model from the speech act gold standard data
  and loads it into the GenServer.
  """
  def load_models(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)

    case GenServer.call(name, :reload, 10_000) do
      :ok ->
        {:ok, :loaded}

      {:error, _} ->
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
    gold = Brain.ML.EvaluationStore.load_gold_standard("speech_act")

    training_data =
      gold
      |> Enum.filter(fn ex -> is_binary(ex["text"]) and is_binary(ex["speech_act"]) end)
      |> Enum.map(fn ex -> {ex["text"], ex["speech_act"]} end)

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
        Logger.info("SpeechActClassifierSimple: model loaded from disk",
          vocab_size: map_size(model.vocabulary),
          label_count: map_size(model.label_centroids)
        )

        {:noreply, %{state | model: model}}

      {:error, _disk_reason} ->
        Logger.error(
          "SpeechActClassifierSimple: no model on disk. " <>
            "Run `mix train` to train the speech act classifier."
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
        Logger.info("SpeechActClassifierSimple: model reloaded")
        {:reply, :ok, %{state | model: model}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:load_trained_model, model}, _from, state) do
    Logger.info("SpeechActClassifierSimple: model loaded from gold standard training",
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
          Logger.warning("SpeechActClassifierSimple: failed to load model: #{inspect(e)}")
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

    Path.join(models_path, "speech_act_classifier.term")
  end
end
