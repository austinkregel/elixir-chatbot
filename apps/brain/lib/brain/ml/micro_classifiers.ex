defmodule Brain.ML.MicroClassifiers do
  @moduledoc """
  GenServer that hosts multiple small TF-IDF classifiers for lightweight
  NLP classification tasks.

  Each classifier is trained from a JSON data file in `data/classifiers/`
  and loaded at startup from `priv/ml_models/micro/`. If a trained model
  file doesn't exist, the classifier falls back to training from raw data.

  ## Available Classifiers

  - `:personal_question` - Detects personal questions about the bot
  - `:clarification_response` - Detects clarification/disambiguation responses
  - `:modal_directive` - Detects "can you / would you" directive patterns
  - `:fallback_response` - Detects generic fallback/error responses
  - `:goal_type` - Classifies research goal type (reasoning/sentiment/factual/general)
  - `:entity_type` - Infers entity type from entity name + category

  ## Usage

      MicroClassifiers.classify(:personal_question, "what is your name")
      # => {:ok, "personal", 0.87}

      MicroClassifiers.classify(:fallback_response, "The weather is sunny.")
      # => {:ok, "not_fallback", 0.92}
  """

  use GenServer
  require Logger

  alias Brain.ML.SimpleClassifier

  @classifier_names [
    :personal_question,
    :clarification_response,
    :modal_directive,
    :fallback_response,
    :goal_type,
    :entity_type
  ]

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Classify text using the named micro-classifier."
  @spec classify(atom(), String.t()) :: {:ok, String.t(), float()} | {:error, :not_loaded}
  def classify(name, text) do
    if ready?() do
      GenServer.call(__MODULE__, {:classify, name, text}, 5_000)
    else
      {:error, :not_loaded}
    end
  end

  @doc "Check if the MicroClassifiers server is ready."
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Hot-reload all models from disk or retrain from data files."
  def reload do
    GenServer.call(__MODULE__, :reload, 30_000)
  end

  @doc "Get status of all loaded classifiers."
  def status do
    if ready?() do
      GenServer.call(__MODULE__, :status, 5_000)
    else
      %{ready: false, classifiers: %{}}
    end
  end

  # --- Server Callbacks ---

  @impl true
  def init(_opts) do
    models = load_all_models()

    {:ok, %{models: models}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call({:classify, name, text}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      model ->
        case SimpleClassifier.classify(text, model) do
          {:ok, label, score, _details} ->
            {:reply, {:ok, label, score}, state}

          _ ->
            {:reply, {:error, :classification_failed}, state}
        end
    end
  end

  @impl true
  def handle_call(:reload, _from, _state) do
    models = load_all_models()
    {:reply, :ok, %{models: models}}
  end

  @impl true
  def handle_call({:load_trained_models, models_map}, _from, state) when is_map(models_map) do
    merged = Map.merge(state.models, models_map)
    {:reply, :ok, %{state | models: merged}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    status =
      Enum.into(@classifier_names, %{}, fn name ->
        loaded = Map.has_key?(state.models, name)
        {name, %{loaded: loaded}}
      end)

    {:reply, %{ready: true, classifiers: status}, state}
  end

  @impl true
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # --- Private ---

  defp load_all_models do
    Enum.reduce(@classifier_names, %{}, fn name, acc ->
      case load_model(name) do
        {:ok, model} ->
          Map.put(acc, name, model)

        {:error, reason} ->
          Logger.warning("MicroClassifiers: failed to load #{name}: #{inspect(reason)}")
          acc
      end
    end)
  end

  defp load_model(name) do
    model_path = model_file_path(name)

    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          _ -> train_from_data(name)
        end

      {:error, _} ->
        train_from_data(name)
    end
  end

  defp train_from_data(name) do
    data_path = data_file_path(name)

    case File.read(data_path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, entries} ->
            training_data =
              Enum.map(entries, fn entry ->
                {Map.get(entry, "text", ""), Map.get(entry, "label", "unknown")}
              end)

            if length(training_data) > 0 do
              Logger.warning("MicroClassifiers: training #{name} on-the-fly (no pre-trained model found at #{model_file_path(name)})")
              model = SimpleClassifier.train(training_data)

              model_path = model_file_path(name)
              File.mkdir_p!(Path.dirname(model_path))
              File.write!(model_path, :erlang.term_to_binary(model))

              Logger.warning("MicroClassifiers: trained and cached #{name} (#{length(training_data)} examples) to #{model_path}")
              {:ok, model}
            else
              {:error, :empty_training_data}
            end

          {:error, reason} ->
            {:error, {:json_decode, reason}}
        end

      {:error, _} ->
        {:error, :no_data_file}
    end
  end

  defp model_file_path(name) do
    Path.join([Brain.priv_path("ml_models/micro"), "#{name}.term"])
  end

  defp data_file_path(name) do
    # In umbrella apps, the data/ dir is at the project root.
    # Resolve priv_dir symlink to find the real source path, then navigate to project root.
    priv_dir = :code.priv_dir(:brain) |> to_string()

    umbrella_root =
      case File.read_link(priv_dir) do
        {:ok, link_target} ->
          # Symlink: resolve relative to the parent of priv_dir
          parent = Path.dirname(priv_dir)
          real_priv = Path.join(parent, link_target) |> Path.expand()
          # real_priv is now e.g. /-/chat_bot/apps/brain/priv, go up 3
          Path.join(real_priv, "../../..") |> Path.expand()

        {:error, _} ->
          # No symlink: priv_dir is e.g. /-/chat_bot/_build/env/lib/brain/priv, go up 5
          Path.join(priv_dir, "../../../../..") |> Path.expand()
      end

    Path.join(umbrella_root, "data/classifiers/#{name}.json")
  end
end
