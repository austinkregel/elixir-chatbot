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
  - `:user_fact_type` - Classifies whether entity/fact is user-specific or general knowledge
  - `:directed_at_bot` - Detects whether text is addressed to the bot

  ## Usage

      MicroClassifiers.classify(:personal_question, "what is your name")
      # => {:ok, "personal", 0.87}

      MicroClassifiers.classify(:fallback_response, "The weather is sunny.")
      # => {:ok, "not_fallback", 0.92}

      # Incremental learning from usage
      MicroClassifiers.incremental_update(:user_fact_type, [
        {"my favorite color blue", "user_specific"},
        {"photosynthesis plants energy", "general_knowledge"}
      ])
      # => {:ok, 1}
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
    :entity_type,
    :user_fact_type,
    :directed_at_bot,
    :event_argument_role
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

  @doc "Incrementally update a named micro-classifier with new examples.\n\nExamples should be a list of `{text, label}` tuples. Uses\n`SimpleClassifier.update_model/3` to blend new examples into\nthe existing model without a full retrain. After 200 incremental\nupdates, the model is persisted to disk.\n\nReturns `{:ok, incremental_count}` or `{:error, reason}`.\n"
  @spec incremental_update(atom(), [{String.t(), String.t()}]) ::
          {:ok, non_neg_integer()} | {:error, atom()}
  def incremental_update(name, examples) when is_atom(name) and is_list(examples) do
    if ready?() do
      GenServer.call(__MODULE__, {:incremental_update, name, examples}, 10_000)
    else
      {:error, :not_loaded}
    end
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
  def handle_call({:incremental_update, name, examples}, _from, state) do
    case Map.get(state.models, name) do
      nil ->
        {:reply, {:error, :not_loaded}, state}

      model ->
        incremental_count = Map.get(model, :incremental_update_count, 0)
        {updated_model, new_count} = SimpleClassifier.update_model(model, examples, incremental_count)
        updated_model = Map.put(updated_model, :incremental_update_count, new_count)

        if rem(new_count, 200) == 0 and new_count > 0 do
          persist_model(name, updated_model)
        end

        Logger.debug("MicroClassifiers: incremental update for #{name}",
          examples: length(examples),
          incremental_count: new_count
        )

        {:reply, {:ok, new_count}, %{state | models: Map.put(state.models, name, updated_model)}}
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
        case Map.get(state.models, name) do
          nil ->
            {name, %{loaded: false, incremental_updates: 0}}

          model ->
            count = Map.get(model, :incremental_update_count, 0)
            {name, %{loaded: true, incremental_updates: count}}
        end
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
    Brain.ML.ModelStore.ensure_local("micro/#{name}.term", model_path)

    case File.read(model_path) do
      {:ok, binary} ->
        try do
          model = :erlang.binary_to_term(binary)
          {:ok, model}
        rescue
          _ ->
            Logger.error("MicroClassifiers: corrupt model file for #{name}. Run `mix train_micro` to retrain.")
            {:error, :corrupted_model}
        end

      {:error, _} ->
        Logger.error("MicroClassifiers: no model file for #{name} at #{model_file_path(name)}. Run `mix train_micro` to train.")
        {:error, :no_model_file}
    end
  end

  defp persist_model(name, model) do
    path = model_file_path(name)
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, :erlang.term_to_binary(model))
    Logger.info("MicroClassifiers: persisted #{name} to #{path}")
  rescue
    e -> Logger.warning("MicroClassifiers: failed to persist #{name}: #{inspect(e)}")
  end

  defp model_file_path(name) do
    base =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Brain.priv_path("ml_models")
        path -> path
      end

    Path.join([base, "micro", "#{name}.term"])
  end

end
