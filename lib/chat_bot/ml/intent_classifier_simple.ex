defmodule ChatBot.ML.IntentClassifierSimple do
  @moduledoc """
  Intent classifier using simple TF-IDF and nearest centroid classification.

  ## World Scoping

  Supports world-specific models with inheritance fallback:
  1. Try world-specific model (priv/training_worlds/{world_id}/models/classifier.term)
  2. Fall back to default model (priv/ml_models/classifier.term)

  World-specific models can be trained using:
  `mix train_models --world star_trek`

  ## Integration with WorldModelRegistry

  This classifier subscribes to `world_models:status` PubSub events to:
  - Reload models when a world's models are updated
  - Unload models when requested
  """

  use GenServer
  require Logger

  @default_world_id "default"
  @pubsub ChatBot.PubSub

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the intent classifier.

  ## Options
    - `:name` - The name to register the GenServer under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Loads the default classifier model.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def load_models(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(server, {:load_model, world_id})
  end

  @doc """
  Returns true if the default classifier model is loaded.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def is_loaded?(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(server, {:is_loaded, world_id})
  end

  @doc """
  Unloads the classifier model for a specific world to free memory.
  Cannot unload the default world model.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def unload_world(world_id, opts \\ []) when is_binary(world_id) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, {:unload_world, world_id})
  end

  @doc """
  Returns status of all loaded models.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def get_status(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    GenServer.call(server, :get_status)
  end

  @doc """
  Classifies text using the default world's model.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
    - `:world_id` - The world whose model to use (default: "default")

  Falls back through the world inheritance chain if the world's
  model is not available.
  """
  def classify(text, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(server, {:classify, text, world_id})
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Subscribe to world model status events
    Phoenix.PubSub.subscribe(@pubsub, "world_models:status")

    # Models are loaded on-demand, keyed by world_id
    state = %{
      models: %{},
      loading: MapSet.new()
    }

    # Try to load default model at startup
    send(self(), {:load_default})

    {:ok, state}
  end

  @impl true
  def handle_info({:load_default}, state) do
    case do_load_model(@default_world_id) do
      {:ok, model} ->
        {:noreply, %{state | models: Map.put(state.models, @default_world_id, model)}}

      {:error, _} ->
        {:noreply, state}
    end
  end

  # Handle world model reload requests from WorldModelRegistry
  @impl true
  def handle_info({:world_models_loaded, world_id, _status}, state) do
    # Reload this world's model if it was already loaded
    if Map.has_key?(state.models, world_id) do
      Logger.debug("IntentClassifier: Reloading model for world #{world_id}")

      case do_load_model(world_id) do
        {:ok, model} ->
          {:noreply, %{state | models: Map.put(state.models, world_id, model)}}

        {:error, _} ->
          {:noreply, state}
      end
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:world_models_loading, _world_id}, state) do
    # Could show loading state, but for now just acknowledge
    {:noreply, state}
  end

  @impl true
  def handle_info({:world_models_error, _world_id, _reason}, state) do
    # Log but continue operating
    {:noreply, state}
  end

  @impl true
  def handle_call({:load_model, world_id}, _from, state) do
    case do_load_model(world_id) do
      {:ok, model} ->
        new_models = Map.put(state.models, world_id, model)
        {:reply, {:ok, model}, %{state | models: new_models}}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:is_loaded, world_id}, _from, state) do
    loaded = Map.has_key?(state.models, world_id)
    {:reply, loaded, state}
  end

  @impl true
  def handle_call({:unload_world, world_id}, _from, state) do
    if world_id == @default_world_id do
      {:reply, {:error, :cannot_unload_default}, state}
    else
      new_state = %{state | models: Map.delete(state.models, world_id)}
      Logger.info("IntentClassifier: Unloaded model for world #{world_id}")
      {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = %{
      loaded_worlds: Map.keys(state.models),
      loading: MapSet.to_list(state.loading),
      models:
        Enum.map(state.models, fn {world_id, model} ->
          {world_id,
           %{
             vocab_size: map_size(Map.get(model, :vocabulary, %{})),
             intent_count: map_size(Map.get(model, :intent_centroids, %{}))
           }}
        end)
        |> Map.new()
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:classify, text, world_id}, _from, state) do
    # Try to get model for this world, or load it
    {model, new_state} = get_or_load_model(world_id, state)

    result =
      case model do
        nil ->
          # Try fallback to default
          case Map.get(state.models, @default_world_id) do
            nil -> {:error, :no_model_available}
            default_model -> do_classify(text, default_model)
          end

        model ->
          do_classify(text, model)
      end

    {:reply, result, new_state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_or_load_model(world_id, state) do
    case Map.get(state.models, world_id) do
      nil ->
        # Try to load it
        case do_load_model(world_id) do
          {:ok, model} ->
            new_models = Map.put(state.models, world_id, model)
            {model, %{state | models: new_models}}

          {:error, _} ->
            {nil, state}
        end

      model ->
        {model, state}
    end
  end

  defp do_load_model(world_id) do
    model_path = get_model_path(world_id)

    if File.exists?(model_path) do
      try do
        model_binary = File.read!(model_path)
        model = :erlang.binary_to_term(model_binary)

        Logger.info("Classifier model loaded", %{
          world_id: world_id,
          vocab_size: map_size(model.vocabulary)
        })

        {:ok, model}
      rescue
        e ->
          Logger.warning("Failed to load classifier model", %{
            world_id: world_id,
            error: inspect(e)
          })

          {:error, :load_failed}
      end
    else
      # If world-specific model doesn't exist, try default
      if world_id != @default_world_id do
        Logger.debug("No world-specific model, using default", %{world_id: world_id})
        {:error, :not_found}
      else
        Logger.warning("Default classifier model not found", %{path: model_path})
        {:error, :model_not_found}
      end
    end
  end

  defp get_model_path(@default_world_id) do
    models_path = Application.get_env(:chat_bot, :ml)[:models_path] || "priv/ml_models"
    Path.join(models_path, "classifier.term")
  end

  defp get_model_path(world_id) do
    # World-specific model path - use WorldPersistence.world_path() for isolation
    world_path = ChatBot.Learning.WorldPersistence.world_path(world_id)
    Path.join([world_path, "models", "classifier.term"])
  end

  defp do_classify(text, model) do
    case ChatBot.ML.SimpleClassifier.classify(text, model) do
      {:ok, label, score} ->
        {:ok, %{intent: label, confidence: score}}

      error ->
        error
    end
  rescue
    e ->
      Logger.error("Classification failed", %{error: inspect(e)})
      {:error, "Classification failed: #{inspect(e)}"}
  end
end
