defmodule Brain.ML.IntentClassifierSimple do
  @moduledoc "Intent classifier using simple TF-IDF and nearest centroid classification.\n\n## World Scoping\n\nSupports world-specific models with inheritance fallback:\n1. Try world-specific model (priv/training_worlds/{world_id}/models/classifier.term)\n2. Fall back to default model (priv/ml_models/classifier.term)\n\nWorld-specific models can be trained using:\n`mix train_models --world star_trek`\n\n## Integration with WorldModelRegistry\n\nThis classifier subscribes to `world_models:status` PubSub events to:\n- Reload models when a world's models are updated\n- Unload models when requested\n"

  # World.Persistence is in a sibling umbrella app that depends on :brain.
  # It's available at runtime but not at compile time.
  @compile {:no_warn_undefined, World.Persistence}

  alias Brain.ML.SimpleClassifier
  alias World.Persistence
  alias Phoenix.PubSub
  use GenServer
  require Logger

  @default_world_id "default"
  @pubsub Brain.PubSub

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
  Incrementally updates the classifier model for a world with new training examples.

  `new_examples` is a list of `{text, label}` tuples.
  Returns `{:ok, incremental_count}` or `{:error, reason}`.
  Triggers a full retrain after 200 incremental updates.

  ## Options
    - `:server` - The server to call (default: `#{__MODULE__}`)
  """
  def incremental_update(new_examples, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(server, {:incremental_update, world_id, new_examples})
  end

  def load_models(opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    GenServer.call(server, {:load_model, world_id})
  end

  @doc """
  Returns true if the default classifier model is loaded and ready.

  This is a convenience function for readiness checks that follows
  the standard `ready?/0` pattern used across the codebase.
  """
  def ready? do
    try do
      is_loaded?()
    catch
      :exit, _ -> false
    end
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
    - `:with_details` - If true, returns detailed results with top-k and margin (default: false)
    - `:top_k` - Number of top intents to return when with_details is true (default: 5)

  Falls back through the world inheritance chain if the world's
  model is not available.
  """
  def classify(text, opts \\ []) do
    server = Keyword.get(opts, :server, __MODULE__)
    world_id = Keyword.get(opts, :world_id, @default_world_id)
    with_details = Keyword.get(opts, :with_details, false)
    top_k = Keyword.get(opts, :top_k, 5)
    GenServer.call(server, {:classify, text, world_id, with_details, top_k})
  end

  @impl true
  def init(_opts) do
    PubSub.subscribe(@pubsub, "world_models:status")

    state = %{
      models: %{},
      loading: MapSet.new()
    }

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

  @impl true
  def handle_info({:world_models_loaded, world_id, _status}, state) do
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
    {:noreply, state}
  end

  @impl true
  def handle_info({:world_models_error, _world_id, _reason}, state) do
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
  def handle_call({:load_trained_model, model}, _from, state) do
    new_models = Map.put(state.models, @default_world_id, model)
    {:reply, :ok, %{state | models: new_models}}
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
  def handle_call({:incremental_update, world_id, new_examples}, _from, state) do
    case Map.get(state.models, world_id) do
      nil ->
        {:reply, {:error, :no_model_loaded}, state}

      model ->
        incremental_count = Map.get(model, :incremental_update_count, 0)

        {updated_model, new_count} =
          SimpleClassifier.update_model(model, new_examples, incremental_count)

        updated_model = Map.put(updated_model, :incremental_update_count, new_count)

        # Full retrain after 200 incremental updates
        if new_count >= 200 do
          Logger.info("IntentClassifier: incremental drift guard triggered, scheduling full retrain",
            world_id: world_id,
            incremental_updates: new_count
          )

          Phoenix.PubSub.broadcast(
            Brain.PubSub,
            "training:requests",
            {:retrain_requested, world_id, :drift_guard}
          )
        end

        new_models = Map.put(state.models, world_id, updated_model)

        Logger.debug("IntentClassifier: incremental update applied",
          world_id: world_id,
          examples: length(new_examples),
          incremental_count: new_count
        )

        {:reply, {:ok, new_count}, %{state | models: new_models}}
    end
  end

  @impl true
  def handle_call({:classify, text, world_id, with_details, top_k}, _from, state) do
    {model, new_state} = get_or_load_model(world_id, state)

    result =
      case model do
        nil ->
          case Map.get(state.models, @default_world_id) do
            nil -> {:error, :no_model_available}
            default_model -> do_classify(text, default_model, with_details, top_k)
          end

        model ->
          do_classify(text, model, with_details, top_k)
      end

    {:reply, result, new_state}
  end

  defp get_or_load_model(world_id, state) do
    case Map.get(state.models, world_id) do
      nil ->
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
    Brain.ML.ModelStore.ensure_local("classifier.term", model_path)

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
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")
    Path.join(models_path, "classifier.term")
  end

  defp get_model_path(world_id) do
    world_path = Persistence.world_path(world_id)
    Path.join([world_path, "models", "classifier.term"])
  end

  defp do_classify(text, model, with_details, top_k) do
    if with_details do
      case SimpleClassifier.classify_with_details(text, model, top_k: top_k) do
        {:ok, label, score, details} ->
          {:ok,
           %{
             intent: label,
             confidence: score,
             second_score: details.second_score,
             margin: details.margin,
             top_k: details.top_k
           }}

        error ->
          error
      end
    else
      case SimpleClassifier.classify(text, model) do
        {:ok, label, score, _details} ->
          {:ok, %{intent: label, confidence: score}}

        error ->
          error
      end
    end
  rescue
    e ->
      Logger.error("Classification failed", %{error: inspect(e)})
      {:error, "Classification failed: #{inspect(e)}"}
  end
end
