defmodule World.ModelRegistry do
  @moduledoc "Central registry for per-world ML models.\n\nTracks which worlds' models are loaded and which world is active:\n- Embedder vocabulary/IDF weights (`World.Embedder`, read from the world's own copy)\n\nA world's entity model is not here: `Brain.ML.EntityModel` owns it. POS tagging\nis not world-scoped: every world uses the shared `Brain.ML.POSTagger`.\n\nActivating a world loads its models from its own copies. A world with no copy\nraises naming the world and the seed command; there is no fallback to another\nworld's models or to the trained ones. No world is active until one is activated.\n\n## PubSub Events\n\nSubscribes to:\n- `world_context:global` - `{:world_changed, user_id, world_id}`, sent by the web interface when a user selects a world - Triggers model switch\n\nBroadcasts on `world_models:status`:\n- `{:world_models_loading, world_id}` - Models are being loaded\n- `{:world_models_loaded, world_id, status}` - Models finished loading\n"

  alias Phoenix.PubSub
  alias World.Persistence
  use GenServer
  require Logger

  alias World.Embedder, as: WorldEmbedder
  alias Brain.ML.Gazetteer

  @pubsub Brain.PubSub
  @model_types [:embedder]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Activates a world's models, loading them if necessary.\nThis is the primary entry point for world switching.\n\nReturns {:ok, status}; raises when the world has no copy of a model.\n"
  def activate_world(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:activate_world, world_id}, 60_000)
  end

  @doc "Returns the currently active world ID, or nil when no world has been activated.\n"
  def get_active_world do
    GenServer.call(__MODULE__, :get_active_world, 5_000)
  end

  @doc "Gets a specific model for a loaded world.\nReturns {:ok, model} or {:error, :not_loaded}\n"
  def get_model(world_id, model_type) when is_binary(world_id) and model_type in @model_types do
    GenServer.call(__MODULE__, {:get_model, world_id, model_type}, 5_000)
  end

  @doc "Gets all models for a world.\nReturns {:ok, models_map} or {:error, reason}\n"
  def get_world_models(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:get_world_models, world_id}, 5_000)
  end

  @doc "Gets the active world's models.\nReturns {:ok, models_map}, {:error, :no_active_world} or {:error, :not_loaded}\n"
  def get_active_models do
    GenServer.call(__MODULE__, :get_active_models, 5_000)
  end

  @doc "Reloads models for a world from its copies.\n"
  def reload_world_models(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:reload_world_models, world_id}, 60_000)
  end

  @doc "Unloads models for a world to free memory.\n"
  def unload_world(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:unload_world, world_id}, 30_000)
  end

  @doc "Returns the status of models for a world.\n"
  def get_world_status(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:get_world_status, world_id}, 5_000)
  end

  @doc "Returns the status of all loaded worlds.\n"
  def get_all_status do
    GenServer.call(__MODULE__, :get_all_status, 5_000)
  end

  @doc "Checks if the registry is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Returns the path where a world's models are stored.\n"
  def models_dir(world_id) do
    base = Persistence.base_path()
    Path.join([base, world_id, "models"])
  end

  @doc """
  Checks if a world holds its own embedder: a copy on disk for a persistent
  world, or one held in memory for an ephemeral world.
  """
  def world_has_models?(world_id) do
    File.regular?(WorldEmbedder.path(world_id)) or WorldEmbedder.ready?(world_id)
  end

  @impl true
  def init(_opts) do
    PubSub.subscribe(@pubsub, "world_context:global")

    state = %{
      active_world_id: nil,
      models: %{},
      ready: true
    }

    {:ok, state}
  end

  @impl true
  def handle_info({:world_changed, _session_id, world_id}, state) do
    Logger.info("WorldModelRegistry: World change detected", %{world_id: world_id})
    {:noreply, do_activate_world(world_id, state)}
  end

  @impl true
  def handle_info({:world_changed, world_id}, state) do
    Logger.info("WorldModelRegistry: World change detected (simple)", %{world_id: world_id})
    {:noreply, do_activate_world(world_id, state)}
  end

  @impl true
  def handle_call({:activate_world, world_id}, _from, state) do
    new_state = do_activate_world(world_id, state)
    {:reply, {:ok, get_world_status_from_state(world_id, new_state)}, new_state}
  end

  @impl true
  def handle_call(:get_active_world, _from, state) do
    {:reply, state.active_world_id, state}
  end

  @impl true
  def handle_call({:get_model, world_id, model_type}, _from, state) do
    result =
      case Map.get(state.models, world_id) do
        nil -> {:error, :not_loaded}
        world_models -> {:ok, Map.fetch!(world_models, model_type)}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_world_models, world_id}, _from, state) do
    result =
      case Map.get(state.models, world_id) do
        nil -> {:error, :not_loaded}
        models -> {:ok, models}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_active_models, _from, %{active_world_id: nil} = state) do
    {:reply, {:error, :no_active_world}, state}
  end

  def handle_call(:get_active_models, _from, state) do
    result =
      case Map.get(state.models, state.active_world_id) do
        nil -> {:error, :not_loaded}
        models -> {:ok, models}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:reload_world_models, world_id}, _from, state) do
    broadcast_models_loading(world_id)
    models = do_load_world_models(world_id)
    new_state = %{state | models: Map.put(state.models, world_id, models)}
    broadcast_models_loaded(world_id, models)
    {:reply, {:ok, models}, new_state}
  end

  @impl true
  def handle_call({:unload_world, world_id}, _from, state) do
    new_state = %{state | models: Map.delete(state.models, world_id)}
    Logger.info("WorldModelRegistry: Unloaded world models", %{world_id: world_id})
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:get_world_status, world_id}, _from, state) do
    {:reply, get_world_status_from_state(world_id, state), state}
  end

  @impl true
  def handle_call(:get_all_status, _from, state) do
    status = %{
      active_world_id: state.active_world_id,
      loaded_worlds: Map.keys(state.models),
      ready: state.ready,
      worlds:
        Enum.map(state.models, fn {world_id, _models} ->
          {world_id, get_world_status_from_state(world_id, state)}
        end)
        |> Map.new()
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, state.ready, state}
  end

  defp do_activate_world(world_id, state) do
    broadcast_models_loading(world_id)

    models =
      case Map.fetch(state.models, world_id) do
        {:ok, models} -> models
        :error -> do_load_world_models(world_id)
      end

    broadcast_models_loaded(world_id, models)

    %{state | active_world_id: world_id, models: Map.put(state.models, world_id, models)}
  end

  # Raises when the world has no copy of a model (`World.Embedder.loaded!/1`).
  defp do_load_world_models(world_id) do
    Logger.debug("Loading models for world: #{world_id}")

    models = %{embedder: WorldEmbedder.loaded!(world_id)}
    preload_gazetteer_overlay(world_id)

    models
  end

  defp preload_gazetteer_overlay(world_id) do
    overlay_path = Path.join(Persistence.world_path(world_id), "gazetteer_overlay.json")

    if File.exists?(overlay_path) do
      try do
        data = File.read!(overlay_path) |> Jason.decode!(keys: :atoms)

        overlay_list =
          case data do
            list when is_list(list) ->
              list
              |> Enum.reject(fn entry -> entry[:key] == "_meta" end)
              |> Enum.map(fn entry ->
                {to_string(entry[:key]), entry[:info] || %{}}
              end)

            map when is_map(map) ->
              Enum.map(map, fn {key, value} ->
                {to_string(key), value}
              end)

            _ ->
              []
          end

        if overlay_list != [] do
          Gazetteer.restore_world_overlay(world_id, overlay_list)
          Logger.debug("Preloaded gazetteer overlay for world: #{world_id}")
        end
      rescue
        e ->
          Logger.warning("Failed to preload gazetteer overlay: #{inspect(e)}")
      end
    end
  end

  defp get_world_status_from_state(world_id, state) do
    models = Map.get(state.models, world_id, %{})

    %{
      world_id: world_id,
      is_active: state.active_world_id == world_id,
      is_loaded: Map.has_key?(state.models, world_id),
      has_embedder: models[:embedder] != nil,
      embedder_vocab_size: get_embedder_vocab_size(models[:embedder])
    }
  end

  defp get_embedder_vocab_size(nil) do
    0
  end

  defp get_embedder_vocab_size(%{vocabulary: vocabulary}) do
    map_size(vocabulary)
  end

  defp broadcast_models_loading(world_id) do
    PubSub.broadcast(@pubsub, "world_models:status", {:world_models_loading, world_id})
  end

  defp broadcast_models_loaded(world_id, models) do
    status = %{
      embedder: models[:embedder] != nil
    }

    PubSub.broadcast(
      @pubsub,
      "world_models:status",
      {:world_models_loaded, world_id, status}
    )
  end
end
