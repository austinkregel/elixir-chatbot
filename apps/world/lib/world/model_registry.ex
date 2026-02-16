defmodule World.ModelRegistry do
  @moduledoc "Central registry for per-world ML models.\n\nManages loading, caching, and switching of world-specific models:\n- Intent classifier\n- Embedder vocabulary/IDF weights\n- POS tagger\n- Entity model\n\nWhen a world is activated, all its models are loaded (or created from defaults).\nComponents can query this registry to get the active world's models.\n\n## PubSub Events\n\nSubscribes to:\n- `world_context:global` - `{:world_changed, session_id, world_id}` - Triggers model switch\n\nBroadcasts on `world_models:status`:\n- `{:world_models_loading, world_id}` - Models are being loaded\n- `{:world_models_loaded, world_id, status}` - Models finished loading\n- `{:world_models_error, world_id, reason}` - Loading failed\n"

  alias Phoenix.PubSub
  alias World.Persistence
  use GenServer
  require Logger

  alias World.Embedder, as: WorldEmbedder
  alias Brain.ML.Gazetteer

  @pubsub Brain.PubSub
  @default_world_id "default"
  @model_types [:classifier, :embedder, :pos_model, :entity_model]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Activates a world's models, loading them if necessary.\nThis is the primary entry point for world switching.\n\nReturns {:ok, status} or {:error, reason}\n"
  def activate_world(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:activate_world, world_id}, 60_000)
  end

  @doc "Returns the currently active world ID.\n"
  def get_active_world do
    GenServer.call(__MODULE__, :get_active_world, 5_000)
  end

  @doc "Gets a specific model for a world.\nReturns {:ok, model} or {:error, reason}\n"
  def get_model(world_id, model_type) when is_binary(world_id) and model_type in @model_types do
    GenServer.call(__MODULE__, {:get_model, world_id, model_type}, 5_000)
  end

  @doc "Gets all models for a world.\nReturns {:ok, models_map} or {:error, reason}\n"
  def get_world_models(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:get_world_models, world_id}, 5_000)
  end

  @doc "Gets the active world's models.\n"
  def get_active_models do
    GenServer.call(__MODULE__, :get_active_models, 5_000)
  end

  @doc "Reloads models for a world from disk.\n"
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

  @doc "Checks if the registry is ready (has loaded default world).\n"
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

  @doc "Returns the path for a specific model file.\n"
  def model_path(world_id, model_type) when model_type in @model_types do
    filename =
      case model_type do
        :classifier -> "classifier.term"
        :embedder -> "embedder.term"
        :pos_model -> "pos_model.term"
        :entity_model -> "entity_model.term"
      end

    Path.join(models_dir(world_id), filename)
  end

  @doc "Returns the path for the default (fallback) model.\n"
  def default_model_path(model_type) when model_type in @model_types do
    models_path = Application.get_env(:brain, :ml)[:models_path] || Brain.priv_path("ml_models")

    filename =
      case model_type do
        :classifier -> "classifier.term"
        :embedder -> "embedder.term"
        :pos_model -> "pos_model.term"
        :entity_model -> "entity_model.term"
      end

    Path.join(models_path, filename)
  end

  @doc "Checks if a world has trained models.\n"
  def world_has_models?(world_id) do
    File.exists?(model_path(world_id, :classifier)) or
      (world_id == @default_world_id and File.exists?(default_model_path(:classifier)))
  end

  @impl true
  def init(_opts) do
    PubSub.subscribe(@pubsub, "world_context:global")

    state = %{
      active_world_id: @default_world_id,
      models: %{},
      loading: MapSet.new(),
      ready: false
    }

    send(self(), :load_default_world)

    {:ok, state}
  end

  @impl true
  def handle_info(:load_default_world, state) do
    Logger.info("WorldModelRegistry: Loading default world models...")

    case do_load_world_models(@default_world_id) do
      {:ok, models} ->
        Logger.info("WorldModelRegistry: Default world models loaded", %{
          models: Map.keys(models) |> Enum.filter(&Map.get(models, &1))
        })

        new_state = %{
          state
          | models: Map.put(state.models, @default_world_id, models),
            ready: true
        }

        broadcast_models_loaded(@default_world_id, models)
        {:noreply, new_state}

      {:error, reason} ->
        Logger.warning(
          "WorldModelRegistry: Failed to load default world models: #{inspect(reason)}"
        )

        {:noreply, %{state | ready: true}}
    end
  end

  @impl true
  def handle_info({:world_changed, _session_id, world_id}, state) do
    Logger.info("WorldModelRegistry: World change detected", %{world_id: world_id})

    case do_activate_world(world_id, state) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:error, _reason, new_state} ->
        {:noreply, new_state}
    end
  end

  @impl true
  def handle_info({:world_changed, world_id}, state) do
    Logger.info("WorldModelRegistry: World change detected (simple)", %{world_id: world_id})

    case do_activate_world(world_id, state) do
      {:ok, new_state} ->
        {:noreply, new_state}

      {:error, _reason, new_state} ->
        {:noreply, new_state}
    end
  end

  @impl true
  def handle_call({:activate_world, world_id}, _from, state) do
    case do_activate_world(world_id, state) do
      {:ok, new_state} ->
        {:reply, {:ok, get_world_status_from_state(world_id, new_state)}, new_state}

      {:error, reason, new_state} ->
        {:reply, {:error, reason}, new_state}
    end
  end

  @impl true
  def handle_call(:get_active_world, _from, state) do
    {:reply, state.active_world_id, state}
  end

  @impl true
  def handle_call({:get_model, world_id, model_type}, _from, state) do
    result =
      case Map.get(state.models, world_id) do
        nil ->
          case Map.get(state.models, @default_world_id) do
            nil -> {:error, :no_models_loaded}
            default_models -> {:ok, Map.get(default_models, model_type)}
          end

        world_models ->
          case Map.get(world_models, model_type) do
            nil ->
              default_models = Map.get(state.models, @default_world_id, %{})
              {:ok, Map.get(default_models, model_type)}

            model ->
              {:ok, model}
          end
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
  def handle_call(:get_active_models, _from, state) do
    result =
      case Map.get(state.models, state.active_world_id) do
        nil ->
          case Map.get(state.models, @default_world_id) do
            nil -> {:error, :no_models_loaded}
            models -> {:ok, models}
          end

        models ->
          {:ok, models}
      end

    {:reply, result, state}
  end

  @impl true
  def handle_call({:reload_world_models, world_id}, _from, state) do
    broadcast_models_loading(world_id)

    case do_load_world_models(world_id) do
      {:ok, models} ->
        new_state = %{state | models: Map.put(state.models, world_id, models)}
        broadcast_models_loaded(world_id, models)
        {:reply, {:ok, models}, new_state}

      {:error, reason} ->
        broadcast_models_error(world_id, reason)
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call({:unload_world, world_id}, _from, state) do
    if world_id == @default_world_id do
      {:reply, {:error, :cannot_unload_default}, state}
    else
      new_state = %{state | models: Map.delete(state.models, world_id)}
      Logger.info("WorldModelRegistry: Unloaded world models", %{world_id: world_id})
      {:reply, :ok, new_state}
    end
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
      loading: MapSet.to_list(state.loading),
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
    if MapSet.member?(state.loading, world_id) do
      {:error, :already_loading, state}
    else
      state = %{state | loading: MapSet.put(state.loading, world_id)}
      broadcast_models_loading(world_id)

      if Map.has_key?(state.models, world_id) do
        new_state = %{
          state
          | active_world_id: world_id,
            loading: MapSet.delete(state.loading, world_id)
        }

        broadcast_models_loaded(world_id, Map.get(state.models, world_id))
        {:ok, new_state}
      else
        case do_load_world_models(world_id) do
          {:ok, models} ->
            new_state = %{
              state
              | active_world_id: world_id,
                models: Map.put(state.models, world_id, models),
                loading: MapSet.delete(state.loading, world_id)
            }

            broadcast_models_loaded(world_id, models)
            {:ok, new_state}

          {:error, reason} ->
            new_state = %{state | loading: MapSet.delete(state.loading, world_id)}
            broadcast_models_error(world_id, reason)
            new_state = %{new_state | active_world_id: world_id}
            {:error, reason, new_state}
        end
      end
    end
  end

  defp do_load_world_models(world_id) do
    Logger.debug("Loading models for world: #{world_id}")

    models = %{
      classifier: load_model_file(world_id, :classifier),
      embedder: load_embedder_model(world_id),
      pos_model: load_model_file(world_id, :pos_model),
      entity_model: load_model_file(world_id, :entity_model)
    }

    if Enum.all?(Map.values(models), &is_nil/1) do
      {:error, :no_models_found}
    else
      preload_gazetteer_overlay(world_id)
      ensure_world_embedder(world_id)

      {:ok, models}
    end
  end

  defp load_model_file(world_id, model_type) do
    world_path = model_path(world_id, model_type)

    cond do
      File.exists?(world_path) ->
        load_term_file(world_path)

      world_id != @default_world_id ->
        default_path = default_model_path(model_type)

        if File.exists?(default_path) do
          load_term_file(default_path)
        else
          nil
        end

      true ->
        default_path = default_model_path(model_type)

        if File.exists?(default_path) do
          load_term_file(default_path)
        else
          nil
        end
    end
  end

  defp load_term_file(path) do
    try do
      binary = File.read!(path)
      :erlang.binary_to_term(binary)
    rescue
      e ->
        Logger.warning("Failed to load model from #{path}: #{inspect(e)}")
        nil
    end
  end

  defp load_embedder_model(world_id) do
    world_path = model_path(world_id, :embedder)

    cond do
      File.exists?(world_path) ->
        load_term_file(world_path)

      true ->
        default_path = default_model_path(:embedder)

        if File.exists?(default_path) do
          load_term_file(default_path)
        else
          nil
        end
    end
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

  defp ensure_world_embedder(world_id) do
    case WorldEmbedder.get_status(world_id) do
      %{ready: true} ->
        :ok

      %{phase: :not_initialized} ->
        :ok

      _ ->
        :ok
    end
  end

  defp get_world_status_from_state(world_id, state) do
    models = Map.get(state.models, world_id, %{})

    %{
      world_id: world_id,
      is_active: state.active_world_id == world_id,
      is_loaded: Map.has_key?(state.models, world_id),
      is_loading: MapSet.member?(state.loading, world_id),
      has_classifier: models[:classifier] != nil,
      has_embedder: models[:embedder] != nil,
      has_pos_model: models[:pos_model] != nil,
      has_entity_model: models[:entity_model] != nil,
      classifier_vocab_size: get_vocab_size(models[:classifier]),
      embedder_vocab_size: get_embedder_vocab_size(models[:embedder])
    }
  end

  defp get_vocab_size(nil) do
    0
  end

  defp get_vocab_size(model) when is_map(model) do
    Map.get(model, :vocabulary, %{}) |> map_size()
  end

  defp get_vocab_size(_) do
    0
  end

  defp get_embedder_vocab_size(nil) do
    0
  end

  defp get_embedder_vocab_size(model) when is_map(model) do
    Map.get(model, :vocabulary, %{}) |> map_size()
  end

  defp get_embedder_vocab_size(_) do
    0
  end

  defp broadcast_models_loading(world_id) do
    PubSub.broadcast(@pubsub, "world_models:status", {:world_models_loading, world_id})
  end

  defp broadcast_models_loaded(world_id, models) do
    status = %{
      classifier: models[:classifier] != nil,
      embedder: models[:embedder] != nil,
      pos_model: models[:pos_model] != nil,
      entity_model: models[:entity_model] != nil
    }

    PubSub.broadcast(
      @pubsub,
      "world_models:status",
      {:world_models_loaded, world_id, status}
    )
  end

  defp broadcast_models_error(world_id, reason) do
    PubSub.broadcast(
      @pubsub,
      "world_models:status",
      {:world_models_error, world_id, reason}
    )
  end
end
