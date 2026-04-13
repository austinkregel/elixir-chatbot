defmodule ChatWeb.WorldContext do
  @moduledoc "LiveView on_mount hook for world context persistence.\n\nProvides a consistent world context across all LiveViews in a session.\nThe selected world persists across page navigations via PubSub broadcasts\nand session storage.\n\n## Usage\n\nIn the router:\n\n    live_session :world_context, on_mount: [{ChatWeb.WorldContext, :default}] do\n      live \"/chat\", ChatLive\n      live \"/explorer\", ExplorerLive\n      ...\n    end\n\nIn LiveViews, the following assigns are available:\n- `@current_world_id` - The currently selected world ID\n- `@available_worlds` - List of available worlds\n- `@system_ready` - Whether all systems are ready\n- `@current_path` - The current request path\n\n## World Change Notifications\n\nWhen a world is switched, a PubSub message is broadcast on the topic\n`world_context:<session_id>` with the payload `{:world_changed, world_id}`.\nSystems can subscribe to this to react to world changes.\n"

  alias Phoenix.PubSub
  alias World.Manager
  alias Brain.SystemStatus
  import Phoenix.LiveView
  import Phoenix.Component

  require Logger

  @default_world_id "default"
  @pubsub Brain.PubSub

  @doc "On mount hook that sets up world context.\n"
  def on_mount(:default, params, session, socket) do
    session_id = get_session_id(session, socket)

    world_id =
      params["world_id"] ||
        get_cached_world_id(session_id) ||
        Map.get(session, "world_id") ||
        @default_world_id

    cache_world_id(session_id, world_id)
    available_worlds = get_available_worlds()
    system_ready = SystemStatus.all_ready?()
    current_path = get_current_path(socket)

    if connected?(socket) do
      PubSub.subscribe(@pubsub, world_topic(session_id))
      PubSub.subscribe(@pubsub, "world_context:global")
    end

    socket =
      socket
      |> assign(:current_world_id, world_id)
      |> assign(:available_worlds, available_worlds)
      |> assign(:system_ready, system_ready)
      |> assign(:current_path, current_path)
      |> assign(:world_session_id, session_id)
      |> attach_hook(:world_context_events, :handle_event, &handle_world_events/3)
      |> attach_hook(:world_context_params, :handle_params, &handle_params/3)
      |> attach_hook(:world_context_info, :handle_info, &handle_world_info/2)

    {:cont, socket}
  end

  defp handle_world_events("switch_world", %{"world_id" => world_id}, socket) do
    old_world_id = socket.assigns.current_world_id
    session_id = socket.assigns.world_session_id

    if world_id != old_world_id do
      Logger.info("Switching world context",
        from: old_world_id,
        to: world_id,
        session: String.slice(session_id, 0, 8)
      )

      cache_world_id(session_id, world_id)
      PubSub.broadcast(@pubsub, world_topic(session_id), {:world_changed, world_id})

      PubSub.broadcast(
        @pubsub,
        "world_context:global",
        {:world_changed, session_id, world_id}
      )

      socket =
        socket
        |> assign(:current_world_id, world_id)
        |> put_flash(:info, "Switched to world: #{world_id}")

      {:cont, socket}
    else
      {:cont, socket}
    end
  end

  defp handle_world_events("refresh_worlds", _params, socket) do
    available_worlds = get_available_worlds()
    {:cont, assign(socket, :available_worlds, available_worlds)}
  end

  defp handle_world_events(_event, _params, socket) do
    {:cont, socket}
  end

  defp handle_world_info({:world_changed, world_id}, socket) do
    socket =
      if socket.assigns.current_world_id != world_id do
        cache_world_id(socket.assigns.world_session_id, world_id)
        send(self(), {:world_context_changed, world_id})
        assign(socket, :current_world_id, world_id)
      else
        socket
      end

    {:halt, socket}
  end

  defp handle_world_info({:world_changed, _other_session_id, world_id}, socket) do
    socket =
      if socket.assigns.current_world_id != world_id do
        cache_world_id(socket.assigns.world_session_id, world_id)

        send(self(), {:world_context_changed, world_id})
        assign(socket, :current_world_id, world_id)
      else
        socket
      end

    {:halt, socket}
  end

  defp handle_world_info(_msg, socket) do
    {:cont, socket}
  end

  defp handle_params(_params, uri, socket) do
    %URI{path: path} = URI.parse(uri)
    {:cont, assign(socket, :current_path, path || "/")}
  end

  defp get_available_worlds do
    try do
      Manager.list_worlds()
      |> Enum.map(fn world ->
        %{id: world.id, name: world.name}
      end)
      |> Enum.sort_by(& &1.name)
      |> ensure_default_world()
    rescue
      _ -> [%{id: "default", name: "default"}]
    end
  end

  defp ensure_default_world(worlds) do
    if Enum.any?(worlds, &(&1.id == "default")) do
      worlds
    else
      [%{id: "default", name: "default"} | worlds]
    end
  end

  defp get_current_path(socket) do
    case socket.private do
      %{connect_info: %{request_path: path}} when is_binary(path) -> path
      _ -> "/"
    end
  end

  defp get_session_id(session, socket) do
    cond do
      session_id = Map.get(session, "session_id") ->
        session_id

      csrf = Map.get(session, "_csrf_token") ->
        csrf

      socket.root_pid != nil ->
        :erlang.pid_to_list(socket.root_pid) |> to_string() |> Base.encode64(padding: false)

      socket.id != nil ->
        socket.id

      true ->
        id = :crypto.strong_rand_bytes(16) |> Base.url_encode64(padding: false)

        Logger.warning(
          "WorldContext: Generated random session_id - world sync may not work: #{String.slice(id, 0, 8)}"
        )

        id
    end
  end

  defp world_topic(session_id) do
    "world_context:#{session_id}"
  end

  @doc "Returns the default world ID.\n"
  def default_world_id do
    @default_world_id
  end

  @doc "Subscribe to global world change events.\nReceives `{:world_changed, session_id, world_id}` messages.\n"
  def subscribe_global do
    PubSub.subscribe(@pubsub, "world_context:global")
  end

  @doc "Broadcast a world change event for a session.\nUsed by backend systems that need to trigger a world switch.\n"
  def broadcast_world_change(session_id, world_id) do
    PubSub.broadcast(@pubsub, world_topic(session_id), {:world_changed, world_id})
  end

  @ets_table :world_context_cache
  @cache_ttl_ms 3_600_000

  defp ensure_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end
  rescue
    ArgumentError ->
      :ok
  end

  defp cache_world_id(session_id, world_id) do
    ensure_ets_table()

    try do
      :ets.insert(@ets_table, {session_id, world_id, System.monotonic_time(:millisecond)})
    rescue
      _ -> :ok
    end
  end

  defp get_cached_world_id(session_id) do
    ensure_ets_table()

    try do
      case :ets.lookup(@ets_table, session_id) do
        [{^session_id, world_id, cached_at}] ->
          if System.monotonic_time(:millisecond) - cached_at < @cache_ttl_ms do
            world_id
          else
            :ets.delete(@ets_table, session_id)
            nil
          end

        [] ->
          nil
      end
    rescue
      _ -> nil
    end
  end
end