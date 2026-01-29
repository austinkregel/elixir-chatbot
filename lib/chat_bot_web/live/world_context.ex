defmodule ChatBotWeb.WorldContext do
  @moduledoc """
  LiveView on_mount hook for world context persistence.

  Provides a consistent world context across all LiveViews in a session.
  The selected world persists across page navigations via PubSub broadcasts
  and session storage.

  ## Usage

  In the router:

      live_session :world_context, on_mount: [{ChatBotWeb.WorldContext, :default}] do
        live "/chat", ChatLive
        live "/explorer", ExplorerLive
        ...
      end

  In LiveViews, the following assigns are available:
  - `@current_world_id` - The currently selected world ID
  - `@available_worlds` - List of available worlds
  - `@system_ready` - Whether all systems are ready
  - `@current_path` - The current request path

  ## World Change Notifications

  When a world is switched, a PubSub message is broadcast on the topic
  `world_context:<session_id>` with the payload `{:world_changed, world_id}`.
  Systems can subscribe to this to react to world changes.
  """

  import Phoenix.LiveView
  import Phoenix.Component

  require Logger

  @default_world_id "default"
  @pubsub ChatBot.PubSub

  @doc """
  On mount hook that sets up world context.
  """
  def on_mount(:default, params, session, socket) do
    # Get session_id for PubSub topic
    session_id = get_session_id(session, socket)

    # Get world_id from params, session, or ETS cache, or default
    # We use ETS to persist across LiveView navigations since session isn't writable
    world_id =
      params["world_id"] ||
        get_cached_world_id(session_id) ||
        Map.get(session, "world_id") ||
        @default_world_id

    # Cache the world_id for this session
    cache_world_id(session_id, world_id)

    # Load available worlds
    available_worlds = get_available_worlds()

    # Check system status
    system_ready = ChatBot.SystemStatus.all_ready?()

    # Get current path
    current_path = get_current_path(socket)

    # Subscribe to world context changes for this session AND globally
    if connected?(socket) do
      Phoenix.PubSub.subscribe(@pubsub, world_topic(session_id))
      # Also subscribe to global topic to catch cross-session broadcasts
      Phoenix.PubSub.subscribe(@pubsub, "world_context:global")
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

  # ============================================================================
  # Event Handlers
  # ============================================================================

  defp handle_world_events("switch_world", %{"world_id" => world_id}, socket) do
    old_world_id = socket.assigns.current_world_id
    session_id = socket.assigns.world_session_id

    # Only do something if world actually changed
    if world_id != old_world_id do
      Logger.info("Switching world context",
        from: old_world_id,
        to: world_id,
        session: String.slice(session_id, 0, 8)
      )

      # Cache the new world_id for this session
      cache_world_id(session_id, world_id)

      # Broadcast world change to all LiveViews in this session
      Phoenix.PubSub.broadcast(@pubsub, world_topic(session_id), {:world_changed, world_id})

      # Also broadcast globally for backend systems (includes session_id for filtering)
      Phoenix.PubSub.broadcast(
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

  # ============================================================================
  # Info Handlers (PubSub messages)
  # ============================================================================

  defp handle_world_info({:world_changed, world_id}, socket) do
    # Another LiveView in this session changed the world - sync up
    socket =
      if socket.assigns.current_world_id != world_id do
        # Update cache for consistency
        cache_world_id(socket.assigns.world_session_id, world_id)

        # Let the LiveView handle the world change for page-specific data reload
        send(self(), {:world_context_changed, world_id})
        assign(socket, :current_world_id, world_id)
      else
        socket
      end

    # Halt to prevent the raw PubSub message from reaching the LiveView
    # The LiveView will receive {:world_context_changed, world_id} instead
    {:halt, socket}
  end

  defp handle_world_info({:world_changed, _other_session_id, world_id}, socket) do
    # Global broadcast from another session - check if same session
    # We receive this because we subscribe to global, but we filter by our session
    # For now, we also sync to this world if it changed (cross-tab sync)
    socket =
      if socket.assigns.current_world_id != world_id do
        # Update cache
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

  # ============================================================================
  # Params Handler
  # ============================================================================

  defp handle_params(_params, uri, socket) do
    # Update current path on navigation
    %URI{path: path} = URI.parse(uri)
    {:cont, assign(socket, :current_path, path || "/")}
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp get_available_worlds do
    try do
      ChatBot.Learning.WorldManager.list_worlds()
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
    # Try to get a stable session identifier
    # Priority: explicit session_id > CSRF token > socket private key > live_session name
    cond do
      # Check explicit session_id in session
      session_id = Map.get(session, "session_id") ->
        session_id

      # Check CSRF token (stable across live_session)
      csrf = Map.get(session, "_csrf_token") ->
        csrf

      # Check socket's root_pid as a stable identifier for this browser tab
      socket.root_pid != nil ->
        # Use the root LiveView's PID as identifier - stable within a browser tab
        :erlang.pid_to_list(socket.root_pid) |> to_string() |> Base.encode64(padding: false)

      # Fallback to socket ID if available
      socket.id != nil ->
        socket.id

      # Last resort - generate one (this means each LV would have different ID, but we log it)
      true ->
        id = :crypto.strong_rand_bytes(16) |> Base.url_encode64(padding: false)

        Logger.warning(
          "WorldContext: Generated random session_id - world sync may not work: #{String.slice(id, 0, 8)}"
        )

        id
    end
  end

  defp world_topic(session_id), do: "world_context:#{session_id}"

  # ============================================================================
  # Public API for other modules
  # ============================================================================

  @doc """
  Returns the default world ID.
  """
  def default_world_id, do: @default_world_id

  @doc """
  Subscribe to global world change events.
  Receives `{:world_changed, session_id, world_id}` messages.
  """
  def subscribe_global do
    Phoenix.PubSub.subscribe(@pubsub, "world_context:global")
  end

  @doc """
  Broadcast a world change event for a session.
  Used by backend systems that need to trigger a world switch.
  """
  def broadcast_world_change(session_id, world_id) do
    Phoenix.PubSub.broadcast(@pubsub, world_topic(session_id), {:world_changed, world_id})
  end

  # ============================================================================
  # ETS Cache for World ID Persistence
  # ============================================================================

  @ets_table :world_context_cache
  # 1 hour
  @cache_ttl_ms 3_600_000

  defp ensure_ets_table do
    if :ets.whereis(@ets_table) == :undefined do
      :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
    end
  rescue
    ArgumentError ->
      # Table already exists in another process - that's fine
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
          # Check if cache is still valid
          if System.monotonic_time(:millisecond) - cached_at < @cache_ttl_ms do
            world_id
          else
            # Expired - delete and return nil
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
