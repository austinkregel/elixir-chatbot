defmodule ChatWeb.WorldContext do
  @moduledoc "LiveView on_mount hook that gives every page the web user and the world they selected.\n\nThe browser's web user comes from the session (`ChatWeb.Plugs.CurrentUser`), and\nthe selected world is stored on that user (`Atlas.WebUsers`), so it is the same on\nevery page, after a reload, and on the next visit. New conversations run in it.\n\n## Usage\n\nIn the router:\n\n    live_session :world_context, on_mount: [{ChatWeb.WorldContext, :default}] do\n      live \"/chat\", ChatLive\n      live \"/explorer\", ExplorerLive\n      ...\n    end\n\nIn LiveViews, the following assigns are available:\n- `@current_user_id` - The web user's id\n- `@current_world_id` - The world the user has selected\n- `@available_worlds` - Every world `World.Manager` holds\n- `@system_ready` - Whether all systems are ready\n- `@current_path` - The current request path\n- `@scheduler_notifications` - The user's alarms, reminders, timers and calendar items that fired while the page was open, newest first, and their alarms ringing when it opened (`Brain.Services.LocalScheduler.notification/1`)\n\nA selected world that is no longer loaded is kept as the selection, and the app\nshell marks it as not in the world list until the user picks another.\n\n## World Change Notifications\n\nSelecting a world (the `\"switch_world\"` event) stores it on the user and\nbroadcasts `{:world_changed, world_id}` on `web_user:<user_id>`, so the user's\nother open pages follow. The hook handles `\"switch_world\"` and `\"refresh_worlds\"`\nitself; every page, the one switched on included, is then sent\n`{:world_context_changed, world_id}` and reloads what depends on the world there.\n`{:world_changed, user_id, world_id}` also goes to `world_context:global`, where\n`Brain.Worlds.ModelRegistry` loads that world's models.\n\n## Scheduler Notifications\n\nWhen one of the user's items fires, `Brain.Services.LocalScheduler` broadcasts\n`{:scheduler_fired, notification}` on the same topic. The hook handles it on\nevery page, keeping it in `@scheduler_notifications` (the chat page shows them),\nand handles `\"dismiss_notification\"`, which takes one off the page.\n"

  alias Phoenix.PubSub
  alias Atlas.WebUsers
  alias Brain.Services.LocalScheduler
  alias Brain.SystemStatus
  alias ChatWeb.Worlds
  import Phoenix.LiveView
  import Phoenix.Component

  require Logger

  @pubsub Brain.PubSub

  @doc "On mount hook that sets up the user's world context.\n"
  def on_mount(:default, _params, session, socket) do
    user =
      case Map.fetch(session, "user_id") do
        {:ok, user_id} ->
          WebUsers.get!(user_id)

        :error ->
          raise "WorldContext: the session has no user_id; ChatWeb.Plugs.CurrentUser must run in the browser pipeline"
      end

    if connected?(socket) do
      PubSub.subscribe(@pubsub, user_topic(user.id))
    end

    socket =
      socket
      |> assign(:current_user_id, user.id)
      |> assign(:current_world_id, user.selected_world_id)
      |> assign(:scheduler_notifications, LocalScheduler.ringing_notifications(user.id))
      |> assign(:available_worlds, Worlds.list())
      |> assign(:system_ready, SystemStatus.all_ready?())
      |> assign(:current_path, get_current_path(socket))
      |> attach_hook(:world_context_events, :handle_event, &handle_world_events/3)
      |> attach_hook(:world_context_params, :handle_params, &handle_params/3)
      |> attach_hook(:world_context_info, :handle_info, &handle_world_info/2)

    {:cont, socket}
  end

  defp handle_world_events("switch_world", %{"world_id" => world_id}, socket) do
    if world_id == socket.assigns.current_world_id do
      {:halt, socket}
    else
      {:halt, select_world(socket, world_id)}
    end
  end

  defp handle_world_events("refresh_worlds", _params, socket) do
    {:halt, assign(socket, :available_worlds, Worlds.list())}
  end

  # Takes a notification off the page; an alarm keeps ringing until it is
  # snoozed or cancelled, and shows again on the next page opened.
  defp handle_world_events("dismiss_notification", %{"id" => id}, socket) do
    {:halt, update(socket, :scheduler_notifications, &Enum.reject(&1, fn notification -> notification.id == id end))}
  end

  defp handle_world_events(_event, _params, socket) do
    {:cont, socket}
  end

  @doc """
  Stores `world_id` as the user's selected world and tells the user's open pages.
  Raises when no such world is loaded: the selector only offers loaded worlds, so
  anything else is not a choice the page made.
  """
  def select_world(socket, world_id) do
    Worlds.fetch!(world_id)
    user_id = socket.assigns.current_user_id
    old_world_id = socket.assigns.current_world_id

    user_id |> WebUsers.get!() |> WebUsers.select_world!(world_id)

    Logger.info("Switching world", user_id: user_id, from: old_world_id, to: world_id)

    PubSub.broadcast(@pubsub, user_topic(user_id), {:world_changed, world_id})
    PubSub.broadcast(@pubsub, "world_context:global", {:world_changed, user_id, world_id})
    send(self(), {:world_context_changed, world_id})

    socket
    |> assign(:current_world_id, world_id)
    |> put_flash(:info, "Switched to world: #{world_id}")
  end

  defp handle_world_info({:world_changed, world_id}, socket) do
    socket =
      if socket.assigns.current_world_id != world_id do
        send(self(), {:world_context_changed, world_id})
        assign(socket, :current_world_id, world_id)
      else
        socket
      end

    {:halt, socket}
  end

  # One of the user's items fired: it shows first, in place of what was shown
  # of it before (an alarm snoozed and ringing again).
  defp handle_world_info({:scheduler_fired, %{id: id} = notification}, socket) do
    {:halt,
     update(socket, :scheduler_notifications, fn shown ->
       [notification | Enum.reject(shown, &(&1.id == id))]
     end)}
  end

  defp handle_world_info(_msg, socket) do
    {:cont, socket}
  end

  defp handle_params(_params, uri, socket) do
    %URI{path: path} = URI.parse(uri)
    {:cont, assign(socket, :current_path, path || "/")}
  end

  defp get_current_path(socket) do
    case socket.private do
      %{connect_info: %{request_path: path}} when is_binary(path) -> path
      _ -> "/"
    end
  end

  defp user_topic(user_id), do: WebUsers.topic(user_id)
end
