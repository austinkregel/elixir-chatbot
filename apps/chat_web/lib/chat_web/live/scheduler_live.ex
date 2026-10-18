defmodule ChatWeb.SchedulerLive do
  @moduledoc """
  The `/scheduler` page: alarms, reminders, timers and calendar items, and the
  time zone their times are read in, checked by hand in a browser.

  - **Read a time in your zone.** What is typed is resolved now, as a turn
    would be (`Brain.Temporal.Resolver`), in the zone this browser sends
    (the BrowserTimeZone hook, as the chat page's message form does). The
    page shows the zone the server received: its name, its offset now, the
    span its offsets are known for and how many changes of offset that span
    holds, and what the text resolved to, in UTC and in this browser's zone,
    with the offset its start was read with; or why it could not be read.
  - **Fired.** The items of yours that fired while a page was open, and your
    alarms ringing (`ChatWeb.WorldContext`), as the chat page shows them.
  - **Items in this world.** Every item of the selected world, with its state,
    when it is due, when it fired, and whose it is. Refresh reads them again.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Brain.Services.LocalScheduler
  alias Brain.Temporal.{Resolver, Zone}

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(:zone_years, Application.fetch_env!(:chat_web, :browser_zone_years))
     |> assign(:said, "")
     |> assign(:reading, nil)
     |> load_items()}
  end

  @impl true
  def handle_event("resolve", %{"said" => said, "time_zone" => json}, socket) when is_binary(json) and json != "" do
    zone = json |> Jason.decode!() |> Zone.from_browser!()
    turn = DateTime.utc_now()

    reading = %{zone: zone, turn: turn, offset: Zone.offset_at!(zone, turn), result: Resolver.resolve(said, turn, time_zone: zone)}
    {:noreply, socket |> assign(:said, said) |> assign(:reading, reading)}
  end

  def handle_event("resolve", params, _socket) do
    raise ArgumentError,
          "SchedulerLive: a reading carries what was said as said and the browser's time zone as time_zone (the " <>
            "form's BrowserTimeZone hook fills it); got: #{inspect(params)}"
  end

  def handle_event("refresh_items", _params, socket), do: {:noreply, load_items(socket)}

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket), do: {:noreply, load_items(socket)}

  defp load_items(socket), do: assign(socket, :items, LocalScheduler.items(socket.assigns.current_world_id))

  @impl true
  def render(assigns) do
    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div>
          <h1 class="text-title text-ink">Scheduler</h1>
          <p class="text-body text-ink-muted">
            Times read in this browser's zone, and the alarms, reminders, timers and calendar items that fire.
          </p>
        </div>
      </:page_header>

      <div class="space-y-space-xl p-space-lg">
        <.card>
          <.card_body class="space-y-space-md">
            <.section_header icon="hero-globe-americas">Read a time in your zone</.section_header>

            <form
              id="resolve-form"
              phx-submit="resolve"
              phx-hook="BrowserTimeZone"
              data-zone-years={@zone_years}
              class="flex flex-wrap items-end gap-space-md"
            >
              <div id="resolve-form-time-zone" phx-update="ignore" class="hidden">
                <input type="hidden" name="time_zone" value="" />
              </div>
              <div class="flex-1 min-w-48">
                <.input id="resolve-said" name="said" value={@said} label="When" placeholder="tomorrow at 9 am" />
              </div>
              <.btn type="submit" variant={:primary}>Read</.btn>
            </form>

            <div :if={@reading} id="reading" class="space-y-space-sm">
              <dl class="grid grid-cols-[max-content_1fr] gap-x-space-md gap-y-space-xs text-body">
                <dt class="text-ink-muted">Zone</dt>
                <dd id="reading-zone" class="text-ref text-ink">{@reading.zone.name}</dd>
                <dt class="text-ink-muted">Offset now</dt>
                <dd id="reading-offset" class="text-ref text-ink">{format_offset(@reading.offset)}</dd>
                <dt class="text-ink-muted">Offsets known</dt>
                <dd id="reading-span" class="text-ref text-ink">
                  {span(@reading.zone)}, {length(@reading.zone.transitions)} changes of offset
                </dd>
              </dl>

              <%= case @reading.result do %>
                <% {:ok, resolution} -> %>
                  <dl
                    id="reading-result"
                    class="grid grid-cols-[max-content_1fr] gap-x-space-md gap-y-space-xs text-body"
                  >
                    <dt class="text-ink-muted">Said</dt>
                    <dd class="text-ink">{resolution.said}</dd>
                    <dt class="text-ink-muted">Starts</dt>
                    <dd>
                      <.instant id="reading-starts" at={resolution.starts_at} />
                    </dd>
                    <dt class="text-ink-muted">Ends</dt>
                    <dd>
                      <%= if resolution.ends_at do %>
                        <.instant id="reading-ends" at={resolution.ends_at} />
                      <% else %>
                        <span class="text-ink-muted">no end</span>
                      <% end %>
                    </dd>
                    <dt class="text-ink-muted">Time of day said</dt>
                    <dd class="text-ref text-ink">{resolution.clock || "none"}</dd>
                    <dt class="text-ink-muted">Offset read with</dt>
                    <dd id="reading-utc-offset" class="text-ref text-ink">{format_offset(resolution.utc_offset)}</dd>
                  </dl>
                <% {:error, message} -> %>
                  <.alert id="reading-error" variant={:warning}>{message}</.alert>
              <% end %>
            </div>
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-space-md">
            <.section_header icon="hero-bell-alert">Fired</.section_header>
            <.empty_panel
              :if={@scheduler_notifications == []}
              kind={:plain}
              words="Nothing of yours has fired while this page was open, and no alarm of yours is ringing."
            />
            <div :if={@scheduler_notifications != []} id="fired" class="space-y-space-sm">
              <ChatWeb.ChatLive.scheduler_notification
                :for={notification <- @scheduler_notifications}
                notification={notification}
              />
            </div>
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-space-md">
            <div class="flex items-center justify-between gap-space-md">
              <.section_header icon="hero-clock">Items in this world</.section_header>
              <.btn variant={:outline} size={:sm} phx-click="refresh_items">Refresh</.btn>
            </div>
            <.empty_panel :if={@items == []} kind={:plain} words="This world has no alarms, reminders, timers or calendar items." />
            <div :if={@items != []} class="overflow-x-auto">
              <table id="scheduler-items" class="w-full text-left text-body-dense text-ink">
                <thead class="bg-surface-sunk">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Kind</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Set with</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Status</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Firing</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Due</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Fired</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Whose</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <tr :for={item <- @items} id={"scheduler-item-#{item.id}"}>
                    <td class="px-space-sm py-space-xs text-ink">{item.kind}</td>
                    <td class="px-space-sm py-space-xs text-ink">{set_with(item.said)}</td>
                    <td class="px-space-sm py-space-xs text-ref text-ink">{item.status || "none"}</td>
                    <td class="px-space-sm py-space-xs text-ref text-ink">{item.fire_state}</td>
                    <td class="px-space-sm py-space-xs">
                      <.instant :if={item.due_at} id={"scheduler-item-#{item.id}-due"} at={item.due_at} />
                    </td>
                    <td class="px-space-sm py-space-xs">
                      <.instant :if={item.fired_at} id={"scheduler-item-#{item.id}-fired"} at={item.fired_at} />
                    </td>
                    <td class="px-space-sm py-space-xs text-caption text-ink">{whose(item.web_user_id, @current_user_id)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </.card_body>
        </.card>
      </div>
    </.app_shell>
    """
  end

  # An instant in this browser's zone (the LocalTime hook), with its UTC form.
  attr :id, :string, required: true
  attr :at, DateTime, required: true

  defp instant(assigns) do
    ~H"""
    <span class="text-ink">
      <time
        id={@id}
        phx-hook="LocalTime"
        phx-update="ignore"
        data-format="datetime"
        datetime={ChatWeb.ChatLive.browser_instant(@at)}
      ></time>
      <span class="text-caption text-ink-muted">({ChatWeb.ChatLive.browser_instant(@at)})</span>
    </span>
    """
  end

  defp format_offset(seconds) when is_integer(seconds) do
    sign = if seconds < 0, do: "-", else: "+"
    minutes = div(abs(seconds), 60)
    "#{sign}#{pad(div(minutes, 60))}:#{pad(rem(minutes, 60))}"
  end

  defp pad(number), do: number |> Integer.to_string() |> String.pad_leading(2, "0")

  defp span(%Zone{from: nil, until: nil}), do: "every instant"
  defp span(%Zone{from: from, until: until}), do: "#{from.year} to #{until.year}"

  defp set_with(said), do: said |> Map.values() |> Enum.reject(&(&1 in [nil, ""])) |> Enum.join(" · ")

  defp whose(nil, _current_user_id), do: "no owner recorded"
  defp whose(owner, owner), do: "yours"
  defp whose(_owner, _current_user_id), do: "another web user's"
end
