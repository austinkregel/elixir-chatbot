defmodule ChatWeb.SchedulerNotificationTest do
  @moduledoc """
  How the chat page shows an alarm, reminder, timer or calendar item that
  fired (`ChatWeb.ChatLive.scheduler_notification/1`), from the notification
  `Brain.Services.LocalScheduler` tells its owner's pages.
  """
  use ExUnit.Case, async: true

  import Phoenix.LiveViewTest

  alias ChatWeb.ChatLive

  defp notification(changes) do
    Map.merge(
      %{
        id: "4f1d1c7e-6a52-4c39-9d43-2b8f5d7a9e10",
        world_id: "test_scheduler_notification",
        kind: :alarm,
        said: %{time: "7 am", date: "tomorrow"},
        due_at: ~U[2026-10-13 12:00:00.000000Z],
        fired_at: ~U[2026-10-13 12:00:00.000000Z],
        late: false,
        status: "ringing"
      },
      changes
    )
  end

  defp render_notification(changes),
    do: render_component(&ChatLive.scheduler_notification/1, notification: notification(changes))

  test "a ringing alarm shows what it was set for, that it rings, and its due time for the browser to show in its zone" do
    html = render_notification(%{})

    assert html =~ ~s(id="notification-4f1d1c7e-6a52-4c39-9d43-2b8f5d7a9e10")
    assert html =~ "Alarm"
    assert html =~ "Ringing"
    assert html =~ "tomorrow · 7 am"
    assert html =~ ~r/<time[^>]*phx-hook="LocalTime"[^>]*datetime="2026-10-13T12:00:00.000Z"|<time[^>]*datetime="2026-10-13T12:00:00.000Z"[^>]*phx-hook="LocalTime"/
    assert html =~ "Ask to snooze it"
    assert html =~ ~s(phx-click="dismiss_notification")
    refute html =~ "Fired late"
  end

  test "an item that fired late says so, with when it fired" do
    html =
      render_notification(%{
        kind: :reminder,
        said: %{text: "call home", time: "6 pm", date: ""},
        status: nil,
        late: true,
        fired_at: ~U[2026-10-13 19:00:00.000000Z]
      })

    assert html =~ "Reminder"
    assert html =~ "call home · 6 pm"
    assert html =~ "Fired late"
    assert html =~ ~s(datetime="2026-10-13T19:00:00.000Z")
    refute html =~ "Ringing"
  end

  test "a finished timer shows its duration and that it has finished" do
    html = render_notification(%{kind: :timer, said: %{duration: "5 minutes"}, status: "finished"})

    assert html =~ "Timer"
    assert html =~ "Finished"
    assert html =~ "5 minutes"
    refute html =~ "Ask to snooze it"
  end

  test "an instant is given to the browser in UTC, to the millisecond" do
    assert ChatLive.browser_instant(~U[2026-10-13 12:00:00.123456Z]) == "2026-10-13T12:00:00.123Z"
  end
end
