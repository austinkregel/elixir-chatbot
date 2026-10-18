defmodule ChatWeb.ChatLiveHistoryTest do
  alias Phoenix.HTML
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias Brain

  setup do
    start_brain_services()
    :ok
  end

  # The world the test suite runs in, which a new web user starts in.
  @world_id "default"

  # A world a manager creates schedules with the local scheduler
  # (`World.Manager.create/2`, `mix world.assign_local_scheduling` for one that
  # existed before), and a turn the classifier reads as a timer, alarm,
  # reminder or calendar intent is dispatched to it; the suite's world gets the
  # same driver, inside the test's transaction.
  setup do
    :ok = Brain.Services.Dispatcher.assign_scheduling_driver(@world_id, Brain.Services.LocalScheduler)
  end

  test "clicking a conversation loads its message history", %{conn: conn} do
    {:ok, c1} = Brain.create_conversation(world_id: @world_id)
    {:ok, %{response: r1}} = Brain.evaluate(c1, "History test message 1", time_zone: Brain.Temporal.Zone.utc())

    {:ok, c2} = Brain.create_conversation(world_id: @world_id)
    {:ok, _r2} = Brain.evaluate(c2, "History test message 2", time_zone: Brain.Temporal.Zone.utc())

    {:ok, view, _html} = live(conn, "/chat")

    view
    |> element("[phx-click='select_conversation'][phx-value-conversation_id='#{c1}']")
    |> render_click()

    html = render(view)
    escaped_r1 = HTML.html_escape(r1) |> HTML.safe_to_string()
    assert html =~ "History test message 1"
    assert html =~ escaped_r1
    refute html =~ "History test message 2"
  end
end