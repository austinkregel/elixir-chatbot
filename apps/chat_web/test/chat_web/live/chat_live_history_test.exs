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

  test "clicking a conversation loads its message history", %{conn: conn} do
    {:ok, c1} = Brain.create_conversation()
    {:ok, %{response: r1}} = Brain.evaluate(c1, "History test message 1")

    {:ok, c2} = Brain.create_conversation()
    {:ok, _r2} = Brain.evaluate(c2, "History test message 2")

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