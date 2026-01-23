defmodule ChatBotWeb.ChatLiveProgressTest do
  use ChatBotWeb.ConnCase, async: false

  import Phoenix.LiveViewTest
  import ChatBot.TestHelpers

  setup do
    start_brain_services()
    :ok
  end

  test "shows live analysis progress steps while evaluating", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")

    view
    |> form("#message-form", %{input: "Hello there"})
    |> render_submit()

    html =
      eventually(
        fn -> render(view) end,
        fn h ->
          String.contains?(h, "Developer panel") and
            (String.contains?(h, "pipeline") or String.contains?(h, "chunk"))
        end,
        200
      )

    assert html =~ "Developer panel"
    assert html =~ "pipeline" or html =~ "chunk"
  end

  defp eventually(fetch_html, predicate, attempts) do
    Enum.reduce_while(1..attempts, nil, fn _, _ ->
      html = fetch_html.()

      if predicate.(html) do
        {:halt, html}
      else
        Process.sleep(50)
        {:cont, nil}
      end
    end) || flunk("Condition not met within timeout")
  end
end

