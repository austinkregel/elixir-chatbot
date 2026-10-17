defmodule ChatWeb.ChatLiveProgressTest do
  use ChatWeb.ConnCase, async: false

  import Phoenix.LiveViewTest
  import Brain.TestHelpers, except: [eventually: 3, eventually: 4]

  setup do
    start_brain_services()
    :ok
  end

  test "shows live analysis progress steps while evaluating", %{conn: conn} do
    {:ok, view, _html} = live(conn, "/chat")

    # The time zone the form's BrowserTimeZone hook sends with a message, as a
    # browser in UTC sends it. The hook writes it into the form's hidden
    # time_zone input as the form is submitted, so it is given to
    # render_submit/2 as the hook's value; form/3 takes only what is typed.
    browser_zone =
      Jason.encode!(%{
        "name" => "Etc/UTC",
        "at" => DateTime.to_unix(~U[2026-10-14 12:00:00Z], :millisecond),
        "offset" => 0,
        "from" => DateTime.to_unix(~U[2000-01-01 00:00:00Z], :millisecond),
        "until" => DateTime.to_unix(~U[2100-01-01 00:00:00Z], :millisecond),
        "offset_at_from" => 0,
        "transitions" => []
      })

    view
    |> form("#message-form", %{input: "Hello there"})
    |> render_submit(%{time_zone: browser_zone})

    # Wait for the processing summary pill to appear (shown inline with user messages)
    html =
      eventually(
        fn -> render(view) end,
        fn h ->
          # The processing pill shows strategy info like "can respond" or "Processing"
          String.contains?(h, "select_message") or
            String.contains?(h, "Processing") or
            String.contains?(h, "can respond")
        end,
        200
      )

    # Processing pill should be visible with a click handler
    assert html =~ "select_message" or html =~ "Processing" or html =~ "can respond"
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
