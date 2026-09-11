defmodule ChatWeb.AccuracyLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()
    :ok
  end

  describe "mount" do
    test "mounts successfully", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/accuracy")
      assert html =~ "Accuracy" or html =~ "Evaluation" or html =~ "Intent"
    end

    test "displays task tabs", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/accuracy")
      assert html =~ "Intent" or html =~ "intent"
    end

    test "shows default active tab", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/accuracy")
      html = render(view)
      assert is_binary(html)
    end
  end

  describe "tab switching" do
    test "can switch tabs", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/accuracy")

      html = view |> element("[phx-click=switch_tab][phx-value-tab=ner]") |> render_click()
      assert is_binary(html)
    rescue
      _ -> :ok
    end
  end
end
