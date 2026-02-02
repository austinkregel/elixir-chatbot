defmodule ChatWeb.DashboardLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()

    # Start required GenServers
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(World.Manager)

    :ok
  end

  describe "mount" do
    test "mounts successfully with default world", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Operations Dashboard"
      assert html =~ "default"
    end

    test "displays world memory stats section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Episodes in World"
      assert html =~ "Semantic Facts"
      assert html =~ "Knowledge Categories"
    end

    test "shows world models section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "World Models"
    end

    test "displays auto-refresh toggle", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Auto-refresh"
    end

    test "shows refresh button", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Refresh"
    end
  end

  describe "auto-refresh toggle" do
    test "can toggle auto-refresh off", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Toggle auto-refresh
      view |> element("[phx-click=toggle_auto_refresh]") |> render_click()

      # Should still render the page
      html = render(view)
      assert html =~ "Operations Dashboard"
    end
  end

  describe "manual refresh" do
    test "manual refresh button works", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Click manual refresh
      view |> element("button", "Refresh") |> render_click()

      # Should still be on dashboard
      html = render(view)
      assert html =~ "Operations Dashboard"
    end
  end

  describe "page structure" do
    test "displays last updated time", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Updated:"
    end

    test "shows current world context", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "World:"
      assert html =~ "default"
    end
  end
end
