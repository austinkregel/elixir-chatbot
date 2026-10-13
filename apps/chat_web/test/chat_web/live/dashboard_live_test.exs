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

    test "shows the selected world's semantic facts in responses", %{conn: conn} do
      {:ok, view, html} = live(conn, "/dashboard")

      assert html =~ "Semantic Facts in Responses"
      assert has_element?(view, "#semantic-fact-retrievals")
      assert has_element?(view, "#semantic-fact-patterns-found")
      assert has_element?(view, "#semantic-fact-used-synthesizer")
      assert has_element?(view, "#semantic-fact-used-realization-packet")
      assert has_element?(view, "#semantic-fact-retrieved_similarity")
      assert has_element?(view, "#semantic-fact-used_similarity")
    end

    test "shows world models section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "World Models"
    end

    test "displays GenServer status section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Core" or html =~ "GenServer" or html =~ "Systems"
    end

    test "displays metrics section from Aggregator", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Metrics" or html =~ "Telemetry" or html =~ "Updated"
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

  describe "semantic fact measurement window" do
    # A world the world selector lists, so the page can be switched to it: it
    # is created without the test mark, its id starts with `test_`, it is
    # ephemeral, so nothing of it is on disk, and it is destroyed when the test
    # ends. Its metric rows are written inside the test's sandbox transaction.
    setup do
      world_id = "test_dashboard_semantic_window_#{System.unique_integer([:positive])}"
      {:ok, world} = World.Manager.create(world_id, id: world_id, mode: :ephemeral)
      on_exit(fn -> World.Manager.destroy(world.id) end)
      %{world: world}
    end

    defp dashboard_in(conn, world) do
      {:ok, view, _html} = live(conn, "/dashboard")
      view |> form("form[phx-change=switch_world]", %{world_id: world.id}) |> render_change()
      view
    end

    test "says when nothing was counted in the world, and shows no previous window", %{conn: conn, world: world} do
      view = dashboard_in(conn, world)

      assert view |> element("#semantic-fact-window") |> render() =~ "Nothing counted in this world yet"
      refute render(view) =~ "since the brain started"
      refute has_element?(view, "#semantic-fact-previous-window")
    end

    test "shows when the current window started, and starting a new one keeps the ended window's counts",
         %{conn: conn, world: world} do
      :ok = Atlas.SemanticFactMetrics.increment!(world.id, %{"retrievals" => 3, "patterns_found" => 2})
      [%{window: first}] = Atlas.SemanticFactMetrics.windows(world.id, 1)

      view = dashboard_in(conn, world)

      assert view |> element("#semantic-fact-window") |> render() =~
               ChatWeb.DashboardLive.format_window_time(first.started_at)

      assert view |> element("#semantic-fact-retrievals") |> render() =~ "3"
      refute has_element?(view, "#semantic-fact-previous-window")

      view |> element("#semantic-fact-start-window") |> render_click()
      assert has_element?(view, "#semantic-fact-window-confirm")
      view |> element("#semantic-fact-window-confirm-confirm") |> render_click()

      refute has_element?(view, "#semantic-fact-window-confirm")
      assert [%{window: second, counts: new_counts}, %{window: ended}] = Atlas.SemanticFactMetrics.windows(world.id, 5)
      assert new_counts == %{}
      assert ended.id == first.id

      assert view |> element("#semantic-fact-window") |> render() =~
               ChatWeb.DashboardLive.format_window_time(second.started_at)

      assert view |> element("#semantic-fact-retrievals") |> render() =~ ~r/>\s*0\s*</

      previous = view |> element("#semantic-fact-previous-window") |> render()
      assert previous =~ ChatWeb.DashboardLive.format_window_time(first.started_at)
      assert previous =~ ChatWeb.DashboardLive.format_window_time(ended.ended_at)
      assert previous =~ ~r/>\s*3\s*<\/span> turn\(s\) retrieving facts/
      assert previous =~ ~r/>\s*2\s*<\/span> recurring pattern\(s\) found/
    end

    test "cancelling leaves the window as it was", %{conn: conn, world: world} do
      :ok = Atlas.SemanticFactMetrics.increment!(world.id, %{"retrievals" => 1})
      view = dashboard_in(conn, world)

      view |> element("#semantic-fact-start-window") |> render_click()
      view |> element("#semantic-fact-window-confirm-cancel") |> render_click()

      refute has_element?(view, "#semantic-fact-window-confirm")
      assert [%{counts: %{"retrievals" => 1}}] = Atlas.SemanticFactMetrics.windows(world.id, 5)
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

  describe "readiness matrix" do
    test "renders readiness matrix panel", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Readiness Matrix"
      assert html =~ "Brain Core"
      assert html =~ "Embedder"
      assert html =~ "Memory Store"
      assert html =~ "NLP Pipeline"
      assert html =~ "Intent Classifier"
      assert html =~ "Micro Classifiers"
    end

    test "shows overall readiness badge", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "All Systems Ready" or html =~ "Some Systems Initializing"
    end
  end

  describe "response timing panel" do
    test "renders response timing section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Response Timing"
      assert html =~ "Brain Evaluate"
      assert html =~ "Analysis Pipeline"
    end
  end

  describe "micro classifiers panel" do
    test "renders micro classifiers section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Micro Classifiers"
      assert html =~ "Lightweight TF-IDF"
    end
  end
end
