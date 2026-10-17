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
      count_semantic_facts!(world, [{"retrievals", :sum, 3}, {"patterns_found", :sum, 2}])
      [%{window: first}] = semantic_fact_windows(world, 1)

      view = dashboard_in(conn, world)

      assert view |> element("#semantic-fact-window") |> render() =~
               ChatWeb.DashboardLive.format_window_time(first.started_at)

      assert view |> element("#semantic-fact-retrievals") |> render() =~ "3"
      refute has_element?(view, "#semantic-fact-previous-window")

      view |> element("#semantic-fact-start-window") |> render_click()
      assert has_element?(view, "#semantic-fact-window-confirm")
      view |> element("#semantic-fact-window-confirm-confirm") |> render_click()

      refute has_element?(view, "#semantic-fact-window-confirm")
      assert [%{window: second, values: new_counts}, %{window: ended}] = semantic_fact_windows(world, 5)
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
      count_semantic_facts!(world, [{"retrievals", :sum, 1}])
      view = dashboard_in(conn, world)

      view |> element("#semantic-fact-start-window") |> render_click()
      view |> element("#semantic-fact-window-confirm-cancel") |> render_click()

      refute has_element?(view, "#semantic-fact-window-confirm")
      assert [%{values: %{"retrievals" => 1}}] = semantic_fact_windows(world, 5)
    end

    defp count_semantic_facts!(world, values) do
      :ok = Atlas.Metrics.write!([%{scope: {:world, world.id}, family: "semantic_facts", values: values}])
    end

    defp semantic_fact_windows(world, limit), do: Atlas.Metrics.windows({:world, world.id}, "semantic_facts", limit)
  end

  describe "telemetry health" do
    test "shows the telemetry events the aggregator did not count, by who sent them and why", %{conn: conn} do
      # Counted in the node's telemetry window, inside the test's transaction.
      :ok =
        Atlas.Metrics.write!([
          %{
            scope: :node,
            family: "telemetry",
            values: [{"missed:chatbot-semantic-fact-used:aggregator_not_running", :sum, 2}]
          }
        ])

      {:ok, view, _html} = live(conn, "/dashboard")
      missed = view |> element("#telemetry-missed-events") |> render()

      assert missed =~ "telemetry event(s) not counted"
      assert missed =~ ~r/>\s*2\s*<\/span> from chatbot-semantic-fact-used: the aggregator was not running/
    end

    test "shows the activity the aggregator counted: KG signals, learning, consistency, racing early exits and code languages",
         %{conn: conn} do
      :ok =
        Atlas.Metrics.write!([
          %{
            scope: :node,
            family: "telemetry",
            values: [
              {"kg_signal:scored:count", :sum, 3},
              {"kg_signal:scored:items", :sum, 7},
              {"learning:document_processed", :sum, 2},
              {"consistency:total", :sum, 1},
              {"consistency:high", :sum, 1},
              {"racing_early_exit:fast_path", :sum, 4},
              {"code_language:elixir", :sum, 5}
            ]
          }
        ])

      {:ok, view, _html} = live(conn, "/dashboard")

      assert view |> element("#activity-kg-signals") |> render() =~ ~r/scored.*3.*7 items/s
      assert view |> element("#activity-learning") |> render() =~ ~r/document_processed.*2/s
      assert view |> element("#activity-consistency") |> render() =~ ~r/high.*1/s
      assert view |> element("#activity-racing-early-exits") |> render() =~ ~r/fast_path.*4/s
      assert view |> element("#activity-code-languages") |> render() =~ ~r/elixir.*5/s
    end

    test "a refresh reads the node's telemetry window from Atlas once", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")
      test_pid = self()
      handler = "dashboard-test-telemetry-reads"

      # The reads of the node's metric windows the LiveView itself makes.
      :ok =
        :telemetry.attach(
          handler,
          [:atlas, :repo, :query],
          fn _event, _measurements, %{source: source, params: params}, live_view ->
            if self() == live_view and source == "atlas_metric_windows" and "telemetry" in params do
              send(test_pid, :telemetry_window_read)
            end
          end,
          view.pid
        )

      on_exit(fn -> :telemetry.detach(handler) end)

      send(view.pid, :refresh_dashboard)
      _html = render(view)
      :telemetry.detach(handler)

      assert_received :telemetry_window_read
      refute_received :telemetry_window_read
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
