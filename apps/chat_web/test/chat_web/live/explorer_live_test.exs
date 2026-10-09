defmodule ChatWeb.ExplorerLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  setup do
    ensure_pubsub_started()

    # Start required GenServers
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.Memory.Store)
    ensure_started(World.Manager)

    :ok
  end

  describe "mount" do
    test "mounts successfully with default world", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      assert html =~ "Data Explorer"
      assert html =~ "default"
    end

    test "displays page header", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      assert html =~ "Data Explorer"
      assert html =~ "Explore data in world"
    end

    test "shows tab navigation", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      # Should have all tab buttons
      assert html =~ "Entities"
      assert html =~ "Candidates"
      assert html =~ "Episodes"
      assert html =~ "Semantics"
      assert html =~ "Knowledge"
    end

    test "shows search input", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      assert html =~ "Search..."
    end
  end

  describe "tab switching" do
    test "can switch to candidates tab", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/explorer")

      view |> element("button", "Candidates") |> render_click()

      # URL should update with tab param
      assert_patched(view, "/explorer?tab=candidates")
    end

    test "can switch to episodes tab", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/explorer")

      view |> element("button", "Episodes") |> render_click()

      assert_patched(view, "/explorer?tab=episodes")
    end

    test "can switch to semantics tab", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/explorer")

      view |> element("button", "Semantics") |> render_click()

      assert_patched(view, "/explorer?tab=semantics")
    end

    test "can switch to knowledge tab", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/explorer")

      view |> element("button", "Knowledge") |> render_click()

      assert_patched(view, "/explorer?tab=knowledge")
    end
  end

  describe "semantics tab" do
    test "shows a fact's text and the action it records", %{conn: conn} do
      fact =
        Brain.Memory.Types.SemanticFact.new("is it sunny in paris | is it raining in paris", [], [], [],
          action: "weather.query"
        )

      {:ok, _id} = Brain.Memory.Store.add_semantic(fact, world_id: "default")

      # The rows roll back with the test; the store's index is cleared with them.
      on_exit(fn -> Brain.Memory.Store.clear(world_id: "default") end)

      {:ok, _view, html} = live(conn, "/explorer?tab=semantics")

      assert html =~ "is it sunny in paris | is it raining in paris"
      assert html =~ "Action: weather.query"
      refute html =~ "Examples:"
    end
  end

  describe "URL params" do
    test "respects tab param in URL", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer?tab=episodes")

      # Should be on episodes tab
      assert html =~ "Episodes"
    end

    test "displays search input on entities tab", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/explorer")

      # Search input should be present
      assert html =~ ~r/input.*Search\.\.\./s or html =~ "Search..."
    end
  end

  describe "refresh" do
    test "refresh button works", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/explorer")

      # Click refresh button
      view |> element("button", "Refresh") |> render_click()

      # Should still be on explorer page
      html = render(view)
      assert html =~ "Data Explorer"
    end
  end
end
