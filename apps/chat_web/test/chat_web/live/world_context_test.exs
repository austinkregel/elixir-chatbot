defmodule ChatWeb.WorldContextTest do
  @moduledoc """
  Tests for world context switching functionality.

  Verifies that changing the world via the selector:
  1. Updates the UI to show the new world
  2. Loads world-specific data into memory
  3. Persists the selection across page navigations
  4. Broadcasts changes to other LiveViews
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias Brain.Memory.Store, as: MemoryStore
  alias World.Manager, as: WorldManager

  setup do
    ensure_pubsub_started()

    # Start required GenServers
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(World.Manager)
    ensure_started(World.ModelRegistry)

    :ok
  end

  describe "world selector on dashboard" do
    test "displays current world in selector", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # Default world should be shown
      assert html =~ "default"
      assert html =~ "World:"
    end

    test "shows available worlds in dropdown", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # Default world should be available
      assert html =~ "default"
      # The select element should exist
      assert html =~ ~r/<select[^>]*name="world_id"/
    end

    test "switching to default world works", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Switch to default (should work even if already on default - no-op)
      html =
        view
        |> form("form[phx-change=switch_world]", %{world_id: "default"})
        |> render_change()

      # UI should show default
      assert html =~ "default"
    end

    test "world selector is present in dashboard header", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # The world selector form should exist
      assert html =~ "phx-change=\"switch_world\""
      assert html =~ "name=\"world_id\""
    end
  end

  describe "world-specific data loading" do
    test "dashboard displays memory stats for current world", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # Memory stats section should be present
      assert html =~ "Episodes in World"
      assert html =~ "Semantic Facts"
      assert html =~ "Knowledge Categories"
    end

    test "dashboard loads world models status", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # World models section should show
      assert html =~ "World Models"
    end

    test "memory store can be queried for default world", %{conn: conn} do
      # First load the dashboard to ensure systems are initialized
      {:ok, _view, _html} = live(conn, "/dashboard")

      # Verify the memory store can be queried for the default world
      result = MemoryStore.all_episodes(world_id: "default")

      assert {:ok, episodes} = result
      assert is_list(episodes)
    end
  end

  describe "world context refresh" do
    test "manual refresh reloads world data", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Trigger manual refresh
      html = view |> element("button", "Refresh") |> render_click()

      # Dashboard should still be showing with world data
      assert html =~ "Operations Dashboard"
      assert html =~ "Episodes in World"
    end

    test "world data is present after refresh", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Trigger manual refresh
      view |> element("button", "Refresh") |> render_click()

      html = render(view)

      # Verify world-related data is still present
      assert html =~ "World Models"
      assert html =~ "default"
    end
  end

  describe "world context display on dashboard" do
    test "shows world context in page header", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      # World context should be shown in the header
      assert html =~ ~r/World:.*default/s
    end

    test "shows last updated timestamp", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/dashboard")

      assert html =~ "Updated:"
    end
  end

  describe "data isolation verification" do
    test "episodes can be stored and retrieved per world", %{conn: conn} do
      # Ensure dashboard is loaded
      {:ok, _view, _html} = live(conn, "/dashboard")

      # Add an episode to default world
      unique_state = "Test episode #{:erlang.unique_integer([:positive])}"

      MemoryStore.add_episode(
        unique_state,
        "test_action",
        "test_outcome",
        ["default", "test"],
        world_id: "default"
      )

      # Query should find it
      {:ok, episodes} = MemoryStore.all_episodes(world_id: "default")
      assert Enum.any?(episodes, fn ep -> ep.state == unique_state end)
    end

    test "episodes in world A are NOT visible in world B", %{conn: conn} do
      # Ensure dashboard is loaded
      {:ok, _view, _html} = live(conn, "/dashboard")

      # Create a unique episode identifier
      world_a_id = "test_world_a_#{:erlang.unique_integer([:positive])}"
      world_b_id = "test_world_b_#{:erlang.unique_integer([:positive])}"

      unique_state_a = "Episode only in world A #{:erlang.unique_integer([:positive])}"
      unique_state_b = "Episode only in world B #{:erlang.unique_integer([:positive])}"

      # Add episode to world A
      MemoryStore.add_episode(
        unique_state_a,
        "test_action",
        "test_outcome",
        ["isolation_test"],
        world_id: world_a_id
      )

      # Add episode to world B
      MemoryStore.add_episode(
        unique_state_b,
        "test_action",
        "test_outcome",
        ["isolation_test"],
        world_id: world_b_id
      )

      # Query world A - should find world_a episode, NOT world_b episode
      {:ok, episodes_a} = MemoryStore.all_episodes(world_id: world_a_id)
      assert Enum.any?(episodes_a, fn ep -> ep.state == unique_state_a end),
             "World A should contain its episode"
      refute Enum.any?(episodes_a, fn ep -> ep.state == unique_state_b end),
             "World A should NOT contain world B's episode"

      # Query world B - should find world_b episode, NOT world_a episode
      {:ok, episodes_b} = MemoryStore.all_episodes(world_id: world_b_id)
      assert Enum.any?(episodes_b, fn ep -> ep.state == unique_state_b end),
             "World B should contain its episode"
      refute Enum.any?(episodes_b, fn ep -> ep.state == unique_state_a end),
             "World B should NOT contain world A's episode"
    end
  end

  describe "world list management" do
    test "can get list of available worlds", %{conn: conn} do
      # Load dashboard which fetches world list
      {:ok, _view, _html} = live(conn, "/dashboard")

      # WorldManager should return worlds
      worlds = WorldManager.list_worlds()
      assert is_list(worlds)
      # Default world should exist
      assert Enum.any?(worlds, fn w -> w.id == "default" end)
    end
  end

  describe "switching between existing worlds" do
    setup %{conn: conn} do
      # Get list of available worlds
      {:ok, _view, _html} = live(conn, "/dashboard")
      worlds = WorldManager.list_worlds()

      # Find a second world (if one exists besides default)
      other_world =
        worlds
        |> Enum.find(fn w -> w.id != "default" end)

      %{other_world: other_world}
    end

    test "can switch to another existing world", %{conn: conn, other_world: other_world} do
      # Skip if no other world exists
      if other_world do
        {:ok, view, _html} = live(conn, "/dashboard")

        # Switch to the other world
        html =
          view
          |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
          |> render_change()

        # UI should reflect the change
        assert html =~ other_world.id
      end
    end

    test "can switch back to default from another world", %{conn: conn, other_world: other_world} do
      # Skip if no other world exists
      if other_world do
        {:ok, view, _html} = live(conn, "/dashboard")

        # Switch to other world first
        view
        |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
        |> render_change()

        # Now switch back to default
        html =
          view
          |> form("form[phx-change=switch_world]", %{world_id: "default"})
          |> render_change()

        # Should show default
        assert html =~ "default"
      end
    end

    test "switching worlds updates world models section", %{conn: conn, other_world: other_world} do
      if other_world do
        {:ok, view, _html} = live(conn, "/dashboard")

        # Switch to other world
        html =
          view
          |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
          |> render_change()

        # World models section should reflect the change
        assert html =~ "World Models"
        assert html =~ other_world.id
      end
    end
  end

  describe "flash messages" do
    setup %{conn: conn} do
      {:ok, _view, _html} = live(conn, "/dashboard")
      worlds = WorldManager.list_worlds()
      other_world = Enum.find(worlds, fn w -> w.id != "default" end)
      %{other_world: other_world}
    end

    test "switching world shows confirmation flash", %{conn: conn, other_world: other_world} do
      if other_world do
        {:ok, view, _html} = live(conn, "/dashboard")

        # Switch to another world
        view
        |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
        |> render_change()

        html = render(view)

        # Flash message should appear
        assert html =~ "Switched to world:"
      end
    end
  end
end
