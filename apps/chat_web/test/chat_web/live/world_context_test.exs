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

    setup_world_sandbox()

    :ok
  end

  # A world the world selector lists. `World.Manager.list_worlds/0` leaves out
  # worlds marked as test worlds, and only listed worlds can be selected, so
  # this one is created without that mark. Its id starts with `test_`, it is
  # ephemeral, so nothing of it is on disk, and it is destroyed when the test
  # ends. Create it before mounting a page: the selector offers the worlds
  # loaded when the page mounted.
  defp create_listed_world(name) do
    world_id = "test_#{name}_#{System.unique_integer([:positive])}"
    {:ok, world} = WorldManager.create(world_id, id: world_id, mode: :ephemeral)
    on_exit(fn -> WorldManager.destroy(world.id) end)
    world
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

      # A world of this test's own, created with its embedder, so nothing is
      # written to a real world.
      {:ok, world} = create_test_world("episode_store")

      unique_state = "Test episode #{:erlang.unique_integer([:positive])}"

      assert {:ok, _id} =
               MemoryStore.add_episode(
                 unique_state,
                 "test_action",
                 "test_outcome",
                 ["test"],
                 world_id: world.id
               )

      # The world holds that episode and nothing else.
      {:ok, episodes} = MemoryStore.all_episodes(world_id: world.id)
      assert [%{state: ^unique_state}] = episodes
    end

    test "episodes in world A are NOT visible in world B", %{conn: conn} do
      # Ensure dashboard is loaded
      {:ok, _view, _html} = live(conn, "/dashboard")

      # Two worlds of this test's own, each created with its embedder.
      {:ok, world_a} = create_test_world("world_a")
      {:ok, world_b} = create_test_world("world_b")
      world_a_id = world_a.id
      world_b_id = world_b.id

      unique_state_a = "Episode only in world A #{:erlang.unique_integer([:positive])}"
      unique_state_b = "Episode only in world B #{:erlang.unique_integer([:positive])}"

      # Add episode to world A
      assert {:ok, _} =
               MemoryStore.add_episode(
                 unique_state_a,
                 "test_action",
                 "test_outcome",
                 ["isolation_test"],
                 world_id: world_a_id
               )

      # Add episode to world B
      assert {:ok, _} =
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
    setup do
      %{other_world: create_listed_world("world_context_switch")}
    end

    test "can switch to another existing world", %{conn: conn, other_world: other_world} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Switch to the other world
      html =
        view
        |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
        |> render_change()

      # UI should reflect the change
      assert html =~ other_world.id
      assert has_element?(view, ~s(#world-select option[value="#{other_world.id}"][selected]))
      refute has_element?(view, ~s(#world-select option[value="default"][selected]))
    end

    test "can switch back to default from another world", %{conn: conn, other_world: other_world} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Switch to other world first
      view
      |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
      |> render_change()

      assert has_element?(view, ~s(#world-select option[value="#{other_world.id}"][selected]))

      # Now switch back to default
      view
      |> form("form[phx-change=switch_world]", %{world_id: "default"})
      |> render_change()

      # Should show default
      assert has_element?(view, ~s(#world-select option[value="default"][selected]))
      refute has_element?(view, ~s(#world-select option[value="#{other_world.id}"][selected]))
    end

    test "switching worlds updates world models section", %{conn: conn, other_world: other_world} do
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

  describe "the selected world belongs to one web user" do
    setup do
      %{other: create_listed_world("world_context_other")}
    end

    test "switching in one user's session leaves another user's world alone", %{other: other} do
      alice = get(build_conn(), "/worlds")
      bob = get(build_conn(), "/worlds")
      bob_id = get_session(bob, "user_id")

      {:ok, alice_view, _html} = live(alice)
      {:ok, bob_view, _html} = live(bob)

      initial = ChatWeb.Worlds.initial_world_id()
      assert bob_view |> element("#web-user-world") |> render() =~ initial

      alice_view |> element("#select-world-#{other.id}") |> render_click()

      assert alice_view |> element("#web-user-world") |> render() =~ other.id
      assert bob_view |> element("#web-user-world") |> render() =~ initial
      refute bob_view |> element("#web-user-world") |> render() =~ other.id
      assert Atlas.WebUsers.get!(bob_id).selected_world_id == initial
    end

    test "switching on one page moves the same user's other open pages", %{other: other} do
      conn = get(build_conn(), "/worlds")

      {:ok, first_page, _html} = live(conn)
      {:ok, second_page, _html} = live(recycle(conn), "/worlds")

      first_page |> element("#select-world-#{other.id}") |> render_click()

      assert second_page |> element("#web-user-world") |> render() =~ other.id
    end
  end

  describe "flash messages" do
    setup do
      %{other_world: create_listed_world("world_context_flash")}
    end

    test "switching world shows confirmation flash", %{conn: conn, other_world: other_world} do
      {:ok, view, _html} = live(conn, "/dashboard")

      # Switch to another world
      view
      |> form("form[phx-change=switch_world]", %{world_id: other_world.id})
      |> render_change()

      html = render(view)

      # Flash message should appear
      assert html =~ "Switched to world: #{other_world.id}"
    end
  end
end
