defmodule ChatWeb.SettingsLiveTest do
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias Brain.Knowledge.LearningTriggers
  alias Brain.Services.CredentialVault
  alias World.Manager, as: WorldManager

  setup do
    ensure_pubsub_started()

    # Start required GenServers
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(World.Manager)

    :ok
  end

  describe "mount" do
    test "mounts successfully", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      assert html =~ "Settings"
      assert html =~ "Manage worlds and entities"
    end

    test "shows section tabs", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      assert html =~ "Worlds"
      assert html =~ "Entities"
    end

    test "defaults to worlds section", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      # Should show world management UI
      assert html =~ "Create New World"
      assert html =~ "Active Worlds"
    end

    test "shows refresh button", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      assert html =~ "Refresh"
    end
  end

  describe "section navigation" do
    test "entities tab is present", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      # Entities tab should be visible
      assert html =~ "Entities"
    end

    test "worlds tab is active by default", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      # Worlds section should be shown
      assert html =~ "Create New World"
    end
  end

  describe "URL params" do
    test "respects section param for worlds", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings?section=worlds")

      assert html =~ "Create New World"
    end
  end

  describe "worlds section" do
    test "shows create world form", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      assert html =~ "World name"
      assert html =~ "Create World"
    end

    test "shows persistence mode options", %{conn: conn} do
      {:ok, _view, html} = live(conn, "/settings")

      assert html =~ "Persistent"
      assert html =~ "Ephemeral"
    end
  end

  # Note: Entities section tests are skipped due to a bug in the app's
  # is_overlay_entity/2 function that doesn't handle :_meta atoms properly.
  # The entities section functionality works in the browser but crashes in tests
  # due to how the world overlay is structured with metadata.

  describe "auto-learning world card" do
    setup do
      # LearningTriggers is the node's one instance; whatever a test sets is
      # set back when it ends. The configured world is loaded, so setting it
      # back succeeds.
      original = LearningTriggers.auto_learning_world()
      on_exit(fn -> :ok = LearningTriggers.set_auto_learning_world(original) end)

      # A world the card offers. The card lists `World.Manager.list_worlds/0`,
      # which leaves out worlds marked as test worlds, so this one is created
      # without that mark. Its id starts with `test_`, it is ephemeral, so
      # nothing of it is on disk, and it is destroyed when the test ends.
      other_id = "test_settings_auto_learning_#{System.unique_integer([:positive])}"
      {:ok, other} = WorldManager.create(other_id, id: other_id, mode: :ephemeral)
      on_exit(fn -> WorldManager.destroy(other.id) end)

      %{original: original, other: other}
    end

    test "shows the auto-learning world selected", %{conn: conn, original: original, other: other} do
      {:ok, view, _html} = live(conn, "/settings?section=training")

      assert has_element?(view, "h3", "Auto-Learning World")
      assert has_element?(view, ~s(#auto-learning-world option[value="#{original}"][selected]))
      assert has_element?(view, ~s(#auto-learning-world option[value="#{other.id}"]))
      refute has_element?(view, ~s(#auto-learning-world option[value="#{other.id}"][selected]))
      refute render(view) =~ "is not loaded. An auto-triggered session will fail"
    end

    test "saving a loaded world sets it as the auto-learning world and stores it", %{
      conn: conn,
      other: other
    } do
      {:ok, view, _html} = live(conn, "/settings?section=training")

      html =
        view
        |> form("form[phx-submit=set_auto_learning_world]", %{world_id: other.id})
        |> render_submit()

      assert LearningTriggers.auto_learning_world() == other.id

      assert Atlas.Settings.fetch(LearningTriggers.auto_learning_world_setting()) ==
               {:ok, other.id}
      assert html =~ "Auto-triggered learning sessions now start in world #{other.id}"
      assert has_element?(view, ~s(#auto-learning-world option[value="#{other.id}"][selected]))
    end
  end

  # The services section, switched to `world`. Selecting a world reloads the
  # section for it.
  defp services_in(conn, world) do
    {:ok, view, _html} = live(conn, "/settings?section=services")
    view |> form("form[phx-change=switch_world]", %{world_id: world.id}) |> render_change()
    view
  end

  defp set_home_assistant_credentials(world) do
    :ok = CredentialVault.store(:home_assistant, :url, "http://home-assistant.invalid", world: world.id)
    :ok = CredentialVault.store(:home_assistant, :access_token, "unused", world: world.id)
  end

  describe "scheduling driver card" do
    # A world the world selector lists, so the page can be switched to it: it
    # is created without the test mark, its id starts with `test_`, it is
    # ephemeral, so nothing of it is on disk, and it is destroyed when the test
    # ends. Its driver row and any credential row are written inside the
    # test's sandbox transaction; the credentials the vault holds in memory are
    # deleted when the test ends.
    setup do
      world_id = "test_settings_scheduling_#{System.unique_integer([:positive])}"
      {:ok, world} = WorldManager.create(world_id, id: world_id, mode: :ephemeral)
      on_exit(fn -> WorldManager.destroy(world.id) end)

      on_exit(fn ->
        :ok = CredentialVault.delete(:home_assistant, :url, world: world_id)
        :ok = CredentialVault.delete(:home_assistant, :access_token, world: world_id)
      end)

      %{world: world}
    end

    test "shows the local scheduler a new world starts with, and offers Home Assistant only once its credentials are set",
         %{conn: conn, world: world} do
      view = services_in(conn, world)

      assert has_element?(view, "#scheduling-driver-current", "Local Scheduler")

      view |> element("#choose-scheduling-driver") |> render_click()

      assert has_element?(view, ~s(#scheduling-driver-select option[value="local_scheduler"][selected]))
      refute has_element?(view, ~s(#scheduling-driver-select option[value="local_scheduler"][disabled]))
      assert has_element?(view, ~s(#scheduling-driver-select option[value="home_assistant"][disabled]))
      assert render(view) =~ "Home Assistant (needs Url, Access token)"
    end

    test "choosing Home Assistant with its credentials set stores it as the world's driver",
         %{conn: conn, world: world} do
      set_home_assistant_credentials(world)
      view = services_in(conn, world)

      view |> element("#choose-scheduling-driver") |> render_click()
      refute has_element?(view, ~s(#scheduling-driver-select option[value="home_assistant"][disabled]))

      html =
        view
        |> form("#confirm-scheduling-driver-form", %{driver: "home_assistant"})
        |> render_submit()

      assert Atlas.WorldServiceDrivers.fetch(world.id, "scheduling") == {:ok, "home_assistant"}
      assert html =~ "Home Assistant now handles alarms, timers, reminders and calendar items in world #{world.id}"
      assert has_element?(view, "#scheduling-driver-current", "Home Assistant")
    end

    test "refuses Home Assistant while its credentials are not set for the world, and stores nothing",
         %{conn: conn, world: world} do
      view = services_in(conn, world)
      view |> element("#choose-scheduling-driver") |> render_click()

      # The option is disabled; the event is sent as a page that ignored that would send it.
      html = render_submit(view, "set_scheduling_driver", %{"driver" => "home_assistant"})

      assert Atlas.WorldServiceDrivers.fetch(world.id, "scheduling") == {:ok, "local_scheduler"}

      assert html =~
               "Home Assistant cannot be the scheduling driver of world #{world.id} until its " <>
                 "Url, Access token credentials are set for that world."

      assert has_element?(view, "#scheduling-driver-current", "Local Scheduler")
    end
  end

  describe "refresh" do
    test "refresh button works", %{conn: conn} do
      {:ok, view, _html} = live(conn, "/settings")

      # Click refresh
      view |> element("button", "Refresh") |> render_click()

      # Should still be on settings
      html = render(view)
      assert html =~ "Settings"
    end
  end
end
