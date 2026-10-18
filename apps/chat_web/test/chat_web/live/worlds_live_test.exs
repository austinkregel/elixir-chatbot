defmodule ChatWeb.WorldsLiveTest do
  @moduledoc """
  The `/worlds` page shows this browser's web user and the world it has
  selected, and selecting a world there stores it on that user.
  """
  use ChatWeb.ConnCase, async: false
  import Phoenix.LiveViewTest
  import Brain.TestHelpers

  alias Atlas.WebUsers
  alias World.Manager, as: WorldManager

  setup do
    ensure_pubsub_started()
    ensure_started(World.Manager)
    ensure_started(Brain.Worlds.ModelRegistry)

    # A world the page lists. `World.Manager.list_worlds/0` leaves out worlds
    # marked as test worlds, so this one is created without that mark. Its id
    # starts with `test_`, it is ephemeral, so nothing of it is on disk, and it
    # is destroyed when the test ends.
    other_id = "test_worlds_live_other_#{System.unique_integer([:positive])}"
    {:ok, other} = WorldManager.create(other_id, id: other_id, mode: :ephemeral)
    on_exit(fn -> WorldManager.destroy(other.id) end)

    %{other: other}
  end

  test "shows the web user's id and selected world", %{conn: conn} do
    conn = get(conn, "/worlds")
    user_id = get_session(conn, "user_id")

    {:ok, view, _html} = live(conn)

    assert view |> element("#web-user-id") |> render() =~ user_id
    assert view |> element("#web-user-world") |> render() =~ ChatWeb.Worlds.initial_world_id()
    assert has_element?(view, "#world-#{ChatWeb.Worlds.initial_world_id()}")
  end

  test "selecting a world stores it on the user and the page shows it", %{conn: conn, other: other} do
    conn = get(conn, "/worlds")
    user_id = get_session(conn, "user_id")
    {:ok, view, _html} = live(conn)

    view |> element("#select-world-#{other.id}") |> render_click()

    assert WebUsers.get!(user_id).selected_world_id == other.id
    assert view |> element("#web-user-world") |> render() =~ other.id
    refute has_element?(view, "#select-world-#{other.id}")

    # A reload is the same user with the same selection.
    {:ok, reloaded, _html} = live(recycle(conn), "/worlds")
    assert reloaded |> element("#web-user-id") |> render() =~ user_id
    assert reloaded |> element("#web-user-world") |> render() =~ other.id
  end

  describe "your profile" do
    setup do
      ensure_started(Brain.Epistemic.UserModelStore)

      # User models live in GenServer state, which the Sandbox does not roll
      # back; empty is the boot state (0 rows in atlas_user_models).
      Brain.Epistemic.UserModelStore.clear_all()
      on_exit(fn -> Brain.Epistemic.UserModelStore.clear_all() end)
    end

    test "lists the facts held about the web user", %{conn: conn} do
      conn = get(conn, "/worlds")
      user_id = get_session(conn, "user_id")

      {:ok, _} = Brain.Epistemic.UserModelStore.update_fact(user_id, :location, "Seattle", :explicit, 0.9)
      {:ok, _} = Brain.Epistemic.UserModelStore.update_fact(user_id, :occupation, "Engineer", :inferred, 0.6)

      {:ok, view, _html} = live(conn)

      location = view |> element("#profile-fact-location") |> render()
      assert location =~ "Seattle"
      assert location =~ "0.90"
      assert location =~ "explicit"

      occupation = view |> element("#profile-fact-occupation") |> render()
      assert occupation =~ "Engineer"
      assert occupation =~ "inferred"
    end

    test "is the same whichever world is selected", %{conn: conn, other: other} do
      conn = get(conn, "/worlds")
      user_id = get_session(conn, "user_id")
      {:ok, _} = Brain.Epistemic.UserModelStore.update_fact(user_id, :location, "Seattle", :explicit, 0.9)

      {:ok, view, _html} = live(conn)
      view |> element("#select-world-#{other.id}") |> render_click()

      assert view |> element("#web-user-world") |> render() =~ other.id
      assert view |> element("#profile-fact-location") |> render() =~ "Seattle"
    end

    test "shows another user's facts to no one else", %{conn: conn} do
      first = get(conn, "/worlds")
      {:ok, _} =
        Brain.Epistemic.UserModelStore.update_fact(get_session(first, "user_id"), :location, "Seattle", :explicit, 0.9)

      {:ok, view, _html} = live(get(build_conn(), "/worlds"))

      refute has_element?(view, "#profile-facts")
      assert render(view) =~ "No facts are held about you yet."
    end
  end

  test "a browser without the cookie is a different user", %{conn: conn} do
    first = get(conn, "/worlds")
    second = get(build_conn(), "/worlds")

    first_id = get_session(first, "user_id")
    second_id = get_session(second, "user_id")
    assert first_id != second_id

    {:ok, view, _html} = live(second)
    assert view |> element("#web-user-id") |> render() =~ second_id
    refute view |> element("#web-user-id") |> render() =~ first_id
  end
end
