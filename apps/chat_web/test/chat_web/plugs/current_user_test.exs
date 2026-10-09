defmodule ChatWeb.Plugs.CurrentUserTest do
  @moduledoc """
  Every browser gets a web user, kept in its session as `"user_id"`.

  The first request creates the user in the world a new user starts in; a later
  request carrying that session is the same user; and a session naming a user
  with no row raises instead of quietly starting someone new.
  """
  use ChatWeb.ConnCase, async: false
  import Brain.TestHelpers

  alias Atlas.WebUsers
  alias ChatWeb.Plugs.CurrentUser

  setup do
    ensure_pubsub_started()
    ensure_started(World.Manager)

    :ok
  end

  describe "call/2" do
    test "the first request creates a user in the initial world and stores its id", %{conn: conn} do
      conn = conn |> init_test_session(%{}) |> CurrentUser.call(CurrentUser.init([]))

      user_id = get_session(conn, "user_id")

      assert is_binary(user_id)
      assert WebUsers.get!(user_id).selected_world_id == ChatWeb.Worlds.initial_world_id()
    end

    test "a request carrying the session keeps the same user", %{conn: conn} do
      user = WebUsers.create!(ChatWeb.Worlds.initial_world_id())

      conn =
        conn
        |> init_test_session(%{"user_id" => user.id})
        |> CurrentUser.call(CurrentUser.init([]))

      assert get_session(conn, "user_id") == user.id
    end

    test "a session naming a user with no row raises", %{conn: conn} do
      conn = init_test_session(conn, %{"user_id" => Ecto.UUID.generate()})

      assert_raise Ecto.NoResultsError, fn -> CurrentUser.call(conn, CurrentUser.init([])) end
    end
  end

  describe "in the browser pipeline" do
    test "the session cookie brings the same user back on the next request", %{conn: conn} do
      first = get(conn, "/")
      user_id = get_session(first, "user_id")
      assert is_binary(user_id)

      second = first |> recycle() |> get("/")

      assert get_session(second, "user_id") == user_id
    end

    test "a request without the cookie is a different user", %{conn: conn} do
      first = get(conn, "/")
      second = get(build_conn(), "/")

      assert get_session(first, "user_id") != get_session(second, "user_id")
    end
  end
end
