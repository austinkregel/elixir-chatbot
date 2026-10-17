defmodule ChatWeb.Plugs.CurrentUser do
  @moduledoc """
  Gives every browser a web user (`Atlas.WebUsers`) and keeps its id in the
  session as `"user_id"`.

  On a browser's first visit the user is created in
  `ChatWeb.Worlds.initial_world_id/0`. On later visits the id comes back in the
  signed session cookie, and the user must still exist: a session naming a user
  with no row raises rather than quietly starting someone new.
  """

  import Plug.Conn

  alias Atlas.WebUsers
  alias ChatWeb.Worlds

  def init(opts), do: opts

  def call(conn, _opts) do
    case get_session(conn, "user_id") do
      nil ->
        user = WebUsers.create!(Worlds.initial_world_id())
        put_session(conn, "user_id", user.id)

      user_id ->
        WebUsers.get!(user_id)
        conn
    end
  end
end
