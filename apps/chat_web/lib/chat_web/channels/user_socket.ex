defmodule ChatWeb.UserSocket do
  @moduledoc """
  User socket for WebSocket connections.

  A connection belongs to the browser's web user (`ChatWeb.Plugs.CurrentUser`),
  read from the session the socket is opened with; the client sends the page's
  `_csrf_token` so Phoenix hands the session over. A connection with no web user
  is refused. The user id is assigned as `:user_id`.
  """

  use Phoenix.Socket

  alias Atlas.WebUsers

  ## Channels
  channel "brain:*", ChatWeb.BrainChannel

  @impl true
  def connect(_params, socket, %{session: %{"user_id" => user_id}}) do
    WebUsers.get!(user_id)
    {:ok, assign(socket, :user_id, user_id)}
  end

  def connect(_params, _socket, _connect_info) do
    :error
  end

  @impl true
  def id(_socket), do: nil
end
