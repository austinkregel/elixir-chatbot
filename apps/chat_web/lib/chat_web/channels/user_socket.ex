defmodule ChatWeb.UserSocket do
  @moduledoc """
  User socket for WebSocket connections.
  """

  use Phoenix.Socket

  ## Channels
  channel "brain:*", ChatWeb.BrainChannel

  @impl true
  def connect(_params, socket, _connect_info) do
    {:ok, socket}
  end

  @impl true
  def id(_socket), do: nil
end
