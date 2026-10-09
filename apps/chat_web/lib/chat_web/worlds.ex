defmodule ChatWeb.Worlds do
  @moduledoc """
  The worlds the web interface offers, as `World.Manager` holds them.

  A web user starts in the world whose id is `"default"`, a real world like any
  other. That is the only place the web interface names a world itself: every
  other world it uses is one the user selected.
  """

  alias World.Manager

  @initial_world_id "default"

  @doc "The world a new web user starts in. Raises if no such world is loaded."
  @spec initial_world_id() :: String.t()
  def initial_world_id do
    fetch!(@initial_world_id)
    @initial_world_id
  end

  @doc "Every world, as `%{id, name, mode}`, sorted by name."
  @spec list() :: [%{id: String.t(), name: String.t(), mode: atom()}]
  def list do
    Manager.list_worlds()
    |> Enum.map(&%{id: &1.id, name: &1.name, mode: &1.mode})
    |> Enum.sort_by(& &1.name)
  end

  @doc "Whether a world with this id is loaded."
  @spec exists?(String.t()) :: boolean()
  def exists?(world_id) when is_binary(world_id) do
    match?({:ok, _}, Manager.get(world_id))
  end

  @doc "The world with this id, or raises naming it."
  def fetch!(world_id) when is_binary(world_id) do
    case Manager.get(world_id) do
      {:ok, world} ->
        world

      {:error, reason} ->
        raise ArgumentError, "world #{inspect(world_id)} is not available: #{inspect(reason)}"
    end
  end
end
