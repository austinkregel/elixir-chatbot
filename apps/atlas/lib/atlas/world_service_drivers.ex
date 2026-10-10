defmodule Atlas.WorldServiceDrivers do
  @moduledoc """
  Context module for the driver chosen to serve each capability in each world
  (`Atlas.Schemas.WorldServiceDriver`). Every read and write names the world
  and the capability; nothing is assumed for a world that has no row.
  """

  alias Atlas.Repo
  alias Atlas.Schemas.WorldServiceDriver

  @doc "The driver chosen for `capability` in `world_id`, or `:error` when none has been chosen."
  @spec fetch(String.t(), String.t()) :: {:ok, String.t()} | :error
  def fetch(world_id, capability) when is_binary(world_id) and is_binary(capability) do
    case Repo.get_by(WorldServiceDriver, world_id: world_id, capability: capability) do
      %WorldServiceDriver{driver: driver} -> {:ok, driver}
      nil -> :error
    end
  end

  @doc """
  Chooses `driver` for `capability` in `world_id`, replacing the driver chosen
  before. Raises on a blank world or driver, or an unknown capability.
  """
  @spec put!(String.t(), String.t(), String.t()) :: WorldServiceDriver.t()
  def put!(world_id, capability, driver)
      when is_binary(world_id) and is_binary(capability) and is_binary(driver) do
    %WorldServiceDriver{}
    |> WorldServiceDriver.changeset(%{world_id: world_id, capability: capability, driver: driver})
    |> Repo.insert!(
      on_conflict: {:replace, [:driver, :updated_at]},
      conflict_target: [:world_id, :capability],
      returning: true
    )
  end
end
