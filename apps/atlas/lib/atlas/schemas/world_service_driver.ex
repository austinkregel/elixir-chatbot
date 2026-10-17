defmodule Atlas.Schemas.WorldServiceDriver do
  @moduledoc """
  The driver chosen to serve one capability in one world, such as the driver
  that sets and reads the world's alarms, timers, reminders and calendar items
  (capability `"scheduling"`). A world has at most one driver per capability.

  `driver` names the driver; what the name means is the owning module's to say
  (`Brain.Services.Dispatcher`).
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @capabilities ~w(scheduling)

  @type t :: %__MODULE__{}

  schema "atlas_world_service_drivers" do
    field :world_id, :string
    field :capability, :string
    field :driver, :string

    timestamps(type: :utc_datetime_usec)
  end

  @doc "The capabilities a driver can be chosen for."
  def capabilities, do: @capabilities

  @doc false
  def changeset(row, attrs) do
    row
    |> cast(attrs, [:world_id, :capability, :driver])
    |> validate_required([:world_id, :capability, :driver])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_change(:driver, &not_blank/2)
    |> validate_inclusion(:capability, @capabilities)
    |> unique_constraint([:world_id, :capability])
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
