defmodule Atlas.Schemas.Setting do
  @moduledoc """
  A node-wide setting changed at runtime, stored by name so it survives a
  restart. The value is a string; what it means is the owning module's to say.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:name, :string, autogenerate: false}

  @type t :: %__MODULE__{}

  schema "atlas_settings" do
    field :value, :string

    timestamps(type: :utc_datetime_usec)
  end

  @doc false
  def changeset(setting, attrs) do
    setting
    |> cast(attrs, [:name, :value])
    |> validate_required([:name, :value])
    |> validate_change(:name, &not_blank/2)
    |> validate_change(:value, &not_blank/2)
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
