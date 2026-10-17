defmodule Atlas.Schemas.MetricWindow do
  @moduledoc """
  One measurement window of a family of metrics (`Atlas.Metrics`) in one
  scope: the values and records written from `started_at` until `ended_at`.
  `world_id` names the world the window measures, or is `nil` for a window of
  the whole node. The open window of a family and scope has no `ended_at` and
  is the one being written to; there is at most one.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  schema "atlas_metric_windows" do
    field :world_id, :string
    field :family, :string
    field :started_at, :utc_datetime_usec
    field :ended_at, :utc_datetime_usec
  end

  @doc false
  def changeset(window, attrs) do
    window
    |> cast(attrs, [:world_id, :family, :started_at, :ended_at])
    |> validate_required([:family, :started_at])
    |> validate_change(:world_id, &not_blank/2)
    |> validate_change(:family, &not_blank/2)
    |> unique_constraint([:family, :world_id], name: :atlas_metric_windows_open_index)
  end

  defp not_blank(field, value) do
    if String.trim(value) == "", do: [{field, "must not be blank"}], else: []
  end
end
