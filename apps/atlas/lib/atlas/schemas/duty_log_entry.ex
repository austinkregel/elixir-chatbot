defmodule Atlas.Schemas.DutyLogEntry do
  @moduledoc """
  The agent's OWN duty log — its working notes/journal while on an order. Unlike
  `atlas_service_records` (runtime-written accountability), the agent MAY write
  here; this is its memory of its own reasoning, not the accountable record. It
  is still append-only: the agent can add notes, never edit or delete prior ones.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_duty_log_entries" do
    field :soul_id, :string
    field :order_id, :string
    field :world_id, :string
    field :note, :string
    field :tags, {:array, :string}, default: []
    field :authored_by, :string, default: "agent"
    field :payload, :map, default: %{}
    field :occurred_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(soul_id note)a
  @optional_fields ~w(order_id world_id tags authored_by payload occurred_at)a

  def changeset(entry, attrs) do
    entry
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
  end

  def for_soul(query \\ __MODULE__, soul_id),
    do: from(e in query, where: e.soul_id == ^soul_id, order_by: [asc: e.inserted_at])

  def for_order(query \\ __MODULE__, order_id),
    do: from(e in query, where: e.order_id == ^order_id, order_by: [asc: e.inserted_at])
end
