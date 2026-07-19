defmodule Atlas.Schemas.ServiceSummary do
  @moduledoc """
  The durable rehydration snapshot for one soul: its standing chain position,
  standing grants, current assignment, duty status, mind-world binding, and
  career counters. One row per soul (upserted by the runtime). This is the
  projection an ensign reads on restart to rebuild its durable self in a single
  query; it is rebuildable from `atlas_service_records` and is never the record
  of truth.

  On court martial (a later phase) the soul + memory are deleted; a sterilized
  stub of this row (sterile service facts only) is what remains.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_service_summaries" do
    field :soul_id, :string
    field :ship_id, :string
    field :duty_status, :string, default: "active"
    field :rank, :string, default: "ensign"
    field :home_world_id, :string
    field :mind_world_id, :string
    field :co_id, :string
    field :reports, {:array, :string}, default: []
    field :standing_grants, {:array, :string}, default: []
    field :current_assignment, :map, default: %{}
    field :served_since, :utc_datetime_usec
    field :last_commissioned_at, :utc_datetime_usec
    field :orders_completed, :integer, default: 0
    field :orders_dissented, :integer, default: 0
    field :orders_failed, :integer, default: 0
    field :reliefs, :integer, default: 0
    field :achievements, {:array, :map}, default: []
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(soul_id)a
  @optional_fields ~w(ship_id duty_status rank home_world_id mind_world_id co_id reports
                      standing_grants current_assignment served_since
                      last_commissioned_at orders_completed orders_dissented
                      orders_failed reliefs achievements metadata)a

  def changeset(summary, attrs) do
    summary
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:duty_status, ~w(active relieved))
    |> unique_constraint(:soul_id)
  end

  def for_soul(query \\ __MODULE__, soul_id), do: from(s in query, where: s.soul_id == ^soul_id)
end
