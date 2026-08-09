defmodule Atlas.Schemas.ServiceRecord do
  @moduledoc """
  Append-only, agent-centric accountable record — the milestone-level service
  history of one soul: commissioning, order outcomes, dissents, reliefs,
  reinstatements, achievements, rehydrations, retirement.

  Distinct from `Atlas.Schemas.CommandRecord` (the per-MESSAGE command-channel
  audit): this is the per-AGENT service history keyed by the stable `soul_id`.
  Written by the Fleet *runtime* only (`Fleet.Service`) — an agent can neither
  forge nor scrub it (same integrity property as CommandRecord).
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_service_records" do
    field :soul_id, :string
    field :agent_id, :string
    field :kind, :string
    field :order_id, :string
    field :world_id, :string
    field :under_order_of, :string
    field :outcome, :string
    field :reason, :string
    field :authored_by, :string, default: "runtime"
    field :payload, :map, default: %{}
    field :occurred_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @valid_kinds ~w(commissioned order_assigned order_blocked order_completed
                  order_dissented order_failed relieved reinstated achievement
                  rehydrated retired decommissioned)

  @required_fields ~w(soul_id kind)a
  @optional_fields ~w(agent_id order_id world_id under_order_of outcome reason
                      authored_by payload occurred_at)a

  def valid_kinds, do: @valid_kinds

  def changeset(record, attrs) do
    record
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:kind, @valid_kinds)
  end

  @doc "Service history for a soul, oldest first (the career)."
  def for_soul(query \\ __MODULE__, soul_id),
    do: from(r in query, where: r.soul_id == ^soul_id, order_by: [asc: r.inserted_at])

  def of_kind(query \\ __MODULE__, kind), do: from(r in query, where: r.kind == ^kind)
end
