defmodule Atlas.Schemas.CommandRecord do
  @moduledoc """
  Append-only audit record of a single command-channel message.

  Every ORDER / ACK / SITREP / REQUEST / GRANT / DENY / DISSENT / REPORT /
  RELIEVE / REINSTATE / provenance-anomaly is written here by the Fleet
  *runtime* (not by agents), so an agent can neither forge a superior's order
  nor scrub its own record. `from_agent` is always the principal the runtime
  *attributed* via the Registry — never a value asserted in a message payload.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_command_records" do
    field :kind, :string
    field :order_id, :string
    field :from_agent, :string
    field :to_agent, :string
    field :world_id, :string
    field :authority, :string
    field :verdict, :string
    field :reason, :string
    field :payload, :map, default: %{}
    field :issued_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @valid_kinds ~w(order ack sitrep request grant deny dissent report relieve reinstate provenance_anomaly)

  @required_fields ~w(kind from_agent)a
  @optional_fields ~w(order_id to_agent world_id authority verdict reason payload issued_at)a

  def valid_kinds, do: @valid_kinds

  def changeset(record, attrs) do
    record
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:kind, @valid_kinds)
  end

  @doc "Query records for an order, oldest first (the order's lifecycle)."
  def for_order(query \\ __MODULE__, order_id) do
    from(r in query, where: r.order_id == ^order_id, order_by: [asc: r.inserted_at])
  end

  @doc "Query records of a given kind."
  def of_kind(query \\ __MODULE__, kind) do
    from(r in query, where: r.kind == ^kind)
  end
end
