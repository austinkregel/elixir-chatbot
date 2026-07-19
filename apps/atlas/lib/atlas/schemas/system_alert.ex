defmodule Atlas.Schemas.SystemAlert do
  @moduledoc """
  A durable, append-only alert *event* — the ship's alarm log, and the substrate the
  relief-of-duty anomaly monitor / fleet brake consume (LCARS §3 #10).

  Written by `Fleet.Systems.Monitor` (never by agents). Two event kinds keep the log
  append-only: a `raised` row when a system first goes non-nominal (or the ship's
  health crosses a threshold), and a `resolved` row when it recovers. The *active*
  set is the Monitor's in-memory projection; this is the history of record.

  Ship-stamped for the fleet-of-ships future; ordered by `occurred_at`.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_system_alerts" do
    field :ship_id, :string
    field :system_id, :string
    field :severity, :string
    field :kind, :string
    field :message, :string
    field :occurred_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec, updated_at: false)
  end

  @kinds ~w(raised resolved)
  @severities ~w(warning critical)

  @required_fields ~w(ship_id system_id severity kind occurred_at)a
  @optional_fields ~w(message)a

  def changeset(alert, attrs) do
    alert
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:kind, @kinds)
    |> validate_inclusion(:severity, @severities)
  end

  @doc "Recent alert events for a ship, newest first."
  def recent(query \\ __MODULE__, ship_id, limit \\ 100) do
    from(a in query, where: a.ship_id == ^ship_id, order_by: [desc: a.occurred_at], limit: ^limit)
  end
end
