defmodule Atlas.Schemas.SystemStatusSample do
  @moduledoc """
  A durable black-box sample of subsystem state over time — the flight-recorder
  substrate for the ship's systems.

  Written by `Fleet.Systems.Sampler` (never by agents). Two row kinds keep the
  volume bounded while still giving a continuous trend:

    * a periodic **health rollup** (`system_id: "ship"`, `metric` = health score),
      one per sample — the trend line, and
    * per-system **status transitions** (`system_id` = a system's id) — a row only
      when a system's status actually changes.

  Every row is stamped with `ship_id` so the recorder is filterable per ship for
  the fleet-of-ships future. Append-only; ordered by `sampled_at`.
  """
  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_system_status_samples" do
    field :ship_id, :string
    field :system_id, :string
    field :status, :string
    field :metric, :float
    field :sampled_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec, updated_at: false)
  end

  @required_fields ~w(ship_id system_id status sampled_at)a
  @optional_fields ~w(metric)a

  def changeset(sample, attrs) do
    sample
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
  end

  @doc "Recent samples for one system on a ship, newest first."
  def for_system(query \\ __MODULE__, ship_id, system_id, limit \\ 200) do
    from(s in query,
      where: s.ship_id == ^ship_id and s.system_id == ^system_id,
      order_by: [desc: s.sampled_at],
      limit: ^limit
    )
  end

  @doc "The ship's health-rollup trend, newest first."
  def health_trend(query \\ __MODULE__, ship_id, limit \\ 200) do
    for_system(query, ship_id, "ship", limit)
  end
end
