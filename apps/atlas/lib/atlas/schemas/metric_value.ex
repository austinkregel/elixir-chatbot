defmodule Atlas.Schemas.MetricValue do
  @moduledoc """
  One named value of one measurement window (`Atlas.Metrics`). `name` says
  what was measured; what the name means is the writer's to say
  (`Brain.Metrics.Aggregator`). `aggregate` says how a new value combines with
  it: `"sum"` adds it, `"min"` keeps the smaller, `"max"` the larger. A
  `"sum"` row exists only once something was counted, so it is positive.

  Rows are written by `Atlas.Metrics.write!/1` only, which combines the values
  in the database.
  """

  use Ecto.Schema

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  @type t :: %__MODULE__{}

  schema "atlas_metric_values" do
    field :world_id, :string
    field :name, :string
    field :aggregate, :string
    field :value, :integer

    belongs_to :window, Atlas.Schemas.MetricWindow

    timestamps(type: :utc_datetime_usec)
  end
end
