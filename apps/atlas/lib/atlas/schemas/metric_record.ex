defmodule Atlas.Schemas.MetricRecord do
  @moduledoc """
  The latest record of one name in one measurement window (`Atlas.Metrics`),
  such as the last training run of a model: `record` is the map written, as
  JSON, at `recorded_at`. Each newer record of the name replaces it. What the
  name and the record's fields mean is the writer's to say
  (`Brain.Metrics.Aggregator`).

  Rows are written by `Atlas.Metrics.write!/1` only.
  """

  use Ecto.Schema

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  @type t :: %__MODULE__{}

  schema "atlas_metric_records" do
    field :world_id, :string
    field :name, :string
    field :record, :map
    field :recorded_at, :utc_datetime_usec

    belongs_to :window, Atlas.Schemas.MetricWindow
  end
end
