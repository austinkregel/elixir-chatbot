defmodule Atlas.Schemas.SemanticFactMetricCount do
  @moduledoc """
  One counter of one measurement window of a world's semantic-fact metrics
  (`Atlas.SemanticFactMetrics`). `counter` names what was counted; what the
  name means is `Brain.Metrics.Aggregator`'s to say. A row exists only once
  something was counted, so `count` is positive.

  Rows are written by `Atlas.SemanticFactMetrics.increment!/2` only, which adds
  to `count` in the database.
  """

  use Ecto.Schema

  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  @type t :: %__MODULE__{}

  schema "atlas_semantic_fact_metric_counts" do
    field :world_id, :string
    field :counter, :string
    field :count, :integer

    belongs_to :window, Atlas.Schemas.SemanticFactMetricWindow

    timestamps(type: :utc_datetime_usec)
  end
end
