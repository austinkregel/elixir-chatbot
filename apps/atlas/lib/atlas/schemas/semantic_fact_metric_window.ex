defmodule Atlas.Schemas.SemanticFactMetricWindow do
  @moduledoc """
  One measurement window of a world's semantic-fact metrics
  (`Atlas.SemanticFactMetrics`): the counts made from `started_at` until
  `ended_at`. The world's open window has no `ended_at` and is the one being
  counted into; a world has at most one.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :binary_id, autogenerate: true}

  @type t :: %__MODULE__{}

  schema "atlas_semantic_fact_metric_windows" do
    field :world_id, :string
    field :started_at, :utc_datetime_usec
    field :ended_at, :utc_datetime_usec
  end

  @doc false
  def changeset(window, attrs) do
    window
    |> cast(attrs, [:world_id, :started_at, :ended_at])
    |> validate_required([:world_id, :started_at])
    |> validate_change(:world_id, fn :world_id, world_id ->
      if String.trim(world_id) == "", do: [world_id: "must not be blank"], else: []
    end)
    |> unique_constraint(:world_id, name: :atlas_semantic_fact_metric_windows_open_index)
  end
end
