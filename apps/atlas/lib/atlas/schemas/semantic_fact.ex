defmodule Atlas.Schemas.SemanticFact do
  @moduledoc """
  Ecto schema for consolidated semantic facts.

  Replaces the file-based Memory.Store semantic fact persistence.
  Semantic facts are consolidated from episode clusters and represent
  generalized knowledge with confidence scores and TF-IDF embeddings.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_semantic_facts" do
    field :world_id, :string, default: "default"
    field :content, :string
    field :category, :string
    field :confidence, :float
    field :embedding, {:array, :float}, default: []
    field :source_episodes, {:array, :string}, default: []
    field :tags, {:array, :string}, default: []

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(content category confidence)a
  @optional_fields ~w(world_id embedding source_episodes tags)a

  def changeset(semantic_fact, attrs) do
    semantic_fact
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @doc "Query semantic facts for a specific world."
  def for_world(query \\ __MODULE__, world_id) do
    from(sf in query, where: sf.world_id == ^world_id)
  end

  @doc "Query semantic facts by category."
  def for_category(query \\ __MODULE__, category) do
    from(sf in query, where: sf.category == ^category)
  end
end
