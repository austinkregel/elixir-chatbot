defmodule Atlas.Schemas.LearnedFact do
  @moduledoc """
  Ecto schema for the learned facts database.

  Replaces the file-based FactDatabase.Integration `.json` persistence.
  Learned facts represent verified knowledge about entities,
  categorized and scored by confidence.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :string, autogenerate: false}

  schema "atlas_learned_facts" do
    field :entity, :string
    field :entity_type, :string
    field :fact, :string
    field :category, :string, default: "learned"
    field :confidence, :float
    field :verification_source, :string

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(id entity fact confidence)a
  @optional_fields ~w(entity_type category verification_source)a

  def changeset(learned_fact, attrs) do
    learned_fact
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @doc "Query facts for a specific entity."
  def for_entity(query \\ __MODULE__, entity) do
    from(lf in query, where: lf.entity == ^entity)
  end

  @doc "Query facts by category."
  def for_category(query \\ __MODULE__, category) do
    from(lf in query, where: lf.category == ^category)
  end
end
