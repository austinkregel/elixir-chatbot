defmodule Atlas.Schemas.Episode do
  @moduledoc """
  Ecto schema for episodic memory storage.

  Replaces the file-based Memory.Store episode persistence.
  Episodes represent state-action-outcome triples from conversations,
  with TF-IDF embeddings for similarity search.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_episodes" do
    field :world_id, :string
    field :state, :string
    field :action, :string
    field :outcome, :string
    field :tags, {:array, :string}, default: []
    field :embedding, {:array, :float}, default: []
    field :semantic_id, :binary_id
    field :entity_names, {:array, :string}, default: []

    timestamps(type: :utc_datetime_usec)
  end

  # An episode belongs to the world it happened in; the caller names it.
  @required_fields ~w(world_id state action)a
  @optional_fields ~w(outcome tags embedding semantic_id entity_names)a

  @doc false
  def changeset(episode, attrs) do
    episode
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> default_outcome()
  end

  defp default_outcome(changeset) do
    case get_field(changeset, :outcome) do
      nil -> put_change(changeset, :outcome, "")
      _ -> changeset
    end
  end

  @doc "Query episodes for a specific world."
  def for_world(query \\ __MODULE__, world_id) do
    from(e in query, where: e.world_id == ^world_id)
  end

  @doc "Query episodes that have a semantic fact link."
  def with_semantic(query \\ __MODULE__) do
    from(e in query, where: not is_nil(e.semantic_id))
  end
end
