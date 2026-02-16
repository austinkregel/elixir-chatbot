defmodule Atlas.Schemas.KnowledgeEntry do
  @moduledoc """
  Ecto schema for knowledge store entries.

  Replaces the file-based KnowledgeStore per-persona .json persistence.
  Flattens persona knowledge into (world, persona, category, key) entries.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_knowledge_entries" do
    field :world_id, :string, default: "default"
    field :persona_name, :string
    field :category, :string
    field :key, :string
    field :data, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(persona_name category key)a
  @optional_fields ~w(world_id data)a

  def changeset(knowledge_entry, attrs) do
    knowledge_entry
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint([:world_id, :persona_name, :category, :key],
           name: :atlas_knowledge_entries_unique_entry
         )
  end

  @doc "Query entries for a persona."
  def for_persona(query \\ __MODULE__, persona_name) do
    from(ke in query, where: ke.persona_name == ^persona_name)
  end

  @doc "Query entries for a world and persona."
  def for_world_persona(query \\ __MODULE__, world_id, persona_name) do
    from(ke in query,
      where: ke.world_id == ^world_id and ke.persona_name == ^persona_name
    )
  end
end
