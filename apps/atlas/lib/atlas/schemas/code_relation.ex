defmodule Atlas.Schemas.CodeRelation do
  @moduledoc """
  A directed edge between two code symbols in a world's corpus — `calls`,
  `called_by`, `extends`, `implements`, `imports`, `uses`.

  Stored as a plain indexed table rather than in the Apache AGE graph. The
  queries this actually serves are one hop deep and start from a known
  qualified name ("who calls this?", "what does this import?"), which an index
  answers directly; a graph traversal would cost more and buy nothing until
  something needs multi-hop reachability.

  Edges reference symbols by qualified name rather than by foreign key: a
  relation is routinely discovered before — or without — the symbol it points
  at ever being defined in the indexed corpus, and dropping those edges would
  quietly under-report a call graph at exactly its most interesting boundary.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  @relation_types ~w(calls called_by extends implements imports uses)

  schema "atlas_code_relations" do
    field :world_id, :string
    field :from_qualified, :string
    field :relation_type, :string
    field :to_qualified, :string

    timestamps(type: :utc_datetime_usec)
  end

  @fields ~w(world_id from_qualified relation_type to_qualified)a

  @doc "The closed vocabulary of edge kinds."
  def relation_types, do: @relation_types

  def changeset(relation, attrs) do
    relation
    |> cast(attrs, @fields)
    |> validate_required(@fields)
    |> validate_inclusion(:relation_type, @relation_types)
    |> unique_constraint([:world_id, :from_qualified, :relation_type, :to_qualified],
      name: :atlas_code_relations_identity
    )
  end

  @doc "Edges of one kind leaving a symbol."
  def from(query \\ __MODULE__, world_id, from_qualified, relation_type) do
    from(r in query,
      where:
        r.world_id == ^world_id and
          r.from_qualified == ^from_qualified and
          r.relation_type == ^to_string(relation_type)
    )
  end
end
