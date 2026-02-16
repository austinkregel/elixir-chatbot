defmodule Atlas.Schemas.SourceAuthority do
  @moduledoc """
  Ecto schema for learned source authority tracking.

  Replaces the file-based source_authority_learned.term persistence.
  Tracks credibility and outcome counts per authority key.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_source_authority" do
    field :authority_key, :string
    field :confirmed_count, :integer, default: 0
    field :contradicted_count, :integer, default: 0
    field :total_added, :integer, default: 0
    field :credibility, :float
    field :last_updated, :utc_datetime_usec
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(authority_key)a
  @optional_fields ~w(confirmed_count contradicted_count total_added credibility last_updated metadata)a

  def changeset(source_authority, attrs) do
    source_authority
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint(:authority_key)
  end

  @doc "Query by authority key."
  def for_key(query \\ __MODULE__, key) do
    from(sa in query, where: sa.authority_key == ^key)
  end
end
