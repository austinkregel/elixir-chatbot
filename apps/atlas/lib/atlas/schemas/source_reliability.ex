defmodule Atlas.Schemas.SourceReliability do
  @moduledoc """
  Ecto schema for learned source reliability data.

  Replaces the file-based source_reliability_learned.term persistence.
  Tracks per-domain reliability scores, bias ratings, and admin feedback.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  schema "atlas_source_reliability" do
    field :domain, :string
    field :reliability_score, :float
    field :bias_rating, :string
    field :trust_tier, :string
    field :confirmed_count, :integer, default: 0
    field :rejected_count, :integer, default: 0
    field :admin_decisions, {:array, :map}, default: []
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(domain)a
  @optional_fields ~w(reliability_score bias_rating trust_tier confirmed_count rejected_count admin_decisions metadata)a

  def changeset(source_reliability, attrs) do
    source_reliability
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> unique_constraint(:domain)
  end

  @doc "Query by domain."
  def for_domain(query \\ __MODULE__, domain) do
    from(sr in query, where: sr.domain == ^domain)
  end
end
