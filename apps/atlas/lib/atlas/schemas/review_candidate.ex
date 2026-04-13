defmodule Atlas.Schemas.ReviewCandidate do
  @moduledoc """
  Ecto schema for knowledge review queue candidates.

  Replaces the file-based ReviewQueue `.term` persistence.
  Review candidates represent findings from autonomous research
  that need human review before being accepted as knowledge.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :string, autogenerate: false}

  schema "atlas_review_candidates" do
    field :status, :string, default: "pending"
    field :finding, :map, default: %{}
    field :aggregate_confidence, :float
    field :corroborating_sources, {:array, :map}, default: []
    field :conflicting_findings, {:array, :map}, default: []
    field :existing_contradictions, {:array, :map}, default: []
    field :reviewer_notes, :string
    field :reviewed_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(id status finding aggregate_confidence)a
  @optional_fields ~w(corroborating_sources conflicting_findings existing_contradictions reviewer_notes reviewed_at)a

  def changeset(candidate, attrs) do
    candidate
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, ~w(pending approved rejected auto_approved))
  end

  @doc "Query pending review candidates."
  def pending(query \\ __MODULE__) do
    from(rc in query, where: rc.status == "pending")
  end

  @doc "Query candidates ordered by confidence (highest first)."
  def by_confidence(query \\ __MODULE__) do
    from(rc in query, order_by: [desc: rc.aggregate_confidence])
  end

  @doc "Query candidates by status."
  def with_status(query \\ __MODULE__, status) do
    from(rc in query, where: rc.status == ^status)
  end
end
