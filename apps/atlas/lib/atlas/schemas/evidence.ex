defmodule Atlas.Schemas.Evidence do
  @moduledoc """
  Ecto schema for gathered findings linked to investigations.

  Evidence items are factual claims extracted from sources,
  associated with investigations and optionally linked to
  specific hypotheses they support or contradict.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.{Investigation, Hypothesis}

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_evidence" do
    belongs_to :investigation, Investigation, type: :binary_id
    belongs_to :hypothesis, Hypothesis, type: :binary_id
    field :claim, :string
    field :entity, :string
    field :entity_type, :string
    field :source_url, :string
    field :source_domain, :string
    field :source_title, :string
    field :source_reliability, :float
    field :source_bias, :string
    field :source_trust_tier, :string
    field :raw_context, :string
    field :confidence, :float, default: 0.5
    field :corroboration_group, :string
    field :evidence_type, :string, default: "unassociated"
    field :extracted_at, :utc_datetime_usec

    timestamps(type: :utc_datetime_usec)
  end

  @valid_evidence_types ~w(supporting contradicting unassociated)

  @required_fields ~w(investigation_id)a
  @optional_fields ~w(hypothesis_id claim entity entity_type source_url source_domain
    source_title source_reliability source_bias source_trust_tier raw_context
    confidence corroboration_group evidence_type extracted_at)a

  def changeset(evidence, attrs) do
    evidence
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:evidence_type, @valid_evidence_types)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> foreign_key_constraint(:investigation_id)
    |> foreign_key_constraint(:hypothesis_id)
  end

  @doc "Query evidence for an investigation."
  def for_investigation(query \\ __MODULE__, investigation_id) do
    from(e in query, where: e.investigation_id == ^investigation_id)
  end

  @doc "Query evidence for a hypothesis."
  def for_hypothesis(query \\ __MODULE__, hypothesis_id) do
    from(e in query, where: e.hypothesis_id == ^hypothesis_id)
  end

  @doc "Query evidence by type."
  def with_type(query \\ __MODULE__, evidence_type) do
    from(e in query, where: e.evidence_type == ^evidence_type)
  end

  @doc "Query evidence for an entity."
  def for_entity(query \\ __MODULE__, entity) do
    from(e in query, where: e.entity == ^entity)
  end
end
