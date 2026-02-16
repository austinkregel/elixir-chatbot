defmodule Atlas.Schemas.Hypothesis do
  @moduledoc """
  Ecto schema for testable claims within investigations.

  Hypotheses follow the scientific method: they are falsifiable claims
  that can be supported or contradicted by gathered evidence.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.{Investigation, Evidence}

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_hypotheses" do
    belongs_to :investigation, Investigation, type: :binary_id
    field :claim, :string
    field :entity, :string
    field :derived_from, :string
    field :prediction, :string
    field :status, :string, default: "untested"
    field :confidence, :float, default: 0.0
    field :confidence_level, :string, default: "none"
    field :source_count, :integer, default: 0
    field :replication_count, :integer, default: 0
    field :tested_at, :utc_datetime_usec

    has_many :evidence, Evidence, foreign_key: :hypothesis_id

    timestamps(type: :utc_datetime_usec)
  end

  @valid_statuses ~w(untested testing supported falsified inconclusive)
  @valid_confidence_levels ~w(none low moderate high very_high)

  @required_fields ~w(investigation_id claim)a
  @optional_fields ~w(entity derived_from prediction status confidence confidence_level
    source_count replication_count tested_at)a

  def changeset(hypothesis, attrs) do
    hypothesis
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, @valid_statuses)
    |> validate_inclusion(:confidence_level, @valid_confidence_levels)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
    |> foreign_key_constraint(:investigation_id)
  end

  @doc "Query hypotheses for an investigation."
  def for_investigation(query \\ __MODULE__, investigation_id) do
    from(h in query, where: h.investigation_id == ^investigation_id)
  end

  @doc "Query hypotheses by status."
  def with_status(query \\ __MODULE__, status) do
    from(h in query, where: h.status == ^status)
  end

  @doc "Query supported hypotheses with high confidence."
  def promotable(query \\ __MODULE__) do
    from(h in query,
      where: h.status == "supported" and h.confidence >= 0.7 and h.source_count >= 2
    )
  end
end
