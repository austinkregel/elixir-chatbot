defmodule Atlas.Schemas.Belief do
  @moduledoc """
  Ecto schema for epistemic belief storage.

  Replaces the file-based BeliefStore `.term` persistence.
  Stores beliefs with subject-predicate-object triples, confidence scores,
  source tracking, and JTMS integration metadata.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_beliefs" do
    field :subject, :string
    field :predicate, :string
    field :object, :string
    field :confidence, :float
    field :source, :string
    field :source_authority, :string
    field :user_id, :string
    field :node_id, :string
    field :retracted, :boolean, default: false
    field :last_confirmed, :utc_datetime_usec
    field :provenance, :map, default: %{}
    field :metadata, :map, default: %{}

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(subject predicate object confidence source)a
  @optional_fields ~w(source_authority user_id node_id retracted last_confirmed provenance metadata)a

  def changeset(belief, attrs) do
    belief
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_number(:confidence, greater_than_or_equal_to: 0.0, less_than_or_equal_to: 1.0)
  end

  @doc "Query active (non-retracted) beliefs."
  def active(query \\ __MODULE__) do
    from(b in query, where: b.retracted == false)
  end

  @doc "Query beliefs by subject."
  def for_subject(query \\ __MODULE__, subject) do
    from(b in query, where: b.subject == ^subject)
  end

  @doc "Query beliefs by user."
  def for_user(query \\ __MODULE__, user_id) do
    from(b in query, where: b.user_id == ^user_id)
  end

  @doc "Query beliefs above a confidence threshold."
  def above_confidence(query \\ __MODULE__, threshold) do
    from(b in query, where: b.confidence >= ^threshold)
  end
end
