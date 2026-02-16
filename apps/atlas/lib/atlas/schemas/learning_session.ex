defmodule Atlas.Schemas.LearningSession do
  @moduledoc """
  Ecto schema for learning session lifecycle tracking.

  Represents an active, completed, or cancelled learning session
  with its research goals, investigations, and aggregate metrics.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.{ResearchGoal, Investigation}

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_learning_sessions" do
    field :topic, :string
    field :status, :string, default: "active"
    field :started_at, :utc_datetime_usec
    field :completed_at, :utc_datetime_usec
    field :findings_count, :integer, default: 0
    field :approved_count, :integer, default: 0
    field :rejected_count, :integer, default: 0
    field :hypotheses_tested, :integer, default: 0
    field :hypotheses_supported, :integer, default: 0
    field :hypotheses_falsified, :integer, default: 0
    field :source_type, :string, default: "web"
    field :metadata, :map, default: %{}

    has_many :goals, ResearchGoal, foreign_key: :session_id
    has_many :investigations, Investigation, foreign_key: :session_id

    timestamps(type: :utc_datetime_usec)
  end

  @valid_statuses ~w(active completed cancelled)
  @valid_source_types ~w(web academic task auto_trigger)

  @required_fields ~w()a
  @optional_fields ~w(topic status started_at completed_at findings_count approved_count
    rejected_count hypotheses_tested hypotheses_supported hypotheses_falsified
    source_type metadata)a

  def changeset(session, attrs) do
    session
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_inclusion(:status, @valid_statuses)
    |> validate_inclusion(:source_type, @valid_source_types)
    |> validate_number(:findings_count, greater_than_or_equal_to: 0)
    |> validate_number(:approved_count, greater_than_or_equal_to: 0)
    |> validate_number(:rejected_count, greater_than_or_equal_to: 0)
  end

  @doc "Query sessions by status."
  def with_status(query \\ __MODULE__, status) do
    from(s in query, where: s.status == ^status)
  end

  @doc "Query sessions ordered by most recent first."
  def recent_first(query \\ __MODULE__) do
    from(s in query, order_by: [desc: s.started_at])
  end

  @doc "Query active sessions."
  def active(query \\ __MODULE__) do
    from(s in query, where: s.status == "active")
  end
end
