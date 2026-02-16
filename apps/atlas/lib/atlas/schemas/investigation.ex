defmodule Atlas.Schemas.Investigation do
  @moduledoc """
  Ecto schema for scientific investigations.

  Represents a systematic investigation testing one or more hypotheses
  against gathered evidence, following the scientific method.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.{LearningSession, ResearchGoal, Hypothesis, Evidence}

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_investigations" do
    belongs_to :session, LearningSession, type: :binary_id
    belongs_to :goal, ResearchGoal, type: :binary_id
    field :topic, :string
    field :status, :string, default: "planning"
    field :conclusion, :string
    field :independent_variable, :string, default: "source"
    field :dependent_variable, :string, default: "claim"
    field :constants, {:array, :string}, default: []
    field :methodology_notes, :string
    field :started_at, :utc_datetime_usec
    field :concluded_at, :utc_datetime_usec

    has_many :hypotheses, Hypothesis, foreign_key: :investigation_id
    has_many :evidence, Evidence, foreign_key: :investigation_id

    timestamps(type: :utc_datetime_usec)
  end

  @valid_statuses ~w(planning gathering evaluating concluded)
  @valid_conclusions ~w(hypotheses_supported hypotheses_falsified inconclusive mixed)

  @required_fields ~w(session_id topic)a
  @optional_fields ~w(goal_id status conclusion independent_variable dependent_variable
    constants methodology_notes started_at concluded_at)a

  def changeset(investigation, attrs) do
    investigation
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, @valid_statuses)
    |> validate_inclusion(:conclusion, @valid_conclusions ++ [nil])
    |> foreign_key_constraint(:session_id)
    |> foreign_key_constraint(:goal_id)
  end

  @doc "Query investigations for a session."
  def for_session(query \\ __MODULE__, session_id) do
    from(i in query, where: i.session_id == ^session_id)
  end

  @doc "Query investigations by status."
  def with_status(query \\ __MODULE__, status) do
    from(i in query, where: i.status == ^status)
  end
end
