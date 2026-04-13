defmodule Atlas.Schemas.ResearchGoal do
  @moduledoc """
  Ecto schema for per-session research goals.

  Each goal represents a research objective with a topic,
  questions to investigate, constraints, and priority.
  """

  use Ecto.Schema
  import Ecto.Changeset
  import Ecto.Query

  alias Atlas.Schemas.LearningSession

  @primary_key {:id, :binary_id, autogenerate: true}

  schema "atlas_research_goals" do
    belongs_to :session, LearningSession, type: :binary_id
    field :topic, :string
    field :questions, {:array, :string}, default: []
    field :constraints, :map, default: %{}
    field :priority, :string, default: "normal"
    field :status, :string, default: "pending"

    timestamps(type: :utc_datetime_usec)
  end

  @valid_priorities ~w(low normal high)
  @valid_statuses ~w(pending in_progress completed failed)

  @required_fields ~w(session_id topic)a
  @optional_fields ~w(questions constraints priority status)a

  def changeset(goal, attrs) do
    goal
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:priority, @valid_priorities)
    |> validate_inclusion(:status, @valid_statuses)
    |> foreign_key_constraint(:session_id)
  end

  @doc "Query goals for a session."
  def for_session(query \\ __MODULE__, session_id) do
    from(g in query, where: g.session_id == ^session_id)
  end

  @doc "Query goals by status."
  def with_status(query \\ __MODULE__, status) do
    from(g in query, where: g.status == ^status)
  end
end
