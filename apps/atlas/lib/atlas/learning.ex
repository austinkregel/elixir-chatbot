defmodule Atlas.Learning do
  @moduledoc """
  Context module for the ephemeral learning tier.

  Provides CRUD operations for learning sessions, research goals,
  investigations, hypotheses, and evidence. This is the unified
  persistence layer that replaces in-memory-only session tracking.
  """

  import Ecto.Query

  alias Atlas.Repo
  alias Atlas.Schemas.{LearningSession, ResearchGoal, Investigation, Hypothesis, Evidence}

  # ============================================================================
  # Sessions
  # ============================================================================

  @doc "Creates a new learning session."
  def create_session(attrs) when is_map(attrs) do
    %LearningSession{}
    |> LearningSession.changeset(attrs)
    |> Repo.insert()
  end

  @doc "Updates an existing learning session (by struct or by ID)."
  def update_session(%LearningSession{} = session, attrs) when is_map(attrs) do
    session
    |> LearningSession.changeset(attrs)
    |> Repo.update()
  end

  def update_session(session_id, attrs) when is_binary(session_id) and is_map(attrs) do
    case get_session(session_id) do
      nil -> {:error, :not_found}
      session -> update_session(session, attrs)
    end
  end

  @doc "Gets a session by ID (returns nil if not found)."
  def get_session(session_id) when is_binary(session_id) do
    Repo.get(LearningSession, session_id)
  end

  @doc "Lists sessions with optional filters.

  ## Options
    - :status - Filter by status (\"active\", \"completed\", \"cancelled\")
    - :limit - Maximum results (default: 100)
    - :offset - Offset for pagination
  "
  def list_sessions(opts \\ []) do
    status = Keyword.get(opts, :status)
    limit = Keyword.get(opts, :limit, 100)
    offset = Keyword.get(opts, :offset, 0)

    LearningSession
    |> maybe_filter_status(status)
    |> LearningSession.recent_first()
    |> limit(^limit)
    |> offset(^offset)
    |> Repo.all()
  end

  @doc """
  Gets a session with all associations preloaded:
  goals, investigations (with hypotheses and evidence).
  """
  def session_with_details(session_id) when is_binary(session_id) do
    query =
      from(s in LearningSession,
        where: s.id == ^session_id,
        preload: [
          :goals,
          investigations: [:hypotheses, :evidence]
        ]
      )

    case Repo.one(query) do
      nil -> {:error, :not_found}
      session -> {:ok, session}
    end
  end

  # ============================================================================
  # Goals
  # ============================================================================

  @doc "Creates a new research goal."
  def create_goal(attrs) when is_map(attrs) do
    %ResearchGoal{}
    |> ResearchGoal.changeset(attrs)
    |> Repo.insert()
  end

  @doc "Updates the status of a research goal."
  def update_goal_status(goal_id, new_status)
      when is_binary(goal_id) and is_binary(new_status) do
    case Repo.get(ResearchGoal, goal_id) do
      nil ->
        {:error, :not_found}

      goal ->
        goal
        |> ResearchGoal.changeset(%{status: new_status})
        |> Repo.update()
    end
  end

  @doc "Lists goals for a session."
  def goals_for_session(session_id) when is_binary(session_id) do
    ResearchGoal
    |> ResearchGoal.for_session(session_id)
    |> Repo.all()
  end

  # ============================================================================
  # Investigations
  # ============================================================================

  @doc "Creates a new investigation."
  def create_investigation(attrs) when is_map(attrs) do
    %Investigation{}
    |> Investigation.changeset(attrs)
    |> Repo.insert()
  end

  @doc "Concludes an investigation with a status and conclusion."
  def conclude_investigation(investigation_id, attrs)
      when is_binary(investigation_id) and is_map(attrs) do
    case Repo.get(Investigation, investigation_id) do
      nil ->
        {:error, :not_found}

      investigation ->
        conclude_attrs =
          Map.merge(attrs, %{
            status: "concluded",
            concluded_at: DateTime.utc_now()
          })

        investigation
        |> Investigation.changeset(conclude_attrs)
        |> Repo.update()
    end
  end

  @doc "Lists investigations for a session."
  def investigations_for_session(session_id) when is_binary(session_id) do
    Investigation
    |> Investigation.for_session(session_id)
    |> Repo.all()
  end

  # ============================================================================
  # Hypotheses
  # ============================================================================

  @doc "Creates a new hypothesis."
  def create_hypothesis(attrs) when is_map(attrs) do
    %Hypothesis{}
    |> Hypothesis.changeset(attrs)
    |> Repo.insert()
  end

  @doc "Updates a hypothesis (status, confidence, etc.)."
  def update_hypothesis(hypothesis_id, attrs)
      when is_binary(hypothesis_id) and is_map(attrs) do
    case Repo.get(Hypothesis, hypothesis_id) do
      nil ->
        {:error, :not_found}

      hypothesis ->
        hypothesis
        |> Hypothesis.changeset(attrs)
        |> Repo.update()
    end
  end

  @doc "Lists hypotheses for an investigation."
  def hypotheses_for_investigation(investigation_id) when is_binary(investigation_id) do
    Hypothesis
    |> Hypothesis.for_investigation(investigation_id)
    |> Repo.all()
  end

  # ============================================================================
  # Evidence
  # ============================================================================

  @doc "Creates a new evidence record."
  def create_evidence(attrs) when is_map(attrs) do
    %Evidence{}
    |> Evidence.changeset(attrs)
    |> Repo.insert()
  end

  @doc "Lists evidence for an investigation."
  def evidence_for_investigation(investigation_id) when is_binary(investigation_id) do
    Evidence
    |> Evidence.for_investigation(investigation_id)
    |> Repo.all()
  end

  @doc "Lists all evidence across all investigations for a session."
  def all_evidence_for_session(session_id) when is_binary(session_id) do
    from(e in Evidence,
      join: i in Investigation,
      on: e.investigation_id == i.id,
      where: i.session_id == ^session_id,
      order_by: [desc: e.extracted_at]
    )
    |> Repo.all()
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp maybe_filter_status(query, nil), do: query

  defp maybe_filter_status(query, status) when is_atom(status) do
    maybe_filter_status(query, Atom.to_string(status))
  end

  defp maybe_filter_status(query, status) when is_binary(status) do
    from(s in query, where: s.status == ^status)
  end
end
