defmodule Brain.Knowledge.LearningCenterTest do
  alias Brain.Knowledge
  use Brain.Test.GraphCase, async: false
  import Brain.TestHelpers

  alias Knowledge.{LearningCenter, ReviewQueue, SourceReliability}
  alias Brain.Knowledge.Types.{Hypothesis, Investigation, ResearchGoal}

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  setup _context do
    ensure_pubsub_started()

    case Task.Supervisor.start_link(name: Brain.Knowledge.AgentSupervisor) do
      {:ok, _pid} -> :ok
      {:error, {:already_started, _pid}} -> :ok
    end

    ensure_started(SourceReliability)
    ensure_started(ReviewQueue)
    ensure_started(LearningCenter)
    ReviewQueue.clear()

    :ok
  end

  describe "start_session/2" do
    test "starts a new learning session" do
      {:ok, session} = LearningCenter.start_session("European capitals", world_id: @world_id, mock: true)

      assert is_binary(session.id)
      assert session.status == :active
      assert session.topic == "European capitals"
      assert session.goals != []
    end

    test "generates goals from topic" do
      {:ok, session} = LearningCenter.start_session("Test topic", world_id: @world_id, mock: true, max_goals: 2)

      assert length(session.goals) <= 2
      assert Enum.all?(session.goals, &match?(%ResearchGoal{}, &1))
    end

    test "accepts custom questions" do
      {:ok, session} =
        LearningCenter.start_session("France",
          questions: ["What is the capital?", "What is the population?"],
          world_id: @world_id, mock: true
        )

      assert session.goals != []
    end
  end

  describe "a session in Atlas" do
    test "is stored under the session's UUID, with its goals, and updated by it" do
      {:ok, session} = LearningCenter.start_session("Stored session", world_id: @world_id, mock: true)

      assert {:ok, _} = Ecto.UUID.cast(session.id)
      assert Enum.all?(session.goals, &match?({:ok, _}, Ecto.UUID.cast(&1.id)))

      Brain.AtlasIntegration.drain()

      stored = Atlas.Learning.get_session(session.id)
      assert stored.world_id == @world_id
      assert stored.topic == "Stored session"

      assert session.id |> Atlas.Learning.goals_for_session() |> Enum.map(& &1.id) |> Enum.sort() ==
               session.goals |> Enum.map(& &1.id) |> Enum.sort()

      :ok = LearningCenter.cancel_session(session.id)
      Brain.AtlasIntegration.drain()

      assert Atlas.Learning.get_session(session.id).status == "cancelled"
    end
  end

  describe "get_session/1" do
    test "retrieves existing session" do
      {:ok, session} = LearningCenter.start_session("Test", world_id: @world_id, mock: true)

      {:ok, retrieved} = LearningCenter.get_session(session.id)

      assert retrieved.id == session.id
      assert retrieved.topic == "Test"
    end

    test "returns error for non-existent session" do
      {:error, :not_found} = LearningCenter.get_session("nonexistent-id")
    end
  end

  describe "cancel_session/1" do
    test "cancels an active session" do
      {:ok, session} = LearningCenter.start_session("To Cancel", world_id: @world_id, mock: true)

      :ok = LearningCenter.cancel_session(session.id)

      {:ok, cancelled} = LearningCenter.get_session(session.id)
      assert cancelled.status == :cancelled
    end

    test "returns error for non-existent session" do
      {:error, :not_found} = LearningCenter.cancel_session("nonexistent-id")
    end
  end

  describe "list_sessions/1" do
    test "lists all sessions" do
      {:ok, _} = LearningCenter.start_session("Session 1", world_id: @world_id, mock: true)
      {:ok, _} = LearningCenter.start_session("Session 2", world_id: @world_id, mock: true)

      sessions = LearningCenter.list_sessions()

      assert length(sessions) >= 2
    end

    test "filters by status" do
      {:ok, session} = LearningCenter.start_session("To Cancel", world_id: @world_id, mock: true)
      LearningCenter.cancel_session(session.id)

      active = LearningCenter.list_sessions(status: :active)
      cancelled = LearningCenter.list_sessions(status: :cancelled)

      assert not Enum.any?(active, &(&1.id == session.id))
      assert Enum.any?(cancelled, &(&1.id == session.id))
    end

    test "respects limit" do
      for i <- 1..5 do
        LearningCenter.start_session("Session #{i}", world_id: @world_id, mock: true)
      end

      sessions = LearningCenter.list_sessions(limit: 3)

      assert length(sessions) == 3
    end
  end

  describe "stats/0" do
    test "returns learning center statistics" do
      stats = LearningCenter.stats()

      assert is_map(stats)
      assert Map.has_key?(stats, :total_sessions)
      assert Map.has_key?(stats, :active_sessions)
      assert Map.has_key?(stats, :active_agents)
      assert Map.has_key?(stats, :total_findings)
    end
  end

  describe "add_goal/2" do
    test "adds goal to existing session" do
      {:ok, session} = LearningCenter.start_session("Main topic", world_id: @world_id, mock: true)
      goal = ResearchGoal.new("Sub topic")

      :ok = LearningCenter.add_goal(session.id, goal)

      {:ok, updated} = LearningCenter.get_session(session.id)
      assert length(updated.goals) > length(session.goals)
    end

    test "returns error for non-existent session" do
      goal = ResearchGoal.new("Test")
      {:error, :session_not_found} = LearningCenter.add_goal("nonexistent", goal)
    end
  end

  describe "session completion" do
    test "session completes when all agents finish" do
      {:ok, session} = LearningCenter.start_session("Quick topic", world_id: @world_id, mock: true, max_goals: 1)
      Process.sleep(500)

      {:ok, completed} = LearningCenter.get_session(session.id)
      assert completed.status in [:active, :completed]
    end
  end

  describe "ready?/0" do
    test "returns true when service is running" do
      assert LearningCenter.ready?() == true
    end
  end

  describe "test_predictions/2" do
    test "evaluates a belief-query prediction against the world's beliefs" do
      hypothesis = %{
        Hypothesis.new("Paris is the capital of France", entity: "Paris")
        | prediction: %{type: :belief_query, params: %{subject: :world, predicate: :paris}}
      }

      investigation = Investigation.new("capitals", hypotheses: [hypothesis])

      assert {:ok, %Investigation{hypotheses: [evaluated]}} =
               LearningCenter.test_predictions(@world_id, investigation)

      assert evaluated.claim == hypothesis.claim
    end

    test "raises without a world" do
      investigation = Investigation.new("capitals", hypotheses: [])

      for world <- [nil, ""] do
        assert_raise FunctionClauseError, fn -> LearningCenter.test_predictions(world, investigation) end
      end
    end
  end
end
