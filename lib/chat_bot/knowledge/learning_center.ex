defmodule ChatBot.Knowledge.LearningCenter do
  @moduledoc """
  Central orchestrator for the Knowledge Expansion System.

  The Learning Center:
  - Manages learning sessions (triggered or scheduled)
  - Maintains a goal queue with research objectives
  - Dispatches Research Agents as supervised Tasks
  - Collects and synthesizes agent findings
  - Routes vetted findings to the Admin review queue

  ## Example

      # Start a learning session
      {:ok, session} = LearningCenter.start_session("European capitals")
      
      # Check session status
      {:ok, session} = LearningCenter.get_session(session.id)
      
      # List active sessions
      sessions = LearningCenter.list_sessions()
  """

  use GenServer
  require Logger

  alias ChatBot.Knowledge.{ResearchAgent, Corroborator, ReviewQueue}
  alias ChatBot.Knowledge.Types.{ResearchGoal, LearningSession}
  alias ChatBot.Epistemic.BeliefStore

  @max_concurrent_agents 5
  @session_timeout_ms 300_000

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Starts a new learning session for a topic.

  ## Options
    - :questions - List of specific questions to research
    - :max_goals - Maximum number of goals to generate
    - :priority - :low | :normal | :high
    - :mock - If true, uses mock data for testing
  """
  @spec start_session(String.t(), keyword()) :: {:ok, LearningSession.t()} | {:error, term()}
  def start_session(topic, opts \\ []) when is_binary(topic) do
    GenServer.call(__MODULE__, {:start_session, topic, opts})
  end

  @doc """
  Adds a goal to an existing session.
  """
  @spec add_goal(String.t(), ResearchGoal.t()) :: :ok | {:error, term()}
  def add_goal(session_id, %ResearchGoal{} = goal) do
    GenServer.call(__MODULE__, {:add_goal, session_id, goal})
  end

  @doc """
  Gets a session by ID.
  """
  @spec get_session(String.t()) :: {:ok, LearningSession.t()} | {:error, :not_found}
  def get_session(session_id) when is_binary(session_id) do
    GenServer.call(__MODULE__, {:get_session, session_id})
  end

  @doc """
  Cancels a session.
  """
  @spec cancel_session(String.t()) :: :ok | {:error, term()}
  def cancel_session(session_id) when is_binary(session_id) do
    GenServer.call(__MODULE__, {:cancel_session, session_id})
  end

  @doc """
  Lists all sessions.

  ## Options
    - :status - Filter by status (:active, :completed, :cancelled)
    - :limit - Maximum number to return
  """
  @spec list_sessions(keyword()) :: [LearningSession.t()]
  def list_sessions(opts \\ []) do
    GenServer.call(__MODULE__, {:list_sessions, opts})
  end

  @doc """
  Gets statistics about the Learning Center.
  """
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Checks if the service is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Start the agent supervisor if not already started
    ensure_agent_supervisor_started()

    state = %{
      sessions: %{},
      agent_tasks: %{},
      scheduled: [],
      stats: %{
        total_sessions: 0,
        total_findings: 0,
        active_agents: 0
      }
    }

    Logger.info("LearningCenter initialized")

    {:ok, state}
  end

  @impl true
  def handle_call({:start_session, topic, opts}, _from, state) do
    session = LearningSession.new(topic: topic)
    mock? = Keyword.get(opts, :mock, false)

    # Decompose topic into research goals
    goals = decompose_topic(topic, opts)

    session =
      Enum.reduce(goals, session, fn goal, sess ->
        LearningSession.add_goal(sess, goal)
      end)

    # Start research agents for each goal
    {agent_refs, updated_state} = dispatch_agents(goals, session.id, mock?, state)

    # Update state with session and agent refs
    new_state = %{
      updated_state
      | sessions: Map.put(updated_state.sessions, session.id, session),
        agent_tasks: Map.merge(updated_state.agent_tasks, agent_refs),
        stats: %{updated_state.stats | total_sessions: updated_state.stats.total_sessions + 1}
    }

    Logger.info("Learning session started",
      session_id: session.id,
      topic: topic,
      goals: length(goals)
    )

    # Schedule timeout
    Process.send_after(self(), {:session_timeout, session.id}, @session_timeout_ms)

    {:reply, {:ok, session}, new_state}
  end

  @impl true
  def handle_call({:add_goal, session_id, goal}, _from, state) do
    case Map.get(state.sessions, session_id) do
      nil ->
        {:reply, {:error, :session_not_found}, state}

      session ->
        updated_session = LearningSession.add_goal(session, goal)
        new_sessions = Map.put(state.sessions, session_id, updated_session)

        # Dispatch agent for the new goal
        {agent_refs, updated_state} = dispatch_agents([goal], session_id, false, state)

        new_state = %{
          updated_state
          | sessions: new_sessions,
            agent_tasks: Map.merge(updated_state.agent_tasks, agent_refs)
        }

        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:get_session, session_id}, _from, state) do
    case Map.get(state.sessions, session_id) do
      nil -> {:reply, {:error, :not_found}, state}
      session -> {:reply, {:ok, session}, state}
    end
  end

  @impl true
  def handle_call({:cancel_session, session_id}, _from, state) do
    case Map.get(state.sessions, session_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      session ->
        # Cancel any running agents for this session
        {cancelled_refs, remaining_refs} =
          state.agent_tasks
          |> Enum.split_with(fn {_ref, {sid, _gid}} -> sid == session_id end)

        # Demonitor and flush any pending messages for cancelled tasks
        Enum.each(cancelled_refs, fn {ref, _} ->
          Process.demonitor(ref, [:flush])
        end)

        cancelled_session = LearningSession.cancel(session)
        new_sessions = Map.put(state.sessions, session_id, cancelled_session)

        new_state = %{
          state
          | sessions: new_sessions,
            agent_tasks: Map.new(remaining_refs)
        }

        Logger.info("Session cancelled", session_id: session_id)

        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_call({:list_sessions, opts}, _from, state) do
    status_filter = Keyword.get(opts, :status)
    limit = Keyword.get(opts, :limit, 100)

    sessions =
      state.sessions
      |> Map.values()
      |> maybe_filter_by_status(status_filter)
      |> Enum.sort_by(& &1.started_at, {:desc, DateTime})
      |> Enum.take(limit)

    {:reply, sessions, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_sessions: state.stats.total_sessions,
      active_sessions: count_active_sessions(state.sessions),
      active_agents: map_size(state.agent_tasks),
      total_findings: state.stats.total_findings
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_info({ref, {:ok, findings}}, state) do
    # Agent completed successfully
    Process.demonitor(ref, [:flush])

    case Map.pop(state.agent_tasks, ref) do
      {{session_id, goal_id}, remaining_tasks} ->
        Logger.debug("Agent completed",
          session_id: session_id,
          goal_id: goal_id,
          findings: length(findings)
        )

        # Process findings through corroboration
        {:ok, candidates} = Corroborator.corroborate(findings, include_uncorroborated: true)

        # Check for contradictions with existing beliefs
        candidates = check_contradictions(candidates)

        # Add to review queue
        Enum.each(candidates, fn candidate ->
          ReviewQueue.add(%{candidate | session_id: session_id})
        end)

        # Update session metrics
        new_state = update_session_metrics(state, session_id, goal_id, findings, candidates)
        new_state = %{new_state | agent_tasks: remaining_tasks}

        # Check if session is complete
        new_state = maybe_complete_session(new_state, session_id)

        {:noreply, new_state}

      {nil, _} ->
        # Unknown ref, ignore
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({ref, {:error, reason}}, state) do
    # Agent failed
    Process.demonitor(ref, [:flush])

    case Map.pop(state.agent_tasks, ref) do
      {{session_id, goal_id}, remaining_tasks} ->
        Logger.warning("Agent failed",
          session_id: session_id,
          goal_id: goal_id,
          reason: inspect(reason)
        )

        # Update goal status to failed
        new_state = mark_goal_failed(state, session_id, goal_id)
        new_state = %{new_state | agent_tasks: remaining_tasks}

        # Check if session is complete (all goals done or failed)
        new_state = maybe_complete_session(new_state, session_id)

        {:noreply, new_state}

      {nil, _} ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    # Agent process crashed
    case Map.pop(state.agent_tasks, ref) do
      {{session_id, goal_id}, remaining_tasks} ->
        Logger.error("Agent crashed",
          session_id: session_id,
          goal_id: goal_id,
          reason: inspect(reason)
        )

        new_state = mark_goal_failed(state, session_id, goal_id)
        new_state = %{new_state | agent_tasks: remaining_tasks}
        new_state = maybe_complete_session(new_state, session_id)

        {:noreply, new_state}

      {nil, _} ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:session_timeout, session_id}, state) do
    case Map.get(state.sessions, session_id) do
      %LearningSession{status: :active} = session ->
        Logger.warning("Session timed out", session_id: session_id)

        # Force complete the session
        completed = LearningSession.complete(session)
        new_sessions = Map.put(state.sessions, session_id, completed)

        {:noreply, %{state | sessions: new_sessions}}

      _ ->
        {:noreply, state}
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp ensure_agent_supervisor_started do
    case Process.whereis(ChatBot.Knowledge.AgentSupervisor) do
      nil ->
        # Start the supervisor - it should be started by the application
        Logger.debug("Agent supervisor not found, will be started by application")

      _pid ->
        :ok
    end
  end

  defp decompose_topic(topic, opts) do
    questions = Keyword.get(opts, :questions, [])
    max_goals = Keyword.get(opts, :max_goals, 3)
    priority = Keyword.get(opts, :priority, :normal)

    # Generate goals from topic and questions
    base_goal =
      ResearchGoal.new(topic,
        questions: questions,
        priority: priority
      )

    # If no questions provided, generate some default ones
    additional_goals =
      if questions == [] do
        generate_default_questions(topic)
        |> Enum.take(max_goals - 1)
        |> Enum.map(fn q ->
          ResearchGoal.new(topic,
            questions: [q],
            priority: priority
          )
        end)
      else
        []
      end

    [base_goal | additional_goals]
    |> Enum.take(max_goals)
  end

  defp generate_default_questions(topic) do
    # Generate common questions about a topic
    [
      "What is #{topic}?",
      "What are the key facts about #{topic}?",
      "What is the history of #{topic}?"
    ]
  end

  defp dispatch_agents(goals, session_id, mock?, state) do
    # Limit concurrent agents
    available_slots = @max_concurrent_agents - map_size(state.agent_tasks)
    goals_to_dispatch = Enum.take(goals, available_slots)

    agent_refs =
      goals_to_dispatch
      |> Enum.map(fn goal ->
        # Update goal status
        updated_goal = ResearchGoal.update_status(goal, :in_progress)

        task =
          Task.Supervisor.async_nolink(
            ChatBot.Knowledge.AgentSupervisor,
            fn -> ResearchAgent.research(updated_goal, mock: mock?) end
          )

        {task.ref, {session_id, goal.id}}
      end)
      |> Map.new()

    new_stats = %{
      state.stats
      | active_agents: map_size(state.agent_tasks) + map_size(agent_refs)
    }

    {agent_refs, %{state | stats: new_stats}}
  end

  defp check_contradictions(candidates) do
    if BeliefStore.ready?() do
      Enum.map(candidates, fn candidate ->
        entity = candidate.finding.entity
        predicate = normalize_predicate(entity)

        case BeliefStore.query_beliefs(predicate: predicate) do
          {:ok, existing} ->
            conflicts = find_conflicting_beliefs(candidate.finding.claim, existing)
            %{candidate | existing_contradictions: conflicts}

          _ ->
            candidate
        end
      end)
    else
      candidates
    end
  end

  defp find_conflicting_beliefs(claim, existing_beliefs) do
    claim_lower = String.downcase(claim)

    existing_beliefs
    |> Enum.filter(fn belief ->
      belief_text = to_string(belief.object) |> String.downcase()

      # Check for potential conflicts
      has_negation_difference?(claim_lower, belief_text) or
        has_number_disagreement?(claim_lower, belief_text)
    end)
    |> Enum.map(fn belief ->
      %{
        id: belief.id,
        object: belief.object,
        confidence: belief.confidence,
        source: belief.source
      }
    end)
  end

  defp has_negation_difference?(c1, c2) do
    negation_words = ["not", "no", "never", "none"]
    c1_has_negation = Enum.any?(negation_words, &String.contains?(c1, &1))
    c2_has_negation = Enum.any?(negation_words, &String.contains?(c2, &1))
    c1_has_negation != c2_has_negation
  end

  defp has_number_disagreement?(c1, c2) do
    numbers1 = Regex.scan(~r/\d+/, c1) |> List.flatten()
    numbers2 = Regex.scan(~r/\d+/, c2) |> List.flatten()

    if length(numbers1) > 0 and length(numbers2) > 0 do
      # Check if numbers are significantly different
      n1 = numbers1 |> Enum.map(&String.to_integer/1) |> Enum.max()
      n2 = numbers2 |> Enum.map(&String.to_integer/1) |> Enum.max()
      min_val = min(n1, n2)
      max_val = max(n1, n2)
      min_val > 0 and (max_val - min_val) / min_val > 0.2
    else
      false
    end
  end

  defp normalize_predicate(entity) when is_binary(entity) do
    entity
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.to_atom()
  end

  defp normalize_predicate(_), do: :unknown

  defp update_session_metrics(state, session_id, goal_id, findings, candidates) do
    case Map.get(state.sessions, session_id) do
      nil ->
        state

      session ->
        # Update goal status
        updated_goals =
          Enum.map(session.goals, fn goal ->
            if goal.id == goal_id do
              ResearchGoal.update_status(goal, :completed)
            else
              goal
            end
          end)

        updated_session =
          session
          |> Map.put(:goals, updated_goals)
          |> LearningSession.record_findings(length(candidates))

        new_sessions = Map.put(state.sessions, session_id, updated_session)

        new_stats = %{
          state.stats
          | total_findings: state.stats.total_findings + length(findings)
        }

        %{state | sessions: new_sessions, stats: new_stats}
    end
  end

  defp mark_goal_failed(state, session_id, goal_id) do
    case Map.get(state.sessions, session_id) do
      nil ->
        state

      session ->
        updated_goals =
          Enum.map(session.goals, fn goal ->
            if goal.id == goal_id do
              ResearchGoal.update_status(goal, :failed)
            else
              goal
            end
          end)

        updated_session = Map.put(session, :goals, updated_goals)
        new_sessions = Map.put(state.sessions, session_id, updated_session)

        %{state | sessions: new_sessions}
    end
  end

  defp maybe_complete_session(state, session_id) do
    case Map.get(state.sessions, session_id) do
      nil ->
        state

      %LearningSession{status: :active} = session ->
        # Check if all goals are done
        all_done =
          Enum.all?(session.goals, fn goal ->
            goal.status in [:completed, :failed]
          end)

        # Check if any agents still running for this session
        agents_running =
          Enum.any?(state.agent_tasks, fn {_ref, {sid, _gid}} ->
            sid == session_id
          end)

        if all_done and not agents_running do
          completed = LearningSession.complete(session)
          new_sessions = Map.put(state.sessions, session_id, completed)

          Logger.info("Session completed",
            session_id: session_id,
            findings: completed.findings_count
          )

          %{state | sessions: new_sessions}
        else
          state
        end

      _ ->
        state
    end
  end

  defp maybe_filter_by_status(sessions, nil), do: sessions

  defp maybe_filter_by_status(sessions, status) do
    Enum.filter(sessions, &(&1.status == status))
  end

  defp count_active_sessions(sessions) do
    sessions
    |> Map.values()
    |> Enum.count(&(&1.status == :active))
  end
end
