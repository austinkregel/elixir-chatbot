defmodule Brain.Knowledge.LearningCenter do
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

  alias Brain.Knowledge.{ResearchAgent, Corroborator, ReviewQueue}
  alias Brain.Knowledge.Types.{ResearchGoal, LearningSession, Investigation}
  alias Brain.Epistemic.BeliefStore

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
  Starts a training session using domain-specific NLP tasks.

  This uses curated benchmark tasks (Question Answering, Commonsense, etc.)
  instead of web sources, providing high-quality training data for child agents.

  ## Options
    - :capability - Training capability (:question_answering, :commonsense, :sentiment, :all)
    - :max_tasks - Maximum task files to use (default: 5)
    - :max_instances - Maximum instances per task (default: 20)

  ## Example

      {:ok, session} = LearningCenter.start_task_training(:commonsense)
      {:ok, session} = LearningCenter.start_task_training(:question_answering, max_tasks: 10)
  """
  @spec start_task_training(atom(), keyword()) :: {:ok, LearningSession.t()} | {:error, term()}
  def start_task_training(capability \\ :all, opts \\ []) do
    topic = "task_training:#{capability}"
    task_opts = Keyword.merge(opts, sources: [:task], capability: capability)
    start_session(topic, task_opts)
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

    # Decompose topic into research goals
    goals = decompose_topic(topic, opts)

    session =
      Enum.reduce(goals, session, fn goal, sess ->
        LearningSession.add_goal(sess, goal)
      end)

    # Start research agents for each goal (pass full opts for sources, etc.)
    {agent_refs, updated_state} = dispatch_agents(goals, session.id, opts, state)

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

        # Dispatch agent for the new goal (use default opts)
        {agent_refs, updated_state} = dispatch_agents([goal], session_id, [], state)

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

        # === SCIENTIFIC METHOD APPROACH ===
        # 1. Create investigation from goal
        # 2. Test hypotheses against evidence (findings)
        # 3. Convert supported hypotheses to review candidates
        # 4. Track falsified hypotheses for learning

        new_state =
          process_findings_scientifically(
            state,
            session_id,
            goal_id,
            findings,
            remaining_tasks
          )

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
    case Process.whereis(Brain.Knowledge.AgentSupervisor) do
      nil ->
        # Start the supervisor - it should be started by the application
        Logger.debug("Agent supervisor not found, will be started by application")

      _pid ->
        :ok
    end
  end

  # Scientific investigation logging
  defp log_investigation_results(investigation) do
    summary = Investigation.summary(investigation)

    Logger.info("Investigation concluded",
      topic: summary.topic,
      hypotheses_tested: summary.total_hypotheses,
      supported: summary.supported,
      falsified: summary.falsified,
      inconclusive: summary.inconclusive,
      promotable: summary.promotable,
      conclusion: summary.conclusion
    )

    # Log any falsified hypotheses for learning
    investigation.hypotheses
    |> Enum.filter(&(&1.status == :falsified))
    |> Enum.each(fn hyp ->
      Logger.debug("Hypothesis falsified",
        claim: hyp.claim,
        contradicting_sources: length(hyp.contradicting_evidence)
      )
    end)
  end

  defp find_goal(nil, _goal_id), do: nil
  defp find_goal(session, goal_id) do
    Enum.find(session.goals, &(&1.id == goal_id))
  end

  # Process findings using the scientific method
  defp process_findings_scientifically(state, session_id, goal_id, findings, remaining_tasks) do
    session = Map.get(state.sessions, session_id)
    goal = find_goal(session, goal_id)

    if goal do
      # Create investigation from goal's questions
      investigation = ResearchGoal.to_investigation(goal)

      Logger.info("Starting scientific investigation",
        session_id: session_id,
        hypotheses: length(investigation.hypotheses),
        evidence: length(findings)
      )

      # Test hypotheses against the evidence
      {:ok, concluded} = Corroborator.test_hypotheses(investigation, findings)

      # Log scientific outcomes
      log_investigation_results(concluded)

      # Convert supported hypotheses to review candidates
      candidates = Corroborator.hypotheses_to_candidates(concluded, session_id: session_id)

      # Check for contradictions with existing beliefs
      candidates = check_contradictions(candidates)

      # Add to review queue
      Enum.each(candidates, fn candidate ->
        ReviewQueue.add(%{candidate | session_id: session_id})
      end)

      # Update session with investigation results
      updated_session = LearningSession.add_investigation(session, concluded)
      updated_session = LearningSession.record_findings(updated_session, length(findings))

      # Update state
      new_sessions = Map.put(state.sessions, session_id, updated_session)
      new_stats = %{state.stats | total_findings: state.stats.total_findings + length(findings)}

      %{state |
        sessions: new_sessions,
        stats: new_stats,
        agent_tasks: remaining_tasks
      }
    else
      # Fallback to traditional corroboration if goal not found
      process_findings_traditional(state, session_id, goal_id, findings, remaining_tasks)
    end
  end

  # Traditional corroboration (fallback)
  defp process_findings_traditional(state, session_id, goal_id, findings, remaining_tasks) do
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
    %{new_state | agent_tasks: remaining_tasks}
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
    # Generate questions using data-driven approaches
    # 1. Find similar past questions from memory
    # 2. Extract question patterns from the topic using POS tagging
    # 3. Fall back to minimal defaults only if needed

    memory_questions = extract_questions_from_memory(topic)
    pos_questions = generate_questions_with_pos(topic)

    # Combine unique questions, preferring memory-based ones
    combined = (memory_questions ++ pos_questions) |> Enum.uniq()

    if Enum.empty?(combined) do
      # Minimal fallback - just the topic as a query
      [topic]
    else
      Enum.take(combined, 5)
    end
  end

  # Search memory for similar topics and extract question patterns
  defp extract_questions_from_memory(topic) do
    alias Brain.Memory.Store

    case Store.query_similar(topic, 10) do
      {:ok, episodes} ->
        episodes
        |> Enum.flat_map(fn {episode, _similarity} ->
          # Extract question-like text from episode state
          extract_questions_from_text(episode.state)
        end)
        |> Enum.uniq()
        |> Enum.take(3)

      {:error, _} ->
        []
    end
  end

  # Extract question sentences from text using tokenizer
  defp extract_questions_from_text(text) when is_binary(text) do
    alias Brain.ML.Tokenizer

    # Split into sentences and find questions
    text
    |> Tokenizer.split_sentences()
    |> Enum.filter(&Tokenizer.ends_with_question?/1)
    |> Enum.take(2)
  end

  defp extract_questions_from_text(_), do: []

  # Generate question variations using POS tagging to understand topic structure
  defp generate_questions_with_pos(topic) do
    alias Brain.ML.{Tokenizer, POSTagger}

    tokens = Tokenizer.tokenize_words(topic)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)
        generate_questions_from_pos_analysis(tokens, tags, topic)

      {:error, _} ->
        # If POS tagger unavailable, use topic directly
        [topic]
    end
  end

  # Generate contextually appropriate questions based on POS analysis
  defp generate_questions_from_pos_analysis(tokens, tags, topic) do
    token_tags = Enum.zip(tokens, tags)

    # Find the main noun(s) in the topic
    nouns =
      token_tags
      |> Enum.filter(fn {_token, tag} -> tag in ["NOUN", "PROPN"] end)
      |> Enum.map(fn {token, _tag} -> token end)

    # Find verbs if present (for process/action topics)
    verbs =
      token_tags
      |> Enum.filter(fn {_token, tag} -> tag == "VERB" end)
      |> Enum.map(fn {token, _tag} -> token end)

    # Build questions based on what we found
    questions = []

    # If we have nouns, they're likely the subject
    questions =
      if length(nouns) > 0 do
        main_noun = Enum.join(nouns, " ")
        questions ++ ["#{main_noun}"]
      else
        questions
      end

    # If we have verbs, the topic might be about a process
    questions =
      if length(verbs) > 0 do
        questions ++ [topic]
      else
        questions
      end

    # Add the full topic as-is if it's substantive
    questions =
      if length(tokens) > 1 do
        questions ++ [topic]
      else
        questions
      end

    Enum.uniq(questions)
  end

  defp dispatch_agents(goals, session_id, opts, state) do
    # Limit concurrent agents
    available_slots = @max_concurrent_agents - map_size(state.agent_tasks)
    goals_to_dispatch = Enum.take(goals, available_slots)

    # Extract research options (sources, mock, etc.)
    research_opts = [
      mock: Keyword.get(opts, :mock, false),
      sources: Keyword.get(opts, :sources, [:web]),
      max_pages: Keyword.get(opts, :max_tasks, 5),
      max_instances: Keyword.get(opts, :max_instances, 20)
    ]

    agent_refs =
      goals_to_dispatch
      |> Enum.map(fn goal ->
        # Update goal status
        updated_goal = ResearchGoal.update_status(goal, :in_progress)

        task =
          Task.Supervisor.async_nolink(
            Brain.Knowledge.AgentSupervisor,
            fn -> ResearchAgent.research(updated_goal, research_opts) end
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
    negation_words = Brain.LinguisticData.negation_words()
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
