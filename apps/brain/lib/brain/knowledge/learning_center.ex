defmodule Brain.Knowledge.LearningCenter do
  # Tasks.Source lives in the tasks app and may be unavailable at compile time
  @compile {:no_warn_undefined, Tasks.Source}
  @moduledoc "Central orchestrator for the Knowledge Expansion System.\n\nThe Learning Center:\n- Manages learning sessions (triggered or scheduled)\n- Maintains a goal queue with research objectives\n- Dispatches Research Agents as supervised Tasks\n- Collects and synthesizes agent findings\n- Routes vetted findings to the Admin review queue\n\n## Example\n\n    # Start a learning session\n    {:ok, session} = LearningCenter.start_session(\"European capitals\")\n\n    # Check session status\n    {:ok, session} = LearningCenter.get_session(session.id)\n\n    # List active sessions\n    sessions = LearningCenter.list_sessions()\n"

  alias Brain.ML
  alias Brain.Knowledge.Types
  alias Brain.Knowledge
  use GenServer
  require Logger

  alias Knowledge.{ResearchAgent, Corroborator, ReviewQueue}
  alias Types.{ResearchGoal, LearningSession, Investigation, Hypothesis}
  alias Brain.Epistemic.BeliefStore
  alias Brain.Lattice

  @max_concurrent_agents 5
  @session_timeout_ms 300_000

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Starts a new learning session for a topic.\n\n## Options\n  - :questions - List of specific questions to research\n  - :max_goals - Maximum number of goals to generate\n  - :priority - :low | :normal | :high\n  - :mock - If true, uses mock data for testing\n"
  @spec start_session(String.t(), keyword()) :: {:ok, LearningSession.t()} | {:error, term()}
  def start_session(topic, opts \\ []) when is_binary(topic) do
    GenServer.call(__MODULE__, {:start_session, topic, opts})
  end

  @doc "Starts a training session using domain-specific NLP tasks.\n\nThis uses curated benchmark tasks (Question Answering, Commonsense, etc.)\ninstead of web sources, providing high-quality training data for child agents.\n\n## Options\n  - :capability - Training capability (:question_answering, :commonsense, :sentiment, :all)\n  - :max_tasks - Maximum task files to use (default: 5)\n  - :max_instances - Maximum instances per task (default: 20)\n\n## Example\n\n    {:ok, session} = LearningCenter.start_task_training(:commonsense)\n    {:ok, session} = LearningCenter.start_task_training(:question_answering, max_tasks: 10)\n"
  @spec start_task_training(atom(), keyword()) :: {:ok, LearningSession.t()} | {:error, term()}
  def start_task_training(capability \\ :all, opts \\ []) do
    topic = "task_training:#{capability}"
    task_opts = Keyword.merge(opts, sources: [:task], capability: capability)
    start_session(topic, task_opts)
  end

  @doc "Adds a goal to an existing session.\n"
  @spec add_goal(String.t(), ResearchGoal.t()) :: :ok | {:error, term()}
  def add_goal(session_id, %ResearchGoal{} = goal) do
    GenServer.call(__MODULE__, {:add_goal, session_id, goal})
  end

  @doc "Gets a session by ID.\n"
  @spec get_session(String.t()) :: {:ok, LearningSession.t()} | {:error, :not_found}
  def get_session(session_id) when is_binary(session_id) do
    GenServer.call(__MODULE__, {:get_session, session_id})
  end

  @doc "Cancels a session.\n"
  @spec cancel_session(String.t()) :: :ok | {:error, term()}
  def cancel_session(session_id) when is_binary(session_id) do
    GenServer.call(__MODULE__, {:cancel_session, session_id})
  end

  @doc "Lists all sessions.\n\n## Options\n  - :status - Filter by status (:active, :completed, :cancelled)\n  - :limit - Maximum number to return\n"
  @spec list_sessions(keyword()) :: [LearningSession.t()]
  def list_sessions(opts \\ []) do
    GenServer.call(__MODULE__, {:list_sessions, opts})
  end

  @doc "Gets statistics about the Learning Center.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Checks if the service is ready.\n"
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Alias for `add_goal/2` — enqueue a follow-up goal on an active session."
  def spawn_followup_goal(session_id, %ResearchGoal{} = goal) when is_binary(session_id) do
    add_goal(session_id, goal)
  end

  @doc """
  Evaluates structured predictions on hypotheses using `BeliefStore` (when ready).

  Returns `{:error, :not_ready}` if `BeliefStore.ready?/0` is false.
  """
  @spec test_predictions(Investigation.t()) ::
          {:ok, Investigation.t()} | {:error, :not_ready}
  def test_predictions(%Investigation{} = investigation) do
    GenServer.call(__MODULE__, {:test_predictions, investigation}, 60_000)
  end

  @impl true
  def init(_opts) do
    ensure_agent_supervisor_started()

    state = %{
      sessions: %{},
      agent_tasks: %{},
      task_sessions: MapSet.new(),
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

    goals =
      if Keyword.get(opts, :sources) == [:task] do
        build_task_goals(topic, opts)
      else
        decompose_topic(topic, opts)
      end

    session =
      Enum.reduce(goals, session, fn goal, sess ->
        LearningSession.add_goal(sess, goal)
      end)

    {agent_refs, updated_state} = dispatch_agents(goals, session.id, opts, state)

    is_task_session = Keyword.get(opts, :sources) == [:task]

    new_state = %{
      updated_state
      | sessions: Map.put(updated_state.sessions, session.id, session),
        agent_tasks: Map.merge(updated_state.agent_tasks, agent_refs),
        task_sessions:
          if(is_task_session,
            do: MapSet.put(updated_state.task_sessions, session.id),
            else: updated_state.task_sessions
          ),
        stats: %{updated_state.stats | total_sessions: updated_state.stats.total_sessions + 1}
    }

    Logger.info("Learning session started",
      session_id: session.id,
      topic: topic,
      goals: length(goals)
    )

    # Persist session to Atlas
    source_type = if is_task_session, do: "task", else: "web"
    persist_session_to_atlas(session, source_type)
    Enum.each(goals, &persist_goal_to_atlas(&1, session.id))

    Process.send_after(self(), {:session_timeout, session.id}, @session_timeout_ms)

    {:reply, {:ok, session}, new_state}
  end

  @impl true
  def handle_call({:test_predictions, %Investigation{} = inv}, _from, state) do
    if BeliefStore.ready?() do
      {:reply, {:ok, do_test_predictions(inv)}, state}
    else
      {:reply, {:error, :not_ready}, state}
    end
  end

  @impl true
  def handle_call({:add_goal, session_id, goal}, _from, state) do
    case Map.get(state.sessions, session_id) do
      nil ->
        {:reply, {:error, :session_not_found}, state}

      session ->
        updated_session = LearningSession.add_goal(session, goal)
        new_sessions = Map.put(state.sessions, session_id, updated_session)
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
        {cancelled_refs, remaining_refs} =
          state.agent_tasks
          |> Enum.split_with(fn {_ref, {sid, _gid}} -> sid == session_id end)

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
        persist_session_update_to_atlas(cancelled_session)

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
    Process.demonitor(ref, [:flush])

    case Map.pop(state.agent_tasks, ref) do
      {{session_id, goal_id}, remaining_tasks} ->
        Logger.debug("Agent completed",
          session_id: session_id,
          goal_id: goal_id,
          findings: length(findings)
        )

        new_state =
          if MapSet.member?(state.task_sessions, session_id) do
            process_task_findings(state, session_id, goal_id, findings, remaining_tasks)
          else
            process_findings_scientifically(
              state,
              session_id,
              goal_id,
              findings,
              remaining_tasks
            )
          end

        new_state = maybe_complete_session(new_state, session_id)

        {:noreply, new_state}

      {nil, _} ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({ref, {:error, reason}}, state) do
    Process.demonitor(ref, [:flush])

    case Map.pop(state.agent_tasks, ref) do
      {{session_id, goal_id}, remaining_tasks} ->
        Logger.warning("Agent failed",
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
  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
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
        completed = LearningSession.complete(session)
        new_sessions = Map.put(state.sessions, session_id, completed)
        persist_session_update_to_atlas(completed)

        {:noreply, %{state | sessions: new_sessions}}

      _ ->
        {:noreply, state}
    end
  end

  defp ensure_agent_supervisor_started do
    case Process.whereis(Brain.Knowledge.AgentSupervisor) do
      nil ->
        Logger.debug("Agent supervisor not found, will be started by application")

      _pid ->
        :ok
    end
  end

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

    investigation.hypotheses
    |> Enum.filter(&(&1.status == :falsified))
    |> Enum.each(fn hyp ->
      Logger.debug("Hypothesis falsified",
        claim: hyp.claim,
        contradicting_sources: length(hyp.contradicting_evidence)
      )
    end)
  end

  defp find_goal(nil, _goal_id) do
    nil
  end

  defp find_goal(session, goal_id) do
    Enum.find(session.goals, &(&1.id == goal_id))
  end

  defp process_findings_scientifically(state, session_id, goal_id, findings, remaining_tasks) do
    session = Map.get(state.sessions, session_id)
    goal = find_goal(session, goal_id)

    if goal do
      investigation = ResearchGoal.to_investigation(goal)

      Logger.info("Starting scientific investigation",
        session_id: session_id,
        hypotheses: length(investigation.hypotheses),
        evidence: length(findings)
      )

      {:ok, concluded} = Corroborator.test_hypotheses(investigation, findings)
      log_investigation_results(concluded)
      candidates = Corroborator.hypotheses_to_candidates(concluded, session_id: session_id)
      candidates = check_contradictions(candidates)

      Enum.each(candidates, fn candidate ->
        ReviewQueue.add(%{candidate | session_id: session_id})
      end)

      updated_session = LearningSession.add_investigation(session, concluded)
      updated_session = LearningSession.record_findings(updated_session, length(findings))
      new_sessions = Map.put(state.sessions, session_id, updated_session)
      new_stats = %{state.stats | total_findings: state.stats.total_findings + length(findings)}

      lattice = Investigation.rank_hypotheses(concluded)
      margin = Lattice.margin(lattice)
      dominant_gap = %{dimension: :hypothesis_margin, score: margin}

      if Process.whereis(Brain.PubSub) do
        Phoenix.PubSub.broadcast(
          Brain.PubSub,
          "learning:investigation",
          {:investigation_concluded, session_id, concluded.id, lattice, dominant_gap, goal.topic}
        )
      end

      # Persist investigation + goal status + session update to Atlas
      persist_investigation_to_atlas(concluded, session_id)
      persist_goal_status_to_atlas(goal_id, :completed)
      persist_session_update_to_atlas(updated_session)

      %{state | sessions: new_sessions, stats: new_stats, agent_tasks: remaining_tasks}
    else
      process_findings_traditional(state, session_id, goal_id, findings, remaining_tasks)
    end
  end

  defp process_task_findings(state, session_id, goal_id, findings, remaining_tasks) do
    session = Map.get(state.sessions, session_id)

    # Route task findings to TrainingExampleBuffer for model improvement
    if Code.ensure_loaded?(Brain.ML.TrainingExampleBuffer) and
         function_exported?(Brain.ML.TrainingExampleBuffer, :add_example, 2) do
      Enum.each(findings, fn finding ->
        if finding.claim && finding.entity do
          try do
            Brain.ML.TrainingExampleBuffer.add_example(finding.claim, finding.entity)
          rescue
            _ -> :ok
          end
        end
      end)
    end

    # Mark goal as completed and update session metrics directly (no hypothesis testing)
    updated_session =
      if session do
        updated_goals =
          Enum.map(session.goals, fn goal ->
            if goal.id == goal_id do
              ResearchGoal.update_status(goal, :completed)
            else
              goal
            end
          end)

        session
        |> Map.put(:goals, updated_goals)
        |> LearningSession.record_findings(length(findings))
      else
        session
      end

    new_sessions =
      if updated_session do
        Map.put(state.sessions, session_id, updated_session)
      else
        state.sessions
      end

    new_stats = %{
      state.stats
      | total_findings: state.stats.total_findings + length(findings)
    }

    Logger.info("Task findings processed directly (no hypothesis testing)",
      session_id: session_id,
      goal_id: goal_id,
      findings: length(findings)
    )

    # Persist goal status and session update to Atlas
    persist_goal_status_to_atlas(goal_id, :completed)

    if updated_session do
      persist_session_update_to_atlas(updated_session)
    end

    %{state | sessions: new_sessions, stats: new_stats, agent_tasks: remaining_tasks}
  end

  defp process_findings_traditional(state, session_id, goal_id, findings, remaining_tasks) do
    {:ok, candidates} = Corroborator.corroborate(findings, include_uncorroborated: true)
    candidates = check_contradictions(candidates)

    Enum.each(candidates, fn candidate ->
      ReviewQueue.add(%{candidate | session_id: session_id})
    end)

    new_state = update_session_metrics(state, session_id, goal_id, findings, candidates)
    %{new_state | agent_tasks: remaining_tasks}
  end

  defp build_task_goals(topic, opts) do
    capability = Keyword.get(opts, :capability, :all)
    priority = Keyword.get(opts, :priority, :normal)

    categories =
      if Code.ensure_loaded?(Tasks.Source) and function_exported?(Tasks.Source, :capability_categories, 1) do
        Tasks.Source.capability_categories(capability)
      else
        [to_string(capability)]
      end

    categories
    |> Enum.map(fn category ->
      ResearchGoal.new("#{category} training",
        questions: ["Train on #{category} benchmark tasks"],
        constraints: %{source: :task, category: category},
        priority: priority
      )
    end)
    |> case do
      [] -> [ResearchGoal.new(topic, priority: priority)]
      goals -> goals
    end
  end

  defp decompose_topic(topic, opts) do
    questions = Keyword.get(opts, :questions, [])
    max_goals = Keyword.get(opts, :max_goals, 3)
    priority = Keyword.get(opts, :priority, :normal)

    base_goal =
      ResearchGoal.new(topic, questions: questions, priority: priority)

    additional_goals =
      if questions == [] do
        generate_default_questions(topic)
        |> Enum.take(max_goals - 1)
        |> Enum.map(fn q ->
          ResearchGoal.new(topic, questions: [q], priority: priority)
        end)
      else
        []
      end

    [base_goal | additional_goals]
    |> Enum.take(max_goals)
  end

  defp generate_default_questions(topic) do
    memory_questions = extract_questions_from_memory(topic)
    pos_questions = generate_questions_with_pos(topic)
    combined = (memory_questions ++ pos_questions) |> Enum.uniq()

    if Enum.empty?(combined) do
      [topic]
    else
      Enum.take(combined, 5)
    end
  end

  defp extract_questions_from_memory(topic) do
    alias Brain.Memory.Store

    case Store.query_similar(topic, 10) do
      {:ok, episodes} ->
        episodes
        |> Enum.flat_map(fn {episode, _similarity} ->
          extract_questions_from_text(episode.state)
        end)
        |> Enum.uniq()
        |> Enum.take(3)

      {:error, _} ->
        []
    end
  end

  defp extract_questions_from_text(text) when is_binary(text) do
    alias Brain.ML.Tokenizer

    text
    |> Tokenizer.split_sentences()
    |> Enum.filter(&Tokenizer.ends_with_question?/1)
    |> Enum.take(2)
  end

  defp extract_questions_from_text(_) do
    []
  end

  defp generate_questions_with_pos(topic) do
    alias ML.{Tokenizer, POSTagger}

    tokens = Tokenizer.tokenize_words(topic)

    case POSTagger.load_model() do
      {:ok, model} ->
        tags = POSTagger.predict_tags(tokens, model)
        generate_questions_from_pos_analysis(tokens, tags, topic)

      {:error, _} ->
        [topic]
    end
  end

  defp generate_questions_from_pos_analysis(tokens, tags, topic) do
    token_tags = Enum.zip(tokens, tags)

    nouns =
      token_tags
      |> Enum.filter(fn {_token, tag} -> tag in ["NOUN", "PROPN"] end)
      |> Enum.map(fn {token, _tag} -> token end)

    verbs =
      token_tags
      |> Enum.filter(fn {_token, tag} -> tag == "VERB" end)
      |> Enum.map(fn {token, _tag} -> token end)

    questions = []

    questions =
      if nouns != [] do
        main_noun = Enum.join(nouns, " ")
        questions ++ ["#{main_noun}"]
      else
        questions
      end

    questions =
      if verbs != [] do
        questions ++ [topic]
      else
        questions
      end

    questions =
      if length(tokens) > 1 do
        questions ++ [topic]
      else
        questions
      end

    Enum.uniq(questions)
  end

  defp dispatch_agents(goals, session_id, opts, state) do
    available_slots = @max_concurrent_agents - map_size(state.agent_tasks)
    goals_to_dispatch = Enum.take(goals, available_slots)

    # When the HTTP client is a mock (test env), default to mock mode
    # to avoid hitting snapshot misses for auto-triggered research
    http_client = Application.get_env(:brain, :http_client, Req)
    mock_default = http_client != Req

    research_opts = [
      mock: Keyword.get(opts, :mock, mock_default),
      sources: Keyword.get(opts, :sources, [:web]),
      max_pages: Keyword.get(opts, :max_tasks, 5),
      max_instances: Keyword.get(opts, :max_instances, 20)
    ]

    agent_refs =
      goals_to_dispatch
      |> Enum.map(fn goal ->
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

  defp has_negation_difference?(c1, c2),
    do: Brain.Knowledge.ContradictionDetector.has_negation_difference?(c1, c2)

  defp has_number_disagreement?(c1, c2),
    do: Brain.Knowledge.ContradictionDetector.has_number_disagreement?(c1, c2)

  defp normalize_predicate(entity) when is_binary(entity) do
    normalized = entity |> String.downcase() |> String.replace(~r/[^a-z0-9]+/, "_")
    String.to_existing_atom(normalized)
  rescue
    ArgumentError -> :unknown
  end

  defp normalize_predicate(_) do
    :unknown
  end

  defp update_session_metrics(state, session_id, goal_id, findings, candidates) do
    case Map.get(state.sessions, session_id) do
      nil ->
        state

      session ->
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

        persist_goal_status_to_atlas(goal_id, :failed)

        %{state | sessions: new_sessions}
    end
  end

  defp maybe_complete_session(state, session_id) do
    case Map.get(state.sessions, session_id) do
      nil ->
        state

      %LearningSession{status: :active} = session ->
        all_done =
          Enum.all?(session.goals, fn goal ->
            goal.status in [:completed, :failed]
          end)

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

          persist_session_update_to_atlas(completed)

          %{state | sessions: new_sessions}
        else
          state
        end

      _ ->
        state
    end
  end

  defp maybe_filter_by_status(sessions, nil) do
    sessions
  end

  defp maybe_filter_by_status(sessions, status) do
    Enum.filter(sessions, &(&1.status == status))
  end

  defp count_active_sessions(sessions) do
    sessions
    |> Map.values()
    |> Enum.count(&(&1.status == :active))
  end

  # ============================================================================
  # Atlas Persistence Helpers
  # ============================================================================

  defp atlas_available? do
    Code.ensure_loaded?(Atlas.Learning) and
      Code.ensure_loaded?(Atlas.Repo) and
      match?({:ok, _}, try_atlas_repo())
  end

  defp try_atlas_repo do
    try do
      {:ok, Process.whereis(Atlas.Repo)}
    rescue
      _ -> {:error, :not_available}
    catch
      _, _ -> {:error, :not_available}
    end
  end

  defp persist_session_to_atlas(%LearningSession{} = session, source_type) do
    if atlas_available?() do
      Brain.AtlasIntegration.async(fn ->
        Atlas.Learning.create_session(%{
          id: session.id,
          topic: session.topic,
          status: to_string(session.status),
          started_at: session.started_at,
          findings_count: session.findings_count,
          approved_count: session.approved_count,
          rejected_count: session.rejected_count,
          hypotheses_tested: session.hypotheses_tested,
          hypotheses_supported: session.hypotheses_supported,
          hypotheses_falsified: session.hypotheses_falsified,
          source_type: source_type
        })
      end)
    end
  end

  defp persist_goal_to_atlas(%ResearchGoal{} = goal, session_id) do
    if atlas_available?() do
      Brain.AtlasIntegration.async(fn ->
        Atlas.Learning.create_goal(%{
          id: goal.id,
          session_id: session_id,
          topic: goal.topic,
          questions: goal.questions,
          constraints: goal.constraints,
          priority: to_string(goal.priority),
          status: to_string(goal.status)
        })
      end)
    end
  end

  defp persist_goal_status_to_atlas(goal_id, new_status) do
    if atlas_available?() do
      Brain.AtlasIntegration.async(fn ->
        Atlas.Learning.update_goal_status(goal_id, to_string(new_status))
      end)
    end
  end

  defp persist_investigation_to_atlas(%Investigation{} = investigation, session_id) do
    if atlas_available?() do
      Brain.AtlasIntegration.async(fn ->
        {:ok, db_investigation} =
          Atlas.Learning.create_investigation(%{
            id: investigation.id,
            session_id: session_id,
            topic: investigation.topic,
            status: to_string(investigation.status),
            conclusion: if(investigation.conclusion, do: to_string(investigation.conclusion)),
            independent_variable: investigation.independent_variable,
            dependent_variable: investigation.dependent_variable,
            constants: investigation.constants,
            methodology_notes: investigation.methodology_notes,
            started_at: investigation.started_at,
            concluded_at: investigation.concluded_at
          })

        Enum.each(investigation.hypotheses, fn hyp ->
          Atlas.Learning.create_hypothesis(%{
            id: hyp.id,
            investigation_id: db_investigation.id,
            claim: hyp.claim,
            entity: hyp.entity,
            derived_from: hyp.derived_from,
            prediction: encode_hypothesis_prediction_for_atlas(hyp.prediction),
            status: to_string(hyp.status),
            confidence: hyp.confidence,
            confidence_level: to_string(hyp.confidence_level),
            source_count: hyp.source_count,
            replication_count: hyp.replication_count,
            tested_at: hyp.tested_at
          })
        end)

        Enum.each(investigation.evidence, fn finding ->
          Atlas.Learning.create_evidence(%{
            investigation_id: db_investigation.id,
            claim: finding.claim,
            entity: finding.entity,
            entity_type: finding.entity_type,
            source_url: finding.source && finding.source.url,
            source_domain: finding.source && finding.source.domain,
            source_title: finding.source && finding.source.title,
            source_reliability: finding.source && finding.source.reliability_score,
            source_bias: finding.source && to_string(finding.source.bias_rating),
            source_trust_tier: finding.source && to_string(finding.source.trust_tier),
            raw_context: finding.raw_context,
            confidence: finding.confidence,
            corroboration_group: finding.corroboration_group,
            evidence_type: "unassociated",
            extracted_at: finding.extracted_at
          })
        end)
      end)
    end
  end

  defp persist_session_update_to_atlas(%LearningSession{} = session) do
    if atlas_available?() do
      Brain.AtlasIntegration.async(fn ->
        Atlas.Learning.update_session(session.id, %{
          status: to_string(session.status),
          completed_at: session.completed_at,
          findings_count: session.findings_count,
          approved_count: session.approved_count,
          rejected_count: session.rejected_count,
          hypotheses_tested: session.hypotheses_tested,
          hypotheses_supported: session.hypotheses_supported,
          hypotheses_falsified: session.hypotheses_falsified
        })
      end)
    end
  end

  defp encode_hypothesis_prediction_for_atlas(pred) when is_map(pred) do
    Jason.encode!(pred)
  end

  defp encode_hypothesis_prediction_for_atlas(pred) when is_binary(pred), do: pred
  defp encode_hypothesis_prediction_for_atlas(nil), do: nil

  defp do_test_predictions(%Investigation{hypotheses: hs} = inv) when is_list(hs) do
    evaluated =
      hs
      |> Task.async_stream(
        &apply_prediction_evaluation/1,
        max_concurrency: 4,
        timeout: 5_000,
        on_timeout: :kill_task
      )
      |> Enum.zip(hs)
      |> Enum.map(fn
        {{:ok, h2}, _} -> h2
        {_other, h1} -> h1
      end)

    %{inv | hypotheses: evaluated}
  end

  defp apply_prediction_evaluation(%Hypothesis{} = h) do
    case h.prediction do
      %{type: :unstructured} ->
        h

      pred when is_binary(pred) ->
        h

      %{type: :belief_query, params: params} when is_map(params) ->
        opts = Map.to_list(params)

        case BeliefStore.query_beliefs(opts) do
          {:ok, beliefs} when beliefs == [] -> Hypothesis.evaluate(h)
          {:ok, _} -> Hypothesis.evaluate(h)
          _ -> h
        end

      %{type: :corroboration_count, params: params, expected: expected}
      when is_map(params) and is_integer(expected) ->
        min_sources = Map.get(params, :min_sources, expected)

        if h.source_count >= min_sources do
          Hypothesis.evaluate(h)
        else
          Hypothesis.evaluate(h)
        end

      _ ->
        h
    end
  end
end
