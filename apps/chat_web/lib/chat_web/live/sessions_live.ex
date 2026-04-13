defmodule ChatWeb.SessionsLive do
  @moduledoc "LiveView for detailed learning session inspection.

  Provides a dedicated page for browsing and drilling into learning sessions,
  research goals, scientific investigations, hypotheses, and evidence.

  ## Routes

      /sessions              - List all sessions with status filters
      /sessions/:session_id  - Detail view for a specific session
  "

  alias Phoenix.PubSub
  alias Brain.Knowledge
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Knowledge.LearningCenter

  @refresh_interval_ms 5000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "knowledge:review")
      :timer.send_interval(@refresh_interval_ms, self(), :refresh)
    end

    socket =
      socket
      |> assign(:view_mode, :list)
      |> assign(:sessions, [])
      |> assign(:session, nil)
      |> assign(:status_filter, :all)
      |> assign(:detail_tab, :overview)
      |> assign(:expanded_investigation_id, nil)
      |> assign(:expanded_evidence_ids, MapSet.new())
      |> assign(:lc_stats, %{total_sessions: 0, active_sessions: 0, active_agents: 0, total_findings: 0})
      |> assign(:page_title, "Learning Sessions")

    {:ok, socket}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    case params["session_id"] do
      nil ->
        sessions = load_sessions(socket.assigns.status_filter)
        stats = load_stats()

        socket =
          socket
          |> assign(:view_mode, :list)
          |> assign(:sessions, sessions)
          |> assign(:session, nil)
          |> assign(:lc_stats, stats)
          |> assign(:page_title, "Learning Sessions")

        {:noreply, socket}

      session_id ->
        case load_session(session_id) do
          {:ok, session} ->
            socket =
              socket
              |> assign(:view_mode, :detail)
              |> assign(:session, session)
              |> assign(:detail_tab, :overview)
              |> assign(:expanded_investigation_id, nil)
              |> assign(:expanded_evidence_ids, MapSet.new())
              |> assign(:page_title, session.topic || "Session #{String.slice(session_id, 0..7)}")

            {:noreply, socket}

          {:error, _} ->
            socket =
              socket
              |> put_flash(:error, "Session not found")
              |> push_navigate(to: ~p"/sessions")

            {:noreply, socket}
        end
    end
  end

  # -- Data Loading --
  # Uses Atlas.Learning for historical sessions (survives restarts),
  # falls back to LearningCenter for active session real-time data.

  defp load_sessions(:all) do
    atlas_sessions = load_sessions_from_atlas(nil)

    if atlas_sessions != [] do
      atlas_sessions
    else
      load_sessions_from_learning_center(nil)
    end
  end

  defp load_sessions(status) when status in [:active, :completed, :cancelled] do
    atlas_sessions = load_sessions_from_atlas(status)

    if atlas_sessions != [] do
      atlas_sessions
    else
      load_sessions_from_learning_center(status)
    end
  end

  defp load_session(session_id) do
    # Try Atlas first for full historical data with preloaded associations
    case load_session_from_atlas(session_id) do
      {:ok, session} ->
        {:ok, session}

      _ ->
        # Fall back to LearningCenter for active sessions with real-time agent status
        if LearningCenter.ready?() do
          LearningCenter.get_session(session_id)
        else
          {:error, :not_ready}
        end
    end
  end

  defp load_stats do
    atlas_stats = load_stats_from_atlas()
    lc_stats = load_stats_from_learning_center()

    # Merge: use Atlas for totals (historical), LearningCenter for real-time (active agents)
    %{
      total_sessions: max(atlas_stats[:total_sessions] || 0, lc_stats[:total_sessions] || 0),
      active_sessions: lc_stats[:active_sessions] || atlas_stats[:active_sessions] || 0,
      active_agents: lc_stats[:active_agents] || 0,
      total_findings: max(atlas_stats[:total_findings] || 0, lc_stats[:total_findings] || 0)
    }
  end

  defp load_sessions_from_atlas(status) do
    if atlas_available?() do
      try do
        opts = [limit: 100]
        opts = if status, do: Keyword.put(opts, :status, status), else: opts

        Atlas.Learning.list_sessions(opts)
        |> Enum.map(&atlas_session_to_brain_session/1)
      rescue
        _ -> []
      end
    else
      []
    end
  end

  defp load_session_from_atlas(session_id) do
    if atlas_available?() do
      try do
        case Atlas.Learning.session_with_details(session_id) do
          {:ok, atlas_session} ->
            {:ok, atlas_detail_to_brain_session(atlas_session)}

          error ->
            error
        end
      rescue
        _ -> {:error, :atlas_error}
      end
    else
      {:error, :atlas_unavailable}
    end
  end

  defp load_stats_from_atlas do
    if atlas_available?() do
      try do
        import Ecto.Query

        total =
          Atlas.Schemas.LearningSession
          |> Atlas.Repo.aggregate(:count, :id)

        active =
          Atlas.Schemas.LearningSession
          |> where([s], s.status == "active")
          |> Atlas.Repo.aggregate(:count, :id)

        total_findings =
          Atlas.Schemas.LearningSession
          |> Atlas.Repo.aggregate(:sum, :findings_count) || 0

        %{
          total_sessions: total || 0,
          active_sessions: active || 0,
          total_findings: total_findings || 0
        }
      rescue
        _ -> %{}
      end
    else
      %{}
    end
  end

  defp load_sessions_from_learning_center(nil) do
    if LearningCenter.ready?() do
      LearningCenter.list_sessions(limit: 100)
    else
      []
    end
  end

  defp load_sessions_from_learning_center(status) do
    if LearningCenter.ready?() do
      LearningCenter.list_sessions(status: status, limit: 100)
    else
      []
    end
  end

  defp load_stats_from_learning_center do
    if LearningCenter.ready?() do
      LearningCenter.stats()
    else
      %{total_sessions: 0, active_sessions: 0, active_agents: 0, total_findings: 0}
    end
  end

  defp atlas_available? do
    Code.ensure_loaded?(Atlas.Repo) and is_pid(Process.whereis(Atlas.Repo))
  rescue
    _ -> false
  catch
    _, _ -> false
  end

  # Convert Atlas schema to Brain LearningSession struct for template compatibility
  defp atlas_session_to_brain_session(%Atlas.Schemas.LearningSession{} = s) do
    alias Brain.Knowledge.Types.LearningSession, as: BrainSession

    %BrainSession{
      id: s.id,
      topic: s.topic,
      status: safe_to_atom(s.status),
      started_at: s.started_at,
      completed_at: s.completed_at,
      findings_count: s.findings_count || 0,
      approved_count: s.approved_count || 0,
      rejected_count: s.rejected_count || 0,
      hypotheses_tested: s.hypotheses_tested || 0,
      hypotheses_supported: s.hypotheses_supported || 0,
      hypotheses_falsified: s.hypotheses_falsified || 0,
      goals: [],
      investigations: []
    }
  end

  # Convert Atlas schema with preloaded associations to Brain structs
  defp atlas_detail_to_brain_session(%Atlas.Schemas.LearningSession{} = s) do
    alias Brain.Knowledge.Types.{LearningSession, ResearchGoal, Investigation, Hypothesis, Finding, SourceInfo}

    goals =
      (s.goals || [])
      |> Enum.map(fn g ->
        %ResearchGoal{
          id: g.id,
          topic: g.topic,
          questions: g.questions || [],
          constraints: g.constraints || %{},
          priority: safe_to_atom(g.priority),
          status: safe_to_atom(g.status),
          created_at: g.inserted_at
        }
      end)

    investigations =
      (s.investigations || [])
      |> Enum.map(fn inv ->
        hypotheses =
          (inv.hypotheses || [])
          |> Enum.map(fn h ->
            %Hypothesis{
              id: h.id,
              claim: h.claim,
              entity: h.entity,
              derived_from: h.derived_from,
              prediction: h.prediction,
              status: safe_to_atom(h.status),
              confidence: h.confidence || 0.0,
              confidence_level: safe_to_atom(h.confidence_level),
              source_count: h.source_count || 0,
              replication_count: h.replication_count || 0,
              tested_at: h.tested_at,
              created_at: h.inserted_at,
              supporting_evidence: [],
              contradicting_evidence: []
            }
          end)

        evidence =
          (inv.evidence || [])
          |> Enum.map(fn e ->
            source = %SourceInfo{
              url: e.source_url || "",
              domain: e.source_domain || "unknown",
              title: e.source_title,
              reliability_score: e.source_reliability || 0.5,
              bias_rating: safe_to_atom(e.source_bias),
              trust_tier: safe_to_atom(e.source_trust_tier)
            }

            %Finding{
              id: e.id,
              claim: e.claim || "",
              entity: e.entity || "",
              entity_type: e.entity_type,
              source: source,
              raw_context: e.raw_context || "",
              confidence: e.confidence || 0.5,
              corroboration_group: e.corroboration_group,
              extracted_at: e.extracted_at
            }
          end)

        %Investigation{
          id: inv.id,
          topic: inv.topic,
          hypotheses: hypotheses,
          evidence: evidence,
          control_evidence: [],
          independent_variable: inv.independent_variable || "source",
          dependent_variable: inv.dependent_variable || "claim",
          constants: inv.constants || [],
          status: safe_to_atom(inv.status),
          conclusion: safe_to_atom(inv.conclusion),
          started_at: inv.started_at,
          concluded_at: inv.concluded_at,
          methodology_notes: inv.methodology_notes
        }
      end)

    %LearningSession{
      id: s.id,
      topic: s.topic,
      status: safe_to_atom(s.status),
      started_at: s.started_at,
      completed_at: s.completed_at,
      findings_count: s.findings_count || 0,
      approved_count: s.approved_count || 0,
      rejected_count: s.rejected_count || 0,
      hypotheses_tested: s.hypotheses_tested || 0,
      hypotheses_supported: s.hypotheses_supported || 0,
      hypotheses_falsified: s.hypotheses_falsified || 0,
      goals: goals,
      investigations: investigations
    }
  end

  defp safe_to_atom(nil), do: nil
  defp safe_to_atom(val) when is_atom(val), do: val

  defp safe_to_atom(val) when is_binary(val) do
    String.to_existing_atom(val)
  rescue
    ArgumentError -> String.to_atom(val)
  end

  # -- Events --

  @impl true
  def handle_event("filter_status", %{"status" => status}, socket) do
    status_atom =
      case status do
        "active" -> :active
        "completed" -> :completed
        "cancelled" -> :cancelled
        _ -> :all
      end

    sessions = load_sessions(status_atom)

    {:noreply,
     socket
     |> assign(:status_filter, status_atom)
     |> assign(:sessions, sessions)}
  end

  def handle_event("change_tab", %{"tab" => tab}, socket) do
    tab_atom =
      case tab do
        "goals" -> :goals
        "investigations" -> :investigations
        "evidence" -> :evidence
        _ -> :overview
      end

    {:noreply, assign(socket, :detail_tab, tab_atom)}
  end

  def handle_event("toggle_investigation", %{"id" => id}, socket) do
    current = socket.assigns.expanded_investigation_id
    new_id = if current == id, do: nil, else: id
    {:noreply, assign(socket, :expanded_investigation_id, new_id)}
  end

  def handle_event("toggle_evidence", %{"id" => id}, socket) do
    current = socket.assigns.expanded_evidence_ids

    new_set =
      if MapSet.member?(current, id) do
        MapSet.delete(current, id)
      else
        MapSet.put(current, id)
      end

    {:noreply, assign(socket, :expanded_evidence_ids, new_set)}
  end

  # -- PubSub & Refresh --

  @impl true
  def handle_info(:refresh, socket) do
    {:noreply, refresh_data(socket)}
  end

  def handle_info({event, _data}, socket)
      when event in [
             :candidate_added,
             :candidate_approved,
             :candidate_rejected,
             :candidate_deferred
           ] do
    {:noreply, refresh_data(socket)}
  end

  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, refresh_data(socket)}
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  defp refresh_data(socket) do
    case socket.assigns.view_mode do
      :list ->
        sessions = load_sessions(socket.assigns.status_filter)
        stats = load_stats()

        socket
        |> assign(:sessions, sessions)
        |> assign(:lc_stats, stats)

      :detail ->
        if socket.assigns.session do
          case load_session(socket.assigns.session.id) do
            {:ok, session} -> assign(socket, :session, session)
            {:error, _} -> socket
          end
        else
          socket
        end
    end
  end

  # -- Render --

  @impl true
  def render(assigns) do
    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <%= if @view_mode == :detail and @session do %>
          <div class="flex items-center gap-2 text-sm text-base-content/60 mb-1">
            <.link navigate={~p"/sessions"} class="hover:text-primary transition-colors">
              Sessions
            </.link>
            <.icon name="hero-chevron-right" class="size-3" />
            <span class="text-base-content">{@session.topic || "Session"}</span>
          </div>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-xl font-bold">{@session.topic || "Session Detail"}</h1>
              <p class="text-sm text-base-content/60 font-mono">{@session.id}</p>
            </div>
            <span class={["badge", session_badge(@session.status)]}>
              {@session.status}
            </span>
          </div>
        <% else %>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-xl font-bold">Learning Sessions</h1>
              <p class="text-sm text-base-content/60">
                Inspect research sessions, goals, investigations, and evidence
              </p>
            </div>
          </div>
        <% end %>
      </:page_header>

      <div class="p-4 sm:p-6">
        <%= if @view_mode == :list do %>
          <.list_view
            sessions={@sessions}
            status_filter={@status_filter}
            lc_stats={@lc_stats}
          />
        <% else %>
          <.detail_view
            session={@session}
            detail_tab={@detail_tab}
            expanded_investigation_id={@expanded_investigation_id}
            expanded_evidence_ids={@expanded_evidence_ids}
          />
        <% end %>
      </div>
    </.app_shell>
    """
  end

  # ============================================================================
  # LIST VIEW
  # ============================================================================

  defp list_view(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Stats Bar -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="text-2xl font-bold">{@lc_stats[:total_sessions] || 0}</div>
          <div class="text-sm text-base-content/60">Total Sessions</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="text-2xl font-bold text-warning">{@lc_stats[:active_sessions] || 0}</div>
          <div class="text-sm text-base-content/60">Active</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="text-2xl font-bold text-info">{@lc_stats[:active_agents] || 0}</div>
          <div class="text-sm text-base-content/60">Active Agents</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="text-2xl font-bold text-success">{@lc_stats[:total_findings] || 0}</div>
          <div class="text-sm text-base-content/60">Total Findings</div>
        </div>
      </div>

      <!-- Status Filter Tabs -->
      <div class="flex gap-1 bg-base-100 rounded-xl border border-base-300/50 p-1">
        <%= for {label, value} <- [{"All", "all"}, {"Active", "active"}, {"Completed", "completed"}, {"Cancelled", "cancelled"}] do %>
          <button
            phx-click="filter_status"
            phx-value-status={value}
            class={[
              "px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              if(to_string(@status_filter) == value,
                do: "bg-primary text-primary-content",
                else: "text-base-content/60 hover:bg-base-200")
            ]}
          >
            {label}
          </button>
        <% end %>
      </div>

      <!-- Session Cards -->
      <%= if length(@sessions) == 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-12 text-center">
          <.icon name="hero-beaker" class="size-12 mx-auto mb-4 text-base-content/20" />
          <p class="text-base-content/50">No sessions found</p>
          <p class="text-sm text-base-content/40 mt-1">
            Start a training session from the Settings page
          </p>
        </div>
      <% else %>
        <div class="space-y-3">
          <%= for session <- @sessions do %>
            <.session_card session={session} />
          <% end %>
        </div>
      <% end %>
    </div>
    """
  end

  defp session_card(assigns) do
    failed_goals = Enum.count(assigns.session.goals, &(&1.status == :failed))
    completed_goals = Enum.count(assigns.session.goals, &(&1.status == :completed))
    total_goals = length(assigns.session.goals)

    assigns =
      assigns
      |> assign(:failed_goals, failed_goals)
      |> assign(:completed_goals, completed_goals)
      |> assign(:total_goals, total_goals)

    ~H"""
    <.link
      navigate={~p"/sessions/#{@session.id}"}
      class="block bg-base-100 rounded-xl border border-base-300/50 p-4 hover:border-primary/30 hover:shadow-lg transition-all group"
    >
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-3 min-w-0">
          <div class="min-w-0">
            <div class="font-semibold group-hover:text-primary transition-colors truncate">
              {@session.topic || "Untitled Session"}
            </div>
            <div class="text-xs text-base-content/40 font-mono">{@session.id}</div>
          </div>
        </div>
        <div class="flex items-center gap-2 shrink-0">
          <span class={["badge badge-sm", session_badge(@session.status)]}>
            {@session.status}
          </span>
          <.icon name="hero-chevron-right" class="size-4 text-base-content/30 group-hover:text-primary transition-colors" />
        </div>
      </div>

      <!-- Progress & Stats -->
      <div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-base-content/60">
        <span class="flex items-center gap-1">
          <.icon name="hero-flag" class="size-3" />
          {@completed_goals}/{@total_goals} goals
        </span>
        <%= if @failed_goals > 0 do %>
          <span class="flex items-center gap-1 text-error">
            <.icon name="hero-exclamation-triangle" class="size-3" />
            {@failed_goals} failed
          </span>
        <% end %>
        <%= if @session.findings_count > 0 do %>
          <span class="flex items-center gap-1 text-info">
            <.icon name="hero-document-magnifying-glass" class="size-3" />
            {@session.findings_count} findings
          </span>
        <% end %>
        <%= if @session.approved_count > 0 do %>
          <span class="flex items-center gap-1 text-success">
            <.icon name="hero-check-circle" class="size-3" />
            {@session.approved_count} approved
          </span>
        <% end %>
        <%= if @session.rejected_count > 0 do %>
          <span class="flex items-center gap-1 text-error/70">
            <.icon name="hero-x-circle" class="size-3" />
            {@session.rejected_count} rejected
          </span>
        <% end %>
        <%= if @session.hypotheses_tested > 0 do %>
          <span class="flex items-center gap-1">
            <.icon name="hero-beaker" class="size-3" />
            {@session.hypotheses_tested} hypotheses
          </span>
        <% end %>
        <%= if @session.started_at do %>
          <span class="text-base-content/40">
            {Calendar.strftime(@session.started_at, "%Y-%m-%d %H:%M:%S")}
          </span>
        <% end %>
      </div>
    </.link>
    """
  end

  # ============================================================================
  # DETAIL VIEW
  # ============================================================================

  defp detail_view(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Timestamps -->
      <div class="flex flex-wrap gap-4 text-sm text-base-content/60">
        <%= if @session.started_at do %>
          <span>
            Started: <span class="text-base-content font-medium">{Calendar.strftime(@session.started_at, "%Y-%m-%d %H:%M:%S")}</span>
          </span>
        <% end %>
        <%= if @session.completed_at do %>
          <span>
            Completed: <span class="text-base-content font-medium">{Calendar.strftime(@session.completed_at, "%Y-%m-%d %H:%M:%S")}</span>
          </span>
        <% end %>
      </div>

      <!-- Tab Navigation -->
      <div class="flex gap-1 bg-base-100 rounded-xl border border-base-300/50 p-1">
        <%= for {label, value, icon} <- [
          {"Overview", "overview", "hero-chart-bar-square"},
          {"Goals", "goals", "hero-flag"},
          {"Investigations", "investigations", "hero-beaker"},
          {"Evidence", "evidence", "hero-document-magnifying-glass"}
        ] do %>
          <button
            phx-click="change_tab"
            phx-value-tab={value}
            class={[
              "flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
              if(to_string(@detail_tab) == value,
                do: "bg-primary text-primary-content",
                else: "text-base-content/60 hover:bg-base-200")
            ]}
          >
            <.icon name={icon} class="size-4" />
            {label}
          </button>
        <% end %>
      </div>

      <!-- Tab Content -->
      <%= case @detail_tab do %>
        <% :overview -> %>
          <.overview_tab session={@session} />
        <% :goals -> %>
          <.goals_tab session={@session} />
        <% :investigations -> %>
          <.investigations_tab
            session={@session}
            expanded_investigation_id={@expanded_investigation_id}
          />
        <% :evidence -> %>
          <.evidence_tab
            session={@session}
            expanded_evidence_ids={@expanded_evidence_ids}
          />
      <% end %>
    </div>
    """
  end

  # ============================================================================
  # OVERVIEW TAB
  # ============================================================================

  defp overview_tab(assigns) do
    total_goals = length(assigns.session.goals)
    completed = Enum.count(assigns.session.goals, &(&1.status == :completed))
    failed = Enum.count(assigns.session.goals, &(&1.status == :failed))
    in_progress = Enum.count(assigns.session.goals, &(&1.status == :in_progress))
    pending = total_goals - completed - failed - in_progress

    support_rate =
      if assigns.session.hypotheses_tested > 0 do
        Float.round(assigns.session.hypotheses_supported / assigns.session.hypotheses_tested * 100, 1)
      else
        nil
      end

    assigns =
      assigns
      |> assign(:total_goals, total_goals)
      |> assign(:completed, completed)
      |> assign(:failed, failed)
      |> assign(:in_progress, in_progress)
      |> assign(:pending, pending)
      |> assign(:support_rate, support_rate)

    ~H"""
    <div class="space-y-6">
      <!-- Metric Cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-5 text-center">
          <div class="text-3xl font-bold">{@session.findings_count}</div>
          <div class="text-sm text-base-content/60 mt-1">Findings</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-5 text-center">
          <div class="text-3xl font-bold text-success">{@session.approved_count}</div>
          <div class="text-sm text-base-content/60 mt-1">Approved</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-5 text-center">
          <div class="text-3xl font-bold text-error">{@session.rejected_count}</div>
          <div class="text-sm text-base-content/60 mt-1">Rejected</div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-5 text-center">
          <div class="text-3xl font-bold">
            <%= if @support_rate, do: "#{@support_rate}%", else: "N/A" %>
          </div>
          <div class="text-sm text-base-content/60 mt-1">Support Rate</div>
        </div>
      </div>

      <!-- Goal Progress -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-5">
        <h3 class="font-semibold mb-3">Goal Progress</h3>
        <div class="flex gap-1 h-4 rounded-full overflow-hidden bg-base-300/30 mb-3">
          <%= if @total_goals > 0 do %>
            <%= if @completed > 0 do %>
              <div
                class="bg-success h-full transition-all"
                style={"width: #{@completed / @total_goals * 100}%"}
                title={"#{@completed} completed"}
              />
            <% end %>
            <%= if @failed > 0 do %>
              <div
                class="bg-error h-full transition-all"
                style={"width: #{@failed / @total_goals * 100}%"}
                title={"#{@failed} failed"}
              />
            <% end %>
            <%= if @in_progress > 0 do %>
              <div
                class="bg-warning h-full transition-all"
                style={"width: #{@in_progress / @total_goals * 100}%"}
                title={"#{@in_progress} in progress"}
              />
            <% end %>
            <%= if @pending > 0 do %>
              <div
                class="bg-base-300 h-full transition-all"
                style={"width: #{@pending / @total_goals * 100}%"}
                title={"#{@pending} pending"}
              />
            <% end %>
          <% end %>
        </div>
        <div class="flex flex-wrap gap-4 text-xs">
          <span class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-success"></span>
            {@completed} completed
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-error"></span>
            {@failed} failed
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-warning"></span>
            {@in_progress} in progress
          </span>
          <span class="flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-full bg-base-300"></span>
            {@pending} pending
          </span>
        </div>
      </div>

      <!-- Hypotheses Summary -->
      <%= if @session.hypotheses_tested > 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-5">
          <h3 class="font-semibold mb-3">Scientific Investigation Summary</h3>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div class="text-2xl font-bold">{length(@session.investigations)}</div>
              <div class="text-xs text-base-content/60">Investigations</div>
            </div>
            <div>
              <div class="text-2xl font-bold">{@session.hypotheses_tested}</div>
              <div class="text-xs text-base-content/60">Hypotheses Tested</div>
            </div>
            <div>
              <div class="text-2xl font-bold text-success">{@session.hypotheses_supported}</div>
              <div class="text-xs text-base-content/60">Supported</div>
            </div>
            <div>
              <div class="text-2xl font-bold text-error">{@session.hypotheses_falsified}</div>
              <div class="text-xs text-base-content/60">Falsified</div>
            </div>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  # ============================================================================
  # GOALS TAB
  # ============================================================================

  defp goals_tab(assigns) do
    ~H"""
    <div class="space-y-3">
      <%= if length(@session.goals) == 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-8 text-center">
          <.icon name="hero-flag" class="size-10 mx-auto mb-3 text-base-content/20" />
          <p class="text-base-content/50">No research goals defined</p>
        </div>
      <% else %>
        <%= for goal <- @session.goals do %>
          <.goal_card goal={goal} />
        <% end %>
      <% end %>
    </div>
    """
  end

  defp goal_card(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
      <div class="flex items-start justify-between gap-3">
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-2 mb-2">
            <span class={["badge badge-sm", goal_badge(@goal.status)]}>
              {@goal.status}
            </span>
            <span class="font-semibold truncate">{@goal.topic}</span>
            <%= if @goal.priority != :normal do %>
              <span class={["badge badge-sm badge-outline", priority_badge(@goal.priority)]}>
                {@goal.priority}
              </span>
            <% end %>
          </div>

          <!-- Questions -->
          <%= if length(@goal.questions) > 0 do %>
            <div class="space-y-1 mb-2">
              <%= for question <- @goal.questions do %>
                <div class="flex items-start gap-2 text-sm text-base-content/70">
                  <.icon name="hero-question-mark-circle" class="size-4 shrink-0 mt-0.5 text-info" />
                  <span>{safe_display(question)}</span>
                </div>
              <% end %>
            </div>
          <% end %>

          <!-- Constraints -->
          <%= if map_size(@goal.constraints) > 0 do %>
            <div class="flex flex-wrap gap-1.5 mb-2">
              <%= for {key, val} <- @goal.constraints do %>
                <span class="badge badge-sm badge-ghost">{key}: {safe_display(val)}</span>
              <% end %>
            </div>
          <% end %>

          <!-- Timestamps -->
          <%= if @goal.created_at do %>
            <div class="text-xs text-base-content/40">
              Created {Calendar.strftime(@goal.created_at, "%Y-%m-%d %H:%M:%S")}
            </div>
          <% end %>
        </div>

        <div class="shrink-0">
          <%= case @goal.status do %>
            <% :completed -> %>
              <.icon name="hero-check-circle" class="size-6 text-success" />
            <% :failed -> %>
              <.icon name="hero-x-circle" class="size-6 text-error" />
            <% :in_progress -> %>
              <span class="loading loading-spinner loading-sm text-warning"></span>
            <% _ -> %>
              <.icon name="hero-clock" class="size-6 text-base-content/30" />
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  # ============================================================================
  # INVESTIGATIONS TAB
  # ============================================================================

  defp investigations_tab(assigns) do
    ~H"""
    <div class="space-y-3">
      <%= if length(@session.investigations) == 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-8 text-center">
          <.icon name="hero-beaker" class="size-10 mx-auto mb-3 text-base-content/20" />
          <p class="text-base-content/50">No scientific investigations conducted</p>
          <p class="text-sm text-base-content/40 mt-1">
            Investigations are created when research goals generate testable hypotheses
          </p>
        </div>
      <% else %>
        <%= for investigation <- @session.investigations do %>
          <.investigation_card
            investigation={investigation}
            expanded={@expanded_investigation_id == investigation.id}
          />
        <% end %>
      <% end %>
    </div>
    """
  end

  defp investigation_card(assigns) do
    supported = Enum.count(assigns.investigation.hypotheses, &(&1.status == :supported))
    falsified = Enum.count(assigns.investigation.hypotheses, &(&1.status == :falsified))
    inconclusive = Enum.count(assigns.investigation.hypotheses, &(&1.status == :inconclusive))

    assigns =
      assigns
      |> assign(:supported, supported)
      |> assign(:falsified, falsified)
      |> assign(:inconclusive, inconclusive)

    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
      <!-- Header (clickable) -->
      <div
        class="p-4 cursor-pointer hover:bg-base-200/50 transition-colors"
        phx-click="toggle_investigation"
        phx-value-id={@investigation.id}
      >
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <.icon
              name={if @expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
              class="size-4 text-base-content/40"
            />
            <span class="font-semibold">{@investigation.topic}</span>
          </div>
          <div class="flex items-center gap-2">
            <span class={["badge badge-sm", investigation_badge(@investigation.status)]}>
              {@investigation.status}
            </span>
            <%= if @investigation.conclusion do %>
              <span class={["badge badge-sm", conclusion_badge(@investigation.conclusion)]}>
                {format_conclusion(@investigation.conclusion)}
              </span>
            <% end %>
          </div>
        </div>

        <!-- Summary stats -->
        <div class="flex flex-wrap gap-3 ml-6 text-xs text-base-content/60">
          <span>{length(@investigation.hypotheses)} hypotheses</span>
          <span>{length(@investigation.evidence)} evidence items</span>
          <%= if @supported > 0 do %>
            <span class="text-success">{@supported} supported</span>
          <% end %>
          <%= if @falsified > 0 do %>
            <span class="text-error">{@falsified} falsified</span>
          <% end %>
          <%= if @inconclusive > 0 do %>
            <span class="text-warning">{@inconclusive} inconclusive</span>
          <% end %>
          <%= if @investigation.concluded_at do %>
            <span class="text-base-content/40">
              Concluded {Calendar.strftime(@investigation.concluded_at, "%H:%M:%S")}
            </span>
          <% end %>
        </div>
      </div>

      <!-- Expanded Content -->
      <%= if @expanded do %>
        <div class="border-t border-base-300/50">
          <!-- Methodology -->
          <div class="p-4 bg-base-200/20 border-b border-base-300/30">
            <h4 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">
              Methodology
            </h4>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <div>
                <span class="text-base-content/50">Independent Variable:</span>
                <span class="ml-1 font-medium">{@investigation.independent_variable}</span>
              </div>
              <div>
                <span class="text-base-content/50">Dependent Variable:</span>
                <span class="ml-1 font-medium">{@investigation.dependent_variable}</span>
              </div>
              <div>
                <span class="text-base-content/50">Constants:</span>
                <span class="ml-1 font-medium">{Enum.join(@investigation.constants, ", ")}</span>
              </div>
            </div>
            <%= if @investigation.methodology_notes do %>
              <div class="mt-2 text-sm text-base-content/60 italic">
                {@investigation.methodology_notes}
              </div>
            <% end %>
          </div>

          <!-- Hypotheses -->
          <div class="p-4">
            <h4 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-3">
              Hypotheses ({length(@investigation.hypotheses)})
            </h4>
            <div class="space-y-3">
              <%= for hypothesis <- @investigation.hypotheses do %>
                <.hypothesis_card hypothesis={hypothesis} />
              <% end %>
            </div>
          </div>

          <!-- Control Evidence -->
          <%= if length(@investigation.control_evidence) > 0 do %>
            <div class="p-4 border-t border-base-300/30">
              <h4 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-3">
                Control Evidence ({length(@investigation.control_evidence)})
              </h4>
              <div class="space-y-2">
                <%= for finding <- @investigation.control_evidence do %>
                  <.finding_inline finding={finding} />
                <% end %>
              </div>
            </div>
          <% end %>

          <!-- Evidence -->
          <div class="p-4 border-t border-base-300/30">
            <h4 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-3">
              Gathered Evidence ({length(@investigation.evidence)})
            </h4>
            <%= if length(@investigation.evidence) == 0 do %>
              <p class="text-sm text-base-content/40">No evidence gathered</p>
            <% else %>
              <div class="space-y-2">
                <%= for finding <- @investigation.evidence do %>
                  <.finding_inline finding={finding} />
                <% end %>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  defp hypothesis_card(assigns) do
    ~H"""
    <div class="bg-base-200/30 rounded-lg p-3 border border-base-300/20">
      <div class="flex items-start justify-between gap-2 mb-2">
        <div class="flex items-center gap-2 min-w-0">
          <span class={["badge badge-xs shrink-0", hypothesis_badge(@hypothesis.status)]}>
            {@hypothesis.status}
          </span>
          <span class="font-medium text-sm">{@hypothesis.claim}</span>
        </div>
        <%= if @hypothesis.confidence > 0 do %>
          <div class="shrink-0 text-right">
            <div class="text-sm font-bold">{Float.round(@hypothesis.confidence * 100, 1)}%</div>
            <div class="text-[10px] text-base-content/40">{@hypothesis.confidence_level}</div>
          </div>
        <% end %>
      </div>

      <!-- Prediction -->
      <%= if @hypothesis.prediction do %>
        <div class="text-xs text-base-content/60 mb-2 pl-2 border-l-2 border-info/30 italic">
          {@hypothesis.prediction}
        </div>
      <% end %>

      <!-- Derived From -->
      <%= if @hypothesis.derived_from do %>
        <div class="text-xs text-base-content/50 mb-2">
          Derived from: <span class="text-base-content/70">{@hypothesis.derived_from}</span>
        </div>
      <% end %>

      <!-- Evidence Counts -->
      <div class="flex flex-wrap gap-3 text-xs text-base-content/50">
        <span class="flex items-center gap-1">
          <.icon name="hero-check" class="size-3 text-success" />
          {length(@hypothesis.supporting_evidence)} supporting
        </span>
        <span class="flex items-center gap-1">
          <.icon name="hero-x-mark" class="size-3 text-error" />
          {length(@hypothesis.contradicting_evidence)} contradicting
        </span>
        <%= if @hypothesis.source_count > 0 do %>
          <span>{@hypothesis.source_count} sources</span>
        <% end %>
        <%= if @hypothesis.replication_count > 0 do %>
          <span>{@hypothesis.replication_count} replications</span>
        <% end %>
        <%= if @hypothesis.tested_at do %>
          <span class="text-base-content/40">
            Tested {Calendar.strftime(@hypothesis.tested_at, "%H:%M:%S")}
          </span>
        <% end %>
      </div>

      <!-- Confidence bar -->
      <%= if @hypothesis.confidence > 0 do %>
        <div class="mt-2 h-1.5 rounded-full bg-base-300/50 overflow-hidden">
          <div
            class={[
              "h-full rounded-full transition-all",
              confidence_color(@hypothesis.confidence)
            ]}
            style={"width: #{@hypothesis.confidence * 100}%"}
          />
        </div>
      <% end %>
    </div>
    """
  end

  defp finding_inline(assigns) do
    ~H"""
    <div class="flex items-start gap-2 text-sm bg-base-200/20 rounded-lg p-2.5 border border-base-300/10">
      <.icon name="hero-document-text" class="size-4 shrink-0 mt-0.5 text-base-content/40" />
      <div class="min-w-0 flex-1">
        <div class="text-base-content/80">{@finding.claim}</div>
        <div class="flex flex-wrap gap-2 mt-1 text-xs text-base-content/50">
          <%= if @finding.entity do %>
            <span class="badge badge-xs badge-ghost">{@finding.entity}</span>
          <% end %>
          <%= if @finding.entity_type do %>
            <span class="badge badge-xs badge-outline">{@finding.entity_type}</span>
          <% end %>
          <%= if @finding.source do %>
            <span class="truncate max-w-[200px]" title={source_url(@finding.source)}>
              {source_domain(@finding.source)}
            </span>
          <% end %>
          <span>Conf: {Float.round(@finding.confidence * 100, 1)}%</span>
        </div>
      </div>
    </div>
    """
  end

  # ============================================================================
  # EVIDENCE TAB
  # ============================================================================

  defp evidence_tab(assigns) do
    all_evidence =
      assigns.session.investigations
      |> Enum.flat_map(& &1.evidence)

    assigns = assign(assigns, :all_evidence, all_evidence)

    ~H"""
    <div class="space-y-3">
      <%= if length(@all_evidence) == 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-8 text-center">
          <.icon name="hero-document-magnifying-glass" class="size-10 mx-auto mb-3 text-base-content/20" />
          <p class="text-base-content/50">No evidence collected</p>
          <p class="text-sm text-base-content/40 mt-1">
            Evidence is gathered during scientific investigations
          </p>
        </div>
      <% else %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4 mb-3">
          <span class="text-sm text-base-content/60">
            {length(@all_evidence)} evidence items across {length(@session.investigations)} investigation(s)
          </span>
        </div>

        <%= for finding <- @all_evidence do %>
          <.evidence_card
            finding={finding}
            expanded={MapSet.member?(@expanded_evidence_ids, finding.id)}
          />
        <% end %>
      <% end %>
    </div>
    """
  end

  defp evidence_card(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
      <!-- Header (clickable) -->
      <div
        class="p-4 cursor-pointer hover:bg-base-200/30 transition-colors"
        phx-click="toggle_evidence"
        phx-value-id={@finding.id}
      >
        <div class="flex items-start justify-between gap-3">
          <div class="flex items-start gap-2 min-w-0 flex-1">
            <.icon
              name={if @expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
              class="size-4 shrink-0 mt-0.5 text-base-content/40"
            />
            <div class="min-w-0">
              <div class="font-medium text-sm">{@finding.claim}</div>
              <div class="flex flex-wrap gap-2 mt-1 text-xs text-base-content/50">
                <%= if @finding.entity do %>
                  <span class="badge badge-xs badge-ghost">{@finding.entity}</span>
                <% end %>
                <%= if @finding.entity_type do %>
                  <span class="badge badge-xs badge-outline">{@finding.entity_type}</span>
                <% end %>
                <%= if @finding.source do %>
                  <span>{source_domain(@finding.source)}</span>
                <% end %>
              </div>
            </div>
          </div>
          <div class="shrink-0">
            <div class={[
              "badge badge-sm",
              confidence_badge(@finding.confidence)
            ]}>
              {Float.round(@finding.confidence * 100, 1)}%
            </div>
          </div>
        </div>
      </div>

      <!-- Expanded Detail -->
      <%= if @expanded do %>
        <div class="border-t border-base-300/30 p-4 space-y-4">
          <!-- Source Info -->
          <%= if @finding.source do %>
            <div>
              <h5 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">Source</h5>
              <div class="bg-base-200/30 rounded-lg p-3 space-y-1.5 text-sm">
                <div class="flex items-center gap-2">
                  <span class="text-base-content/50 w-24 shrink-0">Domain:</span>
                  <span class="font-medium">{source_domain(@finding.source)}</span>
                </div>
                <div class="flex items-start gap-2">
                  <span class="text-base-content/50 w-24 shrink-0">URL:</span>
                  <span class="text-info break-all text-xs">{source_url(@finding.source)}</span>
                </div>
                <%= if source_title(@finding.source) do %>
                  <div class="flex items-start gap-2">
                    <span class="text-base-content/50 w-24 shrink-0">Title:</span>
                    <span>{source_title(@finding.source)}</span>
                  </div>
                <% end %>
                <div class="flex items-center gap-2">
                  <span class="text-base-content/50 w-24 shrink-0">Reliability:</span>
                  <div class="flex items-center gap-2">
                    <div class="w-20 h-2 rounded-full bg-base-300/50 overflow-hidden">
                      <div
                        class={["h-full rounded-full", confidence_color(source_reliability(@finding.source))]}
                        style={"width: #{source_reliability(@finding.source) * 100}%"}
                      />
                    </div>
                    <span class="text-xs">{Float.round(source_reliability(@finding.source) * 100, 1)}%</span>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  <span class="text-base-content/50 w-24 shrink-0">Bias:</span>
                  <span class={["badge badge-xs", bias_badge(source_bias(@finding.source))]}>
                    {source_bias(@finding.source)}
                  </span>
                </div>
                <div class="flex items-center gap-2">
                  <span class="text-base-content/50 w-24 shrink-0">Trust Tier:</span>
                  <span class={["badge badge-xs", trust_badge(source_trust(@finding.source))]}>
                    {source_trust(@finding.source)}
                  </span>
                </div>
              </div>
            </div>
          <% end %>

          <!-- Raw Context -->
          <%= if @finding.raw_context && @finding.raw_context != "" do %>
            <div>
              <h5 class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">Raw Context</h5>
              <div class="bg-base-200/30 rounded-lg p-3 text-sm text-base-content/70 whitespace-pre-wrap font-mono text-xs max-h-48 overflow-y-auto">
                {@finding.raw_context}
              </div>
            </div>
          <% end %>

          <!-- Metadata -->
          <div class="flex flex-wrap gap-4 text-xs text-base-content/50">
            <span>ID: <span class="font-mono">{@finding.id}</span></span>
            <%= if @finding.extracted_at do %>
              <span>Extracted: {Calendar.strftime(@finding.extracted_at, "%Y-%m-%d %H:%M:%S")}</span>
            <% end %>
            <%= if @finding.corroboration_group do %>
              <span>Corroboration Group: <span class="font-mono">{@finding.corroboration_group}</span></span>
            <% end %>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  # ============================================================================
  # HELPERS
  # ============================================================================

  defp session_badge(:active), do: "badge-warning"
  defp session_badge(:completed), do: "badge-success"
  defp session_badge(:cancelled), do: "badge-error"
  defp session_badge(_), do: "badge-ghost"

  defp goal_badge(:completed), do: "badge-success"
  defp goal_badge(:failed), do: "badge-error"
  defp goal_badge(:in_progress), do: "badge-warning"
  defp goal_badge(:pending), do: "badge-ghost"
  defp goal_badge(_), do: "badge-ghost"

  defp priority_badge(:high), do: "badge-error"
  defp priority_badge(:low), do: "badge-ghost"
  defp priority_badge(_), do: ""

  defp investigation_badge(:concluded), do: "badge-success"
  defp investigation_badge(:evaluating), do: "badge-warning"
  defp investigation_badge(:gathering_evidence), do: "badge-info"
  defp investigation_badge(:planning), do: "badge-ghost"
  defp investigation_badge(_), do: "badge-ghost"

  defp conclusion_badge(:hypotheses_supported), do: "badge-success"
  defp conclusion_badge(:hypotheses_falsified), do: "badge-error"
  defp conclusion_badge(:inconclusive), do: "badge-warning"
  defp conclusion_badge(:mixed), do: "badge-warning"
  defp conclusion_badge(_), do: "badge-ghost"

  defp format_conclusion(:hypotheses_supported), do: "supported"
  defp format_conclusion(:hypotheses_falsified), do: "falsified"
  defp format_conclusion(:inconclusive), do: "inconclusive"
  defp format_conclusion(:mixed), do: "mixed"
  defp format_conclusion(other), do: to_string(other)

  defp hypothesis_badge(:supported), do: "badge-success"
  defp hypothesis_badge(:falsified), do: "badge-error"
  defp hypothesis_badge(:inconclusive), do: "badge-warning"
  defp hypothesis_badge(:testing), do: "badge-info"
  defp hypothesis_badge(:untested), do: "badge-ghost"
  defp hypothesis_badge(_), do: "badge-ghost"

  defp confidence_color(c) when c >= 0.7, do: "bg-success"
  defp confidence_color(c) when c >= 0.4, do: "bg-warning"
  defp confidence_color(_), do: "bg-error"

  defp confidence_badge(c) when c >= 0.7, do: "badge-success"
  defp confidence_badge(c) when c >= 0.4, do: "badge-warning"
  defp confidence_badge(_), do: "badge-error"

  defp bias_badge(:center), do: "badge-success"
  defp bias_badge(:center_left), do: "badge-info"
  defp bias_badge(:center_right), do: "badge-info"
  defp bias_badge(:left), do: "badge-warning"
  defp bias_badge(:right), do: "badge-warning"
  defp bias_badge(:unknown), do: "badge-ghost"
  defp bias_badge(_), do: "badge-ghost"

  defp trust_badge(:verified), do: "badge-success"
  defp trust_badge(:neutral), do: "badge-ghost"
  defp trust_badge(:untrusted), do: "badge-warning"
  defp trust_badge(:blocked), do: "badge-error"
  defp trust_badge(_), do: "badge-ghost"

  defp source_domain(%{domain: domain}) when is_binary(domain), do: domain
  defp source_domain(_), do: "unknown"

  defp source_url(%{url: url}) when is_binary(url), do: url
  defp source_url(_), do: ""

  defp source_title(%{title: title}) when is_binary(title), do: title
  defp source_title(_), do: nil

  defp source_reliability(%{reliability_score: score}) when is_number(score), do: score
  defp source_reliability(_), do: 0.5

  defp source_bias(%{bias_rating: bias}), do: bias
  defp source_bias(_), do: :unknown

  defp source_trust(%{trust_tier: tier}), do: tier
  defp source_trust(_), do: :neutral

  defp safe_display(val) when is_binary(val), do: val
  defp safe_display(val) when is_atom(val), do: Atom.to_string(val)
  defp safe_display(val) when is_number(val), do: to_string(val)
  defp safe_display(%{text: text}) when is_binary(text), do: text
  defp safe_display(%{claim: claim}) when is_binary(claim), do: claim
  defp safe_display(val) when is_map(val), do: inspect(val, limit: 5, pretty: false)
  defp safe_display(val) when is_list(val), do: Enum.map_join(val, ", ", &safe_display/1)
  defp safe_display(val), do: inspect(val)
end
