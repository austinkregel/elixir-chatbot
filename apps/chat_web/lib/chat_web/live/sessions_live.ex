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
          <div class="flex items-center gap-space-sm text-body-dense text-ink-muted mb-space-xs">
            <.link navigate={~p"/sessions"} class="text-accent hover:underline">
              Sessions
            </.link>
            <.icon name="hero-chevron-right" class="size-3" />
            <span class="text-ink">{@session.topic || "Session"}</span>
          </div>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-title text-ink">{@session.topic || "Session Detail"}</h1>
              <p class="text-ref text-ink-muted">{@session.id}</p>
            </div>
            <.badge variant={session_variant(@session.status)}>
              {@session.status}
            </.badge>
          </div>
        <% else %>
          <div class="flex items-center justify-between">
            <div>
              <h1 class="text-title text-ink">Learning Sessions</h1>
              <p class="text-body text-ink-muted">
                Inspect research sessions, goals, investigations, and evidence
              </p>
            </div>
          </div>
        <% end %>
      </:page_header>

      <div class="p-space-lg sm:p-space-xl">
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
    <div class="space-y-space-xl">
      <!-- Stats Bar -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-space-lg">
        <.stat_kpi label="Total Sessions" value={to_string(@lc_stats[:total_sessions] || 0)} />
        <.stat_kpi label="Active" value={to_string(@lc_stats[:active_sessions] || 0)} />
        <.stat_kpi label="Active Agents" value={to_string(@lc_stats[:active_agents] || 0)} />
        <.stat_kpi label="Total Findings" value={to_string(@lc_stats[:total_findings] || 0)} />
      </div>

      <!-- Status Filter Tabs -->
      <.tabs>
        <%= for {label, value} <- [{"All", "all"}, {"Active", "active"}, {"Completed", "completed"}, {"Cancelled", "cancelled"}] do %>
          <.tab
            active={to_string(@status_filter) == value}
            phx-click="filter_status"
            phx-value-status={value}
          >
            {label}
          </.tab>
        <% end %>
      </.tabs>

      <!-- Session Cards -->
      <%= if length(@sessions) == 0 do %>
        <.card class="p-space-3xl text-center">
          <.icon name="hero-beaker" class="size-12 mx-auto mb-space-lg text-ink-muted" />
          <p class="text-body text-ink-muted">No sessions found</p>
          <p class="text-body-dense text-ink-muted mt-space-xs">
            Start a training session from the Settings page
          </p>
        </.card>
      <% else %>
        <div class="space-y-space-md">
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
      class="block rounded-md border border-border bg-surface p-space-lg hover:border-border-strong transition-colors group"
    >
      <div class="flex items-center justify-between mb-space-md">
        <div class="flex items-center gap-space-md min-w-0">
          <div class="min-w-0">
            <div class="text-subheading text-ink group-hover:text-accent transition-colors truncate">
              {@session.topic || "Untitled Session"}
            </div>
            <div class="text-ref text-ink-muted">{@session.id}</div>
          </div>
        </div>
        <div class="flex items-center gap-space-sm shrink-0">
          <.badge variant={session_variant(@session.status)}>
            {@session.status}
          </.badge>
          <.icon name="hero-chevron-right" class="size-4 text-ink-muted group-hover:text-accent transition-colors" />
        </div>
      </div>

      <!-- Progress & Stats -->
      <div class="flex flex-wrap gap-x-space-lg gap-y-space-xs text-caption text-ink-muted">
        <span class="flex items-center gap-space-xs">
          <.icon name="hero-flag" class="size-3" />
          {@completed_goals}/{@total_goals} goals
        </span>
        <%= if @failed_goals > 0 do %>
          <span class="flex items-center gap-space-xs text-red">
            <.icon name="hero-exclamation-triangle" class="size-3" />
            {@failed_goals} failed
          </span>
        <% end %>
        <%= if @session.findings_count > 0 do %>
          <span class="flex items-center gap-space-xs">
            <.icon name="hero-document-magnifying-glass" class="size-3" />
            {@session.findings_count} findings
          </span>
        <% end %>
        <%= if @session.approved_count > 0 do %>
          <span class="flex items-center gap-space-xs">
            <.icon name="hero-check-circle" class="size-3" />
            {@session.approved_count} approved
          </span>
        <% end %>
        <%= if @session.rejected_count > 0 do %>
          <span class="flex items-center gap-space-xs">
            <.icon name="hero-x-circle" class="size-3" />
            {@session.rejected_count} rejected
          </span>
        <% end %>
        <%= if @session.hypotheses_tested > 0 do %>
          <span class="flex items-center gap-space-xs">
            <.icon name="hero-beaker" class="size-3" />
            {@session.hypotheses_tested} hypotheses
          </span>
        <% end %>
        <%= if @session.started_at do %>
          <span class="text-ink-muted">
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
    <div class="space-y-space-xl">
      <!-- Timestamps -->
      <div class="flex flex-wrap gap-space-lg text-body text-ink-muted">
        <%= if @session.started_at do %>
          <span>
            Started: <span class="text-value-strong text-ink">{Calendar.strftime(@session.started_at, "%Y-%m-%d %H:%M:%S")}</span>
          </span>
        <% end %>
        <%= if @session.completed_at do %>
          <span>
            Completed: <span class="text-value-strong text-ink">{Calendar.strftime(@session.completed_at, "%Y-%m-%d %H:%M:%S")}</span>
          </span>
        <% end %>
      </div>

      <!-- Tab Navigation -->
      <.tabs>
        <%= for {label, value, icon} <- [
          {"Overview", "overview", "hero-chart-bar-square"},
          {"Goals", "goals", "hero-flag"},
          {"Investigations", "investigations", "hero-beaker"},
          {"Evidence", "evidence", "hero-document-magnifying-glass"}
        ] do %>
          <.tab
            active={to_string(@detail_tab) == value}
            phx-click="change_tab"
            phx-value-tab={value}
          >
            <span class="inline-flex items-center gap-space-xs">
              <.icon name={icon} class="size-4" />
              {label}
            </span>
          </.tab>
        <% end %>
      </.tabs>

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
    <div class="space-y-space-xl">
      <!-- Metric Cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-space-lg">
        <.stat_kpi label="Findings" value={to_string(@session.findings_count)} />
        <.stat_kpi label="Approved" value={to_string(@session.approved_count)} />
        <.stat_kpi label="Rejected" value={to_string(@session.rejected_count)} />
        <.stat_kpi label="Support Rate" value={if @support_rate, do: "#{@support_rate}%", else: "N/A"} />
      </div>

      <!-- Goal Progress -->
      <.card>
        <.card_body>
          <h3 class="text-heading text-ink mb-space-md">Goal Progress</h3>
          <div class="flex gap-space-xs h-space-lg rounded-sm overflow-hidden bg-surface-sunk mb-space-md">
            <%= if @total_goals > 0 do %>
              <%= if @completed > 0 do %>
                <div
                  class="bg-ink h-full transition-all"
                  style={"width: #{@completed / @total_goals * 100}%"}
                  title={"#{@completed} completed"}
                />
              <% end %>
              <%= if @failed > 0 do %>
                <div
                  class="bg-red h-full transition-all"
                  style={"width: #{@failed / @total_goals * 100}%"}
                  title={"#{@failed} failed"}
                />
              <% end %>
              <%= if @in_progress > 0 do %>
                <div
                  class="bg-ochre h-full transition-all"
                  style={"width: #{@in_progress / @total_goals * 100}%"}
                  title={"#{@in_progress} in progress"}
                />
              <% end %>
              <%= if @pending > 0 do %>
                <div
                  class="bg-border h-full transition-all"
                  style={"width: #{@pending / @total_goals * 100}%"}
                  title={"#{@pending} pending"}
                />
              <% end %>
            <% end %>
          </div>
          <div class="flex flex-wrap gap-space-lg text-caption text-ink">
            <span class="flex items-center gap-space-xs">
              <span class="size-3 rounded-pip bg-ink"></span>
              {@completed} completed
            </span>
            <span class="flex items-center gap-space-xs">
              <span class="size-3 rounded-pip bg-red"></span>
              {@failed} failed
            </span>
            <span class="flex items-center gap-space-xs">
              <span class="size-3 rounded-pip bg-ochre"></span>
              {@in_progress} in progress
            </span>
            <span class="flex items-center gap-space-xs">
              <span class="size-3 rounded-pip bg-border"></span>
              {@pending} pending
            </span>
          </div>
        </.card_body>
      </.card>

      <!-- Hypotheses Summary -->
      <%= if @session.hypotheses_tested > 0 do %>
        <.card>
          <.card_body>
            <h3 class="text-heading text-ink mb-space-md">Scientific Investigation Summary</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-space-lg">
              <div>
                <div class="text-title tabular-nums text-ink">{length(@session.investigations)}</div>
                <div class="text-label text-ink-muted">Investigations</div>
              </div>
              <div>
                <div class="text-title tabular-nums text-ink">{@session.hypotheses_tested}</div>
                <div class="text-label text-ink-muted">Hypotheses Tested</div>
              </div>
              <div>
                <div class="text-title tabular-nums text-ink">{@session.hypotheses_supported}</div>
                <div class="text-label text-ink-muted">Supported</div>
              </div>
              <div>
                <div class="text-title tabular-nums text-ink">{@session.hypotheses_falsified}</div>
                <div class="text-label text-ink-muted">Falsified</div>
              </div>
            </div>
          </.card_body>
        </.card>
      <% end %>
    </div>
    """
  end

  # ============================================================================
  # GOALS TAB
  # ============================================================================

  defp goals_tab(assigns) do
    ~H"""
    <div class="space-y-space-md">
      <%= if length(@session.goals) == 0 do %>
        <.card class="p-space-2xl text-center">
          <.icon name="hero-flag" class="size-10 mx-auto mb-space-md text-ink-muted" />
          <p class="text-body text-ink-muted">No research goals defined</p>
        </.card>
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
    <.card class="p-space-lg">
      <div class="flex items-start justify-between gap-space-md">
        <div class="flex-1 min-w-0">
          <div class="flex items-center gap-space-sm mb-space-sm">
            <.badge variant={goal_variant(@goal.status)}>
              {@goal.status}
            </.badge>
            <span class="text-subheading text-ink truncate">{@goal.topic}</span>
            <%= if @goal.priority != :normal do %>
              <.badge variant={priority_variant(@goal.priority)} class="border border-border-strong">
                {@goal.priority} priority
              </.badge>
            <% end %>
          </div>

          <!-- Questions -->
          <%= if length(@goal.questions) > 0 do %>
            <div class="space-y-space-xs mb-space-sm">
              <%= for question <- @goal.questions do %>
                <div class="flex items-start gap-space-sm text-body text-ink">
                  <.icon name="hero-question-mark-circle" class="size-4 shrink-0 mt-space-2xs text-ink-muted" />
                  <span>{safe_display(question)}</span>
                </div>
              <% end %>
            </div>
          <% end %>

          <!-- Constraints -->
          <%= if map_size(@goal.constraints) > 0 do %>
            <div class="flex flex-wrap gap-space-xs mb-space-sm">
              <%= for {key, val} <- @goal.constraints do %>
                <.badge>{key}: {safe_display(val)}</.badge>
              <% end %>
            </div>
          <% end %>

          <!-- Timestamps -->
          <%= if @goal.created_at do %>
            <div class="text-caption text-ink-muted">
              Created {Calendar.strftime(@goal.created_at, "%Y-%m-%d %H:%M:%S")}
            </div>
          <% end %>
        </div>

        <div class="shrink-0">
          <%= case @goal.status do %>
            <% :completed -> %>
              <.icon name="hero-check-circle" class="size-6 text-ink" />
            <% :failed -> %>
              <.icon name="hero-x-circle" class="size-6 text-red" />
            <% :in_progress -> %>
              <.icon name="hero-arrow-path" class="size-6 motion-safe:animate-spin text-progress-fill" />
            <% :pending -> %>
              <.icon name="hero-clock" class="size-6 text-ink-muted" />
          <% end %>
        </div>
      </div>
    </.card>
    """
  end

  # ============================================================================
  # INVESTIGATIONS TAB
  # ============================================================================

  defp investigations_tab(assigns) do
    ~H"""
    <div class="space-y-space-md">
      <%= if length(@session.investigations) == 0 do %>
        <.card class="p-space-2xl text-center">
          <.icon name="hero-beaker" class="size-10 mx-auto mb-space-md text-ink-muted" />
          <p class="text-body text-ink-muted">No scientific investigations conducted</p>
          <p class="text-body-dense text-ink-muted mt-space-xs">
            Investigations are created when research goals generate testable hypotheses
          </p>
        </.card>
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
    <.card class="overflow-hidden">
      <!-- Header (clickable) -->
      <div
        class="p-space-lg cursor-pointer hover:bg-surface-sunk transition-colors"
        phx-click="toggle_investigation"
        phx-value-id={@investigation.id}
      >
        <div class="flex items-center justify-between mb-space-sm">
          <div class="flex items-center gap-space-sm">
            <.icon
              name={if @expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
              class="size-4 text-ink-muted"
            />
            <span class="text-subheading text-ink">{@investigation.topic}</span>
          </div>
          <div class="flex items-center gap-space-sm">
            <.badge variant={investigation_variant(@investigation.status)}>
              {@investigation.status}
            </.badge>
            <%= if @investigation.conclusion do %>
              <.badge variant={conclusion_variant(@investigation.conclusion)}>
                {format_conclusion(@investigation.conclusion)}
              </.badge>
            <% end %>
          </div>
        </div>

        <!-- Summary stats -->
        <div class="flex flex-wrap gap-space-md ml-space-xl text-caption text-ink-muted">
          <span>{length(@investigation.hypotheses)} hypotheses</span>
          <span>{length(@investigation.evidence)} evidence items</span>
          <%= if @supported > 0 do %>
            <span>{@supported} supported</span>
          <% end %>
          <%= if @falsified > 0 do %>
            <span>{@falsified} falsified</span>
          <% end %>
          <%= if @inconclusive > 0 do %>
            <span>{@inconclusive} inconclusive</span>
          <% end %>
          <%= if @investigation.concluded_at do %>
            <span class="text-ink-muted">
              Concluded {Calendar.strftime(@investigation.concluded_at, "%H:%M:%S")}
            </span>
          <% end %>
        </div>
      </div>

      <!-- Expanded Content -->
      <%= if @expanded do %>
        <div class="border-t border-border">
          <!-- Methodology -->
          <div class="p-space-lg bg-surface-sunk border-b border-border">
            <h4 class="text-label text-ink-muted mb-space-sm">
              Methodology
            </h4>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-space-md text-body">
              <div>
                <span class="text-ink-muted">Independent Variable:</span>
                <span class="ml-space-xs font-semibold text-ink">{@investigation.independent_variable}</span>
              </div>
              <div>
                <span class="text-ink-muted">Dependent Variable:</span>
                <span class="ml-space-xs font-semibold text-ink">{@investigation.dependent_variable}</span>
              </div>
              <div>
                <span class="text-ink-muted">Constants:</span>
                <span class="ml-space-xs font-semibold text-ink">{Enum.join(@investigation.constants, ", ")}</span>
              </div>
            </div>
            <%= if @investigation.methodology_notes do %>
              <div class="mt-space-sm text-body text-ink-muted italic">
                {@investigation.methodology_notes}
              </div>
            <% end %>
          </div>

          <!-- Hypotheses -->
          <div class="p-space-lg">
            <h4 class="text-label text-ink-muted mb-space-md">
              Hypotheses ({length(@investigation.hypotheses)})
            </h4>
            <div class="space-y-space-md">
              <%= for hypothesis <- @investigation.hypotheses do %>
                <.hypothesis_card hypothesis={hypothesis} />
              <% end %>
            </div>
          </div>

          <!-- Control Evidence -->
          <%= if length(@investigation.control_evidence) > 0 do %>
            <div class="p-space-lg border-t border-border">
              <h4 class="text-label text-ink-muted mb-space-md">
                Control Evidence ({length(@investigation.control_evidence)})
              </h4>
              <div class="space-y-space-sm">
                <%= for finding <- @investigation.control_evidence do %>
                  <.finding_inline finding={finding} />
                <% end %>
              </div>
            </div>
          <% end %>

          <!-- Evidence -->
          <div class="p-space-lg border-t border-border">
            <h4 class="text-label text-ink-muted mb-space-md">
              Gathered Evidence ({length(@investigation.evidence)})
            </h4>
            <%= if length(@investigation.evidence) == 0 do %>
              <p class="text-body text-ink-muted">No evidence gathered</p>
            <% else %>
              <div class="space-y-space-sm">
                <%= for finding <- @investigation.evidence do %>
                  <.finding_inline finding={finding} />
                <% end %>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>
    </.card>
    """
  end

  defp hypothesis_card(assigns) do
    ~H"""
    <div class="bg-surface-sunk rounded-md p-space-md border border-border">
      <div class="flex items-start justify-between gap-space-sm mb-space-sm">
        <div class="flex items-center gap-space-sm min-w-0">
          <.badge variant={hypothesis_variant(@hypothesis.status)} size={:xs} class="shrink-0">
            {@hypothesis.status}
          </.badge>
          <span class="text-subheading text-ink">{@hypothesis.claim}</span>
        </div>
        <%= if @hypothesis.confidence > 0 do %>
          <div class="shrink-0 text-right">
            <div class="text-value-strong text-ink">{Float.round(@hypothesis.confidence * 100, 1)}%</div>
            <div class="text-caption text-ink-muted">{@hypothesis.confidence_level}</div>
          </div>
        <% end %>
      </div>

      <!-- Prediction -->
      <%= if @hypothesis.prediction do %>
        <div class="text-caption text-ink-muted mb-space-sm pl-space-sm border-l-2 border-border-strong italic">
          {@hypothesis.prediction}
        </div>
      <% end %>

      <!-- Derived From -->
      <%= if @hypothesis.derived_from do %>
        <div class="text-caption text-ink-muted mb-space-sm">
          Derived from: <span class="text-ink">{@hypothesis.derived_from}</span>
        </div>
      <% end %>

      <!-- Evidence Counts -->
      <div class="flex flex-wrap gap-space-md text-caption text-ink-muted">
        <span class="flex items-center gap-space-xs">
          <.icon name="hero-check" class="size-3 text-ink" />
          {length(@hypothesis.supporting_evidence)} supporting
        </span>
        <span class="flex items-center gap-space-xs">
          <.icon name="hero-x-mark" class="size-3 text-ink" />
          {length(@hypothesis.contradicting_evidence)} contradicting
        </span>
        <%= if @hypothesis.source_count > 0 do %>
          <span>{@hypothesis.source_count} sources</span>
        <% end %>
        <%= if @hypothesis.replication_count > 0 do %>
          <span>{@hypothesis.replication_count} replications</span>
        <% end %>
        <%= if @hypothesis.tested_at do %>
          <span class="text-ink-muted">
            Tested {Calendar.strftime(@hypothesis.tested_at, "%H:%M:%S")}
          </span>
        <% end %>
      </div>

      <!-- Confidence bar -->
      <%= if @hypothesis.confidence > 0 do %>
        <div class="mt-space-sm h-space-xs rounded-sm bg-surface overflow-hidden">
          <div
            class="h-full rounded-sm bg-ink-muted transition-all"
            style={"width: #{@hypothesis.confidence * 100}%"}
          />
        </div>
      <% end %>
    </div>
    """
  end

  defp finding_inline(assigns) do
    ~H"""
    <div class="flex items-start gap-space-sm text-body bg-surface-sunk rounded-md p-space-sm border border-border">
      <.icon name="hero-document-text" class="size-4 shrink-0 mt-space-2xs text-ink-muted" />
      <div class="min-w-0 flex-1">
        <div class="text-ink">{@finding.claim}</div>
        <div class="flex flex-wrap gap-space-sm mt-space-xs text-caption text-ink-muted">
          <%= if @finding.entity do %>
            <.badge size={:xs}>{@finding.entity}</.badge>
          <% end %>
          <%= if @finding.entity_type do %>
            <.badge size={:xs} class="border border-border-strong">{@finding.entity_type}</.badge>
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
    <div class="space-y-space-md">
      <%= if length(@all_evidence) == 0 do %>
        <.card class="p-space-2xl text-center">
          <.icon name="hero-document-magnifying-glass" class="size-10 mx-auto mb-space-md text-ink-muted" />
          <p class="text-body text-ink-muted">No evidence collected</p>
          <p class="text-body-dense text-ink-muted mt-space-xs">
            Evidence is gathered during scientific investigations
          </p>
        </.card>
      <% else %>
        <.card class="p-space-lg mb-space-md">
          <span class="text-body text-ink-muted">
            {length(@all_evidence)} evidence items across {length(@session.investigations)} investigation(s)
          </span>
        </.card>

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
    <.card class="overflow-hidden">
      <!-- Header (clickable) -->
      <div
        class="p-space-lg cursor-pointer hover:bg-surface-sunk transition-colors"
        phx-click="toggle_evidence"
        phx-value-id={@finding.id}
      >
        <div class="flex items-start justify-between gap-space-md">
          <div class="flex items-start gap-space-sm min-w-0 flex-1">
            <.icon
              name={if @expanded, do: "hero-chevron-down", else: "hero-chevron-right"}
              class="size-4 shrink-0 mt-space-2xs text-ink-muted"
            />
            <div class="min-w-0">
              <div class="text-subheading text-ink">{@finding.claim}</div>
              <div class="flex flex-wrap gap-space-sm mt-space-xs text-caption text-ink-muted">
                <%= if @finding.entity do %>
                  <.badge size={:xs}>{@finding.entity}</.badge>
                <% end %>
                <%= if @finding.entity_type do %>
                  <.badge size={:xs} class="border border-border-strong">{@finding.entity_type}</.badge>
                <% end %>
                <%= if @finding.source do %>
                  <span>{source_domain(@finding.source)}</span>
                <% end %>
              </div>
            </div>
          </div>
          <div class="shrink-0">
            <.badge class="font-mono">
              {Float.round(@finding.confidence * 100, 1)}%
            </.badge>
          </div>
        </div>
      </div>

      <!-- Expanded Detail -->
      <%= if @expanded do %>
        <div class="border-t border-border p-space-lg space-y-space-lg">
          <!-- Source Info -->
          <%= if @finding.source do %>
            <div>
              <h5 class="text-label text-ink-muted mb-space-sm">Source</h5>
              <div class="bg-surface-sunk rounded-md p-space-md space-y-space-xs text-body">
                <div class="flex items-center gap-space-sm">
                  <span class="text-ink-muted w-24 shrink-0">Domain:</span>
                  <span class="font-semibold text-ink">{source_domain(@finding.source)}</span>
                </div>
                <div class="flex items-start gap-space-sm">
                  <span class="text-ink-muted w-24 shrink-0">URL:</span>
                  <span class="text-ref text-ink break-all">{source_url(@finding.source)}</span>
                </div>
                <%= if source_title(@finding.source) do %>
                  <div class="flex items-start gap-space-sm">
                    <span class="text-ink-muted w-24 shrink-0">Title:</span>
                    <span class="text-ink">{source_title(@finding.source)}</span>
                  </div>
                <% end %>
                <div class="flex items-center gap-space-sm">
                  <span class="text-ink-muted w-24 shrink-0">Reliability:</span>
                  <div class="flex items-center gap-space-sm">
                    <div class="w-20 h-space-sm rounded-sm bg-surface overflow-hidden">
                      <div
                        class="h-full rounded-sm bg-ink-muted"
                        style={"width: #{source_reliability(@finding.source) * 100}%"}
                      />
                    </div>
                    <span class="text-offset text-ink">{Float.round(source_reliability(@finding.source) * 100, 1)}%</span>
                  </div>
                </div>
                <div class="flex items-center gap-space-sm">
                  <span class="text-ink-muted w-24 shrink-0">Bias:</span>
                  <.badge size={:xs}>
                    {source_bias(@finding.source)}
                  </.badge>
                </div>
                <div class="flex items-center gap-space-sm">
                  <span class="text-ink-muted w-24 shrink-0">Trust Tier:</span>
                  <.badge variant={trust_variant(source_trust(@finding.source))} size={:xs}>
                    {source_trust(@finding.source)}
                  </.badge>
                </div>
              </div>
            </div>
          <% end %>

          <!-- Raw Context -->
          <%= if @finding.raw_context && @finding.raw_context != "" do %>
            <div>
              <h5 class="text-label text-ink-muted mb-space-sm">Raw Context</h5>
              <div class="bg-surface-sunk rounded-md p-space-md text-term text-ink whitespace-pre-wrap max-h-48 overflow-y-auto">
                {@finding.raw_context}
              </div>
            </div>
          <% end %>

          <!-- Metadata -->
          <div class="flex flex-wrap gap-space-lg text-caption text-ink-muted">
            <span>ID: <span class="text-ref">{@finding.id}</span></span>
            <%= if @finding.extracted_at do %>
              <span>Extracted: {Calendar.strftime(@finding.extracted_at, "%Y-%m-%d %H:%M:%S")}</span>
            <% end %>
            <%= if @finding.corroboration_group do %>
              <span>Corroboration Group: <span class="text-ref">{@finding.corroboration_group}</span></span>
            <% end %>
          </div>
        </div>
      <% end %>
    </.card>
    """
  end

  # ============================================================================
  # HELPERS
  # ============================================================================

  @session_variants %{active: :warning, completed: :success, cancelled: :default}

  @goal_variants %{completed: :success, failed: :error, in_progress: :warning, pending: :default}

  @priority_variants %{high: :default, low: :default}

  @investigation_variants %{
    concluded: :success,
    evaluating: :warning,
    gathering_evidence: :info,
    planning: :default
  }

  @conclusion_variants %{
    hypotheses_supported: :success,
    hypotheses_falsified: :default,
    inconclusive: :warning,
    mixed: :warning
  }

  @hypothesis_variants %{
    supported: :success,
    falsified: :default,
    inconclusive: :warning,
    testing: :info,
    untested: :default
  }

  @trust_variants %{
    nil => :default,
    :verified => :success,
    :neutral => :default,
    :untrusted => :warning,
    :blocked => :default
  }

  defp session_variant(status), do: fetch_variant!(@session_variants, status, "session status")
  defp goal_variant(status), do: fetch_variant!(@goal_variants, status, "goal status")
  defp priority_variant(priority), do: fetch_variant!(@priority_variants, priority, "goal priority")

  defp investigation_variant(status),
    do: fetch_variant!(@investigation_variants, status, "investigation status")

  defp conclusion_variant(conclusion),
    do: fetch_variant!(@conclusion_variants, conclusion, "investigation conclusion")

  defp hypothesis_variant(status), do: fetch_variant!(@hypothesis_variants, status, "hypothesis status")
  defp trust_variant(tier), do: fetch_variant!(@trust_variants, tier, "source trust tier")

  defp fetch_variant!(variants, value, what) do
    case Map.fetch(variants, value) do
      {:ok, variant} ->
        variant

      :error ->
        raise ArgumentError,
              "ChatWeb.SessionsLive: no badge treatment for #{what} #{inspect(value)}. " <>
                "The mapped values are #{inspect(Map.keys(variants))}."
    end
  end

  defp format_conclusion(:hypotheses_supported), do: "supported"
  defp format_conclusion(:hypotheses_falsified), do: "falsified"
  defp format_conclusion(:inconclusive), do: "inconclusive"
  defp format_conclusion(:mixed), do: "mixed"
  defp format_conclusion(other), do: to_string(other)

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
