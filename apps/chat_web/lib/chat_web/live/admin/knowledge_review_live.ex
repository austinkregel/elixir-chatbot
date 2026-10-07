defmodule ChatWeb.Admin.KnowledgeReviewLive do
  @moduledoc "LiveView for reviewing knowledge expansion candidates.\n\nProvides an admin interface for:\n- Viewing pending knowledge candidates\n- Source reliability badges and bias indicators\n- Corroboration evidence display\n- Contradiction highlighting with existing beliefs\n- Approve/Reject/Defer actions\n- Bulk review capabilities\n"

  alias Phoenix.PubSub
  alias Brain.Knowledge
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Knowledge.{ReviewQueue, LearningCenter}

  @refresh_interval_ms 5000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "knowledge:review")
      :timer.send_interval(@refresh_interval_ms, self(), :refresh)
    end

    {:ok, assign_initial_state(socket)}
  end

  defp assign_initial_state(socket) do
    socket
    |> assign(:current_tab, :pending)
    |> assign(:candidates, load_candidates(:pending))
    |> assign(:stats, load_stats())
    |> assign(:sessions, load_sessions())
    |> assign(:selected_ids, MapSet.new())
    |> assign(:filter, :all)
    |> assign(:sort_by, :confidence)
    |> assign(:new_session_topic, "")
    |> assign(:show_start_session_modal, false)
    |> assign(:page_title, "Knowledge Review")
  end

  defp load_candidates(:pending) do
    if ReviewQueue.ready?() do
      ReviewQueue.get_pending(limit: 50, sort_by: :confidence)
    else
      []
    end
  end

  defp load_candidates(status) when status in [:approved, :rejected, :deferred] do
    if ReviewQueue.ready?() do
      ReviewQueue.get_by_status(status, limit: 50, sort_by: :reviewed_at)
    else
      []
    end
  end

  defp load_stats do
    if ReviewQueue.ready?() do
      ReviewQueue.stats()
    else
      %{pending: 0, approved: 0, rejected: 0, deferred: 0, approved_today: 0, rejected_today: 0}
    end
  end

  defp load_sessions do
    if LearningCenter.ready?() do
      LearningCenter.list_sessions(limit: 10)
    else
      []
    end
  end

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
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-title text-ink">Knowledge Review Queue</h1>
            <p class="text-body text-ink-muted">Review and approve knowledge expansion candidates</p>
          </div>
          <.btn variant={:primary} size={:sm} phx-click="show_start_session">
            Start Learning Session
          </.btn>
        </div>
      </:page_header>

      <div class="p-space-lg sm:p-space-xl">
        <!-- Stats Bar -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-space-lg mb-space-xl">
          <.stat_kpi label="Pending" value={to_string(@stats.pending)} />
          <.stat_kpi label="Approved Today" value={to_string(@stats.approved_today)} />
          <.stat_kpi label="Rejected Today" value={to_string(@stats.rejected_today)} />
          <.stat_kpi label="Total Approved" value={to_string(@stats.approved)} />
        </div>

      <!-- Status Tabs -->
      <.tabs class="mb-space-lg">
        <.tab active={@current_tab == :pending} phx-click="change_tab" phx-value-tab="pending">
          Pending
          <.badge class="ml-space-sm"><%= @stats.pending %></.badge>
        </.tab>
        <.tab active={@current_tab == :approved} phx-click="change_tab" phx-value-tab="approved">
          Approved
          <.badge variant={:success} class="ml-space-sm"><%= @stats.approved %></.badge>
        </.tab>
        <.tab active={@current_tab == :rejected} phx-click="change_tab" phx-value-tab="rejected">
          Rejected
          <.badge variant={:error} class="ml-space-sm"><%= @stats.rejected %></.badge>
        </.tab>
        <.tab active={@current_tab == :deferred} phx-click="change_tab" phx-value-tab="deferred">
          Deferred
          <.badge variant={:warning} class="ml-space-sm"><%= @stats.deferred %></.badge>
        </.tab>
      </.tabs>

      <!-- Active Sessions -->
      <%= if length(@sessions) > 0 do %>
        <div class="mb-space-xl">
          <h2 class="text-heading text-ink mb-space-sm">Active Sessions</h2>
          <div class="flex flex-wrap gap-space-sm">
            <%= for session <- @sessions do %>
              <.badge variant={session_status_variant(session.status)}>
                <span><%= session.topic || "Session" %></span>
                <.badge size={:xs}><%= session.findings_count %> findings</.badge>
              </.badge>
            <% end %>
          </div>
        </div>
      <% end %>

      <!-- Bulk Actions (only for pending) -->
      <%= if @current_tab == :pending do %>
        <div class="flex items-center gap-space-sm mb-space-lg">
          <.reach_shared_badge />
          <.btn
            variant={:primary}
            size={:sm}
            class="outline-mark outline-reach-shared"
            phx-click="bulk_approve"
            disabled={MapSet.size(@selected_ids) == 0}
          >
            Approve Selected (<%= MapSet.size(@selected_ids) %>)
          </.btn>
          <.btn
            variant={:primary}
            size={:sm}
            phx-click="bulk_reject"
            disabled={MapSet.size(@selected_ids) == 0}
          >
            Reject Selected
          </.btn>
          <.btn
            variant={:primary}
            size={:sm}
            phx-click="cleanup_html"
            data-confirm="This will reject all pending items containing HTML/JavaScript fragments. Continue?"
          >
            <.icon name="hero-trash" class="size-4" />
            Cleanup HTML Fragments
          </.btn>
          <div class="flex-1"></div>
          <select
            class="h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink"
            phx-change="change_sort"
          >
            <option value="confidence" selected={@sort_by == :confidence}>Sort by Confidence</option>
            <option value="created_at" selected={@sort_by == :created_at}>Sort by Date</option>
          </select>
        </div>
      <% end %>

      <!-- Candidate List -->
      <div class="space-y-space-lg">
        <%= if length(@candidates) == 0 do %>
          <.card class="bg-surface-sunk">
            <.card_body class="text-center">
              <p class="text-body text-ink-muted">No <%= @current_tab %> candidates.</p>
              <%= if @current_tab == :pending do %>
                <p class="text-body-dense text-ink-muted">Start a learning session to discover new knowledge.</p>
              <% end %>
            </.card_body>
          </.card>
        <% else %>
          <%= for candidate <- @candidates do %>
            <.candidate_card
              candidate={candidate}
              selected={MapSet.member?(@selected_ids, candidate.id)}
              show_actions={@current_tab == :pending}
              show_reviewed_at={@current_tab != :pending}
            />
          <% end %>
        <% end %>
      </div>

      <!-- Start Session Modal -->
      <%= if @show_start_session_modal do %>
        <div class="fixed inset-0 z-50 flex items-center justify-center p-space-lg">
          <div class="fixed inset-0 bg-ground/80" phx-click="hide_start_session"></div>
          <div class="relative w-full max-w-lg rounded-md border border-border-strong bg-surface-raised shadow-overlay p-space-lg">
            <h3 class="text-heading text-ink">Start Learning Session</h3>
            <form phx-submit="start_session">
              <div class="mt-space-lg">
                <label class="block mb-space-xs text-label text-ink-muted">
                  Topic to research
                </label>
                <input
                  type="text"
                  name="topic"
                  placeholder="e.g., European capitals, Nobel Prize winners"
                  class="w-full h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink placeholder:text-ink-muted"
                  value={@new_session_topic}
                  phx-change="update_topic"
                  autofocus
                />
              </div>
              <div class="mt-space-lg flex justify-end gap-space-sm">
                <.btn type="button" variant={:outline} phx-click="hide_start_session">Cancel</.btn>
                <.btn type="submit" variant={:primary}>Start Session</.btn>
              </div>
            </form>
          </div>
        </div>
      <% end %>
      </div>
    </.app_shell>
    """
  end

  defp candidate_card(assigns) do
    assigns =
      assigns
      |> Map.put_new(:show_actions, true)
      |> Map.put_new(:show_reviewed_at, false)

    ~H"""
    <.card class={if @selected, do: "border-accent bg-accent-wash"}>
      <.card_body>
        <div class="flex items-start gap-space-lg">
          <!-- Checkbox (only for pending) -->
          <%= if @show_actions do %>
            <input
              type="checkbox"
              class="size-4 accent-primary mt-space-xs"
              checked={@selected}
              phx-click="toggle_select"
              phx-value-id={@candidate.id}
            />
          <% else %>
            <!-- Status badge for non-pending -->
            <.badge variant={candidate_status_variant(@candidate.status)}>
              <%= @candidate.status %>
            </.badge>
          <% end %>

          <div class="flex-1">
            <!-- Claim -->
            <h2 class="text-heading text-ink"><%= @candidate.finding.claim %></h2>

            <!-- Entity and Type -->
            <div class="flex gap-space-sm mt-space-sm">
              <.badge class="border border-border-strong">
                <%= @candidate.finding.entity %>
              </.badge>
              <%= if @candidate.finding.entity_type do %>
                <.badge>
                  <%= @candidate.finding.entity_type %>
                </.badge>
              <% end %>
            </div>

            <!-- Source Info -->
            <div class="flex items-center gap-space-sm mt-space-md">
              <.badge variant={trust_tier_variant(@candidate.finding.source.trust_tier)}>
                <%= @candidate.finding.source.domain %>
              </.badge>
              <.badge title={"Bias: #{@candidate.finding.source.bias_rating}"}>
                <%= bias_label(@candidate.finding.source.bias_rating) %>
              </.badge>
              <.badge class="border border-border-strong" title="Reliability score">
                <%= format_confidence(@candidate.finding.source.reliability_score) %>%
              </.badge>
            </div>

            <!-- Corroboration -->
            <div class="text-body mt-space-sm text-ink">
              <strong>Sources:</strong>
              <%= length(@candidate.corroborating_sources) + 1 %>
              <%= if length(@candidate.corroborating_sources) > 0 do %>
                <span class="text-caption text-ink-muted">
                  (<%= format_domains(@candidate.corroborating_sources) %>)
                </span>
              <% end %>
            </div>

            <!-- Contradiction Warning -->
            <%= if length(@candidate.existing_contradictions) > 0 do %>
              <.alert variant={:warning} icon="hero-exclamation-triangle" class="mt-space-md">
                <span class="font-semibold">Contradicts existing belief:</span>
                <%= for conflict <- Enum.take(@candidate.existing_contradictions, 2) do %>
                  <div class="text-body-dense"><%= conflict.object || inspect(conflict) %></div>
                <% end %>
              </.alert>
            <% end %>

            <!-- Confidence -->
            <div class="mt-space-md">
              <div class="flex justify-between text-caption text-ink mb-space-xs">
                <span>Confidence</span>
                <span class="text-offset"><%= format_confidence(@candidate.aggregate_confidence) %>%</span>
              </div>
              <div
                class="w-full h-space-sm rounded-sm bg-surface-sunk"
                role="meter"
                aria-label="Confidence"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={@candidate.aggregate_confidence * 100}
              >
                <div
                  class="h-full rounded-sm bg-ink-muted"
                  style={"width: #{@candidate.aggregate_confidence * 100}%"}
                />
              </div>
            </div>
          </div>

          <!-- Action Buttons (only for pending) -->
          <%= if @show_actions do %>
            <div class="flex flex-col items-stretch gap-space-sm">
              <.reach_shared_badge />
              <.btn
                variant={:primary}
                size={:sm}
                class="outline-mark outline-reach-shared"
                phx-click="approve"
                phx-value-id={@candidate.id}
              >
                Approve
              </.btn>
              <.btn
                variant={:primary}
                size={:sm}
                phx-click="defer"
                phx-value-id={@candidate.id}
              >
                Defer
              </.btn>
              <.btn
                variant={:primary}
                size={:sm}
                phx-click="reject"
                phx-value-id={@candidate.id}
              >
                Reject
              </.btn>
            </div>
          <% else %>
            <!-- Reviewed timestamp for non-pending -->
            <%= if @show_reviewed_at && @candidate.reviewed_at do %>
              <div class="text-caption text-ink-muted">
                Reviewed: <%= format_datetime(@candidate.reviewed_at) %>
              </div>
            <% end %>
          <% end %>
        </div>
      </.card_body>
    </.card>
    """
  end

  defp reach_shared_badge(assigns) do
    ~H"""
    <span class="inline-flex items-center gap-space-xs rounded-sm border border-reach-shared px-space-xs text-caption font-semibold text-reach-shared">
      <.icon name="hero-share-micro" class="size-3" /> writes shared
    </span>
    """
  end

  @candidate_status_variants %{
    pending: :default,
    approved: :success,
    auto_approved: :default,
    rejected: :error,
    deferred: :warning
  }

  @trust_tier_variants %{
    verified: :success,
    neutral: :info,
    untrusted: :warning,
    blocked: :error
  }

  @session_status_variants %{
    active: :primary,
    completed: :success,
    cancelled: :error
  }

  defp candidate_status_variant(status),
    do: fetch_variant!(@candidate_status_variants, status, "candidate status")

  defp trust_tier_variant(tier), do: fetch_variant!(@trust_tier_variants, tier, "source trust tier")

  defp session_status_variant(status),
    do: fetch_variant!(@session_status_variants, status, "session status")

  defp fetch_variant!(variants, value, what) do
    case Map.fetch(variants, value) do
      {:ok, variant} ->
        variant

      :error ->
        raise ArgumentError,
              "ChatWeb.Admin.KnowledgeReviewLive: no badge treatment for #{what} #{inspect(value)}. " <>
                "The mapped values are #{inspect(Map.keys(variants))}."
    end
  end

  defp format_datetime(nil) do
    "N/A"
  end

  defp format_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y-%m-%d %H:%M")
  end

  defp format_datetime(_) do
    "N/A"
  end

  defp bias_label(bias_rating) do
    case bias_rating do
      :left -> "L"
      :center_left -> "CL"
      :center -> "C"
      :center_right -> "CR"
      :right -> "R"
      _ -> "?"
    end
  end

  @impl true
  def handle_event("toggle_select", %{"id" => id}, socket) do
    selected = socket.assigns.selected_ids

    new_selected =
      if MapSet.member?(selected, id) do
        MapSet.delete(selected, id)
      else
        MapSet.put(selected, id)
      end

    {:noreply, assign(socket, :selected_ids, new_selected)}
  end

  @impl true
  def handle_event("approve", %{"id" => id}, socket) do
    case ReviewQueue.approve(id) do
      {:ok, _} ->
        {:noreply, refresh_data(socket)}

      {:error, reason} ->
        Logger.warning("Failed to approve candidate", id: id, reason: inspect(reason))
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("reject", %{"id" => id}, socket) do
    case ReviewQueue.reject(id) do
      {:ok, _} ->
        {:noreply, refresh_data(socket)}

      {:error, reason} ->
        Logger.warning("Failed to reject candidate", id: id, reason: inspect(reason))
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("defer", %{"id" => id}, socket) do
    case ReviewQueue.defer(id) do
      {:ok, _} ->
        {:noreply, refresh_data(socket)}

      {:error, reason} ->
        Logger.warning("Failed to defer candidate", id: id, reason: inspect(reason))
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("bulk_approve", _, socket) do
    ids = MapSet.to_list(socket.assigns.selected_ids)
    ReviewQueue.bulk_approve(ids)

    socket =
      socket
      |> assign(:selected_ids, MapSet.new())
      |> refresh_data()

    {:noreply, socket}
  end

  @impl true
  def handle_event("bulk_reject", _, socket) do
    ids = MapSet.to_list(socket.assigns.selected_ids)
    ReviewQueue.bulk_reject(ids)

    socket =
      socket
      |> assign(:selected_ids, MapSet.new())
      |> refresh_data()

    {:noreply, socket}
  end

  @impl true
  def handle_event("change_sort", %{"value" => sort_by}, socket) do
    sort_atom = String.to_existing_atom(sort_by)
    {:noreply, assign(socket, :sort_by, sort_atom) |> refresh_data()}
  end

  @impl true
  def handle_event("change_tab", %{"tab" => tab}, socket) do
    tab_atom = String.to_existing_atom(tab)

    socket =
      socket
      |> assign(:current_tab, tab_atom)
      |> assign(:selected_ids, MapSet.new())
      |> assign(:candidates, load_candidates(tab_atom))

    {:noreply, socket}
  end

  @impl true
  def handle_event("show_start_session", _, socket) do
    {:noreply, assign(socket, :show_start_session_modal, true)}
  end

  @impl true
  def handle_event("hide_start_session", _, socket) do
    {:noreply, assign(socket, show_start_session_modal: false, new_session_topic: "")}
  end

  @impl true
  def handle_event("update_topic", %{"topic" => topic}, socket) do
    {:noreply, assign(socket, :new_session_topic, topic)}
  end

  @impl true
  def handle_event("start_session", %{"topic" => topic}, socket) do
    if String.trim(topic) != "" do
      case LearningCenter.start_session(topic) do
        {:ok, session} ->
          Logger.info("Started learning session", session_id: session.id, topic: topic)

          socket =
            socket
            |> assign(show_start_session_modal: false, new_session_topic: "")
            |> refresh_data()

          {:noreply, socket}

        {:error, reason} ->
          Logger.warning("Failed to start session", topic: topic, reason: inspect(reason))
          {:noreply, socket}
      end
    else
      {:noreply, socket}
    end
  end

  @impl true
  def handle_event("cleanup_html", _, socket) do
    case ReviewQueue.cleanup_html_fragments() do
      {:ok, count} ->
        Logger.info("Cleaned up HTML fragments", rejected: count)

        socket =
          socket
          |> put_flash(:info, "Rejected #{count} HTML fragment items")
          |> refresh_data()

        {:noreply, socket}

      {:error, reason} ->
        Logger.warning("Failed to cleanup HTML fragments", reason: inspect(reason))

        socket =
          socket
          |> put_flash(:error, "Failed to cleanup: #{inspect(reason)}")

        {:noreply, socket}
    end
  end

  @impl true
  def handle_info({event, _data}, socket)
      when event in [
             :candidate_added,
             :candidate_approved,
             :candidate_rejected,
             :candidate_deferred,
             :bulk_approved,
             :bulk_rejected
           ] do
    {:noreply, refresh_data(socket)}
  end

  @impl true
  def handle_info(:refresh, socket) do
    {:noreply, refresh_data(socket)}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, refresh_data(socket)}
  end

  @impl true
  def handle_info(_msg, socket) do
    {:noreply, socket}
  end

  defp refresh_data(socket) do
    current_tab = socket.assigns[:current_tab] || :pending

    socket
    |> assign(:candidates, load_candidates(current_tab))
    |> assign(:stats, load_stats())
    |> assign(:sessions, load_sessions())
  end

  defp format_confidence(score) when is_float(score) do
    Float.round(score * 100, 1)
  end

  defp format_confidence(_) do
    "N/A"
  end

  defp format_domains(sources) do
    sources
    |> Enum.map(& &1.domain)
    |> Enum.uniq()
    |> Enum.take(3)
    |> Enum.join(", ")
  end

end