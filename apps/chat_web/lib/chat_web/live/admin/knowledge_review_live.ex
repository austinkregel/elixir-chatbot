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
            <h1 class="text-xl font-bold">Knowledge Review Queue</h1>
            <p class="text-sm text-base-content/60">Review and approve knowledge expansion candidates</p>
          </div>
          <button
            class="btn btn-primary btn-sm"
            phx-click="show_start_session"
          >
            Start Learning Session
          </button>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6">
        <!-- Stats Bar -->
        <div class="stats shadow mb-6 w-full">
          <div class="stat">
            <div class="stat-title">Pending</div>
            <div class="stat-value text-primary"><%= @stats.pending %></div>
          </div>
          <div class="stat">
            <div class="stat-title">Approved Today</div>
            <div class="stat-value text-success"><%= @stats.approved_today %></div>
          </div>
          <div class="stat">
            <div class="stat-title">Rejected Today</div>
            <div class="stat-value text-error"><%= @stats.rejected_today %></div>
          </div>
          <div class="stat">
            <div class="stat-title">Total Approved</div>
            <div class="stat-value"><%= @stats.approved %></div>
          </div>
        </div>

      <!-- Status Tabs -->
      <div class="tabs tabs-boxed mb-4">
        <button
          class={"tab #{if @current_tab == :pending, do: "tab-active"}"}
          phx-click="change_tab"
          phx-value-tab="pending"
        >
          Pending
          <span class="badge badge-sm ml-2"><%= @stats.pending %></span>
        </button>
        <button
          class={"tab #{if @current_tab == :approved, do: "tab-active"}"}
          phx-click="change_tab"
          phx-value-tab="approved"
        >
          Approved
          <span class="badge badge-sm badge-success ml-2"><%= @stats.approved %></span>
        </button>
        <button
          class={"tab #{if @current_tab == :rejected, do: "tab-active"}"}
          phx-click="change_tab"
          phx-value-tab="rejected"
        >
          Rejected
          <span class="badge badge-sm badge-error ml-2"><%= @stats.rejected %></span>
        </button>
        <button
          class={"tab #{if @current_tab == :deferred, do: "tab-active"}"}
          phx-click="change_tab"
          phx-value-tab="deferred"
        >
          Deferred
          <span class="badge badge-sm badge-warning ml-2"><%= @stats.deferred %></span>
        </button>
      </div>

      <!-- Active Sessions -->
      <%= if length(@sessions) > 0 do %>
        <div class="mb-6">
          <h2 class="text-lg font-semibold mb-2">Active Sessions</h2>
          <div class="flex flex-wrap gap-2">
            <%= for session <- @sessions do %>
              <div class={"badge badge-lg gap-2 #{session_status_class(session.status)}"}>
                <span><%= session.topic || "Session" %></span>
                <span class="badge badge-sm"><%= session.findings_count %> findings</span>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>

      <!-- Bulk Actions (only for pending) -->
      <%= if @current_tab == :pending do %>
        <div class="flex gap-2 mb-4">
          <button
            class="btn btn-success btn-sm"
            phx-click="bulk_approve"
            disabled={MapSet.size(@selected_ids) == 0}
          >
            Approve Selected (<%= MapSet.size(@selected_ids) %>)
          </button>
          <button
            class="btn btn-error btn-sm"
            phx-click="bulk_reject"
            disabled={MapSet.size(@selected_ids) == 0}
          >
            Reject Selected
          </button>
          <button
            class="btn btn-warning btn-sm"
            phx-click="cleanup_html"
            data-confirm="This will reject all pending items containing HTML/JavaScript fragments. Continue?"
          >
            <.icon name="hero-trash" class="size-4 mr-1" />
            Cleanup HTML Fragments
          </button>
          <div class="flex-1"></div>
          <select class="select select-bordered select-sm" phx-change="change_sort">
            <option value="confidence" selected={@sort_by == :confidence}>Sort by Confidence</option>
            <option value="created_at" selected={@sort_by == :created_at}>Sort by Date</option>
          </select>
        </div>
      <% end %>

      <!-- Candidate List -->
      <div class="space-y-4">
        <%= if length(@candidates) == 0 do %>
          <div class="card bg-base-200">
            <div class="card-body text-center">
              <p class="text-base-content/70">No <%= @current_tab %> candidates.</p>
              <%= if @current_tab == :pending do %>
                <p class="text-sm text-base-content/50">Start a learning session to discover new knowledge.</p>
              <% end %>
            </div>
          </div>
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
        <div class="modal modal-open">
          <div class="modal-box">
            <h3 class="font-bold text-lg">Start Learning Session</h3>
            <form phx-submit="start_session">
              <div class="form-control mt-4">
                <label class="label">
                  <span class="label-text">Topic to research</span>
                </label>
                <input
                  type="text"
                  name="topic"
                  placeholder="e.g., European capitals, Nobel Prize winners"
                  class="input input-bordered"
                  value={@new_session_topic}
                  phx-change="update_topic"
                  autofocus
                />
              </div>
              <div class="modal-action">
                <button type="button" class="btn" phx-click="hide_start_session">Cancel</button>
                <button type="submit" class="btn btn-primary">Start Session</button>
              </div>
            </form>
          </div>
          <div class="modal-backdrop" phx-click="hide_start_session"></div>
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
    <div class={"card bg-base-100 shadow-xl #{if @selected, do: "ring-2 ring-primary"}"}>
      <div class="card-body">
        <div class="flex items-start gap-4">
          <!-- Checkbox (only for pending) -->
          <%= if @show_actions do %>
            <input
              type="checkbox"
              class="checkbox checkbox-primary mt-1"
              checked={@selected}
              phx-click="toggle_select"
              phx-value-id={@candidate.id}
            />
          <% else %>
            <!-- Status badge for non-pending -->
            <span class={"badge #{status_badge_class(@candidate.status)}"}>
              <%= @candidate.status %>
            </span>
          <% end %>

          <div class="flex-1">
            <!-- Claim -->
            <h2 class="card-title text-base"><%= @candidate.finding.claim %></h2>

            <!-- Entity and Type -->
            <div class="flex gap-2 mt-2">
              <span class="badge badge-outline">
                <%= @candidate.finding.entity %>
              </span>
              <%= if @candidate.finding.entity_type do %>
                <span class="badge badge-ghost">
                  <%= @candidate.finding.entity_type %>
                </span>
              <% end %>
            </div>

            <!-- Source Info -->
            <div class="flex items-center gap-2 mt-3">
              <.source_badge source={@candidate.finding.source} />
            </div>

            <!-- Corroboration -->
            <div class="text-sm mt-2 text-base-content/70">
              <strong>Sources:</strong>
              <%= length(@candidate.corroborating_sources) + 1 %>
              <%= if length(@candidate.corroborating_sources) > 0 do %>
                <span class="text-xs opacity-70">
                  (<%= format_domains(@candidate.corroborating_sources) %>)
                </span>
              <% end %>
            </div>

            <!-- Contradiction Warning -->
            <%= if length(@candidate.existing_contradictions) > 0 do %>
              <div class="alert alert-warning mt-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="stroke-current shrink-0 h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <span class="font-semibold">Contradicts existing belief:</span>
                  <%= for conflict <- Enum.take(@candidate.existing_contradictions, 2) do %>
                    <div class="text-sm"><%= conflict.object || inspect(conflict) %></div>
                  <% end %>
                </div>
              </div>
            <% end %>

            <!-- Confidence -->
            <div class="mt-3">
              <div class="flex justify-between text-xs mb-1">
                <span>Confidence</span>
                <span><%= format_confidence(@candidate.aggregate_confidence) %>%</span>
              </div>
              <progress
                class={"progress #{confidence_class(@candidate.aggregate_confidence)} w-full"}
                value={@candidate.aggregate_confidence * 100}
                max="100"
              ></progress>
            </div>
          </div>

          <!-- Action Buttons (only for pending) -->
          <%= if @show_actions do %>
            <div class="flex flex-col gap-2">
              <button
                class="btn btn-success btn-sm"
                phx-click="approve"
                phx-value-id={@candidate.id}
              >
                Approve
              </button>
              <button
                class="btn btn-warning btn-sm"
                phx-click="defer"
                phx-value-id={@candidate.id}
              >
                Defer
              </button>
              <button
                class="btn btn-error btn-sm"
                phx-click="reject"
                phx-value-id={@candidate.id}
              >
                Reject
              </button>
            </div>
          <% else %>
            <!-- Reviewed timestamp for non-pending -->
            <%= if @show_reviewed_at && @candidate.reviewed_at do %>
              <div class="text-xs text-base-content/50">
                Reviewed: <%= format_datetime(@candidate.reviewed_at) %>
              </div>
            <% end %>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp status_badge_class(:approved) do
    "badge-success"
  end

  defp status_badge_class(:rejected) do
    "badge-error"
  end

  defp status_badge_class(:deferred) do
    "badge-warning"
  end

  defp status_badge_class(_) do
    "badge-ghost"
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

  defp source_badge(assigns) do
    tier_class =
      case assigns.source.trust_tier do
        :verified -> "badge-success"
        :neutral -> "badge-info"
        :untrusted -> "badge-warning"
        :blocked -> "badge-error"
        _ -> "badge-ghost"
      end

    bias_label =
      case assigns.source.bias_rating do
        :left -> "L"
        :center_left -> "CL"
        :center -> "C"
        :center_right -> "CR"
        :right -> "R"
        _ -> "?"
      end

    assigns = assign(assigns, :tier_class, tier_class)
    assigns = assign(assigns, :bias_label, bias_label)

    ~H"""
    <div class="flex items-center gap-2">
      <span class={"badge #{@tier_class}"}>
        <%= @source.domain %>
      </span>
      <span class="badge badge-ghost" title={"Bias: #{@source.bias_rating}"}>
        <%= @bias_label %>
      </span>
      <span class="badge badge-outline" title="Reliability score">
        <%= format_confidence(@source.reliability_score) %>%
      </span>
    </div>
    """
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

  defp confidence_class(score) when is_float(score) do
    cond do
      score >= 0.8 -> "progress-success"
      score >= 0.5 -> "progress-warning"
      true -> "progress-error"
    end
  end

  defp confidence_class(_) do
    "progress-info"
  end

  defp session_status_class(:active) do
    "badge-primary"
  end

  defp session_status_class(:completed) do
    "badge-success"
  end

  defp session_status_class(:cancelled) do
    "badge-error"
  end

  defp session_status_class(_) do
    "badge-ghost"
  end
end