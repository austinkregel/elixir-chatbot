defmodule ChatWeb.Admin.IntentReviewLive do
  @moduledoc "LiveView for reviewing novel intent candidates.\n\nProvides an admin interface for:\n- Viewing pending intent candidates\n- Rich annotation UI (tags, notes, span annotations)\n- Promoting candidates as variations or new intents\n- Filtering and searching candidates\n"

  alias Phoenix.PubSub
  alias Brain.Analysis
  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Analysis.{IntentReviewQueue, IntentRegistry, IntentPromoter}

  @refresh_interval_ms 5000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "intent:review")
      :timer.send_interval(@refresh_interval_ms, self(), :refresh)
    end

    {:ok, assign_initial_state(socket)}
  end

  defp assign_initial_state(socket) do
    socket
    |> assign(:current_tab, :pending)
    |> assign(:candidates, load_candidates(:pending))
    |> assign(:stats, load_stats())
    |> assign(:selected_ids, MapSet.new())
    |> assign(:filter, :all)
    |> assign(:sort_by, :novelty_score)
    |> assign(:search_query, "")
    |> assign(:show_annotation_modal, false)
    |> assign(:show_promote_modal, false)
    |> assign(:selected_candidate, nil)
    |> assign(:all_intents, IntentRegistry.list_intents())
    |> assign(:promotion_action, :variation)
    |> assign(:promoted_to_intent, "")
    |> assign(:new_intent_domain, "")
    |> assign(:new_intent_category, "")
    |> assign(:new_intent_speech_act, "")
    |> assign(:page_title, "Intent Review")
  end

  defp load_candidates(:pending) do
    if IntentReviewQueue.ready?() do
      IntentReviewQueue.get_pending(limit: 50, sort_by: :novelty_score)
    else
      []
    end
  end

  defp load_candidates(status) when status in [:approved, :rejected, :deferred] do
    if IntentReviewQueue.ready?() do
      IntentReviewQueue.get_by_status(status, limit: 50, sort_by: :reviewed_at)
    else
      []
    end
  end

  defp load_stats do
    if IntentReviewQueue.ready?() do
      IntentReviewQueue.stats()
    else
      %{pending: 0, approved: 0, rejected: 0, deferred: 0, approved_today: 0, rejected_today: 0}
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
            <h1 class="text-xl font-bold">Intent Review Queue</h1>
            <p class="text-sm text-base-content/60">Review and promote novel intent candidates</p>
          </div>
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

      <!-- Search and Filters -->
      <%= if @current_tab == :pending do %>
        <div class="flex gap-2 mb-4">
          <input
            type="text"
            placeholder="Search utterances..."
            class="input input-bordered flex-1"
            phx-debounce="300"
            phx-change="search"
            value={@search_query}
          />
          <select class="select select-bordered select-sm" phx-change="change_sort">
            <option value="novelty_score" selected={@sort_by == :novelty_score}>Sort by Novelty</option>
            <option value="created_at" selected={@sort_by == :created_at}>Sort by Date</option>
            <option value="margin" selected={@sort_by == :margin}>Sort by Margin</option>
          </select>
        </div>
      <% end %>

      <!-- Candidate List -->
      <div class="space-y-4">
        <%= if length(@candidates) == 0 do %>
          <div class="card bg-base-200">
            <div class="card-body text-center">
              <p class="text-base-content/70">No <%= @current_tab %> candidates.</p>
            </div>
          </div>
        <% else %>
          <%= for candidate <- @candidates do %>
            <.candidate_card
              candidate={candidate}
              selected={MapSet.member?(@selected_ids, candidate.id)}
              show_actions={@current_tab == :pending}
              all_intents={@all_intents}
            />
          <% end %>
        <% end %>
      </div>

      <!-- Annotation Modal -->
      <%= if @show_annotation_modal && @selected_candidate do %>
        <.annotation_modal candidate={@selected_candidate} />
      <% end %>

      <!-- Promotion Modal -->
      <%= if @show_promote_modal && @selected_candidate do %>
        <.promotion_modal
          candidate={@selected_candidate}
          all_intents={@all_intents}
          promotion_action={@promotion_action}
          promoted_to_intent={@promoted_to_intent}
          new_intent_domain={@new_intent_domain}
          new_intent_category={@new_intent_category}
          new_intent_speech_act={@new_intent_speech_act}
        />
      <% end %>
      </div>
    </.app_shell>
    """
  end

  defp candidate_card(assigns) do
    ~H"""
    <div class={"card bg-base-100 shadow-xl #{if @selected, do: "ring-2 ring-primary"}"}>
      <div class="card-body">
        <div class="flex items-start gap-4">
          <div class="flex-1">
            <!-- Utterance -->
            <h2 class="card-title text-base mb-2"><%= @candidate.text %></h2>

            <!-- Intent Predictions -->
            <div class="mb-3">
              <div class="flex gap-2 items-center mb-1">
                <span class="badge badge-outline">Predicted: <%= @candidate.predicted_intent %></span>
                <span class="text-xs text-base-content/70">
                  Score: <%= format_score(@candidate.best_score) %> | Margin: <%= format_score(@candidate.margin) %>
                </span>
              </div>
              <%= if length(@candidate.top_k || []) > 1 do %>
                <div class="text-xs text-base-content/60 mt-1">
                  Top alternatives:
                  <%= for {intent, score} <- Enum.take(@candidate.top_k || [], 3) do %>
                    <span class="ml-2"><%= intent %> (<%= format_score(score) %>)</span>
                  <% end %>
                </div>
              <% end %>
            </div>

            <!-- Entities -->
            <%= if length(@candidate.extracted_entities || []) > 0 do %>
              <div class="flex gap-2 flex-wrap mb-2">
                <%= for entity <- @candidate.extracted_entities do %>
                  <span class="badge badge-ghost">
                    <%= entity[:entity_type] || "unknown" %>: <%= entity[:value] %>
                  </span>
                <% end %>
              </div>
            <% end %>

            <!-- Slot Fill Summary -->
            <%= if @candidate.slot_fill_summary do %>
              <div class="text-sm text-base-content/70 mb-2">
                <%= if map_size(@candidate.slot_fill_summary[:filled_slots] || %{}) > 0 do %>
                  <span class="text-success">Filled: <%= Enum.join(Map.keys(@candidate.slot_fill_summary[:filled_slots] || %{}), ", ") %></span>
                <% end %>
                <%= if length(@candidate.slot_fill_summary[:missing_required] || []) > 0 do %>
                  <span class="text-error ml-2">Missing: <%= Enum.join(@candidate.slot_fill_summary[:missing_required] || [], ", ") %></span>
                <% end %>
              </div>
            <% end %>

            <!-- Tags -->
            <%= if length(@candidate.annotation[:tags] || []) > 0 do %>
              <div class="flex gap-1 flex-wrap mb-2">
                <%= for tag <- @candidate.annotation[:tags] do %>
                  <span class="badge badge-sm"><%= tag %></span>
                <% end %>
              </div>
            <% end %>

            <!-- Notes -->
            <%= if @candidate.annotation[:notes] do %>
              <div class="text-sm text-base-content/60 italic">
                <%= @candidate.annotation[:notes] %>
              </div>
            <% end %>
          </div>

          <!-- Action Buttons -->
          <%= if @show_actions do %>
            <div class="flex flex-col gap-2">
              <button
                class="btn btn-sm btn-info"
                phx-click="show_annotate"
                phx-value-id={@candidate.id}
              >
                Annotate
              </button>
              <button
                class="btn btn-sm btn-success"
                phx-click="show_promote"
                phx-value-id={@candidate.id}
              >
                Promote
              </button>
              <button
                class="btn btn-sm btn-warning"
                phx-click="defer"
                phx-value-id={@candidate.id}
              >
                Defer
              </button>
              <button
                class="btn btn-sm btn-error"
                phx-click="reject"
                phx-value-id={@candidate.id}
              >
                Reject
              </button>
            </div>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp annotation_modal(assigns) do
    ~H"""
    <div class="modal modal-open">
      <div class="modal-box max-w-3xl">
        <h3 class="font-bold text-lg mb-4">Annotate Candidate</h3>
        <form phx-submit="save_annotation" phx-value-id={@candidate.id}>
          <!-- Tags -->
          <div class="form-control mb-4">
            <label class="label">
              <span class="label-text">Tags</span>
            </label>
            <div class="flex flex-wrap gap-2">
              <%= for tag <- [:priority, :domain_guess, :needs_entity_type, :typo, :multi_intent, :out_of_scope] do %>
                <label class="label cursor-pointer gap-2">
                  <input
                    type="checkbox"
                    class="checkbox checkbox-sm"
                    name="tags[]"
                    value={tag}
                    checked={tag in (@candidate.annotation[:tags] || [])}
                  />
                  <span class="label-text"><%= tag %></span>
                </label>
              <% end %>
            </div>
          </div>

          <!-- Notes -->
          <div class="form-control mb-4">
            <label class="label">
              <span class="label-text">Notes</span>
            </label>
            <textarea
              class="textarea textarea-bordered"
              name="notes"
              placeholder="Add reviewer notes..."
            ><%= @candidate.annotation[:notes] || "" %></textarea>
          </div>

          <!-- Domain Guess -->
          <div class="form-control mb-4">
            <label class="label">
              <span class="label-text">Domain Guess</span>
            </label>
            <input
              type="text"
              class="input input-bordered"
              name="domain_guess"
              placeholder="e.g., weather, music, device"
              value={@candidate.annotation[:domain_guess] || ""}
            />
          </div>

          <div class="modal-action">
            <button type="button" class="btn" phx-click="hide_annotation_modal">Cancel</button>
            <button type="submit" class="btn btn-primary">Save</button>
          </div>
        </form>
      </div>
      <div class="modal-backdrop" phx-click="hide_annotation_modal"></div>
    </div>
    """
  end

  defp promotion_modal(assigns) do
    ~H"""
    <div class="modal modal-open">
      <div class="modal-box max-w-3xl">
        <h3 class="font-bold text-lg mb-4">Promote Intent Candidate</h3>
        <form phx-submit="promote" phx-value-id={@candidate.id}>
          <!-- Promotion Action -->
          <div class="form-control mb-4">
            <label class="label">
              <span class="label-text">Promotion Type</span>
            </label>
            <select class="select select-bordered" name="promotion_action" phx-change="update_promotion_action">
              <option value="variation" selected={@promotion_action == :variation}>Variation of Existing Intent</option>
              <option value="new_intent" selected={@promotion_action == :new_intent}>New Intent</option>
            </select>
          </div>

          <!-- Variation: Select Existing Intent -->
          <%= if @promotion_action == :variation do %>
            <div class="form-control mb-4">
              <label class="label">
                <span class="label-text">Existing Intent</span>
              </label>
              <select class="select select-bordered" name="promoted_to_intent">
                <option value="">Select intent...</option>
                <%= for intent <- @all_intents do %>
                  <option value={intent} selected={intent == @promoted_to_intent}><%= intent %></option>
                <% end %>
              </select>
            </div>
          <% end %>

          <!-- New Intent: Enter Details -->
          <%= if @promotion_action == :new_intent do %>
            <div class="form-control mb-4">
              <label class="label">
                <span class="label-text">Intent Name</span>
              </label>
              <input
                type="text"
                class="input input-bordered"
                name="promoted_to_intent"
                placeholder="e.g., calendar.create_event"
                value={@promoted_to_intent}
                required
              />
            </div>

            <div class="form-control mb-4">
              <label class="label">
                <span class="label-text">Domain</span>
              </label>
              <input
                type="text"
                class="input input-bordered"
                name="new_intent_domain"
                placeholder="e.g., calendar, reminder, task"
                value={@new_intent_domain}
                required
              />
            </div>

            <div class="form-control mb-4">
              <label class="label">
                <span class="label-text">Category</span>
              </label>
              <select class="select select-bordered" name="new_intent_category">
                <option value="directive" selected={@new_intent_category == "directive"}>Directive</option>
                <option value="expressive" selected={@new_intent_category == "expressive"}>Expressive</option>
                <option value="assertive" selected={@new_intent_category == "assertive"}>Assertive</option>
              </select>
            </div>

            <div class="form-control mb-4">
              <label class="label">
                <span class="label-text">Speech Act</span>
              </label>
              <input
                type="text"
                class="input input-bordered"
                name="new_intent_speech_act"
                placeholder="e.g., command, request_information, greeting"
                value={@new_intent_speech_act}
                required
              />
            </div>
          <% end %>

          <div class="modal-action">
            <button type="button" class="btn" phx-click="hide_promote_modal">Cancel</button>
            <button type="submit" class="btn btn-primary">Promote</button>
          </div>
        </form>
      </div>
      <div class="modal-backdrop" phx-click="hide_promote_modal"></div>
    </div>
    """
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
  def handle_event("change_sort", %{"value" => sort_by}, socket) do
    sort_atom = String.to_existing_atom(sort_by)
    {:noreply, assign(socket, :sort_by, sort_atom) |> refresh_data()}
  end

  @impl true
  def handle_event("search", %{"value" => query}, socket) do
    {:noreply, assign(socket, :search_query, query) |> refresh_data()}
  end

  @impl true
  def handle_event("show_annotate", %{"id" => id}, socket) do
    case IntentReviewQueue.get(id) do
      {:ok, candidate} ->
        {:noreply, assign(socket, show_annotation_modal: true, selected_candidate: candidate)}

      {:error, _} ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("hide_annotation_modal", _, socket) do
    {:noreply, assign(socket, show_annotation_modal: false, selected_candidate: nil)}
  end

  @impl true
  def handle_event("save_annotation", %{"id" => id} = params, socket) do
    tags = Map.get(params, "tags", [])
    notes = Map.get(params, "notes", "")
    domain_guess = Map.get(params, "domain_guess", "")

    tags_list =
      cond do
        is_list(tags) -> Enum.map(tags, &String.to_existing_atom/1)
        is_binary(tags) -> [String.to_existing_atom(tags)]
        true -> []
      end

    annotation_updates = %{
      tags: tags_list,
      notes:
        if(notes != "") do
          notes
        else
          nil
        end,
      domain_guess:
        if(domain_guess != "") do
          domain_guess
        else
          nil
        end
    }

    case IntentReviewQueue.update_annotation(id, annotation_updates) do
      {:ok, _} ->
        {:noreply,
         assign(socket, show_annotation_modal: false, selected_candidate: nil) |> refresh_data()}

      {:error, _} ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("show_promote", %{"id" => id}, socket) do
    case IntentReviewQueue.get(id) do
      {:ok, candidate} ->
        {:noreply,
         assign(socket,
           show_promote_modal: true,
           selected_candidate: candidate,
           promotion_action: :variation,
           promoted_to_intent: "",
           new_intent_domain: candidate.annotation[:domain_guess] || "",
           new_intent_category: "",
           new_intent_speech_act: ""
         )}

      {:error, _} ->
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("hide_promote_modal", _, socket) do
    {:noreply,
     assign(socket,
       show_promote_modal: false,
       selected_candidate: nil,
       promotion_action: :variation,
       promoted_to_intent: ""
     )}
  end

  @impl true
  def handle_event("update_promotion_action", %{"promotion_action" => action}, socket) do
    action_atom = String.to_existing_atom(action)
    {:noreply, assign(socket, promotion_action: action_atom)}
  end

  @impl true
  def handle_event("promote", %{"id" => id} = params, socket) do
    action_str = Map.get(params, "promotion_action", "variation")
    action_atom = String.to_existing_atom(action_str)
    intent_name = Map.get(params, "promoted_to_intent", "")

    cond do
      intent_name == "" ->
        {:noreply, put_flash(socket, :error, "Intent name is required")}

      action_atom == :new_intent and Map.get(params, "new_intent_domain", "") == "" ->
        {:noreply, put_flash(socket, :error, "Domain is required for new intents")}

      true ->
        case IntentReviewQueue.approve(id, "Promoted via admin UI", action_atom, intent_name) do
          {:ok, candidate} ->
            Task.start(fn ->
              IntentPromoter.promote(candidate,
                domain: Map.get(params, "new_intent_domain", ""),
                category: Map.get(params, "new_intent_category", ""),
                speech_act: Map.get(params, "new_intent_speech_act", "")
              )
            end)

            {:noreply,
             assign(socket, show_promote_modal: false, selected_candidate: nil)
             |> put_flash(:info, "Promotion started - training will complete in background")
             |> refresh_data()}

          {:error, reason} ->
            Logger.warning("Failed to promote candidate", id: id, reason: inspect(reason))
            {:noreply, put_flash(socket, :error, "Failed to promote: #{inspect(reason)}")}
        end
    end
  end

  @impl true
  def handle_event("reject", %{"id" => id}, socket) do
    case IntentReviewQueue.reject(id) do
      {:ok, _} ->
        {:noreply, refresh_data(socket)}

      {:error, reason} ->
        Logger.warning("Failed to reject candidate", id: id, reason: inspect(reason))
        {:noreply, socket}
    end
  end

  @impl true
  def handle_event("defer", %{"id" => id}, socket) do
    case IntentReviewQueue.defer(id) do
      {:ok, _} ->
        {:noreply, refresh_data(socket)}

      {:error, reason} ->
        Logger.warning("Failed to defer candidate", id: id, reason: inspect(reason))
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
             :annotation_updated
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
    search_query = socket.assigns[:search_query] || ""

    candidates = load_candidates(current_tab)

    filtered_candidates =
      if search_query != "" do
        query_lower = String.downcase(search_query)

        Enum.filter(candidates, fn c ->
          String.contains?(String.downcase(c.text), query_lower) or
            (c.annotation[:notes] &&
               String.contains?(String.downcase(c.annotation[:notes]), query_lower))
        end)
      else
        candidates
      end

    socket
    |> assign(:candidates, filtered_candidates)
    |> assign(:stats, load_stats())
  end

  defp format_score(score) when is_float(score) do
    Float.round(score, 3)
  end

  defp format_score(_) do
    "N/A"
  end
end