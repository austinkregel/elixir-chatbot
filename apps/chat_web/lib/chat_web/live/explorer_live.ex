defmodule ChatWeb.ExplorerLive do
  @moduledoc """
  Unified data explorer for training world data.

  Shows all data types for the currently selected world:
  - Entities (promoted gazetteer entries and candidates)
  - Episodes (episodic memories)
  - Semantic Facts (consolidated knowledge)
  - Knowledge (learned facts and relationships)
  """

  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias World.Manager, as: WorldManager
  alias World.Metrics, as: WorldMetrics
  alias Brain.ML.Gazetteer
  alias Brain.Memory.Store, as: MemoryStore
  alias Brain.KnowledgeStore
  alias Brain.Epistemic.BeliefStore
  alias Brain.Epistemic.SourceAuthority
  alias Brain.Epistemic.JTMS
  alias Brain.Epistemic.UserModelStore
  alias Brain.FactDatabase

  @default_page_size 50

  @impl true
  def mount(_params, _session, socket) do
    {:ok, socket}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    world_id = socket.assigns.current_world_id

    # Load data for the current world
    socket = load_world_data(socket, world_id)

    # Apply URL params
    socket = apply_url_params(socket, params)

    {:noreply, socket}
  end

  defp load_world_data(socket, world_id) do
    # Load entities
    overlay = Gazetteer.get_world_overlay(world_id)
    overlay_by_type = group_overlay_by_type(overlay)
    entity_types = Map.keys(overlay_by_type) |> Enum.sort()

    # Load candidates
    candidates =
      try do
        WorldManager.get_candidates(world_id, sort: :confidence, limit: 1000)
      rescue
        _ -> []
      end

    # Load metrics
    metrics = get_world_metrics(world_id)

    # Load episodes
    episodes = load_world_episodes(world_id)

    # Load semantics
    semantics = load_world_semantics(world_id)

    # Load knowledge
    knowledge = load_world_knowledge(world_id)

    socket
    |> assign(:overlay, overlay)
    |> assign(:overlay_by_type, overlay_by_type)
    |> assign(:entity_types, entity_types)
    |> assign(:candidates, candidates)
    |> assign(:metrics, metrics)
    |> assign(:episodes, episodes)
    |> assign(:semantics, semantics)
    |> assign(:knowledge, knowledge)
    |> assign(:selected_type, List.first(entity_types))
    |> assign(:tab, :entities)
    |> assign(:page, 1)
    |> assign(:page_size, @default_page_size)
    |> assign(:search_query, "")
    |> assign(:expanded_id, nil)
    |> assign(:loading, nil)
    |> assign(:recently_promoted, MapSet.new())
    # Beliefs tab data (loaded lazily)
    |> assign(:beliefs_data, nil)
    |> assign(:beliefs_sub_tab, :beliefs)
    |> assign(:beliefs_source_filter, nil)
    |> assign(:beliefs_category_filter, nil)
    |> assign(:selected_user, nil)
    # Add belief form state
    |> assign(:show_add_belief_form, false)
    |> assign(:add_belief_form, %{"subject" => "", "predicate" => "", "object" => "", "confidence" => "100", "authority" => "mentor"})
    |> assign(:authority_filter, nil)
    # Inline confidence editing
    |> assign(:editing_confidence_id, nil)
  end

  defp apply_url_params(socket, params) do
    # Parse tab from URL
    tab =
      case params["tab"] do
        "candidates" -> :candidates
        "episodes" -> :episodes
        "semantics" -> :semantics
        "knowledge" -> :knowledge
        "beliefs" -> :beliefs
        _ -> :entities
      end

    # Parse type for entities view
    selected_type =
      case params["type"] do
        nil ->
          socket.assigns[:selected_type] || List.first(socket.assigns.entity_types)

        type ->
          if type in socket.assigns.entity_types,
            do: type,
            else: List.first(socket.assigns.entity_types)
      end

    # Parse page
    page = parse_int(params["page"], 1)

    # Parse search
    search_query = params["q"] || ""

    socket
    |> assign(:tab, tab)
    |> assign(:selected_type, selected_type)
    |> assign(:page, page)
    |> assign(:search_query, search_query)
    |> apply_filters()
  end

  defp apply_filters(socket) do
    tab = socket.assigns.tab
    search = socket.assigns.search_query |> String.downcase()
    page = socket.assigns.page
    page_size = socket.assigns.page_size

    {filtered, total} =
      case tab do
        :entities ->
          type = socket.assigns.selected_type
          entities = Map.get(socket.assigns.overlay_by_type, type, [])
          filtered = filter_by_search(entities, search, fn {key, _info} -> key end)
          {paginate(filtered, page, page_size), length(filtered)}

        :candidates ->
          filtered = filter_by_search(socket.assigns.candidates, search, & &1.value)
          {paginate(filtered, page, page_size), length(filtered)}

        :episodes ->
          filtered = filter_by_search(socket.assigns.episodes, search, & &1.state)
          {paginate(filtered, page, page_size), length(filtered)}

        :semantics ->
          filtered = filter_by_search(socket.assigns.semantics, search, & &1.representation)
          {paginate(filtered, page, page_size), length(filtered)}

        :knowledge ->
          # Knowledge is a map, don't paginate
          {socket.assigns.knowledge, map_size(socket.assigns.knowledge)}

        :beliefs ->
          # Load beliefs data lazily on first access
          socket = maybe_load_beliefs_data(socket)
          beliefs_data = socket.assigns.beliefs_data || %{}
          sub_tab = socket.assigns.beliefs_sub_tab

          case sub_tab do
            :beliefs ->
              items = Map.get(beliefs_data, :beliefs, [])
              source_filter = socket.assigns.beliefs_source_filter
              authority_filter = socket.assigns.authority_filter

              items =
                if source_filter do
                  Enum.filter(items, fn b -> to_string(b.source) == to_string(source_filter) end)
                else
                  items
                end

              items =
                if authority_filter do
                  Enum.filter(items, fn b ->
                    to_string(Map.get(b, :source_authority, "")) == to_string(authority_filter)
                  end)
                else
                  items
                end

              filtered = filter_by_search(items, search, fn b ->
                "#{b.subject} #{b.predicate} #{b.object}"
              end)

              {paginate(filtered, page, page_size), length(filtered)}

            :facts ->
              items = Map.get(beliefs_data, :facts, [])
              cat_filter = socket.assigns.beliefs_category_filter

              items =
                if cat_filter do
                  Enum.filter(items, fn f -> f.category == cat_filter end)
                else
                  items
                end

              filtered = filter_by_search(items, search, fn f ->
                "#{f.entity} #{f.fact}"
              end)

              {paginate(filtered, page, page_size), length(filtered)}

            _ ->
              {%{}, 0}
          end
      end

    total_pages = max(1, ceil(total / page_size))

    socket
    |> assign(:filtered_data, filtered)
    |> assign(:total_entries, total)
    |> assign(:total_pages, total_pages)
  end

  # ============================================================================
  # Event Handlers
  # ============================================================================

  @impl true
  def handle_event("switch_world", %{"world_id" => world_id}, socket) do
    # World context hook already updated current_world_id, reload data for new world
    {:noreply, reload_for_world(socket, world_id)}
  end

  def handle_event("refresh_worlds", _params, socket) do
    # World context hook already refreshed available_worlds
    {:noreply, socket}
  end

  def handle_event("switch_tab", %{"tab" => tab}, socket) do
    params =
      build_url_params(assign(socket, :tab, String.to_existing_atom(tab)) |> assign(:page, 1))

    {:noreply, push_patch(socket, to: ~p"/explorer?#{params}")}
  end

  def handle_event("select_type", %{"type" => type}, socket) do
    params = build_url_params(assign(socket, :selected_type, type) |> assign(:page, 1))
    {:noreply, push_patch(socket, to: ~p"/explorer?#{params}")}
  end

  def handle_event("search", %{"query" => query}, socket) do
    params = build_url_params(assign(socket, :search_query, query) |> assign(:page, 1))
    {:noreply, push_patch(socket, to: ~p"/explorer?#{params}")}
  end

  def handle_event("change_page", %{"page" => page}, socket) do
    params = build_url_params(assign(socket, :page, parse_int(page, 1)))
    {:noreply, push_patch(socket, to: ~p"/explorer?#{params}")}
  end

  def handle_event("toggle_expand", %{"id" => id}, socket) do
    expanded = if socket.assigns.expanded_id == id, do: nil, else: id
    {:noreply, assign(socket, :expanded_id, expanded)}
  end

  def handle_event("promote_candidate", %{"value" => value, "type" => type}, socket) do
    world_id = socket.assigns.current_world_id

    # Show loading state
    socket = assign(socket, :loading, value)

    # Perform promotion asynchronously
    Task.start(fn ->
      result =
        Gazetteer.add_to_world(world_id, value, type, %{
          source: :candidate_promotion,
          promoted_at: DateTime.utc_now()
        })

      send(self(), {:promotion_complete, value, type, result})
    end)

    {:noreply, socket}
  end

  def handle_event("refresh", _params, socket) do
    world_id = socket.assigns.current_world_id
    socket = load_world_data(socket, world_id) |> apply_filters()
    {:noreply, socket}
  end

  def handle_event("beliefs_sub_tab", %{"sub_tab" => sub_tab}, socket) do
    sub = String.to_existing_atom(sub_tab)

    socket =
      socket
      |> assign(:beliefs_sub_tab, sub)
      |> assign(:page, 1)
      |> assign(:search_query, "")
      |> apply_filters()

    {:noreply, socket}
  end

  def handle_event("beliefs_source_filter", %{"source" => source}, socket) do
    current = socket.assigns.beliefs_source_filter
    new_filter = if current == source, do: nil, else: source

    socket =
      socket
      |> assign(:beliefs_source_filter, new_filter)
      |> assign(:page, 1)
      |> apply_filters()

    {:noreply, socket}
  end

  def handle_event("beliefs_category_filter", %{"category" => category}, socket) do
    current = socket.assigns.beliefs_category_filter
    new_filter = if current == category, do: nil, else: category

    socket =
      socket
      |> assign(:beliefs_category_filter, new_filter)
      |> assign(:page, 1)
      |> apply_filters()

    {:noreply, socket}
  end

  def handle_event("select_user", %{"user_id" => user_id}, socket) do
    current = socket.assigns.selected_user
    new_selected = if current == user_id, do: nil, else: user_id
    {:noreply, assign(socket, :selected_user, new_selected)}
  end

  def handle_event("refresh_beliefs", _params, socket) do
    socket =
      socket
      |> assign(:beliefs_data, nil)
      |> maybe_load_beliefs_data()
      |> apply_filters()

    {:noreply, socket}
  end

  # ---- Belief management actions ----

  def handle_event("toggle_add_belief_form", _params, socket) do
    {:noreply, assign(socket, :show_add_belief_form, !socket.assigns.show_add_belief_form)}
  end

  def handle_event("update_add_belief_form", %{"belief" => params}, socket) do
    {:noreply, assign(socket, :add_belief_form, params)}
  end

  def handle_event("add_guided_belief", %{"belief" => params}, socket) do
    subject = params["subject"] |> String.trim()
    predicate = params["predicate"] |> String.trim()
    object = params["object"] |> String.trim()
    authority = params["authority"] |> String.trim()

    if subject != "" and predicate != "" and object != "" do
      predicate_atom =
        try do
          String.to_existing_atom(predicate)
        rescue
          _ -> String.to_atom(predicate)
        end

      authority_atom =
        try do
          String.to_existing_atom(authority)
        rescue
          _ -> String.to_atom(authority)
        end

      case BeliefStore.add_belief_with_authority(
             normalize_subject(subject),
             predicate_atom,
             object,
             authority_atom
           ) do
        {:ok, _id} ->
          socket =
            socket
            |> assign(:show_add_belief_form, false)
            |> assign(:add_belief_form, %{"subject" => "", "predicate" => "", "object" => "", "confidence" => "100", "authority" => "mentor"})
            |> assign(:beliefs_data, nil)
            |> maybe_load_beliefs_data()
            |> apply_filters()
            |> put_flash(:info, "Guided belief added (#{authority})")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add belief: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Subject, predicate, and object are all required")}
    end
  end

  def handle_event("retract_belief", %{"id" => belief_id}, socket) do
    case BeliefStore.retract_belief(belief_id) do
      :ok ->
        socket =
          socket
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()
          |> put_flash(:info, "Belief retracted")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to retract: #{inspect(reason)}")}
    end
  end

  def handle_event("confirm_belief", %{"id" => belief_id}, socket) do
    case BeliefStore.confirm_belief(belief_id) do
      {:ok, _updated} ->
        socket =
          socket
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()
          |> put_flash(:info, "Belief confirmed")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to confirm: #{inspect(reason)}")}
    end
  end

  def handle_event("edit_confidence", %{"id" => belief_id}, socket) do
    {:noreply, assign(socket, :editing_confidence_id, belief_id)}
  end

  def handle_event("save_confidence", %{"belief_id" => belief_id, "confidence" => conf_str}, socket) do
    confidence = parse_confidence(conf_str)

    case BeliefStore.update_confidence(belief_id, confidence) do
      {:ok, _updated} ->
        socket =
          socket
          |> assign(:editing_confidence_id, nil)
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to update confidence: #{inspect(reason)}")}
    end
  end

  def handle_event("cancel_edit_confidence", _params, socket) do
    {:noreply, assign(socket, :editing_confidence_id, nil)}
  end

  def handle_event("authority_filter", %{"authority" => authority}, socket) do
    current = socket.assigns.authority_filter
    new_filter = if current == authority, do: nil, else: authority

    socket =
      socket
      |> assign(:authority_filter, new_filter)
      |> assign(:page, 1)
      |> apply_filters()

    {:noreply, socket}
  end

  defp parse_confidence(str) when is_binary(str) do
    case Float.parse(str) do
      {val, _} -> min(max(val / 100.0, 0.0), 1.0)
      :error -> 1.0
    end
  end

  defp parse_confidence(_), do: 1.0

  defp normalize_subject("user"), do: :user
  defp normalize_subject("world"), do: :world
  defp normalize_subject("self"), do: :self
  defp normalize_subject(other), do: other

  defp reload_for_world(socket, world_id) do
    load_world_data(socket, world_id) |> apply_filters()
  end

  @impl true
  def handle_info({:promotion_complete, value, type, result}, socket) do
    world_id = socket.assigns.current_world_id

    socket =
      case result do
        :ok ->
          recently_promoted = MapSet.put(socket.assigns.recently_promoted, value)
          Process.send_after(self(), {:clear_recently_promoted, value}, 2000)

          socket
          |> load_world_data(world_id)
          |> apply_filters()
          |> assign(:recently_promoted, recently_promoted)
          |> put_flash(:info, "Promoted \"#{value}\" as #{type}")

        {:error, reason} ->
          put_flash(socket, :error, "Failed to promote: #{inspect(reason)}")
      end

    {:noreply, assign(socket, :loading, nil)}
  end

  @impl true
  def handle_info({:clear_recently_promoted, value}, socket) do
    recently_promoted = MapSet.delete(socket.assigns.recently_promoted, value)
    {:noreply, assign(socket, :recently_promoted, recently_promoted)}
  end

  def handle_info({:world_context_changed, world_id}, socket) do
    # World was changed from another LiveView or tab
    {:noreply, reload_for_world(socket, world_id)}
  end

  # ============================================================================
  # Render
  # ============================================================================

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
            <h1 class="text-xl font-bold">Data Explorer</h1>
            <p class="text-sm text-base-content/60">
              Explore data in world: <span class="font-medium text-primary">{@current_world_id}</span>
            </p>
          </div>
          <button phx-click="refresh" class="btn btn-ghost btn-sm">
            <.icon name="hero-arrow-path" class="size-4" /> Refresh
          </button>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6 space-y-6">
        <!-- Stats Summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <.stat_card label="Entities" value={length(@overlay)} />
          <.stat_card label="Candidates" value={length(@candidates)} />
          <.stat_card label="Episodes" value={length(@episodes)} />
          <.stat_card label="Semantics" value={length(@semantics)} />
          <.stat_card label="Knowledge" value={map_size(@knowledge)} />
          <.stat_card
            label="Beliefs"
            value={if @beliefs_data, do: length(@beliefs_data[:beliefs] || []), else: "-"}
          />
        </div>

    <!-- Tabs -->
        <div class="flex flex-wrap items-center gap-4">
          <div class="tabs tabs-boxed">
            <.tab_button
              tab={:entities}
              current={@tab}
              count={length(@overlay)}
              icon="hero-tag"
              label="Entities"
            />
            <.tab_button
              tab={:candidates}
              current={@tab}
              count={length(@candidates)}
              icon="hero-queue-list"
              label="Candidates"
            />
            <.tab_button
              tab={:episodes}
              current={@tab}
              count={length(@episodes)}
              icon="hero-clock"
              label="Episodes"
            />
            <.tab_button
              tab={:semantics}
              current={@tab}
              count={length(@semantics)}
              icon="hero-light-bulb"
              label="Semantics"
            />
            <.tab_button
              tab={:knowledge}
              current={@tab}
              count={map_size(@knowledge)}
              icon="hero-book-open"
              label="Knowledge"
            />
            <.tab_button
              tab={:beliefs}
              current={@tab}
              count={if @beliefs_data, do: length(@beliefs_data[:beliefs] || []), else: 0}
              icon="hero-eye"
              label="Beliefs"
            />
          </div>

    <!-- Search -->
          <div class="flex-1 max-w-md">
            <input
              type="text"
              placeholder="Search..."
              value={@search_query}
              phx-keyup="search"
              name="query"
              phx-debounce="150"
              class="input input-sm input-bordered w-full"
            />
          </div>
        </div>

    <!-- Type Selector (for entities tab) -->
        <%= if @tab == :entities and length(@entity_types) > 0 do %>
          <div class="flex flex-wrap gap-2">
            <%= for type <- @entity_types do %>
              <button
                phx-click="select_type"
                phx-value-type={type}
                class={[
                  "btn btn-sm",
                  if(type == @selected_type, do: "btn-primary", else: "btn-ghost")
                ]}
              >
                {type}
                <span class="badge badge-xs">{length(Map.get(@overlay_by_type, type, []))}</span>
              </button>
            <% end %>
          </div>
        <% end %>

    <!-- Recently Promoted Toast -->
        <%= if MapSet.size(@recently_promoted) > 0 do %>
          <div class="bg-success/10 border border-success/30 rounded-xl px-4 py-3 flex items-center gap-3 animate-fade-in">
            <.icon name="hero-check" class="size-5 text-success" />
            <span class="text-sm">Just promoted:</span>
            <div class="flex flex-wrap gap-1">
              <%= for value <- MapSet.to_list(@recently_promoted) do %>
                <span class="badge badge-success badge-sm">{value}</span>
              <% end %>
            </div>
          </div>
        <% end %>

    <!-- Content -->
        <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
          <%= case @tab do %>
            <% :entities -> %>
              <.entities_table
                data={@filtered_data}
                loading={@loading}
                page={@page}
                total_pages={@total_pages}
                total_entries={@total_entries}
                page_size={@page_size}
              />
            <% :candidates -> %>
              <.candidates_table
                data={@filtered_data}
                loading={@loading}
                page={@page}
                total_pages={@total_pages}
                total_entries={@total_entries}
                page_size={@page_size}
              />
            <% :episodes -> %>
              <.episodes_list
                data={@filtered_data}
                expanded_id={@expanded_id}
                page={@page}
                total_pages={@total_pages}
                total_entries={@total_entries}
                page_size={@page_size}
              />
            <% :semantics -> %>
              <.semantics_list
                data={@filtered_data}
                expanded_id={@expanded_id}
                page={@page}
                total_pages={@total_pages}
                total_entries={@total_entries}
                page_size={@page_size}
              />
            <% :knowledge -> %>
              <.knowledge_view data={@filtered_data} />
            <% :beliefs -> %>
              <.beliefs_view
                beliefs_data={@beliefs_data || %{}}
                sub_tab={@beliefs_sub_tab}
                filtered_data={@filtered_data}
                source_filter={@beliefs_source_filter}
                category_filter={@beliefs_category_filter}
                selected_user={@selected_user}
                show_add_belief_form={@show_add_belief_form}
                add_belief_form={@add_belief_form}
                editing_confidence_id={@editing_confidence_id}
                expanded_id={@expanded_id}
                page={@page}
                total_pages={@total_pages}
                total_entries={@total_entries}
                page_size={@page_size}
              />
          <% end %>
        </div>
      </div>
    </.app_shell>
    """
  end

  # ============================================================================
  # Sub-components
  # ============================================================================

  defp stat_card(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
      <div class="text-2xl font-bold">{@value}</div>
      <div class="text-xs text-base-content/60">{@label}</div>
    </div>
    """
  end

  defp tab_button(assigns) do
    ~H"""
    <button
      phx-click="switch_tab"
      phx-value-tab={@tab}
      class={["tab gap-1", if(@tab == @current, do: "tab-active", else: "")]}
    >
      <.icon name={@icon} class="size-4" />
      <span class="hidden sm:inline">{@label}</span>
      <span class="badge badge-xs">{@count}</span>
    </button>
    """
  end

  defp entities_table(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_state icon="hero-tag" message="No entities found" />
    <% else %>
      <table class="table table-sm">
        <thead class="bg-base-200/50">
          <tr>
            <th>Lookup Key</th>
            <th>Canonical Value</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          <%= for {key, info} <- @data do %>
            <tr class="hover:bg-base-200/30">
              <td class="font-medium">{key}</td>
              <td>
                <%= if info[:value] && info[:value] != key do %>
                  {info[:value]}
                <% else %>
                  <span class="text-base-content/40">—</span>
                <% end %>
              </td>
              <td>
                <span class="badge badge-sm badge-ghost">{info[:source] || "unknown"}</span>
              </td>
            </tr>
          <% end %>
        </tbody>
      </table>
      <.pagination
        page={@page}
        total_pages={@total_pages}
        total_entries={@total_entries}
        page_size={@page_size}
      />
    <% end %>
    """
  end

  defp candidates_table(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_state icon="hero-queue-list" message="No candidates found" />
    <% else %>
      <table class="table table-sm">
        <thead class="bg-base-200/50">
          <tr>
            <th>Value</th>
            <th>Inferred Type</th>
            <th>Confidence</th>
            <th>Occurrences</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          <%= for candidate <- @data do %>
            <% is_loading = @loading == candidate.value %>
            <tr class="hover:bg-base-200/30 group">
              <td class="font-medium">
                <div class="flex items-center gap-2">
                  <%= if is_loading do %>
                    <span class="loading loading-spinner loading-xs text-success"></span>
                  <% end %>
                  {candidate.value}
                </div>
              </td>
              <td>
                <span class="badge badge-sm badge-ghost">{candidate.inferred_type || "unknown"}</span>
              </td>
              <td>
                <span class={confidence_color(candidate.confidence)}>
                  {format_confidence(candidate.confidence)}
                </span>
              </td>
              <td>{candidate[:occurrences] || 1}</td>
              <td>
                <%= if candidate.inferred_type && candidate.inferred_type != "unknown" do %>
                  <button
                    phx-click="promote_candidate"
                    phx-value-value={candidate.value}
                    phx-value-type={candidate.inferred_type}
                    disabled={is_loading}
                    class="btn btn-ghost btn-xs text-success opacity-0 group-hover:opacity-100"
                    title="Promote to gazetteer"
                  >
                    <.icon name="hero-arrow-up-circle" class="size-4" />
                  </button>
                <% end %>
              </td>
            </tr>
          <% end %>
        </tbody>
      </table>
      <.pagination
        page={@page}
        total_pages={@total_pages}
        total_entries={@total_entries}
        page_size={@page_size}
      />
    <% end %>
    """
  end

  defp episodes_list(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_state icon="hero-clock" message="No episodes found" />
    <% else %>
      <div class="divide-y divide-base-300/50">
        <%= for episode <- @data do %>
          <div class="p-4 hover:bg-base-200/50 transition-colors">
            <div
              class="flex items-start justify-between cursor-pointer"
              phx-click="toggle_expand"
              phx-value-id={episode.id}
            >
              <div class="flex-1 min-w-0">
                <div class="font-medium text-sm truncate">{episode.state}</div>
                <div class="text-xs text-base-content/60 mt-1">
                  Action: {episode.action}
                </div>
              </div>
              <div class="flex items-center gap-2 ml-4">
                <div class="flex flex-wrap gap-1">
                  <%= for tag <- Enum.take(episode.tags, 3) do %>
                    <span class="badge badge-xs badge-ghost">{tag}</span>
                  <% end %>
                </div>
                <.icon
                  name={
                    if @expanded_id == episode.id, do: "hero-chevron-up", else: "hero-chevron-down"
                  }
                  class="size-4 text-base-content/40"
                />
              </div>
            </div>
            <%= if @expanded_id == episode.id do %>
              <div class="mt-4 pt-4 border-t border-base-300/50 text-sm space-y-2">
                <div>
                  <span class="text-base-content/60">ID:</span>
                  <span class="font-mono text-xs">{episode.id}</span>
                </div>
                <%= if episode.outcome && episode.outcome != "" do %>
                  <div>
                    <span class="text-base-content/60">Outcome:</span>
                    <p class="mt-1 bg-base-200 rounded-lg p-2">{episode.outcome}</p>
                  </div>
                <% end %>
                <div class="flex flex-wrap gap-1">
                  <%= for tag <- episode.tags do %>
                    <span class="badge badge-sm badge-ghost">{tag}</span>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      <.pagination
        page={@page}
        total_pages={@total_pages}
        total_entries={@total_entries}
        page_size={@page_size}
      />
    <% end %>
    """
  end

  defp semantics_list(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_state icon="hero-light-bulb" message="No semantic facts found" />
    <% else %>
      <div class="divide-y divide-base-300/50">
        <%= for semantic <- @data do %>
          <div class="p-4 hover:bg-base-200/50 transition-colors">
            <div
              class="flex items-start justify-between cursor-pointer"
              phx-click="toggle_expand"
              phx-value-id={semantic.id}
            >
              <div class="flex-1 min-w-0">
                <div class="font-medium text-sm">{semantic.representation}</div>
                <div class="text-xs text-base-content/60 mt-1">
                  Evidence: {length(semantic.evidence_ids)} episodes
                </div>
              </div>
              <.icon
                name={
                  if @expanded_id == semantic.id, do: "hero-chevron-up", else: "hero-chevron-down"
                }
                class="size-4 text-base-content/40 ml-4"
              />
            </div>
            <%= if @expanded_id == semantic.id do %>
              <div class="mt-4 pt-4 border-t border-base-300/50 text-sm space-y-2">
                <div>
                  <span class="text-base-content/60">ID:</span>
                  <span class="font-mono text-xs">{semantic.id}</span>
                </div>
                <div class="flex flex-wrap gap-1">
                  <%= for ep_id <- semantic.evidence_ids do %>
                    <span class="badge badge-sm badge-ghost font-mono text-xs">
                      {String.slice(ep_id, 0, 8)}...
                    </span>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      <.pagination
        page={@page}
        total_pages={@total_pages}
        total_entries={@total_entries}
        page_size={@page_size}
      />
    <% end %>
    """
  end

  defp knowledge_view(assigns) do
    ~H"""
    <%= if map_size(@data) == 0 do %>
      <.empty_state icon="hero-book-open" message="No knowledge stored" />
    <% else %>
      <div class="divide-y divide-base-300/50">
        <%= for {category, items} <- @data do %>
          <div class="p-4">
            <h3 class="font-semibold text-sm flex items-center gap-2 mb-3">
              <.icon name="hero-folder" class="size-4 text-primary" />
              {category}
              <span class="badge badge-sm badge-primary/20 text-primary">
                {if is_map(items), do: map_size(items), else: length(items)} items
              </span>
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              <%= if is_map(items) do %>
                <%= for {key, value} <- Enum.take(items, 12) do %>
                  <div class="bg-base-200/50 rounded-lg px-3 py-2 text-sm">
                    <span class="font-medium">{key}:</span>
                    <span class="text-base-content/60 ml-1">{format_value(value)}</span>
                  </div>
                <% end %>
                <%= if map_size(items) > 12 do %>
                  <div class="text-xs text-base-content/40 px-3 py-2">
                    ... and {map_size(items) - 12} more
                  </div>
                <% end %>
              <% else %>
                <%= for item <- Enum.take(items, 12) do %>
                  <div class="bg-base-200/50 rounded-lg px-3 py-2 text-sm">{format_value(item)}</div>
                <% end %>
              <% end %>
            </div>
          </div>
        <% end %>
      </div>
    <% end %>
    """
  end

  # ============================================================================
  # Beliefs Tab Components
  # ============================================================================

  defp beliefs_view(assigns) do
    authority_profiles =
      try do
        SourceAuthority.list_profiles()
      rescue
        _ -> []
      end

    assigns = assign(assigns, :authority_profiles, authority_profiles)

    ~H"""
    <div class="divide-y divide-base-300/50">
      <!-- Sub-tab navigation -->
      <div class="p-4 bg-base-200/30">
        <div class="flex flex-wrap gap-2">
          <%= for {sub, label, icon} <- [
            {:beliefs, "Beliefs", "hero-eye"},
            {:facts, "Facts", "hero-book-open"},
            {:jtms, "JTMS Graph", "hero-share"},
            {:users, "User Models", "hero-user-group"}
          ] do %>
            <button
              phx-click="beliefs_sub_tab"
              phx-value-sub_tab={sub}
              class={[
                "btn btn-sm gap-1",
                if(sub == @sub_tab, do: "btn-primary", else: "btn-ghost")
              ]}
            >
              <.icon name={icon} class="size-4" />
              {label}
              <%= case sub do %>
                <% :beliefs -> %>
                  <span class="badge badge-xs">{length(@beliefs_data[:beliefs] || [])}</span>
                <% :facts -> %>
                  <span class="badge badge-xs">{length(@beliefs_data[:facts] || [])}</span>
                <% :jtms -> %>
                  <span class="badge badge-xs">
                    {Map.get(@beliefs_data[:jtms_stats] || %{}, :total_nodes, 0)}
                  </span>
                <% :users -> %>
                  <span class="badge badge-xs">{length(@beliefs_data[:user_ids] || [])}</span>
              <% end %>
            </button>
          <% end %>

          <div class="flex items-center gap-2 ml-auto">
            <%= if @sub_tab == :beliefs do %>
              <button phx-click="toggle_add_belief_form" class="btn btn-primary btn-sm gap-1" title="Add guided belief">
                <.icon name="hero-plus" class="size-4" />
                Add Belief
              </button>
            <% end %>
            <button phx-click="refresh_beliefs" class="btn btn-ghost btn-sm btn-square" title="Refresh beliefs data">
              <.icon name="hero-arrow-path" class="size-4" />
            </button>
          </div>
        </div>
      </div>

      <!-- Authority Credibility Overview (only when beliefs sub-tab) -->
      <%= if @sub_tab == :beliefs and length(@authority_profiles) > 0 do %>
        <% active_profiles = Enum.filter(@authority_profiles, fn p -> p.total_added > 0 end) %>
        <%= if length(active_profiles) > 0 do %>
          <div class="p-4 bg-base-200/20">
            <div class="text-xs font-semibold text-base-content/70 mb-2 flex items-center gap-1">
              <.icon name="hero-shield-check" class="size-4" />
              Authority Credibility
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
              <%= for p <- active_profiles do %>
                <div class="bg-base-100 rounded-lg p-2 border border-base-300/50">
                  <div class="flex items-center justify-between mb-1">
                    <span class={["badge badge-xs", authority_badge_class(p.profile.category)]}>
                      {p.profile.label}
                    </span>
                    <span class="text-[10px] text-base-content/50">{p.total_added} beliefs</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <div class="flex-1 bg-base-300 rounded-full h-1.5">
                      <div
                        class={[
                          "h-1.5 rounded-full",
                          cond do
                            p.credibility >= 0.7 -> "bg-success"
                            p.credibility >= 0.5 -> "bg-warning"
                            true -> "bg-error"
                          end
                        ]}
                        style={"width: #{Float.round(p.credibility * 100, 1)}%"}
                      >
                      </div>
                    </div>
                    <span class="text-[10px] font-mono">{Float.round(p.credibility * 100, 0)}%</span>
                  </div>
                  <div class="flex items-center gap-2 mt-1 text-[10px] text-base-content/50">
                    <span class="text-success">{p.confirmed_count} confirmed</span>
                    <span class="text-error">{p.contradicted_count} contradicted</span>
                  </div>
                </div>
              <% end %>
            </div>
          </div>
        <% end %>
      <% end %>

      <!-- Add belief form (collapsible) -->
      <%= if @show_add_belief_form do %>
        <div class="p-4 bg-primary/5 border-b border-primary/20">
          <form phx-submit="add_guided_belief" phx-change="update_add_belief_form" class="space-y-3">
            <div class="flex items-center gap-2 mb-2">
              <.icon name="hero-light-bulb" class="size-5 text-primary" />
              <span class="text-sm font-semibold">Add Guided Belief</span>
              <span class="text-xs text-base-content/50 ml-2">Confidence set by authority tier</span>
            </div>
            <div class="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div>
                <label class="label text-xs">Subject</label>
                <select name="belief[subject]" class="select select-bordered select-sm w-full">
                  <option value="world" selected={@add_belief_form["subject"] == "world"}>world</option>
                  <option value="self" selected={@add_belief_form["subject"] == "self"}>self</option>
                  <option value="user" selected={@add_belief_form["subject"] == "user"}>user</option>
                </select>
              </div>
              <div>
                <label class="label text-xs">Predicate</label>
                <input
                  type="text"
                  name="belief[predicate]"
                  value={@add_belief_form["predicate"]}
                  placeholder="e.g. name, likes, location"
                  class="input input-bordered input-sm w-full"
                  required
                />
              </div>
              <div>
                <label class="label text-xs">Object</label>
                <input
                  type="text"
                  name="belief[object]"
                  value={@add_belief_form["object"]}
                  placeholder="The value"
                  class="input input-bordered input-sm w-full"
                  required
                />
              </div>
              <div>
                <label class="label text-xs">Authority</label>
                <select name="belief[authority]" class="select select-bordered select-sm w-full">
                  <%= for {category, profiles} <- group_authority_profiles(@authority_profiles) do %>
                    <optgroup label={String.capitalize(category)}>
                      <%= for p <- profiles do %>
                        <option value={p.key} selected={to_string(p.key) == @add_belief_form["authority"]}>
                          {p.profile.label} ({Float.round(p.profile.initial_confidence * 100, 0)}%)
                        </option>
                      <% end %>
                    </optgroup>
                  <% end %>
                </select>
              </div>
            </div>
            <div class="flex items-center gap-4">
              <div class="text-xs text-base-content/50">
                Confidence will be based on authority tier and tracked credibility
              </div>
              <div class="flex gap-2 ml-auto">
                <button type="button" phx-click="toggle_add_belief_form" class="btn btn-ghost btn-sm">Cancel</button>
                <button type="submit" class="btn btn-primary btn-sm">Add Belief</button>
              </div>
            </div>
          </form>
        </div>
      <% end %>

      <!-- Sub-tab content -->
      <%= case @sub_tab do %>
        <% :beliefs -> %>
          <.beliefs_sub_view
            data={@filtered_data}
            sources={@beliefs_data[:belief_sources] || []}
            source_filter={@source_filter}
            authority_filter={assigns[:authority_filter]}
            authority_types={@beliefs_data[:belief_authorities] || []}
            editing_confidence_id={@editing_confidence_id}
            page={@page}
            total_pages={@total_pages}
            total_entries={@total_entries}
            page_size={@page_size}
          />
        <% :facts -> %>
          <.facts_sub_view
            data={@filtered_data}
            categories={@beliefs_data[:fact_categories] || []}
            category_filter={@category_filter}
            page={@page}
            total_pages={@total_pages}
            total_entries={@total_entries}
            page_size={@page_size}
          />
        <% :jtms -> %>
          <.jtms_sub_view
            stats={@beliefs_data[:jtms_stats] || %{}}
            contradictions={@beliefs_data[:jtms_contradictions] || []}
            expanded_id={@expanded_id}
          />
        <% :users -> %>
          <.users_sub_view
            user_ids={@beliefs_data[:user_ids] || []}
            selected_user={@selected_user}
          />
      <% end %>
    </div>
    """
  end

  defp beliefs_sub_view(assigns) do
    ~H"""
    <!-- Source + Authority filter buttons -->
    <div class="px-4 pt-3 flex flex-wrap gap-2">
      <%= if length(@sources) > 0 do %>
        <%= for source <- @sources do %>
          <button
            phx-click="beliefs_source_filter"
            phx-value-source={source}
            class={[
              "btn btn-xs",
              if(to_string(@source_filter) == to_string(source), do: "btn-primary", else: "btn-ghost")
            ]}
          >
            {source}
          </button>
        <% end %>
      <% end %>
      <%= if length(@authority_types) > 0 do %>
        <span class="text-base-content/30 self-center">|</span>
        <%= for auth <- @authority_types do %>
          <button
            phx-click="authority_filter"
            phx-value-authority={auth}
            class={[
              "btn btn-xs",
              if(to_string(@authority_filter) == to_string(auth), do: "btn-secondary", else: "btn-ghost")
            ]}
          >
            {auth}
          </button>
        <% end %>
      <% end %>
    </div>

    <%= if is_list(@data) and length(@data) == 0 do %>
      <.empty_state icon="hero-eye" message="No beliefs found" />
    <% else %>
      <%= if is_list(@data) do %>
        <div class="divide-y divide-base-300/50">
          <%= for belief <- @data do %>
            <% b_conf = belief.confidence || 0.0
            b_authority = Map.get(belief, :source_authority) %>
            <div class={[
              "p-4 hover:bg-base-200/30 transition-colors",
              if(b_authority, do: "border-l-2 border-primary/40", else: "")
            ]}>
              <div class="flex items-start justify-between gap-4">
                <div class="flex-1 min-w-0">
                  <!-- Subject / Predicate / Object -->
                  <div class="flex items-center gap-1.5 text-sm font-medium">
                    <span class="text-primary font-mono">{belief.subject}</span>
                    <span class="text-base-content/40">/</span>
                    <span class="text-base-content/70 font-mono">{belief.predicate}</span>
                    <span class="text-base-content/40">/</span>
                    <span class="font-mono">{inspect(belief.object)}</span>
                  </div>

                  <!-- Confidence bar or editor -->
                  <%= if @editing_confidence_id == belief.id do %>
                    <form phx-submit="save_confidence" class="flex items-center gap-2 mt-2 max-w-xs">
                      <input type="hidden" name="belief_id" value={belief.id} />
                      <input
                        type="range"
                        name="confidence"
                        min="0"
                        max="100"
                        value={round(b_conf * 100)}
                        class="range range-xs range-primary flex-1"
                      />
                      <span class="text-xs font-mono w-10 text-right">{round(b_conf * 100)}%</span>
                      <button type="submit" class="btn btn-xs btn-success btn-square" title="Save">
                        <.icon name="hero-check" class="size-3" />
                      </button>
                      <button type="button" phx-click="cancel_edit_confidence" class="btn btn-xs btn-ghost btn-square" title="Cancel">
                        <.icon name="hero-x-mark" class="size-3" />
                      </button>
                    </form>
                  <% else %>
                    <div class="flex items-center gap-2 mt-2 max-w-xs">
                      <div class="flex-1 bg-base-300 rounded-full h-1.5">
                        <div
                          class={[
                            "h-1.5 rounded-full",
                            cond do
                              b_conf >= 0.8 -> "bg-success"
                              b_conf >= 0.5 -> "bg-warning"
                              true -> "bg-error"
                            end
                          ]}
                          style={"width: #{Float.round(b_conf * 100, 1)}%"}
                        >
                        </div>
                      </div>
                      <span class="text-xs font-mono text-base-content/60">
                        {Float.round(b_conf * 100, 1)}%
                      </span>
                    </div>
                  <% end %>
                </div>

                <!-- Badges + Actions -->
                <div class="flex items-center gap-2 shrink-0">
                  <%= if b_authority do %>
                    <span class={["badge badge-sm", authority_badge_class(authority_category(b_authority))]}>
                      {b_authority}
                    </span>
                  <% end %>
                  <%= if belief.source do %>
                    <span class={[
                      "badge badge-sm",
                      case belief.source do
                        :explicit -> "badge-primary"
                        :learned -> "badge-info"
                        :inferred -> "badge-warning"
                        :assumed -> "badge-ghost"
                        _ -> "badge-ghost"
                      end
                    ]}>
                      {belief.source}
                    </span>
                  <% end %>
                  <%= if belief.node_id do %>
                    <span class="badge badge-sm badge-secondary badge-outline">JTMS</span>
                  <% end %>

                  <!-- Action buttons -->
                  <div class="flex items-center gap-1 ml-1">
                    <button
                      phx-click="confirm_belief"
                      phx-value-id={belief.id}
                      class="btn btn-xs btn-ghost btn-square text-success"
                      title="Confirm (boost confidence +10%)"
                    >
                      <.icon name="hero-check-circle" class="size-4" />
                    </button>
                    <button
                      phx-click="edit_confidence"
                      phx-value-id={belief.id}
                      class="btn btn-xs btn-ghost btn-square"
                      title="Adjust confidence"
                    >
                      <.icon name="hero-adjustments-horizontal" class="size-4" />
                    </button>
                    <button
                      phx-click="retract_belief"
                      phx-value-id={belief.id}
                      data-confirm="Retract this belief?"
                      class="btn btn-xs btn-ghost btn-square text-error"
                      title="Retract belief"
                    >
                      <.icon name="hero-x-circle" class="size-4" />
                    </button>
                  </div>
                </div>
              </div>

              <!-- Metadata row -->
              <div class="flex items-center gap-4 mt-2 text-xs text-base-content/50">
                <%= if belief.user_id do %>
                  <span>User: {String.slice(belief.user_id, 0, 12)}...</span>
                <% end %>
                <%= if belief.created_at do %>
                  <span>Created: {format_datetime(belief.created_at)}</span>
                <% end %>
                <span class="font-mono text-[10px]">{String.slice(belief.id || "", 0, 8)}</span>
              </div>
            </div>
          <% end %>
        </div>
        <.pagination
          page={@page}
          total_pages={@total_pages}
          total_entries={@total_entries}
          page_size={@page_size}
        />
      <% end %>
    <% end %>
    """
  end

  defp facts_sub_view(assigns) do
    ~H"""
    <!-- Category filter buttons -->
    <%= if length(@categories) > 0 do %>
      <div class="px-4 pt-3 flex flex-wrap gap-2">
        <%= for category <- @categories do %>
          <button
            phx-click="beliefs_category_filter"
            phx-value-category={category}
            class={[
              "btn btn-xs",
              if(@category_filter == category, do: "btn-primary", else: "btn-ghost")
            ]}
          >
            {category}
          </button>
        <% end %>
      </div>
    <% end %>

    <%= if is_list(@data) and length(@data) == 0 do %>
      <.empty_state icon="hero-book-open" message="No facts found" />
    <% else %>
      <%= if is_list(@data) do %>
        <table class="table table-sm">
          <thead class="bg-base-200/50">
            <tr>
              <th>Entity</th>
              <th>Fact</th>
              <th>Category</th>
              <th>Confidence</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            <%= for fact <- @data do %>
              <tr class="hover:bg-base-200/30">
                <td class="font-medium text-primary font-mono">{fact.entity}</td>
                <td class="max-w-md">
                  <div class="truncate">{fact.fact}</div>
                </td>
                <td>
                  <span class="badge badge-sm badge-ghost">{fact.category}</span>
                </td>
                <td>
                  <span class={confidence_color(fact.confidence)}>
                    {format_confidence(fact.confidence)}
                  </span>
                </td>
                <td class="text-xs text-base-content/50">
                  {fact.verification_source || "-"}
                </td>
              </tr>
            <% end %>
          </tbody>
        </table>
        <.pagination
          page={@page}
          total_pages={@total_pages}
          total_entries={@total_entries}
          page_size={@page_size}
        />
      <% end %>
    <% end %>
    """
  end

  defp jtms_sub_view(assigns) do
    ~H"""
    <div class="p-4 space-y-4">
      <!-- JTMS Stats -->
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div class="bg-base-200/50 rounded-lg p-3 text-center">
          <div class="text-xl font-bold">{Map.get(@stats, :total_nodes, 0)}</div>
          <div class="text-xs text-base-content/60">Total Nodes</div>
        </div>
        <div class="bg-success/10 rounded-lg p-3 text-center">
          <div class="text-xl font-bold text-success">{Map.get(@stats, :in_count, 0)}</div>
          <div class="text-xs text-base-content/60">IN</div>
        </div>
        <div class="bg-base-200/50 rounded-lg p-3 text-center">
          <div class="text-xl font-bold text-base-content/50">{Map.get(@stats, :out_count, 0)}</div>
          <div class="text-xs text-base-content/60">OUT</div>
        </div>
        <div class="bg-error/10 rounded-lg p-3 text-center">
          <div class="text-xl font-bold text-error">{Map.get(@stats, :contradiction_count, 0)}</div>
          <div class="text-xs text-base-content/60">Contradictions</div>
        </div>
        <div class="bg-info/10 rounded-lg p-3 text-center">
          <div class="text-xl font-bold text-info">{Map.get(@stats, :justification_count, 0)}</div>
          <div class="text-xs text-base-content/60">Justifications</div>
        </div>
      </div>

      <!-- Contradictions list -->
      <div>
        <h3 class="font-semibold text-sm mb-2 flex items-center gap-2">
          <.icon name="hero-exclamation-triangle" class="size-4 text-error" />
          Active Contradictions
        </h3>
        <%= if length(@contradictions) == 0 do %>
          <div class="text-sm text-base-content/50 italic p-4 bg-base-200/30 rounded-lg text-center">
            No active contradictions
          </div>
        <% else %>
          <div class="space-y-2">
            <%= for node <- @contradictions do %>
              <div class="bg-error/5 border border-error/20 rounded-lg p-3">
                <div
                  class="flex items-start justify-between cursor-pointer"
                  phx-click="toggle_expand"
                  phx-value-id={node.id}
                >
                  <div class="flex-1">
                    <div class="font-medium text-sm text-error">
                      {inspect(node.datum)}
                    </div>
                    <div class="text-xs text-base-content/50 mt-1">
                      Type: {node.node_type} | Label: {node.label} | {length(node.justifications)} justification(s)
                    </div>
                  </div>
                  <.icon
                    name={if @expanded_id == node.id, do: "hero-chevron-up", else: "hero-chevron-down"}
                    class="size-4 text-base-content/40"
                  />
                </div>
                <%= if @expanded_id == node.id do %>
                  <div class="mt-3 pt-3 border-t border-error/20 text-xs space-y-1">
                    <div>
                      <span class="text-base-content/60">Node ID:</span>
                      <span class="font-mono">{node.id}</span>
                    </div>
                    <div>
                      <span class="text-base-content/60">Justifications:</span>
                      <span>{Enum.join(node.justifications, ", ")}</span>
                    </div>
                    <div>
                      <span class="text-base-content/60">Consequences:</span>
                      <span>{Enum.join(node.consequences, ", ")}</span>
                    </div>
                  </div>
                <% end %>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>

      <!-- Node type distribution -->
      <%= if Map.get(@stats, :total_nodes, 0) > 0 do %>
        <div>
          <h3 class="font-semibold text-sm mb-2">Node Distribution</h3>
          <div class="bg-base-200/30 rounded-lg p-3">
            <div class="grid grid-cols-2 gap-2 text-sm">
              <div class="flex justify-between">
                <span class="text-base-content/60">Premises:</span>
                <span class="font-medium">{Map.get(@stats, :premise_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-base-content/60">Assumptions:</span>
                <span class="font-medium">{Map.get(@stats, :assumption_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-base-content/60">Derived:</span>
                <span class="font-medium">{Map.get(@stats, :derived_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-base-content/60">Contradiction nodes:</span>
                <span class="font-medium">{Map.get(@stats, :contradiction_count, 0)}</span>
              </div>
            </div>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  defp users_sub_view(assigns) do
    assigns = assign_new(assigns, :user_data, fn -> load_user_data(assigns.selected_user) end)

    ~H"""
    <div class="divide-y divide-base-300/50">
      <%= if length(@user_ids) == 0 do %>
        <.empty_state icon="hero-user-group" message="No user models found" />
      <% else %>
        <!-- User list -->
        <div class="p-4">
          <h3 class="font-semibold text-sm mb-3">Users ({length(@user_ids)})</h3>
          <div class="flex flex-wrap gap-2">
            <%= for user_id <- @user_ids do %>
              <button
                phx-click="select_user"
                phx-value-user_id={user_id}
                class={[
                  "btn btn-sm",
                  if(@selected_user == user_id, do: "btn-primary", else: "btn-ghost")
                ]}
              >
                {String.slice(user_id, 0, 16)}{if String.length(user_id) > 16, do: "...", else: ""}
              </button>
            <% end %>
          </div>
        </div>

        <!-- Selected user details -->
        <%= if @selected_user && @user_data do %>
          <div class="p-4 space-y-4">
            <h3 class="font-semibold text-sm flex items-center gap-2">
              <.icon name="hero-user" class="size-4 text-primary" />
              User: <span class="font-mono text-primary">{@selected_user}</span>
            </h3>

            <!-- Facts table -->
            <% facts = Map.get(@user_data, :facts, %{})
            bounds = Map.get(@user_data, :epistemic_bounds, %{})
            provenance = Map.get(@user_data, :provenance_map, %{}) %>

            <%= if map_size(facts) > 0 do %>
              <div>
                <h4 class="text-xs font-semibold text-base-content/60 mb-2">
                  Known Facts ({map_size(facts)})
                </h4>
                <table class="table table-sm">
                  <thead class="bg-base-200/50">
                    <tr>
                      <th>Key</th>
                      <th>Value</th>
                      <th>Confidence</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    <%= for {key, value} <- Enum.sort(facts) do %>
                      <% conf = Map.get(bounds, key, 0.5)
                      source = Map.get(provenance, key) %>
                      <tr class="hover:bg-base-200/30">
                        <td class="font-mono text-primary text-sm">{key}</td>
                        <td class="text-sm">{inspect(value)}</td>
                        <td>
                          <span class={confidence_color(conf)}>{format_confidence(conf)}</span>
                        </td>
                        <td class="text-xs text-base-content/50">{source || "-"}</td>
                      </tr>
                    <% end %>
                  </tbody>
                </table>
              </div>
            <% else %>
              <div class="text-sm text-base-content/50 italic p-4 bg-base-200/30 rounded-lg text-center">
                No facts recorded for this user
              </div>
            <% end %>

            <!-- Interaction patterns -->
            <% patterns = Map.get(@user_data, :interaction_patterns, %{}) %>
            <%= if map_size(patterns) > 0 do %>
              <div>
                <h4 class="text-xs font-semibold text-base-content/60 mb-2">
                  Interaction Patterns
                </h4>
                <div class="grid grid-cols-2 gap-2">
                  <%= for {pattern_type, data} <- Enum.sort(patterns) do %>
                    <div class="bg-base-200/50 rounded-lg px-3 py-2 text-sm">
                      <span class="font-medium">{pattern_type}:</span>
                      <span class="text-base-content/60 ml-1">{inspect(data)}</span>
                    </div>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      <% end %>
    </div>
    """
  end

  defp load_user_data(nil), do: nil

  defp load_user_data(user_id) do
    case UserModelStore.get(user_id) do
      nil -> nil
      model -> Map.from_struct(model)
    end
  rescue
    _ -> nil
  end

  defp format_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y-%m-%d %H:%M")
  end

  defp format_datetime(_), do: "-"

  defp empty_state(assigns) do
    ~H"""
    <div class="p-16 text-center text-base-content/50">
      <.icon name={@icon} class="size-12 mx-auto mb-4 text-base-content/30" />
      <p>{@message}</p>
    </div>
    """
  end

  defp pagination(assigns) do
    ~H"""
    <%= if @total_pages > 1 do %>
      <div class="flex items-center justify-between px-4 py-3 border-t border-base-300">
        <div class="text-sm text-base-content/60">
          Showing {(@page - 1) * @page_size + 1}-{min(@page * @page_size, @total_entries)} of {@total_entries}
        </div>
        <div class="flex items-center gap-1">
          <button
            phx-click="change_page"
            phx-value-page={@page - 1}
            disabled={@page == 1}
            class="btn btn-ghost btn-xs btn-square"
          >
            <.icon name="hero-chevron-left" class="size-4" />
          </button>
          <span class="px-2 text-sm">Page {@page} of {@total_pages}</span>
          <button
            phx-click="change_page"
            phx-value-page={@page + 1}
            disabled={@page == @total_pages}
            class="btn btn-ghost btn-xs btn-square"
          >
            <.icon name="hero-chevron-right" class="size-4" />
          </button>
        </div>
      </div>
    <% end %>
    """
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp build_url_params(socket) do
    params = %{}

    params =
      if socket.assigns.tab != :entities,
        do: Map.put(params, "tab", socket.assigns.tab),
        else: params

    params =
      if socket.assigns.tab == :entities and socket.assigns.selected_type,
        do: Map.put(params, "type", socket.assigns.selected_type),
        else: params

    params =
      if socket.assigns.page > 1, do: Map.put(params, "page", socket.assigns.page), else: params

    params =
      if socket.assigns.search_query != "",
        do: Map.put(params, "q", socket.assigns.search_query),
        else: params

    params
  end

  defp get_world_metrics(world_id) do
    case WorldManager.get_metrics(world_id) do
      {:ok, metrics} -> WorldMetrics.summary(metrics)
      _ -> nil
    end
  rescue
    _ -> nil
  end

  defp load_world_episodes(world_id) do
    case MemoryStore.all_episodes(world_id: world_id) do
      {:ok, episodes} -> episodes
      _ -> []
    end
  rescue
    _ -> []
  end

  defp load_world_semantics(world_id) do
    case MemoryStore.all_semantics(world_id: world_id) do
      {:ok, semantics} -> semantics
      _ -> []
    end
  rescue
    _ -> []
  end

  defp load_world_knowledge(world_id) do
    KnowledgeStore.get_world_knowledge(world_id)
  rescue
    _ -> %{}
  end

  defp maybe_load_beliefs_data(socket) do
    if socket.assigns.beliefs_data do
      socket
    else
      beliefs_data = load_beliefs_data()
      assign(socket, :beliefs_data, beliefs_data)
    end
  end

  defp load_beliefs_data do
    beliefs =
      try do
        case BeliefStore.query_beliefs([]) do
          {:ok, list} -> list
          _ -> []
        end
      rescue
        _ -> []
      end

    facts =
      try do
        FactDatabase.query([])
      rescue
        _ -> []
      end

    jtms_stats =
      try do
        JTMS.stats()
      rescue
        _ -> %{}
      end

    jtms_contradictions =
      try do
        JTMS.get_contradictions()
      rescue
        _ -> []
      end

    user_ids =
      try do
        case UserModelStore.list_all_users() do
          {:ok, ids} -> ids
          _ -> []
        end
      rescue
        _ -> []
      end

    fact_categories =
      facts
      |> Enum.map(& &1.category)
      |> Enum.uniq()
      |> Enum.sort()

    belief_sources =
      beliefs
      |> Enum.map(& &1.source)
      |> Enum.uniq()
      |> Enum.reject(&is_nil/1)

    belief_authorities =
      beliefs
      |> Enum.map(&Map.get(&1, :source_authority))
      |> Enum.reject(&is_nil/1)
      |> Enum.uniq()

    %{
      beliefs: beliefs,
      facts: facts,
      jtms_stats: jtms_stats,
      jtms_contradictions: jtms_contradictions,
      user_ids: user_ids,
      fact_categories: fact_categories,
      belief_sources: belief_sources,
      belief_authorities: belief_authorities
    }
  end

  defp group_overlay_by_type(overlay) when is_list(overlay) do
    overlay
    |> Enum.flat_map(fn {key, info} ->
      case info do
        infos when is_list(infos) -> Enum.map(infos, fn i -> {key, normalize_info(i)} end)
        info when is_map(info) -> [{key, normalize_info(info)}]
        _ -> []
      end
    end)
    |> Enum.group_by(fn {_key, info} ->
      Map.get(info, :entity_type) || Map.get(info, :type) || "unknown"
    end)
  end

  defp normalize_info(info) when is_map(info) do
    Map.new(info, fn
      {k, v} when is_binary(k) ->
        atom_key =
          case k do
            "entity_type" -> :entity_type
            "type" -> :type
            "value" -> :value
            "source" -> :source
            _ -> String.to_atom(k)
          end

        {atom_key, v}

      {k, v} when is_atom(k) ->
        {k, v}

      other ->
        other
    end)
  rescue
    _ -> info
  end

  defp filter_by_search(items, "", _getter), do: items

  defp filter_by_search(items, search, getter) do
    Enum.filter(items, fn item ->
      value = getter.(item) || ""
      String.contains?(String.downcase(value), search)
    end)
  end

  defp paginate(items, page, page_size) do
    items
    |> Enum.drop((page - 1) * page_size)
    |> Enum.take(page_size)
  end

  defp parse_int(nil, default), do: default

  defp parse_int(str, default) when is_binary(str) do
    String.to_integer(str)
  rescue
    _ -> default
  end

  defp confidence_color(nil), do: "text-base-content/40"
  defp confidence_color(c) when c >= 0.8, do: "text-success font-medium"
  defp confidence_color(c) when c >= 0.5, do: "text-warning"
  defp confidence_color(_), do: "text-error"

  defp format_confidence(nil), do: "—"
  defp format_confidence(c), do: "#{Float.round(c * 100, 1)}%"

  defp authority_badge_class("professional"), do: "badge-info"
  defp authority_badge_class("personal"), do: "badge-secondary"
  defp authority_badge_class("academic"), do: "badge-accent"
  defp authority_badge_class("entertainment"), do: "badge-warning"
  defp authority_badge_class("unknown"), do: "badge-ghost"
  defp authority_badge_class(_), do: "badge-ghost"

  defp authority_category(authority_key) do
    case SourceAuthority.get_profile(authority_key) do
      nil -> "unknown"
      profile -> profile.category
    end
  rescue
    _ -> "unknown"
  end

  defp group_authority_profiles(profiles) do
    profiles
    |> Enum.group_by(fn p -> p.profile.category end)
    |> Enum.sort_by(fn {cat, _} ->
      case cat do
        "professional" -> 0
        "academic" -> 1
        "personal" -> 2
        "unknown" -> 3
        "entertainment" -> 4
        _ -> 5
      end
    end)
  end

  defp format_value(value) when is_binary(value), do: value
  defp format_value(value), do: inspect(value)
end
