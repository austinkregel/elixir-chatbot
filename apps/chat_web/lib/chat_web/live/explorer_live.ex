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
    # The open confirmation panel, if any, and its failure
    |> assign(:confirming, nil)
    |> assign(:confirm_error, nil)
    # The confidence (0-100) chosen on the open Set-confidence slider, nil until it moves
    |> assign(:chosen_confidence, nil)
    |> assign(:add_belief_form, %{"subject" => "", "predicate" => "", "object" => "", "confidence" => "100", "authority" => "mentor"})
    |> assign(:authority_filter, nil)
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
    page_size = socket.assigns.page_size

    {items, total, total_rows, filter_active, paged?} =
      case tab do
        :entities ->
          type = socket.assigns.selected_type
          entities = Map.get(socket.assigns.overlay_by_type, type, [])
          filtered = filter_by_search(entities, search, fn {key, _info} -> key end)
          {filtered, length(filtered), length(entities), search != "", true}

        :candidates ->
          filtered = filter_by_search(socket.assigns.candidates, search, & &1.value)

          {filtered, length(filtered), length(socket.assigns.candidates), search != "", true}

        :episodes ->
          filtered = filter_by_search(socket.assigns.episodes, search, & &1.state)

          {filtered, length(filtered), length(socket.assigns.episodes), search != "", true}

        :semantics ->
          filtered = filter_by_search(socket.assigns.semantics, search, & &1.representation)

          {filtered, length(filtered), length(socket.assigns.semantics), search != "", true}

        :knowledge ->
          # Knowledge is a map, don't paginate
          knowledge_size = map_size(socket.assigns.knowledge)
          {socket.assigns.knowledge, knowledge_size, knowledge_size, false, false}

        :beliefs ->
          # Load beliefs data lazily on first access
          socket = maybe_load_beliefs_data(socket)
          beliefs_data = socket.assigns.beliefs_data || %{}
          sub_tab = socket.assigns.beliefs_sub_tab

          case sub_tab do
            :beliefs ->
              all_items = Map.get(beliefs_data, :beliefs, [])
              items = all_items
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

              {filtered, length(filtered), length(all_items),
               search != "" or not is_nil(source_filter) or not is_nil(authority_filter), true}

            :facts ->
              all_items = Map.get(beliefs_data, :facts, [])
              items = all_items
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

              {filtered, length(filtered), length(all_items),
               search != "" or not is_nil(cat_filter), true}

            _ ->
              {%{}, 0, 0, false, false}
          end
      end

    page = clamp_page(socket.assigns.page, total, page_size)
    filtered = if paged?, do: paginate(items, page, page_size), else: items

    socket
    |> assign(:page, page)
    |> assign(:filtered_data, filtered)
    |> assign(:total_entries, total)
    |> assign(:total_rows, total_rows)
    |> assign(:matching_rows, if(filter_active, do: total))
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

  def handle_event("open_confirm", %{"id" => confirm_id}, socket) do
    {:noreply, assign(socket, confirming: confirm_id, confirm_error: nil, chosen_confidence: nil)}
  end

  def handle_event("close_confirm", _params, socket) do
    {:noreply, assign(socket, confirming: nil, confirm_error: nil, chosen_confidence: nil)}
  end

  # The Set-confidence slider reports each position as it moves, so the panel
  # can show the value the person is about to confirm.
  def handle_event("choose_confidence", %{"confidence" => value}, socket) do
    {:noreply, assign(socket, :chosen_confidence, slider_percent!(value))}
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
            |> assign(confirming: nil, confirm_error: nil)
            |> assign(:add_belief_form, %{"subject" => "", "predicate" => "", "object" => "", "confidence" => "100", "authority" => "mentor"})
            |> assign(:beliefs_data, nil)
            |> maybe_load_beliefs_data()
            |> apply_filters()
            |> put_flash(:info, "Guided belief added (#{authority})")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply, assign(socket, :confirm_error, "Adding the belief failed: #{inspect(reason)}")}
      end
    else
      {:noreply, assign(socket, :confirm_error, "Subject, predicate, and object are all required.")}
    end
  end

  def handle_event("retract_belief", %{"id" => belief_id}, socket) do
    case BeliefStore.retract_belief(belief_id) do
      :ok ->
        socket =
          socket
          |> assign(confirming: nil, confirm_error: nil)
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()
          |> put_flash(:info, "Belief retracted")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, assign(socket, :confirm_error, "Retract failed: #{inspect(reason)}")}
    end
  end

  def handle_event("confirm_belief", %{"id" => belief_id}, socket) do
    case BeliefStore.confirm_belief(belief_id) do
      {:ok, _updated} ->
        socket =
          socket
          |> assign(confirming: nil, confirm_error: nil)
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()
          |> put_flash(:info, "Belief confirmed")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, assign(socket, :confirm_error, "Confirm failed: #{inspect(reason)}")}
    end
  end

  def handle_event("save_confidence", %{"belief_id" => belief_id, "confidence" => conf_str}, socket) do
    confidence = parse_confidence(conf_str)

    case BeliefStore.update_confidence(belief_id, confidence) do
      {:ok, _updated} ->
        socket =
          socket
          |> assign(confirming: nil, confirm_error: nil, chosen_confidence: nil)
          |> assign(:beliefs_data, nil)
          |> maybe_load_beliefs_data()
          |> apply_filters()
          |> put_flash(:info, "Belief confidence set to #{round(confidence * 100)}%")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply, assign(socket, :confirm_error, "Setting the confidence failed: #{inspect(reason)}")}
    end
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

  # A range input with min 0 and max 100 sends a whole number in that range;
  # anything else did not come from the slider.
  defp slider_percent!(value) when is_binary(value) do
    case Integer.parse(value) do
      {percent, ""} when percent in 0..100 ->
        percent

      _ ->
        raise ArgumentError,
              "ChatWeb.ExplorerLive: choose_confidence expects a whole number from 0 to 100 " <>
                "from the confidence slider, got #{inspect(value)}."
    end
  end

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
            <h1 class="text-title text-ink">Data Explorer</h1>
            <p class="text-body text-ink-muted">
              Explore data in world: <span class="text-value-strong text-ink">{@current_world_id}</span>
            </p>
          </div>
          <.btn phx-click="refresh" variant={:ghost} size={:sm}>
            <.icon name="hero-arrow-path" class="size-4" /> Refresh
          </.btn>
        </div>
      </:page_header>

      <div class="p-space-lg sm:p-space-xl space-y-space-xl">
        <!-- Stats Summary -->
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-space-lg">
          <.stat_kpi label="Entities" value={to_string(length(@overlay))} />
          <.stat_kpi label="Candidates" value={to_string(length(@candidates))} />
          <.stat_kpi label="Episodes" value={to_string(length(@episodes))} />
          <.stat_kpi label="Semantics" value={to_string(length(@semantics))} />
          <.stat_kpi label="Knowledge" value={to_string(map_size(@knowledge))} />
          <.stat_kpi
            label="Beliefs"
            value={if @beliefs_data, do: to_string(length(@beliefs_data[:beliefs] || [])), else: "-"}
          />
        </div>

    <!-- Tabs -->
        <div class="flex flex-wrap items-center gap-space-lg">
          <.tabs class="flex-wrap">
            <.tab active={@tab == :entities} phx-click="switch_tab" phx-value-tab={:entities}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-tag" class="size-4" />
                <span class="hidden sm:inline">Entities</span>
                <span class="text-offset">{length(@overlay)}</span>
              </span>
            </.tab>
            <.tab active={@tab == :candidates} phx-click="switch_tab" phx-value-tab={:candidates}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-queue-list" class="size-4" />
                <span class="hidden sm:inline">Candidates</span>
                <span class="text-offset">{length(@candidates)}</span>
              </span>
            </.tab>
            <.tab active={@tab == :episodes} phx-click="switch_tab" phx-value-tab={:episodes}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-clock" class="size-4" />
                <span class="hidden sm:inline">Episodes</span>
                <span class="text-offset">{length(@episodes)}</span>
              </span>
            </.tab>
            <.tab active={@tab == :semantics} phx-click="switch_tab" phx-value-tab={:semantics}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-light-bulb" class="size-4" />
                <span class="hidden sm:inline">Semantics</span>
                <span class="text-offset">{length(@semantics)}</span>
              </span>
            </.tab>
            <.tab active={@tab == :knowledge} phx-click="switch_tab" phx-value-tab={:knowledge}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-book-open" class="size-4" />
                <span class="hidden sm:inline">Knowledge</span>
                <span class="text-offset">{map_size(@knowledge)}</span>
              </span>
            </.tab>
            <.tab active={@tab == :beliefs} phx-click="switch_tab" phx-value-tab={:beliefs}>
              <span class="inline-flex items-center gap-space-xs">
                <.icon name="hero-eye" class="size-4" />
                <span class="hidden sm:inline">Beliefs</span>
                <span class="text-offset">
                  {if @beliefs_data, do: length(@beliefs_data[:beliefs] || []), else: 0}
                </span>
              </span>
            </.tab>
          </.tabs>

    <!-- Search -->
          <div class="flex-1 max-w-md">
            <input
              type="text"
              placeholder="Search..."
              value={@search_query}
              phx-keyup="search"
              name="query"
              phx-debounce="150"
              class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
            />
          </div>
        </div>

    <!-- Type Selector (for entities tab) -->
        <%= if @tab == :entities and length(@entity_types) > 0 do %>
          <div class="flex flex-wrap gap-space-sm">
            <%= for type <- @entity_types do %>
              <.btn
                phx-click="select_type"
                phx-value-type={type}
                variant={if(type == @selected_type, do: :secondary, else: :ghost)}
                size={:sm}
              >
                {type}
                <.badge size={:xs}>{length(Map.get(@overlay_by_type, type, []))}</.badge>
              </.btn>
            <% end %>
          </div>
        <% end %>

    <!-- Recently Promoted Toast -->
        <%= if MapSet.size(@recently_promoted) > 0 do %>
          <.alert variant={:success} icon="hero-check" class="animate-fade-in">
            <div class="flex flex-wrap items-center gap-space-sm">
              <span>Just promoted:</span>
              <%= for value <- MapSet.to_list(@recently_promoted) do %>
                <.badge variant={:success}>{value}</.badge>
              <% end %>
            </div>
          </.alert>
        <% end %>

    <!-- Content -->
        <.card class="overflow-hidden">
          <%= case @tab do %>
            <% :entities -> %>
              <.entities_table
                data={@filtered_data}
                loading={@loading}
                page={@page}
                total_rows={@total_rows}
                matching_rows={@matching_rows}
                page_size={@page_size}
              />
            <% :candidates -> %>
              <.candidates_table
                data={@filtered_data}
                loading={@loading}
                world_id={@current_world_id}
                page={@page}
                total_rows={@total_rows}
                matching_rows={@matching_rows}
                page_size={@page_size}
              />
            <% :episodes -> %>
              <.episodes_list
                data={@filtered_data}
                expanded_id={@expanded_id}
                page={@page}
                total_rows={@total_rows}
                matching_rows={@matching_rows}
                page_size={@page_size}
              />
            <% :semantics -> %>
              <.semantics_list
                data={@filtered_data}
                expanded_id={@expanded_id}
                page={@page}
                total_rows={@total_rows}
                matching_rows={@matching_rows}
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
                confirming={@confirming}
                confirm_error={@confirm_error}
                chosen_confidence={@chosen_confidence}
                add_belief_form={@add_belief_form}
                expanded_id={@expanded_id}
                page={@page}
                total_rows={@total_rows}
                matching_rows={@matching_rows}
                page_size={@page_size}
              />
          <% end %>
        </.card>
      </div>
    </.app_shell>
    """
  end

  # ============================================================================
  # Sub-components
  # ============================================================================

  defp entities_table(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No entities found" class="m-space-lg" />
    <% else %>
      <table class="w-full text-left text-body-dense text-ink">
        <thead class="bg-surface-sunk">
          <tr>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Lookup Key</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Canonical Value</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Source</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-border">
          <%= for {key, info} <- @data do %>
            <tr class="even:bg-surface-sunk">
              <td class="h-row-compact px-space-sm font-semibold">{key}</td>
              <td class="h-row-compact px-space-sm">
                <%= if info[:value] && info[:value] != key do %>
                  {info[:value]}
                <% else %>
                  <span class="text-ink-muted">—</span>
                <% end %>
              </td>
              <td class="h-row-compact px-space-sm">
                <.badge>{info[:source] || "unknown"}</.badge>
              </td>
            </tr>
          <% end %>
        </tbody>
      </table>
      <.page_bar
        page={@page}
        page_size={@page_size}
        total={@total_rows}
        matching={@matching_rows}
        event="change_page"
        class="px-space-lg py-space-md border-t border-border"
      />
    <% end %>
    """
  end

  defp candidates_table(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No candidates found" class="m-space-lg" />
    <% else %>
      <table class="w-full text-left text-body-dense text-ink">
        <thead class="bg-surface-sunk">
          <tr>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Value</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Inferred Type</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Confidence</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">Occurrences</th>
            <th class="h-row-compact px-space-sm text-label text-ink-muted">
              <span class="inline-flex items-center gap-space-sm">
                Action
                <.reach_badge
                  id="explorer-promote-reach"
                  reach={:local}
                  target={"gazetteer overlay · #{@world_id}"}
                />
              </span>
            </th>
          </tr>
        </thead>
        <tbody class="divide-y divide-border">
          <%= for candidate <- @data do %>
            <% is_loading = @loading == candidate.value %>
            <tr class="even:bg-surface-sunk group">
              <td class="h-row-compact px-space-sm font-semibold">
                <div class="flex items-center gap-space-sm">
                  <%= if is_loading do %>
                    <.icon name="hero-arrow-path" class="size-4 motion-safe:animate-spin text-progress-fill" />
                  <% end %>
                  {candidate.value}
                </div>
              </td>
              <td class="h-row-compact px-space-sm">
                <.badge>{candidate.inferred_type || "unknown"}</.badge>
              </td>
              <td class="h-row-compact px-space-sm">
                <span class={confidence_text_class(candidate.confidence)}>
                  {format_confidence(candidate.confidence)}
                </span>
              </td>
              <td class="h-row-compact px-space-sm text-value">{candidate.occurrences}</td>
              <td class="h-row-compact px-space-sm">
                <%= if candidate.inferred_type && candidate.inferred_type != "unknown" do %>
                  <.btn
                    phx-click="promote_candidate"
                    phx-value-value={candidate.value}
                    phx-value-type={candidate.inferred_type}
                    disabled={is_loading}
                    variant={:primary}
                    size={:xs}
                    class="opacity-0 group-hover:opacity-100"
                    title="Promote to gazetteer"
                    aria-describedby="explorer-promote-reach"
                  >
                    <.icon name="hero-arrow-up-circle" class="size-4" />
                  </.btn>
                <% end %>
              </td>
            </tr>
          <% end %>
        </tbody>
      </table>
      <.page_bar
        page={@page}
        page_size={@page_size}
        total={@total_rows}
        matching={@matching_rows}
        event="change_page"
        class="px-space-lg py-space-md border-t border-border"
      />
    <% end %>
    """
  end

  defp episodes_list(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No episodes found" class="m-space-lg" />
    <% else %>
      <div class="divide-y divide-border">
        <%= for episode <- @data do %>
          <div class="p-space-lg hover:bg-surface-sunk transition-colors">
            <div
              class="flex items-start justify-between cursor-pointer"
              phx-click="toggle_expand"
              phx-value-id={episode.id}
            >
              <div class="flex-1 min-w-0">
                <div class="text-subheading text-ink truncate">{episode.state}</div>
                <div class="text-caption text-ink-muted mt-space-xs">
                  Action: {episode.action}
                </div>
              </div>
              <div class="flex items-center gap-space-sm ml-space-lg">
                <div class="flex flex-wrap gap-space-xs">
                  <%= for tag <- Enum.take(episode.tags, 3) do %>
                    <.badge size={:xs}>{tag}</.badge>
                  <% end %>
                </div>
                <.icon
                  name={
                    if @expanded_id == episode.id, do: "hero-chevron-up", else: "hero-chevron-down"
                  }
                  class="size-4 text-ink-muted"
                />
              </div>
            </div>
            <%= if @expanded_id == episode.id do %>
              <div class="mt-space-lg pt-space-lg border-t border-border text-body space-y-space-sm">
                <div>
                  <span class="text-ink-muted">ID:</span>
                  <span class="text-ref text-ink">{episode.id}</span>
                </div>
                <%= if episode.outcome && episode.outcome != "" do %>
                  <div>
                    <span class="text-ink-muted">Outcome:</span>
                    <p class="mt-space-xs bg-surface-sunk rounded-sm p-space-sm text-ink">{episode.outcome}</p>
                  </div>
                <% end %>
                <div class="flex flex-wrap gap-space-xs">
                  <%= for tag <- episode.tags do %>
                    <.badge>{tag}</.badge>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      <.page_bar
        page={@page}
        page_size={@page_size}
        total={@total_rows}
        matching={@matching_rows}
        event="change_page"
        class="px-space-lg py-space-md border-t border-border"
      />
    <% end %>
    """
  end

  defp semantics_list(assigns) do
    ~H"""
    <%= if length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No semantic facts found" class="m-space-lg" />
    <% else %>
      <div class="divide-y divide-border">
        <%= for semantic <- @data do %>
          <div class="p-space-lg hover:bg-surface-sunk transition-colors">
            <div
              class="flex items-start justify-between cursor-pointer"
              phx-click="toggle_expand"
              phx-value-id={semantic.id}
            >
              <div class="flex-1 min-w-0">
                <div class="text-subheading text-ink">{semantic.representation}</div>
                <div class="text-caption text-ink-muted mt-space-xs">
                  Evidence: {length(semantic.evidence_ids)} episodes
                </div>
              </div>
              <.icon
                name={
                  if @expanded_id == semantic.id, do: "hero-chevron-up", else: "hero-chevron-down"
                }
                class="size-4 text-ink-muted ml-space-lg"
              />
            </div>
            <%= if @expanded_id == semantic.id do %>
              <div class="mt-space-lg pt-space-lg border-t border-border text-body space-y-space-sm">
                <div>
                  <span class="text-ink-muted">ID:</span>
                  <span class="text-ref text-ink">{semantic.id}</span>
                </div>
                <div class="flex flex-wrap gap-space-xs">
                  <%= for ep_id <- semantic.evidence_ids do %>
                    <.badge class="font-mono">
                      {String.slice(ep_id, 0, 8)}...
                    </.badge>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      <.page_bar
        page={@page}
        page_size={@page_size}
        total={@total_rows}
        matching={@matching_rows}
        event="change_page"
        class="px-space-lg py-space-md border-t border-border"
      />
    <% end %>
    """
  end

  defp knowledge_view(assigns) do
    ~H"""
    <%= if map_size(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No knowledge stored" class="m-space-lg" />
    <% else %>
      <div class="divide-y divide-border">
        <%= for {category, items} <- @data do %>
          <div class="p-space-lg">
            <h3 class="text-subheading text-ink flex items-center gap-space-sm mb-space-md">
              <.icon name="hero-folder" class="size-4 text-ink-muted" />
              {category}
              <.badge>
                {if is_map(items), do: map_size(items), else: length(items)} items
              </.badge>
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-space-sm">
              <%= if is_map(items) do %>
                <%= for {key, value} <- Enum.take(items, 12) do %>
                  <div class="bg-surface-sunk rounded-sm px-space-md py-space-sm text-body">
                    <span class="font-semibold text-ink">{key}:</span>
                    <span class="text-ink-muted ml-space-xs">{format_value(value)}</span>
                  </div>
                <% end %>
                <%= if map_size(items) > 12 do %>
                  <div class="text-caption text-ink-muted px-space-md py-space-sm">
                    ... and {map_size(items) - 12} more
                  </div>
                <% end %>
              <% else %>
                <%= for item <- Enum.take(items, 12) do %>
                  <div class="bg-surface-sunk rounded-sm px-space-md py-space-sm text-body text-ink">{format_value(item)}</div>
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
    <div class="divide-y divide-border">
      <!-- Sub-tab navigation -->
      <div class="p-space-lg bg-surface-sunk">
        <div class="flex flex-wrap items-center gap-space-sm">
          <.tabs class="flex-wrap">
            <%= for {sub, label, icon} <- [
              {:beliefs, "Beliefs", "hero-eye"},
              {:facts, "Facts", "hero-book-open"},
              {:jtms, "JTMS Graph", "hero-share"},
              {:users, "User Models", "hero-user-group"}
            ] do %>
              <.tab active={sub == @sub_tab} phx-click="beliefs_sub_tab" phx-value-sub_tab={sub}>
                <span class="inline-flex items-center gap-space-xs">
                  <.icon name={icon} class="size-4" />
                  {label}
                  <%= case sub do %>
                    <% :beliefs -> %>
                      <span class="text-offset">{length(@beliefs_data[:beliefs] || [])}</span>
                    <% :facts -> %>
                      <span class="text-offset">{length(@beliefs_data[:facts] || [])}</span>
                    <% :jtms -> %>
                      <span class="text-offset">
                        {Map.get(@beliefs_data[:jtms_stats] || %{}, :total_nodes, 0)}
                      </span>
                    <% :users -> %>
                      <span class="text-offset">{length(@beliefs_data[:user_ids] || [])}</span>
                  <% end %>
                </span>
              </.tab>
            <% end %>
          </.tabs>

          <div class="flex items-center gap-space-sm ml-auto">
            <%= if @sub_tab == :beliefs do %>
              <.btn
                id="explorer-add-belief"
                phx-click="open_confirm"
                phx-value-id="explorer-confirm-add-belief"
                variant={:primary}
                size={:sm}
                reach={:shared}
                target="belief store"
                title="Add guided belief"
              >
                <.icon name="hero-plus" class="size-4" />
                Add Belief
              </.btn>
            <% end %>
            <.icon_btn phx-click="refresh_beliefs" variant={:ghost} size={:sm} title="Refresh beliefs data">
              <.icon name="hero-arrow-path" class="size-4" />
            </.icon_btn>
          </div>
        </div>
      </div>

      <!-- Authority Credibility Overview (only when beliefs sub-tab) -->
      <%= if @sub_tab == :beliefs and length(@authority_profiles) > 0 do %>
        <% active_profiles = Enum.filter(@authority_profiles, fn p -> p.total_added > 0 end) %>
        <%= if length(active_profiles) > 0 do %>
          <div class="p-space-lg bg-surface-sunk">
            <div class="text-label text-ink-muted mb-space-sm flex items-center gap-space-xs">
              <.icon name="hero-shield-check" class="size-4" />
              Authority Credibility
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-space-sm">
              <%= for p <- active_profiles do %>
                <div class="bg-surface rounded-md p-space-sm border border-border">
                  <div class="flex items-center justify-between mb-space-xs">
                    <.badge size={:xs}>
                      {p.profile.label}
                    </.badge>
                    <span class="text-caption text-ink-muted">{p.total_added} beliefs</span>
                  </div>
                  <div class="flex items-center gap-space-sm">
                    <div class="flex-1 bg-surface-sunk rounded-sm h-space-xs">
                      <div
                        class="h-space-xs rounded-sm bg-ink-muted"
                        style={"width: #{Float.round(p.credibility * 100, 1)}%"}
                      >
                      </div>
                    </div>
                    <span class="text-offset text-ink">{Float.round(p.credibility * 100, 0)}%</span>
                  </div>
                  <div class="flex items-center gap-space-sm mt-space-xs text-caption text-ink-muted">
                    <span>{p.confirmed_count} confirmed</span>
                    <span>{p.contradicted_count} contradicted</span>
                  </div>
                </div>
              <% end %>
            </div>
          </div>
        <% end %>
      <% end %>

      <!-- Add belief: the form is the confirmation -->
      <%= if @sub_tab == :beliefs and @confirming == "explorer-confirm-add-belief" do %>
        <div class="p-space-lg bg-surface-sunk border-b border-border flex justify-end">
          <.execute_confirm
            id="explorer-confirm-add-belief"
            open={true}
            reach={:shared}
            verb="Add belief"
            target="belief store"
            consequence="Adds this belief to the belief store every query reads, with a JTMS node, and saves it to Atlas. Its confidence comes from the chosen authority's tier and tracked credibility."
            on_confirm="add_guided_belief"
            on_cancel="close_confirm"
            trigger_id="explorer-add-belief"
            error={@confirm_error}
            class="w-full"
          >
            <:fields>
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-space-md">
                <div>
                  <label for="explorer-belief-subject" class="block mb-space-xs text-label text-ink-muted">Subject</label>
                  <select
                    id="explorer-belief-subject"
                    name="belief[subject]"
                    phx-change="update_add_belief_form"
                    class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink"
                  >
                    <option value="world" selected={@add_belief_form["subject"] == "world"}>world</option>
                    <option value="self" selected={@add_belief_form["subject"] == "self"}>self</option>
                    <option value="user" selected={@add_belief_form["subject"] == "user"}>user</option>
                  </select>
                </div>
                <div>
                  <label for="explorer-belief-predicate" class="block mb-space-xs text-label text-ink-muted">Predicate</label>
                  <input
                    id="explorer-belief-predicate"
                    type="text"
                    name="belief[predicate]"
                    value={@add_belief_form["predicate"]}
                    phx-change="update_add_belief_form"
                    placeholder="e.g. name, likes, location"
                    class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
                    required
                  />
                </div>
                <div>
                  <label for="explorer-belief-object" class="block mb-space-xs text-label text-ink-muted">Object</label>
                  <input
                    id="explorer-belief-object"
                    type="text"
                    name="belief[object]"
                    value={@add_belief_form["object"]}
                    phx-change="update_add_belief_form"
                    placeholder="The value"
                    class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink placeholder:text-ink-muted"
                    required
                  />
                </div>
                <div>
                  <label for="explorer-belief-authority" class="block mb-space-xs text-label text-ink-muted">Authority</label>
                  <select
                    id="explorer-belief-authority"
                    name="belief[authority]"
                    phx-change="update_add_belief_form"
                    class="w-full h-control-sm px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body-dense text-ink"
                  >
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
            </:fields>
          </.execute_confirm>
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
            confirming={@confirming}
            confirm_error={@confirm_error}
            chosen_confidence={@chosen_confidence}
            page={@page}
            total_rows={@total_rows}
            matching_rows={@matching_rows}
            page_size={@page_size}
          />
        <% :facts -> %>
          <.facts_sub_view
            data={@filtered_data}
            categories={@beliefs_data[:fact_categories] || []}
            category_filter={@category_filter}
            page={@page}
            total_rows={@total_rows}
            matching_rows={@matching_rows}
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
    <div class="px-space-lg pt-space-md flex flex-wrap gap-space-sm">
      <%= if length(@sources) > 0 do %>
        <%= for source <- @sources do %>
          <.btn
            phx-click="beliefs_source_filter"
            phx-value-source={source}
            variant={if(to_string(@source_filter) == to_string(source), do: :secondary, else: :ghost)}
            size={:xs}
          >
            {source}
          </.btn>
        <% end %>
      <% end %>
      <%= if length(@authority_types) > 0 do %>
        <span class="text-ink-muted self-center">|</span>
        <%= for auth <- @authority_types do %>
          <.btn
            phx-click="authority_filter"
            phx-value-authority={auth}
            variant={if(to_string(@authority_filter) == to_string(auth), do: :secondary, else: :ghost)}
            size={:xs}
          >
            {auth}
          </.btn>
        <% end %>
      <% end %>
    </div>

    <%= if is_list(@data) and length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No beliefs found" class="m-space-lg" />
    <% else %>
      <%= if is_list(@data) do %>
        <div class="px-space-lg pt-space-md flex items-center gap-space-sm text-caption text-ink-muted">
          Confirm, Adjust confidence and Retract
          <.reach_badge id="explorer-belief-actions-reach" reach={:shared} target="belief store" />
        </div>
        <div class="divide-y divide-border">
          <%= for belief <- @data do %>
            <% b_conf = belief.confidence || 0.0
            b_authority = Map.get(belief, :source_authority) %>
            <div class={[
              "p-space-lg hover:bg-surface-sunk transition-colors",
              if(b_authority, do: "border-l-2 border-border-strong", else: "")
            ]}>
              <div class="flex items-start justify-between gap-space-lg">
                <div class="flex-1 min-w-0">
                  <!-- Subject / Predicate / Object -->
                  <div class="flex items-center gap-space-xs text-value">
                    <span class="text-value-strong text-ink">{belief.subject}</span>
                    <span class="text-ink-muted">/</span>
                    <span class="text-ink-muted">{belief.predicate}</span>
                    <span class="text-ink-muted">/</span>
                    <span class="text-ink">{inspect(belief.object)}</span>
                  </div>

                  <!-- Confidence bar -->
                  <div class="flex items-center gap-space-sm mt-space-sm max-w-xs">
                    <div class="flex-1 bg-surface-sunk rounded-sm h-space-xs">
                      <div
                        class="h-space-xs rounded-sm bg-ink-muted"
                        style={"width: #{Float.round(b_conf * 100, 1)}%"}
                      >
                      </div>
                    </div>
                    <span class="text-offset text-ink-muted">
                      {Float.round(b_conf * 100, 1)}%
                    </span>
                  </div>
                </div>

                <!-- Badges + Actions -->
                <div class="flex items-center gap-space-sm shrink-0">
                  <%= if b_authority do %>
                    <.badge>
                      {b_authority}
                    </.badge>
                  <% end %>
                  <%= if belief.source do %>
                    <.badge>
                      {belief.source}
                    </.badge>
                  <% end %>
                  <%= if belief.node_id do %>
                    <.badge class="border border-border-strong">JTMS</.badge>
                  <% end %>

                  <!-- Action buttons -->
                  <div class="flex items-center gap-space-xs ml-space-xs">
                    <.icon_btn
                      id={belief_trigger_id("confirm", belief.id)}
                      phx-click="open_confirm"
                      phx-value-id={belief_confirm_id("confirm", belief.id)}
                      aria-describedby="explorer-belief-actions-reach"
                      variant={:primary}
                      size={:sm}
                      title="Confirm (boost confidence +10%)"
                    >
                      <.icon name="hero-check-circle" class="size-4" />
                    </.icon_btn>
                    <.icon_btn
                      id={belief_trigger_id("adjust", belief.id)}
                      phx-click="open_confirm"
                      phx-value-id={belief_confirm_id("adjust", belief.id)}
                      aria-describedby="explorer-belief-actions-reach"
                      variant={:ghost}
                      size={:sm}
                      title="Adjust confidence"
                    >
                      <.icon name="hero-adjustments-horizontal" class="size-4" />
                    </.icon_btn>
                    <.icon_btn
                      id={belief_trigger_id("retract", belief.id)}
                      phx-click="open_confirm"
                      phx-value-id={belief_confirm_id("retract", belief.id)}
                      aria-describedby="explorer-belief-actions-reach"
                      variant={:primary}
                      size={:sm}
                      title="Retract belief"
                    >
                      <.icon name="hero-x-circle" class="size-4" />
                    </.icon_btn>
                  </div>
                </div>
              </div>

              <div class="flex justify-end">
                <.execute_confirm
                  id={belief_confirm_id("confirm", belief.id)}
                  open={@confirming == belief_confirm_id("confirm", belief.id)}
                  reach={:shared}
                  verb="Confirm"
                  target={belief_target(belief)}
                  consequence="Raises this belief's confidence by 0.1, to at most 1.0, in the belief store every query reads, and records a confirmation against its authority's credibility when it has one. Nothing is written to Atlas."
                  on_confirm={JS.push("confirm_belief", value: %{id: belief.id})}
                  on_cancel="close_confirm"
                  trigger_id={belief_trigger_id("confirm", belief.id)}
                  error={@confirm_error}
                  class="mt-space-sm"
                />
                <.execute_confirm
                  id={belief_confirm_id("adjust", belief.id)}
                  open={@confirming == belief_confirm_id("adjust", belief.id)}
                  reach={:shared}
                  verb="Set confidence"
                  target={belief_target(belief)}
                  consequence="Sets this belief's confidence to the chosen value in the belief store every query reads, and saves it to Atlas."
                  on_confirm="save_confidence"
                  on_cancel="close_confirm"
                  trigger_id={belief_trigger_id("adjust", belief.id)}
                  error={@confirm_error}
                  class="mt-space-sm w-full"
                >
                  <:fields>
                    <input type="hidden" name="belief_id" value={belief.id} />
                    <label
                      for={"explorer-belief-confidence-#{belief.id}"}
                      class="block mb-space-xs text-label text-ink-muted"
                    >
                      Confidence (now {Float.round(b_conf * 100, 1)}%)
                    </label>
                    <div class="flex items-center gap-space-sm">
                      <input
                        id={"explorer-belief-confidence-#{belief.id}"}
                        type="range"
                        name="confidence"
                        min="0"
                        max="100"
                        value={@chosen_confidence || round(b_conf * 100)}
                        phx-change="choose_confidence"
                        class="min-w-0 flex-1 accent-primary"
                      />
                      <output
                        id={"explorer-belief-confidence-#{belief.id}-chosen"}
                        for={"explorer-belief-confidence-#{belief.id}"}
                        aria-live="polite"
                        class="w-12 shrink-0 text-right text-value-strong tabular-nums text-ink"
                      >
                        {@chosen_confidence || round(b_conf * 100)}%
                      </output>
                    </div>
                  </:fields>
                </.execute_confirm>
                <.execute_confirm
                  id={belief_confirm_id("retract", belief.id)}
                  open={@confirming == belief_confirm_id("retract", belief.id)}
                  reach={:shared}
                  verb="Retract"
                  target={belief_target(belief)}
                  consequence="Retracts this belief: the belief store stops returning it to every query, Atlas marks it retracted, and its authority's credibility records a contradiction when it has one."
                  on_confirm={JS.push("retract_belief", value: %{id: belief.id})}
                  on_cancel="close_confirm"
                  trigger_id={belief_trigger_id("retract", belief.id)}
                  error={@confirm_error}
                  class="mt-space-sm"
                />
              </div>

              <!-- Metadata row -->
              <div class="flex items-center gap-space-lg mt-space-sm text-caption text-ink-muted">
                <%= if belief.user_id do %>
                  <span>User: {String.slice(belief.user_id, 0, 12)}...</span>
                <% end %>
                <%= if belief.created_at do %>
                  <span>Created: {format_datetime(belief.created_at)}</span>
                <% end %>
                <span class="text-ref">{String.slice(belief.id || "", 0, 8)}</span>
              </div>
            </div>
          <% end %>
        </div>
        <.page_bar
          page={@page}
          page_size={@page_size}
          total={@total_rows}
          matching={@matching_rows}
          event="change_page"
          class="px-space-lg py-space-md border-t border-border"
        />
      <% end %>
    <% end %>
    """
  end

  defp facts_sub_view(assigns) do
    ~H"""
    <!-- Category filter buttons -->
    <%= if length(@categories) > 0 do %>
      <div class="px-space-lg pt-space-md flex flex-wrap gap-space-sm">
        <%= for category <- @categories do %>
          <.btn
            phx-click="beliefs_category_filter"
            phx-value-category={category}
            variant={if(@category_filter == category, do: :secondary, else: :ghost)}
            size={:xs}
          >
            {category}
          </.btn>
        <% end %>
      </div>
    <% end %>

    <%= if is_list(@data) and length(@data) == 0 do %>
      <.empty_panel kind={:plain} words="No facts found" class="m-space-lg" />
    <% else %>
      <%= if is_list(@data) do %>
        <table class="w-full text-left text-body-dense text-ink">
          <thead class="bg-surface-sunk">
            <tr>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Entity</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Fact</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Category</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Confidence</th>
              <th class="h-row-compact px-space-sm text-label text-ink-muted">Source</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-border">
            <%= for fact <- @data do %>
              <tr class="even:bg-surface-sunk">
                <td class="h-row-compact px-space-sm text-value-strong text-ink">{fact.entity}</td>
                <td class="h-row-compact px-space-sm max-w-md">
                  <div class="truncate">{fact.fact}</div>
                </td>
                <td class="h-row-compact px-space-sm">
                  <.badge>{fact.category}</.badge>
                </td>
                <td class="h-row-compact px-space-sm">
                  <span class={confidence_text_class(fact.confidence)}>
                    {format_confidence(fact.confidence)}
                  </span>
                </td>
                <td class="h-row-compact px-space-sm text-caption text-ink-muted">
                  {fact.verification_source || "-"}
                </td>
              </tr>
            <% end %>
          </tbody>
        </table>
        <.page_bar
          page={@page}
          page_size={@page_size}
          total={@total_rows}
          matching={@matching_rows}
          event="change_page"
          class="px-space-lg py-space-md border-t border-border"
        />
      <% end %>
    <% end %>
    """
  end

  defp jtms_sub_view(assigns) do
    ~H"""
    <div class="p-space-lg space-y-space-lg">
      <!-- JTMS Stats -->
      <div class="grid grid-cols-2 md:grid-cols-5 gap-space-md">
        <.stat_kpi label="Total Nodes" value={to_string(Map.get(@stats, :total_nodes, 0))} />
        <.stat_kpi label="IN" value={to_string(Map.get(@stats, :in_count, 0))} />
        <.stat_kpi label="OUT" value={to_string(Map.get(@stats, :out_count, 0))} />
        <.stat_kpi label="Contradictions" value={to_string(Map.get(@stats, :contradiction_count, 0))} />
        <.stat_kpi label="Justifications" value={to_string(Map.get(@stats, :justification_count, 0))} />
      </div>

      <!-- Contradictions list -->
      <div>
        <h3 class="text-subheading text-ink mb-space-sm flex items-center gap-space-sm">
          <.icon name="hero-exclamation-triangle" class="size-4 text-ochre" />
          Active Contradictions
        </h3>
        <%= if length(@contradictions) == 0 do %>
          <.empty_panel kind={:plain} words="No active contradictions" />
        <% else %>
          <div class="space-y-space-sm">
            <%= for node <- @contradictions do %>
              <div class="bg-ochre-wash border border-ochre rounded-md p-space-md">
                <div
                  class="flex items-start justify-between cursor-pointer"
                  phx-click="toggle_expand"
                  phx-value-id={node.id}
                >
                  <div class="flex-1">
                    <div class="text-value-strong text-ink">
                      {inspect(node.datum)}
                    </div>
                    <div class="text-caption text-ink-muted mt-space-xs">
                      Type: {node.node_type} | Label: {node.label} | {length(node.justifications)} justification(s)
                    </div>
                  </div>
                  <.icon
                    name={if @expanded_id == node.id, do: "hero-chevron-up", else: "hero-chevron-down"}
                    class="size-4 text-ink-muted"
                  />
                </div>
                <%= if @expanded_id == node.id do %>
                  <div class="mt-space-md pt-space-md border-t border-ochre text-caption text-ink space-y-space-xs">
                    <div>
                      <span class="text-ink-muted">Node ID:</span>
                      <span class="text-ref">{node.id}</span>
                    </div>
                    <div>
                      <span class="text-ink-muted">Justifications:</span>
                      <span>{Enum.join(node.justifications, ", ")}</span>
                    </div>
                    <div>
                      <span class="text-ink-muted">Consequences:</span>
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
          <h3 class="text-subheading text-ink mb-space-sm">Node Distribution</h3>
          <div class="bg-surface-sunk rounded-md p-space-md">
            <div class="grid grid-cols-2 gap-space-sm text-body">
              <div class="flex justify-between">
                <span class="text-ink-muted">Premises:</span>
                <span class="text-value-strong text-ink">{Map.get(@stats, :premise_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-ink-muted">Assumptions:</span>
                <span class="text-value-strong text-ink">{Map.get(@stats, :assumption_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-ink-muted">Derived:</span>
                <span class="text-value-strong text-ink">{Map.get(@stats, :derived_count, 0)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-ink-muted">Contradiction nodes:</span>
                <span class="text-value-strong text-ink">{Map.get(@stats, :contradiction_count, 0)}</span>
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
    <div class="divide-y divide-border">
      <%= if length(@user_ids) == 0 do %>
        <.empty_panel kind={:plain} words="No user models found" class="m-space-lg" />
      <% else %>
        <!-- User list -->
        <div class="p-space-lg">
          <h3 class="text-subheading text-ink mb-space-md">Users ({length(@user_ids)})</h3>
          <div class="flex flex-wrap gap-space-sm">
            <%= for user_id <- @user_ids do %>
              <.btn
                phx-click="select_user"
                phx-value-user_id={user_id}
                variant={if(@selected_user == user_id, do: :secondary, else: :ghost)}
                size={:sm}
              >
                {String.slice(user_id, 0, 16)}{if String.length(user_id) > 16, do: "...", else: ""}
              </.btn>
            <% end %>
          </div>
        </div>

        <!-- Selected user details -->
        <%= if @selected_user && @user_data do %>
          <div class="p-space-lg space-y-space-lg">
            <h3 class="text-subheading text-ink flex items-center gap-space-sm">
              <.icon name="hero-user" class="size-4 text-ink-muted" />
              User: <span class="text-value-strong text-ink">{@selected_user}</span>
            </h3>

            <!-- Facts table -->
            <% facts = Map.get(@user_data, :facts, %{})
            bounds = Map.get(@user_data, :epistemic_bounds, %{})
            provenance = Map.get(@user_data, :provenance_map, %{}) %>

            <%= if map_size(facts) > 0 do %>
              <div>
                <h4 class="text-label text-ink-muted mb-space-sm">
                  Known Facts ({map_size(facts)})
                </h4>
                <table class="w-full text-left text-body-dense text-ink">
                  <thead class="bg-surface-sunk">
                    <tr>
                      <th class="h-row-compact px-space-sm text-label text-ink-muted">Key</th>
                      <th class="h-row-compact px-space-sm text-label text-ink-muted">Value</th>
                      <th class="h-row-compact px-space-sm text-label text-ink-muted">Confidence</th>
                      <th class="h-row-compact px-space-sm text-label text-ink-muted">Source</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-border">
                    <%= for {key, value} <- Enum.sort(facts) do %>
                      <% conf = Map.get(bounds, key, 0.5)
                      source = Map.get(provenance, key) %>
                      <tr class="even:bg-surface-sunk">
                        <td class="h-row-compact px-space-sm text-value-strong text-ink">{key}</td>
                        <td class="h-row-compact px-space-sm text-value">{inspect(value)}</td>
                        <td class="h-row-compact px-space-sm">
                          <span class={confidence_text_class(conf)}>{format_confidence(conf)}</span>
                        </td>
                        <td class="h-row-compact px-space-sm text-caption text-ink-muted">{source || "-"}</td>
                      </tr>
                    <% end %>
                  </tbody>
                </table>
              </div>
            <% else %>
              <.empty_panel kind={:plain} words="No facts recorded for this user" />
            <% end %>

            <!-- Interaction patterns -->
            <% patterns = Map.get(@user_data, :interaction_patterns, %{}) %>
            <%= if map_size(patterns) > 0 do %>
              <div>
                <h4 class="text-label text-ink-muted mb-space-sm">
                  Interaction Patterns
                </h4>
                <div class="grid grid-cols-2 gap-space-sm">
                  <%= for {pattern_type, data} <- Enum.sort(patterns) do %>
                    <div class="bg-surface-sunk rounded-sm px-space-md py-space-sm text-body">
                      <span class="font-semibold text-ink">{pattern_type}:</span>
                      <span class="text-ink-muted ml-space-xs">{inspect(data)}</span>
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

  defp belief_trigger_id(action, belief_id), do: "explorer-#{action}-belief-#{belief_id}"
  defp belief_confirm_id(action, belief_id), do: "explorer-confirm-#{action}-belief-#{belief_id}"

  defp belief_target(belief), do: "belief #{String.slice(belief.id, 0, 8)}"

  defp format_datetime(%DateTime{} = dt) do
    Calendar.strftime(dt, "%Y-%m-%d %H:%M")
  end

  defp format_datetime(_), do: "-"

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

  # A page from the URL or a page event can be anything; the page shown is the
  # nearest one that exists, from 1 to the last page of the matching rows.
  defp clamp_page(page, total, page_size) do
    last_page = max(div(total + page_size - 1, page_size), 1)
    page |> max(1) |> min(last_page)
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

  defp confidence_text_class(nil), do: "text-value text-ink-muted"
  defp confidence_text_class(c) when is_number(c), do: "text-value text-ink"

  defp format_confidence(nil), do: "—"
  defp format_confidence(c), do: "#{Float.round(c * 100, 1)}%"

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
