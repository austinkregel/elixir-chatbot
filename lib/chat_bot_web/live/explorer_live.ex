defmodule ChatBotWeb.ExplorerLive do
  @moduledoc """
  Unified data explorer for training world data.

  Shows all data types for the currently selected world:
  - Entities (promoted gazetteer entries and candidates)
  - Episodes (episodic memories)
  - Semantic Facts (consolidated knowledge)
  - Knowledge (learned facts and relationships)
  """

  use ChatBotWeb, :live_view
  require Logger

  import ChatBotWeb.AppShell

  alias ChatBot.Learning.{WorldManager, WorldMetrics}
  alias ChatBot.ML.Gazetteer
  alias ChatBot.Memory.Store, as: MemoryStore
  alias ChatBot.KnowledgeStore

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
  end

  defp apply_url_params(socket, params) do
    # Parse tab from URL
    tab =
      case params["tab"] do
        "candidates" -> :candidates
        "episodes" -> :episodes
        "semantics" -> :semantics
        "knowledge" -> :knowledge
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

  defp reload_for_world(socket, world_id) do
    load_world_data(socket, world_id) |> apply_filters()
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
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
          <.stat_card label="Entities" value={length(@overlay)} />
          <.stat_card label="Candidates" value={length(@candidates)} />
          <.stat_card label="Episodes" value={length(@episodes)} />
          <.stat_card label="Semantics" value={length(@semantics)} />
          <.stat_card label="Knowledge" value={map_size(@knowledge)} />
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

  defp format_value(value) when is_binary(value), do: value
  defp format_value(value), do: inspect(value)
end
