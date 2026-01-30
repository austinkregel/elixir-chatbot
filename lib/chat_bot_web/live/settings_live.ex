defmodule ChatBotWeb.SettingsLive do
  @moduledoc """
  Settings page for world management and entity administration.

  Features:
  - World management (create, delete, configure)
  - Gazetteer entity management
  - System configuration
  """

  use ChatBotWeb, :live_view
  require Logger

  import ChatBotWeb.AppShell

  alias ChatBot.Learning.{WorldManager, WorldPersistence}
  alias ChatBot.Knowledge.{LearningCenter, TaskSource}
  alias ChatBot.Knowledge.Types.Investigation
  alias ChatBot.Testing.CapabilityTest
  alias ChatBot.ML.Gazetteer

  @impl true
  def mount(_params, _session, socket) do
    {:ok, socket}
  end

  @impl true
  def handle_params(params, _uri, socket) do
    # Determine which section to show
    section =
      case params["section"] do
        "entities" -> :entities
        "worlds" -> :worlds
        "training" -> :training
        "testing" -> :testing
        _ -> :worlds
      end

    socket =
      socket
      |> assign(:section, section)
      |> assign(:new_world_name, "")
      |> assign(:new_world_mode, "persistent")
      |> assign(:creating_world, false)
      |> assign(:entity_search, "")
      |> assign(:selected_entity_type, nil)
      |> assign(:new_entity_key, "")
      |> assign(:new_entity_value, "")
      |> assign(:new_entity_type, "location")
      # Training section assigns
      |> assign(:training_sessions, [])
      |> assign(:available_tasks, %{})
      |> assign(:selected_capability, :all)
      |> assign(:starting_training, false)
      |> assign(:tasks_loading, false)
      |> assign(:lc_stats, %{total_sessions: 0, active_agents: 0})
      # Testing section assigns
      |> assign(:test_capability, :question_answering)
      |> assign(:running_test, false)
      |> assign(:test_results, nil)
      |> assign(:capability_stats, %{})
      |> load_section_data()

    {:noreply, socket}
  end

  defp load_section_data(socket) do
    case socket.assigns.section do
      :worlds -> load_worlds_data(socket)
      :entities -> load_entities_data(socket)
      :training -> load_training_data(socket)
      :testing -> load_testing_data(socket)
    end
  end

  defp load_worlds_data(socket) do
    worlds =
      try do
        WorldManager.list_worlds()
      rescue
        _ -> []
      end

    persisted =
      try do
        WorldPersistence.list_persisted_worlds()
      rescue
        _ -> []
      end

    socket
    |> assign(:worlds, worlds)
    |> assign(:persisted_worlds, persisted)
  end

  defp load_entities_data(socket) do
    world_id = socket.assigns.current_world_id

    # Get global entities grouped by type
    entity_types = Gazetteer.list_types()

    # Get world overlay
    world_overlay = Gazetteer.get_world_overlay(world_id)

    socket
    |> assign(:entity_types, entity_types)
    |> assign(:world_overlay, world_overlay)
    |> assign(:selected_entity_type, List.first(entity_types))
    |> load_type_entities()
  end

  defp load_type_entities(socket) do
    type = socket.assigns[:selected_entity_type]

    entities =
      if type do
        Gazetteer.list_by_type(type)
        |> Enum.map(fn {key, info} ->
          # Convert tuple to map for template compatibility
          %{
            key: key,
            value: Map.get(info, :value) || Map.get(info, :original) || key,
            source: Map.get(info, :source, "unknown"),
            entity_type: Map.get(info, :entity_type) || Map.get(info, :type)
          }
        end)
      else
        []
      end

    assign(socket, :type_entities, entities)
  end

  defp load_training_data(socket) do
    # Load active training sessions
    sessions =
      try do
        LearningCenter.list_sessions()
      rescue
        _ -> []
      catch
        :exit, _ -> []
      end

    # Get Learning Center stats
    lc_stats =
      try do
        LearningCenter.stats()
      rescue
        _ -> %{total_sessions: 0, active_agents: 0}
      catch
        :exit, _ -> %{total_sessions: 0, active_agents: 0}
      end

    # Load available task categories in background to not block UI
    # Start with empty map, then load async
    socket =
      socket
      |> assign(:training_sessions, sessions)
      |> assign(:lc_stats, lc_stats)
      |> assign(:available_tasks, socket.assigns[:available_tasks] || %{})
      |> assign(:tasks_loading, true)

    # Spawn async task to load available tasks (heavy operation)
    if connected?(socket) do
      self_pid = self()

      Task.start(fn ->
        available =
          try do
            case TaskSource.available_tasks() do
              {:ok, grouped} -> grouped
              _ -> %{}
            end
          rescue
            _ -> %{}
          catch
            :exit, _ -> %{}
          end

        send(self_pid, {:tasks_loaded, available})
      end)
    end

    socket
  end

  defp load_testing_data(socket) do
    # Load capability statistics
    capability_stats =
      try do
        CapabilityTest.task_stats()
      rescue
        _ -> %{}
      catch
        :exit, _ -> %{}
      end

    socket
    |> assign(:capability_stats, capability_stats)
    |> assign(:test_results, nil)
    |> assign(:running_test, false)
  end

  # ============================================================================
  # Event Handlers - World Context
  # ============================================================================

  @impl true
  def handle_event("switch_world", %{"world_id" => _world_id}, socket) do
    # World context hook already updated current_world_id, reload section data
    {:noreply, load_section_data(socket)}
  end

  def handle_event("refresh_worlds", _params, socket) do
    # World context hook already refreshed available_worlds
    {:noreply, socket}
  end

  def handle_event("switch_section", %{"section" => section}, socket) do
    {:noreply, push_patch(socket, to: ~p"/settings?section=#{section}")}
  end

  def handle_event("update_new_world", %{"name" => name, "mode" => mode}, socket) do
    {:noreply, socket |> assign(:new_world_name, name) |> assign(:new_world_mode, mode)}
  end

  def handle_event("create_world", _params, socket) do
    name = socket.assigns.new_world_name
    mode = String.to_existing_atom(socket.assigns.new_world_mode)

    if name != "" do
      socket = assign(socket, :creating_world, true)

      case WorldManager.create(name, mode: mode, base_world: "default") do
        {:ok, world} ->
          socket =
            socket
            |> assign(:creating_world, false)
            |> assign(:new_world_name, "")
            |> load_worlds_data()
            |> put_flash(:info, "Created world: #{world.name}")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply,
           socket
           |> assign(:creating_world, false)
           |> put_flash(:error, "Failed to create world: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "World name is required")}
    end
  end

  @impl true
  def handle_event("delete_world", %{"id" => world_id}, socket) do
    if world_id != "default" do
      case WorldManager.destroy(world_id) do
        :ok ->
          {:noreply,
           socket |> load_worlds_data() |> put_flash(:info, "Deleted world: #{world_id}")}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to delete world: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Cannot delete the default world")}
    end
  end

  @impl true
  def handle_event("save_world", %{"id" => world_id}, socket) do
    case WorldManager.checkpoint(world_id) do
      :ok ->
        {:noreply, put_flash(socket, :info, "World saved: #{world_id}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to save world: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_event("load_world", %{"id" => _world_id}, socket) do
    case WorldManager.reload_persisted_worlds() do
      :ok ->
        {:noreply, socket |> load_worlds_data() |> put_flash(:info, "Reloaded persisted worlds")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to reload: #{inspect(reason)}")}
    end
  end

  # ============================================================================
  # Event Handlers - Entities
  # ============================================================================

  @impl true
  def handle_event("select_entity_type", %{"type" => type}, socket) do
    {:noreply, socket |> assign(:selected_entity_type, type) |> load_type_entities()}
  end

  @impl true
  def handle_event("search_entities", %{"query" => query}, socket) do
    {:noreply, assign(socket, :entity_search, query)}
  end

  @impl true
  def handle_event("update_new_entity", params, socket) do
    socket =
      socket
      |> assign(:new_entity_key, params["key"] || socket.assigns.new_entity_key)
      |> assign(:new_entity_value, params["value"] || socket.assigns.new_entity_value)
      |> assign(:new_entity_type, params["type"] || socket.assigns.new_entity_type)

    {:noreply, socket}
  end

  @impl true
  def handle_event("add_entity", _params, socket) do
    world_id = socket.assigns.current_world_id
    key = socket.assigns.new_entity_key
    value = socket.assigns.new_entity_value
    type = socket.assigns.new_entity_type

    if key != "" do
      case Gazetteer.add_to_world(world_id, key, type, %{
             value: if(value == "", do: key, else: value),
             source: :admin,
             added_at: DateTime.utc_now()
           }) do
        :ok ->
          {:noreply,
           socket
           |> assign(:new_entity_key, "")
           |> assign(:new_entity_value, "")
           |> load_entities_data()
           |> put_flash(:info, "Added entity: #{key}")}

        {:error, reason} ->
          {:noreply, put_flash(socket, :error, "Failed to add entity: #{inspect(reason)}")}
      end
    else
      {:noreply, put_flash(socket, :error, "Entity key is required")}
    end
  end

  @impl true
  def handle_event("remove_entity", %{"key" => key}, socket) do
    world_id = socket.assigns.current_world_id

    case Gazetteer.remove_from_world(world_id, key) do
      :ok ->
        {:noreply, socket |> load_entities_data() |> put_flash(:info, "Removed entity: #{key}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to remove entity: #{inspect(reason)}")}
    end
  end

  @impl true
  def handle_event("refresh", _params, socket) do
    {:noreply, load_section_data(socket)}
  end

  # ============================================================================
  # Event Handlers - Training
  # ============================================================================

  def handle_event("select_capability", %{"capability" => capability}, socket) do
    capability = String.to_existing_atom(capability)
    {:noreply, assign(socket, :selected_capability, capability)}
  end

  def handle_event("start_task_training", _params, socket) do
    capability = socket.assigns.selected_capability
    socket = assign(socket, :starting_training, true)

    case LearningCenter.start_task_training(capability, max_tasks: 5) do
      {:ok, session} ->
        socket =
          socket
          |> assign(:starting_training, false)
          |> load_training_data()
          |> put_flash(:info, "Started training session: #{session.id}")

        {:noreply, socket}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:starting_training, false)
         |> put_flash(:error, "Failed to start training: #{inspect(reason)}")}
    end
  end

  def handle_event("cancel_session", %{"id" => session_id}, socket) do
    case LearningCenter.cancel_session(session_id) do
      :ok ->
        {:noreply,
         socket |> load_training_data() |> put_flash(:info, "Cancelled session: #{session_id}")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to cancel session: #{inspect(reason)}")}
    end
  end

  # ============================================================================
  # Event Handlers - Capability Testing
  # ============================================================================

  def handle_event("select_test_capability", %{"capability" => capability}, socket) do
    capability_atom = String.to_existing_atom(capability)
    {:noreply, assign(socket, :test_capability, capability_atom)}
  end

  def handle_event("run_capability_test", _params, socket) do
    capability = socket.assigns.test_capability
    socket = assign(socket, :running_test, true)

    # Run test async to not block UI
    self_pid = self()

    Task.start(fn ->
      result =
        try do
          CapabilityTest.test_capability(capability,
            limit: 10,
            max_tasks: 3,
            verbose: false
          )
        rescue
          e -> {:error, Exception.message(e)}
        catch
          :exit, reason -> {:error, inspect(reason)}
        end

      send(self_pid, {:test_complete, result})
    end)

    {:noreply, socket}
  end

  def handle_event("clear_test_results", _params, socket) do
    {:noreply, assign(socket, :test_results, nil)}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    # World was changed from another LiveView or tab
    {:noreply, load_section_data(socket)}
  end

  def handle_info({:tasks_loaded, available}, socket) do
    {:noreply,
     socket
     |> assign(:available_tasks, available)
     |> assign(:tasks_loading, false)}
  end

  def handle_info({:test_complete, result}, socket) do
    case result do
      {:ok, investigation} ->
        summary = Investigation.summary(investigation)

        {:noreply,
         socket
         |> assign(:running_test, false)
         |> assign(:test_results, %{
           investigation: investigation,
           summary: summary
         })
         |> put_flash(:info, "Capability test complete")}

      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:running_test, false)
         |> put_flash(:error, "Test failed: #{inspect(reason)}")}
    end
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
            <h1 class="text-xl font-bold">Settings</h1>
            <p class="text-sm text-base-content/60">Manage worlds and entities</p>
          </div>
          <button phx-click="refresh" class="btn btn-ghost btn-sm">
            <.icon name="hero-arrow-path" class="size-4" /> Refresh
          </button>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6">
        <!-- Section Tabs -->
        <div class="tabs tabs-boxed mb-6">
          <button
            phx-click="switch_section"
            phx-value-section="worlds"
            class={["tab gap-1", if(@section == :worlds, do: "tab-active", else: "")]}
          >
            <.icon name="hero-globe-alt" class="size-4" /> Worlds
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="entities"
            class={["tab gap-1", if(@section == :entities, do: "tab-active", else: "")]}
          >
            <.icon name="hero-tag" class="size-4" /> Entities
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="training"
            class={["tab gap-1", if(@section == :training, do: "tab-active", else: "")]}
          >
            <.icon name="hero-academic-cap" class="size-4" /> Training
          </button>
          <button
            phx-click="switch_section"
            phx-value-section="testing"
            class={["tab gap-1", if(@section == :testing, do: "tab-active", else: "")]}
          >
            <.icon name="hero-beaker" class="size-4" /> Testing
          </button>
        </div>
        
    <!-- Content -->
        <%= case @section do %>
          <% :worlds -> %>
            <.worlds_section
              worlds={@worlds}
              persisted_worlds={@persisted_worlds}
              new_world_name={@new_world_name}
              new_world_mode={@new_world_mode}
              creating_world={@creating_world}
            />
          <% :entities -> %>
            <.entities_section
              entity_types={@entity_types}
              type_entities={@type_entities}
              world_overlay={@world_overlay}
              selected_entity_type={@selected_entity_type}
              entity_search={@entity_search}
              new_entity_key={@new_entity_key}
              new_entity_value={@new_entity_value}
              new_entity_type={@new_entity_type}
              current_world_id={@current_world_id}
            />
          <% :training -> %>
            <.training_section
              training_sessions={@training_sessions}
              available_tasks={@available_tasks}
              selected_capability={@selected_capability}
              starting_training={@starting_training}
              tasks_loading={@tasks_loading}
              lc_stats={@lc_stats}
            />
          <% :testing -> %>
            <.testing_section
              test_capability={@test_capability}
              running_test={@running_test}
              test_results={@test_results}
              capability_stats={@capability_stats}
            />
        <% end %>
      </div>
    </.app_shell>
    """
  end

  # ============================================================================
  # Section Components
  # ============================================================================

  defp worlds_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Create World -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Create New World</h3>
        <form phx-change="update_new_world" phx-submit="create_world" class="flex flex-wrap gap-4">
          <input
            type="text"
            name="name"
            value={@new_world_name}
            placeholder="World name"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <select name="mode" class="select select-bordered">
            <option value="persistent" selected={@new_world_mode == "persistent"}>Persistent</option>
            <option value="ephemeral" selected={@new_world_mode == "ephemeral"}>Ephemeral</option>
          </select>
          <button type="submit" class="btn btn-primary" disabled={@creating_world}>
            <%= if @creating_world do %>
              <span class="loading loading-spinner loading-sm"></span>
            <% else %>
              <.icon name="hero-plus" class="size-4" />
            <% end %>
            Create World
          </button>
        </form>
      </div>
      
    <!-- Active Worlds -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Active Worlds</h3>
        </div>
        <%= if length(@worlds) == 0 do %>
          <div class="p-8 text-center text-base-content/50">
            <.icon name="hero-globe-alt" class="size-12 mx-auto mb-4 text-base-content/30" />
            <p>No active worlds</p>
          </div>
        <% else %>
          <div class="divide-y divide-base-300/50">
            <%= for world <- @worlds do %>
              <div class="p-4 flex items-center justify-between hover:bg-base-200/50">
                <div>
                  <div class="font-medium">{world.name}</div>
                  <div class="text-sm text-base-content/60 font-mono">{world.id}</div>
                </div>
                <div class="flex items-center gap-2">
                  <span class={[
                    "badge badge-sm",
                    if(world.mode == :persistent, do: "badge-info", else: "badge-ghost")
                  ]}>
                    {world.mode}
                  </span>
                  <%= if world.mode == :persistent do %>
                    <button
                      phx-click="save_world"
                      phx-value-id={world.id}
                      class="btn btn-ghost btn-xs"
                      title="Save to disk"
                    >
                      <.icon name="hero-cloud-arrow-up" class="size-4" />
                    </button>
                  <% end %>
                  <%= if world.id != "default" do %>
                    <button
                      phx-click="delete_world"
                      phx-value-id={world.id}
                      class="btn btn-ghost btn-xs text-error"
                      title="Delete world"
                      data-confirm="Are you sure you want to delete this world?"
                    >
                      <.icon name="hero-trash" class="size-4" />
                    </button>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      
    <!-- Persisted Worlds (not loaded) -->
      <% not_loaded =
        Enum.filter(@persisted_worlds, fn pw -> not Enum.any?(@worlds, &(&1.id == pw.id)) end) %>
      <%= if length(not_loaded) > 0 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50">
          <div class="p-4 border-b border-base-300">
            <h3 class="font-semibold">Persisted Worlds (Not Loaded)</h3>
          </div>
          <div class="divide-y divide-base-300/50">
            <%= for world <- not_loaded do %>
              <div class="p-4 flex items-center justify-between hover:bg-base-200/50">
                <div>
                  <div class="font-medium text-base-content/70">{world.name}</div>
                  <div class="text-sm text-base-content/50 font-mono">{world.id}</div>
                </div>
                <button
                  phx-click="load_world"
                  phx-value-id={world.id}
                  class="btn btn-ghost btn-xs"
                >
                  <.icon name="hero-arrow-down-tray" class="size-4" /> Load
                </button>
              </div>
            <% end %>
          </div>
        </div>
      <% end %>
    </div>
    """
  end

  defp entities_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Add Entity -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">
          Add Entity to World: <span class="text-primary">{@current_world_id}</span>
        </h3>
        <form phx-change="update_new_entity" phx-submit="add_entity" class="flex flex-wrap gap-4">
          <input
            type="text"
            name="key"
            value={@new_entity_key}
            placeholder="Lookup key (e.g., 'new york')"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <input
            type="text"
            name="value"
            value={@new_entity_value}
            placeholder="Canonical value (optional)"
            class="input input-bordered flex-1 min-w-[200px]"
          />
          <select name="type" class="select select-bordered">
            <%= for type <- @entity_types do %>
              <option value={type} selected={@new_entity_type == type}>{type}</option>
            <% end %>
          </select>
          <button type="submit" class="btn btn-primary">
            <.icon name="hero-plus" class="size-4" /> Add
          </button>
        </form>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <!-- Entity Type Sidebar -->
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <h3 class="font-semibold mb-4">Entity Types</h3>
          <ul class="space-y-1">
            <%= for type <- @entity_types do %>
              <li>
                <button
                  phx-click="select_entity_type"
                  phx-value-type={type}
                  class={[
                    "w-full text-left px-3 py-2 rounded-lg text-sm transition-colors",
                    if(type == @selected_entity_type,
                      do: "bg-primary/10 text-primary font-medium",
                      else: "hover:bg-base-200"
                    )
                  ]}
                >
                  {type}
                </button>
              </li>
            <% end %>
          </ul>
        </div>
        
    <!-- Entities List -->
        <div class="lg:col-span-3 bg-base-100 rounded-xl border border-base-300/50">
          <div class="p-4 border-b border-base-300 flex items-center gap-4">
            <h3 class="font-semibold">{@selected_entity_type || "Select a type"}</h3>
            <input
              type="text"
              placeholder="Search entities..."
              value={@entity_search}
              phx-keyup="search_entities"
              name="query"
              phx-debounce="150"
              class="input input-sm input-bordered flex-1 max-w-xs"
            />
          </div>
          <%= if length(@type_entities) == 0 do %>
            <div class="p-8 text-center text-base-content/50">
              <.icon name="hero-tag" class="size-12 mx-auto mb-4 text-base-content/30" />
              <p>No entities of this type</p>
            </div>
          <% else %>
            <% filtered = filter_entities(@type_entities, @entity_search) %>
            <div class="max-h-96 overflow-y-auto">
              <table class="table table-sm">
                <thead class="bg-base-200/50 sticky top-0">
                  <tr>
                    <th>Key</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  <%= for entity <- Enum.take(filtered, 100) do %>
                    <% is_from_overlay = is_overlay_entity(entity, @world_overlay) %>
                    <tr class="hover:bg-base-200/30">
                      <td class="font-medium">{entity.key}</td>
                      <td>{entity.value || entity.key}</td>
                      <td>
                        <span class={[
                          "badge badge-xs",
                          if(is_from_overlay, do: "badge-primary", else: "badge-ghost")
                        ]}>
                          {if is_from_overlay, do: "world", else: "global"}
                        </span>
                      </td>
                      <td>
                        <%= if is_from_overlay do %>
                          <button
                            phx-click="remove_entity"
                            phx-value-key={entity.key}
                            class="btn btn-ghost btn-xs text-error"
                            title="Remove from world"
                          >
                            <.icon name="hero-x-mark" class="size-4" />
                          </button>
                        <% end %>
                      </td>
                    </tr>
                  <% end %>
                </tbody>
              </table>
              <%= if length(filtered) > 100 do %>
                <div class="p-4 text-center text-sm text-base-content/60">
                  Showing 100 of {length(filtered)} entities
                </div>
              <% end %>
            </div>
          <% end %>
        </div>
      </div>
    </div>
    """
  end

  defp training_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Stats Overview -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
              <.icon name="hero-academic-cap" class="size-5 text-primary" />
            </div>
            <div>
              <div class="text-2xl font-bold">{@lc_stats[:total_sessions] || 0}</div>
              <div class="text-sm text-base-content/60">Total Sessions</div>
            </div>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-warning/10 flex items-center justify-center">
              <.icon name="hero-cpu-chip" class="size-5 text-warning" />
            </div>
            <div>
              <div class="text-2xl font-bold">{@lc_stats[:active_agents] || 0}</div>
              <div class="text-sm text-base-content/60">Active Agents</div>
            </div>
          </div>
        </div>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
              <.icon name="hero-document-text" class="size-5 text-success" />
            </div>
            <div>
              <div class="text-2xl font-bold">{map_size(@available_tasks)}</div>
              <div class="text-sm text-base-content/60">Task Categories</div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Start Training -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Start Task-Based Training</h3>
        <p class="text-sm text-base-content/60 mb-4">
          Train child agents using curated NLP benchmark tasks. Select a capability to focus the training.
        </p>
        <div class="flex flex-wrap gap-4 items-end">
          <div class="form-control">
            <label class="label">
              <span class="label-text">Capability</span>
            </label>
            <select
              phx-change="select_capability"
              name="capability"
              class="select select-bordered"
            >
              <option value="all" selected={@selected_capability == :all}>All Capabilities</option>
              <option value="question_answering" selected={@selected_capability == :question_answering}>
                Question Answering
              </option>
              <option value="commonsense" selected={@selected_capability == :commonsense}>
                Commonsense Reasoning
              </option>
              <option value="sentiment" selected={@selected_capability == :sentiment}>
                Sentiment Analysis
              </option>
              <option value="reasoning" selected={@selected_capability == :reasoning}>
                Explanation & Reasoning
              </option>
            </select>
          </div>
          <button
            phx-click="start_task_training"
            class="btn btn-primary"
            disabled={@starting_training}
          >
            <%= if @starting_training do %>
              <span class="loading loading-spinner loading-sm"></span>
            <% else %>
              <.icon name="hero-play" class="size-4" />
            <% end %>
            Start Training
          </button>
        </div>
      </div>
      
      <!-- Active Sessions -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Training Sessions</h3>
        </div>
        <%= if length(@training_sessions) == 0 do %>
          <div class="p-8 text-center text-base-content/50">
            <.icon name="hero-academic-cap" class="size-12 mx-auto mb-4 text-base-content/30" />
            <p>No active training sessions</p>
            <p class="text-sm mt-2">Start a training session above to begin</p>
          </div>
        <% else %>
          <div class="divide-y divide-base-300/50">
            <%= for session <- @training_sessions do %>
              <div class="p-4 hover:bg-base-200/50">
                <div class="flex items-center justify-between mb-2">
                  <div>
                    <div class="font-medium">{session.topic}</div>
                    <div class="text-sm text-base-content/60 font-mono">{session.id}</div>
                  </div>
                  <div class="flex items-center gap-2">
                    <%= if session.status == :active do %>
                      <button
                        phx-click="cancel_session"
                        phx-value-id={session.id}
                        class="btn btn-ghost btn-xs text-error"
                        title="Cancel session"
                      >
                        <.icon name="hero-stop" class="size-4" /> Cancel
                      </button>
                    <% end %>
                  </div>
                </div>
                <div class="flex flex-wrap gap-2">
                  <span class={["badge badge-sm", session_status_badge(session.status)]}>
                    {session.status}
                  </span>
                  <span class="text-xs text-base-content/50">
                    {length(session.goals)} goal(s)
                  </span>
                  <%= if session.hypotheses_tested > 0 do %>
                    <span class="badge badge-sm badge-outline" title="Scientific Investigation">
                      <.icon name="hero-beaker" class="size-3 mr-1" />
                      {session.hypotheses_tested} tested
                    </span>
                    <%= if session.hypotheses_supported > 0 do %>
                      <span class="badge badge-sm badge-success badge-outline">
                        {session.hypotheses_supported} supported
                      </span>
                    <% end %>
                    <%= if session.hypotheses_falsified > 0 do %>
                      <span class="badge badge-sm badge-error badge-outline">
                        {session.hypotheses_falsified} falsified
                      </span>
                    <% end %>
                  <% end %>
                </div>
              </div>
            <% end %>
          </div>
        <% end %>
      </div>
      
      <!-- Available Task Categories -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Available Task Categories</h3>
          <p class="text-sm text-base-content/60">Domain-specific NLP tasks from benchmarks</p>
        </div>
        <%= if @tasks_loading do %>
          <div class="p-8 text-center text-base-content/50">
            <span class="loading loading-spinner loading-lg text-primary"></span>
            <p class="mt-4">Scanning task files...</p>
            <p class="text-sm mt-2">This may take a moment on first load</p>
          </div>
        <% else %>
          <%= if map_size(@available_tasks) == 0 do %>
            <div class="p-8 text-center text-base-content/50">
              <.icon name="hero-document-text" class="size-12 mx-auto mb-4 text-base-content/30" />
              <p>No tasks available</p>
              <p class="text-sm mt-2">Check that domain task files are in data/domain_specific_tasks/</p>
            </div>
          <% else %>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
              <%= for {category, tasks} <- @available_tasks do %>
                <div class="bg-base-200/50 rounded-lg p-3">
                  <div class="flex items-center justify-between mb-2">
                    <span class="font-medium text-sm">{category}</span>
                    <span class="badge badge-sm badge-ghost">{length(tasks)} tasks</span>
                  </div>
                  <div class="text-xs text-base-content/60">
                    <%= for task <- Enum.take(tasks, 3) do %>
                      <div class="truncate">{task.task_id}</div>
                    <% end %>
                    <%= if length(tasks) > 3 do %>
                      <div class="text-primary">+{length(tasks) - 3} more</div>
                    <% end %>
                  </div>
                </div>
              <% end %>
            </div>
          <% end %>
        <% end %>
      </div>
    </div>
    """
  end

  defp testing_section(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Scientific Method Overview -->
      <div class="bg-gradient-to-r from-primary/10 to-secondary/10 rounded-xl border border-base-300/50 p-6">
        <div class="flex items-center gap-3 mb-4">
          <div class="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
            <.icon name="hero-beaker" class="size-6 text-primary" />
          </div>
          <div>
            <h2 class="text-xl font-bold">Capability Testing</h2>
            <p class="text-sm text-base-content/60">
              Test NLP capabilities using the scientific method
            </p>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div class="bg-base-100/50 rounded-lg p-3">
            <div class="font-medium text-primary">Hypothesis</div>
            <div class="text-xs text-base-content/60">Testable claim from question</div>
          </div>
          <div class="bg-base-100/50 rounded-lg p-3">
            <div class="font-medium text-warning">Evidence</div>
            <div class="text-xs text-base-content/60">Test results from tasks</div>
          </div>
          <div class="bg-base-100/50 rounded-lg p-3">
            <div class="font-medium text-success">Supported</div>
            <div class="text-xs text-base-content/60">Evidence confirms hypothesis</div>
          </div>
          <div class="bg-base-100/50 rounded-lg p-3">
            <div class="font-medium text-error">Falsified</div>
            <div class="text-xs text-base-content/60">Evidence contradicts hypothesis</div>
          </div>
        </div>
      </div>

      <!-- Run Capability Test -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
        <h3 class="font-semibold mb-4">Run Capability Test</h3>
        <p class="text-sm text-base-content/60 mb-4">
          Test a specific NLP capability against curated benchmark tasks. 
          Results are evaluated using the scientific method.
        </p>
        <div class="flex flex-wrap gap-4 items-end">
          <div class="form-control">
            <label class="label">
              <span class="label-text">Capability to Test</span>
            </label>
            <select
              phx-change="select_test_capability"
              name="capability"
              class="select select-bordered"
            >
              <option value="question_answering" selected={@test_capability == :question_answering}>
                Question Answering
              </option>
              <option value="entity_recognition" selected={@test_capability == :entity_recognition}>
                Entity Recognition
              </option>
              <option value="sentiment" selected={@test_capability == :sentiment}>
                Sentiment Analysis
              </option>
              <option value="classification" selected={@test_capability == :classification}>
                Classification
              </option>
              <option value="temporal_reasoning" selected={@test_capability == :temporal_reasoning}>
                Temporal Reasoning
              </option>
              <option value="coreference" selected={@test_capability == :coreference}>
                Coreference Resolution
              </option>
              <option value="commonsense" selected={@test_capability == :commonsense}>
                Commonsense Reasoning
              </option>
            </select>
          </div>
          <button
            phx-click="run_capability_test"
            class="btn btn-primary"
            disabled={@running_test}
          >
            <%= if @running_test do %>
              <span class="loading loading-spinner loading-sm"></span>
              Running Test...
            <% else %>
              <.icon name="hero-play" class="size-4" />
              Run Test
            <% end %>
          </button>
        </div>
      </div>

      <!-- Test Results -->
      <%= if @test_results do %>
        <.test_results_panel results={@test_results} />
      <% end %>

      <!-- Available Task Categories -->
      <div class="bg-base-100 rounded-xl border border-base-300/50">
        <div class="p-4 border-b border-base-300">
          <h3 class="font-semibold">Task Category Statistics</h3>
          <p class="text-sm text-base-content/60">
            <%= map_size(@capability_stats) %> categories across 1600+ benchmark tasks
          </p>
        </div>
        <%= if map_size(@capability_stats) > 0 do %>
          <div class="p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 max-h-64 overflow-y-auto">
            <%= for {category, count} <- Enum.sort_by(@capability_stats, fn {_, c} -> -c end) |> Enum.take(16) do %>
              <div class="flex items-center justify-between bg-base-200/50 rounded-lg px-3 py-2">
                <span class="text-sm truncate" title={category}>{category}</span>
                <span class="badge badge-sm badge-primary">{count}</span>
              </div>
            <% end %>
          </div>
        <% else %>
          <div class="p-8 text-center text-base-content/50">
            <p>Loading task statistics...</p>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  defp test_results_panel(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50">
      <div class="p-4 border-b border-base-300 flex items-center justify-between">
        <div>
          <h3 class="font-semibold">Investigation Results</h3>
          <p class="text-sm text-base-content/60">{@results.summary.topic}</p>
        </div>
        <button
          phx-click="clear_test_results"
          class="btn btn-ghost btn-xs"
        >
          <.icon name="hero-x-mark" class="size-4" /> Clear
        </button>
      </div>
      
      <!-- Scientific Summary -->
      <div class="p-4 grid grid-cols-2 md:grid-cols-5 gap-4">
        <div class="text-center">
          <div class="text-2xl font-bold">{@results.summary.total_hypotheses}</div>
          <div class="text-xs text-base-content/60">Hypotheses</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-success">{@results.summary.supported}</div>
          <div class="text-xs text-base-content/60">Supported</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-error">{@results.summary.falsified}</div>
          <div class="text-xs text-base-content/60">Falsified</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-warning">{@results.summary.inconclusive}</div>
          <div class="text-xs text-base-content/60">Inconclusive</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold">{@results.summary.evidence_count}</div>
          <div class="text-xs text-base-content/60">Evidence</div>
        </div>
      </div>
      
      <!-- Conclusion Badge -->
      <div class="px-4 pb-4">
        <div class={[
          "alert",
          conclusion_alert_class(@results.summary.conclusion)
        ]}>
          <.icon name={conclusion_icon(@results.summary.conclusion)} class="size-5" />
          <div>
            <div class="font-medium">Conclusion: {format_conclusion(@results.summary.conclusion)}</div>
            <div class="text-sm opacity-75">
              <%= case @results.summary.conclusion do %>
                <% :hypotheses_supported -> %>
                  All tested hypotheses are supported by evidence
                <% :hypotheses_falsified -> %>
                  Evidence contradicts the hypotheses - improvement needed
                <% :inconclusive -> %>
                  Insufficient or mixed evidence - more testing required
                <% :mixed -> %>
                  Some hypotheses supported, others falsified
                <% _ -> %>
                  Test completed
              <% end %>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Hypothesis Details -->
      <div class="border-t border-base-300 divide-y divide-base-300/50">
        <%= for hypothesis <- @results.investigation.hypotheses do %>
          <div class="p-4 flex items-start gap-4">
            <div class={[
              "w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0",
              hypothesis_badge_bg(hypothesis.status)
            ]}>
              <.icon name={hypothesis_icon(hypothesis.status)} class="size-4" />
            </div>
            <div class="flex-1 min-w-0">
              <div class="font-medium truncate" title={hypothesis.claim}>{hypothesis.claim}</div>
              <div class="text-sm text-base-content/60">
                Derived from: {hypothesis.derived_from || "N/A"}
              </div>
              <div class="flex flex-wrap gap-2 mt-2">
                <span class={["badge badge-sm", hypothesis_badge(hypothesis.status)]}>
                  {hypothesis.status}
                </span>
                <span class="badge badge-sm badge-ghost">
                  Confidence: {Float.round(hypothesis.confidence * 100, 1)}%
                </span>
                <span class="badge badge-sm badge-ghost">
                  Sources: {hypothesis.source_count}
                </span>
              </div>
            </div>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  # Result styling helpers
  defp conclusion_alert_class(:hypotheses_supported), do: "alert-success"
  defp conclusion_alert_class(:hypotheses_falsified), do: "alert-error"
  defp conclusion_alert_class(:inconclusive), do: "alert-warning"
  defp conclusion_alert_class(:mixed), do: "alert-info"
  defp conclusion_alert_class(_), do: "alert-info"

  defp conclusion_icon(:hypotheses_supported), do: "hero-check-circle"
  defp conclusion_icon(:hypotheses_falsified), do: "hero-x-circle"
  defp conclusion_icon(:inconclusive), do: "hero-question-mark-circle"
  defp conclusion_icon(:mixed), do: "hero-exclamation-triangle"
  defp conclusion_icon(_), do: "hero-information-circle"

  defp format_conclusion(:hypotheses_supported), do: "Supported"
  defp format_conclusion(:hypotheses_falsified), do: "Falsified"
  defp format_conclusion(:inconclusive), do: "Inconclusive"
  defp format_conclusion(:mixed), do: "Mixed Results"
  defp format_conclusion(other), do: to_string(other)

  defp hypothesis_icon(:supported), do: "hero-check"
  defp hypothesis_icon(:falsified), do: "hero-x-mark"
  defp hypothesis_icon(:inconclusive), do: "hero-question-mark-circle"
  defp hypothesis_icon(:testing), do: "hero-arrow-path"
  defp hypothesis_icon(:untested), do: "hero-minus"
  defp hypothesis_icon(_), do: "hero-minus"

  defp hypothesis_badge(:supported), do: "badge-success"
  defp hypothesis_badge(:falsified), do: "badge-error"
  defp hypothesis_badge(:inconclusive), do: "badge-warning"
  defp hypothesis_badge(:testing), do: "badge-info"
  defp hypothesis_badge(_), do: "badge-ghost"

  defp hypothesis_badge_bg(:supported), do: "bg-success/20 text-success"
  defp hypothesis_badge_bg(:falsified), do: "bg-error/20 text-error"
  defp hypothesis_badge_bg(:inconclusive), do: "bg-warning/20 text-warning"
  defp hypothesis_badge_bg(:testing), do: "bg-info/20 text-info"
  defp hypothesis_badge_bg(_), do: "bg-base-200 text-base-content/50"

  defp session_status_badge(:active), do: "badge-warning"
  defp session_status_badge(:completed), do: "badge-success"
  defp session_status_badge(:cancelled), do: "badge-error"
  defp session_status_badge(_), do: "badge-ghost"

  # ============================================================================
  # Helpers
  # ============================================================================

  defp filter_entities(entities, ""), do: entities

  defp filter_entities(entities, search) do
    search = String.downcase(search)

    Enum.filter(entities, fn entity ->
      String.contains?(String.downcase(entity.key || ""), search) ||
        String.contains?(String.downcase(entity.value || ""), search)
    end)
  end

  defp is_overlay_entity(entity, overlay) do
    Enum.any?(overlay, fn {key, _info} ->
      String.downcase(key) == String.downcase(entity.key || "")
    end)
  end
end
