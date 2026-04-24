defmodule ChatWeb.AppShell do
  @moduledoc "App shell component with global sidebar navigation.\n\nProvides a consistent layout across all pages with:\n- Global sidebar with navigation links\n- World selector dropdown\n- Current world indicator\n- System status indicator\n- Collapsible on mobile\n"

  alias World.ModelRegistry
  use Phoenix.Component
  use ChatWeb, :verified_routes
  import ChatWeb.CoreComponents

  alias Phoenix.LiveView.JS

  @doc "Renders the app shell with sidebar navigation.\n\n## Examples\n\n    <.app_shell\n      current_world_id={@current_world_id}\n      available_worlds={@available_worlds}\n      current_path={@current_path}\n      system_ready={@system_ready}\n    >\n      <:page_header>\n        <h1>Page Title</h1>\n      </:page_header>\n\n      Page content here\n    </.app_shell>\n"
  attr(:current_world_id, :string, required: true)
  attr(:available_worlds, :list, default: [])
  attr(:current_path, :string, default: "/")
  attr(:system_ready, :boolean, default: true)
  attr(:flash, :map, default: %{})
  attr(:world_models_loading, :boolean, default: false)

  slot(:page_header, doc: "Optional page header content")
  slot(:inner_block, required: true)

  def app_shell(assigns) do
    assigns =
      assign(assigns, :worlds_with_status, add_world_model_status(assigns.available_worlds))

    ~H"""
    <div class="flex h-screen bg-base-200">
      <!-- Mobile Menu Toggle -->
      <div class="lg:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between p-3 bg-base-100 border-b border-base-300">
        <button
          phx-click={toggle_sidebar()}
          class="btn btn-ghost btn-sm btn-square"
        >
          <.icon name="hero-bars-3" class="size-5" />
        </button>
        <span class="text-sm font-semibold">ChatBot</span>
        <div class="w-8" />
      </div>

    <!-- Sidebar -->
      <aside
        id="app-sidebar"
        class="hidden lg:flex flex-col w-64 shrink-0 bg-base-100 border-r border-base-300 fixed lg:relative inset-y-0 left-0 z-40"
      >
        <!-- World Selector -->
        <div class="p-4 border-b border-base-300">
          <div class="flex items-center justify-between mb-2">
            <div class="text-xs font-semibold text-base-content/50 uppercase tracking-wider">
              Training World
            </div>
            <%= if @world_models_loading do %>
              <.icon name="hero-arrow-path" class="size-3 text-warning animate-spin" />
            <% end %>
          </div>
          <form phx-change="switch_world" class="flex gap-2">
            <select
              name="world_id"
              class="select select-sm select-bordered flex-1 font-medium"
            >
              <%= for world <- @worlds_with_status do %>
                <option value={world.id} selected={world.id == @current_world_id}>
                  {world.name}{if world.has_models, do: " ✓", else: ""}
                </option>
              <% end %>
            </select>
            <button
              type="button"
              phx-click="refresh_worlds"
              class="btn btn-sm btn-ghost btn-square"
              title="Refresh worlds"
            >
              <.icon name="hero-arrow-path" class="size-4" />
            </button>
          </form>
          <div class="mt-2 text-[10px] text-base-content/40">
            ✓ = has trained models
          </div>
        </div>

    <!-- Navigation -->
        <nav class="flex-1 overflow-y-auto p-4 space-y-6">
          <!-- Main Section -->
          <div>
            <div class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">
              Main
            </div>
            <ul class="space-y-1">
              <.nav_item
                href={~p"/chat"}
                icon="hero-chat-bubble-left-right"
                label="Chat"
                active={String.starts_with?(@current_path, "/chat")}
              />
              <.nav_item
                href={~p"/explorer"}
                icon="hero-magnifying-glass-circle"
                label="Data Explorer"
                active={String.starts_with?(@current_path, "/explorer")}
              />
            </ul>
          </div>

          <!-- System Section -->
          <div>
            <div class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">
              System
            </div>
            <ul class="space-y-1">
              <.nav_item
                href={~p"/dashboard"}
                icon="hero-chart-bar"
                label="Dashboard"
                active={String.starts_with?(@current_path, "/dashboard")}
              />
              <.nav_item
                href={~p"/code"}
                icon="hero-code-bracket"
                label="Code Analysis"
                active={String.starts_with?(@current_path, "/code")}
              />
              <.nav_item
                href={~p"/accuracy"}
                icon="hero-chart-pie"
                label="Accuracy"
                active={String.starts_with?(@current_path, "/accuracy")}
              />
              <.nav_item
                href={~p"/settings"}
                icon="hero-cog-6-tooth"
                label="Settings"
                active={String.starts_with?(@current_path, "/settings")}
              />
            </ul>
          </div>

          <!-- Admin Section -->
          <div>
            <div class="text-xs font-semibold text-base-content/50 uppercase tracking-wider mb-2">
              Admin
            </div>
            <ul class="space-y-1">
              <.nav_item
                href={~p"/sessions"}
                icon="hero-beaker"
                label="Sessions"
                active={String.starts_with?(@current_path, "/sessions")}
              />
              <.nav_item
                href={~p"/knowledge-review"}
                icon="hero-academic-cap"
                label="Knowledge Review"
                active={String.starts_with?(@current_path, "/knowledge-review")}
              />
            </ul>
          </div>
        </nav>

    <!-- Status Footer -->
        <div class="p-4 border-t border-base-300">
          <div class="flex items-center gap-2 text-xs">
            <%= if @system_ready do %>
              <span class="flex h-2 w-2 rounded-full bg-success"></span>
              <span class="text-base-content/60">All systems ready</span>
            <% else %>
              <span class="flex h-2 w-2 rounded-full bg-warning animate-pulse"></span>
              <span class="text-base-content/60">Initializing...</span>
            <% end %>
          </div>

    <!-- Theme Toggle -->
          <div class="mt-3 flex items-center justify-between">
            <span class="text-xs text-base-content/50">Theme</span>
            <.theme_toggle />
          </div>
        </div>
      </aside>

    <!-- Mobile Sidebar Backdrop -->
      <div
        id="sidebar-backdrop"
        class="hidden fixed inset-0 bg-black/50 z-30 lg:hidden"
        phx-click={toggle_sidebar()}
      />

    <!-- Main Content -->
      <main class="flex-1 flex flex-col min-h-screen lg:min-h-0 overflow-hidden">
        <!-- Page Header (optional) -->
        <%= if @page_header != [] do %>
          <header class="bg-base-100/80 backdrop-blur-lg border-b border-base-300/50 px-4 sm:px-6 py-4 mt-14 lg:mt-0">
            {render_slot(@page_header)}
          </header>
        <% end %>

    <!-- Page Content -->
        <div class={[
          "flex-1 overflow-y-auto",
          if(@page_header == [], do: "mt-14 lg:mt-0", else: "")
        ]}>
          {render_slot(@inner_block)}
        </div>

    <!-- Flash Messages -->
        <.flash_group flash={@flash} />
      </main>
    </div>
    """
  end

  attr(:href, :string, required: true)
  attr(:icon, :string, required: true)
  attr(:label, :string, required: true)
  attr(:active, :boolean, default: false)

  defp nav_item(assigns) do
    ~H"""
    <li>
      <.link
        navigate={@href}
        class={[
          "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
          if(@active,
            do: "bg-primary/10 text-primary",
            else: "text-base-content/70 hover:bg-base-200 hover:text-base-content"
          )
        ]}
      >
        <.icon name={@icon} class="size-5" />
        {@label}
      </.link>
    </li>
    """
  end

  defp theme_toggle(assigns) do
    ~H"""
    <div class="flex items-center gap-1 bg-base-200 rounded-lg p-1">
      <button
        class="p-1.5 rounded hover:bg-base-300 transition-colors"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="system"
        title="System"
      >
        <.icon name="hero-computer-desktop-micro" class="size-3.5 opacity-60 hover:opacity-100" />
      </button>
      <button
        class="p-1.5 rounded hover:bg-base-300 transition-colors"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="light"
        title="Light"
      >
        <.icon name="hero-sun-micro" class="size-3.5 opacity-60 hover:opacity-100" />
      </button>
      <button
        class="p-1.5 rounded hover:bg-base-300 transition-colors"
        phx-click={JS.dispatch("phx:set-theme")}
        data-phx-theme="dark"
        title="Dark"
      >
        <.icon name="hero-moon-micro" class="size-3.5 opacity-60 hover:opacity-100" />
      </button>
    </div>
    """
  end

  attr(:flash, :map, required: true)

  defp flash_group(assigns) do
    ~H"""
    <div class="fixed bottom-4 right-4 z-50 space-y-2">
      <.flash kind={:info} flash={@flash} />
      <.flash kind={:error} flash={@flash} />
    </div>
    """
  end

  defp toggle_sidebar do
    JS.toggle(to: "#app-sidebar", display: "flex")
    |> JS.toggle(to: "#sidebar-backdrop")
  end

  defp add_world_model_status(worlds) when is_list(worlds) do
    Enum.map(worlds, fn world ->
      has_models = ModelRegistry.world_has_models?(world.id)
      Map.put(world, :has_models, has_models)
    end)
  end

  defp add_world_model_status(_) do
    []
  end
end
