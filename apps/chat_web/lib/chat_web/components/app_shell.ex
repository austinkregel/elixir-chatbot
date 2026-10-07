defmodule ChatWeb.AppShell do
  @moduledoc "App shell component with global sidebar navigation.\n\nProvides a consistent layout across all pages with:\n- The world selector, the shell's first control, and a readout of the selected\n  world that stays visible at every width (in the top bar on narrow screens)\n- Global sidebar with navigation links\n- System status indicator\n- The system, light and dark theme switch (`ChatWeb.Layouts.theme_toggle/1`)\n- Collapsible on mobile\n"

  alias World.ModelRegistry
  use Phoenix.Component
  use ChatWeb, :verified_routes
  import ChatWeb.CoreComponents
  import ChatWeb.UI, only: [status_dot: 1, badge: 1]

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
    worlds = add_world_model_status(assigns.available_worlds)
    current = Enum.find(worlds, &(&1.id == assigns.current_world_id))

    assigns =
      assigns
      |> assign(:worlds_with_status, worlds)
      |> assign(:current_world, current)

    ~H"""
    <div class="flex h-screen bg-ground text-ink">
      <!-- Mobile top bar: the selected world stays visible here when the sidebar is hidden -->
      <div class="lg:hidden fixed top-0 left-0 right-0 z-50 flex items-center justify-between gap-space-sm h-14 px-space-md bg-surface border-b border-border">
        <button
          phx-click={toggle_sidebar()}
          class="inline-flex size-control-md items-center justify-center rounded-md text-ink-muted hover:bg-primary-wash hover:text-primary cursor-pointer"
          aria-label="Open navigation"
        >
          <.icon name="hero-bars-3" class="size-5" />
        </button>
        <span class="text-subheading">ChatBot</span>
        <.world_stamp world_id={@current_world_id} />
      </div>

    <!-- Sidebar -->
      <aside
        id="app-sidebar"
        class="hidden lg:flex flex-col w-64 shrink-0 bg-surface border-r border-border fixed lg:relative inset-y-0 left-0 z-40"
      >
        <!-- World Selector -->
        <div class="p-space-lg border-b border-border">
          <div class="flex items-center justify-between mb-space-xs">
            <div class="text-label text-ink-muted">Training World</div>
            <%= if @world_models_loading do %>
              <.icon name="hero-arrow-path" class="size-3 text-progress-fill motion-safe:animate-spin" />
            <% end %>
          </div>

          <div id="selected-world" class="mb-space-sm">
            <div class="text-subheading text-ink truncate">
              {if @current_world, do: @current_world.name, else: @current_world_id}
            </div>
            <div class="flex flex-wrap items-center gap-space-xs mt-space-2xs">
              <.world_stamp world_id={@current_world_id} />
              <span :if={@current_world && @current_world.has_models} class="text-caption text-ink-muted">
                ✓ trained models
              </span>
              <span :if={is_nil(@current_world)} class="text-caption text-red">
                not in the world list
              </span>
            </div>
          </div>

          <form phx-change="switch_world" class="flex gap-space-xs">
            <label for="world-select" class="sr-only">Switch world</label>
            <select
              id="world-select"
              name="world_id"
              class="min-w-0 flex-1 h-control-md px-space-sm rounded-sm border border-border-strong bg-surface-sunk text-body text-ink cursor-pointer"
            >
              <option :if={is_nil(@current_world)} value={@current_world_id} selected>
                {@current_world_id} (not in the world list)
              </option>
              <%= for world <- @worlds_with_status do %>
                <option value={world.id} selected={world.id == @current_world_id}>
                  {world.name}{if world.has_models, do: " ✓", else: ""}
                </option>
              <% end %>
            </select>
            <button
              type="button"
              phx-click="refresh_worlds"
              class="inline-flex size-control-md shrink-0 items-center justify-center rounded-md text-ink-muted hover:bg-primary-wash hover:text-primary cursor-pointer"
              title="Refresh worlds"
              aria-label="Refresh worlds"
            >
              <.icon name="hero-arrow-path" class="size-4" />
            </button>
          </form>
          <div class="mt-space-xs text-caption text-ink-muted">
            ✓ = has trained models
          </div>
        </div>

    <!-- Navigation -->
        <nav class="flex-1 overflow-y-auto p-space-lg space-y-space-xl">
          <!-- Main Section -->
          <div>
            <div class="text-label text-ink-muted mb-space-sm">
              Main
            </div>
            <ul class="space-y-space-2xs">
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
            <div class="text-label text-ink-muted mb-space-sm">
              System
            </div>
            <ul class="space-y-space-2xs">
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
                href={~p"/lexicon"}
                icon="hero-book-open"
                label="Lexicon"
                active={String.starts_with?(@current_path, "/lexicon")}
              />
              <.nav_item
                href={~p"/training-studio"}
                icon="hero-clipboard-document-list"
                label="Training Studio"
                active={String.starts_with?(@current_path, "/training-studio")}
              />
              <.nav_item
                href={~p"/training/pos"}
                icon="hero-chart-bar"
                label="POS Training"
                active={String.starts_with?(@current_path, "/training/pos")}
              />
              <.nav_item
                href={~p"/verify"}
                icon="hero-check-badge"
                label="Verification"
                active={String.starts_with?(@current_path, "/verify")}
              />
              <.nav_item
                href={~p"/design"}
                icon="hero-swatch"
                label="Design Language"
                active={String.starts_with?(@current_path, "/design")}
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
            <div class="text-label text-ink-muted mb-space-sm">
              Admin
            </div>
            <ul class="space-y-space-2xs">
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
        <div class="p-space-lg border-t border-border">
          <div class="flex items-center gap-space-sm text-caption text-ink-muted">
            <%= if @system_ready do %>
              <.status_dot status={:ready} />
              <span>All systems ready</span>
            <% else %>
              <.status_dot status={:initializing} pulse />
              <span>Initializing...</span>
            <% end %>
          </div>

    <!-- Theme Toggle -->
          <div class="mt-space-md flex items-center justify-between">
            <span class="text-caption text-ink-muted">Theme</span>
            <ChatWeb.Layouts.theme_toggle />
          </div>
        </div>
      </aside>

    <!-- Mobile Sidebar Backdrop -->
      <div
        id="sidebar-backdrop"
        class="hidden fixed inset-0 bg-ground/80 z-30 lg:hidden"
        phx-click={toggle_sidebar()}
      />

    <!-- Main Content -->
      <main class="flex-1 flex flex-col min-h-screen lg:min-h-0 overflow-hidden">
        <!-- Page Header (optional) -->
        <%= if @page_header != [] do %>
          <header class="bg-surface border-b border-border px-space-lg sm:px-space-xl py-space-lg mt-14 lg:mt-0">
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

  attr(:world_id, :string, required: true)

  defp world_stamp(assigns) do
    ~H"""
    <.badge mono>world {@world_id}</.badge>
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
        aria-current={@active && "page"}
        class={[
          "flex items-center gap-space-md h-control-md px-space-md rounded-md text-body transition-colors",
          if(@active,
            do: "bg-accent-wash text-accent font-semibold",
            else: "text-ink-muted hover:bg-surface-sunk hover:text-ink"
          )
        ]}
      >
        <.icon name={@icon} class="size-5" />
        {@label}
      </.link>
    </li>
    """
  end

  attr(:flash, :map, required: true)

  defp flash_group(assigns) do
    ~H"""
    <div class="fixed bottom-space-lg right-space-lg z-50 space-y-space-sm">
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
