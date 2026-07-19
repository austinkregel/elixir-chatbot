defmodule ChatWeb.AppShell do
  @moduledoc "Fleet app shell: a compact top bar (training-world context + system\nstatus + theme) over a full-width content area. This is a Fleet-management\nworkbench — there is a single destination (the Fleet console), so there is no\nmulti-page sidebar; the shell just carries the world context and status the\nconsole needs.\n"

  use Phoenix.Component
  use ChatWeb, :verified_routes
  import ChatWeb.CoreComponents

  alias Phoenix.LiveView.JS

  @doc "Renders the Fleet shell: top bar + optional page header + content.\n\n## Examples\n\n    <.app_shell\n      current_world_id={@current_world_id}\n      available_worlds={@available_worlds}\n      current_path={@current_path}\n      system_ready={@system_ready}\n    >\n      <:page_header>\n        <h1>Fleet</h1>\n      </:page_header>\n\n      Console content here\n    </.app_shell>\n"
  attr(:current_world_id, :string, required: true)
  attr(:available_worlds, :list, default: [])
  attr(:current_path, :string, default: "/")
  attr(:system_ready, :boolean, default: true)
  attr(:flash, :map, default: %{})
  attr(:world_models_loading, :boolean, default: false)

  slot(:page_header, doc: "Optional page header content (title + actions)")
  slot(:inner_block, required: true)

  def app_shell(assigns) do
    ~H"""
    <div class="flex flex-col h-screen bg-base-200">
      <!-- Fleet top bar -->
      <header class="flex items-center gap-3 px-4 py-2 bg-base-100 border-b border-base-300 shrink-0">
        <div class="flex items-center gap-1.5 font-semibold shrink-0">
          <.icon name="hero-rocket-launch" class="size-5 text-primary" />
        </div>

    <!-- Nav: the two views of this Fleet workbench -->
        <nav class="flex items-center gap-1 shrink-0 text-sm">
          <.link navigate="/" class={nav_class(@current_path, ["/", "/fleet"])}>Fleet</.link>
          <.link navigate="/systems" class={nav_class(@current_path, ["/systems"])}>Systems</.link>
        </nav>

    <!-- Optional page header (title + actions), else a spacer -->
        <%= if @page_header != [] do %>
          <div class="flex-1 min-w-0">{render_slot(@page_header)}</div>
        <% else %>
          <div class="flex-1"></div>
        <% end %>

    <!-- System status -->
        <div class="flex items-center gap-1.5 text-xs shrink-0">
          <%= if @system_ready do %>
            <span class="flex h-2 w-2 rounded-full bg-success"></span>
            <span class="text-base-content/60 hidden sm:inline">ready</span>
          <% else %>
            <span class="flex h-2 w-2 rounded-full bg-warning animate-pulse"></span>
            <span class="text-base-content/60 hidden sm:inline">initializing…</span>
          <% end %>
        </div>

        <.theme_toggle />
      </header>

    <!-- Full-width content -->
      <main class="flex-1 overflow-y-auto">
        {render_slot(@inner_block)}
      </main>

    <!-- Flash Messages -->
      <.flash_group flash={@flash} />
    </div>
    """
  end

  defp nav_class(current, matches) do
    base = "px-2 py-1 rounded transition-colors"

    if current in matches,
      do: base <> " bg-primary/10 text-primary font-medium",
      else: base <> " text-base-content/60 hover:text-base-content hover:bg-base-200"
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
end
