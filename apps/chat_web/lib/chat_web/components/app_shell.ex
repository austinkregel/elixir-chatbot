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
      <!-- Command bar -->
      <header class="flex items-center gap-4 h-12 px-5 bg-base-100/95 backdrop-blur border-b border-base-300 shrink-0">
        <!-- Ship identity -->
        <div class="flex items-center gap-2.5 shrink-0">
          <div class="grid place-items-center size-7 rounded-md bg-primary/10 text-primary ring-1 ring-primary/20">
            <.icon name="hero-rocket-launch" class="size-4" />
          </div>
          <div class="leading-none hidden sm:block">
            <div class="text-sm font-semibold tracking-[0.18em] uppercase">{ship_name()}</div>
            <div class="text-[10px] text-base-content/40 tracking-[0.2em] uppercase">Command Net</div>
          </div>
        </div>

        <div class="h-6 w-px bg-base-300 shrink-0"></div>

    <!-- Nav -->
        <nav class="flex items-center gap-5 shrink-0 text-xs font-medium uppercase tracking-wider">
          <.link navigate="/" class={nav_class(@current_path, ["/", "/fleet"])}>Bridge</.link>
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
            <span class="relative flex size-2">
              <span class="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 animate-ping"></span>
              <span class="relative inline-flex size-2 rounded-full bg-success"></span>
            </span>
            <span class="text-base-content/50 hidden md:inline uppercase tracking-wider text-[11px]">Nominal</span>
          <% else %>
            <span class="flex size-2 rounded-full bg-warning animate-pulse"></span>
            <span class="text-base-content/50 hidden md:inline uppercase tracking-wider text-[11px]">Initializing</span>
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

  # Dehyphenated ship id for the wordmark, e.g. "USS-DREAMCOM" -> "USS DREAMCOM".
  defp ship_name, do: Fleet.Ship.id() |> to_string() |> String.replace("-", " ")

  defp nav_class(current, matches) do
    base = "pb-0.5 border-b-2 -mb-px transition-colors"

    if current in matches,
      do: base <> " border-primary text-primary",
      else: base <> " border-transparent text-base-content/50 hover:text-base-content"
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
