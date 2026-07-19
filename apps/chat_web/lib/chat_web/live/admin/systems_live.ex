defmodule ChatWeb.Admin.SystemsLive do
  @moduledoc """
  The ship's black box — a comprehensive, real-time surfacing of every subsystem on
  this instance, as independent ground truth (the LLM systems can hallucinate; this
  is the flight-recorder view that doesn't). Reads `Fleet.Systems.snapshot/0`:
  Layer 0 services, Layer 1 supervised processes, Layer 2 subsystems (200+ modules).

  Live via a 3s refresh (Phase 1); Phase 2 adds the durable sampler + `"systems:status"`
  push + sparklines, and the Admiralty ad-hoc query console.
  """
  use ChatWeb, :live_view
  import ChatWeb.AppShell

  @refresh_ms 3000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: :timer.send_interval(@refresh_ms, self(), :refresh)

    {:ok,
     socket
     |> assign(:page_title, "Systems")
     |> assign_snapshot()}
  end

  @impl true
  def handle_info(:refresh, socket), do: {:noreply, assign_snapshot(socket)}
  def handle_info(_msg, socket), do: {:noreply, socket}

  defp assign_snapshot(socket) do
    snap = safe(fn -> Fleet.Systems.snapshot() end) || %{}
    assign(socket, :snap, snap)
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
        <div class="flex items-center gap-3 flex-wrap">
          <.icon name="hero-cpu-chip" class="size-6 text-primary" />
          <h1 class="text-xl font-bold">Systems — {@snap[:ship_id]}</h1>
          <span class="text-xs text-base-content/50">black box · ground truth</span>
        </div>
      </:page_header>

      <div :if={@snap == %{}} class="p-8 text-center text-base-content/50">Sensors offline — no snapshot.</div>

      <div :if={@snap != %{}} class="p-4 space-y-6">
        <!-- Health + counts -->
        <div class="flex flex-wrap items-center gap-6">
          <div class={["radial-progress", health_color(@snap.health[:health_status])]}
               style={"--value:#{@snap.health[:health_score] || 0}; --size:5rem;"} role="progressbar">
            {@snap.health[:health_score]}%
          </div>
          <div class="stats stats-horizontal shadow bg-base-100">
            <.count_stat label="Systems" value={@snap.counts.total} />
            <.count_stat label="Live up" value={@snap.counts.up} cls="text-success" />
            <.count_stat label="Degraded" value={@snap.counts.degraded} cls="text-warning" />
            <.count_stat label="Down" value={@snap.counts.down} cls={@snap.counts.down > 0 && "text-error"} />
            <.count_stat label="Subsystems" value={@snap.counts.subsystems} />
          </div>
        </div>

        <!-- Layer 0: services -->
        <section>
          <h2 class="text-sm font-semibold text-base-content/60 uppercase tracking-wide mb-2">Services</h2>
          <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
            <.cell :for={s <- @snap.services} sys={s} />
          </div>
        </section>

        <!-- Layer 1: processes, non-nominal first -->
        <section>
          <h2 class="text-sm font-semibold text-base-content/60 uppercase tracking-wide mb-2">
            Processes <span class="text-base-content/40">({length(@snap.processes)})</span>
          </h2>
          <div :if={down_procs(@snap) != []} class="mb-2">
            <div class="text-xs text-error mb-1">Not nominal</div>
            <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
              <.cell :for={s <- down_procs(@snap)} sys={s} />
            </div>
          </div>
          <div class="flex flex-wrap gap-1">
            <span :for={s <- up_procs(@snap)} class={["h-3 w-3 rounded-sm", dot_class(s.status)]}
                  title={"#{s.name} · #{s.deck}/#{s.section} · #{s.status}"}></span>
          </div>
        </section>

        <!-- Layer 2: subsystems by deck -->
        <section>
          <h2 class="text-sm font-semibold text-base-content/60 uppercase tracking-wide mb-2">
            Subsystems <span class="text-base-content/40">({@snap.counts.subsystems} modules)</span>
          </h2>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            <details :for={{deck, sections} <- by_deck(@snap.subsystems)} class="collapse collapse-arrow bg-base-100 border border-base-300">
              <summary class="collapse-title text-sm font-medium py-2 min-h-0">
                {deck} <span class="badge badge-ghost badge-sm ml-1">{Enum.reduce(sections, 0, fn {_, n}, a -> a + n end)}</span>
              </summary>
              <div class="collapse-content text-xs">
                <div :for={{section, n} <- sections} class="flex justify-between py-0.5">
                  <span class="text-base-content/70 font-mono">{section}</span>
                  <span class="text-base-content/40">{n}</span>
                </div>
              </div>
            </details>
          </div>
        </section>
      </div>
    </.app_shell>
    """
  end

  # ── components ──────────────────────────────────────────────────────────────

  attr :label, :string, required: true
  attr :value, :any, required: true
  attr :cls, :any, default: nil

  defp count_stat(assigns) do
    ~H"""
    <div class="stat py-2 px-4">
      <div class="stat-title text-xs">{@label}</div>
      <div class={["stat-value text-lg", @cls]}>{@value}</div>
    </div>
    """
  end

  attr :sys, :map, required: true

  defp cell(assigns) do
    ~H"""
    <div class="flex items-center gap-2 rounded-lg border border-base-300 bg-base-100 px-3 py-2"
         title={"#{@sys.deck}/#{@sys.section} · #{@sys.status}"}>
      <span class={["h-2.5 w-2.5 rounded-full shrink-0", dot_class(@sys.status)]}></span>
      <span class="text-sm truncate">{@sys.name}</span>
      <span :if={@sys.key_metric} class="badge badge-ghost badge-xs ml-auto">{@sys.key_metric}</span>
    </div>
    """
  end

  # ── helpers ─────────────────────────────────────────────────────────────────

  defp down_procs(snap), do: Enum.reject(snap.processes, &(&1.status == :up))
  defp up_procs(snap), do: Enum.filter(snap.processes, &(&1.status == :up))

  defp by_deck(subsystems) do
    subsystems
    |> Enum.group_by(& &1.deck)
    |> Enum.map(fn {deck, systems} ->
      sections =
        systems
        |> Enum.group_by(& &1.section)
        |> Enum.map(fn {sec, list} -> {sec, length(list)} end)
        |> Enum.sort_by(fn {_, n} -> -n end)

      {deck, sections}
    end)
    |> Enum.sort_by(fn {_, secs} -> -Enum.reduce(secs, 0, fn {_, n}, a -> a + n end) end)
  end

  defp dot_class(:up), do: "bg-success"
  defp dot_class(:degraded), do: "bg-warning animate-pulse"
  defp dot_class(:down), do: "bg-error animate-pulse"
  defp dot_class(:static), do: "bg-base-300"
  defp dot_class(:idle), do: "bg-base-300"
  defp dot_class(_), do: "bg-base-content/20"

  defp health_color(:healthy), do: "text-success"
  defp health_color(:degraded), do: "text-warning"
  defp health_color(:warning), do: "text-warning"
  defp health_color(:critical), do: "text-error"
  defp health_color(_), do: "text-base-content"

  defp safe(fun) do
    fun.()
  rescue
    _ -> nil
  catch
    _, _ -> nil
  end
end
