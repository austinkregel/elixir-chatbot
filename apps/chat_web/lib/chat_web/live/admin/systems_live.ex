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

  # A slow fallback; the primary update path is the sampler's PubSub push.
  @refresh_ms 10_000

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(Brain.PubSub, Fleet.Systems.Sampler.topic())
      :timer.send_interval(@refresh_ms, self(), :refresh)
    end

    {:ok,
     socket
     |> assign(:page_title, "Systems")
     |> assign(:query_result, nil)
     |> assign_snapshot()}
  end

  @impl true
  # The Admiralty's ad-hoc query — this workbench has no auth, so the human on the
  # console IS the Admiral (a real auth layer would mint a lower principal here).
  # Every query is clearance-gated and audited via Fleet.Systems.query/2.
  def handle_event("query", %{"q" => q}, socket) do
    q = String.trim(q)

    result =
      if q == "" do
        nil
      else
        case safe(fn -> Fleet.Systems.query(Fleet.Principal.admiral(), %{system_id: q}) end) do
          {:ok, %{matches: matches}} -> %{q: q, matches: matches}
          _ -> %{q: q, matches: []}
        end
      end

    {:noreply, assign(socket, :query_result, result)}
  end

  @impl true
  # Live push from the black-box sampler.
  def handle_info({:systems_snapshot, snap}, socket) do
    {:noreply, socket |> assign(:snap, snap) |> assign(:trend, safe(fn -> Fleet.Systems.health_history() end) || [])}
  end

  def handle_info(:refresh, socket), do: {:noreply, assign_snapshot(socket)}
  def handle_info(_msg, socket), do: {:noreply, socket}

  # Cheap: reads the sampler's last snapshot (a lock-free ETS read), not a live probe.
  defp assign_snapshot(socket) do
    snap = safe(fn -> Fleet.Systems.cached_snapshot() end) || %{}

    socket
    |> assign(:snap, snap)
    |> assign(:trend, safe(fn -> Fleet.Systems.health_history() end) || [])
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
          <div :if={length(@trend) >= 2} class="flex flex-col gap-0.5">
            <span class="text-[10px] text-base-content/40 uppercase tracking-wide">health trend</span>
            <.sparkline points={spark_points(@trend)} />
          </div>
          <div class="stats stats-horizontal shadow bg-base-100">
            <.count_stat label="Systems" value={@snap.counts.total} />
            <.count_stat label="Live up" value={@snap.counts.up} cls="text-success" />
            <.count_stat label="Degraded" value={@snap.counts.degraded} cls="text-warning" />
            <.count_stat label="Down" value={@snap.counts.down} cls={@snap.counts.down > 0 && "text-error"} />
            <.count_stat label="Subsystems" value={@snap.counts.subsystems} />
          </div>
        </div>

        <!-- Ad-hoc query — ask any system for its current + recent state -->
        <section class="rounded-lg border border-base-300 bg-base-100 p-3">
          <form phx-submit="query" class="flex items-center gap-2">
            <.icon name="hero-magnifying-glass" class="size-4 text-base-content/50" />
            <input name="q" placeholder="Query a system by id or name (ground truth — clearance-gated, audited)…"
                   autocomplete="off" class="input input-sm input-bordered flex-1" />
            <button class="btn btn-sm btn-primary">Query</button>
          </form>
          <div :if={@query_result} class="mt-3">
            <div :if={@query_result.matches == []} class="text-sm text-base-content/50">
              No live system matches “{@query_result.q}”.
            </div>
            <div :for={m <- @query_result.matches} class="flex items-center gap-3 py-1 border-t border-base-200 first:border-0">
              <span class={["h-2.5 w-2.5 rounded-full shrink-0", dot_class(m.system.status)]}></span>
              <span class="text-sm font-medium">{m.system.name}</span>
              <span class="text-xs text-base-content/50">{m.system.deck}/{m.system.section} · {m.system.status}</span>
              <.sparkline :if={length(m.history) >= 2} points={status_points(m.history)} />
            </div>
          </div>
        </section>

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

  attr :points, :string, required: true

  defp sparkline(assigns) do
    ~H"""
    <svg viewBox="0 0 100 30" class="w-32 h-8 text-primary" preserveAspectRatio="none">
      <polyline fill="none" stroke="currentColor" stroke-width="1.5" points={@points} />
    </svg>
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

  # `trend` is newest-first `[{ts, score}]`; render oldest→newest across x=0..100.
  defp spark_points(trend) do
    scores = trend |> Enum.map(fn {_ts, score} -> score end) |> Enum.reverse()
    n = length(scores)

    scores
    |> Enum.with_index()
    |> Enum.map_join(" ", fn {score, i} ->
      x = if n > 1, do: i / (n - 1) * 100, else: 0
      y = 30 - score / 100 * 30
      "#{Float.round(x, 1)},#{Float.round(y * 1.0, 1)}"
    end)
  end

  # A per-system status ring (newest-first atoms) as a sparkline: up high, down low.
  defp status_points(history) do
    vals = history |> Enum.reverse() |> Enum.map(&status_val/1)
    n = length(vals)

    vals
    |> Enum.with_index()
    |> Enum.map_join(" ", fn {v, i} ->
      x = if n > 1, do: i / (n - 1) * 100, else: 0
      "#{Float.round(x, 1)},#{Float.round((30 - v * 30) * 1.0, 1)}"
    end)
  end

  defp status_val(:up), do: 1.0
  defp status_val(:degraded), do: 0.5
  defp status_val(:down), do: 0.0
  defp status_val(_), do: 0.75

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
