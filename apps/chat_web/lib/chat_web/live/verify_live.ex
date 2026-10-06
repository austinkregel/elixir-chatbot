defmodule ChatWeb.VerifyLive do
  @moduledoc """
  The `/verify` index: every subsystem that should have a verification page,
  with how many hand-checked cases pass, fail, raised, or have never run.

  Task 039's point is that a feature nobody can exercise in isolation is a
  feature nobody can check. This page is the inventory of that, and it is
  deliberately built to be **uncomfortable while it is empty** — a subsystem
  with no page and no cases is listed with zeros and the task number that would
  build it, rather than being absent. An index that only showed what exists
  would report an all-clear on day one.

  Counts come from `Atlas.Verification.counts_by_subsystem/1`, scoped to the
  selected world, because a subsystem's answer depends on which world's models
  are loaded and so a case verified in one world is not evidence about another.

  ## No page links yet

  None of the 18 pages exist (tasks 040-057). Rather than linking to routes that
  are not there, each row says which task builds it. When a page lands it gets a
  route and this page starts linking to it.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Atlas.Verification
  alias Atlas.Verification.Subsystems
  alias ChatWeb.Harness.Diff

  @impl true
  def mount(_params, _session, socket) do
    {:ok, load_counts(socket)}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, load_counts(socket)}
  end

  @impl true
  def handle_event("refresh", _params, socket) do
    {:noreply, load_counts(socket)}
  end

  defp load_counts(socket) do
    world_id = socket.assigns.current_world_id
    counts = Verification.counts_by_subsystem(world_id: world_id)

    rows =
      Enum.map(Subsystems.all(), fn subsystem ->
        subsystem_counts = Map.get(counts, subsystem.id, zero_counts())

        Map.merge(subsystem, %{
          counts: subsystem_counts,
          total: subsystem_counts |> Map.values() |> Enum.sum()
        })
      end)

    # Orphans: cases whose subsystem is no longer declared. Shown rather than
    # dropped, because a declaration that changed under existing data leaves
    # those cases unreachable, and the only way anyone finds out is if the index
    # says so.
    orphans =
      counts
      |> Map.drop(Subsystems.ids())
      |> Enum.map(fn {id, subsystem_counts} ->
        %{id: id, counts: subsystem_counts, total: subsystem_counts |> Map.values() |> Enum.sum()}
      end)

    socket
    |> assign(:rows, rows)
    |> assign(:orphans, orphans)
    |> assign(:totals, totals(rows))
  end

  defp zero_counts, do: %{"pending" => 0, "pass" => 0, "fail" => 0, "error" => 0}

  defp totals(rows) do
    base = %{
      cases: 0,
      pass: 0,
      fail: 0,
      error: 0,
      pending: 0,
      subsystems_with_cases: 0
    }

    Enum.reduce(rows, base, fn row, acc ->
      %{
        cases: acc.cases + row.total,
        pass: acc.pass + row.counts["pass"],
        fail: acc.fail + row.counts["fail"],
        error: acc.error + row.counts["error"],
        pending: acc.pending + row.counts["pending"],
        subsystems_with_cases:
          acc.subsystems_with_cases + if(row.total > 0, do: 1, else: 0)
      }
    end)
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
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 class="text-lg font-semibold">Verification</h1>
            <p class="text-sm text-base-content/60">
              Every subsystem that should be exercisable in isolation, and whether anyone has.
            </p>
          </div>
          <.btn variant={:outline} size={:sm} phx-click="refresh">Refresh</.btn>
        </div>
      </:page_header>

      <div class="space-y-6 p-4 sm:p-6">
        <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <.stat_kpi label="Subsystems" value={to_string(length(@rows))} />
          <.stat_kpi label="With any case" value={to_string(@totals.subsystems_with_cases)} />
          <.stat_kpi label="Cases" value={to_string(@totals.cases)} />
          <.stat_kpi label="Passing" value={to_string(@totals.pass)} variant={:success} />
          <.stat_kpi label="Failing" value={to_string(@totals.fail)} variant={:error} />
          <.stat_kpi label="Never run" value={to_string(@totals.pending)} variant={:warning} />
        </div>

        <.card :if={@totals.cases == 0} class="border-warning/40 bg-warning/5">
          <.card_body>
            <p class="text-sm">
              No verification cases exist in world
              <span class="font-mono">{@current_world_id}</span>. Nothing below has been
              checked by hand, and the zeros are the honest reading — not an all-clear.
            </p>
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-3">
            <.section_header>Subsystems</.section_header>

            <div class="overflow-x-auto">
              <table class="w-full text-left text-sm">
                <thead class="text-xs uppercase tracking-wider text-base-content/50">
                  <tr>
                    <th class="py-2 pr-4 font-semibold">Subsystem</th>
                    <th class="py-2 pr-4 font-semibold">Page</th>
                    <th class="py-2 pr-4 text-right font-semibold">Pass</th>
                    <th class="py-2 pr-4 text-right font-semibold">Fail</th>
                    <th class="py-2 pr-4 text-right font-semibold">Raised</th>
                    <th class="py-2 pr-4 text-right font-semibold">Never run</th>
                    <th class="py-2 font-semibold">State</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-base-300">
                  <tr :for={row <- @rows}>
                    <td class="py-2 pr-4">
                      <div class="font-medium">{row.title}</div>
                      <div class="font-mono text-[11px] text-base-content/50">{row.id}</div>
                    </td>
                    <td class="py-2 pr-4 text-xs text-base-content/60">
                      not built — task {row.task}
                    </td>
                    <td class="py-2 pr-4 text-right font-mono">{row.counts["pass"]}</td>
                    <td class="py-2 pr-4 text-right font-mono">{row.counts["fail"]}</td>
                    <td class="py-2 pr-4 text-right font-mono">{row.counts["error"]}</td>
                    <td class="py-2 pr-4 text-right font-mono">{row.counts["pending"]}</td>
                    <td class="py-2">
                      <.badge :if={row.total == 0} variant={:warning} size={:xs}>
                        unverified
                      </.badge>
                      <Diff.verdict :if={row.total > 0} status={state_for(row.counts)} />
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </.card_body>
        </.card>

        <.card :if={@orphans != []} class="border-error/40 bg-error/5">
          <.card_body class="space-y-2">
            <.section_header>Cases with no declared subsystem</.section_header>
            <p class="text-xs text-base-content/60">
              These were saved against a subsystem that is no longer declared in
              <span class="font-mono">Atlas.Verification.Subsystems</span>, so no page can run
              them. They are listed rather than dropped: the declaration changed under existing
              data, and counting them is how anyone finds out.
            </p>
            <ul class="space-y-1 text-sm">
              <li :for={orphan <- @orphans} class="font-mono text-xs">
                {orphan.id} — {orphan.total} case(s)
              </li>
            </ul>
          </.card_body>
        </.card>
      </div>
    </.app_shell>
    """
  end

  # The worst state present wins, so a subsystem is not reported as passing
  # while one of its cases raises. "pending" ranks above "pass" for the same
  # reason: a case nobody has run is not evidence of correctness.
  defp state_for(%{"error" => error}) when error > 0, do: "error"
  defp state_for(%{"fail" => fail}) when fail > 0, do: "fail"
  defp state_for(%{"pending" => pending}) when pending > 0, do: "pending"
  defp state_for(_counts), do: "pass"
end
