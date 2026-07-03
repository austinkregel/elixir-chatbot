defmodule ChatWeb.Admin.FleetLive do
  @moduledoc """
  The Admiral's console: see the crew live and drive it. Commission agents from
  souls, wire the chain of command, issue orders, relieve/reinstate, retire, and
  drill into any agent's accountability record (service history + duty log).
  """
  use ChatWeb, :live_view

  import ChatWeb.AppShell
  import Ecto.Query, only: [from: 2]

  alias Phoenix.PubSub
  alias Atlas.Schemas.CommandRecord
  require Logger

  @refresh_ms 3000
  @feed_cap 60

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      PubSub.subscribe(Brain.PubSub, "fleet:events")
      :timer.send_interval(@refresh_ms, self(), :refresh)
    end

    {:ok,
     socket
     |> assign(:page_title, "Fleet")
     |> assign(:souls, load_souls())
     |> assign(:souls_dir, safe(fn -> Brain.Soul.souls_dir() end))
     |> assign(:billets, Fleet.Rank.all())
     |> assign(:feed, backfill_feed())
     |> assign(:selected, nil)
     |> assign(:detail, nil)
     # Controlled select values, so the 3s roster refresh never resets a form
     # the Admiral is mid-way through filling out.
     |> assign(:picks, %{})
     |> assign_roster()}
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
        <div class="flex items-center gap-2">
          <.icon name="hero-rocket-launch" class="size-6 text-primary" />
          <h1 class="text-xl font-bold">Fleet — Admiral's Console</h1>
          <span class="badge badge-neutral">{length(@roster)} crew</span>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6 space-y-6">
        <!-- Stat bar -->
        <div class="stats stats-horizontal shadow bg-base-100 w-full overflow-x-auto">
          <div class="stat">
            <div class="stat-title">Crew</div>
            <div class="stat-value text-primary">{length(@roster)}</div>
          </div>
          <div class="stat">
            <div class="stat-title">Active / Relieved</div>
            <div class="stat-value text-success">{count(@roster, &(&1.duty == :active))}</div>
            <div class="stat-desc">{count(@roster, &(&1.duty == :relieved))} relieved</div>
          </div>
          <div class="stat">
            <div class="stat-title">Working</div>
            <div class="stat-value">{count(@roster, & &1.working)}</div>
          </div>
        </div>

        <!-- Command panel -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div class="card bg-base-100 shadow">
            <div class="card-body p-4">
              <h2 class="card-title text-sm">Commission</h2>
              <form phx-submit="commission" class="space-y-2">
                <select name="soul_id" phx-change="pick" class="select select-sm select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["soul_id"])}>choose a soul…</option>
                  <option :for={sid <- @souls} value={sid} selected={sid == @picks["soul_id"]}>{sid}</option>
                </select>
                <p :if={@souls == []} class="text-xs text-warning">
                  No souls found in <code>{@souls_dir}</code>. Add a soul file (e.g. <code>souls/ensign-jj7.json</code>) and refresh.
                </p>
                <select name="rank" phx-change="pick" class="select select-sm select-bordered w-full">
                  <option :for={{key, meta} <- @billets} value={key} selected={to_string(key) == picked_rank(@picks)}>
                    {meta.label}
                  </option>
                </select>
                <p class="text-xs text-base-content/60">{Fleet.Rank.describe(picked_rank(@picks))}</p>
                <p class="text-xs">
                  <span class="text-base-content/50">Standing authorities:</span>
                  <%= case Fleet.Rank.standing_authorities(picked_rank(@picks)) do %>
                    <% [] -> %><span class="text-base-content/40">none (cognition is order-conferred)</span>
                    <% auths -> %><span :for={a <- auths} class="badge badge-ghost badge-xs ml-1">{fmt_grant(a)}</span>
                  <% end %>
                </p>
                <button class="btn btn-primary btn-sm w-full">Commission</button>
              </form>
            </div>
          </div>

          <div class="card bg-base-100 shadow">
            <div class="card-body p-4">
              <h2 class="card-title text-sm">Issue order</h2>
              <form phx-submit="issue_order" class="space-y-2">
                <select name="agent_id" phx-change="pick" class="select select-sm select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["agent_id"])}>target agent…</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["agent_id"]}>{a.soul_id}</option>
                </select>
                <input name="directive" class="input input-sm input-bordered w-full"
                       placeholder="directive, e.g. Summarize sector 7 readiness." required />
                <button class="btn btn-sm btn-outline w-full">Order</button>
              </form>
            </div>
          </div>

          <div class="card bg-base-100 shadow">
            <div class="card-body p-4">
              <h2 class="card-title text-sm">Assign CO</h2>
              <form phx-submit="assign_co" class="space-y-2">
                <select name="report_id" phx-change="pick" class="select select-sm select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["report_id"])}>report…</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["report_id"]}>{a.soul_id}</option>
                </select>
                <select name="co_id" phx-change="pick" class="select select-sm select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["co_id"])}>commanding officer…</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["co_id"]}>{a.soul_id}</option>
                </select>
                <button class="btn btn-sm btn-outline w-full">Wire chain</button>
              </form>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <!-- Roster -->
          <div class="xl:col-span-2 card bg-base-100 shadow">
            <div class="card-body p-4">
              <h2 class="card-title text-sm">Crew roster</h2>
              <div :if={@roster == []} class="text-base-content/60 text-sm py-6 text-center">
                No crew commissioned. Commission a soul above.
              </div>
              <div class="overflow-x-auto">
                <table :if={@roster != []} class="table table-sm">
                  <thead>
                    <tr><th>Agent</th><th>Duty</th><th>CO</th><th>Assignment</th><th>Grants</th><th></th></tr>
                  </thead>
                  <tbody>
                    <tr :for={a <- @roster} class="hover">
                      <td>
                        <div class="font-medium">{a.soul_id}</div>
                        <div class="text-xs text-base-content/50">{Fleet.Rank.label(a.rank)} · {length(a.reports)} reports · up {div(a.uptime_ms, 1000)}s</div>
                      </td>
                      <td>
                        <span class={["badge badge-sm", duty_class(a.duty)]}>{a.duty}</span>
                        <span :if={a.working} class="loading loading-spinner loading-xs ml-1"></span>
                        <span :if={not a.soul_loaded} class="badge badge-ghost badge-xs ml-1">no soul</span>
                      </td>
                      <td class="text-xs">{a.co || "—"}</td>
                      <td class="max-w-[18rem]">
                        <span :if={a.assignment_status} class={["badge badge-sm", status_class(a.assignment_status)]}>
                          {a.assignment_status}
                        </span>
                        <span :if={is_nil(a.assignment_status)} class="text-base-content/40">idle</span>
                        <div :if={a[:directive]} class="text-xs text-base-content/50 truncate" title={a[:directive]}>“{a.directive}”</div>
                      </td>
                      <td>
                        <span :for={g <- a.grants} class="badge badge-ghost badge-xs mr-1">{fmt_grant(g)}</span>
                      </td>
                      <td class="text-right whitespace-nowrap">
                        <button class="btn btn-ghost btn-xs" phx-click="detail" phx-value-agent={a.agent_id}>Details</button>
                        <button :if={a.duty == :active} class="btn btn-ghost btn-xs text-warning"
                          phx-click="relieve" phx-value-agent={a.agent_id} phx-value-co={a.co || ""}>Relieve</button>
                        <button :if={a.duty == :relieved} class="btn btn-ghost btn-xs text-success"
                          phx-click="reinstate" phx-value-agent={a.agent_id} phx-value-co={a.co || ""}>Reinstate</button>
                        <button class="btn btn-ghost btn-xs text-error"
                          phx-click="retire" phx-value-agent={a.agent_id}
                          data-confirm="Retire this agent?">Retire</button>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Activity feed -->
          <div class="card bg-base-100 shadow">
            <div class="card-body p-4">
              <h2 class="card-title text-sm">Activity</h2>
              <div class="space-y-1 max-h-[28rem] overflow-y-auto text-xs font-mono">
                <div :for={e <- @feed} class="flex items-center gap-2">
                  <span class="text-base-content/40 shrink-0">{e.at}</span>
                  <span class={["badge badge-xs shrink-0", event_class(e.event)]}>{e.event}</span>
                  <span class="truncate">{e.agent}<span :if={e.order_id} class="text-base-content/40"> · {String.slice(e.order_id, 0, 6)}</span></span>
                </div>
                <div :if={@feed == []} class="text-base-content/50 py-4 text-center">No activity yet.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Detail modal -->
      <div :if={@detail} class="modal modal-open">
        <div class="modal-box max-w-2xl">
          <h3 class="font-bold text-lg flex items-center gap-2">
            {@detail.soul_id}
            <span class={["badge badge-sm", duty_class(@detail.duty)]}>{@detail.duty}</span>
          </h3>
          <div class="stats stats-horizontal shadow my-3 w-full">
            <div class="stat py-2"><div class="stat-title text-xs">completed</div><div class="stat-value text-lg text-success">{@detail.completed}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">dissented</div><div class="stat-value text-lg text-warning">{@detail.dissented}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">failed</div><div class="stat-value text-lg text-error">{@detail.failed}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">reliefs</div><div class="stat-value text-lg">{@detail.reliefs}</div></div>
          </div>

          <div :if={@detail.directive} class="my-3">
            <h4 class="font-semibold text-sm mb-1">Current order</h4>
            <div class="bg-base-200 rounded p-2 text-sm">
              <span :if={@detail.assignment_status} class={["badge badge-xs mr-1", status_class(@detail.assignment_status)]}>{@detail.assignment_status}</span>
              “{@detail.directive}”
            </div>
            <div :if={@detail.report} class="mt-2">
              <div class="text-xs font-semibold text-base-content/60 mb-1">Report — what the agent produced</div>
              <div class="bg-base-100 border border-base-300 rounded p-2 text-sm whitespace-pre-wrap break-words">{report_text(@detail.report)}</div>
            </div>
            <details :if={@detail.timeline != []} class="mt-2">
              <summary class="text-xs font-semibold text-base-content/60 cursor-pointer">Command timeline</summary>
              <div class="mt-1 space-y-1 text-xs max-h-40 overflow-y-auto">
              <div :for={r <- @detail.timeline} class="flex gap-2 items-start">
                <span class={["badge badge-xs shrink-0", event_class(safe_atom(r.kind))]}>{r.kind}</span>
                <span class="text-base-content/60 break-words">{timeline_desc(r)}</span>
              </div>
              </div>
            </details>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h4 class="font-semibold text-sm mb-1">Service record</h4>
              <ul class="text-xs space-y-1 max-h-48 overflow-y-auto">
                <li :for={r <- @detail.history} class="flex gap-2">
                  <span class={["badge badge-xs", event_class(String.to_atom(r.kind))]}>{r.kind}</span>
                  <span class="text-base-content/60 truncate">{r.reason || r.outcome || ""}</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 class="font-semibold text-sm mb-1">Duty log (agent's own)</h4>
              <ul class="text-xs space-y-1 max-h-48 overflow-y-auto">
                <li :for={n <- @detail.notes} class="text-base-content/70">• {n.note}</li>
                <li :if={@detail.notes == []} class="text-base-content/40">no notes</li>
              </ul>
            </div>
          </div>

          <div class="modal-action">
            <button class="btn btn-sm" phx-click="close_detail">Close</button>
          </div>
        </div>
        <label class="modal-backdrop" phx-click="close_detail">close</label>
      </div>
    </.app_shell>
    """
  end

  # ── Events ────────────────────────────────────────────────────────────────

  @impl true
  def handle_event("pick", params, socket) do
    picks = Map.merge(socket.assigns.picks, Map.take(params, ~w(soul_id agent_id report_id co_id rank)))

    # Choosing a soul pre-fills the billet from the soul's human-authored
    # suggestion — but only when the soul select itself changed, so it never
    # clobbers a rank the Admiral just picked. The Admiral still confirms/overrides
    # it; authority is only ever conferred by this commission, never by the soul.
    picks =
      case params do
        %{"_target" => ["soul_id"], "soul_id" => sid} -> Map.put(picks, "rank", suggested_billet(sid))
        _ -> picks
      end

    {:noreply, assign(socket, :picks, picks)}
  end

  def handle_event("commission", %{"soul_id" => sid} = params, socket) do
    rank = Fleet.Rank.to_key(params["rank"] || picked_rank(socket.assigns.picks))
    safe(fn -> Fleet.commission(sid, rank: rank) end)

    {:noreply,
     socket
     |> put_flash(:info, "Commissioned #{sid} as #{Fleet.Rank.label(rank)}")
     |> drop_picks(["soul_id", "rank"])
     |> assign_roster()}
  end

  def handle_event("issue_order", %{"agent_id" => id, "directive" => directive}, socket) do
    a = Enum.find(socket.assigns.roster, &(&1.agent_id == id))

    safe(fn ->
      if a && a.co, do: Fleet.issue_order(a.co, id, directive), else: Fleet.order(id, directive)
    end)

    {:noreply, socket |> put_flash(:info, "Order issued to #{id}") |> drop_picks(["agent_id"])}
  end

  def handle_event("assign_co", %{"report_id" => rid, "co_id" => cid}, socket) do
    msg =
      case safe(fn -> Fleet.assign_co(rid, cid) end) do
        {:error, :self_command} -> "An agent cannot be its own CO"
        _ -> "#{cid} now commands #{rid}"
      end

    {:noreply, socket |> put_flash(:info, msg) |> drop_picks(["report_id", "co_id"]) |> assign_roster()}
  end

  def handle_event("relieve", %{"agent" => id, "co" => co}, socket) do
    safe(fn -> if co == "", do: Fleet.relieve(id), else: Fleet.relieve(co, id) end)
    {:noreply, socket |> put_flash(:info, "Relieved #{id}") |> assign_roster()}
  end

  def handle_event("reinstate", %{"agent" => id, "co" => co}, socket) do
    safe(fn -> if co == "", do: Fleet.reinstate(id), else: Fleet.reinstate(co, id) end)
    {:noreply, socket |> put_flash(:info, "Reinstated #{id}") |> assign_roster()}
  end

  def handle_event("retire", %{"agent" => id}, socket) do
    safe(fn ->
      case Registry.lookup(Fleet.Registry, {:ensign, id}) do
        [{pid, _}] -> Fleet.retire(pid)
        _ -> :ok
      end
    end)

    {:noreply, socket |> put_flash(:info, "Retired #{id}") |> assign(:detail, nil) |> assign_roster()}
  end

  def handle_event("detail", %{"agent" => id}, socket) do
    {:noreply, assign(socket, :detail, load_detail(id))}
  end

  def handle_event("close_detail", _params, socket), do: {:noreply, assign(socket, :detail, nil)}

  # ── Live updates ──────────────────────────────────────────────────────────

  @impl true
  def handle_info({:fleet_event, meta}, socket) do
    if meta[:event] in [:tick, nil] do
      {:noreply, socket}
    else
      item = %{
        at: Calendar.strftime(Time.utc_now(), "%H:%M:%S"),
        event: meta[:event],
        agent: meta[:agent_id],
        order_id: meta[:order_id]
      }

      {:noreply, assign(socket, :feed, Enum.take([item | socket.assigns.feed], @feed_cap))}
    end
  end

  def handle_info(:refresh, socket), do: {:noreply, assign_roster(socket)}
  def handle_info({:world_context_changed, _}, socket), do: {:noreply, socket}
  def handle_info(_msg, socket), do: {:noreply, socket}

  # ── Data loading ──────────────────────────────────────────────────────────

  defp assign_roster(socket), do: assign(socket, :roster, safe(fn -> Fleet.roster() end) || [])

  defp load_souls, do: safe(fn -> Brain.Soul.list_ids() end) || []

  defp load_detail(agent_id) do
    {summary, history} =
      case safe(fn -> Fleet.Service.load(agent_id) end) do
        {:ok, %{summary: s, history: h}} -> {s, h}
        _ -> {nil, []}
      end

    notes = safe(fn -> Fleet.DutyLog.for_soul(agent_id) end) || []
    status = safe(fn -> Fleet.status(agent_id) end) || %{}
    current = (summary && summary.current_assignment) || %{}

    order_id = status[:order_id] || current["order_id"]

    timeline =
      if order_id do
        safe(fn -> Atlas.Repo.all(CommandRecord.for_order(order_id)) end) || []
      else
        []
      end

    report =
      timeline
      |> Enum.filter(&(&1.kind == "report"))
      |> List.last()
      |> case do
        nil -> nil
        r -> get_in(r.payload || %{}, ["outcome"])
      end

    %{
      soul_id: agent_id,
      duty: (summary && String.to_atom(summary.duty_status)) || status[:duty] || :active,
      completed: (summary && summary.orders_completed) || 0,
      dissented: (summary && summary.orders_dissented) || 0,
      failed: (summary && summary.orders_failed) || 0,
      reliefs: (summary && summary.reliefs) || 0,
      directive: status[:directive] || current["directive"],
      assignment_status: status[:assignment_status] || current["status"],
      report: report,
      timeline: timeline,
      history: Enum.reverse(history),
      notes: notes
    }
  end

  defp backfill_feed do
    safe(fn ->
      from(r in CommandRecord, order_by: [desc: r.inserted_at], limit: 30)
      |> Atlas.Repo.all()
      |> Enum.map(fn r ->
        %{
          at: Calendar.strftime(r.inserted_at, "%H:%M:%S"),
          event: safe_atom(r.kind),
          agent: r.from_agent,
          order_id: r.order_id
        }
      end)
    end) || []
  end

  # ── Helpers ───────────────────────────────────────────────────────────────

  # The currently-picked billet, as a string key (defaults to the default billet).
  defp picked_rank(picks), do: picks["rank"] || to_string(Fleet.Rank.default())

  # The billet a soul suggests for itself (human-authored `metadata.suggested_billet`),
  # normalised to a known billet key string, or the default. A hint only — the
  # commission is what actually confers the rank.
  defp suggested_billet(soul_id) do
    with {:ok, %Brain.Soul{metadata: %{"suggested_billet" => b}}} <- safe_soul(soul_id) do
      to_string(Fleet.Rank.to_key(b))
    else
      _ -> to_string(Fleet.Rank.default())
    end
  end

  defp safe_soul(soul_id), do: safe(fn -> Brain.Soul.get(soul_id) end) || {:error, :unavailable}

  defp safe(fun) do
    fun.()
  rescue
    e -> Logger.warning("FleetLive: #{inspect(e)}"); nil
  catch
    :exit, _ -> nil
  end

  defp safe_atom(s) when is_binary(s) do
    String.to_existing_atom(s)
  rescue
    ArgumentError -> :event
  end

  defp drop_picks(socket, keys), do: assign(socket, :picks, Map.drop(socket.assigns.picks, keys))

  # One-line description of a command-channel event for the order timeline —
  # for a REPORT, show what the agent actually produced.
  defp timeline_desc(%{kind: "report"} = r), do: "→ " <> to_display(get_in(r.payload || %{}, ["outcome"]))
  defp timeline_desc(%{reason: reason}) when is_binary(reason) and reason != "", do: reason
  defp timeline_desc(%{verdict: v}) when is_binary(v) and v != "", do: "verdict: #{v}"
  defp timeline_desc(%{authority: a}) when is_binary(a) and a != "", do: "authority: #{a}"
  defp timeline_desc(r), do: "#{r.from_agent} → #{r.to_agent || "—"}"

  defp report_text(nil), do: ""
  defp report_text(s) when is_binary(s), do: s
  defp report_text(%{"raw" => raw}), do: to_string(raw)
  defp report_text(other), do: inspect(other)

  defp to_display(nil), do: "(done)"
  defp to_display(s) when is_binary(s), do: String.slice(s, 0, 300)
  defp to_display(other), do: inspect(other) |> String.slice(0, 300)

  defp count(list, fun), do: Enum.count(list, fun)
  defp fmt_grant(g), do: Fleet.Authority.encode(g)

  defp duty_class(:active), do: "badge-success"
  defp duty_class(:relieved), do: "badge-warning"
  defp duty_class(_), do: "badge-ghost"

  defp status_class("completed"), do: "badge-success"
  defp status_class("dissented"), do: "badge-warning"
  defp status_class("failed"), do: "badge-error"
  defp status_class("blocked"), do: "badge-info"
  defp status_class(_), do: "badge-ghost"

  defp event_class(e) when e in [:report, :ack, :cognition_complete, :granted, :grant, :commissioned, :reinstated],
    do: "badge-success"

  defp event_class(e) when e in [:dissent, :deny, :provenance_anomaly, :cognition_failed, :order_failed, :order_dissented],
    do: "badge-error"

  defp event_class(e) when e in [:relieved, :relieve, :request, :rehydrated], do: "badge-warning"
  defp event_class(_), do: "badge-ghost"
end
