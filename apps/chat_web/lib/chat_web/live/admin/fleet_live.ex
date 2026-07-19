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

  @safety_events ~w(dissent deny provenance_anomaly cognition_failed order_failed order_dissented grant_violation)a

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
     |> assign(:feed_filter, :all)
     |> assign(:hail_log, load_hail_log())
     |> assign(:selected, nil)
     |> assign(:detail, nil)
     |> assign(:roster_query, "")
     |> assign(:command_tab, :commission)
     # Conversation state: the agent currently being hailed (spinner) and the last
     # hail Q&A, so the detail modal can show the exchange as it returns.
     |> assign(:hail_pending, nil)
     |> assign(:hail, nil)
     # Controlled select values, so the 3s roster refresh never resets a form
     # the Admiral is mid-way through filling out.
     |> assign(:picks, %{})
     |> assign(:generation, load_generation_status())
     |> assign(:models, load_models())
     |> assign(:model_pull_pending, nil)
     |> assign(:systems, systems_health())
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
        <div class="flex justify-end">
          <.generation_chip generation={@generation} />
        </div>
      </:page_header>

      <div class="p-5 sm:p-6 space-y-6 max-w-7xl mx-auto w-full">
        <!-- Bridge header -->
        <header class="flex flex-wrap items-end justify-between gap-4 border-b border-base-300 pb-5">
          <div>
            <div class="text-[11px] uppercase tracking-[0.2em] text-primary/70 font-medium mb-1.5">Admiral's Console</div>
            <h1 class="text-2xl font-semibold tracking-tight">The Bridge</h1>
            <p class="text-sm text-base-content/60 mt-2">{status_line(@roster, @systems)}</p>
          </div>
          <div class="flex items-center gap-2">
            <span :if={recent_dissent_count(@feed) > 0}
                  class="inline-flex items-center gap-1.5 rounded-full border border-warning/30 bg-warning/10 text-warning px-3 py-1.5 text-sm">
              <.icon name="hero-hand-raised" class="size-4" />
              {recent_dissent_count(@feed)} recent dissent{if recent_dissent_count(@feed) == 1, do: "", else: "s"}
            </span>
            <.link navigate="/systems"
                   class={["inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm transition-colors hover:bg-base-200/60", systems_pill_class(@systems)]}>
              <.icon name="hero-cpu-chip" class="size-4" />
              <span class="font-medium">Systems</span>
              <span class="tabular-nums">{@systems[:health_score] || "—"}<span :if={@systems[:health_score]}>%</span></span>
            </.link>
          </div>
        </header>

        <!-- Command panel -->
        <div class="card bg-base-100 shadow">
          <div class="card-body p-4">
            <div class="flex items-center gap-1 border-b border-base-300 -mx-4 px-4 mb-4">
              <.cmd_tab tab="commission" active={@command_tab} icon="hero-user-plus" label="Commission" />
              <.cmd_tab tab="order" active={@command_tab} icon="hero-flag" label="Orders" />
              <.cmd_tab tab="assign_co" active={@command_tab} icon="hero-share" label="Chain" />
              <.cmd_tab tab="models" active={@command_tab} icon="hero-cpu-chip" label="Minds" />
            </div>

            <form :if={@command_tab == :commission} phx-submit="commission" class="max-w-lg space-y-4">
              <p class="text-sm text-base-content/60">Bring a soul aboard as a commissioned officer.</p>
              <div>
                <.field_label>Officer</.field_label>
                <select name="soul_id" phx-change="pick" class="select select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["soul_id"])}>Select a soul to commission…</option>
                  <option :for={sid <- @souls} value={sid} selected={sid == @picks["soul_id"]}>{sid}</option>
                </select>
                <p :if={@souls == []} class="text-xs text-warning mt-1.5">
                  No souls found in <code class="text-[11px]">{@souls_dir}</code>. Add one (e.g. <code>souls/ensign-jj7.json</code>) and refresh.
                </p>
              </div>
              <div>
                <.field_label>Posting</.field_label>
                <select name="rank" phx-change="pick" class="select select-bordered w-full">
                  <option :for={{key, meta} <- @billets} value={key} selected={to_string(key) == picked_rank(@picks)}>{meta.label}</option>
                </select>
                <p class="text-xs text-base-content/60 mt-1.5">{Fleet.Rank.describe(picked_rank(@picks))}</p>
                <div class="mt-2 flex flex-wrap items-center gap-1.5">
                  <%= case Fleet.Rank.standing_authorities(picked_rank(@picks)) do %>
                    <% [] -> %>
                      <span class="text-xs text-base-content/40">No standing authority — each order confers what its task needs.</span>
                    <% auths -> %>
                      <span class="text-[11px] uppercase tracking-wider text-base-content/40">Confers</span>
                      <span :for={a <- auths} class="badge badge-ghost badge-sm">{fmt_grant(a)}</span>
                  <% end %>
                </div>
              </div>
              <details class="group">
                <summary class="inline-flex items-center gap-1 text-xs text-base-content/50 hover:text-base-content cursor-pointer select-none">
                  <.icon name="hero-chevron-right" class="size-3 group-open:rotate-90 transition-transform" /> Callsign (optional)
                </summary>
                <div class="mt-2 pl-4 space-y-1.5">
                  <input name="commission_agent_id" value={@picks["commission_agent_id"]} phx-change="pick"
                         class="input input-sm input-bordered w-full"
                         placeholder={"defaults to " <> (@picks["soul_id"] || "the soul's id")} autocomplete="off" />
                  <p class="text-xs text-base-content/50">
                    Blank resumes the officer under the soul's own id. A distinct id stands up a second, independent officer from the same soul.
                  </p>
                </div>
              </details>
              <button class="btn btn-primary"><.icon name="hero-user-plus" class="size-4" /> Commission officer</button>
            </form>

            <form :if={@command_tab == :order} phx-submit="issue_order" class="max-w-lg space-y-4">
              <p class="text-sm text-base-content/60">Give a standing officer their orders.</p>
              <div>
                <.field_label>Officer</.field_label>
                <select name="agent_id" phx-change="pick" class="select select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["agent_id"])}>Who receives the order?</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["agent_id"]}>{agent_label(a)}</option>
                </select>
              </div>
              <div>
                <.field_label>Orders</.field_label>
                <input name="directive" class="input input-bordered w-full"
                       placeholder="e.g. Summarize sector 7 readiness and flag anomalies." required />
              </div>
              <button class="btn btn-primary"><.icon name="hero-flag" class="size-4" /> Issue order</button>
            </form>

            <form :if={@command_tab == :assign_co} phx-submit="assign_co" class="max-w-lg space-y-4">
              <p class="text-sm text-base-content/60">Set who answers to whom.</p>
              <div>
                <.field_label>Officer</.field_label>
                <select name="report_id" phx-change="pick" class="select select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["report_id"])}>Which officer reports…</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["report_id"]}>{agent_label(a)}</option>
                </select>
              </div>
              <div>
                <.field_label>Reports to</.field_label>
                <select name="co_id" phx-change="pick" class="select select-bordered w-full" required>
                  <option value="" disabled selected={is_nil(@picks["co_id"])}>…their commanding officer</option>
                  <option :for={a <- @roster} value={a.agent_id} selected={a.agent_id == @picks["co_id"]}>{agent_label(a)}</option>
                </select>
              </div>
              <button class="btn btn-primary"><.icon name="hero-share" class="size-4" /> Assign</button>
            </form>

            <div :if={@command_tab == :models} class="max-w-xl space-y-4">
              <div class="flex items-center justify-between">
                <div class="text-xs text-base-content/60">
                  Generation backend: <span class="font-mono">{@generation.name}</span>
                  <span :if={@generation.name == :openai_compatible}>
                    · active model <span class="font-mono">{Brain.ML.Generation.OllamaAdmin.current_model()}</span>
                  </span>
                </div>
                <button class="btn btn-ghost btn-xs" phx-click="refresh_models">
                  <.icon name="hero-arrow-path" class="size-3.5" /> refresh
                </button>
              </div>

              <form phx-submit="pull_model" class="flex gap-2">
                <input
                  name="model_name"
                  placeholder="model to pull, e.g. llama3.1:8b"
                  class="input input-sm input-bordered flex-1"
                  autocomplete="off"
                  required
                  disabled={not is_nil(@model_pull_pending)}
                />
                <button class="btn btn-sm btn-secondary" disabled={not is_nil(@model_pull_pending)}>
                  Pull
                </button>
              </form>
              <div :if={@model_pull_pending} class="text-xs text-base-content/60">
                <span class="loading loading-dots loading-xs"></span> pulling “{@model_pull_pending}” — this can take a while for large models…
              </div>

              <div class="overflow-x-auto">
                <table :if={@models != []} class="table table-xs">
                  <thead><tr><th>Model</th><th>Size</th><th>Pulled</th><th></th></tr></thead>
                  <tbody>
                    <tr :for={m <- @models}>
                      <td class="font-mono">{m.name}</td>
                      <td>{format_bytes(m.size)}</td>
                      <td class="text-base-content/50">{String.slice(to_string(m.modified_at), 0, 10)}</td>
                      <td class="text-right">
                        <button
                          class={["btn btn-xs", m.name == Brain.ML.Generation.OllamaAdmin.current_model() && "btn-success", m.name != Brain.ML.Generation.OllamaAdmin.current_model() && "btn-ghost"]}
                          phx-click="activate_model" phx-value-name={m.name}
                        >
                          {if m.name == Brain.ML.Generation.OllamaAdmin.current_model(), do: "active", else: "activate"}
                        </button>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <p :if={@models == []} class="text-xs text-base-content/50">
                  No local models found (or Ollama isn't reachable). Pull one above.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-3 gap-4">
          <!-- Roster -->
          <div class="xl:col-span-2 rounded-xl border border-base-300 bg-base-100">
            <div class="p-4">
              <div class="flex items-center justify-between gap-2 flex-wrap mb-3">
                <h2 class="text-sm font-semibold flex items-center gap-2">
                  <.icon name="hero-users" class="size-4 text-base-content/50" /> Chain of command
                </h2>
                <label class="input input-sm input-bordered flex items-center gap-1 w-48" :if={@roster != []}>
                  <.icon name="hero-magnifying-glass" class="size-3.5 text-base-content/40" />
                  <input
                    type="text"
                    phx-change="filter_roster"
                    phx-debounce="150"
                    name="q"
                    value={@roster_query}
                    placeholder="filter crew…"
                    class="grow"
                  />
                </label>
              </div>

              <div :if={@roster == []} class="text-sm py-12 text-center space-y-3">
                <.icon name="hero-user-group" class="size-10 mx-auto text-base-content/20" />
                <p class="font-medium text-base-content/70">No officers aboard.</p>
                <p class="text-xs text-base-content/50">
                  Commission your first from the panel above to stand up the crew.
                </p>
              </div>

              <div :if={@roster != [] and filtered_roster(@roster, @roster_query) == []} class="text-base-content/50 text-sm py-6 text-center">
                No crew match “{@roster_query}”.
              </div>

              <div class="overflow-x-auto">
                <table :if={filtered_roster(@roster, @roster_query) != []} class="table table-sm">
                  <thead>
                    <tr><th>Agent</th><th>Duty</th><th>Assignment</th><th>Grants</th><th></th></tr>
                  </thead>
                  <tbody>
                    <tr :for={a <- hierarchy(filtered_roster(@roster, @roster_query))} class="hover">
                      <td>
                        <div class="flex items-start gap-1 min-w-[14rem]" style={"padding-left: #{a.depth * 1.25}rem"}>
                          <.icon :if={a.depth > 0} name="hero-arrow-turn-down-right" class="size-3.5 text-base-content/30 shrink-0 mt-0.5" />
                          <div class="min-w-0">
                            <div class="flex items-center gap-1.5 flex-wrap">
                              <span class="font-medium whitespace-nowrap">{agent_label(a)}</span>
                              <span class={["badge badge-xs whitespace-nowrap", rank_class(a.rank)]}>{Fleet.Rank.label(a.rank)}</span>
                              <div :if={Map.get(a, :cycle_member, false)} class="tooltip" data-tip="This agent's CO chain loops back on itself and can't be drawn as a tree — fix with Assign CO.">
                                <span class="badge badge-xs badge-error gap-1"><.icon name="hero-exclamation-triangle" class="size-3" />cycle</span>
                              </div>
                            </div>
                            <div class="text-xs text-base-content/50">
                              {length(a.reports)} reports · up {uptime_str(a.uptime_ms)}
                              <span :if={a.co}> · reports to {a.co}</span>
                            </div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span class={["badge badge-sm", duty_class(a.duty)]}>{a.duty}</span>
                        <span :if={a.working} class="loading loading-spinner loading-xs ml-1"></span>
                        <span :if={not a.soul_loaded} class="badge badge-ghost badge-xs ml-1">no soul</span>
                      </td>
                      <td class="max-w-[16rem]">
                        <span :if={a.assignment_status} class={["badge badge-sm", status_class(a.assignment_status)]}>
                          {a.assignment_status}
                        </span>
                        <span :if={is_nil(a.assignment_status)} class="text-base-content/40">idle</span>
                        <div :if={a[:directive]} class="text-xs text-base-content/50 truncate" title={a[:directive]}>“{a.directive}”</div>
                      </td>
                      <td>
                        <span :for={g <- a.grants} class="badge badge-ghost badge-xs mr-1">{fmt_grant(g)}</span>
                        <span :if={a.grants == []} class="text-base-content/30 text-xs">—</span>
                      </td>
                      <td class="text-right whitespace-nowrap">
                        <div class="flex items-center justify-end gap-1">
                          <div class="tooltip" data-tip="details">
                            <button class="btn btn-ghost btn-xs btn-square" phx-click="detail" phx-value-agent={a.agent_id}>
                              <.icon name="hero-magnifying-glass" class="size-3.5" />
                            </button>
                          </div>
                          <div :if={a.duty == :active} class="tooltip" data-tip="relieve">
                            <button class="btn btn-ghost btn-xs btn-square text-warning"
                              phx-click="relieve" phx-value-agent={a.agent_id} phx-value-co={a.co || ""}>
                              <.icon name="hero-pause" class="size-3.5" />
                            </button>
                          </div>
                          <div :if={a.duty == :relieved} class="tooltip" data-tip="reinstate">
                            <button class="btn btn-ghost btn-xs btn-square text-success"
                              phx-click="reinstate" phx-value-agent={a.agent_id} phx-value-co={a.co || ""}>
                              <.icon name="hero-play" class="size-3.5" />
                            </button>
                          </div>
                          <div class="tooltip" data-tip="retire">
                            <button class="btn btn-ghost btn-xs btn-square text-error"
                              phx-click="retire" phx-value-agent={a.agent_id}
                              data-confirm="Retire this agent?">
                              <.icon name="hero-trash" class="size-3.5" />
                            </button>
                          </div>
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Ship's log -->
          <div class="rounded-xl border border-base-300 bg-base-100">
            <div class="p-4">
              <div class="flex items-center justify-between gap-2 mb-3">
                <h2 class="text-sm font-semibold flex items-center gap-2">
                  <.icon name="hero-signal" class="size-4 text-base-content/50" /> Ship's log
                </h2>
                <div class="join">
                  <button
                    class={["btn btn-xs join-item", @feed_filter == :all && "btn-active"]}
                    phx-click="feed_filter" phx-value-filter="all"
                  >all</button>
                  <button
                    class={["btn btn-xs join-item", @feed_filter == :safety && "btn-active"]}
                    phx-click="feed_filter" phx-value-filter="safety"
                  >safety</button>
                  <button
                    class={["btn btn-xs join-item gap-1", @feed_filter == :hails && "btn-active"]}
                    phx-click="feed_filter" phx-value-filter="hails"
                  ><.icon name="hero-chat-bubble-left-right" class="size-3" />hails</button>
                </div>
              </div>

              <div :if={@feed_filter != :hails} class="space-y-1 max-h-[28rem] overflow-y-auto text-xs">
                <div
                  :for={e <- filtered_feed(@feed, @feed_filter)}
                  class={["flex items-center gap-2 py-0.5 pl-1.5 border-l-2", severity_border(e.event)]}
                >
                  <span class="text-base-content/40 shrink-0 font-mono">{e.at}</span>
                  <span class={["badge badge-xs shrink-0", event_class(e.event)]}>{e.event}</span>
                  <span class="truncate">
                    {e.agent}<span :if={e.order_id} class="text-base-content/40 font-mono"> · {String.slice(e.order_id, 0, 6)}</span>
                  </span>
                </div>
                <div :if={filtered_feed(@feed, @feed_filter) == []} class="text-base-content/50 py-4 text-center">No activity yet.</div>
              </div>

              <div :if={@feed_filter == :hails} class="space-y-2 max-h-[28rem] overflow-y-auto text-xs">
                <div :for={h <- @hail_log} class="border-b border-base-200 pb-2 last:border-0">
                  <div class="flex items-center gap-2 text-base-content/50">
                    <span class="font-mono shrink-0">{Calendar.strftime(h.at, "%H:%M:%S")}</span>
                    <.icon name="hero-chat-bubble-left-right" class="size-3 shrink-0 text-secondary" />
                    <span class="truncate"><span class="font-medium text-base-content/70">{h.to}</span> ← {h.from}</span>
                  </div>
                  <div class="mt-1 pl-1 border-l-2 border-secondary/30">
                    <div class="text-base-content/70 break-words">
                      <span class="text-base-content/40">Q</span> “{h.question || "…"}”
                    </div>
                    <div :if={h[:answer]} class="mt-0.5 bg-base-200 rounded p-1.5 whitespace-pre-wrap break-words">
                      {h.answer}
                    </div>
                    <div :if={h[:error]} class="mt-0.5 text-error break-words">
                      <span class="badge badge-error badge-xs mr-1">failed</span>{h.error}
                    </div>
                    <div :if={is_nil(h[:answer]) and is_nil(h[:error])} class="mt-0.5 text-base-content/40 italic">
                      awaiting reply…
                    </div>
                  </div>
                </div>
                <div :if={@hail_log == []} class="text-base-content/50 py-4 text-center">No hails yet.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Detail modal -->
      <div :if={@detail} class="modal modal-open">
        <div class="modal-box max-w-2xl">
          <div class="text-[11px] uppercase tracking-[0.2em] text-primary/70 font-medium mb-1">Officer dossier</div>
          <h3 class="font-semibold text-xl flex items-center gap-2 flex-wrap">
            {agent_label(@detail)}
            <span class={["badge badge-sm", rank_class(@detail.rank)]}>{Fleet.Rank.label(@detail.rank)}</span>
            <span class={["badge badge-sm", duty_class(@detail.duty)]}>{@detail.duty}</span>
          </h3>
          <div class="stats stats-horizontal shadow my-3 w-full">
            <div class="stat py-2"><div class="stat-title text-xs">completed</div><div class="stat-value text-lg text-success">{@detail.completed}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">dissented</div><div class="stat-value text-lg text-warning">{@detail.dissented}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">failed</div><div class="stat-value text-lg text-error">{@detail.failed}</div></div>
            <div class="stat py-2"><div class="stat-title text-xs">reliefs</div><div class="stat-value text-lg">{@detail.reliefs}</div></div>
          </div>

          <div class="text-xs text-base-content/50 mb-3">
            mind-world <code>{@detail.mind_world_id}</code>
            <span :if={@detail.co}> · reports to <code>{@detail.co}</code></span>
          </div>

          <div class="my-3">
            <h4 class="font-semibold text-sm mb-1">Tools</h4>
            <p class="text-xs text-base-content/50 mb-1.5">
              Standing access you grant this officer — what it may call on, over and above what any order confers.
              Click a tool to grant or revoke it.
            </p>
            <div class="flex flex-wrap gap-1.5">
              <button
                :for={{tool_name, spec} <- Fleet.available_tools()}
                type="button"
                class={["btn btn-xs gap-1", tool_name in @detail.granted_tools && "btn-success", tool_name not in @detail.granted_tools && "btn-outline"]}
                phx-click={if tool_name in @detail.granted_tools, do: "revoke_tool", else: "grant_tool"}
                phx-value-agent={@detail.agent_id}
                phx-value-tool={tool_name}
                title={Map.get(spec, :description, "")}
              >
                <.icon name={if tool_name in @detail.granted_tools, do: "hero-check-circle", else: "hero-plus-circle"} class="size-3.5" />
                {tool_name}
              </button>
              <span :if={Fleet.available_tools() == %{}} class="text-xs text-base-content/40">
                No tools registered (Fleet.Tool.registry/0 is empty).
              </span>
            </div>
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

          <!-- Hail: converse with the agent without giving an order -->
          <div class="my-3">
            <h4 class="font-semibold text-sm mb-1">Hail — talk to {agent_label(@detail)}</h4>
            <form phx-submit="hail" class="flex gap-2">
              <input type="hidden" name="agent" value={@detail.agent_id} />
              <input
                name="question"
                placeholder="Ask a question — not an order…"
                required
                autocomplete="off"
                class="input input-sm input-bordered flex-1"
                disabled={@hail_pending == @detail.agent_id}
              />
              <button class="btn btn-sm btn-secondary" disabled={@hail_pending == @detail.agent_id}>Hail</button>
            </form>
            <div :if={@hail_pending == @detail.agent_id} class="mt-2 text-xs text-base-content/60">
              <span class="loading loading-dots loading-xs"></span> awaiting reply…
            </div>
            <div :if={@hail && @hail.agent_id == @detail.agent_id} class="mt-2 space-y-1">
              <div class="text-xs text-base-content/50">Admiral: “{@hail.question}”</div>
              <div
                :if={@hail[:answer]}
                class="bg-base-100 border border-base-300 rounded p-2 text-sm whitespace-pre-wrap break-words"
              >{@hail.answer}</div>
              <div :if={@hail[:error]} class="text-error text-sm">hail failed: {@hail.error}</div>
            </div>
          </div>

          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <h4 class="font-semibold text-sm mb-1">Service record</h4>
              <ul class="text-xs space-y-1 max-h-48 overflow-y-auto">
                <li :for={r <- @detail.history} class="flex gap-2">
                  <span class={["badge badge-xs", event_class(String.to_atom(r.kind))]}>{r.kind}</span>
                  <span class="text-base-content/60 truncate">{r.reason || r.outcome || ""}</span>
                </li>
                <li :if={@detail.history == []} class="text-base-content/40">no history</li>
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

  # ── Small function components ───────────────────────────────────────────────

  attr :tab, :string, required: true
  attr :active, :atom, required: true
  attr :icon, :string, required: true
  attr :label, :string, required: true

  defp cmd_tab(assigns) do
    assigns = assign(assigns, :is_active, assigns.active == String.to_existing_atom(assigns.tab))

    ~H"""
    <button
      type="button"
      phx-click="command_tab"
      phx-value-tab={@tab}
      class={[
        "flex items-center gap-1.5 px-4 py-3 text-sm font-medium border-b-2 -mb-px transition-colors",
        (@is_active && "border-primary text-primary") ||
          "border-transparent text-base-content/50 hover:text-base-content"
      ]}
    >
      <.icon name={@icon} class="size-4" /> {@label}
    </button>
    """
  end

  slot :inner_block, required: true

  defp field_label(assigns) do
    ~H"""
    <label class="block text-[11px] font-medium uppercase tracking-wider text-base-content/50 mb-1.5">
      {render_slot(@inner_block)}
    </label>
    """
  end

  attr :generation, :map, required: true

  defp generation_chip(assigns) do
    ~H"""
    <div class="tooltip tooltip-bottom" data-tip={"generation backend: #{@generation.name}"}>
      <span class={["badge badge-sm gap-1", @generation.ready && "badge-success", not @generation.ready && "badge-error"]}>
        <span class={["inline-block size-1.5 rounded-full", @generation.ready && "bg-success-content animate-pulse", not @generation.ready && "bg-error-content"]}></span>
        {if @generation.ready, do: "LLM ready", else: "LLM unavailable"}
      </span>
    </div>
    """
  end

  # ── Events ────────────────────────────────────────────────────────────────

  @impl true
  def handle_event("pick", params, socket) do
    picks =
      Map.merge(
        socket.assigns.picks,
        Map.take(params, ~w(soul_id agent_id commission_agent_id report_id co_id rank))
      )

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

  def handle_event("command_tab", %{"tab" => tab}, socket) do
    {:noreply, assign(socket, :command_tab, String.to_existing_atom(tab))}
  end

  def handle_event("filter_roster", %{"q" => q}, socket) do
    {:noreply, assign(socket, :roster_query, q)}
  end

  def handle_event("feed_filter", %{"filter" => filter}, socket) do
    {:noreply, assign(socket, :feed_filter, String.to_existing_atom(filter))}
  end

  def handle_event("refresh_models", _params, socket) do
    {:noreply, assign(socket, :models, load_models())}
  end

  def handle_event("activate_model", %{"name" => name}, socket) do
    Brain.ML.Generation.OllamaAdmin.set_model(name)
    {:noreply, put_flash(socket, :info, "Now generating with #{name}")}
  end

  def handle_event("pull_model", %{"model_name" => name}, socket) do
    name = String.trim(name)

    if name == "" or socket.assigns.model_pull_pending do
      {:noreply, socket}
    else
      liveview = self()

      Task.Supervisor.start_child(Fleet.TaskSupervisor, fn ->
        result = Brain.ML.Generation.OllamaAdmin.pull_model(name)
        send(liveview, {:model_pull_result, name, result})
      end)

      {:noreply, assign(socket, :model_pull_pending, name)}
    end
  end

  def handle_event("commission", %{"soul_id" => sid} = params, socket) do
    rank = Fleet.Rank.to_key(params["rank"] || picked_rank(socket.assigns.picks))
    commission_opts = commission_opts(rank, params["commission_agent_id"])

    {kind, msg} =
      case safe(fn -> Fleet.commission(sid, commission_opts) end) do
        {:ok, _pid, ^sid} ->
          {:info, "Commissioned #{sid} as #{Fleet.Rank.label(rank)}"}

        {:ok, _pid, agent_id} ->
          {:info, "Commissioned #{sid} as #{Fleet.Rank.label(rank)}, identified as #{agent_id}"}

        {:error, reason} ->
          {:error, "Could not commission #{sid}: #{inspect(reason)}"}

        _ ->
          {:error, "Could not commission #{sid}"}
      end

    {:noreply,
     socket
     |> put_flash(kind, msg)
     |> drop_picks(["soul_id", "rank", "commission_agent_id"])
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
    {kind, msg} =
      case safe(fn -> Fleet.assign_co(rid, cid) end) do
        {:error, :self_command} -> {:error, "An agent cannot be its own CO"}
        {:error, :cycle} -> {:error, "#{cid} already reports (directly or transitively) to #{rid} — wiring this would close a loop"}
        {:error, reason} -> {:error, "Could not wire chain: #{inspect(reason)}"}
        _ -> {:info, "#{cid} now commands #{rid}"}
      end

    {:noreply, socket |> put_flash(kind, msg) |> drop_picks(["report_id", "co_id"]) |> assign_roster()}
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

  def handle_event("grant_tool", %{"agent" => id, "tool" => tool}, socket) do
    safe(fn -> Fleet.grant_tool(id, tool) end)
    {:noreply, socket |> assign(:detail, load_detail(id)) |> put_flash(:info, "Granted #{tool} to #{id}")}
  end

  def handle_event("revoke_tool", %{"agent" => id, "tool" => tool}, socket) do
    safe(fn -> Fleet.revoke_tool(id, tool) end)
    {:noreply, socket |> assign(:detail, load_detail(id)) |> put_flash(:info, "Revoked #{tool} from #{id}")}
  end

  def handle_event("close_detail", _params, socket), do: {:noreply, assign(socket, :detail, nil)}

  def handle_event("hail", %{"agent" => id, "question" => q}, socket) do
    q = String.trim(q)

    if q == "" do
      {:noreply, socket}
    else
      # Fleet.hail stamps THIS LiveView process as the sender, so the async
      # {:hail_reply, ...} lands in our own handle_info below.
      safe(fn -> Fleet.hail(id, q) end)
      Process.send_after(self(), {:hail_timeout, id, q}, 180_000)

      {:noreply,
       socket
       |> assign(:hail_pending, id)
       |> assign(:hail, %{agent_id: id, question: q})}
    end
  end

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

  def handle_info(:refresh, socket) do
    {:noreply,
     socket
     |> assign(:generation, load_generation_status())
     |> assign(:hail_log, load_hail_log())
     |> assign(:systems, systems_health())
     |> assign_roster()}
  end

  def handle_info({:world_context_changed, _}, socket), do: {:noreply, socket}

  def handle_info({:hail_reply, %{agent_id: id} = reply}, socket) do
    pending = if socket.assigns.hail_pending == id, do: nil, else: socket.assigns.hail_pending
    # Refresh immediately rather than waiting for the next 3s tick, so the
    # hail log reflects this exchange (success or failure — audit_hail_reply
    # records both) as soon as the modal shows it.
    {:noreply,
     socket
     |> assign(:hail, reply)
     |> assign(:hail_pending, pending)
     |> assign(:hail_log, load_hail_log())}
  end

  def handle_info({:hail_timeout, id, q}, socket) do
    h = socket.assigns.hail

    # Only trip the timeout if this exact hail is still outstanding (no reply came).
    if socket.assigns.hail_pending == id and h && h.question == q and is_nil(Map.get(h, :answer)) do
      {:noreply,
       socket
       |> assign(:hail_pending, nil)
       |> assign(:hail, Map.put(h, :error, "timed out waiting for reply"))}
    else
      {:noreply, socket}
    end
  end

  def handle_info({:model_pull_result, name, :ok}, socket) do
    {:noreply,
     socket
     |> assign(:model_pull_pending, nil)
     |> assign(:models, load_models())
     |> put_flash(:info, "Pulled #{name}")}
  end

  def handle_info({:model_pull_result, name, {:error, reason}}, socket) do
    {:noreply,
     socket
     |> assign(:model_pull_pending, nil)
     |> put_flash(:error, "Failed to pull #{name}: #{inspect(reason)}")}
  end

  def handle_info(_msg, socket), do: {:noreply, socket}

  # ── Data loading ──────────────────────────────────────────────────────────

  defp assign_roster(socket), do: assign(socket, :roster, safe(fn -> Fleet.roster() end) || [])

  defp load_souls, do: safe(fn -> Brain.Soul.list_ids() end) || []

  defp load_models do
    case Brain.ML.Generation.OllamaAdmin.list_models() do
      {:ok, models} -> Enum.sort_by(models, & &1.name)
      {:error, _reason} -> []
    end
  end

  defp load_generation_status do
    %{
      name: safe(fn -> Brain.ML.Generation.name() end) || :unknown,
      ready: safe(fn -> Brain.ML.Generation.ready?() end) || false
    }
  end

  # Cheap: the black-box sampler's last snapshot health (a lock-free ETS read), nil
  # until the first sample — never probes ~51 GenServers from the console.
  defp systems_health do
    case safe(fn -> Fleet.Systems.Sampler.latest() end) do
      %{health: h} when is_map(h) -> h
      _ -> %{}
    end
  end

  defp systems_color(%{health_status: :healthy}), do: "text-success"
  defp systems_color(%{health_status: s}) when s in [:degraded, :warning], do: "text-warning"
  defp systems_color(%{health_status: :critical}), do: "text-error"
  defp systems_color(_), do: "text-base-content/40"

  defp systems_pill_class(%{health_status: :healthy}), do: "border-success/30 text-success"
  defp systems_pill_class(%{health_status: s}) when s in [:degraded, :warning], do: "border-warning/30 text-warning"
  defp systems_pill_class(%{health_status: :critical}), do: "border-error/30 text-error"
  defp systems_pill_class(_), do: "border-base-300 text-base-content/60"

  # A human, at-a-glance summary of the crew and the ship — the line under the title.
  defp status_line(roster, systems) do
    if roster == [] do
      "No officers commissioned yet — commission your first below."
    else
      n = length(roster)
      active = count(roster, &(&1.duty == :active))
      working = count(roster, & &1.working)
      officers = if n == 1, do: "1 officer", else: "#{n} officers"
      work = if working > 0, do: " · #{working} at work", else: ""

      sys =
        case systems[:health_status] do
          :healthy -> " · all systems nominal"
          nil -> ""
          s -> " · systems #{s}"
        end

      "#{officers} · #{active} on duty#{work}#{sys}"
    end
  end

  defp load_detail(agent_id) do
    status = safe(fn -> Fleet.status(agent_id) end) || %{}
    # Fleet.Service and Fleet.DutyLog are both keyed by soul_id, not agent_id —
    # the two coincide for the common case (agent_id defaults to soul_id at
    # commission) but diverge for a second instance commissioned under its
    # own identity from the same soul (see the Commission form's "agent id"
    # override). Looking these up by agent_id would silently return another
    # agent's record (or nothing) whenever they differ.
    soul_id = status[:soul_id] || agent_id

    {summary, history} =
      case safe(fn -> Fleet.Service.load(soul_id) end) do
        {:ok, %{summary: s, history: h}} -> {s, h}
        _ -> {nil, []}
      end

    notes = safe(fn -> Fleet.DutyLog.for_soul(soul_id) end) || []
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
      agent_id: agent_id,
      soul_id: soul_id,
      rank: status[:rank] || Fleet.Rank.default(),
      duty: (summary && String.to_atom(summary.duty_status)) || status[:duty] || :active,
      co: status[:co],
      mind_world_id: status[:mind_world_id] || Fleet.MindWorld.id(agent_id),
      granted_tools: Fleet.Authority.granted_tools(Fleet.Authority.to_set(status[:grants])),
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

  # The last 50 hail/hail_reply CommandRecords, paired into conversation
  # exchanges. A "hail" and its "hail_reply" are two separate audit rows with
  # no shared order_id (a hail confers no assignment) — they're mirror images
  # (hail: A→B, reply: B→A) matched chronologically per (A,B) pair. A hail
  # with no matching reply yet (still in flight, or the caller disconnected
  # before Ensign replied) surfaces as "awaiting reply", never silently dropped.
  defp load_hail_log do
    safe(fn ->
      from(r in CommandRecord, where: r.kind in ["hail", "hail_reply"], order_by: [desc: r.inserted_at], limit: 50)
      |> Atlas.Repo.all()
      |> pair_hails()
    end) || []
  end

  defp pair_hails(records) do
    {pending, done} =
      records
      |> Enum.reverse()
      |> Enum.reduce({%{}, []}, fn r, {pending, done} -> apply_hail_record(r, pending, done) end)

    still_pending =
      pending
      |> Map.values()
      |> Enum.map(&Map.take(&1, [:at, :from, :to, :question]))

    (still_pending ++ done) |> Enum.sort_by(& &1.at, {:desc, DateTime})
  end

  defp apply_hail_record(%{kind: "hail"} = r, pending, done) do
    entry = %{at: r.inserted_at, from: r.from_agent, to: r.to_agent, question: get_in(r.payload || %{}, ["question"])}
    {Map.put(pending, {r.from_agent, r.to_agent}, entry), done}
  end

  defp apply_hail_record(%{kind: "hail_reply"} = r, pending, done) do
    key = {r.to_agent, r.from_agent}
    outcome = %{answer: get_in(r.payload || %{}, ["answer"]), error: get_in(r.payload || %{}, ["error"])}

    case Map.pop(pending, key) do
      {nil, pending} ->
        {pending, [Map.merge(%{at: r.inserted_at, from: r.to_agent, to: r.from_agent, question: nil}, outcome) | done]}

      {question_entry, pending} ->
        {pending, [Map.merge(question_entry, outcome) | done]}
    end
  end

  defp apply_hail_record(_r, pending, done), do: {pending, done}

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

  # agent_id defaults to soul_id (CrewSupervisor.start_ensign/1) so
  # re-commissioning the same soul resumes the existing crew member — that's
  # deliberate, for idempotent rehydration. A blank override preserves that;
  # a non-blank one stands up a genuinely separate instance of the soul under
  # its own identity, which is otherwise unreachable from this form.
  defp commission_opts(rank, agent_id_override) do
    case agent_id_override && String.trim(agent_id_override) do
      "" -> [rank: rank]
      nil -> [rank: rank]
      custom_id -> [rank: rank, agent_id: custom_id]
    end
  end

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

  # agent_id is the real unique identity; soul_id is just which soul it's
  # playing (they coincide unless commissioned with a custom agent id — see
  # the Commission form). Show agent_id always, soul_id only when it adds
  # information, so two instances of the same soul are never indistinguishable.
  defp agent_label(%{agent_id: id, soul_id: id}), do: id
  defp agent_label(%{agent_id: agent_id, soul_id: soul_id}), do: "#{agent_id} (#{soul_id})"

  defp uptime_str(ms) when is_integer(ms) and ms >= 60_000, do: "#{div(ms, 60_000)}m"
  defp uptime_str(ms) when is_integer(ms), do: "#{div(ms, 1000)}s"
  defp uptime_str(_), do: "—"

  defp format_bytes(n) when is_integer(n) and n >= 1_000_000_000, do: "#{Float.round(n / 1_000_000_000, 1)} GB"
  defp format_bytes(n) when is_integer(n) and n >= 1_000_000, do: "#{Float.round(n / 1_000_000, 1)} MB"
  defp format_bytes(n) when is_integer(n), do: "#{n} B"
  defp format_bytes(_), do: "—"

  # Filters the roster by a case-insensitive substring match on soul id, rank,
  # or CO — empty query returns everything unchanged.
  defp filtered_roster(roster, ""), do: roster

  defp filtered_roster(roster, query) do
    q = String.downcase(query)

    Enum.filter(roster, fn a ->
      String.contains?(String.downcase(to_string(a.agent_id)), q) or
        String.contains?(String.downcase(to_string(a.soul_id)), q) or
        String.contains?(String.downcase(to_string(a.rank)), q) or
        String.contains?(String.downcase(to_string(a.co || "")), q)
    end)
  end

  # Orders the roster so each CO is immediately followed by its reports
  # (depth-first), and stamps a :depth on each row for indentation — the
  # visual stand-in for the chain-of-command graph that's actually persisted
  # in Atlas, without needing a canvas/graph renderer.
  # Walking down from roots (co == nil, or co not in the roster) covers a
  # well-formed chain. Fleet.assign_co/2 now rejects any wiring that would
  # close a loop, so a cycle should never reach here — but this is a display
  # for the Admiral to trust, not a place to bet on that: every agent not
  # reached from a root (which, by construction, can only happen if it's
  # stuck in a cycle with no path in from a real root — e.g. a bad chain
  # written before this guard existed) is still rendered, flagged, at depth
  # 0, rather than silently dropped from the table.
  defp hierarchy(roster) do
    by_co = Enum.group_by(roster, & &1.co)
    known_ids = MapSet.new(roster, & &1.agent_id)
    roots = Enum.filter(roster, fn a -> is_nil(a.co) or not MapSet.member?(known_ids, a.co) end)

    reached =
      roots
      |> Enum.sort_by(& &1.soul_id)
      |> Enum.flat_map(&walk(&1, by_co, 0, MapSet.new()))

    reached_ids = MapSet.new(reached, & &1.agent_id)

    unreached =
      roster
      |> Enum.reject(&MapSet.member?(reached_ids, &1.agent_id))
      |> Enum.sort_by(& &1.soul_id)
      |> Enum.map(&Map.merge(&1, %{depth: 0, cycle_member: true}))

    reached ++ unreached
  end

  defp walk(agent, by_co, depth, seen) do
    if MapSet.member?(seen, agent.agent_id) do
      # Cycle guard — Fleet.assign_co/2 rejects wiring that would create one,
      # so this only fires on data that predates that guard. Stop walking
      # rather than loop forever; hierarchy/1's unreached-agents pass still
      # picks up anything this cuts off, so nothing silently disappears.
      []
    else
      seen = MapSet.put(seen, agent.agent_id)

      children =
        by_co
        |> Map.get(agent.agent_id, [])
        |> Enum.sort_by(& &1.soul_id)
        |> Enum.flat_map(&walk(&1, by_co, depth + 1, seen))

      [Map.put(agent, :depth, depth) | children]
    end
  end

  defp filtered_feed(feed, :all), do: feed
  defp filtered_feed(feed, :safety), do: Enum.filter(feed, &(&1.event in @safety_events))

  defp recent_dissent_count(feed), do: Enum.count(feed, &(&1.event in [:dissent, :order_dissented, :deny]))

  defp severity_border(e) when e in @safety_events, do: "border-error/60"
  defp severity_border(e) when e in [:hail, :hail_reply], do: "border-secondary/50"
  defp severity_border(_), do: "border-transparent"

  defp duty_class(:active), do: "badge-success"
  defp duty_class(:relieved), do: "badge-warning"
  defp duty_class(_), do: "badge-ghost"

  defp rank_class(:captain), do: "badge-primary"
  defp rank_class(:executive_officer), do: "badge-secondary"
  defp rank_class(:security), do: "badge-accent"
  defp rank_class(:lieutenant), do: "badge-info"
  defp rank_class(_), do: "badge-ghost"

  defp status_class("completed"), do: "badge-success"
  defp status_class("dissented"), do: "badge-warning"
  defp status_class("failed"), do: "badge-error"
  defp status_class("blocked"), do: "badge-info"
  defp status_class(_), do: "badge-ghost"

  defp event_class(e) when e in [:report, :ack, :cognition_complete, :granted, :grant, :commissioned, :reinstated],
    do: "badge-success"

  defp event_class(e) when e in [:hail, :hail_reply], do: "badge-secondary"

  defp event_class(e) when e in [:dissent, :deny, :provenance_anomaly, :cognition_failed, :order_failed, :order_dissented],
    do: "badge-error"

  defp event_class(e) when e in [:relieved, :relieve, :request, :rehydrated], do: "badge-warning"
  defp event_class(_), do: "badge-ghost"
end
