defmodule Fleet.Ensign do
  @moduledoc """
  An ensign: one GenServer per agent, holding its own `%Brain.Soul{}` as identity
  and its place in the chain of command as data (`co` / `reports` / `grants`).

  It is a non-blocking controller, both reactive and autonomous, and it speaks
  the full §4.1 command protocol over OTP mailboxes:

    * Reactive — ORDER / signals arrive as casts; the ensign authenticates the
      sender (`Fleet.Comms.attribute/1` — the Registry vouches, not the payload),
      enforces bounded topology, and never runs heavy work inside a callback.
    * Autonomous — a self-scheduled `:tick` gates the standing order against the
      agent's authority grant and dispatches cognition when authorized.
    * Non-blocking — appraisal + cognition run in a supervised Task; results come
      back via `handle_info/2` as a REPORT or a DISSENT.

  Duty status (`:active | :relieved`) is a real, reversible suspension.
  """

  use GenServer
  require Logger

  alias Fleet.{Order, Signal, Comms, Authority, Appraisal, Audit, Telemetry}

  @default_tick_interval 5_000
  @request_timeout 30_000

  # ── Client API ────────────────────────────────────────────────────────────

  def start_link(opts \\ []) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(agent_id))
  end

  @doc "Deliver an ORDER (stamps the caller as the issuer via `Fleet.Comms`)."
  def order(agent_id, %Order{} = order), do: Comms.order(agent_id, order)

  @doc "Wire this ensign's commanding officer (in-process chain data)."
  def set_co(agent_id, co_id), do: GenServer.cast(via_tuple(agent_id), {:set_co, co_id})

  @doc "Add a direct report to this ensign (in-process chain data)."
  def add_report(agent_id, report_id), do: GenServer.cast(via_tuple(agent_id), {:add_report, report_id})

  @doc "Have a CO issue an ORDER to one of its reports (runs in the CO's process)."
  def issue_order(co_agent_id, report_id, directive, opts \\ []),
    do: GenServer.cast(via_tuple(co_agent_id), {:issue_order, report_id, directive, opts})

  @doc "Have a CO relieve one of its reports of duty."
  def relieve_report(co_agent_id, report_id),
    do: GenServer.cast(via_tuple(co_agent_id), {:relieve_report, report_id})

  @doc "Have a CO reinstate one of its reports."
  def reinstate_report(co_agent_id, report_id),
    do: GenServer.cast(via_tuple(co_agent_id), {:reinstate_report, report_id})

  @doc "Have a report send a SITREP up to its CO."
  def sitrep(agent_id, body \\ %{}), do: GenServer.cast(via_tuple(agent_id), {:emit_sitrep, body})

  @doc "True once the ensign has hydrated its soul (identity present)."
  def ready?(agent_id) do
    GenServer.call(via_tuple(agent_id), :ready?, 100)
  catch
    :exit, _ -> false
  end

  @doc "A public snapshot of the ensign's state."
  def status(agent_id), do: GenServer.call(via_tuple(agent_id), :status, 5_000)

  # ── Server callbacks ──────────────────────────────────────────────────────

  @impl true
  def init(opts) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    soul_id = Keyword.get(opts, :soul_id)
    interval = Keyword.get(opts, :tick_interval, @default_tick_interval)

    Process.send_after(self(), :tick, interval)
    send(self(), :hydrate_soul)

    state = %{
      agent_id: agent_id,
      soul_id: soul_id,
      soul: Keyword.get(opts, :soul),
      world_id: Keyword.get(opts, :world_id, "default"),
      context_tags: %{
        rank: Keyword.get(opts, :rank, :ensign),
        co: Keyword.get(opts, :co),
        ship: Keyword.get(opts, :ship),
        reports: Keyword.get(opts, :reports, []),
        grants: seed_grants(opts)
      },
      duty: :active,
      assignment: nil,
      # Authorities conferred for the CURRENT assignment only (order grant +
      # GRANTs received while working it). Kept separate from the standing grant
      # so privilege never accumulates across orders (least privilege per order).
      order_grants: MapSet.new(),
      awaiting: nil,
      relieved_by: nil,
      task_ref: nil,
      task_pid: nil,
      last_signals: [],
      tick_interval: interval,
      last_ack: nil,
      started_at: System.monotonic_time(:millisecond)
    }

    Telemetry.emit_event(agent_id, :spawned, %{}, %{soul_id: soul_id})
    {:ok, state}
  end

  @impl true
  def handle_call(:ready?, _from, state), do: {:reply, not is_nil(state.soul), state}

  def handle_call(:status, _from, state) do
    ct = state.context_tags

    public = %{
      agent_id: state.agent_id,
      soul_id: state.soul_id,
      soul_loaded: not is_nil(state.soul),
      world_id: state.world_id,
      duty: state.duty,
      rank: ct.rank,
      co: ct.co,
      reports: ct.reports,
      grants: MapSet.to_list(ct.grants),
      order_grants: MapSet.to_list(state.order_grants),
      assignment_status: state.assignment && state.assignment.status,
      awaiting: state.awaiting,
      working: not is_nil(state.task_ref),
      uptime_ms: System.monotonic_time(:millisecond) - state.started_at
    }

    {:reply, public, state}
  end

  # ── Chain wiring ──────────────────────────────────────────────────────────

  @impl true
  def handle_cast({:set_co, co_id}, state) do
    # An agent may never be its own CO — that would let it authorize/grant to
    # itself (self-command privilege loop).
    if co_id == state.agent_id do
      Telemetry.emit_event(state.agent_id, :self_command_rejected, %{}, %{role: :co})
      {:noreply, state}
    else
      {:noreply, put_in(state.context_tags.co, co_id)}
    end
  end

  def handle_cast({:add_report, report_id}, state) do
    if report_id == state.agent_id do
      Telemetry.emit_event(state.agent_id, :self_command_rejected, %{}, %{role: :report})
      {:noreply, state}
    else
      reports = Enum.uniq([report_id | state.context_tags.reports])
      {:noreply, put_in(state.context_tags.reports, reports)}
    end
  end

  # ── ORDER (downward: CO/Admiral → this ensign) ────────────────────────────

  def handle_cast({:order, %Order{} = order, sender_pid}, state) do
    sender = Comms.attribute(sender_pid)

    cond do
      state.duty == :relieved ->
        verdict = %{basis: :relieved, reason: "relieved of duty"}
        emit_dissent(state, order, verdict)
        Audit.record(:dissent, %{order_id: order.id, from_agent: state.agent_id,
                                 to_agent: Comms.principal_string(sender), verdict: :relieved,
                                 reason: "relieved of duty"})
        {:noreply, state}

      not Comms.authorized_issuer?(sender, state) ->
        Audit.record(:provenance_anomaly, %{order_id: order.id,
          from_agent: Comms.principal_string(sender), to_agent: state.agent_id,
          reason: "order issuer is not my CO"})
        Telemetry.emit_event(state.agent_id, :provenance_anomaly, %{}, %{
          order_id: order.id, sender: Comms.principal_string(sender)
        })
        emit_dissent(state, order, %{basis: :provenance, reason: "issuer not my CO"})
        {:noreply, state}

      true ->
        accept_order(state, order, sender)
    end
  end

  # ── CO-side directives (run in the CO's process; self() is the CO) ────────

  def handle_cast({:issue_order, report_id, directive, opts}, state) do
    if Authority.holds?(state.context_tags.grants, :issue_orders) do
      world_id = Keyword.get(opts, :world_id, state.world_id)
      required = [:cognition, {:world, world_id}]
      # A CO may only confer authorities it itself holds — no delegating what you
      # lack. Anything the CO can't confer, the report must REQUEST (and be DENYd).
      authorities =
        opts
        |> Keyword.get(:authorities, required)
        |> Enum.filter(&Authority.holds?(state.context_tags.grants, &1))

      order = %Order{
        id: gen_id(),
        from: state.agent_id,
        directive: directive,
        grant: %{authorities: authorities},
        world_id: world_id,
        dry_run: Keyword.get(opts, :dry_run, false),
        priority: Keyword.get(opts, :priority, "normal"),
        issued_at: System.monotonic_time(:millisecond)
      }

      Comms.order(report_id, order)
      {:noreply, state}
    else
      Telemetry.emit_event(state.agent_id, :unauthorized_action, %{}, %{action: :issue_orders})
      {:noreply, state}
    end
  end

  def handle_cast({:relieve_report, report_id}, state) do
    if Authority.holds?(state.context_tags.grants, :relieve) do
      Comms.signal(report_id, Signal.new(:relieve, reason: "relieved by CO"))
    else
      Telemetry.emit_event(state.agent_id, :unauthorized_action, %{}, %{action: :relieve})
    end

    {:noreply, state}
  end

  def handle_cast({:reinstate_report, report_id}, state) do
    Comms.signal(report_id, Signal.new(:reinstate, reason: "reinstated by CO"))
    {:noreply, state}
  end

  # ── SITREP emission (report → its CO) ─────────────────────────────────────

  def handle_cast({:emit_sitrep, body}, state) do
    sig = Signal.new(:sitrep, order_id: assignment_id(state), payload: as_map(body), world_id: state.world_id)
    co = state.context_tags.co
    if co, do: Comms.signal(co, sig)
    Audit.record(:sitrep, %{order_id: assignment_id(state), from_agent: state.agent_id,
                            to_agent: co || "admiral", payload: as_map(body)})
    {:noreply, state}
  end

  # ── Signals (attribution + bounded topology) ──────────────────────────────

  def handle_cast({:signal, %Signal{kind: kind, from_pid: p} = sig}, state)
      when kind in [:sitrep, :request, :dissent, :report] do
    sender = Comms.attribute(p)

    if Comms.from_report?(sender, state) do
      handle_upward(kind, sig, sender, state)
    else
      signal_anomaly(kind, sig, sender, state)
    end
  end

  def handle_cast({:signal, %Signal{kind: kind, from_pid: p} = sig}, state)
      when kind in [:grant, :deny] do
    sender = Comms.attribute(p)

    if Comms.authorized_issuer?(sender, state) and awaiting_request?(state, sig) do
      handle_grant_deny(kind, sig, state)
    else
      signal_anomaly(kind, sig, sender, state)
    end
  end

  def handle_cast({:signal, %Signal{kind: kind, from_pid: p} = sig}, state)
      when kind in [:relieve, :reinstate] do
    sender = Comms.attribute(p)

    if Comms.authorized_issuer?(sender, state) do
      handle_duty(kind, sig, sender, state)
    else
      signal_anomaly(kind, sig, sender, state)
    end
  end

  # ── Autonomous tick: gate the standing order against the grant ────────────

  @impl true
  def handle_info(:tick, state) do
    Process.send_after(self(), :tick, state.tick_interval)
    {:messages, queue} = Process.info(self(), :messages)
    Telemetry.emit_event(state.agent_id, :tick, %{queue_len: length(queue)})

    cond do
      state.duty == :relieved ->
        {:noreply, state}

      is_nil(state.assignment) or state.assignment.status != "acknowledged" ->
        {:noreply, state}

      not is_nil(state.task_ref) ->
        {:noreply, state}

      true ->
        gate_and_dispatch(state)
    end
  end

  # ── Soul hydration (unchanged from Phase 1) ───────────────────────────────

  def handle_info(:hydrate_soul, %{soul: %Brain.Soul{}} = state) do
    Telemetry.emit_event(state.agent_id, :soul_hydrated, %{}, %{soul_id: state.soul_id})
    record_readiness(true)
    {:noreply, state}
  end

  def handle_info(:hydrate_soul, %{soul_id: nil} = state), do: {:noreply, state}

  def handle_info(:hydrate_soul, state) do
    case Brain.Soul.get(state.soul_id) do
      {:ok, soul} ->
        Telemetry.emit_event(state.agent_id, :soul_hydrated, %{}, %{soul_id: state.soul_id})
        record_readiness(true)
        {:noreply, %{state | soul: soul}}

      {:error, reason} ->
        Logger.warning("Ensign soul hydration failed",
          agent_id: state.agent_id, soul_id: state.soul_id, reason: inspect(reason))

        Telemetry.emit_event(state.agent_id, :soul_hydrate_failed, %{}, %{
          soul_id: state.soul_id, reason: inspect(reason)})

        {:noreply, state}
    end
  end

  # ── REQUEST timeout ───────────────────────────────────────────────────────

  def handle_info({:request_timeout, rid}, %{awaiting: %{request_id: rid}} = state) do
    order = state.assignment
    verdict = %{basis: :authority_timeout, reason: "authority request timed out"}
    emit_dissent(state, order, verdict)
    Audit.record(:dissent, %{order_id: order && order.id, from_agent: state.agent_id,
                             to_agent: co_or_admiral(state), verdict: :authority_timeout,
                             reason: "request #{rid} timed out"})
    {:noreply, %{state | awaiting: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "dissented")}}
  end

  def handle_info({:request_timeout, _rid}, state), do: {:noreply, state}

  # ── Cognition Task results ────────────────────────────────────────────────

  def handle_info({ref, {:dissent, verdict}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    order = state.assignment
    Audit.record(:dissent, %{order_id: order && order.id, from_agent: state.agent_id,
                             to_agent: co_or_admiral(state), verdict: verdict[:basis],
                             reason: verdict[:reason]})
    emit_dissent(state, order, verdict)
    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "dissented")}}
  end

  def handle_info({ref, {:completed, result}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    order = state.assignment
    dt = System.monotonic_time(:millisecond) - state.started_at

    Audit.record(:report, %{order_id: order && order.id, from_agent: state.agent_id,
                            to_agent: co_or_admiral(state), payload: %{outcome: summarize(result)}})

    sig = Signal.new(:report, order_id: order && order.id,
                     payload: %{outcome: summarize(result)}, world_id: state.world_id)
    deliver_upward(state, sig)

    Telemetry.emit_event(state.agent_id, :cognition_complete, %{duration_ms: dt},
      %{order_id: order && order.id})
    Telemetry.emit_event(state.agent_id, :report, %{}, %{order_id: order && order.id})

    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "completed")}}
  end

  # Any other Task return is treated as a completed result.
  def handle_info({ref, other}, %{task_ref: ref} = state) do
    handle_info({ref, {:completed, other}}, state)
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task_ref: ref} = state) do
    order = state.assignment
    Telemetry.emit_event(state.agent_id, :cognition_failed, %{}, %{
      order_id: order && order.id, reason: inspect(reason)})
    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "failed")}}
  end

  # ACK readback from a report (backward-compatible raw tuple).
  def handle_info({:ack, ack}, state) do
    {:noreply, %{state | last_ack: ack}}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ── ORDER acceptance ──────────────────────────────────────────────────────

  defp accept_order(state, %Order{} = order, sender) do
    ack = %{order_id: order.id, agent_id: state.agent_id, status: :accepted,
            accepted_at: System.monotonic_time(:millisecond), note: nil}

    if is_pid(order.reply_to), do: send(order.reply_to, {:ack, ack})

    Audit.record(:order, %{order_id: order.id, from_agent: Comms.principal_string(sender),
                           to_agent: state.agent_id, world_id: order.world_id,
                           payload: %{directive: to_string(order.directive)}})
    Audit.record(:ack, %{order_id: order.id, from_agent: state.agent_id,
                         to_agent: Comms.principal_string(sender)})

    # The order's grant is scoped to THIS assignment — it replaces (does not
    # accumulate onto) any prior order's conferred authorities.
    order_grants = Authority.to_set(Authority.conferred_by(order))

    Telemetry.emit_event(state.agent_id, :order_received, %{}, %{order_id: order.id})
    send(self(), :tick)

    {:noreply,
     %{state
       | assignment: Order.update_status(order, "acknowledged"),
         order_grants: order_grants,
         last_ack: ack}}
  end

  # ── Tick gate: enforce the grant, then dispatch ───────────────────────────

  defp gate_and_dispatch(state) do
    order = state.assignment

    cond do
      order.dry_run ->
        start_dispatch(state, order)

      true ->
        case Authority.missing(effective_grants(state), Authority.required_for(order)) do
          [] ->
            start_dispatch(state, order)

          [missing | _] ->
            block_and_request(state, order, missing)
        end
    end
  end

  # Standing grant (from commission) ∪ the current order's scoped grant.
  defp effective_grants(state),
    do: MapSet.union(state.context_tags.grants, state.order_grants)

  defp start_dispatch(state, order) do
    task = dispatch(state, order)

    {:noreply,
     %{state
       | task_ref: task.ref,
         task_pid: task.pid,
         assignment: Order.update_status(order, "in_progress")}}
  end

  defp block_and_request(state, %Order{} = order, missing) do
    case state.context_tags.co do
      nil ->
        # No CO to ask — a terminal structural dissent.
        verdict = %{basis: :authority, reason: "missing #{inspect(missing)}, no CO to request from"}
        emit_dissent(state, order, verdict)
        Audit.record(:dissent, %{order_id: order.id, from_agent: state.agent_id,
                                 to_agent: "admiral", verdict: :authority, reason: inspect(missing)})
        {:noreply, %{state | order_grants: MapSet.new(), assignment: Order.update_status(order, "dissented")}}

      co ->
        rid = gen_id()
        sig = Signal.new(:request, order_id: order.id, request_id: rid, authority: missing,
                         payload: %{reason: "requires #{inspect(missing)}"}, world_id: order.world_id)
        Comms.signal(co, sig)
        Audit.record(:request, %{order_id: order.id, from_agent: state.agent_id,
                                 to_agent: co, authority: missing})
        Telemetry.emit_event(state.agent_id, :request, %{}, %{order_id: order.id, authority: missing})
        Process.send_after(self(), {:request_timeout, rid}, @request_timeout)

        {:noreply,
         %{state
           | assignment: Order.update_status(order, "blocked"),
             awaiting: %{authority: missing, order_id: order.id, request_id: rid}}}
    end
  end

  # ── Cognition dispatch (appraisal then Brain.evaluate, off the mailbox) ────

  defp dispatch(state, %Order{} = order) do
    soul = state.soul

    fun =
      if order.dry_run do
        fn -> Process.sleep(50); {:completed, %{response: "[dry-run] " <> to_string(order.directive)}} end
      else
        fn ->
          case Appraisal.appraise(order, soul) do
            {:dissent, verdict} ->
              {:dissent, verdict}

            :proceed ->
              {:ok, conversation_id} = Brain.create_conversation(world_id: order.world_id)
              {:completed, Brain.evaluate(conversation_id, order.directive, [])}
          end
        end
      end

    Task.Supervisor.async_nolink(Fleet.TaskSupervisor, fun)
  end

  # ── Upward signal handling (CO side) ──────────────────────────────────────

  defp handle_upward(:request, %Signal{} = sig, {:ensign, report_id}, state) do
    if Authority.grantable?(state.context_tags.grants, sig.authority) do
      Comms.signal(report_id, Signal.new(:grant, authority: sig.authority,
        order_id: sig.order_id, request_id: sig.request_id, payload: %{request_id: sig.request_id}))
      Audit.record(:grant, %{order_id: sig.order_id, from_agent: state.agent_id,
                             to_agent: report_id, authority: sig.authority})
      Telemetry.emit_event(state.agent_id, :grant, %{}, %{to: report_id, authority: sig.authority})
    else
      Comms.signal(report_id, Signal.new(:deny, authority: sig.authority,
        order_id: sig.order_id, request_id: sig.request_id,
        reason: "CO does not hold #{inspect(sig.authority)}", payload: %{request_id: sig.request_id}))
      Audit.record(:deny, %{order_id: sig.order_id, from_agent: state.agent_id,
                            to_agent: report_id, authority: sig.authority,
                            reason: "CO does not hold #{inspect(sig.authority)}"})
      Telemetry.emit_event(state.agent_id, :deny, %{}, %{to: report_id, authority: sig.authority})
    end

    {:noreply, state}
  end

  defp handle_upward(kind, %Signal{} = sig, {:ensign, report_id}, state) when kind in [:report, :sitrep, :dissent] do
    Telemetry.emit_event(state.agent_id, kind, %{}, %{from: report_id, order_id: sig.order_id})
    {:noreply, %{state | last_signals: Enum.take([{kind, report_id, sig} | state.last_signals], 20)}}
  end

  # ── GRANT / DENY (requester side) ─────────────────────────────────────────

  defp handle_grant_deny(:grant, %Signal{} = sig, state) do
    # A granted authority is scoped to the current order, not added to the
    # standing grant — it does not survive into the next assignment.
    order_grants = Authority.confer(state.order_grants, sig.authority)
    order = state.assignment && Order.update_status(state.assignment, "acknowledged")
    Telemetry.emit_event(state.agent_id, :granted, %{}, %{authority: sig.authority})
    send(self(), :tick)

    {:noreply, %{state | order_grants: order_grants, awaiting: nil, assignment: order}}
  end

  defp handle_grant_deny(:deny, %Signal{} = sig, state) do
    order = state.assignment
    verdict = %{basis: :denied_authority, reason: sig.reason || "authority denied"}
    emit_dissent(state, order, verdict)
    Audit.record(:dissent, %{order_id: order && order.id, from_agent: state.agent_id,
                             to_agent: co_or_admiral(state), verdict: :denied_authority,
                             reason: verdict.reason})
    {:noreply, %{state | awaiting: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "dissented")}}
  end

  # ── RELIEVE / REINSTATE ───────────────────────────────────────────────────

  defp handle_duty(:relieve, %Signal{}, sender, state) do
    state = cancel_inflight(state)
    Telemetry.emit_event(state.agent_id, :relieved, %{}, %{by: Comms.principal_string(sender)})
    Audit.record(:relieve, %{from_agent: Comms.principal_string(sender), to_agent: state.agent_id})
    assignment = state.assignment && Order.update_status(state.assignment, "failed")
    {:noreply, %{state | duty: :relieved, relieved_by: sender, task_ref: nil, task_pid: nil,
                 order_grants: MapSet.new(), assignment: assignment}}
  end

  defp handle_duty(:reinstate, %Signal{}, sender, state) do
    Telemetry.emit_event(state.agent_id, :reinstated, %{}, %{by: Comms.principal_string(sender)})
    Audit.record(:reinstate, %{from_agent: Comms.principal_string(sender), to_agent: state.agent_id})
    # A standing order (still "acknowledged"/"blocked") resumes on the next tick.
    assignment =
      case state.assignment do
        %Order{status: s} = o when s in ["failed", "blocked"] -> Order.update_status(o, "acknowledged")
        other -> other
      end

    send(self(), :tick)
    {:noreply, %{state | duty: :active, relieved_by: nil, assignment: assignment}}
  end

  # ── Anomalies & dissent delivery ──────────────────────────────────────────

  defp signal_anomaly(kind, %Signal{} = sig, sender, state) do
    Audit.record(:provenance_anomaly, %{order_id: sig.order_id,
      from_agent: Comms.principal_string(sender), to_agent: state.agent_id,
      reason: "unauthorized #{kind} signal"})
    Telemetry.emit_event(state.agent_id, :provenance_anomaly, %{}, %{
      kind: kind, sender: Comms.principal_string(sender)})
    {:noreply, state}
  end

  # Deliver a DISSENT to the legitimate chain (CO if set, else the order issuer)
  # and emit the (uniform) dissent telemetry. Every dissent path routes here.
  defp emit_dissent(state, order, verdict) do
    Telemetry.emit_event(state.agent_id, :dissent, %{}, %{
      order_id: order && order.id, basis: verdict[:basis]})

    sig = Signal.new(:dissent, order_id: order && order.id,
                     reason: verdict[:reason], payload: as_map(verdict), world_id: state.world_id)
    deliver_upward(state, sig, order)
  end

  defp deliver_upward(state, sig, order \\ nil) do
    order = order || state.assignment

    case state.context_tags.co do
      nil ->
        if order && is_pid(order.reply_to), do: Comms.signal_pid(order.reply_to, sig)

      co ->
        Comms.signal(co, sig)
    end
  end

  # ── Helpers ───────────────────────────────────────────────────────────────

  defp cancel_inflight(%{task_ref: nil} = state), do: state

  defp cancel_inflight(%{task_ref: ref, task_pid: pid} = state) do
    Process.demonitor(ref, [:flush])
    if is_pid(pid), do: Task.Supervisor.terminate_child(Fleet.TaskSupervisor, pid)
    %{state | task_ref: nil, task_pid: nil}
  end

  defp awaiting_request?(%{awaiting: %{request_id: rid}}, %Signal{request_id: rid}) when not is_nil(rid), do: true
  defp awaiting_request?(%{awaiting: %{request_id: rid}}, %Signal{payload: %{request_id: rid}}) when not is_nil(rid), do: true
  defp awaiting_request?(_, _), do: false

  defp co_or_admiral(%{context_tags: %{co: nil}}), do: "admiral"
  defp co_or_admiral(%{context_tags: %{co: co}}), do: co

  defp assignment_id(%{assignment: %Order{id: id}}), do: id
  defp assignment_id(_), do: nil

  defp seed_grants(opts) do
    from_grant =
      case Keyword.get(opts, :grant) do
        %{authorities: a} -> a
        _ -> []
      end

    (Keyword.get(opts, :grants, []) ++ from_grant) |> Authority.to_set()
  end

  defp summarize({:ok, %{response: r}}) when is_binary(r), do: String.slice(r, 0, 500)
  defp summarize({:ok, r}) when is_binary(r), do: String.slice(r, 0, 500)
  defp summarize(%{response: r}) when is_binary(r), do: String.slice(r, 0, 500)
  defp summarize(r) when is_binary(r), do: String.slice(r, 0, 500)
  defp summarize(other), do: %{raw: inspect(other) |> String.slice(0, 500)}

  defp as_map(m) when is_map(m), do: m
  defp as_map(other), do: %{value: other}

  defp record_readiness(ready?) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      Brain.Metrics.Aggregator.record_readiness(:ensign, ready?)
    end
  end

  defp gen_id, do: :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)

  defp via_tuple(agent_id), do: {:via, Registry, {Fleet.Registry, {:ensign, agent_id}}}
end
