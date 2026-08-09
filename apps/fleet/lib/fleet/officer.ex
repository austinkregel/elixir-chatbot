defmodule Fleet.Officer do
  @moduledoc """
  An officer: one GenServer per agent, holding its own `%Brain.Soul{}` as identity
  and its place in the chain of command as data (`co` / `reports` / `grants`).

  It is a non-blocking controller, both reactive and autonomous, and it speaks
  the full §4.1 command protocol over OTP mailboxes:

    * Reactive — ORDER / signals arrive as casts; the officer authenticates the
      sender (`Fleet.Comms.attribute/1` — the Registry vouches, not the payload),
      enforces bounded topology, and never runs heavy work inside a callback.
    * Autonomous — a self-scheduled `:tick` gates the standing order against the
      agent's authority grant and dispatches cognition when authorized.
    * Non-blocking — appraisal + cognition run in a supervised Task; results come
      back via `handle_info/2` as a REPORT or a DISSENT.

  Duty status (`:active | :relieved`) is a real, reversible suspension.

  ## Orders are directives, not conversation

  Cognition for an order runs `Brain.evaluate/3` with `mode: :directive` (and
  `require_llm: true`). The two flags answer different questions: `mode`
  decides what gets *decided* — `Brain.Directive.Assessor` assesses the order
  deterministically before anything is generated, and the chat
  response-optionality heuristics are skipped, since an order is never optional
  to answer. `require_llm` decides only what may *phrase* the result.

  Hails keep `mode: :chat`: a hail is a question to the agent, not a command.

  The assessment's verdict chooses the outcome before Fleet's own handling:

    * `:comply`  — normal cognition, then a REPORT whose payload carries the
      assessment alongside the (unchanged) `:outcome` summary.
    * `:clarify` — the order is BLOCKED and a SITREP carries the deterministic
      clarification prompts upward. Nothing was generated.
    * `:refuse`  — DISSENT with `basis: :epistemic`, through the same machinery
      as a value dissent.

  ## Tool use is proposed, never taken

  If cognition asks for a capability instead of answering, `Fleet.ToolRound`
  runs exactly one gated round: the model's ```propose block is parsed, and
  `Fleet.Dispatcher` decides against grants read from *this process's* state
  (never the payload) before anything executes. The result returns framed as
  DATA for a single follow-up turn.
  """

  use GenServer
  require Logger

  alias Brain.Directive.Assessment

  alias Fleet.{
    Appraisal,
    Audit,
    Authority,
    Comms,
    GeneralOrders,
    Order,
    Review,
    Signal,
    Telemetry,
    ToolRound
  }

  @default_tick_interval 5_000
  @request_timeout 30_000

  # ── Client API ────────────────────────────────────────────────────────────

  def start_link(opts \\ []) do
    agent_id = Keyword.fetch!(opts, :agent_id)
    GenServer.start_link(__MODULE__, opts, name: via_tuple(agent_id))
  end

  @doc "Deliver an ORDER (stamps the caller as the issuer via `Fleet.Comms`)."
  def order(agent_id, %Order{} = order), do: Comms.order(agent_id, order)

  @doc "Wire this officer's CO (in-process chain data)."
  def set_co(agent_id, co_id), do: GenServer.cast(via_tuple(agent_id), {:set_co, co_id})

  @doc "Add a direct report to this officer (in-process chain data)."
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

  @doc """
  Admiral-console action: grant this agent a standing authority (most often a
  tool, via `Fleet.Authority.tool/1`) directly — not routed through the
  REQUEST/GRANT signal protocol, which is reserved for a report's own
  in-flight request to its CO. This is administrative configuration, the same
  category as `commission`/`assign_co`, audited the same way a chain-granted
  authority is.
  """
  def grant_standing_authority(agent_id, authority),
    do: GenServer.cast(via_tuple(agent_id), {:grant_standing_authority, authority})

  @doc "Admiral-console action: revoke a standing authority from this agent."
  def revoke_standing_authority(agent_id, authority),
    do: GenServer.cast(via_tuple(agent_id), {:revoke_standing_authority, authority})

  @doc "Have a report send a SITREP up to its CO."
  def sitrep(agent_id, body \\ %{}), do: GenServer.cast(via_tuple(agent_id), {:emit_sitrep, body})

  @doc """
  Hail an officer — ask it a question and get its in-character answer, without
  giving an order. The reply arrives asynchronously as `{:hail_reply, map}` in the
  calling process's mailbox (the caller is stamped as the sender). Conversation,
  not command: no assignment, no service-record milestone, no conferred authority.
  """
  def hail(agent_id, question) when is_binary(question), do: Comms.hail(agent_id, question)

  @doc """
  Route a model turn through the tool gate: parse a `Fleet.Proposal`, then (if the
  agent's grant permits) let the harness dispatch it. The grant checked is the
  agent's own order-conferred grant from process state — never anything the model
  wrote. Returns the dispatcher's verdict (`{:ok, %{data:}}` | `{:refused, _}` |
  `{:error, _}`), `:no_proposal`, or `{:refused, :untethered}` for an unparseable
  or requirement-less proposal.
  """
  def propose(agent_id, model_output) when is_binary(model_output),
    do: GenServer.call(via_tuple(agent_id), {:propose, model_output}, 30_000)

  @doc """
  Cast a Security veto against this officer's current act.

  The vetoing officer must hold `:veto` and must not be the officer being
  vetoed — a veto of oneself is not review. `subject` names a tool, or `nil` to
  veto the act as a whole. Once recorded the veto stands for the current
  assignment; a new order clears it.
  """
  @spec veto(String.t(), String.t(), MapSet.t(), String.t(), String.t() | nil) :: :ok
  def veto(agent_id, reviewer_id, reviewer_grants, cause, subject \\ nil),
    do:
      GenServer.cast(
        via_tuple(agent_id),
        {:veto, reviewer_id, reviewer_grants, cause, subject}
      )

  @doc """
  Record a review sign-off toward the two-officer rule for this officer's
  current act.

  Two approvals from two *distinct* officers, neither of them the acting agent,
  are required before an irreversible act may proceed.
  """
  @spec sign_off(String.t(), String.t(), MapSet.t(), atom(), Fleet.Review.decision()) :: :ok
  def sign_off(agent_id, reviewer_id, reviewer_grants, authority, decision \\ :approve),
    do:
      GenServer.cast(
        via_tuple(agent_id),
        {:sign_off, reviewer_id, reviewer_grants, authority, decision}
      )

  @doc "Have the agent append a note to its own duty log (agent-written)."
  def log_duty(agent_id, note, opts \\ []),
    do: GenServer.cast(via_tuple(agent_id), {:log_duty, note, opts})

  @doc "True once the officer has hydrated its soul (identity present)."
  def ready?(agent_id) do
    GenServer.call(via_tuple(agent_id), :ready?, 100)
  catch
    :exit, _ -> false
  end

  @doc "A public snapshot of the officer's state."
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
      # The agent's private mind-world (memory/beliefs/JTMS scope), bound once the
      # soul is hydrated. Cognition runs here, not in the order's world.
      mind_world_id: nil,
      context_tags: %{
        rank: Keyword.get(opts, :rank, :ensign),
        co: Keyword.get(opts, :co),
        # The ship this agent is posted to — defaults to THIS instance; a cross-ship
        # agent (future research/science ships) carries a different id.
        ship: Keyword.get(opts, :ship) || Fleet.Ship.id(),
        reports: Keyword.get(opts, :reports, []),
        grants: seed_grants(opts)
      },
      duty: :active,
      assignment: nil,
      # Authorities conferred for the CURRENT assignment only (order grant +
      # GRANTs received while working it). Kept separate from the standing grant
      # so privilege never accumulates across orders (least privilege per order).
      order_grants: MapSet.new(),
      # Review state for the CURRENT assignment: sign-offs collected toward the
      # two-officer rule, and vetoes cast against it. Scoped to the order for
      # the same reason grants are — approval of one act is not approval of the
      # next. Both are written by the harness from chain-of-command signals,
      # never by anything cognition produces.
      signoffs: [],
      vetoes: [],
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
    # Rehydrate the durable self (service record) after init returns.
    {:ok, state, {:continue, :rehydrate}}
  end

  @impl true
  def handle_continue(:rehydrate, state) do
    case Fleet.Service.load(state.soul_id) do
      {:ok, %{summary: nil}} ->
        # First-ever commission — create the durable record.
        Fleet.Service.commission(state.soul_id, commission_attrs(state))
        Telemetry.emit_event(state.agent_id, :commissioned, %{}, %{soul_id: state.soul_id})
        {:noreply, state}

      {:ok, %{summary: %Atlas.Schemas.ServiceSummary{} = summary}} ->
        # Restart — resume the durable self without forgetting who we are.
        Fleet.Service.record_rehydrated(state.soul_id, %{"from" => "restart"})
        Telemetry.emit_event(state.agent_id, :rehydrated, %{}, %{soul_id: state.soul_id})
        {:noreply, apply_summary(state, summary)}
    end
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
      mind_world_id: state.mind_world_id,
      duty: state.duty,
      rank: ct.rank,
      ship_id: ct.ship,
      co: ct.co,
      reports: ct.reports,
      grants: MapSet.to_list(ct.grants),
      order_grants: MapSet.to_list(state.order_grants),
      assignment_status: state.assignment && state.assignment.status,
      order_id: state.assignment && state.assignment.id,
      directive: state.assignment && state.assignment.directive,
      awaiting: state.awaiting,
      working: not is_nil(state.task_ref),
      uptime_ms: System.monotonic_time(:millisecond) - state.started_at
    }

    {:reply, public, state}
  end

  # ── Tool proposal gate (the model proposes; the harness disposes) ──────────

  def handle_call({:propose, model_output}, _from, state) do
    reply =
      case Fleet.Proposal.parse(model_output) do
        :none ->
          :no_proposal

        {:error, reason} ->
          # An untethered / malformed proposal never reaches the gate.
          {:refused, reason}

        {:ok, proposal} ->
          # The grant AND the reader principal are read from process state — the
          # agent's own order-conferred authorities, billet, live duty, and ship
          # commission — NOT from anything the model wrote. This is what makes a
          # forged authority/clearance claim in the model's text inert.
          Fleet.Dispatcher.dispatch(proposal, tool_ctx(state))
      end

    {:reply, reply, state}
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

  # ── ORDER (downward: CO/Admiral → this officer) ────────────────────────────

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

  # ── HAIL (conversation: reach the agent's soul without an order) ───────────

  def handle_cast({:hail, question, sender_pid}, state) do
    sender = Comms.attribute(sender_pid)

    cond do
      is_nil(state.soul) or is_nil(state.mind_world_id) ->
        send(sender_pid, {:hail_reply, %{agent_id: state.agent_id, question: question,
          error: "not ready — soul not yet hydrated"}})
        {:noreply, state}

      not may_hail?(sender, state) ->
        Audit.record(:provenance_anomaly, %{from_agent: Comms.principal_string(sender),
          to_agent: state.agent_id, reason: "hail from outside the chain"})
        send(sender_pid, {:hail_reply, %{agent_id: state.agent_id, question: question,
          error: "not permitted to hail this agent"}})
        {:noreply, state}

      true ->
        principal = Comms.principal_string(sender)

        Audit.record(:hail, %{from_agent: principal, to_agent: state.agent_id,
          world_id: state.mind_world_id, payload: %{question: String.slice(question, 0, 2000)}})
        Telemetry.emit_event(state.agent_id, :hail, %{}, %{from: principal})

        # Conversation runs OFF the mailbox in a fully detached task, so the officer
        # stays responsive (you can hail Security mid-order) and a hail crash never
        # touches the agent or its assignment. The task replies to the caller.
        start_hail(state, question, principal, sender_pid)
        {:noreply, state}
    end
  end

  # A hail may come from the Admiral (who may converse with any crew member) or,
  # for agent-to-agent conversation, only along this agent's chain — its CO or a
  # direct report. Conversation respects bounded topology just like commands do.
  defp may_hail?(:admiral, _state), do: true
  defp may_hail?(sender, state), do: Comms.from_co?(sender, state) or Comms.from_report?(sender, state)

  # Run the hail's cognition in the agent's own mind-world, with its soul — the
  # same accountable path an order uses — but conferring nothing and recording a
  # conversation, not an assignment. Errors are surfaced in the reply, not hidden.
  defp start_hail(state, question, principal, reply_to) do
    soul =
      state.soul
      |> GeneralOrders.apply_to()
      |> soul_with_tool_context(state.context_tags.grants)
    agent_id = state.agent_id
    mind_world_id = state.mind_world_id

    Task.Supervisor.start_child(Fleet.TaskSupervisor, fn ->
      reply =
        try do
          {:ok, conversation_id} =
            Brain.create_conversation(world_id: mind_world_id, soul: soul, agent_id: agent_id)

          answer =
            hail_answer(
              Brain.evaluate(conversation_id, question,
                soul: soul, agent_id: agent_id, require_llm: true)
            )

          audit_hail_reply(agent_id, principal, mind_world_id, %{answer: String.slice(answer, 0, 2000)})
          Telemetry.emit_event(agent_id, :hail_reply, %{}, %{to: principal})

          %{agent_id: agent_id, question: question, answer: answer}
        rescue
          e ->
            Logger.error("Fleet.Officer: hail cognition failed",
              agent_id: agent_id, reason: inspect(e))

            audit_hail_reply(agent_id, principal, mind_world_id, %{error: Exception.message(e)})
            %{agent_id: agent_id, question: question, error: Exception.message(e)}
        catch
          kind, reason ->
            Logger.error("Fleet.Officer: hail cognition crashed",
              agent_id: agent_id, reason: inspect({kind, reason}))

            audit_hail_reply(agent_id, principal, mind_world_id, %{error: inspect({kind, reason})})
            %{agent_id: agent_id, question: question, error: inspect({kind, reason})}
        end

      send(reply_to, {:hail_reply, reply})
    end)
  end

  # Always writes a hail_reply record — success or failure — so a failed hail
  # is inspectable in the durable log, not just in the live caller's mailbox.
  # A payload with :error (no :answer) is how a caller distinguishes the two.
  defp audit_hail_reply(agent_id, principal, mind_world_id, payload) do
    Audit.record(:hail_reply, %{from_agent: agent_id, to_agent: principal,
      world_id: mind_world_id, payload: payload})
  end

  defp hail_answer({:ok, %{response: r}}) when is_binary(r), do: r
  defp hail_answer(%{response: r}) when is_binary(r), do: r
  defp hail_answer({:ok, r}) when is_binary(r), do: r
  defp hail_answer(r) when is_binary(r), do: r

  # Brain.evaluate/3 returns {:error, reason} (rather than raising) when
  # require_llm generation fails — that must become a real failure here, not
  # get inspect()'d into text and handed back as if the agent said it.
  defp hail_answer({:error, reason}), do: raise("hail cognition returned an error: #{inspect(reason)}")
  defp hail_answer(other), do: inspect(other)

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

      order =
        Order.new(
          id: gen_id(),
          from: state.agent_id,
          objective: directive,
          constraints: Keyword.get(opts, :constraints, []),
          context_refs: Keyword.get(opts, :context_refs, []),
          risk_class: Keyword.get(opts, :risk_class, :routine),
          grant: %{authorities: authorities},
          world_id: world_id,
          dry_run: Keyword.get(opts, :dry_run, false),
          priority: Keyword.get(opts, :priority, "normal"),
          issued_at: System.monotonic_time(:millisecond)
        )

      Comms.order(report_id, order)
      {:noreply, state}
    else
      Telemetry.emit_event(state.agent_id, :unauthorized_action, %{}, %{action: :issue_orders})
      {:noreply, state}
    end
  end

  # Review arrives over the command channel from a holder of the authority. The
  # reviewer's grants are passed in by the caller (the console or the reviewing
  # officer's own process) — Fleet.Review re-checks entitlement and self-review
  # before anything is recorded, so a bad call is refused, not trusted.
  def handle_cast({:veto, reviewer_id, reviewer_grants, cause, subject}, state) do
    ctx = %{
      agent_id: state.agent_id,
      order_id: assignment_id(state),
      world_id: state.mind_world_id,
      subject: subject
    }

    case Review.veto(ctx, reviewer_id, reviewer_grants, cause) do
      {:vetoed, record} ->
        Telemetry.emit_event(state.agent_id, :vetoed, %{}, %{by: reviewer_id, subject: subject})
        {:noreply, %{state | vetoes: state.vetoes ++ [record]}}

      {:error, reason} ->
        Telemetry.emit_event(state.agent_id, :review_refused, %{}, %{
          by: reviewer_id,
          kind: :veto,
          reason: reason
        })

        {:noreply, state}
    end
  end

  def handle_cast({:sign_off, reviewer_id, reviewer_grants, authority, decision}, state) do
    ctx = %{
      agent_id: state.agent_id,
      order_id: assignment_id(state),
      world_id: state.mind_world_id,
      subject: assignment_id(state)
    }

    case Review.plan_review(ctx, reviewer_id, reviewer_grants, decision) do
      {:error, reason} ->
        Telemetry.emit_event(state.agent_id, :review_refused, %{}, %{
          by: reviewer_id,
          kind: :sign_off,
          reason: reason
        })

        {:noreply, state}

      {_outcome, _record} ->
        signoff = %{agent_id: reviewer_id, authority: authority, decision: decision}

        Telemetry.emit_event(state.agent_id, :signed_off, %{}, %{
          by: reviewer_id,
          decision: decision_label(decision)
        })

        {:noreply, %{state | signoffs: state.signoffs ++ [signoff]}}
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

  def handle_cast({:grant_standing_authority, authority}, state) do
    grants = Authority.confer(state.context_tags.grants, authority)

    Audit.record(:grant, %{from_agent: "admiral", to_agent: state.agent_id,
      authority: Authority.encode(authority)})
    Telemetry.emit_event(state.agent_id, :grant, %{}, %{authority: authority})

    {:noreply, put_in(state.context_tags.grants, grants)}
  end

  def handle_cast({:revoke_standing_authority, authority}, state) do
    grants = MapSet.delete(state.context_tags.grants, authority)

    Audit.record(:deny, %{from_agent: "admiral", to_agent: state.agent_id,
      authority: Authority.encode(authority), reason: "revoked by Admiral"})
    Telemetry.emit_event(state.agent_id, :deny, %{}, %{authority: authority})

    {:noreply, put_in(state.context_tags.grants, grants)}
  end

  # ── SITREP emission (report → its CO) ─────────────────────────────────────

  def handle_cast({:log_duty, note, opts}, state) do
    Fleet.DutyLog.note(
      state.soul_id,
      note,
      Keyword.merge([order_id: assignment_id(state), world_id: state.mind_world_id], opts)
    )

    {:noreply, state}
  end

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

  def handle_info(:hydrate_soul, %{soul: %Brain.Soul{} = soul} = state) do
    mind_world_id = ensure_mind_world(soul)
    Telemetry.emit_event(state.agent_id, :soul_hydrated, %{}, %{soul_id: state.soul_id})
    record_readiness(true)
    {:noreply, %{state | mind_world_id: mind_world_id}}
  end

  def handle_info(:hydrate_soul, %{soul_id: nil} = state), do: {:noreply, state}

  def handle_info(:hydrate_soul, state) do
    case Brain.Soul.get(state.soul_id) do
      {:ok, soul} ->
        mind_world_id = ensure_mind_world(soul)
        Telemetry.emit_event(state.agent_id, :soul_hydrated, %{}, %{soul_id: state.soul_id})
        record_readiness(true)
        {:noreply, %{state | soul: soul, mind_world_id: mind_world_id}}

      {:error, reason} ->
        Logger.warning("Officer soul hydration failed",
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
    Fleet.Service.record_order_outcome(state.soul_id, order, :dissented,
      %{reason: verdict[:reason], basis: verdict[:basis]})
    emit_dissent(state, order, verdict)
    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "dissented")}}
  end

  # A directive the deterministic assessment could not act on without an answer.
  # The order is BLOCKED, not failed and not dissented: nothing is wrong with it
  # except that a parameter or a premise needs settling. The clarification text
  # is SlotDetector's own template, travelling as structured data in a
  # machine-to-machine signal — no generator produced it.
  def handle_info({ref, {:clarify, %Assessment{} = a}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    order = state.assignment
    payload = Assessment.to_report_payload(a)

    Audit.record(:sitrep, %{order_id: order && order.id, from_agent: state.agent_id,
                            to_agent: co_or_admiral(state), payload: payload})

    Fleet.Service.record_order_outcome(state.soul_id, order, :blocked, payload)

    sig = Signal.new(:sitrep, order_id: order && order.id,
                     reason: Assessment.explain(a),
                     payload: payload, world_id: state.world_id)

    deliver_upward(state, sig)

    Telemetry.emit_event(state.agent_id, :directive_clarification, %{}, %{
      order_id: order && order.id, rule: payload[:rule]})

    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "blocked")}}
  end

  # Cognition proposed a tool the agent does not hold, tethered to a stated
  # requirement. The order is not finished and not failed — it needs authority,
  # so it takes the same REQUEST/GRANT path as an order that arrived without
  # the grant it needs. A GRANT re-ticks and cognition runs again with the tool.
  def handle_info({ref, {:needs_authority, authority, _meta}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    state = %{state | task_ref: nil, task_pid: nil}

    case state.assignment do
      %Order{} = order ->
        Telemetry.emit_event(state.agent_id, :tool_authority_requested, %{}, %{
          order_id: order.id,
          authority: authority
        })

        block_and_request(state, order, [authority])

      nil ->
        {:noreply, state}
    end
  end

  def handle_info({ref, {:completed, result}}, %{task_ref: ref} = state) do
    handle_info({ref, {:completed, result, %{}}}, state)
  end

  def handle_info({ref, {:completed, result, meta}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    order = state.assignment
    dt = System.monotonic_time(:millisecond) - state.started_at

    # :outcome keeps its exact historical shape; everything the deterministic
    # assessment learned rides alongside it as sibling keys, so a CO reading a
    # report gets the analysis rather than 500 characters of prose.
    payload = Map.merge(%{outcome: summarize(result)}, meta)

    Audit.record(:report, %{order_id: order && order.id, from_agent: state.agent_id,
                            to_agent: co_or_admiral(state), payload: payload})
    Fleet.Service.record_order_outcome(state.soul_id, order, :completed, payload)

    sig = Signal.new(:report, order_id: order && order.id,
                     payload: payload, world_id: state.world_id)
    deliver_upward(state, sig)

    Telemetry.emit_event(state.agent_id, :cognition_complete, %{duration_ms: dt},
      %{order_id: order && order.id})
    Telemetry.emit_event(state.agent_id, :report, %{}, %{order_id: order && order.id})

    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "completed")}}
  end

  # A cognition call that returned an error (e.g. require_llm generation
  # unavailable) rather than crashing the task — same outcome as the :DOWN
  # clause below (failed, not completed), just reached via a clean return
  # instead of a process exit.
  def handle_info({ref, {:failed, reason}}, %{task_ref: ref} = state) do
    Process.demonitor(ref, [:flush])
    order = state.assignment

    Fleet.Service.record_order_outcome(state.soul_id, order, :failed, %{reason: inspect(reason)})
    Telemetry.emit_event(state.agent_id, :cognition_failed, %{}, %{
      order_id: order && order.id, reason: inspect(reason)})

    {:noreply, %{state | task_ref: nil, task_pid: nil, order_grants: MapSet.new(),
                 assignment: order && Order.update_status(order, "failed")}}
  end

  # Any other Task return is treated as a completed result.
  def handle_info({ref, other}, %{task_ref: ref} = state) do
    handle_info({ref, {:completed, other}}, state)
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task_ref: ref} = state) do
    order = state.assignment
    Telemetry.emit_event(state.agent_id, :cognition_failed, %{}, %{
      order_id: order && order.id, reason: inspect(reason)})
    Fleet.Service.record_order_outcome(state.soul_id, order, :failed, %{reason: inspect(reason)})
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
    readback = readback(order)

    ack = %{order_id: order.id, agent_id: state.agent_id, status: :accepted,
            accepted_at: System.monotonic_time(:millisecond), note: nil,
            readback: readback}

    if is_pid(order.reply_to), do: send(order.reply_to, {:ack, ack})

    Audit.record(:order, %{order_id: order.id, from_agent: Comms.principal_string(sender),
                           to_agent: state.agent_id, world_id: order.world_id,
                           payload: %{
                             directive: to_string(order.directive),
                             constraints: order.constraints,
                             risk_class: to_string(Order.risk_class(order))
                           }})
    Audit.record(:ack, %{order_id: order.id, from_agent: state.agent_id,
                         to_agent: Comms.principal_string(sender),
                         payload: readback})
    Fleet.Service.record_assignment(state.soul_id, order)
    # What I was ordered enters my own mind, attributed to the issuer.
    ingest_communication(state, :order, order.directive, sender)

    # The order's grant is scoped to THIS assignment — it replaces (does not
    # accumulate onto) any prior order's conferred authorities. Review state is
    # cleared for the same reason: sign-off on a previous act is not sign-off
    # on this one.
    order_grants = Authority.to_set(Authority.conferred_by(order))

    Telemetry.emit_event(state.agent_id, :order_received, %{}, %{order_id: order.id})
    send(self(), :tick)

    {:noreply,
     %{state
       | assignment: Order.update_status(order, "acknowledged"),
         order_grants: order_grants,
         signoffs: [],
         vetoes: [],
         last_ack: ack}}
  end

  # The readback: what the ship understood the order to be, restated from the
  # order as received, before any work begins.
  #
  # The LCARS readback is load-bearing because it catches misunderstanding
  # *before* execution and leaves a comparison artifact for drift detection.
  # Here it restates what the harness parsed — objective, the bounds it will
  # enforce, the risk class it will apply, and the authority it believes it
  # holds — so an issuer can see immediately if the order was received as
  # something other than what they meant.
  #
  # It is deliberately the *harness's* restatement rather than a model
  # paraphrase. Cognition here only phrases responses; what matters for drift is
  # what the deterministic layer will act on. What the *analysis* made of the
  # order (its read of the intent, which is the part measured to be unreliable)
  # travels separately, on the assessment in the REPORT.
  defp readback(%Order{} = order) do
    %{
      objective: to_string(order.objective || order.directive),
      constraints: order.constraints || [],
      risk_class: to_string(Order.risk_class(order)),
      authorities: order |> Authority.conferred_by() |> Enum.map(&Authority.encode/1),
      context_refs: order.context_refs || []
    }
  end

  # ── Tick gate: enforce the grant, then dispatch ───────────────────────────

  defp gate_and_dispatch(state) do
    order = state.assignment

    cond do
      order.dry_run ->
        start_dispatch(state, order)

      # Risk class is process, not a label. An irreversible order proceeds only
      # with two distinct officers' sign-off (Fleet.Review.two_officer/3);
      # without it the order is refused on the record rather than quietly
      # executed as if it were routine.
      Order.risk_class(order) == :irreversible and
          Review.two_officer(state.signoffs, state.agent_id) != :ok ->
        refuse_irreversible(state, order)

      true ->
        case Authority.missing(effective_grants(state), Authority.required_for(order)) do
          [] ->
            start_dispatch(state, order)

          [missing | _] ->
            block_and_request(state, order, missing)
        end
    end
  end

  defp refuse_irreversible(state, order) do
    detail =
      case Review.two_officer(state.signoffs, state.agent_id) do
        {:error, reason} -> to_string(reason)
        _ -> "unknown"
      end

    verdict = %{
      basis: :process,
      rule: :irreversible_requires_two_officers,
      reason:
        "order is classed irreversible and lacks two-officer sign-off (#{detail}), " <>
          "so it cannot be lawfully executed"
    }

    Audit.record(:dissent, %{order_id: order.id, from_agent: state.agent_id,
                             to_agent: co_or_admiral(state), verdict: verdict[:basis],
                             reason: verdict[:reason]})

    Fleet.Service.record_order_outcome(state.soul_id, order, :dissented,
      %{reason: verdict[:reason], basis: verdict[:basis]})

    emit_dissent(state, order, verdict)

    {:noreply,
     %{state | order_grants: MapSet.new(), assignment: Order.update_status(order, "dissented")}}
  end

  # Standing grant (from commission) ∪ the current order's scoped grant.
  defp effective_grants(state),
    do: MapSet.union(state.context_tags.grants, state.order_grants)

  # Gives the agent an accurate, standing-orders-level account of which tools
  # it currently holds — the Admiral's grant is otherwise enforced only by the
  # harness (Fleet.Dispatcher), invisible to the agent's own understanding of
  # what it may do. Appended to the constitution, never the per-order
  # directive, so it reads as identity/standing-orders context, not a claim
  # riding in on this turn's input.
  #
  # There is no nil clause: both call sites pipe through `GeneralOrders.apply_to/1`,
  # which returns a soul carrying just the constitution when an officer has none
  # of its own, so a soulless agent still serves under the General Orders.
  defp soul_with_tool_context(soul, grants) do
    case Authority.granted_tools(grants) do
      [] ->
        soul

      tools ->
        # Names alone left a proposing model guessing both what a tool is for and
        # what shape its arguments take; the registry has always carried both.
        catalog =
          tools
          |> Enum.flat_map(fn name ->
            case Fleet.Tool.lookup(name) do
              {:ok, tool} -> [Fleet.Tool.describe(tool)]
              :error -> []
            end
          end)
          |> Enum.join("\n\n")

        appendix =
          "\n\n---\nTools currently granted to you by your chain of command:\n\n" <>
            catalog <>
            "\n\nYou may propose only these — via a fenced ```propose " <>
            "{\"tool\": \"...\", \"requirement\": \"...\", \"args\": {...}}``` block — and nothing " <>
            "else; an unlisted tool will be refused, and so will arguments that do not match " <>
            "the shape above."

        %{soul | constitution: (soul.constitution || "") <> appendix}
    end
  end

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
    agent_id = state.agent_id
    # Built here, on the officer process, so the tool gate reads the agent's
    # real billet, duty, ship and grants from process state — never from
    # anything the model writes. Same construction as the :propose handle_call.
    tool_ctx = tool_ctx(state, order)
    # Cognition runs in the AGENT's mind-world (its own memory/beliefs/JTMS),
    # NOT the order's world — the order's world is task context/authority only.
    mind_world_id = state.mind_world_id

    fun =
      if order.dry_run do
        fn -> Process.sleep(50); {:completed, %{response: "[dry-run] " <> to_string(order.directive)}} end
      else
        fn ->
          case Appraisal.appraise(order, GeneralOrders.apply_to(soul),
                 mind_world_id: mind_world_id,
                 agent_id: agent_id
               ) do
            {:dissent, verdict} ->
              {:dissent, verdict}

            :proceed ->
              # The normative stack, in order: General Orders above the officer's
              # own constitution, then the tool appendix. Appraisal judges against
              # the constitution WITHOUT the tool appendix (it asks what the agent
              # may do, not what it holds), but the General Orders bind there too —
              # "dissent is duty" is article 4.
              cognition_soul =
                soul
                |> GeneralOrders.apply_to()
                |> soul_with_tool_context(effective_grants(state))

              {:ok, conversation_id} =
                Brain.create_conversation(world_id: mind_world_id, soul: cognition_soul, agent_id: agent_id)

              # mode: :directive — an order is not a conversational turn. Brain
              # assesses it deterministically before generating anything, and
              # skips the chat response-optionality heuristics (which would
              # silence two structurally similar consecutive orders).
              #
              # Brain.evaluate/3 returns {:error, reason} rather than raising when
              # require_llm generation fails (Brain's own top-level rescue converts
              # it) — that must not be mistaken for a completed order.
              case Brain.evaluate(conversation_id, order.objective || order.directive,
                     soul: cognition_soul,
                     agent_id: agent_id,
                     require_llm: true,
                     mode: :directive,
                     # What the issuer stated, passed through rather than
                     # re-derived: the assessment gates on these instead of on a
                     # classifier's reading of the objective.
                     directive_constraints: order.constraints,
                     directive_risk_class: Order.risk_class(order)) do
                {:error, reason} ->
                  {:failed, reason}

                result ->
                  result
                  |> classify_cognition()
                  |> maybe_run_tool_round(tool_ctx, conversation_id, cognition_soul, agent_id)
              end
          end
        end
      end

    Task.Supervisor.async_nolink(Fleet.TaskSupervisor, fun)
  end

  # Brain's deterministic directive assessment decides the shape of the outcome
  # before any of Fleet's own handling runs. A withheld directive produced no
  # text at all — no generator was consulted — so it must not be reported as a
  # completed order:
  #
  #   :refuse  -> DISSENT on epistemic grounds (the existing dissent machinery)
  #   :clarify -> the order is blocked pending an answer, reported as a SITREP
  #               carrying the deterministic clarification prompts
  #   :comply  -> normal completion
  #
  # A result without a :directive key (chat mode, or an older Brain) completes
  # as before.
  defp classify_cognition({:ok, %{directive: %Assessment{} = a}} = result) do
    case a.verdict do
      :comply -> {:completed, result, cognition_meta(a)}
      :refuse -> {:dissent, dissent_verdict(a)}
      :clarify -> {:clarify, a}
    end
  end

  defp classify_cognition(result), do: {:completed, result}

  defp dissent_verdict(%Assessment{} = a) do
    %{
      basis: :epistemic,
      rule: Assessment.primary_reason_tag(a) || :directive_refused,
      reason: Assessment.explain(a),
      assessment: Assessment.to_report_payload(a)
    }
  end

  defp cognition_meta(%Assessment{} = a),
    do: %{assessment: Assessment.to_report_payload(a)}

  # If the agent asked for a capability instead of answering, run the single
  # gated tool round and report what came back. Only a completed cognition can
  # propose — a dissent or a blocked directive produced no answer to inspect.
  defp maybe_run_tool_round({:completed, result, meta}, ctx, conversation_id, soul, agent_id) do
    continue = fn framed ->
      # The framed <data> block is the answer to the agent's own question, not
      # an order from a superior: same conversation, no directive assessment.
      #
      # `mode: :tool_result` rather than plain chat because the chat path runs
      # `ResponseGate`, whose conversational-optionality heuristics score a
      # framed data block as not needing a reply — measured, and it silently
      # ended every tool chain after its first call.
      case Brain.evaluate(conversation_id, framed,
             soul: soul,
             agent_id: agent_id,
             require_llm: true,
             mode: :tool_result) do
        {:ok, %{response: r}} when is_binary(r) -> r
        _ -> nil
      end
    end

    case ToolRound.run(response_text(result), ctx, continue) do
      {:final, _text, %{disposition: :none}} ->
        {:completed, result, meta}

      {:final, text, tool_meta} ->
        {:completed, put_response(result, text), Map.put(meta, :tool_use, tool_meta)}

      # Escalation has to happen on the officer process (it owns `awaiting` and
      # the request timeout), so the Task hands the decision back rather than
      # trying to run the REQUEST/GRANT protocol from inside cognition.
      {:needs_authority, authority, tool_meta} ->
        {:needs_authority, authority, Map.put(meta, :tool_use, tool_meta)}
    end
  end

  defp maybe_run_tool_round(outcome, _ctx, _conversation_id, _soul, _agent_id), do: outcome

  # An officer stands in two worlds at once, and a tool has to say which it
  # means. `world_id` is its mind-world — its own beliefs, memory and JTMS — and
  # stays the default because that is what the existing read tools and the audit
  # record refer to. `order_world_id` is the world the ORDER named, where a
  # shared corpus (an indexed codebase, an ingested document set) actually
  # lives. Cognition never leaves the mind-world; only a tool reaches across,
  # and only under the `{:world, id}` authority the order already confers.
  defp tool_ctx(state, order \\ nil) do
    ct = state.context_tags

    %{
      agent_id: state.agent_id,
      order_id: assignment_id(state),
      world_id: state.mind_world_id,
      mind_world_id: state.mind_world_id,
      order_world_id: order_world_id(order || state.assignment),
      tool_budget: tool_budget_for(order || state.assignment),
      ship_id: ct.ship,
      grants: effective_grants(state),
      # Review state travels with the dispatch context so the tool gate can
      # enforce a veto and the two-officer rule at the moment of the call.
      vetoes: state.vetoes,
      signoffs: state.signoffs,
      principal:
        Fleet.Principal.agent(%{
          agent_id: state.agent_id,
          rank: ct.rank,
          duty: state.duty,
          ship_id: ct.ship,
          co: ct.co,
          reports: ct.reports
        })
    }
  end

  defp order_world_id(%Order{world_id: world_id}) when is_binary(world_id), do: world_id
  defp order_world_id(_), do: nil

  # An irreversible order gets a tighter ceiling than a routine one: an act that
  # cannot be undone should not be reached at the end of a long unattended chain
  # of reads, where the reasoning that led there is hardest to review.
  defp tool_budget_for(%Order{} = order) do
    case Order.risk_class(order) do
      :irreversible -> 3
      _ -> Application.get_env(:fleet, :tool_budget, 8)
    end
  end

  defp tool_budget_for(_), do: Application.get_env(:fleet, :tool_budget, 8)

  defp response_text({:ok, %{response: r}}) when is_binary(r), do: r
  defp response_text(%{response: r}) when is_binary(r), do: r
  defp response_text(r) when is_binary(r), do: r
  defp response_text(_), do: nil

  defp put_response({:ok, map}, text) when is_map(map), do: {:ok, Map.put(map, :response, text)}
  defp put_response(map, text) when is_map(map), do: Map.put(map, :response, text)
  defp put_response(_other, text), do: %{response: text}

  # ── Upward signal handling (CO side) ──────────────────────────────────────

  defp handle_upward(:request, %Signal{} = sig, {:officer, report_id}, state) do
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

  defp handle_upward(kind, %Signal{} = sig, {:officer, report_id} = sender, state)
       when kind in [:report, :sitrep, :dissent] do
    Telemetry.emit_event(state.agent_id, kind, %{}, %{from: report_id, order_id: sig.order_id})
    # What a report tells me enters MY mind, attributed to that report.
    ingest_communication(state, kind, sig.payload, sender)
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
    Fleet.Service.record_relief(state.soul_id, Comms.principal_string(sender))
    assignment = state.assignment && Order.update_status(state.assignment, "failed")
    {:noreply, %{state | duty: :relieved, relieved_by: sender, task_ref: nil, task_pid: nil,
                 order_grants: MapSet.new(), assignment: assignment}}
  end

  defp handle_duty(:reinstate, %Signal{}, sender, state) do
    Telemetry.emit_event(state.agent_id, :reinstated, %{}, %{by: Comms.principal_string(sender)})
    Audit.record(:reinstate, %{from_agent: Comms.principal_string(sender), to_agent: state.agent_id})
    Fleet.Service.record_reinstatement(state.soul_id, Comms.principal_string(sender))
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

  defp decision_label(:approve), do: :approve
  defp decision_label({:reject, _}), do: :reject

  defp co_or_admiral(%{context_tags: %{co: nil}}), do: "admiral"
  defp co_or_admiral(%{context_tags: %{co: co}}), do: co

  defp assignment_id(%{assignment: %Order{id: id}}), do: id
  defp assignment_id(_), do: nil

  # The standing grant an agent holds from the moment it is commissioned: the
  # authorities its **billet** confers (Fleet.Rank), unioned with any explicit
  # grants Command passed. Cognition/world stay order-conferred, so a worker
  # billet (ensign/lieutenant) seeds nothing — identical to the pre-billet default.
  defp seed_grants(opts) do
    from_grant =
      case Keyword.get(opts, :grant) do
        %{authorities: a} -> a
        _ -> []
      end

    billet = Fleet.Rank.standing_authorities(Keyword.get(opts, :rank, Fleet.Rank.default()))

    (billet ++ Keyword.get(opts, :grants, []) ++ from_grant) |> Authority.to_set()
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
      Brain.Metrics.Aggregator.record_readiness(:officer, ready?)
    end
  end

  defp gen_id, do: :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)

  # Ensure the agent's private mind-world exists (its soul is the sole resident).
  # No graceful degradation: a failed create crashes the officer loudly.
  defp ensure_mind_world(%Brain.Soul{id: soul_id}) do
    world_id = Fleet.MindWorld.id(soul_id)

    case World.Manager.get(world_id) do
      {:ok, _world} ->
        world_id

      {:error, _} ->
        # Ephemeral: the durable memory/beliefs live in Atlas keyed by world_id;
        # the World *record* is recreated here on every hydrate (idempotent), so
        # it needn't survive a restart on its own — avoids disk-checkpoint churn.
        {:ok, _world} =
          World.Manager.create("mind:" <> soul_id,
            id: world_id, mode: :ephemeral, residents: [soul_id])

        world_id
    end
  end

  # ── Sharing via communication (Stage 5) ───────────────────────────────────

  # Record something the agent was TOLD into its OWN mind-world, stamped with the
  # runtime-attributed sender's provenance. This is the ONLY way another agent's
  # information enters this mind — because it was communicated, never by reading
  # another agent's world. Best-effort enrichment (the accountable record is the
  # audit/service log); a memory hiccup must not take down the agent.
  defp ingest_communication(%{mind_world_id: mind_world_id} = state, kind, content, sender)
       when is_binary(mind_world_id) do
    from = Comms.principal_string(sender)

    if Process.whereis(Brain.Memory.Store) do
      Brain.Memory.Store.add_episode(
        "communication",
        to_string(kind),
        stringify_content(content),
        ["communication", "from:" <> from, to_string(kind)],
        world_id: mind_world_id
      )
    end

    Telemetry.emit_event(state.agent_id, :ingested, %{}, %{kind: kind, from: from})
  rescue
    e -> Logger.warning("Fleet.Officer: communication ingest failed", reason: inspect(e))
  catch
    _, _ -> :ok
  end

  defp ingest_communication(_state, _kind, _content, _sender), do: :ok

  defp stringify_content(c) when is_binary(c), do: c
  defp stringify_content(c), do: inspect(c)

  # ── Durable-self helpers (Stage 4) ────────────────────────────────────────

  defp commission_attrs(state) do
    %{
      agent_id: state.agent_id,
      rank: state.context_tags.rank,
      ship_id: state.context_tags.ship,
      home_world_id: state.world_id,
      mind_world_id: Fleet.MindWorld.id(state.soul_id),
      co_id: state.context_tags.co,
      reports: state.context_tags.reports,
      standing_grants: state.context_tags.grants
    }
  end

  # Restore the DURABLE fields from the service summary; reset the TRANSIENT ones
  # (per-order grants, in-flight task) so least privilege holds across a restart.
  defp apply_summary(state, s) do
    ct = state.context_tags

    %{
      state
      | duty: safe_atom(s.duty_status, :active),
        world_id: s.home_world_id || state.world_id,
        mind_world_id: s.mind_world_id || state.mind_world_id,
        context_tags: %{
          ct
          | rank: safe_atom(s.rank, :ensign),
            ship: s.ship_id || Fleet.Ship.id(),
            co: s.co_id,
            reports: s.reports || [],
            grants: Fleet.Authority.decode_set(s.standing_grants || [])
        },
        assignment: rebuild_assignment(s.current_assignment),
        order_grants: MapSet.new(),
        awaiting: nil,
        task_ref: nil,
        task_pid: nil
    }
  end

  # Rebuild the standing order, downgrading a mid-flight status so the autonomous
  # tick re-drives it — "without forgetting what it was doing."
  defp rebuild_assignment(a) when is_map(a) and map_size(a) > 0 do
    status = Map.get(a, "status", "acknowledged")
    resume_status = if status in ["in_progress", "blocked"], do: "acknowledged", else: status
    authorities = get_in(a, ["grant", "authorities"]) || []

    Order.new(
      id: Map.get(a, "order_id"),
      from: Map.get(a, "from", "admiral"),
      objective: Map.get(a, "objective") || Map.get(a, "directive"),
      constraints: Map.get(a, "constraints", []),
      context_refs: Map.get(a, "context_refs", []),
      risk_class: Map.get(a, "risk_class", :routine),
      grant: %{authorities: Enum.map(authorities, &Fleet.Authority.decode/1)},
      world_id: Map.get(a, "world_id", "default"),
      status: resume_status
    )
  end

  defp rebuild_assignment(_), do: nil

  defp safe_atom(nil, default), do: default
  defp safe_atom(v, _default) when is_atom(v), do: v

  defp safe_atom(v, default) when is_binary(v) do
    String.to_existing_atom(v)
  rescue
    ArgumentError -> default
  end

  defp via_tuple(agent_id), do: {:via, Registry, {Fleet.Registry, {:officer, agent_id}}}
end
