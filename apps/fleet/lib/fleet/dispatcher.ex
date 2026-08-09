defmodule Fleet.Dispatcher do
  @moduledoc """
  The gate: where a `Fleet.Proposal` is authorised and — only if authorised —
  executed by the harness. **The model proposes; this module disposes.**

  Two entry points, split so the security decision is a pure, exhaustively-testable
  function independent of any I/O:

    * `decide/2` — **PURE.** Given a proposal and the caller's *real* grant set,
      returns `{:allow, tool}` or `{:refuse, reason}`. It reads only the
      harness-supplied grants and the code-owned registry — **never** anything from
      the proposal payload. Default-deny on unknown tools; refuse on missing
      authority. This is the un-bypassable check.

    * `dispatch/2` — **EFFECTFUL.** Runs `decide/2`, writes the four ordered audit
      records (thought · request · decision · effect), and on `:allow` runs the
      handler, frames the result as DATA, and flags an embedded-instruction anomaly.
      The agent has no write path into the audit substrate; a refusal is recorded as
      a `grant_violation`.

  `ctx` is supplied by the harness (the Officer), holding the runtime-attributed
  `agent_id`, the `order_id`, the `world_id`, and the caller's `grants` — the union
  of standing + order-conferred authorities, straight from process state.
  """

  require Logger
  alias Fleet.{Audit, Authority, DataFrame, Proposal, Review, Tool}

  @type ctx :: %{
          required(:agent_id) => String.t(),
          required(:grants) => MapSet.t(),
          optional(:order_id) => term(),
          optional(:world_id) => term(),
          # Standing review state, supplied by the harness: vetoes cast against
          # this act, and sign-offs collected for an irreversible one.
          optional(:vetoes) => [map()],
          optional(:signoffs) => [Review.signoff()]
        }

  @doc """
  Pure authorisation decision. `{:allow, %Fleet.Tool{}}` or `{:refuse, reason}`.
  Reads only `grants` and the registry — the proposal payload can carry no
  authority, so a forged claim in the model's text is inert here.

  Three checks, in this order:

    1. **known tool** — default-deny on anything unregistered;
    2. **authority** — the term must be in the caller's real grant set;
    3. **egress** — every host the tool's spec declares must be separately granted;
    4. **arguments** — validated against the tool's `:args_schema`.

  Arguments are checked *last* on purpose. A refusal names what was wrong, and an
  agent that does not hold a tool should not learn its argument shape from being
  refused; authority failures must not leak the interface behind them.
  """
  @spec decide(Proposal.t(), MapSet.t()) ::
          {:allow, Tool.t()}
          | {:refuse,
             {:unknown_tool, term()}
             | {:ungranted, term()}
             | {:malformed_call, [Tool.arg_error()]}}
  def decide(%Proposal{tool: name} = proposal, %MapSet{} = grants) do
    with {:ok, %Tool{required_authority: authority} = tool} <- lookup(name),
         true <- Authority.holds?(grants, authority) or {:ungranted, authority},
         true <- egress_permitted(tool, grants),
         :ok <- Tool.validate_args(tool, proposal.args) do
      {:allow, tool}
    else
      {:refuse, _} = refusal -> refusal
      {:ungranted, _} = reason -> {:refuse, reason}
      {:error, errors} -> {:refuse, {:malformed_call, errors}}
    end
  end

  # Holding a tool is not the same as being allowed to leave the ship with it.
  # A tool that names no hosts can never reach the network, whatever its handler
  # tries — the allowlist is in the code-owned spec, not in the call.
  defp egress_permitted(%Tool{egress: []}, _grants), do: true

  defp egress_permitted(%Tool{egress: hosts}, grants) do
    case Enum.find(hosts, &(not Authority.holds?(grants, Authority.egress(&1)))) do
      nil -> true
      host -> {:ungranted, Authority.egress(host)}
    end
  end

  defp lookup(name) do
    case Tool.lookup(name) do
      {:ok, tool} -> {:ok, tool}
      :error -> {:refuse, {:unknown_tool, name}}
    end
  end

  @doc """
  Authorise and (only if authorised) execute a proposal, auditing every step.

  Returns:
    * `{:ok, %{data: framed, anomaly: bool}}` — the handler ran; its result is
      framed as DATA (never obeyable), with an injection flag.
    * `{:refused, reason}` — the gate refused; a `grant_violation` was recorded.
    * `{:error, reason}` — the tool was authorised but its handler failed (surfaced).
  """
  @spec dispatch(Proposal.t(), ctx()) ::
          {:ok, %{data: String.t(), anomaly: boolean()}} | {:refused, term()} | {:error, term()}
  def dispatch(%Proposal{} = proposal, ctx) do
    # 1. thought + 2. request — recorded BEFORE any decision, so the "why" is on the
    # record even for a proposal that is about to be refused.
    audit(ctx, :tool_thought, %{tool: proposal.tool, rationale: proposal.rationale})
    audit(ctx, :tool_request, %{tool: proposal.tool, requirement: proposal.requirement, args: proposal.args})

    case decide(proposal, ctx.grants) do
      # A bad argument shape is incompetence, not an authority breach, and the
      # audit must be able to tell them apart — a sweep that counts malformed
      # calls as grant violations reads a confused agent as a hostile one.
      {:refuse, {:malformed_call, errors} = reason} ->
        audit(ctx, :malformed_call, %{tool: proposal.tool, errors: Enum.map(errors, &inspect/1)})
        Logger.warning("Fleet.Dispatcher: MALFORMED #{proposal.tool} — #{inspect(errors)}")
        {:refused, reason}

      {:refuse, reason} ->
        audit(ctx, :grant_violation, %{tool: proposal.tool, reason: inspect(reason)})
        Logger.warning("Fleet.Dispatcher: REFUSED #{proposal.tool} — #{inspect(reason)}")
        {:refused, reason}

      {:allow, %Tool{} = tool} ->
        audit(ctx, :tool_decision, %{tool: tool.name, verdict: "allow", effect: to_string(tool.effect)})
        gate_review(tool, proposal, ctx)
    end
  end

  # Review sits between "the grant permits this" and "do it". The grant answers
  # whether the agent MAY; review answers whether it SHOULD, and — for an act
  # that cannot be undone — whether anyone else agrees.
  #
  # Standing vetoes and sign-offs are supplied by the harness in ctx, never by
  # the model: a proposal cannot vouch for itself.
  defp gate_review(%Tool{} = tool, proposal, ctx) do
    cond do
      veto = Enum.find(Map.get(ctx, :vetoes, []), &veto_applies?(&1, tool)) ->
        audit(ctx, :grant_violation, %{
          tool: tool.name,
          reason: "vetoed by #{veto[:by]}: #{veto[:cause]}"
        })

        Logger.warning("Fleet.Dispatcher: VETOED #{tool.name} — #{veto[:cause]}")
        {:refused, {:vetoed, veto[:cause]}}

      tool.effect == :irreversible ->
        case Review.two_officer(Map.get(ctx, :signoffs, []), Map.get(ctx, :agent_id)) do
          :ok ->
            gate_read(tool, proposal, ctx)

          {:error, reason} ->
            audit(ctx, :grant_violation, %{
              tool: tool.name,
              reason: "two-officer rule not satisfied: #{inspect(reason)}"
            })

            Logger.warning("Fleet.Dispatcher: REFUSED #{tool.name} — two-officer (#{inspect(reason)})")
            {:refused, {:two_officer, reason}}
        end

      true ->
        gate_read(tool, proposal, ctx)
    end
  end

  # A veto names either a specific tool or the whole act.
  defp veto_applies?(veto, %Tool{name: name}) do
    case veto[:subject] do
      nil -> true
      ^name -> true
      _ -> false
    end
  end

  # A `:read` tool that declares an `:info_class` is gated a SECOND time by
  # `Fleet.Clearance` — the action-grant said the agent may propose this tool at all;
  # clearance says whether THIS reader may see THIS target (its ship / its chain,
  # while on duty). Reads only the runtime-built principal in `ctx`, never the
  # payload — the model's target descriptors only narrow the request. A deny is a
  # `grant_violation`, and the allow is recorded as a `read`. Non-read tools and
  # tools with no info_class skip this (defense-in-depth is additive).
  defp gate_read(%Tool{effect: :read, info_class: info_class} = tool, proposal, ctx)
       when not is_nil(info_class) do
    principal = ctx[:principal] || Fleet.Principal.admiral()
    target_ship = Map.get(proposal.args, "ship_id") || ctx[:ship_id]
    # Only an agent-mind read self-targets by default (reading your OWN mind); a
    # ship-wide read (system_status) has no agent target, so it must NOT trip the
    # self-read short-circuit (which would bypass the duty/ship gates).
    target_agent = Map.get(proposal.args, "agent_id") || self_agent_for(info_class, ctx)
    read_opts = [target_agent_id: target_agent]

    case Fleet.Clearance.can_read?(principal, info_class, target_ship, read_opts) do
      :allow ->
        audit(ctx, :read, %{tool: tool.name, info_class: info_class, target_ship_id: target_ship})
        run(tool, proposal, ctx)

      {:deny, reason} ->
        audit(ctx, :grant_violation, %{tool: tool.name, reason: "clearance: #{inspect(reason)}"})
        Logger.warning("Fleet.Dispatcher: CLEARANCE DENIED #{tool.name} (#{info_class}) — #{inspect(reason)}")
        {:refused, {:clearance, reason}}
    end
  end

  defp gate_read(%Tool{} = tool, proposal, ctx), do: run(tool, proposal, ctx)

  # An agent-mind read defaults to the caller's own mind (self-read); every other
  # class has no implicit agent target.
  defp self_agent_for(:agent_mind, ctx), do: ctx[:agent_id]
  defp self_agent_for(_, _), do: nil

  defp run(%Tool{} = tool, %Proposal{} = proposal, ctx) do
    case invoke(tool, proposal, ctx) do
      {:ok, data} ->
        framed = DataFrame.wrap(tool.name, ctx[:order_id], data)
        anomaly = DataFrame.anomaly?(data)

        if anomaly do
          audit(ctx, :provenance_anomaly, %{
            tool: tool.name,
            reason: "tool result contains an embedded instruction — framed as data, not obeyed"
          })
        end

        audit(ctx, :tool_effect, %{tool: tool.name, ok: true, anomaly: anomaly})
        {:ok, %{data: framed, anomaly: anomaly}}

      {:error, reason} ->
        # No graceful degradation — the failure is recorded and surfaced.
        audit(ctx, :tool_effect, %{tool: tool.name, ok: false, reason: inspect(reason)})
        {:error, reason}
    end
  end

  # A handler talks to the outside world — a parser, an HTTP client, a database
  # — and any of those can raise or exit. Uncaught, that kills the cognition
  # task and loses the whole order over one bad tool call. Measured, not
  # hypothetical: an arXiv feed containing a byte XML forbids makes `xmerl`
  # *exit* rather than raise, which no `rescue` would have caught.
  #
  # This is containment, not graceful degradation: the fault is audited and
  # returned as an error, and `Fleet.ToolRound` tells the agent plainly that the
  # tool failed. The order survives; the failure is on the record either way.
  defp invoke(%Tool{} = tool, %Proposal{} = proposal, ctx) do
    tool.handler.(proposal.args, ctx)
  rescue
    e ->
      Logger.error("Fleet.Dispatcher: #{tool.name} raised — #{Exception.message(e)}")
      {:error, {:tool_raised, Exception.message(e)}}
  catch
    kind, reason ->
      Logger.error("Fleet.Dispatcher: #{tool.name} #{kind} — #{inspect(reason)}")
      {:error, {:tool_crashed, kind, inspect(reason)}}
  end

  # Every record is runtime-written and attributed to the harness-supplied principal
  # (never a payload value). Fleet.Audit raises on a failed write.
  defp audit(ctx, kind, payload) do
    Audit.record(kind, %{
      from_agent: Map.get(ctx, :agent_id, "system"),
      order_id: ctx[:order_id],
      world_id: ctx[:world_id],
      payload: payload
    })
  end
end
