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

  `ctx` is supplied by the harness (the Ensign), holding the runtime-attributed
  `agent_id`, the `order_id`, the `world_id`, and the caller's `grants` — the union
  of standing + order-conferred authorities, straight from process state.
  """

  require Logger
  alias Fleet.{Authority, Tool, Proposal, DataFrame, Audit}

  @type ctx :: %{
          required(:agent_id) => String.t(),
          required(:grants) => MapSet.t(),
          optional(:order_id) => term(),
          optional(:world_id) => term()
        }

  @doc """
  Pure authorisation decision. `{:allow, %Fleet.Tool{}}` or `{:refuse, reason}`.
  Reads only `grants` and the registry — the proposal payload can carry no
  authority, so a forged claim in the model's text is inert here.
  """
  @spec decide(Proposal.t(), MapSet.t()) ::
          {:allow, Tool.t()} | {:refuse, {:unknown_tool, term()} | {:ungranted, term()}}
  def decide(%Proposal{tool: name}, %MapSet{} = grants) do
    case Tool.lookup(name) do
      :error ->
        {:refuse, {:unknown_tool, name}}

      {:ok, %Tool{required_authority: authority} = tool} ->
        if Authority.holds?(grants, authority),
          do: {:allow, tool},
          else: {:refuse, {:ungranted, authority}}
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
      {:refuse, reason} ->
        audit(ctx, :grant_violation, %{tool: proposal.tool, reason: inspect(reason)})
        Logger.warning("Fleet.Dispatcher: REFUSED #{proposal.tool} — #{inspect(reason)}")
        {:refused, reason}

      {:allow, %Tool{} = tool} ->
        audit(ctx, :tool_decision, %{tool: tool.name, verdict: "allow", effect: to_string(tool.effect)})
        run(tool, proposal, ctx)
    end
  end

  defp run(%Tool{} = tool, %Proposal{} = proposal, ctx) do
    case tool.handler.(proposal.args, ctx) do
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
