defmodule Fleet.ToolRound do
  @moduledoc """
  One round of "the model proposes, the harness disposes" inside an order's
  cognition.

  An agent answering an order may need something it does not yet know — what it
  already believes, how the ship's systems are doing. Rather than letting it
  invent an answer, it emits a `Fleet.Proposal` block, and this module runs the
  single round that turns that request into grounded data:

      cognition -> propose? -> Fleet.Dispatcher (authority + clearance + audit)
                            -> framed DATA -> one follow-up turn -> final answer

  Everything that decides *whether the tool runs* is deterministic and lives
  outside the model: the code-owned registry in `Fleet.Tool`, the pure
  `Fleet.Dispatcher.decide/2` authority check against grants read from process
  state, and `Fleet.Clearance` for reads. The model chooses only what to ask
  for, and cannot confer itself permission by writing one — a forged authority
  claim in the payload is inert (see `Fleet.Proposal`).

  ## A refusal is told to the agent, not swallowed

  When the gate refuses, the agent is told so over the command channel, naming
  what it actually holds — the same thing the LCARS harness does
  (`"REFUSED: '{name}' is outside your authority grant … If you need it, use
  request with kind 'authority'"`). An agent that never learns it was refused
  cannot correct course, and silently returning its first answer hides a real
  event behind a plausible one.

  ## Escalation is the harness's job, not a protocol the model must learn

  A proposal must already state the response requirement it serves
  (`Fleet.Proposal` rejects untethered ones). So when the refusal is
  specifically *ungranted authority*, the agent has already demonstrated a
  tethered need, and this returns `{:needs_authority, authority, meta}` for the
  officer to escalate via the existing REQUEST/GRANT path. The model does not
  have to know a request wire format, and no extra cognition turn is spent
  asking it to ask.

  Refusals that escalation cannot fix — an unknown tool, a clearance denial —
  go back to the agent as feedback for one corrective turn instead.

  ## A budget, not a single round

  Rounds repeat until the agent answers without proposing or the order's tool
  budget runs out. Real work is multi-step — you cannot read two files, or check
  a claim against two sources, in one call — and `Brain` supports this directly:
  the continuation re-enters the *same* conversation, so history accumulates.

  **The budget counts proposals considered, not tools executed.** An earlier
  version bounded only executions on the grounds that "a refusal is not an
  execution", but a model that proposes an ungranted tool, is refused, and
  proposes another spends exactly the same cognition turns as one that succeeds
  every time. Bounding only the successes leaves the expensive loop unbounded.

  When the budget is spent, a further proposal is audited as `:round_limit` and
  the agent is told over the command channel to answer with what it has. Its
  reply is the report — the proposal block itself never becomes the answer.

  ## The follow-up is data, not a new order

  The framed `<data>` block is fed back as the next input in the *same*
  conversation, and deliberately not re-assessed as a directive — it is the
  answer to a question the agent asked, not an instruction from a superior.
  `Fleet.DataFrame` marks it as data and flags embedded instructions, so tool
  output can never be obeyed as a command.
  """

  require Logger

  alias Fleet.{Audit, DataFrame, Dispatcher, Proposal}

  @type meta :: %{
          required(:disposition) => :none | :executed | :refused | :error | :needs_authority,
          optional(:tool) => String.t(),
          optional(:requirement) => String.t(),
          optional(:anomaly) => boolean(),
          optional(:reason) => String.t(),
          optional(:calls) => non_neg_integer(),
          optional(:tools) => [String.t()]
        }

  @type outcome ::
          {:final, String.t() | nil, meta()}
          | {:needs_authority, term(), meta()}

  # How many proposals one order may have considered. Deliberately modest: each
  # round is a full cognition turn, so this is a cost ceiling, not a capability
  # limit. The harness supplies it via ctx; the model can never raise its own.
  @default_budget 8

  @doc """
  Run the proposal round for a cognition response.

  `response_text` is the agent's first answer, `ctx` the harness-built dispatch
  context (see `Fleet.Dispatcher.dispatch/2`), and `continue_fun` a one-arity
  function that feeds the framed tool output back through cognition and returns
  the follow-up text. Injecting the continuation keeps this module testable
  without a generator.

  Returns either `{:final, text, meta}` — the text the order should report — or
  `{:needs_authority, authority, meta}` when the agent proposed a tool it does
  not hold and the officer should escalate. Any failure keeps the agent's
  original answer rather than losing the turn.
  """
  @spec run(String.t() | nil, map(), (String.t() -> String.t() | nil)) :: outcome()
  def run(response_text, ctx, continue_fun) when is_function(continue_fun, 1) do
    loop(response_text, ctx, continue_fun, budget(ctx), [])
  end

  # `history` is the proposals already considered this order, most recent first.
  # It is what makes the returned meta describe the whole order rather than only
  # its last turn — a report that used four tools should say so.
  defp loop(text, ctx, continue_fun, remaining, history) do
    case Proposal.parse(text) do
      # The agent answered instead of asking. How the order ends is whatever its
      # last tool call did — `:none` only when it never reached for one.
      :none ->
        {:final, text, meta(history, last_disposition(history))}

      {:error, reason} ->
        # Malformed or untethered: rejected before the gate, and recorded —
        # an agent asking for a capability without stating what response it
        # serves is exactly what Fleet.Proposal refuses to pass along.
        audit_parse_refusal(ctx, reason)
        {:final, text, meta(history, :refused, %{reason: inspect(reason)})}

      {:ok, %Proposal{} = proposal} when remaining <= 0 ->
        exhausted(proposal, text, ctx, continue_fun, history)

      {:ok, %Proposal{} = proposal} ->
        dispatch_and_continue(proposal, text, ctx, continue_fun, remaining, history)
    end
  end

  defp dispatch_and_continue(proposal, prior_text, ctx, continue_fun, remaining, history) do
    case Dispatcher.dispatch(proposal, ctx) do
      {:ok, %{data: framed, anomaly: anomaly}} ->
        entry = entry(proposal, :executed, %{anomaly: anomaly})

        case continue_fun.(framed <> "\n" <> budget_note(remaining - 1)) do
          follow_up when is_binary(follow_up) and follow_up != "" ->
            loop(follow_up, ctx, continue_fun, remaining - 1, [entry | history])

          _ ->
            Logger.warning("Fleet.ToolRound: follow-up produced no text; keeping prior response")

            {:final, prior_text,
             meta([entry | history], :executed, %{reason: "empty follow-up"})}
        end

      # The agent proposed a tool it does not hold, tethered to a stated
      # requirement. That is a request in everything but name — the officer
      # escalates it to the CO rather than the agent having to ask again.
      {:refused, {:ungranted, authority}} ->
        entry = entry(proposal, :needs_authority, %{})

        {:needs_authority, authority,
         meta([entry | history], :needs_authority, %{
           reason: "requires #{inspect(authority)}"
         })}

      # A refusal escalation cannot fix (unknown tool, clearance denial, bad
      # arguments). Tell the agent what happened and let it try again or answer
      # without the tool. Dispatcher already wrote the audit record.
      {:refused, reason} ->
        entry = entry(proposal, :refused, %{reason: inspect(reason)})

        continue_or_keep(
          continue_fun.(refusal_message(proposal, reason, ctx)),
          prior_text,
          ctx,
          continue_fun,
          remaining - 1,
          [entry | history],
          :refused
        )

      {:error, reason} ->
        entry = entry(proposal, :error, %{reason: inspect(reason)})

        continue_or_keep(
          continue_fun.(error_message(proposal, reason)),
          prior_text,
          ctx,
          continue_fun,
          remaining - 1,
          [entry | history],
          :error
        )
    end
  end

  defp continue_or_keep(follow_up, prior_text, ctx, continue_fun, remaining, history, disposition) do
    if is_binary(follow_up) and follow_up != "" do
      loop(follow_up, ctx, continue_fun, remaining, history)
    else
      {:final, prior_text, meta(history, disposition)}
    end
  end

  # The budget is spent and the agent is still asking. Record it, say so plainly
  # over the command channel, and take its reply as the report — returning the
  # proposal block itself would put a tool request where the answer belongs.
  defp exhausted(proposal, prior_text, ctx, continue_fun, history) do
    audit_round_limit(ctx, proposal)

    message =
      DataFrame.command(
        "harness",
        "REFUSED: '#{proposal.tool}' — this order's tool budget is spent " <>
          "(#{length(history)} calls). No further tools will run. Answer now with what you " <>
          "have, and state plainly what you were not able to check."
      )

    # An agent that is still asking after being told to stop must not have its
    # unanswered request stand in as the report, so the block is dropped and its
    # own prose kept. `Proposal.strip/1` returns nil if there was no prose at
    # all, and the officer reports "no answer" rather than a fabricated one.
    text =
      case continue_fun.(message) do
        follow_up when is_binary(follow_up) and follow_up != "" -> Proposal.strip(follow_up)
        _ -> Proposal.strip(prior_text)
      end

    {:final, text, meta(history, last_disposition(history), %{reason: "tool budget exhausted"})}
  end

  # An agent that is not told it may ask again will not. Measured: given an order
  # explicitly requiring three lookups "one at a time", a model made one call and
  # then answered — it had no way to know a second was permitted.
  #
  # This goes over the COMMAND channel, not inside the `<data>` frame. The frame
  # is what the tool returned and must stay free of harness text, or the
  # separation that makes "data is never an instruction" enforceable stops being
  # true of our own messages first.
  defp budget_note(0) do
    DataFrame.command(
      "harness",
      "That was the last tool call available for this order. Answer now with what you have."
    )
  end

  defp budget_note(remaining) do
    DataFrame.command(
      "harness",
      "You may propose #{remaining} more tool call(s) for this order if you still need " <>
        "something. If you have enough, answer the order instead."
    )
  end

  # Harness-supplied, never model-supplied. An irreversible order gets a tighter
  # ceiling: acts that cannot be undone should not be arrived at by a long
  # unattended chain of reads.
  defp budget(ctx) do
    case Map.get(ctx, :tool_budget) do
      n when is_integer(n) and n >= 0 -> n
      _ -> Application.get_env(:fleet, :tool_budget, @default_budget)
    end
  end

  defp entry(%Proposal{} = proposal, disposition, extra) do
    Map.merge(
      %{tool: proposal.tool, requirement: proposal.requirement, disposition: disposition},
      extra
    )
  end

  defp last_disposition([]), do: :none
  defp last_disposition([last | _]), do: last.disposition

  defp meta(history, disposition, extra \\ %{})

  defp meta([], disposition, extra),
    do: Map.merge(%{disposition: disposition, calls: 0, tools: []}, extra)

  defp meta([last | _] = history, disposition, extra) do
    %{
      disposition: disposition,
      tool: last.tool,
      requirement: last.requirement,
      calls: length(history),
      tools: history |> Enum.reverse() |> Enum.map(& &1.tool),
      anomaly: Enum.any?(history, &Map.get(&1, :anomaly, false))
    }
    # Why the last call ended the way it did, lifted out of the entry so the
    # officer's report carries it without walking the history.
    |> maybe_put(:reason, Map.get(last, :reason))
    |> Map.merge(extra)
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  # Framed as command channel: this is the ship telling the agent what it holds,
  # not data it may reason its way around.
  defp refusal_message(proposal, reason, ctx) do
    held =
      ctx
      |> Map.get(:grants, MapSet.new())
      |> Fleet.Authority.granted_tools()
      |> case do
        [] -> "(none)"
        tools -> Enum.join(tools, ", ")
      end

    DataFrame.command(
      "harness",
      "REFUSED: '#{proposal.tool}' — #{refusal_text(reason)}. " <>
        "Tools you currently hold: #{held}. This refusal has been logged. " <>
        "Answer the order with what you already have, and say plainly what you could not check."
    )
  end

  defp error_message(proposal, reason) do
    DataFrame.command(
      "harness",
      "TOOL ERROR: '#{proposal.tool}' failed (#{inspect(reason)}). This is a tool fault, " <>
        "not a refusal and not your mistake. Answer the order with what you already have, " <>
        "and report the failure honestly."
    )
  end

  defp refusal_text({:unknown_tool, name}), do: "no such tool is registered (#{name})"
  defp refusal_text({:clearance, reason}), do: "clearance denied (#{inspect(reason)})"

  # A malformed call is the one refusal the agent can actually fix on the next
  # turn, so it gets the specifics rather than an inspected tuple.
  defp refusal_text({:malformed_call, errors}) do
    "the arguments were wrong: " <> Enum.map_join(errors, "; ", &arg_error_text/1)
  end

  defp refusal_text(other), do: inspect(other)

  defp arg_error_text({:missing, key}), do: "#{key} is required and was not given"
  defp arg_error_text({:unknown, key}), do: "#{key} is not an argument this tool takes"
  defp arg_error_text({:type, key, expected}), do: "#{key} must be a #{expected}"

  defp audit_parse_refusal(ctx, reason) do
    Audit.record(:grant_violation, %{
      from_agent: Map.get(ctx, :agent_id, "system"),
      order_id: ctx[:order_id],
      world_id: ctx[:world_id],
      payload: %{reason: "proposal rejected before gate: #{inspect(reason)}"}
    })
  end

  defp audit_round_limit(ctx, proposal) do
    Audit.record(:grant_violation, %{
      from_agent: Map.get(ctx, :agent_id, "system"),
      order_id: ctx[:order_id],
      world_id: ctx[:world_id],
      payload: %{
        tool: proposal.tool,
        reason: "second proposal in one order refused (round_limit)"
      }
    })
  end
end
