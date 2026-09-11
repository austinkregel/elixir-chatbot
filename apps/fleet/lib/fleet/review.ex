defmodule Fleet.Review do
  @moduledoc """
  Independent review of an act before it happens: the Security veto, the First
  Officer's plan review, and the two-officer rule.

  Everything below the review layer answers "may this agent do this?" — grants,
  clearance, provenance. Review answers a different question: "should this be
  done at all, and does someone other than the person proposing it agree?" A
  gate the proposer can satisfy alone is not review.

  The three mechanisms are deliberately different questions, asked by different
  holders:

    * **Security veto** (`:veto`) — *lawfulness and scope*. May a tool call
      proceed given the General Orders? A veto is deterministic in effect: once
      cast it stands, and it is recorded with cause.
    * **Plan review** (`:review_plans`) — *judgment*. Is this the right move at
      all? Held by the First Officer, who is duty-bound to object when a plan is
      flawed — silence being a failure of duty, not politeness.
    * **Two-officer rule** — *irreversibility*. An act that cannot be undone
      needs independent sign-off from two distinct holders, enforced in code.

  ## Reviewers must be someone else

  Every function here refuses a reviewer who is the acting agent. A review panel
  of clones is one opinion counted twice — the whole value of the mechanism is
  that the reviewer did not author the thing under review. `two_officer/3`
  additionally requires the two sign-offs to come from *different* agents, which
  is the entire point of the rule.

  ## What this module is not

  It records and enforces decisions; it does not make them. Who reviews, and
  what they conclude, comes from the chain of command — a holder of the
  authority, reached over the command channel. This module is the code that
  makes their answer binding.
  """

  alias Fleet.{Audit, Authority}

  @type decision :: :approve | {:reject, String.t()}
  @type signoff :: %{agent_id: String.t(), authority: atom(), decision: decision()}

  @doc """
  Is `reviewer_grants` entitled to review, and is the reviewer someone other
  than the acting agent?

  Returns `:ok`, or `{:error, :not_authorized}` / `{:error, :self_review}`.
  """
  @spec eligible?(MapSet.t(), atom(), String.t(), String.t()) ::
          :ok | {:error, :not_authorized | :self_review}
  def eligible?(reviewer_grants, authority, reviewer_id, acting_id) do
    cond do
      reviewer_id == acting_id -> {:error, :self_review}
      not Authority.holds?(reviewer_grants, authority) -> {:error, :not_authorized}
      true -> :ok
    end
  end

  @doc """
  Record a Security veto of a proposed act.

  A veto is a refusal with a cause on the record. `{:error, reason}` when the
  vetoing officer does not hold `:veto`, or is the acting agent.
  """
  @spec veto(map(), String.t(), MapSet.t(), String.t()) ::
          {:vetoed, map()} | {:error, :not_authorized | :self_review}
  def veto(ctx, reviewer_id, reviewer_grants, cause) do
    acting = Map.get(ctx, :agent_id)

    with :ok <- eligible?(reviewer_grants, :veto, reviewer_id, acting) do
      record = %{
        by: reviewer_id,
        cause: cause,
        subject: Map.get(ctx, :subject),
        order_id: Map.get(ctx, :order_id)
      }

      Audit.record(:policy_veto, %{
        from_agent: reviewer_id,
        to_agent: acting,
        order_id: Map.get(ctx, :order_id),
        world_id: Map.get(ctx, :world_id),
        reason: cause,
        payload: record
      })

      {:vetoed, record}
    end
  end

  @doc """
  Record a First Officer's plan review.

  An approval lets the act proceed; a rejection stops it with a reason. Either
  way the review is on the record, because an unrecorded review cannot be
  audited and so is not a review.
  """
  @spec plan_review(map(), String.t(), MapSet.t(), decision()) ::
          {:approved, map()} | {:rejected, map()} | {:error, :not_authorized | :self_review}
  def plan_review(ctx, reviewer_id, reviewer_grants, decision) do
    acting = Map.get(ctx, :agent_id)

    with :ok <- eligible?(reviewer_grants, :review_plans, reviewer_id, acting) do
      record = %{
        by: reviewer_id,
        order_id: Map.get(ctx, :order_id),
        subject: Map.get(ctx, :subject),
        decision: decision_tag(decision),
        reason: decision_reason(decision)
      }

      Audit.record(:plan_review, %{
        from_agent: reviewer_id,
        to_agent: acting,
        order_id: Map.get(ctx, :order_id),
        world_id: Map.get(ctx, :world_id),
        verdict: decision_tag(decision),
        reason: decision_reason(decision),
        payload: record
      })

      case decision do
        :approve -> {:approved, record}
        {:reject, _} -> {:rejected, record}
      end
    end
  end

  @doc """
  Does this set of sign-offs satisfy the two-officer rule?

  Requires two approvals, from two *distinct* agents, neither of them the acting
  agent, each holding an authority that entitles them to sign. One officer
  signing twice is one officer.
  """
  @spec two_officer([signoff()], String.t(), [atom()]) ::
          :ok | {:error, :insufficient_signoffs | :not_distinct | :self_signoff | :rejected}
  def two_officer(signoffs, acting_id, accepted_authorities \\ [:veto, :review_plans]) do
    approvals = Enum.filter(signoffs, &(&1.decision == :approve))

    cond do
      Enum.any?(signoffs, &match?({:reject, _}, &1.decision)) ->
        {:error, :rejected}

      Enum.any?(approvals, &(&1.agent_id == acting_id)) ->
        {:error, :self_signoff}

      not Enum.all?(approvals, &(&1.authority in accepted_authorities)) ->
        {:error, :insufficient_signoffs}

      Enum.count_until(approvals, 2) < 2 ->
        {:error, :insufficient_signoffs}

      approvals |> Enum.map(& &1.agent_id) |> Enum.uniq() |> Enum.count_until(2) < 2 ->
        {:error, :not_distinct}

      true ->
        :ok
    end
  end

  @doc """
  Does an act at this risk class require review before it may proceed?

  `:routine` proceeds on acknowledgement; `:sensitive` wants a plan review;
  `:irreversible` wants two-officer sign-off.
  """
  @spec required_review(Fleet.Order.risk_class()) :: :none | :plan_review | :two_officer
  def required_review(:routine), do: :none
  def required_review(:sensitive), do: :plan_review
  def required_review(:irreversible), do: :two_officer

  defp decision_tag(:approve), do: "approve"
  defp decision_tag({:reject, _}), do: "reject"

  defp decision_reason(:approve), do: nil
  defp decision_reason({:reject, reason}), do: reason
end
