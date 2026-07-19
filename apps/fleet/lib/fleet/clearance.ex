defmodule Fleet.Clearance do
  @moduledoc """
  The **read** gate — the peer of `Fleet.Authority`/`Fleet.Dispatcher.decide` for
  *seeing* information rather than *doing* things.

  Read-clearance cannot be a static grant term: the constraints are **relational**
  ("…of *that* ship", "…on *your* chain") and **live** (duty can flip mid-session),
  which a flat `MapSet` grant can't encode. So it is a second **pure** function the
  Dispatcher consults alongside the action-grant check. `can_read?/4` reads only the
  runtime-built `Fleet.Principal`, the `Fleet.InfoClass` taxonomy, and the target
  descriptors in `opts` — target descriptors only *narrow* the request; the
  principal (rank/duty/ship/chain) is built from process state, never the payload.

  Rules, first match wins, else default-deny:
  1. **self** — the subject always reads its own information.
  2. **duty** — a `:relieved` principal reads nothing (this is *why* clearance must
     run: `Fleet.Proposal`/`decide` never checks duty).
  3. **ship-commissioning** — `:ship`-scoped classes require the principal be
     commissioned to the target ship.
  4. **billet floor** — `Fleet.InfoClass.min_billet` vs the principal's `Fleet.Rank`
     seniority.
  5. **chain analog** (`:chain`-scoped classes) — Phase 3; denied for now.

  `:admiral` bypasses ship + chain (fleet-wide, top billet) but is still routed here.
  """

  alias Fleet.{Principal, InfoClass, Rank}

  @type reason ::
          :unknown_info_class
          | :relieved
          | :not_commissioned_to_ship
          | :insufficient_billet
          | :chain_read_unsupported
          | :off_chain

  @spec can_read?(Principal.t(), atom(), String.t() | nil, keyword()) :: :allow | {:deny, reason()}
  def can_read?(%Principal{} = principal, info_class, target_ship_id, opts \\ []) do
    cond do
      not InfoClass.known?(info_class) -> {:deny, :unknown_info_class}
      self_read?(principal, opts) -> :allow
      principal.kind == :admiral -> :allow
      true -> gate(principal, info_class, target_ship_id, opts)
    end
  end

  # Rule 1 — the subject reads its own mind/orders/records.
  defp self_read?(%Principal{kind: :ensign, id: id}, opts) when is_binary(id),
    do: Keyword.get(opts, :target_agent_id) == id

  defp self_read?(_, _), do: false

  defp gate(%Principal{} = p, info_class, target_ship_id, _opts) do
    cond do
      # Rule 2 — duty
      p.duty != :active ->
        {:deny, :relieved}

      # Rule 3 — ship-commissioning (ship-scoped classes)
      InfoClass.scope(info_class) == :ship and not commissioned_to?(p, target_ship_id) ->
        {:deny, :not_commissioned_to_ship}

      # Rule 4 — billet floor
      not Rank.at_least?(p.rank, InfoClass.min_billet(info_class)) ->
        {:deny, :insufficient_billet}

      # Rule 5 — the chain analog for comms/orders/audit is Phase 3.
      InfoClass.scope(info_class) == :chain ->
        {:deny, :chain_read_unsupported}

      true ->
        :allow
    end
  end

  defp commissioned_to?(%Principal{ship_id: sid}, target),
    do: is_binary(sid) and is_binary(target) and sid == target
end
