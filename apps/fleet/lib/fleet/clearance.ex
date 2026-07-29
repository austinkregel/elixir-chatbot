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
  5. **chain analog** (`:chain`-scoped classes — comms/orders/audit) — you may read
     only if you are a **participant** or a **superior of every participant** (the
     read-side peer of `Fleet.Comms.from_co?/from_report?`).

  `:admiral` bypasses ship + chain (fleet-wide, top billet) but is still routed here.
  """

  alias Fleet.{Principal, InfoClass, Rank}

  @type reason ::
          :unknown_info_class
          | :relieved
          | :not_commissioned_to_ship
          | :insufficient_billet
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
  defp self_read?(%Principal{kind: :officer, id: id}, opts) when is_binary(id),
    do: Keyword.get(opts, :target_agent_id) == id

  defp self_read?(_, _), do: false

  defp gate(%Principal{} = p, info_class, target_ship_id, opts) do
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

      # Rule 5 — the chain analog for comms/orders/audit: you may read a chain-scoped
      # item only if you are a PARTICIPANT, or a chain-superior of every participant.
      InfoClass.scope(info_class) == :chain ->
        chain_read(p, opts)

      true ->
        :allow
    end
  end

  defp commissioned_to?(%Principal{ship_id: sid}, target),
    do: is_binary(sid) and is_binary(target) and sid == target

  # The read-side peer of `Fleet.Comms.from_co?/from_report?`. `opts[:participants]`
  # names the item's principals (`:admiral` | `{:officer, id}` | id); it only NARROWS
  # the request — clearance denies if the reader isn't a participant or a superior of
  # every participant. (Phase 3 checks DIRECT reports; a transitive-subtree check for
  # deeper chains is an additive extension when the caller supplies the subtree.)
  defp chain_read(%Principal{} = p, opts) do
    participants =
      opts
      |> Keyword.get(:participants, [])
      |> Enum.map(&participant_id/1)
      |> Enum.reject(&is_nil/1)

    cond do
      participants == [] -> {:deny, :off_chain}
      p.id in participants -> :allow
      Enum.all?(participants, &(&1 in p.reports)) -> :allow
      true -> {:deny, :off_chain}
    end
  end

  defp participant_id(:admiral), do: "admiral"
  defp participant_id({:officer, id}) when is_binary(id), do: id
  defp participant_id(id) when is_binary(id), do: id
  defp participant_id(_), do: nil
end
