defmodule Fleet.InfoClass do
  @moduledoc """
  The closed taxonomy of **readable** information classes — the read-side peer of
  `Fleet.Authority`'s action vocabulary.

  Each class declares:
    * `:scope` — `:ship` (gated by ship-commissioning), `:chain` (gated by chain
      position — the read-side analog of `Fleet.Comms.from_co?/from_report?`), or
      `:self` (only the subject).
    * `:min_billet` — the minimum `Fleet.Rank` billet that may read it at all.
    * `:sensitivity` — advisory label for the UI/audit.

  `Fleet.Clearance` consults this. The model can only *propose* a read of a **known**
  class (default-deny on anything unknown); it can never mint a class. Phase 1
  implements `:system_status`; the rest are reserved so the chain analog and the
  remaining classes land without re-architecting.
  """

  @classes %{
    system_status: %{scope: :ship, min_billet: :ensign, sensitivity: :low},
    orders_assignments: %{scope: :chain, min_billet: :ensign, sensitivity: :moderate},
    command_comms: %{scope: :chain, min_billet: :ensign, sensitivity: :high},
    trust_ledger: %{scope: :ship, min_billet: :executive_officer, sensitivity: :high},
    souls: %{scope: :ship, min_billet: :executive_officer, sensitivity: :moderate},
    agent_mind: %{scope: :ship, min_billet: :executive_officer, sensitivity: :critical},
    audit_blackbox: %{scope: :chain, min_billet: :executive_officer, sensitivity: :critical},

    # Working material rather than command information. These gate what an
    # officer may *study*, and the floor is `:ensign` on purpose — an officer
    # commissioned to do research or engineering that cannot read the corpus it
    # was commissioned for is a compliance exercise, not a crew. The narrower
    # control is the tool grant and, for anything leaving the ship, egress.
    corpus: %{scope: :ship, min_billet: :ensign, sensitivity: :moderate},
    world_knowledge: %{scope: :ship, min_billet: :ensign, sensitivity: :low},
    external_research: %{scope: :ship, min_billet: :ensign, sensitivity: :moderate},
    home_sensors: %{scope: :ship, min_billet: :ensign, sensitivity: :moderate}
  }

  @type t :: atom()

  @doc "Is `class` a known information class?"
  def known?(class), do: Map.has_key?(@classes, class)

  @doc "All known information classes."
  def all, do: Map.keys(@classes)

  @doc "The scope of a class (`:ship` | `:chain` | `:self`), or nil if unknown."
  def scope(class), do: get_in(@classes, [class, :scope])

  @doc "The minimum billet that may read this class, or nil if unknown."
  def min_billet(class), do: get_in(@classes, [class, :min_billet])

  @doc "The full spec map for a class, or nil."
  def spec(class), do: Map.get(@classes, class)
end
