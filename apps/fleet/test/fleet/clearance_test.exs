defmodule Fleet.ClearanceTest do
  @moduledoc """
  Pure tests for the read-clearance gate — the read-side peer of the action-grant
  gate. No stack needed: `can_read?/4` is a pure function of a runtime-built
  principal, the info-class taxonomy, and target descriptors.
  """
  use ExUnit.Case, async: true

  alias Fleet.{Clearance, Principal, InfoClass, Rank, Ship}

  @ship "USS-TEST"
  defp ensign(attrs \\ []) do
    Principal.agent(
      Enum.into(attrs, %{agent_id: "ens-1", rank: :ensign, duty: :active, ship_id: @ship})
    )
  end

  # ── taxonomy + seniority ───────────────────────────────────────────────────

  test "InfoClass taxonomy: known classes, scope, and floors" do
    assert InfoClass.known?(:system_status)
    refute InfoClass.known?(:nonsense)
    assert InfoClass.scope(:system_status) == :ship
    assert InfoClass.min_billet(:system_status) == :ensign
    assert InfoClass.min_billet(:souls) == :executive_officer
    assert InfoClass.scope(:command_comms) == :chain
  end

  test "Rank seniority orders billets and admiral sits above all" do
    assert Rank.seniority(:admiral) > Rank.seniority(:captain)
    assert Rank.seniority(:executive_officer) > Rank.seniority(:ensign)
    assert Rank.at_least?(:executive_officer, :ensign)
    refute Rank.at_least?(:ensign, :executive_officer)
    assert Rank.seniority(:stowaway) == -1
  end

  # ── the gate ───────────────────────────────────────────────────────────────

  test "self-read is always allowed, even for a relieved agent" do
    p = ensign(duty: :relieved)
    assert Clearance.can_read?(p, :agent_mind, "OTHER-SHIP", target_agent_id: "ens-1") == :allow
  end

  test "the admiral reads any known class fleet-wide" do
    adm = Principal.admiral()
    assert Clearance.can_read?(adm, :system_status, "ANY-SHIP") == :allow
    assert Clearance.can_read?(adm, :souls, "ANY-SHIP") == :allow
    assert Clearance.can_read?(adm, :nonsense, "ANY-SHIP") == {:deny, :unknown_info_class}
  end

  test "hard example B: ship status needs active duty AND commission to THAT ship" do
    # active + commissioned to the target ship → allow
    assert Clearance.can_read?(ensign(), :system_status, @ship) == :allow
    # relieved → denied on duty
    assert Clearance.can_read?(ensign(duty: :relieved), :system_status, @ship) == {:deny, :relieved}
    # active but commissioned elsewhere → denied on ship
    assert Clearance.can_read?(ensign(), :system_status, "OTHER-SHIP") ==
             {:deny, :not_commissioned_to_ship}
  end

  test "billet floor gates a command-only class from an ensign" do
    # souls is ship-scoped (ensign is commissioned, so the ship gate passes) but
    # floors at :executive_officer → the ensign is denied on billet.
    assert Clearance.can_read?(ensign(), :souls, @ship) == {:deny, :insufficient_billet}
    # an XO on the ship clears it.
    xo = ensign(rank: :executive_officer)
    assert Clearance.can_read?(xo, :souls, @ship) == :allow
  end

  test "unknown info class is default-denied" do
    assert Clearance.can_read?(ensign(), :nonsense, @ship) == {:deny, :unknown_info_class}
  end

  test "chain-scoped classes are deferred (Phase 3) — denied for non-admirals for now" do
    # hard example A (Admiral<->Commander comms an ensign must not see) is satisfied
    # today by the blanket chain deny; Phase 3 adds the participant/superior analog.
    assert Clearance.can_read?(ensign(), :command_comms, @ship) == {:deny, :chain_read_unsupported}
  end

  test "Fleet.Ship.id is configured" do
    assert is_binary(Ship.id())
  end
end
