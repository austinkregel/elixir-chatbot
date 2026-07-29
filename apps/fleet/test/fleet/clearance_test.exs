defmodule Fleet.ClearanceTest do
  @moduledoc """
  Pure tests for the read-clearance gate — the read-side peer of the action-grant
  gate. No stack needed: `can_read?/4` is a pure function of a runtime-built
  principal, the info-class taxonomy, and target descriptors.
  """
  use ExUnit.Case, async: true

  alias Fleet.{Clearance, Principal, InfoClass, Rank, Ship}

  @ship "USS-TEST"
  defp officer(attrs \\ []) do
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
    p = officer(duty: :relieved)
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
    assert Clearance.can_read?(officer(), :system_status, @ship) == :allow
    # relieved → denied on duty
    assert Clearance.can_read?(officer(duty: :relieved), :system_status, @ship) == {:deny, :relieved}
    # active but commissioned elsewhere → denied on ship
    assert Clearance.can_read?(officer(), :system_status, "OTHER-SHIP") ==
             {:deny, :not_commissioned_to_ship}
  end

  test "billet floor gates a command-only class from an officer" do
    # souls is ship-scoped (officer is commissioned, so the ship gate passes) but
    # floors at :executive_officer → the officer is denied on billet.
    assert Clearance.can_read?(officer(), :souls, @ship) == {:deny, :insufficient_billet}
    # an XO on the ship clears it.
    xo = officer(rank: :executive_officer)
    assert Clearance.can_read?(xo, :souls, @ship) == :allow
  end

  test "unknown info class is default-denied" do
    assert Clearance.can_read?(officer(), :nonsense, @ship) == {:deny, :unknown_info_class}
  end

  test "hard example A: an officer cannot read Admiral<->Commander comms (off chain)" do
    # participants are the Admiral and a Commander; our officer is neither a
    # participant nor a superior of them → off chain.
    parts = [participants: [:admiral, {:officer, "cmdr-1"}]]
    assert Clearance.can_read?(officer(), :command_comms, @ship, parts) == {:deny, :off_chain}
  end

  test "chain analog: a participant, and a superior of every participant, may read" do
    # a CO reading a comm it participates in
    co = officer(agent_id: "co-1", rank: :executive_officer, reports: ["rep-1"])
    assert Clearance.can_read?(co, :command_comms, @ship, participants: [{:officer, "co-1"}, {:officer, "rep-1"}]) == :allow

    # a CO reading a comm between two of its direct reports (superior of all)
    co2 = officer(agent_id: "co-2", rank: :executive_officer, reports: ["r1", "r2"])
    assert Clearance.can_read?(co2, :command_comms, @ship, participants: [{:officer, "r1"}, {:officer, "r2"}]) == :allow

    # but not a comm involving someone off its chain
    assert Clearance.can_read?(co2, :command_comms, @ship, participants: [{:officer, "r1"}, {:officer, "stranger"}]) == {:deny, :off_chain}
  end

  test "a comm read with no named participants is denied (can't establish chain membership)" do
    assert Clearance.can_read?(officer(), :command_comms, @ship) == {:deny, :off_chain}
  end

  test "Fleet.Ship.id is configured" do
    assert is_binary(Ship.id())
  end
end
