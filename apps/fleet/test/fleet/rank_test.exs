defmodule Fleet.RankTest do
  @moduledoc "Unit tests for the billet → standing-authority map."
  use ExUnit.Case, async: true

  alias Fleet.Rank

  test "worker billets confer no standing command authority" do
    assert Rank.standing_authorities(:ensign) == []
    assert Rank.standing_authorities(:lieutenant) == []
  end

  test "command billets confer the authorities their office carries" do
    xo = Rank.standing_authorities(:executive_officer)
    assert :issue_orders in xo
    assert :relieve in xo
    assert :review_plans in xo
    assert :draft_court_martial in xo

    assert Rank.standing_authorities(:security) == [:veto, :flag_anomaly]

    cap = Rank.standing_authorities(:captain)
    assert :issue_orders in cap
    assert :relieve in cap
    assert :delegate in cap
  end

  test "an unknown billet confers nothing — authority never comes from an unrecognised rank" do
    assert Rank.standing_authorities(:stowaway) == []
    assert Rank.standing_authorities("stowaway") == []
    assert Rank.standing_authorities(nil) == []
  end

  test "standing_authorities accepts the string form of a known billet" do
    assert Rank.standing_authorities("executive_officer") == Rank.standing_authorities(:executive_officer)
  end

  test "to_key parses known billets and falls back to the default for anything else" do
    assert Rank.to_key("security") == :security
    assert Rank.to_key(:captain) == :captain
    assert Rank.to_key("nonsense") == Rank.default()
    assert Rank.default() == :ensign
  end

  test "label and describe are populated for every billet and safe on unknowns" do
    for {key, _meta} <- Rank.all() do
      assert is_binary(Rank.label(key)) and Rank.label(key) != ""
      assert is_binary(Rank.describe(key)) and Rank.describe(key) != ""
    end

    assert Rank.label(:mystery) == "Mystery"
    assert Rank.describe(:mystery) == ""
  end

  test "every authority a billet confers round-trips through the Authority codec" do
    for {_key, %{authorities: auths}} <- Rank.all(), a <- auths do
      assert Fleet.Authority.decode(Fleet.Authority.encode(a)) == a
    end
  end
end
