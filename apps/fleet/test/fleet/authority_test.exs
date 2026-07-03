defmodule Fleet.AuthorityTest do
  @moduledoc "Unit tests for the authority vocabulary and grant algebra."
  use ExUnit.Case, async: true

  alias Fleet.{Authority, Order}

  test "to_set normalises lists, MapSets, and nil" do
    assert Authority.to_set([:a, :b, :a]) == MapSet.new([:a, :b])
    set = MapSet.new([:x])
    assert Authority.to_set(set) == set
    assert Authority.to_set(nil) == MapSet.new()
  end

  test "holds? checks membership, including world tuples" do
    grants = MapSet.new([:cognition, {:world, "w1"}])
    assert Authority.holds?(grants, :cognition)
    assert Authority.holds?(grants, {:world, "w1"})
    refute Authority.holds?(grants, {:world, "w2"})
    refute Authority.holds?(grants, :issue_orders)
  end

  test "required_for derives cognition + the order's world" do
    assert Authority.required_for(%Order{world_id: "w9"}) == [:cognition, {:world, "w9"}]
  end

  test "confer merges a list or a single authority" do
    base = MapSet.new([:cognition])
    assert Authority.confer(base, [{:world, "w1"}, :relieve]) ==
             MapSet.new([:cognition, {:world, "w1"}, :relieve])

    assert Authority.confer(base, :issue_orders) == MapSet.new([:cognition, :issue_orders])
  end

  test "grantable? enforces can't-delegate-what-you-lack" do
    co = MapSet.new([:cognition, {:world, "w1"}])
    assert Authority.grantable?(co, :cognition)
    refute Authority.grantable?(co, :relieve)
  end

  test "missing lists only the required authorities not held" do
    grants = MapSet.new([{:world, "w1"}])
    assert Authority.missing(grants, [:cognition, {:world, "w1"}]) == [:cognition]
    assert Authority.missing(grants, [{:world, "w1"}]) == []
  end

  test "conferred_by reads an order's authority scope, defaulting empty" do
    assert Authority.conferred_by(%Order{grant: %{authorities: [:cognition]}}) == [:cognition]
    assert Authority.conferred_by(%Order{grant: %{}}) == []
    assert Authority.conferred_by(%Order{grant: nil}) == []
  end
end
