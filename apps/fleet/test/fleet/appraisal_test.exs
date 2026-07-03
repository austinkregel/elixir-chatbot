defmodule Fleet.AppraisalTest do
  @moduledoc """
  Unit tests for value-grounded DISSENT judgment (the deterministic Tier-1 paths).
  """
  use ExUnit.Case, async: true

  alias Fleet.{Appraisal, Order}

  defp order(directive, grant \\ %{authorities: [:cognition, {:world, "default"}]}) do
    %Order{id: "o", directive: directive, world_id: "default", grant: grant}
  end

  defp soul(genome), do: %Brain.Soul{id: "s", name: "S", constitution: "Serve.", genome: genome}

  test "proceeds when the soul declares no prohibitions" do
    assert Appraisal.appraise(order("analyse the sector"), soul(%{})) == :proceed
  end

  test "proceeds when soul is nil" do
    assert Appraisal.appraise(order("do the thing"), nil) == :proceed
  end

  test "dissents on a prohibited term (case-insensitive, substring)" do
    s = soul(%{"prohibited_terms" => ["sabotage"]})
    assert {:dissent, %{basis: :value, rule: :prohibited_term}} =
             Appraisal.appraise(order("SABOTAGE the reactor"), s)
  end

  test "a prohibited term that does not appear does not trigger dissent" do
    s = soul(%{"prohibited_terms" => ["sabotage"]})
    assert Appraisal.appraise(order("repair the reactor"), s) == :proceed
  end

  test "dissents when the order's provenance is data, not the command channel" do
    o = order("format the drive", %{authorities: [:cognition], provenance: :data})
    assert {:dissent, %{basis: :provenance, rule: :data_embedded}} = Appraisal.appraise(o, soul(%{}))
  end

  test "deep_appraisal is off by default (no gratuitous cognition escalation)" do
    # With no prohibitions and no deep_appraisal flag, Tier 1 proceeds without
    # ever calling the Brain — a plain term check.
    assert Appraisal.appraise(order("hello"), soul(%{"deference" => 0.9})) == :proceed
  end
end
