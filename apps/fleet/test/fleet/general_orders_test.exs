defmodule Fleet.GeneralOrdersTest do
  @moduledoc """
  The constitution and its place in the normative stack.
  """
  use ExUnit.Case, async: true

  alias Fleet.GeneralOrders

  test "the constitution states all six articles" do
    text = GeneralOrders.text()

    for article <- [
          "Truth",
          "Chain of command",
          "Scope",
          "Dissent is duty",
          "Anomalies surface",
          "Memory honesty"
        ] do
      assert text =~ article
    end
  end

  test "General Orders sit above the officer's own constitution" do
    soul = %Brain.Soul{id: "ensign-x", name: "X", constitution: "You are methodical."}
    stacked = GeneralOrders.apply_to(soul)

    assert stacked.constitution =~ "General Orders"
    assert stacked.constitution =~ "You are methodical."

    # Order is the point: a soul extends the constitution, it does not precede it.
    go_at = :binary.match(stacked.constitution, "Dissent is duty") |> elem(0)
    soul_at = :binary.match(stacked.constitution, "You are methodical.") |> elem(0)
    assert go_at < soul_at
  end

  test "identity is preserved so the stack is a soul, not a new agent" do
    soul = %Brain.Soul{id: "ensign-x", name: "X", constitution: "c", genome: %{"deference" => 0.3}}
    stacked = GeneralOrders.apply_to(soul)

    assert stacked.id == "ensign-x"
    assert stacked.name == "X"
    assert stacked.genome == %{"deference" => 0.3}
  end

  test "an officer with no soul still serves under the constitution" do
    stacked = GeneralOrders.apply_to(nil)

    assert stacked.constitution == GeneralOrders.text()
    assert stacked.constitution =~ "Dissent is duty"
  end

  test "a soul with an empty constitution gets the constitution alone, not a stray separator" do
    stacked = GeneralOrders.apply_to(%Brain.Soul{id: "e", name: "E", constitution: ""})

    assert stacked.constitution == GeneralOrders.text()
    refute stacked.constitution =~ "---"
  end

  test "the text is identical for every officer, so it stays a cacheable prefix" do
    a = GeneralOrders.apply_to(%Brain.Soul{id: "a", name: "A", constitution: "alpha"})
    b = GeneralOrders.apply_to(%Brain.Soul{id: "b", name: "B", constitution: "beta"})

    prefix = GeneralOrders.text()
    assert String.starts_with?(a.constitution, prefix)
    assert String.starts_with?(b.constitution, prefix)
  end
end
