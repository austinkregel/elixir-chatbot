defmodule Fleet.OrderTest do
  @moduledoc """
  The ORDER as a structured object: what the issuer states, and what the
  lifecycle keeps in step.
  """
  use ExUnit.Case, async: true

  alias Fleet.Order

  describe "new/1" do
    test "objective and directive stay in step whichever name is given" do
      from_objective = Order.new(objective: "survey the sector")
      from_directive = Order.new(directive: "survey the sector")

      assert from_objective.objective == "survey the sector"
      assert from_objective.directive == "survey the sector"
      assert from_directive.objective == "survey the sector"
      assert from_directive.directive == "survey the sector"
    end

    test "carries stated constraints and context refs" do
      order =
        Order.new(
          objective: "summarize the last three CI runs",
          constraints: ["read-only: do not modify any file", "stay within the repo"],
          context_refs: ["briefing/ci-background.md"]
        )

      assert length(order.constraints) == 2
      assert order.context_refs == ["briefing/ci-background.md"]
    end

    test "defaults to the routine risk class" do
      assert Order.new(objective: "x").risk_class == :routine
      assert Order.risk_class(Order.new(objective: "x")) == :routine
    end

    test "accepts each valid risk class, and a rehydrated string form" do
      for rc <- Order.valid_risk_classes() do
        assert Order.new(objective: "x", risk_class: rc).risk_class == rc
        assert Order.risk_class(Order.new(objective: "x", risk_class: to_string(rc))) == rc
      end
    end

    test "an unknown risk class fails loudly rather than defaulting" do
      assert_raise ArgumentError, ~r/invalid risk_class/, fn ->
        Order.new(objective: "x", risk_class: :catastrophic)
      end
    end

    test "a single constraint is normalized to a list" do
      assert Order.new(objective: "x", constraints: "read-only").constraints == ["read-only"]
      assert Order.new(objective: "x", constraints: nil).constraints == []
    end
  end

  describe "update_status/2" do
    test "rejects a status outside the valid set" do
      order = Order.new(objective: "x")

      assert Order.update_status(order, "blocked").status == "blocked"

      assert_raise ArgumentError, ~r/invalid order status/, fn ->
        Order.update_status(order, "vibing")
      end
    end
  end
end
