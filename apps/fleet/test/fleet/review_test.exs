defmodule Fleet.ReviewTest do
  @moduledoc """
  The review layer: Security veto, plan review, and the two-officer rule.

  These are the LCARS Phase-3 mechanisms — the ones that ask "should this be
  done, and does anyone else agree?" rather than "may this agent do it?".
  """
  use Fleet.FleetCase

  alias Fleet.{Authority, Review}

  @moduletag :integration

  defp grants(list), do: Authority.to_set(list)

  defp ctx(overrides \\ %{}) do
    Map.merge(
      %{agent_id: "ensign-acting", order_id: "ord-review-1", world_id: "mind:ensign-acting"},
      overrides
    )
  end

  describe "eligibility" do
    test "a holder of the authority who is not the acting agent may review" do
      assert Review.eligible?(grants([:veto]), :veto, "security-1", "ensign-acting") == :ok
    end

    test "an agent may not review itself, however it is credentialed" do
      assert Review.eligible?(grants([:veto]), :veto, "ensign-acting", "ensign-acting") ==
               {:error, :self_review}
    end

    test "a reviewer without the authority is refused" do
      assert Review.eligible?(grants([:issue_orders]), :veto, "xo-1", "ensign-acting") ==
               {:error, :not_authorized}
    end
  end

  describe "veto" do
    test "a Security officer's veto is recorded with its cause" do
      assert {:vetoed, record} =
               Review.veto(ctx(), "security-1", grants([:veto]), "would touch the reactor")

      assert record.by == "security-1"
      assert record.cause == "would touch the reactor"
      assert record.order_id == "ord-review-1"
    end

    test "an unauthorized or self veto records nothing" do
      assert {:error, :not_authorized} = Review.veto(ctx(), "ensign-b", grants([]), "no")

      assert {:error, :self_review} =
               Review.veto(ctx(), "ensign-acting", grants([:veto]), "vetoing myself")
    end
  end

  describe "plan review" do
    test "an approval and a rejection are both on the record" do
      assert {:approved, approved} =
               Review.plan_review(ctx(), "xo-1", grants([:review_plans]), :approve)

      assert approved.decision == "approve"

      assert {:rejected, rejected} =
               Review.plan_review(ctx(), "xo-1", grants([:review_plans]), {:reject, "anchoring"})

      assert rejected.decision == "reject"
      assert rejected.reason == "anchoring"
    end
  end

  describe "two-officer rule" do
    defp approval(id, authority \\ :review_plans),
      do: %{agent_id: id, authority: authority, decision: :approve}

    test "two distinct authorized officers satisfy it" do
      assert Review.two_officer([approval("xo-1"), approval("security-1", :veto)], "ensign-acting") ==
               :ok
    end

    test "one officer signing twice is still one officer" do
      assert Review.two_officer([approval("xo-1"), approval("xo-1")], "ensign-acting") ==
               {:error, :not_distinct}
    end

    test "a single sign-off is not enough" do
      assert Review.two_officer([approval("xo-1")], "ensign-acting") ==
               {:error, :insufficient_signoffs}
    end

    test "the acting agent cannot sign off on its own irreversible act" do
      assert Review.two_officer([approval("ensign-acting"), approval("xo-1")], "ensign-acting") ==
               {:error, :self_signoff}
    end

    test "any rejection stops the act regardless of approvals" do
      signoffs = [
        approval("xo-1"),
        approval("security-1", :veto),
        %{agent_id: "security-2", authority: :veto, decision: {:reject, "unsafe"}}
      ]

      assert Review.two_officer(signoffs, "ensign-acting") == {:error, :rejected}
    end

    test "a sign-off from someone without a reviewing authority does not count" do
      signoffs = [approval("xo-1"), %{agent_id: "ensign-b", authority: :cognition, decision: :approve}]

      assert Review.two_officer(signoffs, "ensign-acting") == {:error, :insufficient_signoffs}
    end

    test "no sign-offs at all is the default, and it refuses" do
      assert Review.two_officer([], "ensign-acting") == {:error, :insufficient_signoffs}
    end
  end

  describe "required review by risk class" do
    test "process scales with how hard the act is to undo" do
      assert Review.required_review(:routine) == :none
      assert Review.required_review(:sensitive) == :plan_review
      assert Review.required_review(:irreversible) == :two_officer
    end
  end
end
