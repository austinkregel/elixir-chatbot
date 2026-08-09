defmodule Fleet.ReviewGateTest do
  @moduledoc """
  Review as enforced at the tool gate, not merely recorded: a veto stops a call
  the grant would otherwise allow, and an irreversible effect needs two
  distinct officers behind it.
  """
  use Fleet.FleetCase

  alias Fleet.{Authority, Dispatcher, Proposal}

  @moduletag :integration

  defp base_ctx(extra \\ %{}) do
    Map.merge(
      %{
        agent_id: "ensign-gate",
        order_id: "ord-gate-1",
        world_id: "mind:ensign-gate",
        ship_id: Fleet.Ship.id(),
        grants: Authority.to_set([{:tool, "beliefs.read"}]),
        principal:
          Fleet.Principal.agent(%{
            agent_id: "ensign-gate",
            rank: :ensign,
            duty: :active,
            ship_id: Fleet.Ship.id(),
            co: nil,
            reports: []
          })
      },
      extra
    )
  end

  defp proposal do
    {:ok, p} =
      Proposal.parse("""
      ```propose
      {"tool": "beliefs.read", "requirement": "to answer what I already know", "args": {}}
      ```
      """)

    p
  end

  test "with no veto the granted call proceeds" do
    assert {:ok, %{data: framed}} = Dispatcher.dispatch(proposal(), base_ctx())
    assert framed =~ ~s(<data source="beliefs.read")
  end

  test "a veto stops a call the grant would otherwise allow" do
    ctx = base_ctx(%{vetoes: [%{by: "security-1", cause: "reactor risk", subject: nil}]})

    assert {:refused, {:vetoed, "reactor risk"}} = Dispatcher.dispatch(proposal(), ctx)
  end

  test "a veto naming a different tool does not stop this one" do
    ctx = base_ctx(%{vetoes: [%{by: "security-1", cause: "no", subject: "systems.read"}]})

    assert {:ok, _} = Dispatcher.dispatch(proposal(), ctx)
  end

  test "a veto naming this tool stops it" do
    ctx = base_ctx(%{vetoes: [%{by: "security-1", cause: "leaks the mind", subject: "beliefs.read"}]})

    assert {:refused, {:vetoed, "leaks the mind"}} = Dispatcher.dispatch(proposal(), ctx)
  end

  describe "two-officer rule at the gate" do
    # beliefs.read is a :read tool, so the two-officer branch is exercised by
    # asserting the rule itself against the gate's own contract: an irreversible
    # effect with no sign-offs must refuse. We assert via Review since the
    # registry ships no irreversible tool — deliberately, since nothing in the
    # fleet should be able to take an unrecoverable action yet.
    test "the registry ships no irreversible tool" do
      refute Enum.any?(Fleet.Tool.registry(), fn {_name, tool} -> tool.effect == :irreversible end)
    end

    test "an irreversible effect without two distinct sign-offs is refused" do
      assert {:error, :insufficient_signoffs} =
               Fleet.Review.two_officer([], "ensign-gate")

      assert {:error, :not_distinct} =
               Fleet.Review.two_officer(
                 [
                   %{agent_id: "xo-1", authority: :review_plans, decision: :approve},
                   %{agent_id: "xo-1", authority: :review_plans, decision: :approve}
                 ],
                 "ensign-gate"
               )

      assert :ok =
               Fleet.Review.two_officer(
                 [
                   %{agent_id: "xo-1", authority: :review_plans, decision: :approve},
                   %{agent_id: "security-1", authority: :veto, decision: :approve}
                 ],
                 "ensign-gate"
               )
    end
  end
end
