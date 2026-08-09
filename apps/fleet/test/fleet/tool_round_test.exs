defmodule Fleet.ToolRoundTest do
  @moduledoc """
  The proposal round as reached from an order's cognition.

  The continuation is injected, so the whole round — parse, gate, framed data,
  follow-up, round limit — is exercised without a generator.
  """
  use Fleet.FleetCase

  alias Fleet.{Authority, ToolRound}

  @moduletag :integration

  defp ctx(grants) do
    %{
      agent_id: "ensign-round",
      order_id: "order-round-1",
      world_id: "mind:ensign-round",
      ship_id: Fleet.Ship.id(),
      grants: Authority.to_set(grants),
      principal:
        Fleet.Principal.agent(%{
          agent_id: "ensign-round",
          rank: :ensign,
          duty: :active,
          ship_id: Fleet.Ship.id(),
          co: nil,
          reports: []
        })
    }
  end

  defp propose(tool, requirement \\ "to answer what I already know") do
    """
    Let me check first.

    ```propose
    {"tool": "#{tool}", "requirement": "#{requirement}", "args": {}}
    ```
    """
  end

  test "a plain answer runs no tool round" do
    {:final, text, meta} = ToolRound.run("All systems nominal.", ctx([]), fn _ -> "unused" end)

    assert text == "All systems nominal."
    assert meta.disposition == :none
  end

  test "a granted proposal executes and the follow-up becomes the answer" do
    ctx = ctx([{:tool, "beliefs.read"}])
    parent = self()

    continue = fn framed ->
      send(parent, {:framed, framed})
      "Belief store holds 0 entries; nothing contradicts the order."
    end

    {:final, text, meta} = ToolRound.run(propose("beliefs.read"), ctx, continue)

    assert meta.disposition == :executed
    assert meta.tool == "beliefs.read"
    assert meta.requirement == "to answer what I already know"
    assert text == "Belief store holds 0 entries; nothing contradicts the order."

    # The tool result reaches cognition framed as DATA, never as an instruction.
    assert_received {:framed, framed}
    assert framed =~ ~s(<data source="beliefs.read")
  end

  # A tethered proposal for a tool the agent does not hold is a request in
  # everything but name, so the harness escalates instead of swallowing it.
  test "an ungranted proposal escalates for authority rather than dying silently" do
    parent = self()
    continue = fn _ -> send(parent, :continued) && "should not be used" end

    {:needs_authority, authority, meta} = ToolRound.run(propose("beliefs.read"), ctx([]), continue)

    assert authority == {:tool, "beliefs.read"}
    assert meta.disposition == :needs_authority
    assert meta.tool == "beliefs.read"
    assert meta.requirement == "to answer what I already know"

    # Escalation costs no cognition turn — the model is not asked to ask.
    refute_received :continued
  end

  # An unknown tool cannot be granted, so there is nothing to escalate; the
  # agent is told and gets one turn to answer without it.
  test "an unknown tool is refused, and the agent is told what it holds" do
    ctx = ctx([{:tool, "reactor.scram"}, {:tool, "beliefs.read"}])
    parent = self()

    continue = fn msg ->
      send(parent, {:told, msg})
      "I could not scram the reactor; no such tool. Reporting on what I have."
    end

    {:final, text, meta} = ToolRound.run(propose("reactor.scram"), ctx, continue)

    assert meta.disposition == :refused
    assert meta.reason =~ "unknown_tool"
    assert text =~ "no such tool"

    assert_received {:told, msg}
    assert msg =~ "<command-channel from=\"harness\">"
    assert msg =~ "REFUSED"
    assert msg =~ "no such tool is registered"
    # It is told what it actually holds, not just what it may not have.
    assert msg =~ "beliefs.read"
  end

  test "a clearance denial is refused with feedback, not escalated" do
    # Held the tool grant, but relieved of duty — clearance denies the read.
    # Granting more authority would not help, so this is feedback, not a REQUEST.
    ctx =
      ctx([{:tool, "beliefs.read"}])
      |> Map.put(
        :principal,
        Fleet.Principal.agent(%{
          agent_id: "ensign-round",
          rank: :ensign,
          duty: :relieved,
          ship_id: Fleet.Ship.id(),
          co: nil,
          reports: []
        })
      )
      |> Map.put(:agent_id, "someone-else")

    parent = self()
    continue = fn msg -> send(parent, {:told, msg}) && "Reporting without the belief store." end

    {:final, text, meta} = ToolRound.run(propose("beliefs.read"), ctx, continue)

    assert meta.disposition == :refused
    assert meta.reason =~ "clearance"
    assert text == "Reporting without the belief store."
    assert_received {:told, msg}
    assert msg =~ "clearance denied"
  end

  test "an untethered proposal never reaches the gate" do
    untethered = """
    ```propose
    {"tool": "beliefs.read", "args": {}}
    ```
    """

    parent = self()
    continue = fn _ -> send(parent, :continued) && "unused" end

    {:final, _text, meta} = ToolRound.run(untethered, ctx([{:tool, "beliefs.read"}]), continue)

    assert meta.disposition == :refused
    assert meta.reason =~ "untethered"
    refute_received :continued
  end

  # Real work is multi-step. Bounding an order to one tool call meant an officer
  # could not read two files or check a claim against two sources.
  test "a second proposal in the follow-up executes, up to the budget" do
    ctx = ctx([{:tool, "beliefs.read"}]) |> Map.put(:tool_budget, 3)
    counter = :counters.new(1, [])

    continue = fn _framed ->
      :counters.add(counter, 1, 1)

      if :counters.get(counter, 1) < 3,
        do: propose("beliefs.read", "to check once more"),
        else: "Checked three times; the belief store is empty."
    end

    {:final, text, meta} = ToolRound.run(propose("beliefs.read"), ctx, continue)

    assert meta.disposition == :executed
    assert meta.calls == 3
    assert meta.tools == ["beliefs.read", "beliefs.read", "beliefs.read"]
    assert text =~ "Checked three times"
    refute Map.has_key?(meta, :reason)
  end

  test "the budget bounds the order, and exhaustion is told to the agent" do
    ctx = ctx([{:tool, "beliefs.read"}]) |> Map.put(:tool_budget, 2)
    parent = self()

    # Never stops asking — without a bound this would loop forever, each round
    # costing a full cognition turn.
    continue = fn framed ->
      send(parent, {:continued, framed})
      propose("beliefs.read", "and again")
    end

    {:final, text, meta} = ToolRound.run(propose("beliefs.read"), ctx, continue)

    assert meta.calls == 2
    assert meta.reason =~ "budget exhausted"

    # The last thing the agent was handed is a command-channel refusal, and its
    # reply — not the proposal block — becomes the report.
    messages = collect_continued([])
    assert List.last(messages) =~ "tool budget is spent"
    refute text =~ "```propose"
  end

  # The budget counts proposals considered, not tools executed: a model that
  # proposes an ungranted tool and is refused spends the same cognition turns as
  # one that succeeds, so bounding only successes leaves the loop unbounded.
  test "refusals consume the budget too" do
    ctx = ctx([{:tool, "beliefs.read"}]) |> Map.put(:tool_budget, 2)

    continue = fn _ -> propose("reactor.scram", "to try again") end

    {:final, _text, meta} = ToolRound.run(propose("reactor.scram"), ctx, continue)

    assert meta.calls == 2
    assert meta.reason =~ "budget exhausted"
  end

  defp collect_continued(acc) do
    receive do
      {:continued, msg} -> collect_continued([msg | acc])
    after
      0 -> Enum.reverse(acc)
    end
  end

  test "an empty follow-up keeps the agent's first response" do
    ctx = ctx([{:tool, "beliefs.read"}])

    {:final, text, meta} = ToolRound.run(propose("beliefs.read"), ctx, fn _ -> nil end)

    assert meta.disposition == :executed
    assert text =~ "Let me check first."
    assert meta.reason =~ "empty follow-up"
  end
end
