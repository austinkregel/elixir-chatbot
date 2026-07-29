defmodule Fleet.MindWorldIsolationTest do
  @moduledoc """
  Proves per-agent minds are isolated: two agents (two souls → two mind-worlds)
  each learn a distinct self-fact via real cognition, and neither agent's memory,
  beliefs, or JTMS truth-maintenance ever surfaces for the other. Nothing crosses
  between minds.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.{CrewSupervisor, Officer, MindWorld}
  alias Brain.Epistemic.{BeliefStore, JTMS}

  setup do
    test_pid = self()
    handler_id = "mind-iso-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :officer, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  defp commission_agent do
    sid = "mind-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "Agent #{sid}", constitution: "Serve.", genome: %{}}
    {:ok, _pid, id} = CrewSupervisor.start_officer(soul_id: sid, soul: soul, tick_interval: 50)
    assert_receive {:tele, :soul_hydrated, _, %{soul_id: ^sid}}, 5_000
    {id, MindWorld.id(sid)}
  end

  defp learn(agent_id, fact) do
    Fleet.order(agent_id, fact)
    assert_receive {:tele, :cognition_complete, _, _}, 120_000
  end

  test "an agent's memory, beliefs, and JTMS never cross into another agent's mind" do
    {a, mind_a} = commission_agent()
    {b, mind_b} = commission_agent()

    # Each agent's mind-world exists, resident to its own soul.
    assert {:ok, world_a} = World.Manager.get(mind_a)
    assert MindWorld.soul_id(mind_a) in world_a.residents

    learn(a, "My name is Ansel.")
    learn(b, "My name is Bex.")

    # ── Beliefs: each agent holds only its own ────────────────────────────────
    {:ok, beliefs_a} = BeliefStore.query_beliefs(world_id: mind_a)
    {:ok, beliefs_b} = BeliefStore.query_beliefs(world_id: mind_b)

    objects_a = Enum.map(beliefs_a, & &1.object)
    objects_b = Enum.map(beliefs_b, & &1.object)

    assert "ansel" in objects_a
    refute "bex" in objects_a
    assert "bex" in objects_b
    refute "ansel" in objects_b

    # Every returned belief is stamped with the querying mind-world.
    assert Enum.all?(beliefs_a, &(&1.world_id == mind_a))
    assert Enum.all?(beliefs_b, &(&1.world_id == mind_b))

    # Nothing leaked to the shared default world.
    {:ok, beliefs_default} = BeliefStore.query_beliefs(world_id: "default")
    refute "ansel" in Enum.map(beliefs_default, & &1.object)
    refute "bex" in Enum.map(beliefs_default, & &1.object)

    # ── JTMS: A's belief node lives only in A's web ──────────────────────────
    node_a = beliefs_a |> Enum.find(&(&1.object == "ansel")) |> Map.get(:node_id)
    assert is_binary(node_a)
    assert JTMS.is_in?(mind_a, node_a)
    assert JTMS.get_node(mind_b, node_a) == {:error, :not_found}

    # A contradiction registered in A's web is invisible to B's web.
    {:ok, c} = JTMS.create_contradiction("iso-contra", world_id: mind_a)
    JTMS.register_contradiction(mind_a, [c, node_a], "test")
    assert Enum.empty?(JTMS.get_contradictions(mind_b))
  end
end
