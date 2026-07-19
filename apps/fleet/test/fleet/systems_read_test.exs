defmodule Fleet.SystemsReadTest do
  @moduledoc """
  The systems registry produces a layered inventory of 200+ systems, and reading it
  through the crew's `systems.read` tool is gated twice — by the action grant AND by
  `Fleet.Clearance` (on-duty + commissioned to this ship) — and audited.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  import Ecto.Query
  alias Fleet.{CrewSupervisor, Authority, Ship}
  alias Atlas.Schemas.CommandRecord

  setup do
    test_pid = self()
    handler_id = "sysread-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :ensign, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  defp commission(grants) do
    sid = "sysread-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "Agent #{sid}", constitution: "Serve.", genome: %{}}
    {:ok, _pid, id} = CrewSupervisor.start_ensign(soul_id: sid, soul: soul, grant: %{authorities: grants}, tick_interval: 50)
    assert_receive {:tele, :soul_hydrated, _, %{soul_id: ^sid}}, 5_000
    id
  end

  defp block(json), do: "I need the ship's status first.\n\n```propose\n#{json}\n```\n"

  defp kinds_for(agent_id) do
    Atlas.Repo.all(from r in CommandRecord, where: r.from_agent == ^agent_id, select: r.kind)
  end

  test "the registry snapshots 200+ layered systems, ship-stamped" do
    snap = Fleet.Systems.snapshot()
    assert snap.ship_id == Ship.id()
    assert snap.counts.subsystems > 200
    assert length(snap.services) >= 3
    assert Enum.any?(snap.services, &(&1.name =~ "Postgres"))
    assert is_integer(snap.health.health_score)
    # every system carries this ship's id
    assert Enum.all?(snap.services, &(&1.ship_id == Ship.id()))
  end

  test "an on-duty, commissioned agent may read ship status → framed data + a `read` audit" do
    id = commission([Authority.tool("systems.read")])

    assert {:ok, %{data: framed}} =
             Fleet.propose(id, block(~s({"tool": "systems.read", "requirement": "to report readiness"})))

    assert framed =~ "<data source=\"systems.read\""
    assert framed =~ "health_score"

    kinds = kinds_for(id)
    assert "read" in kinds
    assert "tool_effect" in kinds
    refute "grant_violation" in kinds
  end

  test "a RELIEVED agent is denied by clearance even though it holds the grant" do
    id = commission([Authority.tool("systems.read")])
    Fleet.relieve(id)
    assert_receive {:tele, :relieved, _, _}, 5_000

    assert {:refused, {:clearance, :relieved}} =
             Fleet.propose(id, block(~s({"tool": "systems.read", "requirement": "to report readiness"})))

    assert "grant_violation" in kinds_for(id)
  end

  test "beliefs.read still works as a self-read under the new clearance gate" do
    id = commission([Authority.tool("beliefs.read")])

    assert {:ok, %{data: framed}} =
             Fleet.propose(id, block(~s({"tool": "beliefs.read", "requirement": "to recall what I know"})))

    assert framed =~ "<data source=\"beliefs.read\""
    assert "read" in kinds_for(id)
  end
end
