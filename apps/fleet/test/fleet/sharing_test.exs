defmodule Fleet.SharingTest do
  @moduledoc """
  Proves the ONLY way another agent's information enters a mind is by
  communication: what an agent is TOLD (an order's directive it receives, a
  report delivered to it) is recorded into its OWN mind-world with the sender's
  provenance — never by reading another agent's world.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.{CrewSupervisor, MindWorld}
  alias Brain.Memory.Store

  @world "default"

  setup do
    test_pid = self()
    handler_id = "sharing-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :officer, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  defp commission(grants) do
    sid = "sh-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "T", constitution: "Serve.", genome: %{}}
    {:ok, _pid, id} = CrewSupervisor.start_officer(soul_id: sid, soul: soul, grant: %{authorities: grants}, tick_interval: 50)
    assert_receive {:tele, :soul_hydrated, _, %{soul_id: ^sid}}, 5_000
    id
  end

  defp comms(world_id) do
    {:ok, eps} = Store.query_by_tags(["communication"], 50, world_id: world_id)
    eps
  end

  test "communicated content enters the recipient's mind with provenance; nothing else crosses" do
    co = commission([:cognition, {:world, @world}, :issue_orders])
    rep = commission([])
    other = commission([])
    :ok = Fleet.assign_co(rep, co)
    assert Fleet.status(rep).co == co

    Fleet.issue_order(co, rep, "Investigate the anomaly at sector 7.",
      authorities: [:cognition, {:world, @world}], world_id: @world)

    # The report ingests the directive it was ordered (attributed to its CO)...
    assert_receive {:tele, :ingested, _, %{kind: :order}}, 15_000
    # ...and when it reports, the CO ingests that report (attributed to the report).
    assert_receive {:tele, :ingested, _, %{kind: :report}}, 120_000

    rep_comms = comms(MindWorld.id(rep))
    co_comms = comms(MindWorld.id(co))
    other_comms = comms(MindWorld.id(other))

    # The report's mind holds the directive it was told, stamped from its CO.
    assert Enum.any?(rep_comms, &(&1.outcome =~ "sector 7"))
    assert Enum.any?(rep_comms, fn e -> "from:#{co}" in (e.tags || []) end)

    # The CO's mind holds the report, stamped from the report.
    assert Enum.any?(co_comms, fn e -> "from:#{rep}" in (e.tags || []) end)

    # An uninvolved agent was told nothing — its mind holds none of this.
    refute Enum.any?(other_comms, &(&1.outcome =~ "sector 7"))
  end
end
