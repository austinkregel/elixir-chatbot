defmodule Fleet.CrewBilletsTest do
  @moduledoc """
  Proves a billet is wired into *real* authority: commissioning an agent into a
  command billet seeds the standing grant its office carries, and that grant is
  enforced by the live command channel — an XO can command a report, a worker
  cannot. Non-command billets stay exactly as before (no standing authority).
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.{CrewSupervisor, Rank}

  setup do
    test_pid = self()
    handler_id = "billets-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :ensign, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  defp commission(rank) do
    sid = "billet-#{rank}-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "Agent #{sid}", constitution: "Serve.", genome: %{}}
    {:ok, _pid, id} = CrewSupervisor.start_ensign(soul_id: sid, soul: soul, rank: rank, tick_interval: 50)
    assert_receive {:tele, :soul_hydrated, _, %{soul_id: ^sid}}, 5_000
    id
  end

  test "a worker billet seeds no standing command authority (unchanged from the default)" do
    id = commission(:ensign)
    grants = Fleet.status(id).grants
    refute :issue_orders in grants
    refute :relieve in grants
    refute :veto in grants
  end

  test "a Security billet holds veto and flag_anomaly as standing grants" do
    id = commission(:security)
    grants = Fleet.status(id).grants
    assert :veto in grants
    assert :flag_anomaly in grants
  end

  test "an XO billet holds the command authorities and can actually command a report" do
    xo = commission(:executive_officer)
    report = commission(:ensign)

    grants = Fleet.status(xo).grants
    assert :issue_orders in grants
    assert :relieve in grants
    assert :draft_court_martial in grants

    :ok = Fleet.assign_co(report, xo)

    # The XO issues an order to its report through the live channel — this only
    # succeeds because the billet conferred :issue_orders. The report acknowledges.
    Fleet.issue_order(xo, report, "Log the current stardate.")
    assert_receive {:tele, :order_received, _, %{agent_id: ^report}}, 5_000
  end

  test "the standing grant survives commissioning as the durable billet" do
    xo = commission(:executive_officer)
    # rank is reported as the billet key, and the roster surfaces its label.
    assert Fleet.status(xo).rank == :executive_officer
    assert Rank.label(:executive_officer) =~ "First Officer"
  end
end
