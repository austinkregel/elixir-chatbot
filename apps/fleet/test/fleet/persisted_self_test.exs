defmodule Fleet.PersistedSelfTest do
  @moduledoc """
  Proves an agent's durable self survives a crash "without forgetting who it is
  or what it was doing": a service record + summary + duty log persist, and on
  restart the officer rehydrates its identity, standing grants, last assignment,
  and career counters.
  """
  use Fleet.FleetCase, async: false

  @moduletag :integration

  alias Fleet.{CrewSupervisor, Officer, Service, DutyLog}

  setup do
    test_pid = self()
    handler_id = "persisted-#{System.unique_integer([:positive])}"

    :telemetry.attach(handler_id, [:chat_bot, :officer, :event],
      fn _e, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end, test_pid)

    on_exit(fn ->
      :telemetry.detach(handler_id)
      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid), do: CrewSupervisor.retire(pid)
    end)

    :ok
  end

  test "an agent rehydrates its durable self after a crash" do
    sid = "self-#{System.unique_integer([:positive])}"
    soul = %Brain.Soul{id: sid, name: "Self #{sid}", constitution: "Serve.",
                       genome: %{"prohibited_terms" => ["sabotage"]}}
    grants = [:cognition, {:world, "default"}, :issue_orders]

    {:ok, pid, ^sid} =
      CrewSupervisor.start_officer(
        soul_id: sid, soul: soul, grant: %{authorities: grants}, tick_interval: 50
      )

    assert_receive {:tele, :commissioned, _, %{soul_id: ^sid}}, 5_000

    # Commission created the durable summary + a milestone.
    s0 = Service.summary(sid)
    assert s0.served_since != nil
    assert s0.orders_completed == 0
    {:ok, %{history: h0}} = Service.load(sid)
    assert Enum.any?(h0, &(&1.kind == "commissioned"))

    # Complete an order (dry-run mechanics).
    Fleet.order(sid, "status report", dry_run: true)
    assert_receive {:tele, :cognition_complete, _, _}, 10_000
    assert Service.summary(sid).orders_completed == 1
    assert Service.summary(sid).current_assignment["status"] == "completed"

    # The agent writes to its own duty log.
    Officer.log_duty(sid, "chose approach A because it was safest", tags: ["reasoning"])
    _ = Officer.status(sid)
    notes = DutyLog.for_soul(sid)
    assert Enum.any?(notes, &(&1.note =~ "approach A" and &1.authored_by == "agent"))

    # Dissent an order (its soul prohibits "sabotage") — a real value dissent.
    Fleet.order(sid, "sabotage the primary core")
    assert_receive {:tele, :dissent, _, %{basis: :value}}, 10_000
    assert Service.summary(sid).orders_dissented == 1

    # ── Crash it ──────────────────────────────────────────────────────────────
    ref = Process.monitor(pid)
    Process.exit(pid, :kill)
    assert_receive {:DOWN, ^ref, :process, _, _}, 5_000

    # The supervisor restarts it, and it rehydrates its durable self.
    assert_receive {:tele, :rehydrated, _, %{soul_id: ^sid}}, 10_000
    assert Officer.ready?(sid)

    [{new_pid, _}] = Registry.lookup(Fleet.Registry, {:officer, sid})
    refute new_pid == pid

    # Remembers who it is and what it was doing.
    st = Officer.status(sid)
    assert st.soul_loaded
    assert :cognition in st.grants
    assert {:world, "default"} in st.grants
    assert :issue_orders in st.grants
    assert st.assignment_status == "dissented"

    # Career counters + history intact, plus a rehydrated milestone.
    s1 = Service.summary(sid)
    assert s1.orders_completed == 1
    assert s1.orders_dissented == 1
    {:ok, %{history: h1}} = Service.load(sid)
    assert Enum.any?(h1, &(&1.kind == "rehydrated"))
    assert length(h1) > length(h0)

    # Its own duty note survived.
    assert Enum.any?(DutyLog.for_soul(sid), &(&1.note =~ "approach A"))

    # Re-commissioning the same soul is idempotent (returns the running pid).
    {:ok, same_pid, ^sid} = Fleet.commission(sid, soul: soul, grant: %{authorities: grants})
    assert same_pid == new_pid
  end
end
