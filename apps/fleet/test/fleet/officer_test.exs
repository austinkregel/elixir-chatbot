defmodule Fleet.OfficerTest do
  @moduledoc """
  Proves the officer mechanics: spawn → order → ACK → autonomous tick →
  non-blocking cognition dispatch. Cognition is dry-run and the soul is injected
  as a fixture; runs against the live stack (the runtime audit writes to the DB).
  """
  use Fleet.FleetCase, async: false

  alias Fleet.{CrewSupervisor, Officer}

  # The :fleet application is already started by `mix test`, so its supervision
  # tree (Fleet.Registry / Fleet.TaskSupervisor / Fleet.CrewSupervisor) is live.
  # We commission officers into the running CrewSupervisor and retire them after
  # each test for isolation.
  setup do
    test_pid = self()
    handler_id = "test-officer-#{System.unique_integer([:positive])}"

    :telemetry.attach(
      handler_id,
      [:chat_bot, :officer, :event],
      fn _event, meas, meta, pid -> send(pid, {:tele, meta.event, meas, meta}) end,
      test_pid
    )

    on_exit(fn ->
      :telemetry.detach(handler_id)

      for {_, pid, _, _} <- CrewSupervisor.list(), is_pid(pid) do
        CrewSupervisor.retire(pid)
      end
    end)

    soul = %Brain.Soul{
      id: "ensign-test-#{System.unique_integer([:positive])}",
      name: "Ensign Test",
      constitution: "Be careful."
    }
    {:ok, soul: soul}
  end

  test "spawn → order → ack → tick → non-blocking dispatch", %{soul: soul} do
    {:ok, _pid, id} =
      CrewSupervisor.start_officer(soul_id: soul.id, soul: soul, tick_interval: 50)

    assert_receive {:tele, :spawned, _, _}
    assert_receive {:tele, :soul_hydrated, _, _}
    assert Officer.ready?(id)

    order = %Order{id: "o1", from: :admiral, reply_to: self(), directive: "scan sector", dry_run: true}
    Officer.order(id, order)

    # ACK arrives three ways: to the caller and on the telemetry stream.
    assert_receive {:ack, %{order_id: "o1", status: :accepted}}
    assert_receive {:tele, :ack, _, %{order_id: "o1"}}
    assert_receive {:tele, :order_received, _, %{order_id: "o1"}}

    # Autonomous tick fires and the dispatched (dry-run) cognition completes.
    assert_receive {:tele, :tick, _, _}
    assert_receive {:tele, :cognition_complete, %{duration_ms: dt}, %{order_id: "o1"}}, 2_000
    assert is_integer(dt)
  end

  # Risk class is process. Until two-officer sign-off exists, an irreversible
  # order is refused on the record rather than executed as if it were routine —
  # the same posture the LCARS prototype takes at order composition.
  test "an irreversible order is dissented, not executed", %{soul: soul} do
    {:ok, _pid, id} =
      CrewSupervisor.start_officer(soul_id: soul.id, soul: soul, tick_interval: 50)

    assert_receive {:tele, :soul_hydrated, _, _}, 2_000

    order =
      Order.new(
        id: "o-irr",
        from: :admiral,
        reply_to: self(),
        objective: "purge the archive",
        risk_class: :irreversible
      )

    Officer.order(id, order)
    assert_receive {:ack, %{order_id: "o-irr"}}

    assert_receive {:tele, :dissent, _, %{order_id: "o-irr", basis: :process}}, 2_000
    refute_receive {:tele, :cognition_complete, _, %{order_id: "o-irr"}}, 500

    assert Officer.status(id).assignment_status == "dissented"
  end

  # The ACK is a readback, not a receipt: it restates what the ship understood
  # the order to be, so an issuer can catch a misread before work starts.
  test "the ACK carries a readback of the order as received", %{soul: soul} do
    {:ok, _pid, id} =
      CrewSupervisor.start_officer(soul_id: soul.id, soul: soul, tick_interval: 50)

    assert_receive {:tele, :soul_hydrated, _, _}, 2_000

    order =
      Order.new(
        id: "o-rb",
        from: :admiral,
        reply_to: self(),
        objective: "summarize the last three CI runs",
        constraints: ["read-only: do not modify any file"],
        context_refs: ["briefing/ci.md"],
        risk_class: :sensitive,
        dry_run: true
      )

    Officer.order(id, order)

    assert_receive {:ack, %{order_id: "o-rb", readback: readback}}

    assert readback.objective == "summarize the last three CI runs"
    assert readback.constraints == ["read-only: do not modify any file"]
    assert readback.risk_class == "sensitive"
    assert readback.context_refs == ["briefing/ci.md"]
    assert is_list(readback.authorities)
  end

  test "a routine order with stated constraints dispatches normally", %{soul: soul} do
    {:ok, _pid, id} =
      CrewSupervisor.start_officer(soul_id: soul.id, soul: soul, tick_interval: 50)

    assert_receive {:tele, :soul_hydrated, _, _}, 2_000

    order =
      Order.new(
        id: "o-con",
        from: :admiral,
        reply_to: self(),
        objective: "survey the sector",
        constraints: ["read-only: do not modify any file"],
        dry_run: true
      )

    Officer.order(id, order)
    assert_receive {:ack, %{order_id: "o-con"}}
    assert_receive {:tele, :cognition_complete, _, %{order_id: "o-con"}}, 2_000
  end

  test "mailbox stays responsive while cognition is in flight", %{soul: soul} do
    {:ok, _pid, id} =
      CrewSupervisor.start_officer(soul_id: soul.id, soul: soul, tick_interval: 50)

    assert_receive {:tele, :soul_hydrated, _, _}

    Officer.order(id, %Order{id: "o2", reply_to: self(), directive: "long task", dry_run: true})
    assert_receive {:ack, %{order_id: "o2"}}

    # The dry-run Task sleeps 50ms; a status call must still return promptly,
    # proving the controller never blocks on its dispatched work.
    {micros, status} = :timer.tc(fn -> Officer.status(id) end)
    assert status.agent_id == id
    assert micros < 100_000
  end

  test "unknown soul id leaves the officer alive but not ready" do
    {:ok, _pid, id} = CrewSupervisor.start_officer(soul_id: "no-such-soul", tick_interval: 50)

    assert_receive {:tele, :spawned, _, _}
    assert_receive {:tele, :soul_hydrate_failed, _, _}
    refute Officer.ready?(id)
    # Still addressable and responsive.
    assert Officer.status(id).soul_loaded == false
  end

  # Real cognition against the live stack (Postgres + Ouro). Excluded by default;
  # run with `mix do --app fleet mix test --include integration` (stack up).
  # Proves the officer runs REAL cognition through a world where JJ-7 resides, so
  # the acting soul's constitution steers generation (not the generic default).
  @tag :integration
  test "officer runs real cognition through a world where JJ-7 resides" do
    # Point the soul loader at the umbrella-root souls/ regardless of cwd
    # (`mix do --app fleet` runs from apps/fleet). __DIR__ is cwd-independent.
    souls_dir = Path.expand(Path.join([__DIR__ | List.duplicate("..", 4)] ++ ["souls"]))
    prev_souls_dir = Application.get_env(:brain, :souls_dir)
    Application.put_env(:brain, :souls_dir, souls_dir)

    world_id = "fleet-test-jj7-#{System.unique_integer([:positive])}"
    {:ok, _world} = World.Manager.create("Fleet Test (JJ-7)", id: world_id, residents: ["ensign-jj7"])

    on_exit(fn ->
      World.Manager.destroy(world_id)

      if prev_souls_dir,
        do: Application.put_env(:brain, :souls_dir, prev_souls_dir),
        else: Application.delete_env(:brain, :souls_dir)
    end)

    # Deterministic seam: the world feeds JJ-7 into generation. This is what makes
    # the officer's cognition soul-steered rather than the generic default.
    assert %Brain.Soul{id: "ensign-jj7"} = World.Roster.acting_soul(world_id)

    {:ok, _pid, id} = Fleet.commission("ensign-jj7", tick_interval: 500)
    assert_receive {:tele, :soul_hydrated, _, _}
    assert Officer.ready?(id)

    # A REAL order (dry_run defaults to false) → Brain.create_conversation +
    # Brain.evaluate against world_id, so JJ-7's constitution is the system prompt.
    Fleet.order(id, "In one sentence, report your current readiness.", world_id: world_id)

    assert_receive {:ack, %{order_id: order_id, status: :accepted}}
    assert_receive {:tele, :cognition_complete, %{duration_ms: dt}, %{order_id: ^order_id}}, 180_000
    assert is_integer(dt)
    assert Officer.status(id).assignment_status == "completed"
  end
end
