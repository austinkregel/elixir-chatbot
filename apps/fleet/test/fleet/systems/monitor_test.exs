defmodule Fleet.Systems.MonitorTest do
  @moduledoc "The alarm layer raises a durable alert when a system goes down, and resolves it on recovery."
  use Fleet.FleetCase, async: false

  @moduletag :integration

  import Ecto.Query
  alias Fleet.Systems.{Monitor, System}
  alias Atlas.Schemas.SystemAlert

  defp snap(statuses) do
    ship = Fleet.Ship.id()

    procs =
      for {id, st} <- statuses do
        %System{id: id, name: id, layer: :process, deck: :test, section: "t", status: st, ship_id: ship}
      end

    %{ship_id: ship, services: [], processes: procs, health: %{health_score: 100, health_status: :healthy}}
  end

  test "a down system raises a critical alert; recovery resolves it (telemetry + durable)" do
    tp = self()
    hid = "mon-#{:erlang.unique_integer([:positive])}"
    :telemetry.attach(hid, [:chat_bot, :systems, :alert],
      fn _e, _m, meta, pid -> send(pid, {:alert, meta}) end, tp)
    on_exit(fn -> :telemetry.detach(hid) end)

    sid = "proc:test:#{:erlang.unique_integer([:positive])}"

    # a live system goes down → raised, critical
    send(Monitor, {:systems_snapshot, snap([{sid, :down}])})
    assert_receive {:alert, %{system_id: ^sid, kind: :raised, severity: :critical}}, 5_000
    _ = Monitor.active()  # sync: the call returns only after the snapshot's handle_info (incl. the durable write) completes
    assert Atlas.Repo.exists?(from a in SystemAlert, where: a.system_id == ^sid and a.kind == "raised" and a.severity == "critical")

    # it recovers → resolved
    send(Monitor, {:systems_snapshot, snap([{sid, :up}])})
    assert_receive {:alert, %{system_id: ^sid, kind: :resolved}}, 5_000
    _ = Monitor.active()
    assert Atlas.Repo.exists?(from a in SystemAlert, where: a.system_id == ^sid and a.kind == "resolved")
  end
end
