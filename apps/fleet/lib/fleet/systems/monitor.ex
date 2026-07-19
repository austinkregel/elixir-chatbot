defmodule Fleet.Systems.Monitor do
  @moduledoc """
  Watches the black-box stream and raises/resolves **alerts** — the ship's alarm
  layer, and the substrate the relief-of-duty anomaly monitor and the fleet brake
  consume (LCARS §3 #10, `FLEET.md` §4.5). It subscribes to the sampler's
  `"systems:status"` push and, on each snapshot, applies rules:

    * a live system (service/process) that is `:down` → **critical**;
    * `:degraded` → **warning**;
    * the ship health score below a floor → a ship-health **warning/critical**;
    * recovery of any of the above → **resolved**.

  Each raise/resolve is emitted three ways (per FLEET.md §5.1 "reuse the telemetry
  path"): `:telemetry.execute([:chat_bot, :systems, :alert], …)`, a `"systems:alerts"`
  PubSub broadcast of the active set, and a durable append-only
  `Atlas.Schemas.SystemAlert` event. The *active* set is this GenServer's in-memory
  projection (read via `active/0`); the durable rows are the alarm history.
  """
  use GenServer
  require Logger

  alias Atlas.Schemas.SystemAlert

  @status_topic "systems:status"
  @alerts_topic "systems:alerts"
  @alert_event [:chat_bot, :systems, :alert]
  @health_floor 80
  @health_critical 50

  def start_link(opts \\ []), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc "The current active alerts (a list of maps), newest issues first."
  def active do
    GenServer.call(__MODULE__, :active, 1_000)
  catch
    :exit, _ -> []
  end

  def topic, do: @alerts_topic

  @impl true
  def init(_opts) do
    if Process.whereis(Brain.PubSub), do: Phoenix.PubSub.subscribe(Brain.PubSub, @status_topic)
    {:ok, %{active: %{}}}
  end

  @impl true
  def handle_call(:active, _from, state) do
    list = state.active |> Map.values() |> Enum.sort_by(& &1.since, :desc)
    {:reply, list, state}
  end

  @impl true
  def handle_info({:systems_snapshot, snap}, state) when is_map(snap) do
    {:noreply, evaluate(snap, state)}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # ── rules ──────────────────────────────────────────────────────────────────

  defp evaluate(snap, state) do
    desired = desired_alerts(snap)
    prev = state.active
    ship = snap.ship_id

    # raises: newly alerting, or severity changed
    Enum.each(desired, fn {id, a} ->
      case Map.get(prev, id) do
        %{severity: sev} when sev == a.severity -> :noop
        _ -> emit(ship, id, a.severity, :raised, a.message)
      end
    end)

    # resolves: were active, no longer alerting
    Enum.each(prev, fn {id, a} ->
      unless Map.has_key?(desired, id), do: emit(ship, id, a.severity, :resolved, "#{a.name} recovered")
    end)

    active = Map.new(desired, fn {id, a} -> {id, Map.put_new(a, :since, now())} end)
    broadcast(active)
    %{state | active: active}
  end

  defp desired_alerts(snap) do
    live = snap.services ++ snap.processes

    system_alerts =
      for s <- live, s.status in [:down, :degraded], into: %{} do
        {s.id, %{system_id: s.id, name: s.name, deck: s.deck, severity: severity_for(s.status),
                 message: "#{s.name} (#{s.deck}/#{s.section}) is #{s.status}", since: now()}}
      end

    score = snap.health[:health_score] || 100

    if score < @health_floor do
      sev = if score < @health_critical, do: :critical, else: :warning
      Map.put(system_alerts, "ship", %{system_id: "ship", name: "Ship health", deck: :ship,
        severity: sev, message: "ship health at #{score}%", since: now()})
    else
      system_alerts
    end
  end

  defp severity_for(:down), do: :critical
  defp severity_for(_), do: :warning

  # ── emit (telemetry + durable + [broadcast is per-snapshot]) ───────────────

  defp emit(ship_id, system_id, severity, kind, message) do
    :telemetry.execute(@alert_event, %{count: 1}, %{
      ship_id: ship_id, system_id: system_id, severity: severity, kind: kind, message: message
    })

    Logger.warning("Fleet.Systems.Monitor: #{kind} [#{severity}] #{system_id} — #{message}")
    persist(ship_id, system_id, severity, kind, message)
  end

  defp persist(ship_id, system_id, severity, kind, message) do
    attrs = %{ship_id: ship_id, system_id: system_id, severity: to_string(severity),
              kind: to_string(kind), message: message, occurred_at: DateTime.utc_now()}

    try do
      %SystemAlert{} |> SystemAlert.changeset(attrs) |> Atlas.Repo.insert!()
    rescue
      e -> Logger.error("Fleet.Systems.Monitor: alert write failed: #{inspect(e)}")
    catch
      _, _ -> :ok
    end
  end

  defp broadcast(active) do
    if Process.whereis(Brain.PubSub) do
      list = active |> Map.values() |> Enum.sort_by(& &1.since, :desc)
      Phoenix.PubSub.broadcast(Brain.PubSub, @alerts_topic, {:systems_alerts, list})
    end
  end

  defp now, do: System.system_time(:millisecond)
end
