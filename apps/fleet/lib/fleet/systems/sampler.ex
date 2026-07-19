defmodule Fleet.Systems.Sampler do
  @moduledoc """
  The ship's black box: periodically snapshots `Fleet.Systems` and records it three
  ways so the whole ship's state over time is inspectable, cheaply and durably.

    * **Live ring buffer (ETS)** — the last ~60 samples of the ship health score and
      each live system's status, for sparklines. Lock-free public reads, so the
      board never probes ~51 GenServers on render — it reads the last sample.
    * **Durable samples (Postgres)** — a periodic health rollup row plus a row per
      *status transition* (only when a system actually changes), keeping volume
      bounded. Written to `Atlas.Schemas.SystemStatusSample`, ship-stamped, so the
      recorder survives restarts and is filterable per ship.
    * **Live push (PubSub)** — each snapshot broadcast on `"systems:status"` so the
      board and console update in real time.

  A failed durable write is logged loudly but does NOT crash the recorder — a
  continuous black box must keep running to keep providing live truth (the ring
  buffer is unaffected).

      config :fleet, systems_sample_interval_ms: 30_000
  """
  use GenServer
  require Logger

  alias Atlas.Schemas.SystemStatusSample

  @table :fleet_systems_samples
  @ring 60
  @default_interval 30_000
  @start_delay 5_000
  @topic "systems:status"

  # ── public reads (lock-free ETS) ───────────────────────────────────────────

  @doc "The last full snapshot the sampler took, or nil if none yet."
  def latest, do: lookup(:latest)

  @doc "The ship health-score ring, newest first: `[{unix_ms, score}]`."
  def health_history, do: lookup(:health_history) || []

  @doc "A live system's recent status ring, newest first: `[status_atom]`."
  def history(system_id), do: lookup({:sys, system_id}) || []

  def topic, do: @topic

  defp lookup(key) do
    case :ets.info(@table) do
      :undefined -> nil
      _ -> case :ets.lookup(@table, key), do: ([{^key, v}] -> v; _ -> nil)
    end
  end

  # ── lifecycle ──────────────────────────────────────────────────────────────

  def start_link(opts \\ []), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @impl true
  def init(_opts) do
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
    Process.send_after(self(), :sample, @start_delay)
    {:ok, %{prev: %{}}}
  end

  @impl true
  def handle_info(:sample, state) do
    state = sample(state)
    Process.send_after(self(), :sample, interval())
    {:noreply, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  defp sample(state) do
    snap = safe(fn -> Fleet.Systems.snapshot() end)

    if is_map(snap) do
      :ets.insert(@table, {:latest, snap})
      record_health_ring(snap)
      live = snap.services ++ snap.processes
      record_system_rings(live)
      transitions = record_transitions(state.prev, live, snap.ship_id, snap.health)
      broadcast(snap)
      _ = transitions
      %{state | prev: Map.new(live, &{&1.id, &1.status})}
    else
      state
    end
  end

  # ── ring buffer ────────────────────────────────────────────────────────────

  defp record_health_ring(snap) do
    score = snap.health[:health_score] || 0
    ring = [{now_ms(), score} | health_history()] |> Enum.take(@ring)
    :ets.insert(@table, {:health_history, ring})
  end

  defp record_system_rings(live) do
    for s <- live do
      ring = [s.status | history(s.id)] |> Enum.take(@ring)
      :ets.insert(@table, {{:sys, s.id}, ring})
    end
  end

  # ── durable samples ────────────────────────────────────────────────────────

  # A health rollup every sample + a row for each system whose status changed
  # since the previous sample. The first sample only seeds `prev` (no transitions).
  defp record_transitions(prev, live, ship_id, health) do
    persist(ship_id, "ship", to_string(health[:health_status] || :unknown), (health[:health_score] || 0) * 1.0)

    changed =
      if prev == %{} do
        []
      else
        Enum.filter(live, fn s -> Map.get(prev, s.id) != s.status end)
      end

    for s <- changed, do: persist(ship_id, s.id, to_string(s.status), nil)
    changed
  end

  defp persist(ship_id, system_id, status, metric) do
    attrs = %{ship_id: ship_id, system_id: system_id, status: status, metric: metric, sampled_at: DateTime.utc_now()}

    try do
      %SystemStatusSample{} |> SystemStatusSample.changeset(attrs) |> Atlas.Repo.insert!()
    rescue
      e -> Logger.error("Fleet.Systems.Sampler: durable sample write failed: #{inspect(e)}")
    catch
      _, _ -> :ok
    end
  end

  # ── push ───────────────────────────────────────────────────────────────────

  defp broadcast(snap) do
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.broadcast(Brain.PubSub, @topic, {:systems_snapshot, snap})
    end
  end

  # ── helpers ────────────────────────────────────────────────────────────────

  defp interval, do: Application.get_env(:fleet, :systems_sample_interval_ms, @default_interval)
  defp now_ms, do: System.system_time(:millisecond)

  defp safe(fun) do
    fun.()
  rescue
    e -> Logger.debug("Fleet.Systems.Sampler: snapshot failed: #{inspect(e)}"); nil
  catch
    _, _ -> nil
  end
end
