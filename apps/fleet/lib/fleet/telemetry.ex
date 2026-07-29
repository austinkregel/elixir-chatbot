defmodule Fleet.Telemetry do
  @moduledoc """
  Telemetry for the crew. Every officer emits `[:chat_bot, :officer, :event]` from
  its first line of life — the read surface the Fleet page (Phase 4) will consume.

  Follows the discipline of `Brain.Telemetry`: emission is a thin
  `:telemetry.execute/3`; handlers are fire-and-forget and guarded so a missing
  downstream (e.g. the metrics aggregator not running in a fleet-only test) is a
  no-op rather than a crash.

  Measurements are numeric (`count`, `duration_ms`, `queue_len`); metadata is
  dimensional (`agent_id`, `event`, `order_id`, `status`, `timestamp`).
  """

  require Logger

  @officer_event [:chat_bot, :officer, :event]

  @doc "The officer telemetry event name."
  def officer_event, do: @officer_event

  @doc """
  Emits an officer lifecycle/work event. `event` is the semantic marker
  (`:spawned`, `:soul_hydrated`, `:order_received`, `:ack`, `:tick`,
  `:cognition_complete`, `:cognition_failed`, ...).
  """
  def emit_event(agent_id, event, measurements \\ %{}, metadata \\ %{}) do
    :telemetry.execute(
      @officer_event,
      Map.merge(%{count: 1}, measurements),
      metadata
      |> Map.merge(%{
        agent_id: agent_id,
        event: event,
        timestamp: System.monotonic_time(:millisecond)
      })
    )
  end

  @doc "Attaches the Fleet telemetry handlers. Called during application startup."
  def attach_handlers do
    :telemetry.attach(
      "fleet-officer-event",
      @officer_event,
      &__MODULE__.handle_officer_event/4,
      %{}
    )
  end

  @doc "Detaches the Fleet telemetry handlers (used in tests)."
  def detach_handlers do
    :telemetry.detach("fleet-officer-event")
  end

  @doc false
  def handle_officer_event(_event_name, measurements, metadata, _config) do
    Logger.debug("officer event", metadata)

    # Bridge to Phoenix.PubSub so the Fleet page (and any subscriber) gets a live
    # activity feed. Best-effort: a missing bus or failed broadcast must never
    # break telemetry emission on the hot path.
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.broadcast(
        Brain.PubSub,
        "fleet:events",
        {:fleet_event, Map.put(metadata, :measurements, measurements)}
      )
    end

    :ok
  rescue
    _ -> :ok
  end
end
