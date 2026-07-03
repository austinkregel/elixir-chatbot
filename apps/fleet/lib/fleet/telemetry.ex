defmodule Fleet.Telemetry do
  @moduledoc """
  Telemetry for the crew. Every ensign emits `[:chat_bot, :ensign, :event]` from
  its first line of life — the read surface the Fleet page (Phase 4) will consume.

  Follows the discipline of `Brain.Telemetry`: emission is a thin
  `:telemetry.execute/3`; handlers are fire-and-forget and guarded so a missing
  downstream (e.g. the metrics aggregator not running in a fleet-only test) is a
  no-op rather than a crash.

  Measurements are numeric (`count`, `duration_ms`, `queue_len`); metadata is
  dimensional (`agent_id`, `event`, `order_id`, `status`, `timestamp`).
  """

  require Logger

  @ensign_event [:chat_bot, :ensign, :event]

  @doc "The ensign telemetry event name."
  def ensign_event, do: @ensign_event

  @doc """
  Emits an ensign lifecycle/work event. `event` is the semantic marker
  (`:spawned`, `:soul_hydrated`, `:order_received`, `:ack`, `:tick`,
  `:cognition_complete`, `:cognition_failed`, ...).
  """
  def emit_event(agent_id, event, measurements \\ %{}, metadata \\ %{}) do
    :telemetry.execute(
      @ensign_event,
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
      "fleet-ensign-event",
      @ensign_event,
      &__MODULE__.handle_ensign_event/4,
      %{}
    )
  end

  @doc "Detaches the Fleet telemetry handlers (used in tests)."
  def detach_handlers do
    :telemetry.detach("fleet-ensign-event")
  end

  @doc false
  def handle_ensign_event(_event_name, _measurements, metadata, _config) do
    # Fire-and-forget. Phase 1 keeps a light audit trail; richer aggregation
    # (a fleet-scoped ETS store feeding the Fleet page) is deferred to Phase 4.
    Logger.debug("ensign event", metadata)
    :ok
  rescue
    _ -> :ok
  end
end
