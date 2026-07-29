defmodule Fleet.Audit do
  @moduledoc """
  The runtime audit trail. Every command-channel message is recorded here by the
  runtime — not by agents — so an agent can neither forge a superior's order nor
  scrub its own record. Two substrates:

    * a live telemetry event (`[:chat_bot, :officer, :event]`), and
    * a durable, append-only `Atlas.Schemas.CommandRecord` row.

  `from_agent` is always the principal the runtime **attributed** (see
  `Fleet.Comms.attribute/1`), never a value asserted in a payload.

  No graceful degradation: a failed audit write is surfaced (logged at :error
  and raised). If we cannot account for what happened, we fail loudly.
  """

  require Logger
  alias Atlas.Schemas.CommandRecord
  alias Brain.AtlasIntegration

  @doc """
  Record a command message. `kind` is the message kind (`:order`, `:ack`,
  `:sitrep`, `:request`, `:grant`, `:deny`, `:dissent`, `:report`, `:relieve`,
  `:reinstate`, `:provenance_anomaly`). `attrs` may include `:order_id`,
  `:from_agent`, `:to_agent`, `:world_id`, `:authority`, `:verdict`, `:reason`,
  `:payload`.
  """
  def record(kind, attrs) when is_atom(kind) and is_map(attrs) do
    attrs =
      attrs
      |> Map.put(:kind, Atom.to_string(kind))
      |> Map.put_new(:issued_at, DateTime.utc_now())
      # Stamp THIS ship onto every record, so the black box is filterable by ship
      # for the fleet-of-ships future (a caller may override by passing :ship_id).
      |> Map.put_new(:ship_id, Fleet.Ship.id())
      |> normalize()

    # Durable append-only record FIRST — then the live telemetry event, so an
    # observer that sees the telemetry can rely on the record already existing.
    record =
      case AtlasIntegration.sync(fn ->
             %CommandRecord{} |> CommandRecord.changeset(attrs) |> Atlas.Repo.insert!()
           end) do
        {:ok, record} ->
          record

        {:error, reason} ->
          Logger.error("Fleet.Audit: command audit write failed",
            kind: kind,
            reason: inspect(reason)
          )

          raise "Fleet.Audit: command audit write failed (#{inspect(reason)})"
      end

    Fleet.Telemetry.emit_event(attrs[:from_agent] || "system", kind, %{}, %{
      order_id: attrs[:order_id],
      to_agent: attrs[:to_agent],
      authority: attrs[:authority],
      verdict: attrs[:verdict]
    })

    {:ok, record}
  end

  # CommandRecord string fields must receive strings; normalize terms.
  defp normalize(attrs) do
    attrs
    |> stringify(:authority)
    |> stringify(:verdict)
    |> stringify(:reason)
    |> Map.update(:payload, %{}, fn
      p when is_map(p) -> p
      other -> %{value: inspect(other)}
    end)
  end

  defp stringify(attrs, key) do
    case Map.get(attrs, key) do
      nil -> attrs
      v when is_binary(v) -> attrs
      v when is_atom(v) -> Map.put(attrs, key, Atom.to_string(v))
      v -> Map.put(attrs, key, inspect(v))
    end
  end
end
