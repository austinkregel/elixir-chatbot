defmodule Fleet.DutyLog do
  @moduledoc """
  The agent's own append-only duty log (working notes/journal). Unlike
  `Fleet.Service` (runtime-written accountability), an agent MAY write here — but
  only ever with its own `soul_id`, never a value from a payload. Still
  append-only: notes can be added, never edited or deleted.

  No graceful degradation: a lost note is a real failure, surfaced and raised.
  """
  require Logger
  alias Atlas.Schemas.DutyLogEntry
  alias Brain.AtlasIntegration

  @doc "Append a note authored by the agent identified by `soul_id`."
  def note(soul_id, note, opts \\ []) do
    attrs = %{
      soul_id: soul_id,
      note: to_string(note),
      order_id: Keyword.get(opts, :order_id),
      world_id: Keyword.get(opts, :world_id),
      tags: Keyword.get(opts, :tags, []),
      authored_by: "agent",
      payload: Keyword.get(opts, :payload, %{}),
      occurred_at: DateTime.utc_now()
    }

    case AtlasIntegration.sync(fn ->
           %DutyLogEntry{} |> DutyLogEntry.changeset(attrs) |> Atlas.Repo.insert!()
         end) do
      {:ok, entry} ->
        {:ok, entry}

      {:error, reason} ->
        Logger.error("Fleet.DutyLog: note write failed", reason: inspect(reason))
        raise "Fleet.DutyLog: note write failed (#{inspect(reason)})"
    end
  end

  @doc "All duty-log entries for a soul, oldest first."
  def for_soul(soul_id) do
    case AtlasIntegration.sync(fn -> Atlas.Repo.all(DutyLogEntry.for_soul(soul_id)) end) do
      {:ok, entries} -> entries
      {:error, reason} -> raise "Fleet.DutyLog.for_soul failed (#{inspect(reason)})"
    end
  end
end
