defmodule Fleet.Service do
  @moduledoc """
  The runtime service-record writer: the AGENT-centric accountable career record
  keyed by `soul_id`. Written by the Fleet runtime only (never by an agent), so a
  soul can neither forge nor scrub its own service history. Append-only milestone
  events (`Atlas.Schemas.ServiceRecord`) plus a per-soul projection
  (`Atlas.Schemas.ServiceSummary`) read back on rehydration.

  No graceful degradation: a failed write is logged at :error and raised. If we
  cannot account for what happened, we fail loudly. The append-only record is
  written before the projection is updated (record-of-truth before projection).

  This module is called only from ensign lifecycle code — never from agent
  cognition.
  """
  require Logger
  alias Atlas.Schemas.{ServiceRecord, ServiceSummary}
  alias Brain.AtlasIntegration

  # ── Milestones ──────────────────────────────────────────────────────────

  @doc "First-ever commission: log the milestone and create the summary."
  def commission(soul_id, attrs) do
    now = DateTime.utc_now()

    append!(%{
      soul_id: soul_id,
      agent_id: attrs[:agent_id] || soul_id,
      kind: "commissioned",
      world_id: attrs[:mind_world_id],
      payload: %{"rank" => to_string(attrs[:rank] || "ensign")}
    })

    upsert!(soul_id, %{
      duty_status: "active",
      rank: to_string(attrs[:rank] || "ensign"),
      home_world_id: attrs[:home_world_id],
      mind_world_id: attrs[:mind_world_id],
      co_id: attrs[:co_id],
      reports: attrs[:reports] || [],
      standing_grants: Fleet.Authority.encode_set(attrs[:standing_grants] || []),
      served_since: now,
      last_commissioned_at: now
    })
  end

  @doc "Record that an ensign took on a standing order."
  def record_assignment(soul_id, order) do
    append!(%{
      soul_id: soul_id,
      kind: "order_assigned",
      order_id: order.id,
      world_id: order.world_id,
      under_order_of: to_string(order.from),
      payload: %{"directive" => to_string(order.directive)}
    })

    upsert!(soul_id, %{current_assignment: assignment_map(order, "acknowledged")})
  end

  @doc "Record an order outcome (:completed | :dissented | :failed) and bump the counter."
  def record_order_outcome(soul_id, order, outcome, attrs \\ %{}) do
    kind = "order_" <> to_string(outcome)

    append!(%{
      soul_id: soul_id,
      kind: kind,
      order_id: order && order.id,
      world_id: order && order.world_id,
      outcome: to_string(outcome),
      # :reason is a varchar(255) column — a failure reason built from
      # inspect(error_tuple) can easily run long (an HTTP error body, an
      # arbitrary exit reason). Truncating here, once, protects every
      # caller; the untruncated detail still survives in `payload`, a
      # jsonb column with no length limit.
      reason: truncate_reason(attrs[:reason]),
      payload: stringify(attrs)
    })

    counter =
      case outcome do
        :completed -> :orders_completed
        :dissented -> :orders_dissented
        :failed -> :orders_failed
      end

    bump(soul_id, counter, %{current_assignment: assignment_map(order, to_string(outcome))})
  end

  def record_relief(soul_id, by_principal) do
    append!(%{soul_id: soul_id, kind: "relieved", under_order_of: by_principal})
    bump(soul_id, :reliefs, %{duty_status: "relieved"})
  end

  def record_reinstatement(soul_id, by_principal) do
    append!(%{soul_id: soul_id, kind: "reinstated", under_order_of: by_principal})
    upsert!(soul_id, %{duty_status: "active"})
  end

  def record_achievement(soul_id, achievement) when is_map(achievement) do
    append!(%{soul_id: soul_id, kind: "achievement", payload: stringify(achievement)})
    current = summary(soul_id)
    upsert!(soul_id, %{achievements: (current && current.achievements || []) ++ [stringify(achievement)]})
  end

  @doc "Record that the ensign rehydrated after a restart (audit of the restart)."
  def record_rehydrated(soul_id, attrs) do
    append!(%{soul_id: soul_id, kind: "rehydrated", payload: stringify(attrs)})
  end

  # ── Reads (rehydration) ───────────────────────────────────────────────────

  @doc "Load the durable self: `%{summary, history}`. Raises if Atlas is unreachable."
  def load(soul_id) do
    case AtlasIntegration.sync(fn ->
           %{
             summary: Atlas.Repo.one(ServiceSummary.for_soul(soul_id)),
             history: Atlas.Repo.all(ServiceRecord.for_soul(soul_id))
           }
         end) do
      {:ok, result} -> {:ok, result}
      {:error, reason} -> raise "Fleet.Service.load failed (#{inspect(reason)})"
    end
  end

  @doc "The per-soul summary row, or nil."
  def summary(soul_id) do
    case AtlasIntegration.sync(fn -> Atlas.Repo.one(ServiceSummary.for_soul(soul_id)) end) do
      {:ok, s} -> s
      {:error, reason} -> raise "Fleet.Service.summary failed (#{inspect(reason)})"
    end
  end

  # ── Internals (error-surfacing write discipline) ──────────────────────────

  defp append!(attrs) do
    attrs = Map.put_new(attrs, :occurred_at, DateTime.utc_now())

    case AtlasIntegration.sync(fn ->
           %ServiceRecord{} |> ServiceRecord.changeset(attrs) |> Atlas.Repo.insert!()
         end) do
      {:ok, rec} ->
        rec

      {:error, reason} ->
        Logger.error("Fleet.Service: service record write failed", reason: inspect(reason))
        raise "Fleet.Service: service record write failed (#{inspect(reason)})"
    end
  end

  # Per-soul writes are serialized by the ensign process, so read-modify-write is
  # safe; we upsert the whole projection.
  defp upsert!(soul_id, changes) do
    changes = changes |> Map.put(:soul_id, soul_id)

    case AtlasIntegration.sync(fn ->
           %ServiceSummary{}
           |> ServiceSummary.changeset(changes)
           |> Atlas.Repo.insert!(
             on_conflict: {:replace, Map.keys(changes) ++ [:updated_at]},
             conflict_target: :soul_id
           )
         end) do
      {:ok, s} ->
        {:ok, s}

      {:error, reason} ->
        Logger.error("Fleet.Service: summary upsert failed", reason: inspect(reason))
        raise "Fleet.Service: summary upsert failed (#{inspect(reason)})"
    end
  end

  defp bump(soul_id, counter, extra_changes) do
    n = (summary(soul_id) && Map.get(summary(soul_id), counter)) || 0
    upsert!(soul_id, Map.put(extra_changes, counter, n + 1))
  end

  defp assignment_map(nil, _status), do: %{}

  defp assignment_map(order, status) do
    %{
      "order_id" => order.id,
      "directive" => to_string(order.directive),
      "status" => status,
      "world_id" => order.world_id,
      "from" => to_string(order.from),
      "grant" => %{"authorities" => Enum.map(Fleet.Authority.conferred_by(order), &Fleet.Authority.encode/1)}
    }
  end

  defp truncate_reason(nil), do: nil
  defp truncate_reason(s) when is_binary(s), do: String.slice(s, 0, 250)
  defp truncate_reason(other), do: other |> inspect() |> String.slice(0, 250)

  defp stringify(map) when is_map(map),
    do: Map.new(map, fn {k, v} -> {to_string(k), stringify_value(v)} end)

  defp stringify_value(v) when is_binary(v) or is_number(v) or is_boolean(v) or is_nil(v), do: v
  defp stringify_value(v) when is_atom(v), do: to_string(v)
  defp stringify_value(v) when is_map(v), do: stringify(v)
  defp stringify_value(v) when is_list(v), do: Enum.map(v, &stringify_value/1)
  defp stringify_value(v), do: inspect(v)
end
