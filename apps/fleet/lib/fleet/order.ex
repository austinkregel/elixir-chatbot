defmodule Fleet.Order do
  @moduledoc """
  An ORDER: the unit of command handed to an ensign, and the standing assignment
  record it carries while working. In Phase 1 this is a plain struct (transient
  message + in-process record). Its status shape mirrors
  `Atlas.Schemas.ResearchGoal` so it can later back onto a persisted, auditable
  record without reshaping.

  Fields:
    * `id`        — order id.
    * `from`      — provenance: who issued it (`:admiral` | agent_id). DATA in
      Phase 1; sender authentication (via Registry) is deferred to Phase 2.
    * `reply_to`  — pid the ACK is sent back to (recorded at receipt).
    * `directive` — the task text the ensign acts on (≈ `ResearchGoal.topic`).
    * `grant`     — the order's authority scope: `%{authorities: [term], provenance: :command | :data}`.
      `authorities` are conferred to the report at ACK (see `Fleet.Authority`);
      `provenance: :data` marks a directive that came from data, not the command
      channel (an anomaly the agent dissents from rather than obeys).
    * `world_id`  — binds cognition (soul is resolved via the world roster).
    * `dry_run`   — offline test affordance: skip real cognition. Real orders
      leave this false.
    * `priority`  — scheduling hint (default "normal").
    * `issued_at` — monotonic ms when issued.
    * `status`    — one of `@valid_statuses`.
  """

  @valid_statuses ~w(pending acknowledged in_progress blocked completed failed dissented)

  @type t :: %__MODULE__{
          id: String.t(),
          from: atom() | String.t(),
          reply_to: pid() | nil,
          directive: String.t(),
          grant: map(),
          world_id: String.t(),
          dry_run: boolean(),
          priority: String.t(),
          issued_at: integer() | nil,
          status: String.t()
        }

  defstruct id: nil,
            from: :admiral,
            reply_to: nil,
            directive: nil,
            grant: %{},
            world_id: "default",
            dry_run: false,
            priority: "normal",
            issued_at: nil,
            status: "pending"

  @doc "The valid order statuses."
  def valid_statuses, do: @valid_statuses

  @doc """
  Sets the order's status. Mirrors `ResearchGoal.update_status/2`; a status
  outside `@valid_statuses` is rejected with an `ArgumentError` so an illegal
  transition fails loudly rather than corrupting the record.
  """
  @spec update_status(t(), String.t()) :: t()
  def update_status(%__MODULE__{} = order, status) when status in @valid_statuses do
    %{order | status: status}
  end

  def update_status(%__MODULE__{}, status) do
    raise ArgumentError, "invalid order status: #{inspect(status)}"
  end
end
