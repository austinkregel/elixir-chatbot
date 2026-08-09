defmodule Fleet.Order do
  @moduledoc """
  An ORDER: the unit of command handed to an officer, and the standing assignment
  record it carries while working. In Phase 1 this is a plain struct (transient
  message + in-process record). Its status shape mirrors
  `Atlas.Schemas.ResearchGoal` so it can later back onto a persisted, auditable
  record without reshaping.

  ## An order is a structured object, not a sentence

  "Orders are structured objects, not vibes" is the second of the ten
  structure-over-trust mechanisms: every task carries its issuer, objective,
  constraints and authority explicitly, so that "make it so" is logged,
  attributed, and specific.

  Carrying the objective as bare prose costs more than tidiness. Analysis has to
  *infer* what an order asks for, and inference on terse command language is
  unreliable — measured on real directives, "Report your status." classified as
  `music.search` and "Give a one-line readiness report." as `reminder.create`,
  both around 0.75 confidence. Anything gating on that guess refuses the wrong
  orders. `constraints` and `risk_class` are the issuer stating what a
  classifier would otherwise have to invent.

  Fields:
    * `id`        — order id.
    * `from`      — provenance: who issued it (`:admiral` | agent_id). DATA in
      Phase 1; sender authentication (via Registry) is deferred to Phase 2.
    * `reply_to`  — pid the ACK is sent back to (recorded at receipt).
    * `objective` — what the officer is to accomplish (≈ `ResearchGoal.topic`).
    * `directive` — alias of `objective`, kept because the whole lifecycle
      (audit payloads, `current_assignment`, memory ingest) reads it by that
      name. `new/1` keeps the two in step; prefer `objective` in new code.
    * `constraints` — bounds the officer must respect ("read-only", "stay in the
      repo"). Stated, never inferred; they reach both the assessment and the
      realization prompt.
    * `context_refs` — briefing material the issuer chose to attach.
    * `risk_class` — `:routine | :sensitive | :irreversible`; drives process,
      see `risk_class/1`.
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
  @valid_risk_classes [:routine, :sensitive, :irreversible]

  @type risk_class :: :routine | :sensitive | :irreversible

  @type t :: %__MODULE__{
          id: String.t(),
          from: atom() | String.t(),
          reply_to: pid() | nil,
          objective: String.t(),
          directive: String.t(),
          constraints: [String.t()],
          context_refs: [String.t()],
          risk_class: risk_class(),
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
            objective: nil,
            directive: nil,
            constraints: [],
            context_refs: [],
            risk_class: :routine,
            grant: %{},
            world_id: "default",
            dry_run: false,
            priority: "normal",
            issued_at: nil,
            status: "pending"

  @doc """
  Build an order, keeping `objective` and `directive` in step and validating
  `risk_class`.

  Either name may be given; whichever is present fills the other, so existing
  callers that speak `directive:` keep working while new code can say
  `objective:`.
  """
  @spec new(keyword() | map()) :: t()
  def new(fields) do
    fields = Map.new(fields)
    text = Map.get(fields, :objective) || Map.get(fields, :directive)

    risk_class = normalize_risk_class(Map.get(fields, :risk_class, :routine))

    struct(
      %__MODULE__{},
      fields
      |> Map.put(:objective, text)
      |> Map.put(:directive, text)
      |> Map.put(:risk_class, risk_class)
      |> Map.put(:constraints, normalize_list(Map.get(fields, :constraints, [])))
      |> Map.put(:context_refs, normalize_list(Map.get(fields, :context_refs, [])))
    )
  end

  @doc "The valid order statuses."
  def valid_statuses, do: @valid_statuses

  @doc "The valid risk classes."
  @spec valid_risk_classes() :: [risk_class()]
  def valid_risk_classes, do: @valid_risk_classes

  @doc """
  The order's risk class, tolerating a string form from a rehydrated record.

  Risk class drives process, it is not a label: `:routine` dispatches once
  acknowledged, `:sensitive` requires the issuer to confirm the readback before
  work begins, and `:irreversible` requires two-officer sign-off.
  """
  @spec risk_class(t()) :: risk_class()
  def risk_class(%__MODULE__{risk_class: rc}), do: normalize_risk_class(rc)

  defp normalize_risk_class(rc) when rc in @valid_risk_classes, do: rc

  defp normalize_risk_class(rc) when is_binary(rc) do
    case Enum.find(@valid_risk_classes, &(to_string(&1) == rc)) do
      nil -> raise ArgumentError, "invalid risk_class: #{inspect(rc)}"
      found -> found
    end
  end

  defp normalize_risk_class(nil), do: :routine

  defp normalize_risk_class(rc),
    do: raise(ArgumentError, "invalid risk_class: #{inspect(rc)}")

  defp normalize_list(nil), do: []
  defp normalize_list(list) when is_list(list), do: Enum.map(list, &to_string/1)
  defp normalize_list(one), do: [to_string(one)]

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
