defmodule Brain.Directive.Assessment do
  @moduledoc """
  The deterministic verdict on a directive: may it be carried out, must it be
  clarified first, or must it be refused?

  This is a plain data structure with no runtime dependencies — it is safe to
  pattern-match from any app (notably `Fleet`, which maps verdicts onto its
  signal protocol). `Brain.Directive.Assessor` produces it; nothing generative
  contributes to it.

  ## Verdicts

    * `:comply` — every check passed; the directive may proceed to generation.
    * `:clarify` — the directive is understood too poorly, is missing required
      parameters, or asserts a premise the agent's own beliefs contradict. The
      right next move is to ask, not to act.
    * `:refuse` — the directive names an action the system provably cannot
      perform.

  Precedence when several gating checks fire: `:refuse` > `:clarify` > `:comply`.

  ## Reasons vs advisories

  `reasons` are the findings that produced the verdict. `advisories` are
  findings recorded but deliberately not allowed to block — currently
  capability and slot gaps, whose backing intent classification is not
  trustworthy enough for free-form directives to justify refusing on
  (`Brain.Directive.Assessor` documents the measurements). Advisories still
  reach the report payload and the generation prompt, so the agent can say what
  it noticed while still carrying out the order.

  ## Honest checks

  `checks` records, per check, whether it actually `:ran` or was `:unavailable`
  (its backing service was down). An unavailable check is never silently
  treated as a pass — a caller that needs a guarantee can see it did not run.
  This is deliberate: `Brain.Response.ResponseEvaluator`'s belief-grounding
  score degrades to a flat 0.7 when its stores are down, which reads as a soft
  pass and hides the gap. Directive assessment does not repeat that.
  """

  @type verdict :: :comply | :clarify | :refuse

  @type reason ::
          {:not_comprehended, map()}
          | {:missing_slots, [String.t()]}
          | {:contradicted_premise, map()}
          | {:incapable_action, String.t()}

  @type check_state :: :ran | :unavailable | :skipped

  @type capability :: :cognitive | {:external, :capable | :incapable}

  @type t :: %__MODULE__{
          verdict: verdict(),
          reasons: [reason()],
          advisories: [reason()],
          constraints: [String.t()],
          risk_class: atom(),
          gating: boolean(),
          intent: String.t() | nil,
          comprehension: %{
            verdict: atom() | nil,
            composite: float() | nil,
            gaps: [map()]
          },
          slots: %{filled: map(), missing: [String.t()], prompts: [String.t()]},
          capability: capability(),
          task_frames: %{events: [map()], srl_triples: [map()]},
          premises: [map()],
          confidence: float(),
          checks: %{atom() => check_state()}
        }

  defstruct verdict: :comply,
            reasons: [],
            advisories: [],
            constraints: [],
            risk_class: :routine,
            gating: false,
            intent: nil,
            comprehension: %{verdict: nil, composite: nil, gaps: []},
            slots: %{filled: %{}, missing: [], prompts: []},
            capability: :cognitive,
            task_frames: %{events: [], srl_triples: []},
            premises: [],
            confidence: 0.0,
            checks: %{}

  @doc """
  The single reason tag that best explains a non-`:comply` verdict, for use as
  a dissent/blocked `rule`. Returns `nil` for `:comply`.
  """
  @spec primary_reason_tag(t()) :: atom() | nil
  def primary_reason_tag(%__MODULE__{verdict: :comply}), do: nil
  def primary_reason_tag(%__MODULE__{reasons: []}), do: nil
  def primary_reason_tag(%__MODULE__{reasons: [{tag, _} | _]}), do: tag

  @doc """
  A one-line human-readable explanation of the verdict, suitable for a dissent
  `reason` field or an operator log line.
  """
  @spec explain(t()) :: String.t()
  def explain(%__MODULE__{verdict: :comply}), do: "directive assessed as actionable"

  def explain(%__MODULE__{reasons: []} = a),
    do: "directive #{a.verdict} (no reason recorded)"

  def explain(%__MODULE__{reasons: reasons}) do
    reasons |> Enum.map(&reason_text/1) |> Enum.join("; ")
  end

  @doc """
  A flat, JSON-encodable map of the assessment, for signal payloads and audit
  records. Structs and tuples are flattened; nothing here needs Brain loaded to
  be read back.
  """
  @spec to_report_payload(t()) :: map()
  def to_report_payload(%__MODULE__{} = a) do
    %{
      verdict: to_string(a.verdict),
      reasons: Enum.map(a.reasons, &reason_text/1),
      advisories: Enum.map(a.advisories, &reason_text/1),
      constraints: a.constraints || [],
      risk_class: maybe_to_string(a.risk_class),
      # Which regime the intent-derived checks ran under: a structured order
      # lets them block, an unstructured one only lets them advise.
      gating: a.gating,
      rule: a |> primary_reason_tag() |> maybe_to_string(),
      intent: a.intent,
      confidence: a.confidence,
      comprehension: %{
        verdict: maybe_to_string(a.comprehension[:verdict]),
        composite: a.comprehension[:composite],
        gaps: a.comprehension |> Map.get(:gaps, []) |> Enum.map(&gap_text/1)
      },
      missing_slots: a.slots[:missing] || [],
      clarification_prompts: a.slots[:prompts] || [],
      capability: capability_text(a.capability),
      premises: Enum.map(a.premises, &premise_text/1),
      task_frames: %{
        events: length(a.task_frames[:events] || []),
        srl_triples: length(a.task_frames[:srl_triples] || [])
      },
      checks: Map.new(a.checks, fn {k, v} -> {to_string(k), to_string(v)} end)
    }
  end

  defp reason_text({:not_comprehended, %{verdict: v}}),
    do: "directive not comprehended (#{v})"

  defp reason_text({:not_comprehended, _}), do: "directive not comprehended"

  defp reason_text({:missing_slots, slots}),
    do: "missing required parameters: #{Enum.join(slots, ", ")}"

  defp reason_text({:contradicted_premise, %{text: text}}),
    do: "premise contradicted by existing belief: #{text}"

  defp reason_text({:contradicted_premise, _}), do: "premise contradicted by existing belief"

  defp reason_text({:incapable_action, intent}),
    do: "no capability registered for #{intent}"

  defp reason_text(other), do: inspect(other)

  defp gap_text(%{dimension: d, description: desc}), do: "#{d}: #{desc}"
  defp gap_text(other), do: inspect(other)

  defp premise_text(%{subject: s, predicate: p, object: o}), do: "#{s} #{p} #{o}"
  defp premise_text(%{text: t}), do: t
  defp premise_text(other), do: inspect(other)

  defp capability_text(:cognitive), do: "cognitive"
  defp capability_text({:external, state}), do: "external:#{state}"
  defp capability_text(other), do: inspect(other)

  defp maybe_to_string(nil), do: nil
  defp maybe_to_string(v), do: to_string(v)
end
