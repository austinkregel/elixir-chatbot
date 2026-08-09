defmodule Brain.Directive.Assessor do
  @moduledoc """
  Decides — deterministically, before any text generation runs — whether a
  directive can be carried out.

  A directive ("investigate the anomaly at sector 7") is a different kind of
  input from a conversational turn: it commits the agent to an action, so the
  question "do we understand this well enough to act, and can we act on it?"
  must be answered by machinery that can be audited, not by a model's
  self-report. Every input to this module is already computed by
  `Brain.Analysis.Pipeline`; nothing here consults a generator.

  ## The decision table

  Checks run in this order. Each is either **gating** (it can change the
  verdict) or **advisory** (it is recorded on the assessment for the report and
  the prompt, but never blocks). Among gating reasons the strongest wins:
  `:refuse` > `:clarify` > `:comply`.

  1. **Comprehension** — GATING. `Brain.Analysis.ComprehensionAssessor` scores
     eight dimensions of understanding. `:garbled` or `:opaque` → `:clarify`.
     `:partial` passes: partial understanding of an order is the normal case for
     terse command phrasing, and blocking it would make the fleet unusable.
  2. **Capability** — ADVISORY. Recorded as `:cognitive` or
     `{:external, :capable | :incapable}`; see below.
  3. **Slots** — ADVISORY, for the same reason.
  4. **Premises** — GATING. A chunk the pipeline marked `:contradicted` against
     known facts yields `:clarify`, not `:refuse`: the agent's beliefs may be
     stale, and a subordinate that refuses orders whenever its own memory
     disagrees is worse than one that challenges them.

  ## Why nothing keyed off the classified intent may block

  Both advisory rules read the *classified* intent, and on terse command
  language that classification is not trustworthy. Measured on real Fleet
  directives through the pipeline:

      "Give a one-line readiness report."  -> reminder.create      (conf 0.75)
      "Report your status."                -> music.search         (conf 0.75)
      "scan sector"                        -> reminder.create      (conf 0.75)
      "In one sentence, report your ..."   -> smarthome.device_set (conf 0.76)

  The lattice always resolves *some* registered intent at plausible confidence,
  because the intent registry has no vocabulary for orders. Gating on that
  refuses orders for services they never mentioned.

  **Structured orders do not fix this, and it was worth measuring rather than
  assuming.** The obvious hope is that an issuer-stated `constraints` list
  grounds the finding — but constraints bound *behaviour* ("read-only"), they do
  not identify *intent*. Re-running the sweep with constraints attached still
  classified "Report your status." as `music.search`, so promoting these checks
  to gating refused four of eight real orders exactly as before. The promotion
  was implemented, measured, and reverted.

  What would actually earn these teeth, in increasing order of honesty:

    * directive-shaped intents in `priv/analysis/intent_registry.json` plus a
      speech-act mapping that resolves `command`/`request_action` to them
      (currently both map to `"unknown"`), so the classification is about orders
      rather than smart-home commands; or
    * the issuer declaring the required capability on the order itself — the
      LCARS `ORDER` carries `authority: {tools: [...]}` for exactly this reason.
      A capability check against a *declared* tool is grounded; one against a
      guessed intent is not.

  `gating` on the assessment records whether the order arrived structured, so a
  report never leaves ambiguous which kind of order was assessed — but no check
  currently keys off it.

  Comprehension is likewise a weak gate in practice: real directives score
  `:partial` (composite 0.51–0.63), and so does deliberate gibberish, because
  `:garbled` requires `structural_coherence < 0.2`. It is kept because it is
  honest and cheap; the ACK readback is the mechanism intended to give this
  question real teeth.

  ## Advisory findings

  `Assessment.advisories` holds findings that did not gate, in the same `reason`
  vocabulary as gating ones. They travel in the REPORT payload and steer
  generation, so nothing is lost by not blocking — the agent can say "I have no
  service registered for this" while still doing the cognitive work it was
  ordered to do.

  ## Deliberately not checked here

    * **Issuer credibility.** Whether an order is *lawful* — who may command
      whom — is command-structure logic and belongs to `Fleet.Authority` /
      `Fleet.Clearance`. `Brain.Epistemic.SourceAuthority` models the
      trustworthiness of *knowledge sources*; using it to judge an officer
      would conflate two unrelated notions of trust.
    * **JTMS node consultation.** There is no mapping from directive text to
      JTMS nodes, and inventing one by string-matching would be determinism in
      appearance only. The pipeline's `epistemic_status` is the real, already
      justified signal. JTMS justification chains belong in *explaining* a
      report, not gating it (see `Brain.Response.RealizationPacket`).

  ## Known limitation

  `priv/analysis/speech_act_intent_map.json` maps `command` and
  `request_action` to `"unknown"`, and an unknown intent has no slot schema —
  so the slot check only bites for directives the lattice classifies into a
  registered intent (`smarthome.device_set`, `reminder.create`, …). Free-form
  orders are governed by the comprehension check alone. Widening that mapping
  affects the chat path too and is deliberately out of scope here.
  """

  alias Brain.Analysis.{ChunkPriority, ComprehensionAssessor, InternalModel, SlotDetector}
  alias Brain.Directive.Assessment

  require Logger

  @doc """
  Assess an analyzed directive.

  Takes the `Brain.Analysis.InternalModel` the pipeline already produced (so
  the directive is analyzed exactly once per turn) and returns a
  `Brain.Directive.Assessment`.

  Options:

    * `:world_id` — the mind-world whose beliefs premise checks consult.
    * `:directive_constraints` — bounds the issuer stated on the order. Carried
      onto the assessment and into the realization prompt as binding. They do
      *not* promote intent-derived checks to gating (measured; see moduledoc).
    * `:directive_risk_class` — `:routine | :sensitive | :irreversible`, carried
      onto the assessment for the report.
    * `:comprehension_assessor` — process name override, for tests.

  Every backing service is optional: if one is not running, its check is
  recorded as `:unavailable` in the assessment rather than silently passing.
  """
  @spec assess(InternalModel.t(), keyword()) :: Assessment.t()
  def assess(%InternalModel{} = model, opts \\ []) do
    primary = ChunkPriority.select_primary(model.analyses || [])
    constraints = opts |> Keyword.get(:directive_constraints, []) |> List.wrap()

    %Assessment{
      intent: primary.intent,
      confidence: primary.confidence || 0.0,
      task_frames: task_frames(model),
      constraints: constraints,
      risk_class: Keyword.get(opts, :directive_risk_class, :routine),
      # Whether the order arrived structured. Reported, not acted on — see the
      # moduledoc on why stated constraints do not make a guessed intent safe
      # to gate on.
      gating: constraints != []
    }
    |> check_comprehension(model, opts)
    |> check_capability(primary, opts)
    |> check_slots(primary)
    |> check_premises(model)
    |> resolve_verdict()
  end

  # ── 1. Comprehension ──────────────────────────────────────────────────────

  defp check_comprehension(%Assessment{} = a, %InternalModel{analyses: analyses}, opts)
       when is_list(analyses) and analyses != [] do
    name = Keyword.get(opts, :comprehension_assessor, ComprehensionAssessor)

    if comprehension_available?(name) do
      profile = ComprehensionAssessor.assess(analyses, name)

      a = %{
        a
        | comprehension: %{
            verdict: profile.verdict,
            composite: profile.composite_score,
            gaps: profile.gaps || []
          },
          checks: Map.put(a.checks, :comprehension, :ran)
      }

      if profile.verdict in [:garbled, :opaque] do
        add_reason(a, {:not_comprehended, %{verdict: profile.verdict, gaps: profile.gaps || []}})
      else
        a
      end
    else
      %{a | checks: Map.put(a.checks, :comprehension, :unavailable)}
    end
  rescue
    e ->
      Logger.warning("Directive.Assessor: comprehension check failed: #{Exception.message(e)}")
      %{a | checks: Map.put(a.checks, :comprehension, :unavailable)}
  catch
    :exit, _ -> %{a | checks: Map.put(a.checks, :comprehension, :unavailable)}
  end

  defp check_comprehension(%Assessment{} = a, _model, _opts),
    do: %{a | checks: Map.put(a.checks, :comprehension, :skipped)}

  defp comprehension_available?(name) do
    is_pid(Process.whereis(name)) and ComprehensionAssessor.ready?(name)
  end

  # ── 2. Capability ─────────────────────────────────────────────────────────
  #
  # Scoped by find_service/1 on purpose: action_capability/2 returns :incapable
  # for every intent without a registered service, and only three services are
  # registered — so an unscoped check would refuse essentially every order.

  defp check_capability(%Assessment{} = a, %{intent: intent}, opts) when is_binary(intent) do
    case find_service(intent) do
      nil ->
        %{a | capability: :cognitive, checks: Map.put(a.checks, :capability, :ran)}

      _service ->
        # Dispatcher scopes credential lookup by :world, not :world_id.
        cap_opts = [world: Keyword.get(opts, :world_id, "default")]

        case Brain.Services.Dispatcher.action_capability(intent, cap_opts) do
          :capable ->
            %{
              a
              | capability: {:external, :capable},
                checks: Map.put(a.checks, :capability, :ran)
            }

          :incapable ->
            %{
              a
              | capability: {:external, :incapable},
                checks: Map.put(a.checks, :capability, :ran)
            }
            |> add_intent_finding({:incapable_action, intent})
        end
    end
  rescue
    e ->
      Logger.warning("Directive.Assessor: capability check failed: #{Exception.message(e)}")
      %{a | checks: Map.put(a.checks, :capability, :unavailable)}
  catch
    :exit, _ -> %{a | checks: Map.put(a.checks, :capability, :unavailable)}
  end

  defp check_capability(%Assessment{} = a, _primary, _opts),
    do: %{a | capability: :cognitive, checks: Map.put(a.checks, :capability, :ran)}

  defp find_service(intent), do: Brain.Services.Dispatcher.find_service(intent)

  # ── 3. Slots ──────────────────────────────────────────────────────────────

  defp check_slots(%Assessment{} = a, %{slots: %{missing_required: missing} = slots} = primary)
       when is_list(missing) and missing != [] do
    intent = primary.intent

    prompts =
      if is_binary(intent), do: SlotDetector.get_clarification_prompts(missing, intent), else: []

    %{
      a
      | slots: %{
          filled: Map.get(slots, :filled_slots, %{}),
          missing: missing,
          prompts: prompts
        },
        checks: Map.put(a.checks, :slots, :ran)
    }
    |> add_intent_finding({:missing_slots, missing})
  end

  defp check_slots(%Assessment{} = a, %{slots: %{filled_slots: filled}}),
    do: %{
      a
      | slots: %{filled: filled, missing: [], prompts: []},
        checks: Map.put(a.checks, :slots, :ran)
    }

  defp check_slots(%Assessment{} = a, _primary),
    do: %{a | checks: Map.put(a.checks, :slots, :skipped)}

  # ── 4. Premises ───────────────────────────────────────────────────────────
  #
  # epistemic_status is computed during the pipeline's pass-2 fact verification
  # against the BeliefStore/FactDatabase, so this reads an existing justified
  # signal rather than re-deriving one.

  defp check_premises(%Assessment{} = a, %InternalModel{analyses: analyses})
       when is_list(analyses) do
    contradicted = Enum.filter(analyses, &contradicted?/1)

    a = %{a | checks: Map.put(a.checks, :premises, :ran)}

    case contradicted do
      [] ->
        a

      chunks ->
        premises =
          Enum.map(chunks, fn c ->
            %{text: c.text, beliefs: c.related_beliefs || []}
          end)

        %{a | premises: premises}
        |> add_reason({:contradicted_premise, hd(premises)})
    end
  end

  defp check_premises(%Assessment{} = a, _model),
    do: %{a | checks: Map.put(a.checks, :premises, :skipped)}

  defp contradicted?(%{epistemic_status: :contradicted}), do: true
  defp contradicted?(%{epistemic_status: "contradicted"}), do: true
  defp contradicted?(_), do: false

  # ── Verdict resolution ────────────────────────────────────────────────────

  defp add_reason(%Assessment{reasons: reasons} = a, reason),
    do: %{a | reasons: reasons ++ [reason]}

  # A finding derived from the *classified* intent is always advisory. See the
  # moduledoc: structured orders do not make the classification trustworthy, so
  # nothing keyed off it may block.
  defp add_intent_finding(%Assessment{} = a, finding), do: add_advisory(a, finding)

  defp add_advisory(%Assessment{advisories: advisories} = a, advisory),
    do: %{a | advisories: advisories ++ [advisory]}

  # Only gating reasons reach here; advisories are carried separately and never
  # influence the verdict. Reasons are recorded in check order and re-sorted so
  # the most severe leads (primary_reason_tag/1 and explain/1 read the head).
  defp resolve_verdict(%Assessment{reasons: []} = a), do: %{a | verdict: :comply}

  defp resolve_verdict(%Assessment{reasons: reasons} = a) do
    sorted = Enum.sort_by(reasons, &severity_rank/1)
    verdict = sorted |> hd() |> severity()

    %{a | reasons: sorted, verdict: verdict}
  end

  # No gating check currently produces :refuse. The severity table keeps the
  # distinction real for the day a trustworthy refusal signal exists (see the
  # moduledoc on what capability gating would need).
  defp severity({:incapable_action, _}), do: :refuse
  defp severity(_), do: :clarify

  defp severity_rank(reason), do: if(severity(reason) == :refuse, do: 0, else: 1)

  # ── Task frames (pass-through data, never a gate) ─────────────────────────

  defp task_frames(%InternalModel{analyses: analyses}) when is_list(analyses) do
    %{
      events: analyses |> Enum.flat_map(&(&1.event_frames || [])),
      srl_triples: analyses |> Enum.flat_map(&(&1.srl_frames || []))
    }
  end

  defp task_frames(_), do: %{events: [], srl_triples: []}
end
