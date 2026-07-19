# Fleet — Module Inventory & Doc/Code Consistency Check

> **Status:** point-in-time audit, generated 2026-07-04, using the same
> methodology as [`docs/BRAIN.md`](../BRAIN.md): deep-read every module,
> cross-check its own `@moduledoc`/`@doc` against its implementation, and
> additionally cross-check against [`docs/FLEET.md`](../FLEET.md)'s explicit
> ✅ built / 🔨 in progress / 📋 planned / ❓ open-question status markers,
> since `FLEET.md` is a living design doc making specific build-status claims
> BRAIN.md's subsystems don't have an equivalent of. Also checked for the
> hardcoded-stub-masquerading-as-real-signal pattern documented in
> [`hardcoded_stubs_audit.md`](hardcoded_stubs_audit.md), since that's now an
> established project concern.

`apps/fleet/lib/fleet` has **20 modules, ~2,637 lines, 100% `@moduledoc`
coverage**, exercised by 13 test files (`command_protocol_test.exs`,
`command_security_test.exs`, `tool_gate_test.exs`, `crew_billets_test.exs`,
`hail_test.exs`, `mind_world_isolation_test.exs`, `persisted_self_test.exs`,
`sharing_test.exs`, and per-module unit tests). Small enough to be read in
two batches rather than needing the ten-way split `BRAIN.md` required.

## Headline finding

**Fleet is materially cleaner than Brain on both axes this audit checks.**

- **Zero hardcoded-stub-masquerading violations found across all 20
  modules** — no function was found returning a constant while being
  presented as a live signal. The closest candidates (`Appraisal`'s
  `deference` default of `0.5`, `Rank`'s static billet→authority table) are
  both *legitimate* uses of a fixed value — a documented fallback default and
  intentionally-declared data, respectively, not a stand-in for something
  that's supposed to be computed.
- **`FLEET.md`'s ✅ status markers are accurate.** Every module checked
  against a specific ✅ claim in `FLEET.md` matched the real implementation,
  function-for-function in most cases (`Fleet`, `Application`, `Order`,
  `Authority`, `Rank`, `Appraisal`, `Dispatcher`, `Ensign`, `CrewSupervisor`,
  `CommandGraph`, `Audit`, `Comms`, `DataFrame`, `DutyLog`, `MindWorld`,
  `Proposal`, `Service`, `Signal`, `Tool` all confirmed). This is a real
  contrast with Brain's docs, which had drifted in several places (`docs/BRAIN.md`
  §Executive Summary #3-4). The one genuine attribution drift found
  (`Telemetry`, below) is minor and doesn't misrepresent what's actually
  built, only *which file* it lives in.

That said, three things are worth attention:

1. **One scope-discipline violation.** `Fleet.Service.record_achievement/2`
   is fully built and wired into the `ServiceSummary` projection — but
   `FLEET.md` §4.4 explicitly marks promotions/achievements as ❓ ("deliberately
   undecided... do not build it until that moment"). This is the one place
   code ran ahead of the doc's own stated policy rather than the doc running
   ahead of the code. It's currently dead code (zero callers), so nothing is
   broken, but it's worth a decision: either retroactively bless it in
   `FLEET.md` or pull it until the promotions design actually happens.
2. **Two soft tensions with the "no graceful degradation" doctrine**
   (`.cursorrules` Rule 5 / `[[no-graceful-degradation]]`) — both logged, so
   not *silent*, but both are a fallback-on-failure rather than a surfaced
   error: `Appraisal.cognition_verdict/3`'s `else` branch defers to a
   deference-based heuristic if `Brain.create_conversation` fails, and
   `Ensign`'s catch-all `handle_info` clause discards unrecognized messages
   with no logging at all (the one of the two that's actually silent).
3. **A cluster of dead public functions** — 11 across the whole app (listed
   below). None are hardcoded-stub concerns; they're unused facade/helper
   surface area, most commonly a wrapper that nothing calls because callers
   go one layer deeper instead (e.g. `Fleet.sitrep/2` wraps `Ensign.sitrep/2`,
   but nothing calls the wrapper).

---

## Dead code across the app

| Function | File:line | Note |
|---|---|---|
| `Fleet.list/0` | fleet.ex:127 | Zero callers, including Fleet's own tests |
| `Fleet.ready?/1` | fleet.ex:130 | Zero callers; tests call `Ensign.ready?/1` directly |
| `Fleet.sitrep/2` | fleet.ex:91 | Zero callers outside its own wrapped `Ensign.sitrep/1` call |
| `Order.valid_statuses/0` | order.ex:54 | Zero callers anywhere |
| `Rank.known?/1` | rank.ex:66-67 | Zero callers, not even in `rank_test.exs` |
| `Signal.kinds/0` | signal.ex:38 | Zero callers — every caller uses a literal atom instead |
| `Telemetry.ensign_event/0` | telemetry.ex:20 | Zero callers — callers hardcode the event name literal instead |
| `CommandGraph.graph/0` | command_graph.ex:39 | Zero callers, despite its own doc comment claiming it's "for tests/queries" |
| `Service.record_achievement/2` | service.ex:95-99 | Zero callers — also the scope-discipline item above |
| `MindWorld.mind_world?/1` | mind_world.ex | No caller in `lib/`; not directly exercised by tests either |
| `MindWorld.soul_id/1` | mind_world.ex | Test-only (`mind_world_isolation_test.exs`), no `lib/` caller |

None of these are urgent — they're candidates for deletion during a cleanup
pass, not correctness bugs.

---

## Module catalog

### Core command protocol

#### Fleet (facade)
**File:** fleet.ex
**Purpose:** The Admiral-facing public entry point wrapping `CrewSupervisor`, `Ensign`, `Comms`, `CommandGraph` — commission, chain-wiring, order/sitrep/hail dispatch, roster snapshot.
**FLEET.md check:** Matches §7 Phase 1/2/4 claims function-for-function.
**Notable:** `list/0`, `ready?/1`, `sitrep/2` are dead (see table above). Otherwise a thin, honest delegation layer with no hardcoded-signal issues.

#### Fleet.Application
**File:** application.ex
**Purpose:** OTP application entry — `Registry` (unique keys) → `TaskSupervisor` → `CrewSupervisor`, telemetry attached post-boot, mirroring `Brain.Application`'s pattern.
**FLEET.md check:** Matches §7 Phase 1 exactly, line for line.
**Notable:** None.

#### Fleet.Order
**File:** order.ex
**Purpose:** The ORDER struct (transient message + standing-assignment record), shape deliberately mirroring `Atlas.Schemas.ResearchGoal` for later persistence; status transitions validated against a whitelist, raising on illegal values.
**FLEET.md check:** Matches §7 Phase 1.
**Notable:** `valid_statuses/0` is dead. A doc comment ("sender authentication... deferred to Phase 2") is stale — Phase 2 sender authentication is now live in `Comms`/`Ensign`, but this comment in `order.ex` wasn't updated to reflect that it shipped.

#### Fleet.Authority
**File:** authority.ex
**Purpose:** The authority vocabulary (atoms/tuples: `:cognition`, `{:world, id}`, `:issue_orders`, `:relieve`, `:veto`, etc.) plus set operations — normalization, membership, per-order requirement derivation, delegation-capping (`grantable?/2`: a CO can only grant what it itself holds), and a stable string codec for persisting grants.
**FLEET.md check:** Matches §7 Phase 2's "built for real, no stubs" claim — verified `grantable?/2` is a genuine dynamic MapSet check, not a stub.
**Notable:** None. One of the most solidly verified modules in the app.

#### Fleet.Rank
**File:** rank.ex
**Purpose:** Billet (`:ensign`/`:lieutenant`/`:security`/`:executive_officer`/`:captain`) → standing-authority map — the sole source of an agent's *standing* (not per-order) authorities, conferred at commission and re-derived on rehydrate.
**FLEET.md check:** Matches §7 "Phase 5 precursor" exactly, authority-list-for-authority-list.
**Notable:** `known?/1` is dead. The static table is intentionally-declared data per FLEET.md §3.5 ("rank is data, never code structures") — not a stub.

#### Fleet.Appraisal
**File:** appraisal.ex
**Purpose:** Value-grounded DISSENT judgment run inside the cognition Task, after ACK and before `Brain.evaluate`. Tier 1: deterministic checks (data-provenance smuggling, soul-genome prohibited-terms/speech-acts). Tier 2 (opt-in): a real `Brain.evaluate/3` call parsing a leading `PROCEED`/`DISSENT:` token, defaulting conservatively by the soul's `deference` value on ambiguity.
**FLEET.md check:** Matches §7 Phase 2 exactly — Tier 2 genuinely calls `Brain.evaluate/3`, not a faked verdict.
**Notable:** `default_on_ambiguity/2`'s hardcoded `0.5` deference default is a legitimate fallback (fires only when the soul's own authored data is absent), not a stub-masquerading violation. The `cognition_verdict/3` `else` branch (falls back to the heuristic default if `Brain.create_conversation` fails, logged via `Logger.warning` but not surfaced as an error) is the one soft "graceful degradation" tension flagged above.

#### Fleet.Dispatcher
**File:** dispatcher.ex
**Purpose:** The tool-proposal authorization gate. `decide/2` is pure — reads only the caller's real harness-supplied grant `MapSet` and the code-owned `Tool` registry, never the proposal payload; default-deny on unknown tools. `dispatch/2` wraps it with a four-stage audit trail (thought → request → decision → effect) and DATA-frames results.
**FLEET.md check:** Matches §7 "Tools/MCP — propose-not-dispatch" exactly, line-for-line for the audit-kind sequence.
**Notable:** None — one of the most carefully-built, security-critical modules; no hardcoded-stub concerns.

#### Fleet.Ensign
**File:** ensign.ex
**Purpose:** The one-GenServer-per-agent runtime spine — soul identity, chain-of-command state, duty status, current assignment, full command protocol (ORDER acceptance with sender attribution + bounded-topology checks, grant-gated autonomous dispatch, REQUEST/GRANT/DENY, RELIEVE/REINSTATE with in-flight task cancellation, detached HAIL, tool-proposal gate). Cognition runs in a supervised non-linked Task so the mailbox never blocks. Rehydrates durable state from `Fleet.Service` on restart; incoming communication is written into the agent's own mind-world as a memory episode with attributed provenance.
**FLEET.md check:** Matches §7 Phase 1/2/3 claims (duty state machine, grant-enforcement gate, per-message attribution, mind-world cognition, rehydration, HAIL) verified true against the actual line ranges.
**Notable:** No hardcoded-stub issues — grant checks, appraisal, dispatch are all genuinely dynamic. The `handle_info(_msg, state)` catch-all silently discards unrecognized messages with zero logging — the one fully-silent instance flagged above. A `{:ack, ack}` raw-tuple handler is marked in a comment as a "backward-compatible" Phase-1 vestige but is still reachable (not dead).

#### Fleet.CrewSupervisor
**File:** crew_supervisor.ex
**Purpose:** `DynamicSupervisor` owning ensign process lifecycle only (the command hierarchy is data the ensign carries, not supervision structure). `start_ensign/1` commissions with a stable soul-keyed `agent_id`, making re-commissioning idempotent.
**FLEET.md check:** Matches §7 Phase 1/3 exactly, including the idempotent-rehydration claim.
**Notable:** None.

#### Fleet.CommandGraph
**File:** command_graph.ex
**Purpose:** Persists the chain of command as a real `Agent -[COMMANDS]-> Agent` edge in Apache AGE via `Brain.AtlasIntegration.sync/1`; write failures genuinely propagate as errors to the caller (no swallowed failure path).
**FLEET.md check:** Matches §7 Phase 2 exactly.
**Notable:** `graph/0` is dead, despite its own doc comment claiming it exists "for tests/queries."

### Comms & tooling

#### Fleet.Audit
**File:** audit.ex
**Purpose:** Runtime audit writer for the command channel — normalizes attrs, inserts a durable `Atlas.Schemas.CommandRecord` row *before* emitting the live telemetry event (deliberately ordered), raises loudly on DB write failure.
**FLEET.md check:** Matches §7 Phase 2 exactly, including the durable-then-telemetry ordering and fail-loud behavior.
**Notable:** None — confirmed real, ~20 real call sites in `ensign.ex`.

#### Fleet.Comms
**File:** comms.ex
**Purpose:** The single send-path and sender-authentication layer. Every outbound message stamps the real `self()` at the call site (never accepts a caller-supplied pid); `attribute/1` resolves a sender pid via `Registry.keys/2` — the Registry vouches for identity, not the payload. Bounded-topology predicates (`from_co?/2`, `from_admiral_root?/2`, etc.) check against live `context_tags`, never anything the message itself claims.
**FLEET.md check:** Matches §7 Phase 2 and the HAIL addition exactly.
**Notable:** None — all ten public functions have live callers.

#### Fleet.DataFrame
**File:** data_frame.ex
**Purpose:** Frames tool results as inert `<data>` for the model; flags likely prompt-injection content via a conservative 7-pattern regex heuristic that computes a real boolean from actual matches.
**FLEET.md check:** Matches; moduledoc is honest about the heuristic's limits rather than overclaiming certainty.
**Notable:** None — a genuine signal, not a stub, and it says so itself.

#### Fleet.DutyLog
**File:** duty_log.ex
**Purpose:** The agent's own append-only journal (distinct from the runtime-written `Service`). `note/3` always tagged with the agent's own `soul_id`, never a payload value; raises loudly on Atlas failure.
**FLEET.md check:** Matches §7 Phase 3 exactly.
**Notable:** None.

#### Fleet.MindWorld
**File:** mind_world.ex
**Purpose:** Pure `mind:<soul_id>` world-id convention scoping an agent's private memory/beliefs/JTMS — three one-line functions, each the correct inverse/check of the others.
**FLEET.md check:** Matches §7 Phase 3 exactly.
**Notable:** `mind_world?/1` and `soul_id/1` are effectively dead in production code (see table above).

#### Fleet.Proposal
**File:** proposal.ex
**Purpose:** Parses a model's fenced ` ```propose {...}``` ` block into a typed struct requiring both `tool` and `requirement` — untethered proposals (no requirement) are rejected before ever reaching the authority gate; the struct structurally cannot carry an authority claim.
**FLEET.md check:** Matches §7 Tools/MCP precisely, including exact rejection semantics.
**Notable:** None — thoroughly exercised by `tool_gate_test.exs`.

#### Fleet.Service
**File:** service.ex
**Purpose:** Runtime-only (never agent-written) accountable career record — append-only `ServiceRecord` milestones plus a per-soul `ServiceSummary` projection, always writing the append-only record before the projection; raises loudly on Atlas failure.
**FLEET.md check:** Matches §7 Phase 3 for the milestones it lists — **except** `record_achievement/2`, which is built despite `FLEET.md` §4.4 explicitly saying not to (the scope-discipline item above).
**Notable:** `bump/3` reads `summary(soul_id)` twice in the same expression — an efficiency nit, not a correctness issue (per-agent serialization makes it safe).

#### Fleet.Signal
**File:** signal.ex
**Purpose:** The envelope struct for every non-ORDER/non-ACK message. `@kinds` is the closed list of 8 kinds; `new/2` guards construction against it and stamps `issued_at`. `from_pid` is populated later by `Comms`'s send-path, not by `new/2` — correctly matching the documented separation.
**FLEET.md check:** Matches §7 Phase 2 exactly — the 8 kinds line up one-for-one.
**Notable:** `kinds/0` is dead — callers construct with literal atoms instead of validating against it.

#### Fleet.Telemetry
**File:** telemetry.ex
**Purpose:** Thin `:telemetry.execute/3` wrapper for the ensign lifecycle event, following `Brain.Telemetry`'s pattern; the attached handler debug-logs and best-effort-bridges to the `"fleet:events"` PubSub topic behind the Fleet page's activity feed.
**FLEET.md check:** The Phase-4 Activity-feed claim matches exactly. **The one genuine misattribution found in this audit:** `FLEET.md`'s Phase-1 bullet for this file claims "readiness reuses `Brain.Metrics.Aggregator.record_readiness(:ensign, _)`" is implemented *in this module* — it's actually a private helper in `ensign.ex:863-866`. The underlying claim (readiness reuse, guarded, zero brain edits) is true; the doc just points at the wrong file.
**Notable:** `ensign_event/0` is dead — callers hardcode the event-name literal instead of calling it back.

#### Fleet.Tool
**File:** tool.ex
**Purpose:** The code-owned tool registry and struct. `registry/0`'s hardcoded map is *intentional* — the moduledoc is explicit that a human registers tools here and the model never can. The one registered tool, `"beliefs.read"`, has a real handler querying `Brain.Epistemic.BeliefStore.query_beliefs/1` for genuine per-world data, not a canned response.
**FLEET.md check:** Matches §7 Tools/MCP exactly — registry literally contains exactly the one entry claimed.
**Notable:** None — the static registry is a documented design choice, not a stub.

---

## Coverage note

All 20 `.ex` files under `apps/fleet/lib/fleet` were read in full and
cross-checked against both their own documentation and `docs/FLEET.md`'s
specific status claims. No files were skipped or sampled.
