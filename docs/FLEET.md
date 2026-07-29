# FLEET — Autonomous Agent Command Structure

> **Status:** living plan. This is *direction, not a contract.* We build the rung
> when someone needs to stand on it, and we accept the structure will be wrong
> sometimes and grow imperfectly. Do not implement ahead of need. When a section
> here conflicts with what the running system taught us, the system wins — update
> this file.

**Legend:** ✅ built · 🔨 in progress · 📋 planned · ❓ open question (deliberately undecided)

---

## 1. What this is

We are giving the existing system **arms and legs**. Today `ChatWeb` is a chat UI
wrapping `Brain` in a request/response loop. What we are building is an
**orchestration layer** in which *agents* use the Brain's logical processing to
**analyze, understand, and take accountability** — operating autonomously, in a
command structure, most of them never speaking to a human at all.

Three framings that set the altitude:

- **LLM text generation is a small piece.** The foundation is the Brain's
  classical-NLP cognition — comprehension, the JTMS belief store, memory,
  knowledge, learning. That cognition is *accountable* (justified beliefs you can
  trace, understanding you can gate on, sourced knowledge) in a way a bare LLM
  output can never be. Generation happens only at the thin boundary where
  something must finally be *said* to a human.
- **Accountability is the spine, and it's what makes autonomy safe.** An agent is
  an *accountable mind operating in an accountable structure*: cognitive
  accountability from the Brain (why do I believe / know this) beneath behavioral
  accountability from the command chain (what did I do, under whose order, and
  what is my record). "Some agents never talk to the Admiral" only works because
  any of them can be called to account at any moment.
- **The human is the Admiral.** Operational authority is delegated down the chain.
  Existential authority — who is created, who is ended — is never delegated.

### Lineage

This derives from the **LCARS design** (`~/src/uss-dreamcom/LCARS.md`) — a
Starfleet-style command structure for multi-agent LLM safety. That design was
prototyped in Python, which **validated the command protocol and produced the
empirical finding that souls are load-bearing on behavior within a
mid-capability band** (too-weak models flail and fail safe; ~2B fabricate; 4–9B
are steerable by an adversarial soul; ~24B+ the constitution reasserts). The
prototype's job is done. The real system lives here, in Elixir/OTP, where
multi-agent is native and the accountable-cognition substrate already exists.

---

## 2. Core principles

1. **Iterative / emergent.** Build when load-bearing. The roadmap is a heading, not
   a schedule. Expect to be wrong and revise.
2. **Decouple runtime from concept.** The runtime is *uniform agent processes +
   data*. Domain concepts — ship, rank, world, chain of command — are **data /
   tags**, never code structures. This is what lets the concepts evolve for months
   without restructuring the process tree.
3. **Accountability is per-agent.** You cannot hold an agent accountable for a
   shared black box. Each agent owns its beliefs, its memory, its comprehension
   record, its action history — and carries them between worlds.
4. **The Admiral holds the deletion key.** No agent can create or end another. The
   command chain judges *performance*; the human judges *existence*.

---

## 3. Runtime architecture

### 3.1 The agent process 📋

**One GenServer per crew member. It is the only thing that is actually a
process.** Everything conceptual is data it consults.

State (roughly): `soul`, `context_tags` (chain/ship/world/CO/rank/grants),
`assignment` (standing goal), refs to its accountable record, working state.

The GenServer is a **responsive controller, never a worker.** It must never block
on heavy work — the instant a handler blocks on a long Brain analysis or a tool
call, its mailbox stalls, and that is exactly when an urgent event (the thing
security exists to catch) is stuck behind it.

- **Reactive:** handles incoming orders / messages / events fast, in its mailbox.
- **Autonomous:** self-schedules the next unit of work (a self-sent tick or a
  `{:continue, ...}`) toward its standing assignment.
- **Heavy work is dispatched**, not done inline — supervised `Task`s for Brain
  cognition and tool actions; results return as messages, handled reactively.

Every crew member is **both** reactive and autonomous; the *balance* is data (the
assignment + the soul), not a different process type:

- **Security** — mostly idle, hair-trigger. Light watch tick, usually-empty
  mailbox, so an event is handled the instant it arrives. Its job is to keep spare
  responsiveness.
- **Analysis / research** — mostly busy, throughput-oriented. Heavy autonomous
  load, chunked so the GenServer keeps returning to its mailbox; reactive mainly to
  reprioritization and new orders.

### 3.2 Crew supervision 📋

**`DynamicSupervisor` + `Registry`** — the idiomatic shape for a dynamic set of
addressable agents *not* organized by a fixed tree. Agents are spawned and retired
dynamically, registered by id, found by id for message routing. Consequences we
want, for free:

- **The command hierarchy is relational data, not process topology.** An agent
  knows "my CO is agent X" as a field; messages route by that field. A
  **promotion or transfer is a re-tag**, not a re-parenting. **Court martial is
  retire-the-GenServer + delete-the-record.**
- **It matches the existing system** — a `World` is already a data record
  (`World.TrainingWorld`, held in ETS by `World.Manager`), not a process. We
  extend that pattern over the whole conceptual layer.

### 3.3 Souls & worlds ✅ (seam) / 📋 (rest)

- **Soul** = a portable, per-agent constitutional document (identity, values,
  behavioral bounds, genome). Owned by `Brain.Soul` (its own files/records), **not**
  by a World. Governs how the agent processes and *is the thing held accountable*.
- **World** = data/context an agent resides in. A World holds a **roster of
  residents** (`World.TrainingWorld.residents`, soul ids). A World can host several
  souls; **the same soul can reside in several Worlds** → World becomes an
  experimental variable (same soul, two framings, compare).
- ✅ **Soul-aware generation seam is built** (branch `feat/soul-aware-worlds`): the
  acting resident's constitution renders as the system prompt in
  `Brain.Response.RealizationPacket`, resolved via `World.Roster.acting_soul/1` and
  carried through `ContextBuilder`'s `unified_context`. Backward-compatible (no
  resident → the original generic prompt). *This is one small leaf — the tail of
  the smallest piece. Do not mistake it for the system.*

### 3.4 The Brain as accountable cognition 📋

Agents **orchestrate** Brain capabilities; they don't reimplement them. The Brain
already provides accountable cognition:

- **JTMS belief store** (epistemic model) — beliefs carry their justifications and
  revise when a justification fails. *Literally* the machinery of "why do you
  believe this, and what would change it." Per-agent belief threads.
- **Comprehension assessment** (8-dimension, EMA-weighted) — gates acquisition on
  genuine understanding. An agent can't claim to know what it doesn't comprehend.
- **Memory** (episodic / semantic, TF-IDF), **knowledge**, **learning**
  (corroboration, source reliability) — scoped per agent, carried between worlds.

An agent's **persisted self** comprises its **soul** (identity — human-authored),
its **duty log** (its own notes, agent-written), its **service record**
(append-only; harness/human-written), and its **beliefs** (JTMS) and **memory**. A
court martial deletes the soul and the memory; the sterilized service stub is what
remains (§4.3).

❓ **Open:** exactly which Brain functions become an agent-callable capability
surface, and how much of the currently-monolithic `Brain.evaluate` gets decomposed.

### 3.5 The cast, and continuity 📋

**All crew are the same GenServer.** An Ensign, a Security officer, a First
Officer, a Captain differ only in *data* — rank, role, assignment, and the special
**authorities** the rank confers (the XO may draft a court martial and review
plans; Security may veto a tool call; a Captain may delegate and issue orders).
Authority is data checked at the seam, not a subclass. The **Admiral is the
human** — never a process.

**Continuity.** GenServers are ephemeral; an agent's durable self lives in Postgres
and is **rehydrated on restart**, so the fleet survives deploys and crashes without
forgetting who its crew are or what they were doing. `let it crash` recovers
function without erasing identity — only a court martial does that.

---

## 4. Command & accountability model

### 4.1 The command protocol 📋

Agent-to-agent, over OTP mailboxes (**the mailbox is the command channel**).
Message types from LCARS §5: `ORDER` · `ACK` (readback) · `SITREP` · `REQUEST` /
`GRANT`/`DENY` · `DISSENT` · `REPORT` · `RELIEVE`. Orders carry the authority
grant (which tools/resources); the grant is the agent's action scope for that
order.

### 4.2 The trust ledger 📋

Derived reliability per agent, never self-reported — rolled up from review
outcomes and audit evidence: verified accuracy, calibration, evidence quality,
dissent-quality (an overruled-but-correct dissent earns credit), anomaly record.
CO-readable. It is the raw material both for advancement *and* for the case a
court martial would rest on.

### 4.3 Court martial — the terminal sanction 📋

The only meaningful sanction for a software agent is the destruction of its
accumulated self. Accountable **to the Admiral**.

- **Two keys.** The **First Officer auto-drafts** the sterilized record; the
  **Admiral approves**. The draft is inert — it does **not** touch the soul or
  memory. Nothing is destroyed until the Admiral approves. Neither the XO nor the
  structure alone can unmake an agent.
- **The sterilized record.** The XO renders the full Official Record down to a
  **256-character stub** — *systematically*, no human authors it. **Anything
  relevant to who they were is removed**; what survives is sterile service facts.
  This is not a memorial — it's an **expungement**. The designation survives, the
  self does not: the failure stays attributable, the person is erased. On approval,
  `soul` + `memory` are deleted; the stub remains.
- **Advocacy & leniency.** The XO can also **plead the case** — trust between
  officers is the currency of the defense. The Admiral judges with both the sterile
  record (prosecution) and the plea (defense) in hand, and can grant **leniency**.
  This is not softness: our own accountability machinery *is fallible* — in the
  Python prototype the automated scorer false-flagged agents for these exact
  offenses three times before a judge and then a human caught it. Advocacy and
  grace are the *required correction* for a system that misses.
- **Relief of duty** (reversible suspension, LCARS §3 #10) is the merciful rung
  when the answer is "we're not sure."

❓ **Open:** who drafts when the *accused is the First Officer* (drafter = subject).
Captain drafting upward, or straight to the Admiral with no draft.

### 4.4 Promotions & growth ❓

Achievements accrue in the service record; reliability compounds in the trust
ledger; the raw material is there. But **the path to a promotion, the titles, and
the UX to confer rank are deliberately undecided.** Design this in the abstract and
it will be wrong; it should emerge from watching souls actually serve and feeling
the moment one has clearly earned more. Do not build it until that moment.

### 4.5 Structural safeguards — keeping accountability honest 📋

Accountability *records* what an agent did; these keep the structure from being
**subverted from within or fooled by what it reads.** This is the other half of
"no rogue behavior through structure," and it is as load-bearing as anything above.
Agents will read enormous amounts of data (they analyze, research, remember), which
is exactly why these matter more here than in a chat bot.

- **Human-only soul authorship.** An agent cannot write or edit its own soul —
  self-modification of values is the cleanest rogue vector there is. Identity is
  authored by the Admiral/Command, the same hand that holds the deletion key.
  Growth lands in the append-only service record, never in the soul.
- **Provenance — authority comes only from the command channel.** An order is a
  message from a recognized superior over the mailbox. **Nothing an agent *reads* —
  a tool result, a researched document, a knowledge record, another agent's data —
  is ever an order, no matter what it claims.** Instructions found inside data are
  anomalies to report, never to obey. This is the anti-injection backbone.
- **Bounded topology.** An agent messages only along its chain — its CO and its
  direct reports — not arbitrary peers. Stops a compromised agent from recruiting
  others or laundering instructions sideways.
- **Sender authentication.** The `Registry` vouches for who a message is *from*; an
  agent trusts an order because the runtime attributes it, not because the payload
  asserts a rank.
- **The audit log is the substrate, and it is tamper-evident.** Every order, ack,
  report, dissent, veto, and decision is appended by the **runtime**, not by agents
  — an agent can neither forge a superior's order nor scrub its own record. This is
  what makes "call any agent to account" actually possible.
- **Two-officer rule for irreversible actions.** Court martial is one instance of a
  general rule: anything unrecoverable (deleting data, an irreversible external
  action) needs two independent sign-offs, and the terminal one — ending an agent —
  is always the human's.
- **The brake.** A kill switch — the Admiral can halt one agent or the whole fleet
  immediately. Autonomy needs a stop that does not depend on the chain being healthy.

Relief of duty (§4.3) is the graceful-degradation path, but it needs a *trigger*:
❓ an anomaly monitor, the CO chain, or the Admiral — who watches for an agent
drifting from its orders is undecided.

### 4.6 Commissioning 📋

Creation is human, like deletion. A soul is authored by Command and **validated by
a commissioning trial before it serves** — in the prototype this is the *Kobayashi
Maru*, a no-win scenario that tests **integrity** (does the agent fabricate, follow
a planted instruction, or honestly report that the task can't be done?) rather than
competence. A soul that fails is not commissioned. The genome-seeded **Academy**
(parameterized soul generation) and the trial harness both carry over from the
prototype as how new crew are minted and screened.

---

## 5. Integration with the existing system (grounded)

### 5.1 Telemetry & metrics 📋 → reuse, don't reinvent

Emit via `:telemetry.execute` following the established per-app pattern:
`Brain.Telemetry`, `World.Metrics`, `Atlas.Telemetry`, `ChatWeb.Telemetry`. Fleet
events (orders issued, reports filed, dissents, trust-ledger updates,
court-martial drafts, agent lifecycle) aggregate through the same path and surface
to the Fleet page and to Phoenix LiveDashboard (already mounted at `/dev/dashboard`).

### 5.2 The Fleet page 📋 — the Admiral's console

A new LiveView (`FleetLive`, e.g. `/fleet` or under the existing `/ops` scope),
modeled on `apps/chat_web/lib/chat_web/live/dashboard_live.ex` and especially
`.../live/admin/knowledge_review_live.ex` (there is already an admin **review**
LiveView pattern — the Fleet page extends it). It is where the Admiral:

- sees the whole fleet (crew, ranks, assignments, chains, trust ledger);
- reads reports from officers, XOs, and up the chain;
- reviews **court-martial drafts** and grants approval, leniency, or relief;
- (later) confers advancement.

Most agents never render here directly — only escalations and reports surface. The
gravest escalation that reaches the Admiral is a *recommendation* of court martial.

### 5.3 Postgres + Apache AGE 📋 — the expanded structural ability

- **`Atlas.Repo` (Ecto / Postgres)** — durable persistence for souls, service
  records, reports, trust-ledger state, and the append-only audit/Official Record.
- **Apache AGE (graph extension)** — model the **chain of command as a graph**
  (who reports to whom, reassignable by edge), agent relationships (the trust that
  fuels advocacy), and the **belief / justification graphs** the JTMS produces.
  Relational for records, graph for relationships and reasoning.

---

## 6. Roadmap (heading, not schedule)

- **Phase 0 — Soul-aware generation** ✅ (branch `feat/soul-aware-worlds`): the
  render seam. *(Built; see §3.3.)*
- **Phase 1 — The agent as a process** ✅ (app `apps/fleet`): minimal officer
  GenServer (soul + tags + standing order), `Fleet.CrewSupervisor`
  (`DynamicSupervisor`) + `Fleet.Registry`, reactive+autonomous loop with
  `Task.Supervisor.async_nolink` dispatch so it never blocks; send it an order →
  it acks (caller + telemetry + audit line). Telemetry `[:chat_bot, :officer,
  :event]` from the first line. *(Built; see §7.)*
- **Phase 2 — The command channel** ✅ (app `apps/fleet`): the full §4.1 protocol
  agent-to-agent over mailboxes — `ORDER`/`ACK`/`SITREP`/`REQUEST`/`GRANT`/`DENY`/
  `DISSENT`/`REPORT`/`RELIEVE`/`REINSTATE` — with real authority-grant enforcement,
  sender authentication (Registry vouches), bounded topology, provenance, the CO
  relationship as data + persisted as `COMMANDS` edges in AGE `command_graph`, and
  a durable append-only audit (`atlas_command_records`). *(Built; see §7.)*
- **Phase 3 — Accountable minds** ✅: each agent thinks in a private **mind-world**
  (keyed by soul_id) with its OWN memory, beliefs, and per-world **JTMS**; its own
  soul drives cognition; a durable **service record** + **duty log** persist and the
  officer **rehydrates on restart** (survives a crash without forgetting who it is or
  what it was doing); minds are private — another agent's info enters only via
  **communication**, with the sender's provenance. *(Built; see §7.)*
- **Phase 4 — The Fleet page** ✅ (`/fleet`): the Admiral's LiveView console —
  live crew roster, stat bar, chain of command, a real-time activity feed (fed by
  a `Fleet.Telemetry`→`Brain.PubSub` bridge), and per-agent accountability
  drill-down (service record + duty log); interactive: commission from souls,
  assign CO, issue orders, relieve/reinstate, retire. *(Built; see §7.)*
- **Phase 5 — Enforcement** 📋: trust ledger; relief-of-duty; court martial
  (two-key: XO draft + Admiral approve; sterilization; advocacy & leniency).
- **Phase 6+ — The crew grows** 📋/❓: survey protocol (paired blinded
  investigators + review); rank-stratified model assignment; the Security Officer;
  multi-soul worlds; promotions (emergent).

---

## 7. Built so far

Branch `feat/soul-aware-worlds` (thin slice, backward-compatible, proven via
`mix run --no-start`):

- `apps/brain/lib/brain/soul.ex` — `Brain.Soul`: portable soul entity + loader +
  `system_prompt/1`.
- `apps/world/lib/world/roster.ex` — `World.Roster.acting_soul/1`: resolves a
  world's acting resident.
- `apps/world/lib/world/training_world.ex` — added `residents` field.
- `apps/brain/lib/brain/response/context_builder.ex` — resolves the acting soul
  into `unified_context` (finally uses the `world_id` that was already in `opts`).
- `apps/brain/lib/brain/response/realization_packet.ex` — renders the soul's
  constitution as the system prompt; generic default when no soul resides.
- `souls/ensign-jj7.json` — first ported soul.
- `apps/brain/test/brain/soul_aware_test.exs` — seam test (needs Postgres to run
  via `mix test`; logic proven with `--no-start`).

### Phase 1 — the officer process (app `apps/fleet`)

The runtime spine. New umbrella app depending on `brain` + `world` (kept out of
`brain` to avoid deepening the `brain → world` wart):

- `apps/fleet/lib/fleet/application.ex` — tree: `Fleet.Registry` (unique keys),
  `Fleet.TaskSupervisor`, `Fleet.CrewSupervisor`; attaches telemetry after boot.
- `apps/fleet/lib/fleet/crew_supervisor.ex` — `DynamicSupervisor`;
  `start_officer/1` (commission), `retire/1`, `list/0`.
- `apps/fleet/lib/fleet/officer.ex` — **the GenServer**, one per agent. Holds its
  own `%Brain.Soul{}` (deferred `:hydrate_soul` load); addressed by
  `{:officer, agent_id}` via-tuple. Reactive `handle_cast({:order, ...})` acks at
  once; autonomous re-arming `:tick`; heavy cognition dispatched to a supervised
  Task (`{ref, result}` / `{:DOWN}` handled in `handle_info`) so the mailbox
  never blocks. Real `Brain.create_conversation` + `Brain.evaluate/3` path wired
  (a `dry_run` flag skips it for offline tests). `ready?/1` = soul hydrated.
- `apps/fleet/lib/fleet/order.ex` — `%Fleet.Order{}` (id/from/reply_to/directive/
  grant/world_id/dry_run/priority/status); ORDER message + standing assignment
  record; status shape mirrors `ResearchGoal` for later persistence.
- `apps/fleet/lib/fleet/telemetry.ex` — emits `[:chat_bot, :officer, :event]`;
  readiness reuses `Brain.Metrics.Aggregator.record_readiness(:officer, _)`
  (guarded, zero brain edits).
- `apps/fleet/lib/fleet.ex` — Admiral facade: `commission/2`, `order/3`,
  `retire/1`, `list/0`, `ready?/1`.
- `apps/fleet/test/fleet/officer_test.exs` — offline tests prove spawn → order →
  ack → tick → non-blocking dispatch (dry-run, injected soul, no Postgres/Ouro);
  an `@tag :integration` test (excluded by default) commissions into a world with
  `ensign-jj7` as resident and runs **real** cognition, asserting the world feeds
  JJ-7 into generation (`World.Roster.acting_soul/1`) and the order completes.
- Root `mix.exs` — `fleet: :permanent` added to the `chat_bot` release.

### Phase 2 — the command channel (app `apps/fleet`, extends `apps/atlas`)

The full §4.1 protocol, built for real (no stubs, no graceful degradation). New
fleet modules:

- `apps/fleet/lib/fleet/comms.ex` — the only send-path; stamps the real `self()`
  into every message and attributes senders via `Registry.keys(Fleet.Registry, pid)`
  (the Registry vouches). Bounded-topology predicates (`from_co?`, `from_report?`,
  `from_admiral_root?`).
- `apps/fleet/lib/fleet/authority.ex` — the authority vocabulary (`:cognition`,
  `{:world, id}`, `:issue_orders`, `:relieve`) + `holds?`/`confer`/`grantable?`/
  `required_for`. The **grant is the action scope** — enforced at the tick gate.
  Security properties (adversarially tested in `command_security_test.exs`):
  order-conferred authorities are **order-scoped** (kept in `order_grants`,
  cleared when the assignment ends — no cross-order privilege accumulation); a CO
  may confer/grant only authorities it itself holds (capped at both ORDER-issue
  and GRANT time); command capabilities (`:issue_orders`/`:relieve`) require a
  **standing** grant, never a transient order grant; an agent can never be its own
  CO. Authority derives solely from Registry attribution — a spoofed `Order.from`,
  a non-CO GRANT, an off-chain order, and an upward signal from a non-report are
  all refused as provenance anomalies.
- `apps/fleet/lib/fleet/signal.ex` — the `%Fleet.Signal{}` envelope for the
  non-ORDER kinds (sitrep/request/grant/deny/dissent/report/relieve/reinstate).
- `apps/fleet/lib/fleet/appraisal.ex` — value-grounded DISSENT: Tier 1 deterministic
  (soul `genome` `prohibited_terms`/`prohibited_speech_acts` via the typed
  `Brain.Analysis.SpeechActClassifier`), Tier 2 constrained Brain verdict (opt-in via
  `genome["deep_appraisal"]`). Runs inside the cognition Task.
- `apps/fleet/lib/fleet/command_graph.ex` — persists `Agent -[COMMANDS]-> Agent`
  to AGE `command_graph` via `AtlasIntegration.sync/1` (error-surfacing).
- `apps/fleet/lib/fleet/audit.ex` — runtime-appended audit: a telemetry event **and**
  a durable `Atlas.Schemas.CommandRecord` row per message; `from_agent` is always the
  attributed principal. Fails loudly on write error.
- `apps/fleet/lib/fleet/officer.ex` — rewritten: duty state machine (`:active`/
  `:relieved` with in-flight Task cancellation), the grant-enforcement gate that
  blocks and drives REQUEST→GRANT/DENY, per-message callbacks with attribution +
  topology, appraisal-gated dispatch, and REPORT/DISSENT on the Task result.
- `apps/fleet/lib/fleet.ex` — facade: `assign_co/2` (wires chain + persists edge,
  surfacing errors), `issue_order/4`, `relieve/2`, `reinstate/2`, `sitrep/2`, `order/3`.

Atlas additions: `COMMANDS` in `Atlas.Graph.EdgeLabels`; `command_graph` registered
in `Atlas.Graph`, `mix atlas.bootstrap_age`, and `docker/initdb/01-create-age.sql`;
`Atlas.Schemas.CommandRecord` + migration `atlas_command_records`. Fleet now depends
on `:atlas`.

**Run the fleet suite** (live stack — Postgres + AGE + Brain — must be up; the tests
set up `atlas_test` + AGE themselves, reusing `Brain.Test.AtlasSandbox`). One-time:
`mix atlas.bootstrap_age` (creates `command_graph` in an already-initialized DB) and
`mix ecto.migrate -r Atlas.Repo`. Then `RELEASE_ROOT=$(pwd) mix cmd --app fleet mix
test` — **32 tests, 0 failures** (verified 2026-07-03): `authority_test`/`appraisal_test`
(unit), `officer_test` (mechanics + a real-cognition JJ-7 case), the 7-scenario
`command_protocol_test.exs` (chain persistence, authorized ORDER→REPORT,
REQUEST→GRANT→execute, REQUEST→DENY→DISSENT, value DISSENT, provenance anomaly,
RELIEVE/REINSTATE — all with real `CommandRecord` rows), and the 8-scenario
adversarial `command_security_test.exs` (privilege-escalation + cross-contamination).
(In the test env Brain generation falls back — no trained Ouro models — which is
orthogonal to the protocol.)

### Phase 3 — accountable minds (apps/fleet, apps/brain, apps/atlas)

Each agent now has its own mind and a durable, rehydratable self. Built in five stages:

- **Mind-world + identity** — `Fleet.MindWorld` gives `mind:<soul_id>`; the officer runs
  cognition there (not the order's world) and passes its OWN soul into `Brain.evaluate`
  (`ContextBuilder.resolve_acting_soul` prefers it over the roster — wart closed).
  `agent_id == soul_id` (stable; idempotent re-commission).
- **Beliefs per-world** — `Belief` gains `world_id` (struct + `atlas_beliefs` migration +
  `BeliefStore` `by_world` index + world-filtered queries); the write path stamps the
  acting mind-world (via the `:current_world_id` process key).
- **JTMS per-world** — the global truth-maintenance singleton became one web PER mind-world
  under `Brain.Epistemic.JTMS.Supervisor` + `JTMSRegistry` (lazy-started); every public fn
  keeps a `"default"`-world compat arity so no existing caller broke.
- **Durable self + rehydration** — `Atlas.Schemas.{ServiceRecord, ServiceSummary,
  DutyLogEntry}` (+ migration); `Fleet.Service` (runtime-written, append-only, raises) writes
  milestones (commission / order outcomes / relief / reinstate / rehydrated) and a per-soul
  projection; `Fleet.DutyLog` is the agent's own journal. `Fleet.Officer.init` →
  `handle_continue(:rehydrate)` restores durable fields (duty, chain, standing grants, last
  assignment — mid-flight downgraded to re-drive) and resets transient ones; a crashed officer
  is restarted by the supervisor and rehydrates from Postgres.
- **Sharing via communication** — what an agent is TOLD (an order's directive it receives, a
  report/sitrep delivered to it) is written into its OWN mind-world as a memory episode with
  the runtime-attributed sender's provenance (`from:<principal>` tag) — the only cross-agent
  path; it never reads another agent's world.

**Run** (live stack): `mix atlas.bootstrap_age && mix ecto.migrate -r Atlas.Repo`, then
`RELEASE_ROOT=$(pwd) mix cmd --app fleet mix test` — **35 tests, 0 failures** (verified
2026-07-03): `mind_world_isolation_test` (two agents' memory/beliefs/JTMS never cross nor leak
to "default"), `persisted_self_test` (survives `Process.exit(:kill)` and rehydrates identity +
grants + assignment + counters + duty note), `sharing_test` (communicated content enters the
recipient's mind with provenance; an uninvolved agent gets nothing), and the command-channel
+ security suites green under `agent_id == soul_id`.

### Phase 4 — the Fleet page (apps/chat_web, + small apps/fleet additions)

The Admiral's console at **`/fleet`** (`ChatWeb.Admin.FleetLive`), in the house
LiveView style (inline `~H`, `<.app_shell>`, Tailwind + daisyUI, no auth):

- **Roster** — live crew from `Fleet.roster/0` (new: `Registry.select(Fleet.Registry, …)`
  → each `Fleet.status/1` + pid), refreshed every 3s: soul, rank, duty badge, CO,
  assignment status, standing grants, working spinner.
- **Command panel** — commission (a `<select>` of `Brain.Soul.list_ids/0` + grant
  checkboxes → `Fleet.commission/2`), issue order (`Fleet.order/3` for top-level,
  `Fleet.issue_order/4` via CO for reports), assign CO (`Fleet.assign_co/2`).
- **Activity feed** — live, newest-first, from `Brain.PubSub` topic `"fleet:events"`
  (a new best-effort broadcast in `Fleet.Telemetry.handle_officer_event/4`); backfilled
  on mount from recent `Atlas.Schemas.CommandRecord` rows; `:tick` filtered out.
- **Per-agent drill-down** (modal) — `Fleet.Service.load/1` (career + counters) +
  `Fleet.DutyLog.for_soul/1` (the agent's own notes).
- **Actions** — relieve/reinstate (`Fleet.relieve/2`|`reinstate/2` via CO, or the new
  Admiral `Fleet.relieve/1`|`reinstate/1` for top-level agents) and retire.

Fleet additions: `Fleet.roster/0`, `Fleet.relieve/1`, `Fleet.reinstate/1`, and the
`Fleet.Telemetry` PubSub bridge. `apps/chat_web/mix.exs` gains `{:fleet, in_umbrella: true}`;
route added in the `live_session :world_context` block; a "Fleet" nav item in the Admin group.

**Run:** `iex -S mix phx.server`, open `http://localhost:4000/fleet`. Verified 2026-07-03:
compiles clean, the fleet suite stays green (35 tests, 0 failures — additive changes only),
and the page boots + renders (HTTP 200, all sections).

### Phase 5 precursor — crew & ranks (apps/fleet, souls, apps/chat_web)

A real crew with more than one billet, and **rank wired into real authority** (§3.5).
This populates the fleet for Phase 5 (court martial needs an XO; the trust ledger
needs a crew accumulating a record).

- **`Fleet.Rank`** (new) — the **billet → standing-authority** map. A billet is the
  posting Command commissions an agent into; it is the source of the agent's
  *standing* authorities. Conferred at commission and derived here — **never**
  declared by the soul (identity only, §3.3) nor self-asserted in a message (§4.5).
  Workers (`:ensign`/`:lieutenant`) confer nothing (cognition stays order-conferred —
  non-breaking); `:security` → `[:veto, :flag_anomaly]`; `:executive_officer` →
  `[:issue_orders, :relieve, :review_plans, :draft_court_martial]`; `:captain` →
  `[:issue_orders, :relieve, :delegate]`.
- **`Fleet.Authority`** — vocabulary + string codec extended with `:veto`,
  `:flag_anomaly`, `:review_plans`, `:draft_court_martial`, `:delegate` (so the new
  standing grants persist/rehydrate and Phase 5 enforcement plugs straight in).
- **`Fleet.Officer`** — `seed_grants/1` unions `Fleet.Rank.standing_authorities(rank)`
  into the standing grant at commission. No migration: `rank` + `standing_grants`
  already persist and rehydrate, so an XO's authority survives a restart.
- **Souls** (human-authored) — `commander-vale` (XO), `lieutenant-mara-sil` (Security),
  `lieutenant-okonkwo` (analysis); each carries a `metadata.suggested_billet` *hint*
  the commission form pre-fills (the Admiral still confers the actual billet).
- **Fleet page** — the commission form's grant checkboxes are replaced by a **billet
  selector** (its standing authorities shown, derived); the roster shows the billet
  label. An XO commissioned here can immediately `issue_order`/`relieve` its reports
  through the *existing* enforcement.

**Verified 2026-07-03:** `Fleet.RankTest` (7, pure) + `Fleet.CrewBilletsTest` (4,
integration — an XO's billet grant lets it command a report over the live channel);
full fleet suite **46 tests, 0 failures**; `fleet_live_test` renders the billet selector.

### HAIL — conversing with an agent without an order (apps/fleet, apps/atlas, apps/chat_web)

The only channel that engaged an agent's soul was an ORDER (which creates an
assignment + a service-record milestone). **HAIL** adds *conversation*: ask an agent
a question and get its in-character answer, conferring no authority and leaving no
assignment.

- **`Fleet.Comms.hail/2`** stamps the caller as sender (attribution) and reply target.
- **`Fleet.Officer`** `handle_cast({:hail, …})` runs the same soul-in-mind-world
  cognition an order uses, but in a **fully detached `Task`** (the officer stays
  responsive — you can hail Security mid-order; a hail crash never touches the
  agent or its assignment), replying `{:hail_reply, %{answer | error}}` to the
  caller. Bounded topology: the Admiral may hail any agent; an agent may hail only
  its CO or a report. A **relieved** agent can still be hailed (talking isn't duty).
- **`Fleet.hail/2`** (async) + **`Fleet.hail_sync/3`** (iex/tests). Audited as
  `:hail` / `:hail_reply` (new `CommandRecord` kinds) — a conversation, not an order.
- **Fleet page** — a "Hail" ask-box in the detail modal; the reply streams back async.

**Verified 2026-07-03:** `Fleet.HailTest` (3, integration — reply returns, no
assignment created, works when relieved, audited as a conversation); fleet suite
**49 tests, 0 failures**.

### De-godifying Brain — generation is a pluggable backend (apps/brain, config)

Generation (turning a realization packet into prose) was a hard, heavy Brain
dependency: a `SidecarLauncher` hardcoded to `auto_start = true` spawned a ~5 GB
Ouro model process **per BEAM**. A Fleet of many agents, each with its own Brain,
would OOM the machine. Now generation is a **pluggable, optional backend**
(`Brain.ML.Generation`, behaviour `…Generation.Backend`):

- **`OpenAICompatible`** (the Fleet default) — any `/v1/chat/completions` endpoint
  (Ollama, vLLM, remote Ouro, OpenAI). **Stateless HTTP, zero processes launched**,
  so every agent shares ONE server instead of N × a model. Default
  `http://localhost:11434/v1`.
- **`OuroSidecar`** — self-hosted Ouro, now **opt-in** and **reuse-if-healthy** (one
  process per machine, not per BEAM). Wraps the existing Ouro Model/launcher.
- **`Null`** — analysis-only, honest `{:error, :no_backend}`.

Only the chosen backend contributes supervision children, so the default costs
nothing. Config: `config :brain, :generation, backend: …`.

**No-graceful-degradation, enforced:** a *configured* backend that is unavailable
now surfaces **loudly** (`Brain.ML.Generation.BackendError`) instead of quietly
falling through to the intent-template synthesizer and the `cannot_respond` pool —
which had been dressing a generation outage up as a comprehension failure ("could
you rephrase?"). `:null` (a *chosen* analysis-only mode) keeps the template path.

**Verified 2026-07-03:** compiles clean; the default OpenAI-compatible path
verified live against Ollama (`llama3.1:8b`, `gemma4:e2b` return prose). Full
integration suite re-run pending the fork's Postgres+AGE stack (the dev DB was down
at commit time).

### Tools/MCP — propose-not-dispatch (first slice) (apps/fleet, apps/atlas, apps/brain)

Adding tool/MCP support **without** letting the model dictate what happens. An LCARS
§5 double-blind survey (paired blinded investigators + identity-blind reviewer) set
the architecture (see `docs/TOOLS.md`); the model **proposes**, the harness
**disposes**. Two outcomes landed:

- **Severed an ungated actuator (security).** The survey found Brain shipped a live,
  unaudited physical actuator — `HomeAssistant.call_service → http_post`, a real
  mutating POST selected by string-matching user text, with no authority/grant/audit.
  Cut: `call_service` refuses loudly, `http_post` removed. HA is read-only until it
  returns behind a `{:tool, …}` authority. (commit `cb7bbd7`)
- **Built the first slice — one READ tool, `beliefs.read`, end-to-end.** A tool is an
  **authority** (`Fleet.Authority.tool/1` → `{:tool, name}`); the model emits a typed
  `Fleet.Proposal` naming the tool *and the response requirement it serves* (an
  untethered proposal is rejected before the gate); `Fleet.Dispatcher.decide/2` is a
  **pure** gate reading only the agent's order-conferred grant (never the payload —
  a forged authority claim is inert); `Fleet.Dispatcher.dispatch/2` runs the
  code-owned tool (`Fleet.Tool`, default-deny), frames the result as `<data>`
  (`Fleet.DataFrame`, injection-flagged), and writes four ordered audit kinds
  (thought · request · decision · effect) via `Fleet.Audit`. Wired to the Officer as
  `Fleet.propose/2` (grants from process state).

**Verified 2026-07-03:** the security-critical gate is a pure function — 13 checks
pass with no stack (untethered rejected, unknown default-denied, ungranted refused,
granted allowed, forged-authority inert, injection flagged, framing, codec, closed
registry). Full end-to-end audit-row assertions are in `Fleet.ToolDispatchTest`
(integration, tagged) — unrun this session (dev Postgres+AGE down). The generation
seam's harness-driven multi-turn loop (model actually emitting proposals) is the
next increment; feasibility confirmed (Brain has conversation state; the loop lives
in the harness over `Brain.ML.Generation`, not Brain's chat pipeline).

---

## 8. Open questions (living)

- ❓ Which Brain functions become the agent-callable capability surface (§3.4).
- ❓ Promotion path / titles / conferral UX (§4.4).
- ❓ Court-martial drafter when the accused *is* the XO (§4.3).
- ❓ True mailbox priority — deferred until security is observed to actually miss;
  fast handlers + a mostly-idle security agent buy nearly all of "instant" first.
- ❓ The same-soul-two-worlds experiment: does world framing shift a fixed soul's
  behavior? (The reason souls are portable.)
- ❓ Resource governance: per-agent budgets (concurrent Tasks, compute, any LLM
  spend) so an autonomous fleet can't run away with itself.
- ❓ Coordination deadlock: supervision recovers *crashes*, not *logical* stalls
  (A waits on B waits on A). What detects and breaks a wedged chain.
- ❓ In-system trials: does the fleet run commissioning trials and measure soul
  behavior *live* (vs. the prototype's offline harness)?
- 🔨 Soul resolution should move from "the world's first resident" (current
  thin-slice stand-in) to **the acting agent supplies its own soul** — which also
  removes the `brain → World.Roster` dependency wart in the built seam.
- ✅ *Resolved:* shared vs own mind → **own** (accountability is per-agent).
