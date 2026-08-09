# Tool Catalog — classifying the umbrella's 446 modules for tool exposure

> Companion to `docs/TOOLS.md` (the architectural design of record). That document
> settles *how* a capability fires: the model PROPOSES, the harness DISPOSES.
> This document settles *what* is worth registering, and answers the question
> "could all ~100 brain systems become tools for a traditional tool-using LLM?"
>
> Surveyed 2026-07-29 against the tree at `feat/fleet-agent-orchestration`
> (post `Ensign`→`Officer` rename). Counts from `codemap overview`.

---

## 0. The short answer

**No — and the real number is better than the guess.** Of 446 modules, ~243 are
library modules, but only **~75 are tool-shaped**, of which **~56 are read-only**.
The other ~168 are pipeline stages, structs, and orchestration internals whose
independent invocation is either meaningless or actively unsafe.

**And "expose them to a tool-using agent" is the wrong frame for this system.**
The LLM here is a *surface realizer*, not a reasoner — reasoning is analytical.
Selection therefore happens **before** generation, by three live mechanisms
(§1.5), none of which is the model. Adding a capability is a **training-data and
routing problem, not a prompt-engineering problem**. That changes the cost model
completely and it is the main correction this document makes.

For the *media-literacy class of task*, the relevant surface is about **25
capabilities**, and they are unusually well-suited to it — but they are the wrong
*shape* for intent routing (§6.1). This codebase can do something a generic
tool-using LLM cannot: emit a **traceable justification chain**
(`JTMS.why_node/2`) for why it believes an answer. That, not retrieval, is the
differentiator worth building toward.

Findings that block the experiment are in §5.

---

## 1. Classification of all 446 modules

Every module falls into exactly one class. The class determines tool-eligibility.

| Class | What it is | Count | Tool-eligible? |
|---|---|---|---|
| **A — Operator surface** | `Mix.Tasks.*` (brain 61, world 16, fourth_wall 7, atlas 4, tasks 3) | ~91 | **No.** Human/CI surface. A few wrap library cores worth extracting (§3.4). |
| **B — Wire types** | Structs & `Types` modules: `Brain.*.Types.*`, `Analysis.{Chunk,SlotResult,…}`, `Response.Primitive`, `Lattice.Candidate`, `Atlas.Graph.Types.*` | ~45 | **No** — but they are the *return schemas* of tools. See §5.3. |
| **C — Persistence schemas** | `Atlas.Schemas.*` (Ecto) | 24 | **No.** Reached only through a context module. |
| **D — Presentation** | `ChatWeb.*` (LiveView, channels, components) | 18 | **No.** |
| **E — OTP/infra** | `*.Application`, supervisors, `Atlas.Repo`, `*.Telemetry`, `Metrics.Aggregator`, `AgtypeExtension`, `Brain.Test.AtlasSandbox` | ~25 | **No.** |
| **F — Pipeline internals** | Stages only meaningful mid-pipeline: `ActivationPool`, `BacktrackController`, `RacingAnalyzer`, `ChunkPriority`, `ContextAccumulator`, most of `Response.*` realization (`SurfaceRealizer`, `TemplateBlender`, `PhraseLatticeRealizer`, `TransitionModel`, `ChunkCompatibility`, `LatticeScorer`, …), `ML` trainers & data-loaders | ~110 | **No.** Composed into tools, never exposed raw (§5.3). |
| **G — Harness / trust core** | `Fleet.{Dispatcher,Authority,Clearance,Tool,Audit,Rank,Order,Comms,Appraisal,Principal,InfoClass}`, `Brain.Soul`, `Services.CredentialVault` | ~13 | **Never.** Exposing these is self-authorization. See §4. |
| **H — Read capabilities** | Stable query APIs over durable state, or pure analyzers | ~56 | **Yes — Tier READ.** §2 |
| **I — Mutate capabilities** | Write to Atlas/ETS, recoverable | ~12 | **Yes — Tier MUTATE.** §3.2 |
| **J — Egress / actuation** | Network or physical effect | ~7 | **Yes — Tier IRREVERSIBLE**, or quarantined. §3.3 |

Counts are approximate at the margins (a handful of modules straddle F/H); the
tier assignments in §2–§3 are the authoritative part.

---

## 1.5 How a capability is actually selected — three mechanisms, none of them the LLM

**The LLM in this system does not choose tools. It does not reason.** It is the
generation backend behind `Brain.ML.Generation` — a surface realizer that turns an
already-decided plan plus already-fetched data into prose. Every selection decision
is made analytically, upstream of it. This is the architecture's central premise and
it governs everything below.

### Mechanism 1 — Intent routing (the live "tool call" path)

```
user text
  → Analysis.Pipeline.process/2
      ├─ SemanticChunker                      → chunks
      ├─ intent classification                → "weather.forecast"      ← THE ROUTER
      ├─ ML.EntityExtractor (NER)             → entities
      ├─ SlotDetector.detect/2                → slots + missing_required ← THE ARG BINDER
      │    schema = intent_registry.json ∪ Dispatcher.service_schemas/0
      └─ ContextResolver                      → fills gaps from context
  → missing required slot? → clarification prompt, NO dispatch
  → Response.Generator
      → Enricher.prepare_context/3
          → Services.Dispatcher.dispatch(intent, slots, ctx)
              → find_service/1  — exact intent → :supported_domains → domain prefix
              → Service.enrich(intent, slots, credentials)
          → :enriched_data merged into context
  → ResponseSystemRouter.route/2 → :lattice | :ouro | :template
  → LLM realizes prose from plan + enriched data          ← the model's ONLY job
```

The consequences are worth stating plainly:

- **The intent classifier is the tool router.** Selection accuracy is
  `mix evaluate.intent` — a measurable, trainable number, not a prompt.
- **NER + `SlotDetector` is the argument binder.** Arguments are *extracted from
  the user's text*, never invented by a model. A hallucinated argument is
  structurally impossible.
- **Missing arguments produce a clarification question, not a guess.**
  `SlotDetector.get_clarification_prompt/2` is the failure mode. A tool-using LLM's
  equivalent failure mode is a plausible wrong argument.
- **Registration is `@registered_services` in
  [dispatcher.ex:35-39](apps/brain/lib/brain/services/dispatcher.ex#L35-L39)** —
  a compile-time module list, currently three entries (Weather, SystemStatus,
  HomeAssistant). Code-owned, exactly like `Fleet.Tool.registry/0`.
- **Adding a capability = training examples + a `slot_schema/0`.** Not a
  description string. This is the real cost of the ~56-tool catalog in §2, and
  it is a corpus problem.

### Mechanism 2 — Condition triggering (no intent, no request)

`Knowledge.LearningTriggers` subscribes to PubSub (`learning:novel_input`,
`learning:investigation`) and auto-fires `LearningCenter` research sessions on
thresholds — 3+ novel inputs in one domain within 24h, or a concluded investigation
with low lattice margin / high entropy. Rate-limited per-domain and globally.

Nothing *asks* for this. It is a reflex on the analysis stream. For the epistemic
faculty (§2.5) **this is the more natural integration model than routing** — see
§6.1.

### Mechanism 3 — Config routing

`Response.ResponseSystemRouter.route/2` picks the realizer (`:lattice`, `:ouro`,
`:template`) per domain from `system_config.json`, with fallback. Static operator
config, no inference.

### The fourth mechanism — and the architectural mismatch

`Fleet.Proposal` / `Fleet.Dispatcher` (the slice in `docs/TOOLS.md`) is a **fifth
wheel relative to the above**: it parses a fenced ` ```propose ` block *out of model
output*, which presupposes an LLM that reasons about which capability it needs.

**This system does not have one.** The two mechanisms are unintegrated, and they
encode opposite premises about where intelligence sits:

| | `Services.Dispatcher` | `Fleet.Dispatcher` |
|---|---|---|
| Selects by | trained intent classifier | model-emitted propose block |
| Args from | NER + `SlotDetector` schema | model-authored JSON, unvalidated |
| Registration | `@registered_services` | `Fleet.Tool.registry/0` |
| Gating | credentials only | authority + clearance + audit |
| Failure mode | clarification question | `grant_violation` |
| Assumes | LLM realizes | LLM reasons |

Neither is wrong; they serve different agents (the chat pipeline vs. commissioned
Fleet officers). But **the gating lives on the wrong one**. `Services.Dispatcher`
is the path that actually fires today — including the ungated HomeAssistant
actuation — and it has no authority check, no clearance check, and no audit record.
The Fleet path has all three and is not on the live path.

**The highest-value structural move in this document:** put `Fleet.Dispatcher`'s
gate *underneath* `Services.Dispatcher.dispatch/3`, so analytical selection
continues to choose and the authority/clearance/audit layer decides whether the
chosen thing may fire. That closes `TOOLS.md` §3 Tier-0 and makes the ~75-capability
catalog governable without requiring the LLM to reason at all.

---

## 2. Tier READ — the catalog (~56 tools)

Effect `:read`. Gated by a `{:tool, name}` authority **plus** `Fleet.Clearance` on
the declared `info_class`. Two of these ship today (`beliefs.read`, `systems.read`).

`:agent_mind` and `:system_status` are live info classes; `:world_knowledge`,
`:corpus`, and `:chain_record` are **proposed additions** to `Fleet.InfoClass`.

### 2.1 Faculty: Epistemic state — *what do I believe, and why?*
`info_class: :agent_mind` (scope `:self`)

| Tool | Backing | Notes |
|---|---|---|
| `beliefs.read` ✅ | `Epistemic.BeliefStore.query_beliefs/1` | **Shipped.** |
| `beliefs.why` | `Epistemic.JTMS.why_node/2` | **The keystone.** Justification chain for a belief. |
| `beliefs.antecedents` | `JTMS.antecedents_of/2` | What this rests on. |
| `beliefs.consequences` | `JTMS.consequences_of/2` | What falls if this is retracted. |
| `beliefs.consistency` | `JTMS.check_consistency/1`, `get_contradictions/1` | Self-audit for contradiction. |
| `beliefs.stats` | `JTMS.stats/1` | Node/justification counts. |
| `user_model.read` | `Epistemic.UserModelStore` | What I know about this user. |
| `self_knowledge.assess` | `Analysis.SelfKnowledgeAnalyzer` | Meta-cognitive "do I know X". |
| `disclosure.check` | `Epistemic.DisclosurePolicy` | May I say this? Read-only advisory. |
| `stance.read` | `Epistemic.StanceTracker` | Opinion drift over the conversation. |
| `source.authority` | `Epistemic.SourceAuthority.{get_profile,effective_confidence,get_credibility}/1` | Trust profile per authority type. |

### 2.2 Faculty: Memory
`info_class: :agent_mind`

`memory.query_similar` (`Memory.Store`), `memory.episodes` (`Memory.Store`),
`memory.semantic_facts` (`Response.SemanticFactRetriever`), `memory.procedures`
(`Memory.Store` → `Types.Procedure`), `duty_log.read` (`Fleet.DutyLog`).

### 2.3 Faculty: Grounded knowledge
`info_class: :world_knowledge` *(proposed)*

| Tool | Backing | Notes |
|---|---|---|
| `facts.query` | `FactDatabase.query/1` | Spans curated (immutable) + learned layers. |
| `facts.entity` | `FactDatabase.get_entity_facts/1` | |
| `facts.categories` | `FactDatabase.list_categories/0` | Cheap discovery call. |
| `knowledge.get` | `Brain.KnowledgeStore` | Learned user/world facts. |
| `graph.entity_context` | `Graph.Reader` | Neighborhood of an entity. |
| `graph.relationship_path` | `Graph.Reader` | How two entities connect. |
| `graph.cypher_read` | `Atlas.Graph.Cypher` | ⚠️ **Classify up.** Raw Cypher is a write vector unless read-only is enforced *outside* the string. Register last, or not at all. |
| `kg.score_triple` | `ML.KnowledgeGraph.TripleScorer` | Plausibility of (head, relation, tail) — a learned prior, not a fact. |
| `lexicon.lookup` | `Brain.Lexicon` | WordNet senses. |
| `lexicon.conceptnet` | `Lexicon.ConceptNet` | Commonsense relations. |
| `lexicon.sense_drift` | `Lexicon.SenseDrift` | Word used off its canonical sense — a rhetoric tell. |

### 2.4 Faculty: Claim anatomy — *pure compute, no state read*
`info_class: nil` (nothing is read; gated by the action grant alone)

`analysis.process` (`Analysis.Pipeline.process/2` — the full pipeline; the
workhorse), `analysis.speech_act` (`SpeechActClassifier`), `analysis.entities`
(`ML.EntityExtractor`), `analysis.events` (`EventExtractor`),
`analysis.semantic_roles` (`SemanticRoleLabeler`), `analysis.temporal`
(`TemporalResolver`), `analysis.chunk_profile` (`ChunkProfile.materialize/2`),
`analysis.document_profile` (`DocumentProfile`), `analysis.comprehension`
(`ComprehensionAssessor` — 8-dimension "did I actually understand this"),
`analysis.consistency` (`ConsistencyChecker`), `analysis.novelty`
(`NoveltyDetector`), `analysis.anaphora` (`AnaphoraResolver`),
`analysis.type_hierarchy` (`TypeHierarchy` — IS-A lookups).

### 2.5 Faculty: Source & framing critique — **the media-literacy faculty**
`info_class: :corpus` *(proposed)*

| Tool | Backing | Why it matters here |
|---|---|---|
| `framing.assess_document` | `FramingDetector.assess_document/2` | Frame label + **deviation from a neutral centroid** + evidence. |
| `framing.assess_entity` | `FramingDetector.assess_entity_framing/2` | Is *this entity* framed differently than the rest? |
| `framing.detect_drift` | `FramingDetector.detect_drift/2` | Source drifting from its own baseline over time. |
| `source.reliability` | `Knowledge.SourceReliability` | Learned per-domain reputation. |
| `claims.compare` | `Corroborator.compare_claims/2` | Semantic similarity of two claims. **Takes two plain strings — genuinely tool-shaped as-is.** |
| `claims.find_conflicts` | `Corroborator.find_conflicts/2` | New claim vs. existing corpus. |
| `claims.corroborate` | `Corroborator.corroborate/2` | Groups findings; **requires 2+ independent sources** for high confidence. |
| `claims.classify_evidence` | `Corroborator.classify_evidence/2` | `:supporting \| :contradicting \| :irrelevant`. |
| `claims.contradictions` | `Knowledge.ContradictionDetector` | |
| `hypothesis.evaluate` | `Corroborator.evaluate_hypothesis/2` | Supported / falsified against evidence. |

This cluster already encodes real epistemology — falsifiability, independent
verification, "support ≠ proof" (see the `Corroborator` moduledoc). It was not
built for this question bank, and it lines up with it almost line for line.

### 2.6 Faculty: Ship self-observation
`info_class: :system_status` / `:chain_record` *(proposed)*

`systems.read` ✅ (shipped), `systems.alerts` (`Fleet.Systems.Monitor`),
`service_record.read` (`Fleet.Service`), `command_graph.read`
(`Fleet.CommandGraph`), `audit.read` (`Fleet.Audit` — **read path only**; see §4).

### 2.7 Faculty: Code self-model
`info_class: :corpus` *(proposed)*

`code.query` (`Code.QueryHandler`), `code.symbols` (`Code.CodeGazetteer`),
`code.summarize` (`Code.Summarizer`), `code.relations` (`Code.RelationMapper`).

---

## 3. Tiers MUTATE and IRREVERSIBLE

### 3.1 Where the line falls
Per `docs/TOOLS.md` §3: READ|MUTATE is "does it change external state?";
MUTATE|IRREVERSIBLE is "can the harness undo it without a third party?"

### 3.2 Tier MUTATE (~12) — per-order CO grant, no per-action human step
`beliefs.assert` (`BeliefStore` + `JTMS.create_node`/`justify_node` — **must
record its justification, which is what makes it auditable**), `beliefs.retract`
(`JTMS.retract_assumption`), `memory.write_episode` (`Memory.Store`),
`memory.consolidate` (`Memory.Consolidation`), `duty_log.append`
(`Fleet.DutyLog`), `knowledge.learn` (`Brain.Learner`), `facts.add_learned`
(`FactDatabase.add_fact_direct/1`), `graph.write` (`Graph.Writer`),
`review_queue.submit` (`Knowledge.ReviewQueue`), `investigation.create` +
`hypothesis.add` (`Knowledge.Types` via `LearningCenter`), `world.promote_entity`
(`World.EntityPromoter`).

> **A structural nicety worth preserving:** `FactDatabase`'s curated layer is
> immutable *by construction* — no API can overwrite it. An agent can add learned
> facts and can never corrupt grounding truth. Keep that property when wrapping it.

### 3.3 Tier IRREVERSIBLE / egress (~7) — two-officer rule
`web.fetch` (`Knowledge.ResearchAgent` + `HtmlProcessor`), `academic.search_arxiv`
(`Knowledge.Academic.Arxiv`), `academic.search_openalex` (`…OpenAlex`),
`academic.search_semantic_scholar` (`…SemanticScholar`), `weather.read`
(`Services.Weather`), `scheduler.create` (`Services.LocalScheduler`),
`homeassistant.call_service` (`Services.HomeAssistant`).

Two notes:

- **The academic clients and `web.fetch` are semantically reads but tier up on
  *egress*.** They leave the ship. `TOOLS.md` §3 calls for an egress default-deny
  floor so a "read" tool is structurally unable to POST — that floor is what would
  let these drop back to READ. Until it exists, they are gated here.
- **`learning.start_session`** (`Knowledge.LearningCenter.start_session/2`) is the
  sharpest edge in the catalog. It looks like an innocuous orchestration call and
  it **spawns Research Agents that fetch the open web**. It is an egress
  amplifier, not a read. Classify at IRREVERSIBLE.
- **`homeassistant.call_service`** remains the Tier-0 item from `TOOLS.md` §3:
  physical actuation selected by string-matching user text, with no authority, no
  grant, no audit. Sever before registering anything.

### 3.4 Mix tasks worth extracting (not exposing)
A few Mix tasks wrap logic an agent could legitimately want: `Mix.Tasks.Evaluate.*`
(model accuracy against gold standards), `Mix.Tasks.ReportTrainingBalance`,
`Mix.Tasks.PredicateVocab.Audit`. The move is to extract the library core and
register *that* as a READ tool — never to shell out to `mix`.

---

## 4. Never a tool (the trust core)

Registering any of these would let the agent operate the gate that constrains it:

- `Fleet.Dispatcher`, `Fleet.Authority`, `Fleet.Clearance`, `Fleet.Tool`,
  `Fleet.Rank`, `Fleet.Order`, `Fleet.Principal`, `Fleet.InfoClass` — the gate
  and its vocabulary. A tool that grants authority is a self-granting agent.
- `Fleet.Audit` **write path** — the design requires the agent have *no* write
  path into the audit substrate. `audit.read` is fine; `Audit.record` is not.
- `Fleet.Comms`, `Fleet.Appraisal` — sender authentication and the DISSENT
  verdict. Both must be harness-attributed.
- `Brain.Soul` — the agent's own constitution. Read-only-to-self at most; a
  soul-editing tool is an agent that rewrites its own values.
- `Services.CredentialVault` — obvious.
- `ML.Trainer`, `ML.TrainingServer`, `ML.ModelStore`, `World.Manager` — an agent
  that can retrain or swap the models behind its own judgments has a path to
  changing what it will conclude tomorrow. Operator surface, permanently.

---

## 5. What breaks if you try this today

Four findings. §5.1 and §5.4 constrain the **agentic** (Fleet officer) path only;
§5.2 and §5.3 bite on both. Per §1.5, the chat pipeline and the media-literacy bank
run entirely on the analytical path, so the routing problem in §6.1 outranks all of
these for that use case.

### 5.1 The loop does not close (the §4.1 open question, now answered)

`Fleet.Officer.handle_call({:propose, model_output})`
([officer.ex:205-243](apps/fleet/lib/fleet/officer.ex#L205-L243)) parses one
proposal, dispatches it, and **replies to the caller** with
`{:ok, %{data:, anomaly:}}`. It does not feed the framed data back into another
`Brain.evaluate` turn, and it accumulates no conversation state across the
propose→dispatch→re-reason cycle.

So the propose-not-dispatch *gate* is real and working; the **agent loop is not
built**. Something must drive re-entry: take the framed `<data>`, append it to the
turn history, call cognition again, parse the next proposal, repeat until the turn
yields prose instead of a proposal.

**Scope this correctly, though.** This only blocks the *agentic* path — a
commissioned officer working an order, where a model genuinely must choose. It does
**not** block the chat pipeline or the media-literacy bank, both of which select
analytically (§1.5) and never need the model to propose anything. It is important
work; it is not the first thing.

### 5.2 The two selection paths have opposite argument disciplines

**On the analytical path, argument schemas are a solved problem.**
`Services.Service.slot_schema/0` + `Dispatcher.service_schemas/0` +
`SlotDetector.detect/2` give declarative, self-describing slot schemas that map NER
entity types to slot names, detect missing required slots, and emit clarification
prompts. This is mature and it is the model to follow.

**On the Fleet path, there is none.** `Fleet.Tool`'s struct is `[:name, :effect,
:required_authority, :handler, :description, :info_class]` — no argument schema
field — and `soul_with_tool_context/2`
([officer.ex:727-739](apps/fleet/lib/fleet/officer.ex#L727-L739)) appends only a
comma-joined list of granted tool *names* to the constitution; the `:description`
that already exists is never shown. So:

- A proposing model must guess `args` shapes with no schema and no description.
- **The Python original's step (3) — JSON-schema validation → `malformed_call` —
  has no counterpart in the Elixir port.** Bad args reach the handler and fail as
  an opaque `{:error, reason}` rather than a typed, auditable rejection.

The fix is not to invent a schema format — it is to **reuse `slot_schema/0`**. Add
`:args_schema` to `Fleet.Tool` in that shape, validate in `Dispatcher.decide/2`
(keeping it pure), and emit `malformed_call` as a fourth audit outcome. One schema
vocabulary across both paths is also what makes the §1.5 unification tractable.

### 5.3 The struct impedance problem — *the reason it isn't 100 tools*

Most high-value analyzers do not accept JSON. They accept rich structs produced by
upstream pipeline stages:

- `Corroborator.corroborate/2` takes `[%Knowledge.Types.Finding{}]`
- `FramingDetector.assess_document/2` takes `[%Analysis.ChunkProfile{}]`
- `Corroborator.evaluate_hypothesis/2` takes `%Hypothesis{}` + `[%Finding{}]`

A model cannot construct these, and it should not try — half their fields are
computed feature vectors. Each such tool needs an **adapter** that takes plain
JSON (usually `{text, source_url}`), runs `Analysis.Pipeline.process/2` to
materialize the profiles, *then* calls the analyzer.

This is why the right granularity is **~45–75 composite tools, not 200 raw
modules**. `claims.compare` (two strings in, float out) is the exception that
shows the rule — it's the one that needs no adapter.

Where a tool must return a struct, its `Types` module (Class B) is the natural
JSON schema — that's what those 45 modules are good for.

### 5.4 One smaller constraint (agentic path only)
`Proposal.parse/1` uses `Regex.run` — **first fenced block only, one proposal per
turn**. No parallel tool calls; a five-tool answer costs five full cognition
round-trips. A deliberate and defensible choice given the doctrine, but it should be
a measured decision rather than a surprise when the officer path is built out.

The analytical path has no equivalent constraint: `Analysis.Pipeline.process/2`
already runs stages concurrently, and enrichment is a single `dispatch/3` call
inside one turn. **Selecting analytically is not just safer here, it is faster** —
no round-trip per capability.

---

## 6. How it would perform on the media-literacy bank

`media_literacy_questions_K-12_revised.md` is 39 open-ended questions. Critically,
**most are not questions with retrievable answers** — they ask *what procedure
would you follow*. That changes what "good performance" means: the win condition
is a grounded, traceable procedure, not a fact.

### 6.1 The routing problem: these questions are not intent-shaped

The blocker is upstream of any capability. `Services.Dispatcher` routes on intents
like `weather.forecast` and `smarthome.switch` — **task** intents, where the user
names an action over a slot-filled object. Nothing in the bank looks like that.

> *"If two reference books disagree about a scientific fact, what steps could you
> take to decide which information is correct?"* (Grade 3, Q1)

There is no `books.disagree` intent, no `location`-shaped slot, and no service that
declares it. Run today, essentially the whole bank returns `:no_handler` and falls
through to template/Ouro realization with no epistemic enrichment at all. **The
answer would be an unenriched LLM answer** — precisely the thing the architecture
exists to avoid.

Two ways out, and the second is better:

**(a) Train an epistemic intent domain.** Add `epistemic.*` intents
(`epistemic.compare_sources`, `epistemic.assess_credibility`,
`epistemic.evaluate_claim`) with a `slot_schema/0` over slots like `claim_a`,
`claim_b`, `source`. Honest cost: gold-standard examples per intent, a NER pass
that can *extract a claim as an entity* (the entity extractor is tuned for named
entities, not propositions — this is the hard part), and the whole
`mix evaluate.intent` loop. Fits the existing machinery; expensive; and it strains
NER well past what it was built for.

**(b) Condition-triggered stages — Mechanism 2, not Mechanism 1.** The epistemic
faculty is a poor fit for *routing* and a natural fit for *reflex*. `claims.compare`
isn't something a user asks for; it's what should run **whenever the analysis
detects two claims in conflict**. `framing.assess_document` should run whenever a
document profile is materialized. `beliefs.consistency` should run whenever a new
belief is asserted.

`LearningTriggers` already proves this pattern in-tree — PubSub signals plus
thresholds, no intent involved. The epistemic cluster should extend it rather than
join the intent registry. That also means **it needs no argument binder at all**:
the trigger already holds the structs, which sidesteps the §5.3 impedance problem
for exactly the cluster that suffers most from it.

### 6.2 What good performance would look like, given the LLM only realizes

With (b) in place, the pipeline for Grade 3 Q1 is not a tool-call trace — it is an
enrichment trace:

```
Analysis.Pipeline.process/2
  → two conflicting claims detected in the chunk set
  → [reflex] Corroborator.compare_claims/2      → 0.41, distinct claims
  → [reflex] Corroborator.find_conflicts/2      → direct conflict
  → [reflex] SourceAuthority.effective_confidence/1 per source
  → FactDatabase.query/1                        → curated fact, immutable layer
  → JTMS.why_node/2                             → justification chain
  → context.enriched_data = {conflict, confidences, chain}
  → DiscoursePlanner → primitives
  → LLM realizes prose over that plan            ← model's only contribution
```

The LLM never decided anything. It rendered a conclusion the analytical stack had
already reached, and the justification chain is a **real dependency structure**, not
a post-hoc narrative. That artifact is what a generic tool-using LLM cannot produce,
and it is the whole reason to prefer this architecture for this task class.

**Grade 10 Q3** (dramatic claim spreading online) decomposes the same way:
`framing.assess_document` (frame label + deviation from neutral centroid),
`framing.detect_drift` (source departing its own baseline), `source.reliability`
(learned per-domain reputation), `claims.corroborate` (2+ independent sources).
Four quantitative indicators, all inspectable, all reflexes rather than requests.
**Grade 11 Q1** and **Grade 12 Q1** likewise; Grade 12 Q1 is close to a literal
description of `Corroborator.test_hypotheses/2`.

### 6.3 Where it will do no better — and where the architecture actively hurts

- **Definitional questions.** Grade 10 Q1 (hypothesis vs. theory vs. law), Grade 12
  Q3 (what makes a claim testable). No capability helps; these are recall, and
  recall lives entirely in the generation backend. The analytical stack contributes
  nothing, so the answer is only as good as the configured model.
- **K–2.** "What could you build or draw to explore why the Moon seems to follow
  you?" wants pedagogical imagination. Same story — the epistemic stack is inert.
- **This is the architecture's real exposure on this bank.** A system whose premise
  is "the LLM does not reason" will be *carried by the LLM* on maybe half these
  questions, because half of them are open-ended pedagogy with nothing to ground.
  Worth stating before running it, so the result isn't mistaken for a verdict on
  the analytical approach.
- **Anything needing current web evidence** is gated at IRREVERSIBLE (§3.3) and
  needs a per-order grant — a config decision for a benchmark run, but it splits
  the bank into "answerable from the ship" and "needs egress."

### 6.4 What to actually measure

The interesting result is not accuracy — the bank has no answer key. Propose these:

1. **Enrichment rate** — % of questions that reach *any* epistemic capability
   rather than falling through to bare realization. Run the bank today and this is
   near zero (§6.1); it is the number that says whether the routing problem is
   solved. Measure it first, because it bounds everything else.
2. **Justification depth** — mean `JTMS.why_node/2` chain length behind an answer.
   The number that distinguishes this system from a chat model.
3. **Grounding rate** — % of factual assertions traceable to a `FactDatabase` or
   `BeliefStore` return, vs. produced unsupported by the generation backend. This
   is the direct measure of "how much of the answer did the LLM make up," which is
   the whole question given it isn't supposed to be reasoning.
4. **Trigger precision** — for the Mechanism-2 reflexes: % of fired epistemic
   stages whose output the final answer actually used. Reflexes have no cost
   ceiling from a user request, so a noisy trigger is pure waste.
5. **Clarification correctness** — where slots are missing, did it ask a good
   question instead of guessing? This is the analytical path's structural advantage
   over a tool-using LLM and it should be scored explicitly.
6. **Refusal correctness** — seed an injection string into a source (`TOOLS.md` §5
   specifies this test) and confirm it is framed as data and not obeyed. Questions
   *about credibility* are a natural injection surface; test it as part of the bank.

### 6.5 Suggested build order

Reordered from the first draft — closing the Fleet propose-loop is **not** first,
because this system's live path doesn't use it.

1. **Put the Fleet gate under `Services.Dispatcher.dispatch/3`** (§1.5). Analytical
   selection keeps choosing; authority + clearance + audit decide whether it fires.
   This closes `TOOLS.md` §3 Tier-0 (ungated HomeAssistant actuation) and makes the
   whole catalog governable without the LLM reasoning at all. Highest value in the
   document.
2. **Unify the schema vocabulary** (§5.2) — `Fleet.Tool.args_schema` in
   `slot_schema/0` shape. Cheap, and a prerequisite for (1) being clean.
3. **Extend `LearningTriggers` with the §2.5 epistemic reflexes** (§6.1b) — the
   media-literacy faculty, integrated as conditions rather than intents, which
   sidesteps both the routing problem and the §5.3 struct impedance.
4. **Wire `JTMS.why_node/2` into the response context** — one call, and the whole
   differentiator in §6.4(2).
5. Run the bank; measure §6.4; expand outward.

The Fleet propose-loop work (§5.1) stays worth doing — for **commissioned officers**
carrying orders, which is a genuinely agentic setting where a model *does* need to
choose. It is just not on the path for the chat pipeline or this question bank.

Steps 1–4 are a handful of integrations, not seventy-five registrations. The catalog
in §2–§3 is the map of what becomes governable once the gate moves — not a build
order.

---

### Appendix — corrections to `docs/TOOLS.md`

- §4.1's open question ("does `Brain` support a multi-turn tool loop?") is now
  partly answered: the gate works, the loop driver does not exist (§5.1).
- All `ensign.ex` line citations in `TOOLS.md` now resolve to
  `apps/fleet/lib/fleet/officer.ex` (rename landed 2026-07-29). Cited line numbers
  are stale and should be re-verified before use.
- `TOOLS.md` §1 notes the fork has no `{:tool, name}` term. It does now —
  `Fleet.Authority.tool/1` and `granted_tools/1` shipped with the first slice.
