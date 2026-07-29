# BRAIN — Module Catalog & Doc/Code Consistency Audit

> **Status:** point-in-time audit, generated 2026-07-04. This is a snapshot, not a
> living contract like `FLEET.md` — re-run the audit after significant refactors
> rather than hand-editing entries piecemeal. `apps/brain/lib/brain` had **209
> modules** at audit time, every one of them already carrying a `@moduledoc`
> (100% coverage). What this document adds is the thing `@moduledoc` coverage
> alone can't guarantee: an independent read of each module's *implementation*,
> checked against what its own docs claim.

**Method:** ten parallel review passes, each deep-reading its assigned files
(not just `@moduledoc`/`@doc` text) and cross-checking claims against function
bodies, call sites (via repo-wide grep), and test coverage. "Mismatches" below
means a concrete, cited divergence between what a module's documentation says
and what the code does — not a style nitpick.

**How to use this doc:** the [Executive Summary](#executive-summary) is the
useful part for a first read — it's the small number of cross-cutting patterns
that recurred across independent subsystems. The per-subsystem catalogs after
it are the reference material: what each module is for, its public API, and
its specific doc/code discrepancies (if any).

---

## Executive Summary

Five patterns recurred across independently-reviewed subsystems, which suggests
they're systemic rather than one-off bugs.

> **See also:** [`docs/internal/hardcoded_stubs_audit.md`](internal/hardcoded_stubs_audit.md)
> for the full, dedicated writeup of pattern #3 below — 28 cited instances,
> ranked by severity, checked against `.cursorrules`' explicit "no hardcoded
> data, fail loud, no silent fallbacks" doctrine (Rules 2/4/5/8).

### 1. `persist/0` is a no-op stub in several GenServers, despite its own doc claiming otherwise

At least four core stores document a `persist/0` function as "persists to disk,"
but the implementation is a bare `{:reply, :ok, state}` that writes nothing:

- `Brain.Epistemic.BeliefStore.persist/0` (epistemic/belief_store.ex:473-475)
- `Brain.Epistemic.SourceReliability.persist/0` (knowledge/source_reliability.ex:144-146)
- `Brain.Epistemic.UserModelStore.persist/0` (epistemic/user_model.ex:302-304)
- `Brain.Memory.Store.persist/0` (memory/store.ex:538-540)

In every case this is because the real design has moved to continuous
write-through persistence (each mutating call already persists to Atlas
immediately), and `persist/0` is a leftover from a pre-Atlas design that nobody
has removed. None of these four have any caller anywhere in `apps/` — they're
harmless today, but the doc actively misleads a reader into thinking a manual
flush is meaningful. Recommend either deleting these functions or rewriting
their docs to say "no-op; persistence is continuous write-through."

By contrast, `Brain.Epistemic.SourceAuthority.persist/0` and
`Brain.Knowledge.ReviewQueue.persist/0` are real — they're the positive
counter-examples showing the write-through design elsewhere in the same
subsystem.

### 2. Fully-implemented, well-tested code that nothing calls

Not "dead code" in the usual low-effort sense — these are complete,
non-trivial features with their own test suites, but zero production callers:

- **`Brain.Lexicon.SenseDrift`** — the entire module (drift detection, tier-2
  refinement eligibility, emergent-sense detection) has no callers anywhere,
  despite `Brain.Lexicon`'s own moduledoc listing "sense drift detection" as a
  supported capability.
- **`Brain.Services.LocalScheduler`** — a fully-implemented alarm/reminder/
  timer/calendar service (10 intents, `Brain.Services.Service` behaviour
  correctly implemented) that `Brain.Services.Dispatcher.@registered_services`
  simply never lists. Nothing can ever route to it.
- **`Brain.Analysis.ChunkProfile.Projections`** — a pure-function extraction
  intended to de-bloat `ChunkProfile`, but `ChunkProfile.materialize/2` never
  calls it; the same logic is duplicated inline in `chunk_profile.ex` instead.
  Two copies of the same voting logic, only one of which is live.
- **`Brain.ML.DataLoaders.load_negative_examples/1`** — implements
  negative-example loading for intent-confusion training, documented in
  detail, never wired into `EntityTrainer`/`Trainer`.
- Smaller instances scattered throughout: `Brain.Response.Generator`'s legacy
  `generate_with_path/3`/`generate_with_context/4` path,
  `Brain.ML.Lexicon.WordNetParser.parse_instances/similar/derivations`,
  `Brain.Analysis.EventPatterns`' entire pattern/tensor API (only two of ~20
  functions are actually reached from production), `Brain.Response.
  TransitionModel.build_scores/1`/`save_scores/1` (no automated entry point).

### 3. Silent stub logic masquerading as real signals

Several functions are hardcoded to a constant return value with no TODO/FIXME
disclosing it, so downstream logic that branches on them can never actually
branch the other way:

- `Brain.Analysis.BacktrackController`'s `check_entity_mismatch/1` always
  returns `:ok` — one of three documented contradiction checks is inert.
- `Brain.Analysis.EventLinker`'s `temporal_contains?/2` always returns
  `false` — the documented "sub-event containment" feature never fires.
- `Brain.Analysis.AnalyzerCalibration`'s win-rate tracking
  (`is_dominating?/2`) reads an ETS key nothing ever updates past
  initialization — always `false` in practice.
- `Brain.Response.ContentSpecifier`'s `check_action_capability/2` (and an
  independent hardcode in `DiscoursePlanner`) always returns `:unknown`,
  collapsing capability-gated primitives to a single branch app-wide.
- `Brain.ML.Trainer.train_intent_classifier/2` writes a **constant `1.0`**
  IDF weight for every vocabulary word into the persisted embedder model,
  discarding the real IDF `SimpleClassifier.train/1` computed — silently, with
  no disclosure that the embedder's IDF is fake.
- `Brain.ML.KnowledgeGraph.Embedder.build_embedding_model/1` doesn't actually
  truncate the model as documented (it's an identity wrapper around the
  *full* scorer including the sigmoid head) — dormant only because the real
  caller (`EntityVectorCache`) bypasses it for the correctly-truncated
  `build_extraction_model/2`.

### 4. Undisclosed severed/degraded functionality (one security-relevant)

- **`Brain.Services.HomeAssistant`** — per an in-code comment dated
  2026-07-03, the actuation path (`call_service/5`) was deliberately and
  completely disabled after a prior incident (referenced in `docs/TOOLS.md`'s
  survey as a "live, ungated, unaudited physical-actuation surface"). Every
  action-shaped intent now runs a full resolution pipeline and then returns
  `{:error, :actuation_disabled}`. This is good remediation, but the
  `@moduledoc` still describes a live, working integration — a reader
  cataloging this module from its docs alone would have no way to know
  `smarthome.turn_on` no longer does anything. The read path (state queries)
  remains live and ungated by comparison. Recommend hoisting the
  remediation rationale into the moduledoc itself.
- **`Brain.Code.LanguageGrammar`** discloses prominently (a whole "Current
  Status: Degraded Mode" section) that tree-sitter isn't compiled in and
  every parse uses a regex/line-based fallback. Its sibling
  **`Brain.Code.Parser`**, which does the actual parsing, does **not**
  disclose this anywhere in its own moduledoc — a reader of `Parser` alone
  would believe it produces real tree-sitter ASTs.
- **`Brain.ML.Gazetteer`**'s documented `:table_prefix` test-isolation feature
  is broken: `init/1` computes prefixed table names into `state.tables`, but
  nearly every read/write function (25 of ~29 call sites) references the
  hardcoded global `@table_name`-family module attributes instead, so a
  GenServer started with a prefix still mutates the shared global tables.
  This is a live, exercised code path (used by
  `test/support/genserver_sandbox.ex`), not a theoretical bug. Contrast with
  **`Brain.ML.Lexicon`**, which implements the identical isolation feature
  correctly.

### 5. Telemetry handler catalog is stale

`Brain.Telemetry.attach_handlers/0` attaches 6 handlers for KG-signal and
novelty events that the moduledoc's own event catalog never mentions, **and**
`detach_handlers/0` omits those same 6 handler ids from the list it detaches.
`detach_handlers/0`'s own doc ("Detaches all telemetry handlers") is therefore
inaccurate — a test calling `detach_handlers/0` then `attach_handlers/0` to
reset state will hit a duplicate-attach error from `:telemetry` on those 6
ids.

---

## Table of Contents

1. [Root entry points](#root-entry-points) (12 modules)
2. [Brain.Analysis](#brainanalysis) (48 modules)
3. [Brain.ML](#brainml) (49 modules)
4. [Brain.Response](#brainresponse) (30 modules)
5. [Brain.Knowledge](#brainknowledge) (14 modules)
6. [Brain.Epistemic](#brainepistemic) (10 modules)
7. [Brain.Services](#brainservices) (12 modules)
8. [Brain.Code](#braincode) (9 modules)
9. [Small subsystems](#small-subsystems) — Lexicon, Memory, Graph, Subprocesses, FactDatabase, Lattice, misc singletons (35 modules)

---

## Root entry points

The 12 modules directly under `apps/brain/lib/brain/` (not in any subdirectory)
— the public façade for each major subsystem.

### Brain.Application
**File:** application.ex
**Purpose:** OTP application callback. `start/2` builds the ~60-child supervision tree (PubSub, ML classifiers, epistemic stores, response generation, subprocess supervisors, etc.), then — unless `skip_ml_init` — spawns a detached `Task` that loads the gazetteer, activates world models (or falls back to a bundled embedder/classifier), and initializes `World.Embedder`. `@moduledoc false`.
**Public API:** `start/2`.
**Mismatches:** None (no doc to check against; inline comments match behavior).

### Brain.Soul
**File:** soul.ex
**Purpose:** Loads a per-agent "constitution" (identity/values/behavioral bounds) from a flat JSON file in `souls_dir()`. Confirmed real callers: `RealizationPacket.system_prompt/1`, `World.Roster`, `Fleet.Officer`.
**Public API:** `souls_dir/0`, `get/1`, `list_ids/0`, `system_prompt/1`.
**Mismatches:** None.

### Brain.LinguisticData
**File:** linguistic_data.ex
**Purpose:** Compile-time-loaded negation/intensifier/hedge word lists from `priv/knowledge/linguistic.json`, with hardcoded fallbacks. `has_negation?/1` is token-level, not substring.
**Public API:** `negation_words/0`, `intensifiers/0`, `hedges/0`, `negation?/1`, `has_negation?/1`.
**Mismatches:** None.

### Brain.MemoryStore
**File:** memory_store.ex
**Purpose:** File-per-persona JSON store of conversation "thoughts" (`priv/memory/<persona>.json`).
**Public API:** `start_link/1`, `load_all/2`, `append_thought/5`, `get_memory_window/3`, `clear_memory/2`, `ready?/0`.
**Mismatches:** None in this module itself, but see **Brain.Learner** below — `Learner.store_context_memory/3` pattern-matches on a `"content"` key that entries here never have (they use `"text"`), so a downstream de-dup check silently never fires.

### Brain.FactDatabase
**File:** fact_database.ex
**Purpose:** GenServer holding immutable curated facts (loaded via Atlas with a JSON-file fallback) plus mutable learned facts; ranks search hits via POS-tagged keyword extraction.
**Public API:** `start_link/1`, `query/1`, `get_fact/1`, `get_entity_facts/1`, `get_category_facts/1`, `list_categories/0`, `reload/0`, `add_fact_direct/1`, `curated?/1`, `stats/0`, `ready?/0`.
**Mismatches:** None.

### Brain.KnowledgeStore
**File:** knowledge_store.ex
**Purpose:** Two parallel JSON storage schemes — legacy persona-scoped and world-scoped — with typed entity setters. `add_relationship/5`/`add_fact/4` upsert-and-merge on confidence/timestamp rather than duplicating (undocumented positive behavior).
**Public API:** world-scoped CRUD (`get_world_knowledge/2`, `save_world_knowledge/2`, `add_to_world/4`, `remove_from_world/3`, `clear_world/1`, `list_knowledge_worlds/0`); persona-scoped (`start_link/1`, `load_knowledge/2`, `save_knowledge/2`); typed setters (`add_pet/3`, `add_person/3`, `add_room/3`, `add_device/3`, `add_place/3`, `add_task/3`, `add_event/3`, `add_preference/3`, `set_birthdate/3`, `add_favorite_holiday/3`); `add_relationship/5`, `add_fact/4`, `get_knowledge/2`, `clear/1`, `ready?/0`.
**Mismatches:** `save_world_knowledge/2`, `list_knowledge_worlds/0`, `set_birthdate/3`, `add_favorite_holiday/3` have no callers anywhere in `apps/`. Several public setters have no `@doc` at all.

### Brain.Lattice
**File:** lattice.ex
**Purpose:** Ranked N-best candidate wire format (`%Lattice{}`) with softmax-based constructors and rerank/filter/merge combinators, all funneling through `finalize/1`.
**Public API:** `empty/2`, `empty?/1`, `error/1`, `best/1`, `second/1`, `margin/1`, `entropy/1`, `best_label/1`, `to_top_k/1`, `to_map/1`, `to_context_signal/2`, `from_classifier/2`, `from_top_k/2`, `singleton/3`, `normalize/1`, `rerank/2`, `filter/2`, `map_labels/2`, `take_top_k/2`, `merge/3`.
**Mismatches:** `merge/3` and `to_context_signal/2` have no call sites anywhere — designed-in but unwired.

### Brain.Lexicon (facade)
**File:** lexicon.ex
**Purpose:** Facade over WordNet, ConceptNet, and user-defined lexicon, adding lexicographer-file→domain mapping, OOV detection, and context-overlap word-sense disambiguation.
**Public API:** `domain_atoms/0`, `lexfile_to_domain_map/0`, `lookup/2`, `known?/1`, `oov?/1`, `primary_domain/2`, `domain_histogram/1`, `polysemy_count/1`, `hypernym_depth/2`, `antonyms/1`, `conceptnet_relations/1`, `conceptnet_relation_counts/1`, `disambiguate/3-4`, `word_similarity/2`, `lemma/1`, `synset_ids/2`, `lexical_domain_for_synset/1`.
**Mismatches:** Most of the OOV/disambiguation/lemma API (`lookup/2`, `known?/1`, `oov?/1`, `disambiguate/3-4`, `lemma/1`, `synset_ids/2`, `hypernym_depth/2`, `antonyms/1`, `domain_histogram/1`, `lexfile_to_domain_map/0`) has no callers anywhere — only `word_similarity/2`, `primary_domain/2`, `polysemy_count/1`, `conceptnet_relation_counts/1`, `domain_atoms/0` are used in production. Moduledoc claims "sense drift detection" as a feature, but that logic lives entirely in the separate (also-unused) `Brain.Lexicon.SenseDrift`.

### Brain.AtlasIntegration
**File:** atlas_integration.ex
**Purpose:** Async/sync write-through bridge between in-memory GenServers and Atlas Postgres, covering Belief, Episode/SemanticFact, ReviewCandidate, SourceReliability, LearnedFact, SourceAuthority, UserModel, IntentReviewCandidate, plus generic graph node/edge helpers.
**Public API:** ~30 functions (`available?/0`, `ping/0`, `async/1`, `drain/1`, `sync/1`, plus per-schema load/persist pairs and graph helpers) — see file for full list.
**Mismatches:** None — every function confirmed to have real callers; the most doc-accurate module in the audit.

### Brain.SystemStatus
**File:** system_status.ex
**Purpose:** Read-only status/monitoring facade over ~40 registered GenServers plus embedder/NLP/ML-model/world/services status, health scoring, and a utilization report.
**Public API:** `get_all/0`, `get_all_genservers_status/0`, `get_performance_metrics/0`, `get_health_indicators/0`, `get_utilization_report/0`, per-subsystem status getters, `get_lstm_status/0` (explicitly-documented deprecated stub), `all_ready?/0`.
**Mismatches:** None.

### Brain.Telemetry
**File:** telemetry.ex
**Purpose:** Central registry of telemetry event names, `attach_handlers/0`/`detach_handlers/0`, per-event `span/3` wrappers, `emit_*` convenience functions — all handlers cast non-blockingly to `Brain.Metrics.Aggregator`.
**Public API:** `attach_handlers/0`, `detach_handlers/0`, `span/3` (per-event clauses), ~11 `emit_*/N` functions, `@doc false` handler callbacks.
**Mismatches:** See Executive Summary #5 — KG-signal/novelty handlers attached but undocumented and not detached.

### Brain.Learner
**File:** learner.ex
**Purpose:** NLP-pipeline-driven entity/fact extractor; three entry points funnel into `process_extracted_data/3`, which dispatches entities to `KnowledgeStore` setters and cross-checks facts against `FactDatabase.Integration.verify_fact/2`.
**Public API:** `learn_from_input/3`, `learn_from_classical_extraction/3`, `learn_from_conversation/4`.
**Mismatches:**
- Moduledoc claims "relationship extraction," but all three entry points hardcode `"relationships" => []` — `KnowledgeStore.add_relationship/5`'s one caller (`process_relationships/2`) is therefore always called with an empty list. Relationship extraction is documented but never implemented.
- `store_context_memory/3`'s de-dup check pattern-matches `%{"content" => ^memory_text}` against `MemoryStore` entries, which are always keyed `"text"` — the dedup logic is silently inert; every call appends a fresh memory entry.

---

## Brain.Analysis

48 modules — the NLP comprehension pipeline (chunking, discourse, entities, events, slots, calibration).

### Brain.Analysis.Pipeline
**File:** analysis/pipeline.ex
**Purpose:** The end-to-end orchestrator. `SemanticChunker` splits input; each chunk runs a two-pass process — pass 1 (concurrent discourse/speech-act/sentiment classification + sequential entity/anaphora/POS/event/SRL extraction), then `build_cross_chunk_context/3` selects the primary chunk and accumulates turn-level signals, then pass 2 (only for "substantive" chunks) adds full intent classification, contextual entity inference, slot detection, context resolution, fact verification, and novelty detection.
**Public API:** `process/2`, `analyze_chunk/2`, `summarize/1`, `registered_intent?/1` (`@doc false`), `expected_entity_types_from_domain/1` (`@doc false`, used externally).
**Mismatches:** The moduledoc describes a simple 4-stage pipeline; the actual implementation is a two-pass architecture with cross-chunk aggregation, substantive-chunk gating, contextual entity inference, epistemic fact verification, novelty detection, and async belief extraction — none of which the moduledoc mentions. `registered_intent?/1` is public with zero external callers.

### Brain.Analysis.InternalModel
**File:** analysis/internal_model.ex
**Purpose:** Defines `InternalModel`, `Chunk`, `ChunkAnalysis`, `DiscourseResult`, `SpeechActResult`, `SlotResult` — the pipeline's output model.
**Public API:** `InternalModel.{new/1, with_chunks/2, with_analyses/2, determine_strategy/1, respondable_analyses/1, clarification_analyses/1, bot_addressed?/1}`; `Chunk.new/4-5`; `ChunkAnalysis.{new/2, with_events/2, primary_event/1, has_events?/1, determine_response_strategy/1}`; `DiscourseResult.new/3`; `SpeechActResult.{new/4, expects_response?/1}`; `SlotResult.{new/1, fill_slot/5, get_slot_value/2}`.
**Mismatches:** `ChunkAnalysis`'s documented `@type t` omits five real struct fields (`event_frames`, `srl_frames`, `pos_tags`, `intent_domain`, `accumulated_context`) that the code actually uses.

### Brain.Analysis.SemanticChunker
**File:** analysis/semantic_chunker.ex
**Purpose:** Splits raw input into `%Chunk{}`s: normalize → mask quotes → sentence-split → merge short sentences → tag discourse markers → re-split long sentences on clause boundaries → restore quotes → compute offsets. Thresholds pulled from `LearningStore` when available.
**Public API:** `chunk/1`, `analyze/1`.
**Mismatches:** A learned `discourse_markers` list is fetched from `LearningStore` but never actually used — `detect_discourse_markers/1` always reads the hardcoded `@default_discourse_markers` attribute instead. Chunk-size thresholds are genuinely learnable; discourse markers are fetched as if they were, but silently discarded.

### Brain.Analysis.ChunkPriority
**File:** analysis/chunk_priority.ex
**Purpose:** Picks the "primary" chunk via a strict 5-tier ordering (question-directive > directive > assertive-with-entities > highest confidence > first-in-list).
**Public API:** `select_primary/1`, `question_chunk/1`.
**Mismatches:** None.

### Brain.Analysis.ChunkProfile
**File:** analysis/chunk_profile.ex
**Purpose:** Builds `%ChunkProfile{}` with 17 primary axes + 4 derived interaction axes, from hard copies of analysis fields, MicroClassifier calls, and compositional voting.
**Public API:** `new/0`, `materialize/2`, `derived_label/1`.
**Mismatches:** Duplicates (independently, not by calling) the exact same voting logic that lives in `Brain.Analysis.ChunkProfile.Projections` — see that module and Executive Summary #2.

### Brain.Analysis.ChunkProfile.Projections
**File:** analysis/chunk_profile/projections.ex
**Purpose:** Intended home for `ChunkProfile`'s per-axis projection logic, extracted to avoid bloating the struct module.
**Public API:** `project_target/1`, `project_modality/1`, `project_polarity/1`, `project_sentiment_alignment/1`, `project_response_posture/1`, `project_engagement_level/1`, `project_self_disclosure_level/1`, `project_temporal_framing/1`, `classify_axis/3`, `safe_to_atom/1`.
**Mismatches:** Its stated purpose is false in practice — `ChunkProfile.materialize/2` never calls it; it's exercised only by its own test file. See Executive Summary #2.

### Brain.Analysis.DocumentProfile
**File:** analysis/document_profile.ex
**Purpose:** Aggregates `%ChunkProfile{}` lists into document-level weighted mean/variance/skew vectors, per-entity slices (capped at 50), and rhetorical-mode share.
**Public API:** `aggregate/2`, `similarity/2`, `deviation_from/2`, `vector_dimension/1`.
**Mismatches:** `dominant_pos_distribution`/`dominant_lexical_domains` fields are always `[]` — dead placeholder fields. `similarity/2`/`deviation_from/2`/`vector_dimension/1` have zero production callers; the one real consumer (`FramingDetector`) reimplements its own private cosine-similarity instead of calling this module's public one.

### Brain.Analysis.ComprehensionAssessor
**File:** analysis/comprehension_assessor.ex
**Purpose:** GenServer scoring `ChunkAnalysis` lists across 8 dimensions (via `DimensionEvaluators`), building a `ComprehensionProfile`, storing in ETS (7-day TTL), and evolving per-dimension EMA weights from approve/reject outcomes (direct call or PubSub `"knowledge:review"`).
**Public API:** `start_link/1`, `assess/2`, `record_outcome/3`, `stats/1`, `reset_weights/1`, `ready?/1`.
**Mismatches:** None.

### Brain.Analysis.ComprehensionAssessor.ComprehensionProfile
**File:** analysis/comprehension_assessor/comprehension_profile.ex
**Purpose:** Struct + `build/2` computing weighted composite score across 8 dimensions, with a hard `structural_coherence < 0.2 ⇒ garbled` gate and fixed verdict thresholds.
**Public API:** `dimension_names/0`, `build/2`, `dimension_score/2`.
**Mismatches:** None.

### Brain.Analysis.ComprehensionAssessor.DimensionEvaluators
**File:** analysis/comprehension_assessor/dimension_evaluators.ex
**Purpose:** The 8 individual dimension scorers, each reading only pre-existing `ChunkAnalysis` fields.
**Public API:** `evaluate_all/1` + 8 individual scorers (`referential_clarity/1`, `actor_identification/1`, `propositional_content/1`, `temporal_grounding/1`, `contextual_sufficiency/1`, `epistemic_grounding/1`, `structural_coherence/1`, `illocutionary_clarity/1`).
**Mismatches:** None.

### Brain.Analysis.DiscourseAnalyzer
**File:** analysis/discourse_analyzer.ex
**Purpose:** Additive-scoring addressee detection (bot/user/third-party/ambiguous) combining hardcoded word-list indicators with two ML micro-classifiers and a 1-on-1 baseline bonus.
**Public API:** `analyze/2`, `debug_analyze/2`.
**Mismatches:** `debug_analyze/2` has no production callers despite being framed as a learning/debugging utility. Moduledoc doesn't disclose the hybrid hardcoded-list + ML-classifier nature of address/modal detection.

### Brain.Analysis.EntityDisambiguator
**File:** analysis/entity_disambiguator.ex
**Purpose:** Cascading disambiguation — introduction-pattern confidence, domain-expected-type expansion, static/dynamic preference scoring, WordNet sense overlap, and a knowledge-graph co-occurrence tiebreaker when top candidates are within margin.
**Public API:** `disambiguate/3`, `disambiguate_single/3`, `requires_inference?/1`, `infer_type_with_type_inferrer/3`, `introduction_confidence/3`.
**Mismatches:** `infer_type_with_type_inferrer/3` is public with only one internal caller. `introduction_confidence/3`'s doc claim "no hardcoded token matching" is defensible but borderline — the PRON/VERB sequence patterns it matches against are themselves hardcoded, just applied to model-produced tags.

### Brain.Analysis.ContextualEntityInferrer
**File:** analysis/contextual_entity_inferrer.ex
**Purpose:** Multi-pass IS-A narrowing of generic entity types per candidate intent, scored via 5 weighted signals (TypeInferrer, POS context, syntactic role, hierarchy compatibility, Atlas neighbor match), applying the winning hypothesis and reporting acceptance back for learning.
**Public API:** `infer/5`.
**Mismatches:** Moduledoc's "Signals" list documents 4 signals as "no string matching"; the actual code scores **5** (omitting hierarchy compatibility from the list), and a hardcoded domain→slot-type `case` mapping exists that the "no string matching" framing doesn't account for.

### Brain.Analysis.AnaphoraResolver
**File:** analysis/anaphora_resolver.ex
**Purpose:** Resolves pronoun/demonstrative references by Gazetteer lookup + conversation-history candidate scoring (recency decay × type compatibility).
**Public API:** `resolve/2`, `resolve_and_substitute/2`.
**Mismatches:** None.

### Brain.Analysis.BacktrackController
**File:** analysis/backtrack_controller.ex
**Purpose:** Struct-based controller limiting backtracking to a fixed budget (2), applying a fixed activation penalty, detecting oscillation, building clarification prompts on exhaustion.
**Public API:** `new/1`, `check_for_contradictions/1`, `attempt_backtrack/3`, `stats/1`, `should_clarify?/1`, `max_backtracks/0`, `backtrack_cost/0`.
**Mismatches:** `check_entity_mismatch/1` is a hardcoded `:ok` stub — see Executive Summary #3.

### Brain.Analysis.ActivationPool
**File:** analysis/activation_pool.ex
**Purpose:** Normalizes interpretation activations to sum ≤1.0; applies diminishing-returns boosts with per-source caps and an anti-stacking multiplier.
**Public API:** `normalize/1`, `normalize_with_alternatives/1`, `apply_boost/3`, `apply_stacked_boosts/2`, `max_boost_for/1`, `approaching_limit?/2`, `stats/1`, `diminishing_threshold/0`.
**Mismatches:** None.

### Brain.Analysis.AnalyzerCalibration
**File:** analysis/analyzer_calibration.ex
**Purpose:** Tracks per-analyzer confidence-bucket accuracy via EMA, multiplying raw confidence by historical bucket accuracy.
**Public API:** `start_link/1`, `calibrate/2`, `track_outcome/3`, `get_calibration_error/1`, `get_bucket_accuracy/2`, `recalibrate/1`, `stats/0`, `is_dominating?/2`, `tracked_analyzers/0`, `ready?/0`.
**Mismatches:** `is_dominating?/2`'s win-rate ETS entry is only ever zero-initialized, never updated — always `false` in practice.

### Brain.Analysis.ConsistencyChecker
**File:** analysis/consistency_checker.ex
**Purpose:** Compares up to 4 independent classification signals against the final chosen intent, computing agreement/dissent and severity, with optional telemetry/log/PubSub reporting.
**Public API:** `check/1`, `check_and_report/2`.
**Mismatches:** None (moduledoc's "LSTM/arbitrator removed" note is an accurate historical disclosure).

### Brain.Analysis.ContextAccumulator
**File:** analysis/context_accumulator.ex
**Purpose:** Fuses `{source, value, confidence}` signals via log-odds combination weighted by entropy-derived relevance, computes a Dempster-Shafer-style pairwise conflict measure, queries Memory/Graph for episodic and practical-adaptation context.
**Public API:** `new/2`, `add_signal/4`, `accumulate/1`, `effective_confidence/1`, `should_hedge?/1`, `dominant_strategy/1`, `interlocutor_adaptation/2`, `memory_context/1`, plus `@doc false` math internals.
**Mismatches:** None (the "Dempster-Shafer" conflict measure is a simplified pairwise approximation, not full DS theory, but the doc doesn't overclaim this).

### Brain.Analysis.ContextResolver
**File:** analysis/context_resolver.ex
**Purpose:** Fills missing slots from history (recency-decayed), `UserModelStore` (confidence-gated), then a static profile map, in that priority order.
**Public API:** `resolve/2`, `extract_context/3`, `generate_clarification_prompts/2`.
**Mismatches:** `extract_context/3` and `generate_clarification_prompts/2` have no production callers (only `resolve/2` is called from `pipeline.ex:620`) — moduledoc presents all three as equally live.

### Brain.Analysis.ConsistencyChecker / EntityGraphEnricher / EventExtractor / EventLinker / EventPatterns / FeatureExtractor family

### Brain.Analysis.EntityGraphEnricher
**File:** analysis/entity_graph_enricher.ex
**Purpose:** Batch entity enrichment via one `Reader.entity_context/2` query, plus pairwise relationship-path computation (capped at 10 pairs) and a `familiarity_score/1` helper.
**Public API:** `enrich/1`, `familiarity_score/1`.
**Mismatches:** None — heavily used in production, correctly degrades when Atlas is unavailable.

### Brain.Analysis.EventExtractor
**File:** analysis/event_extractor.ex
**Purpose:** Nx-tensor-based actor-verb-object event extraction (real GPU-capable ops for position-finding), plain-Elixir lemmatization/tense/confidence scoring.
**Public API:** `extract/2`, `extract_parallel/2`, `process_in_batches/3`, tensor primitives (`find_verb_positions/1`, etc.), `pos_tags_to_tensor/1`, `tensor_to_indices/1`, `exla_available?/0`, `backend_info/0`.
**Mismatches:** `compute_event_confidence/3` (a `defn`) is dead — the real path uses an independently-implemented plain-Elixir equivalent. `match_pattern/2` is never called — extraction is pure nearest-neighbor search, not pattern matching, despite the `EventPatterns` framing implying otherwise. `batch_find_verb_positions/1` is exercised only by a benchmark test.

### Brain.Analysis.EventLinker
**File:** analysis/event_linker.ex
**Purpose:** Assigns semantic argument roles via the `:event_argument_role` MicroClassifier, tags temporal arguments, attempts temporal-order and sub-event detection between events.
**Public API:** `link/4`, `assign_argument_roles/4`.
**Mismatches:** `temporal_contains?/2` is a hardcoded `false` stub — sub-event/containment links never form (Executive Summary #3). Temporal-order direction is actually determined by token position, not by the temporal values themselves, despite the doc implying the latter.

### Brain.Analysis.EventPatterns
**File:** analysis/event_patterns.ex
**Purpose:** Compile-time-loaded POS-sequence event patterns and POS↔index mappings.
**Public API:** ~20 functions covering pattern access, POS↔index conversion, tensor construction, role-membership queries.
**Mismatches:** Almost the entire pattern/tensor API is exercised only by tests; production only ever reaches `tagged_tokens_to_indices/1` and `pos_to_index/1` — the JSON-defined "patterns" concept the moduledoc centers on isn't consulted by the live extraction path at all (see `EventExtractor` above).

### Brain.Analysis.FeatureExtractor
**File:** analysis/feature_extractor.ex
**Purpose:** Orchestrates `WordFeatures.extract/1` then `ChunkFeatures.extract/2`.
**Public API:** `extract/1`, `extract_vector/1`, `extract_batch/1`, `vector_dimension/0`, `similarity/2`.
**Mismatches:** `extract_vector/1`, `extract_batch/1`, `similarity/2` have zero callers — every real call site uses `extract/1` directly.

### Brain.Analysis.FeatureExtractor.ChunkFeatures
**File:** analysis/feature_extractor/chunk_features.ex
**Purpose:** Aggregates per-word features + `ChunkAnalysis` signals into a flat vector spanning 23 documented groups (14 base + 9 enrichment, tail-appended to preserve offsets).
**Public API:** `extract/2`, `vector_dimension/0`.
**Mismatches:** None — one of the most doc-accurate modules audited.

### Brain.Analysis.FeatureExtractor.WordFeatures
**File:** analysis/feature_extractor/word_features.ex
**Purpose:** Per-word feature vector — POS one-hot, WordNet-domain one-hot, hypernym depth/polysemy/OOV/length scalars — with lexicon lookups only for content words.
**Public API:** `extract/1`, `extract_content_words/1`, `vector_dimension/0`.
**Mismatches:** None.

### Brain.Analysis.FeatureExtractor.EnrichmentFeatures
**File:** analysis/feature_extractor/enrichment_features.ex
**Purpose:** The 9 "enrichment" feature groups (wh-target type, time typology, POS-conditional supersense fingerprints, ConceptNet edge types, selectional preferences, subcategorization frame, discourse markers, speech-act×wh interaction, entity-type semantics) — designed to be identity-invariant.
**Public API:** ~18 functions, one pair per group (`*_dimension/0` counterparts).
**Mismatches:** `entity_type_semantics/1` **raises** if `TypeHierarchy` isn't ready, unlike every sibling function which degrades to zeros — an undisclosed behavioral asymmetry.

### Brain.Analysis.FollowupDetector
**File:** analysis/followup_detector.ex
**Purpose:** Decides whether a message continues a prior slot-filling intent via missing-slot-filler match, short prepositional phrase, or bare-location heuristics; merges entities via a hardcoded per-domain slot-type table.
**Public API:** `is_followup?/2`, `get_carried_context/2`, `merge_with_previous/2`.
**Mismatches:** Moduledoc claims POS-based grammatical detection "rather than keyword lists," but the non-location branch accepts any ≤5-word message as a slot filler — a pure word-count heuristic. A `profile`-based branch in `fill_slots_from_entities/5` is permanently unreachable since its only caller never passes a profile.

### Brain.Analysis.FramingDetector
**File:** analysis/framing_detector.ex
**Purpose:** GenServer classifying document framing via MicroClassifier + centroid-deviation, tracking per-source EMA drift in ETS.
**Public API:** `start_link/1`, `assess_document/2`, `assess_entity_framing/2`, `detect_drift/2`, `ready?/0`, `get_source_ema/1`.
**Mismatches:** Documented `secondary_frames` is always `[]` (classifier only returns one label). Documented `causal_attribution`/`dominant_lexical_domains` evidence fields are permanent hardcoded stubs, unlike the genuinely-computed `sentiment_skew`/`modality_skew`. Currently dormant in production (started but only exercised by tests) — honestly framed as a future integration seam in its own doc.

### Brain.Analysis.HeuristicStore
**File:** analysis/heuristic_store.ex
**Purpose:** GenServer + 4 ETS tables (global/world/cohort/user) for fast-path heuristic matching, with per-scope confidence caps.
**Public API:** `start_link/1`, `match_best/4`, `match_scope/3`, `add_heuristic/3`, `record_success/1`, `record_failure/1`, `get/1`, `list/1`, `ready?/1`, `stats/0`, `healthy?/1`, `deprecate/1`.
**Mismatches:** `list/1` has no `:world` clause (raises on `list(:world)`; `list(nil)` silently omits the world table). More significantly, `deprecate/1`, the `:record_outcome` handler, and `record_usage/1` all search only 3 of 4 scope tables, omitting `@world_table` — world-scoped heuristics' usage stats never update after creation, silently defeating deprecation for exactly the scope the moduledoc treats as central.

### Brain.Analysis.IntentUtils
**File:** analysis/intent_utils.ex
**Purpose:** Pure domain-prefix string comparison helper.
**Public API:** `same_domain_prefix?/2`, `domain_prefix/1`.
**Mismatches:** None.

### Brain.Analysis.Interpretation
**File:** analysis/interpretation.ex
**Purpose:** `Interpretation` struct implementing "competitive activation" (top analyzer wins, up to 5 runner-up alternatives stay warm for backtracking); also defines `AnalyzerResult`.
**Public API:** `new/4`, `from_analyzer_results/2`, `with_slots/2`, `with_entities/2`, `with_normalized_activation/2`, `with_heuristic/3`, `promote_alternative/1`, `has_missing_required?/1`, `missing_required/1`, `from_heuristic?/1`, `confidence_level/1`, `activation_headroom/1`; `AnalyzerResult.{new/4, with_calibration/3}`.
**Mismatches:** None.

### Brain.Analysis.LearningStore
**File:** analysis/learning_store.ex
**Purpose:** JSON-backed parameter store with admin locking, debounced disk saves, and feedback-driven confidence adjustment for response-optionality thresholds.
**Public API:** `start_link/1`, `get_params/1`, `get_all_params/0`, `update_params/3`, `record_feedback/2`, `lock_params/1`, `unlock_params/1`, `reset_to_defaults/1`, `get_stats/0`, `ready?/0`.
**Mismatches:** The documented `:name` start option is non-functional — every client function hardcodes `GenServer.call(__MODULE__, ...)`. "Threshold tuning from feedback" is narrower than implied: only 2 of 7 feedback types actually adjust thresholds.

### Brain.Analysis.NoveltyDetector
**File:** analysis/novelty_detector.ex
**Purpose:** Determines intent novelty (confidence/margin/unknown-entity) and researchability; KG-aware downweighting near existing beliefs unless a state-change is detected.
**Public API:** `is_novel?/3`, `is_substantive?/2`, `is_researchable?/3`, `maybe_kg_downweight/3`.
**Mismatches:** None against documented behavior (belief-lookup failures silently fall back, undisclosed but reasonable).

### Brain.Analysis.OutcomeLearner
**File:** analysis/outcome_learner.ex
**Purpose:** Assesses turn success/failure, feeds `HeuristicStore`/`AnalyzerCalibration`, extracts candidate patterns that become new heuristics once they recur enough.
**Public API:** `learn_from_outcome/3`, `assess_outcome/3`, `extract_pattern/1`, `determine_scope/3`.
**Mismatches:** Uses raw `Process.get/put` pattern-success counters instead of calling the purpose-built `OutcomeLearner.Store` (see below) — the two modules are unwired from each other despite `Store`'s moduledoc claiming otherwise. `determine_scope/3` ignores its `user_id`/`cohort_id`/`profile` parameters entirely.

### Brain.Analysis.OutcomeLearner.Store
**File:** analysis/outcome_learner/store.ex
**Purpose:** GenServer + ETS table meant to replace `OutcomeLearner`'s process-dictionary counters with cross-process-safe ones.
**Public API:** `start_link/1`, `get_count/1`, `increment/1`.
**Mismatches:** The moduledoc's central claim ("used by OutcomeLearner... replaces Process.get/put") is false — the process is alive in the supervisor and status dashboard, but `OutcomeLearner` never calls it. Functionally unwired to the module it claims to serve.

### Brain.Analysis.ProcessingTrace
**File:** analysis/processing_trace.ex
**Purpose:** Builds a full processing trace (chunking, racing-analyzer output, heuristic hits, backtracking, slot filling) flattened for UI display.
**Public API:** `from_interpretation/2`, `to_display_map/1`, `trace_processing/2`.
**Mismatches:** Zero callers outside its own test — no LiveView/controller currently wires it up despite the "for UI visualization" framing. `find_primary_chunk/1` reads a `:profile` field that `ChunkTrace` doesn't have — permanently `nil`, so that priority branch is dead.

### Brain.Analysis.Progress
**File:** analysis/progress.ex
**Purpose:** Defensive PubSub shim broadcasting analysis progress to a LiveView topic; silent no-op on malformed input.
**Public API:** `report/3`.
**Mismatches:** None.

### Brain.Analysis.RacingAnalyzer
**File:** analysis/racing_analyzer.ex
**Purpose:** Races several concurrent intent-classification strategies (heuristic, ML micro-classifier, pattern/keyword matchers, memory similarity, self-knowledge detector) as `Task`s, with three fast-path shortcuts.
**Public API:** `race/2`, `check_fast_path/4`, `check_fast_path/3` (deprecated shim).
**Mismatches:** `analyze_keywords/1` ignores its `text` argument, always returning a 0.0-confidence stub — the doc's "keyword triggers loaded from JSON" claim is only half-true (parsed, cached, never read). `apply_intent_safeguards/2` is a pure identity function reported to the UI as "corrected results," but corrects nothing.

### Brain.Analysis.ResponseGate
**File:** analysis/response_gate.ex
**Purpose:** Decides respond/optional/silent purely from classified speech acts and conversational timing/history — no raw string matching.
**Public API:** `evaluate/3`, `evaluate_speech_act/3`, `rapid_fire?/1`, `echo_repetition?/2`, `self_correction?/3`, `gratitude_loop?/2`, `recent_thanks?/1`, `has_recent_pattern?/2`.
**Mismatches:** None.

### Brain.Analysis.SelfKnowledgeAnalyzer
**File:** analysis/self_knowledge_analyzer.ex
**Purpose:** Detects meta-cognitive queries ("what do you know about me") via the ML pipeline with a keyword-heuristic fallback.
**Public API:** `analyze/2`, `is_self_knowledge_query?/1`, `build_self_knowledge_assessment/1`, `detect_meta_intent/1`, `detect_query_type/1`, `meta_intents/0` (documented stub, always `[]`).
**Mismatches:** `profile`-aware branches of `is_meta_intent?/2`/`intent_to_query_type/2` are unreachable — no caller ever passes a profile (same pattern as `FollowupDetector`).

### Brain.Analysis.SemanticRoleLabeler
**File:** analysis/semantic_role_labeler.ex
**Purpose:** BIO-tag + token sequences → predicate-argument frames, with entity linking and triple flattening for KG ingestion.
**Public API:** `label/3`, `extract_spans/1`, `to_triples/1`.
**Mismatches:** Doc/example shows a `%PredicateFrame{}` struct return; no such struct exists — `label/3` returns plain maps.

### Brain.Analysis.SlotDetector
**File:** analysis/slot_detector.ex
**Purpose:** JSON-defined slot schemas (with parent/domain fallback), required/optional slot filling via type-hierarchy-compatible matching, clarification prompts, intent suggestion from entities.
**Public API:** `detect/2`, `get_schema/1`, `get_entity_types_for_intent/1`, `list_schemas/0`, `suggest_intent_from_entities/1`, `get_clarification_prompt/2`, `get_clarification_prompts/2`.
**Mismatches:** `list_schemas/0` has no callers. Schema sourcing is broader than documented — also deep-merges schemas dynamically registered via `Dispatcher.service_schemas()`.

### Brain.Analysis.SpeechActClassifier
**File:** analysis/speech_act_classifier.ex
**Purpose:** Confidence-weighted voting among 5 voters (lexicon-regex, structural, statistical-classifier-labeled-"keyword", pragmatic-markers, memory) to classify pragmatic function; `refine_with_intent/2-3` splices in a later intent classification without re-running cheap voters.
**Public API:** `classify/2`, `analyze/2`, `refine_with_intent/2`, `refine_with_intent/3`.
**Mismatches:** Moduledoc's "Keyword Analysis" voter actually delegates to a statistical TF-IDF/centroid model (`SpeechActClassifierSimple`), not a keyword lookup — mislabeled. The real regex/lexicon voter isn't in the moduledoc's voter list at all: 5 voters run, 4 are documented, and one of the 4 is misdescribed.

### Brain.Analysis.TypeHierarchy
**File:** analysis/type_hierarchy.ex
**Purpose:** GenServer/ETS IS-A lookup merging a static JSON hierarchy with Atlas-learned edges, WordNet-hypernym fallback.
**Public API:** `start_link/1`, `is_a?/2`, `specializations/1`, `compatible?/2`, `narrowing_candidates/2`, `parent_type?/1`, `parent_of/1`, `all_types/0`, `parent_types/0`, `config/1-2`, `ready?/1`, `reload/1`, `sync_to_atlas/1`.
**Mismatches:** None.

### Brain.Analysis.Types.Event
**File:** analysis/types/event.ex
**Purpose:** `Event` struct + construction/query helpers.
**Public API:** `new/2`, `complete?/1`, `imperative?/1`, `action_lemma/1`, `to_description/1`.
**Mismatches:** Moduledoc's "no string matching or regex" claim actually describes the sibling `event_extractor.ex`, not this struct file — misplaced, though not itself contradicted.

### Brain.Analysis.Types.IntentReviewCandidate
**File:** analysis/types/intent_review_candidate.ex
**Purpose:** Struct for a human-review-flagged utterance with a full `:pending → :approved/:rejected/:deferred` lifecycle.
**Public API:** `new/4`, `update_annotation/2`, `approve/4`, `reject/2`, `defer/2`.
**Mismatches:** None.

### Remaining `Brain.Analysis.*` modules (verified, no notable mismatches beyond what's noted)

- **Consistency/priority/misc:** `ChunkPriority`, `IntentUtils`, `Progress`, `ResponseGate`, `TypeHierarchy`, `Types.Event`, `Types.IntentReviewCandidate` — all match their documentation.
- **ContextResolver / ContextualEntityInferrer / EntityDisambiguator / EntityGraphEnricher / AnaphoraResolver** — see entries above.

---

## Brain.ML

49 modules — classical NLP models (no LLM), embeddings, training pipelines, the Ouro sidecar integration.

### Brain.ML.Generation (seam)
**File:** ml/generation.ex, ml/generation/{backend,null,openai_compatible,ouro_sidecar}.ex
**Purpose:** The pluggable generation backend seam — resolves a configured backend key to a module implementing `Brain.ML.Generation.Backend`'s `generate/2`/`ready?/0`/`name/0`/`children/0`. Three real implementations: `Null` (honest no-op), `OpenAICompatible` (genuine HTTP client, real `Req.post`/`Req.get` calls), `OuroSidecar` (delegates to the real `Ouro.Model` GenServer).
**Mismatches:** None in any of the four files — this is a clean, honestly-documented seam.

### Brain.ML.Ouro.* (sidecar integration)
**Files:** ml/ouro/{client,model,model_downloader,sidecar_launcher,spec,tokenizer}.ex
**Purpose:** Full integration with a Python "Ouro" LoopLM sidecar — HTTP client with retry semantics, a GenServer wrapping sidecar-or-in-process-Bumblebee backends, model artifact downloading with safetensors/LFS-pointer validation, process supervision for the Python subprocess (with reuse-if-already-healthy logic), a real Bumblebee `ModelSpec` for the 24-layer weight-shared-recurrence architecture, and a patched Llama-based tokenizer with ChatML special tokens.
**Mismatches:**
- `Client.list_models/1` — no callers anywhere, dead code.
- `SidecarLauncher`'s private `diagnostics/1` comment overstates test usage — no test currently exercises the `:diagnostics` message.
- Otherwise clean: health-check gating, reuse-if-healthy, safetensors validation, and the recurrence/sandwich-norm architecture all match their docs precisely.

### Brain.ML.Poincare.* (hyperbolic embeddings)
**Files:** ml/poincare/{distance,embeddings,optimizer}.ex
**Purpose:** Poincaré-ball hyperbolic embeddings for hierarchical (child, parent) entity pairs, trained via Riemannian SGD with negative sampling.
**Mismatches:** None functional — `Optimizer` reimplements the same distance formula as private helpers rather than calling `Distance.distance/2` (reuse opportunity, not a bug).

### Brain.ML.Lexicon (WordNet core) & Brain.ML.Lexicon.WordNetParser
**Files:** ml/lexicon.ex, ml/lexicon/wordnet_parser.ex
**Purpose:** GenServer parsing real WordNet 3.1 Prolog files into ETS at init; serves synonym/hypernym/antonym/Wu-Palmer-similarity lookups. Crashes loudly (by design) if required files are missing.
**Mismatches:** `WordNetParser.parse_instances/1`, `parse_similar/1`, `parse_derivations/1` are correctly implemented but never invoked by `Lexicon.init/1` or anything else — `wn_ins.pl`/`wn_sim.pl`/`wn_der.pl` data is fully parseable but never parsed in practice. Note: this module's `:table_prefix` isolation is implemented **correctly** — contrast with `Brain.ML.Gazetteer` below.

### Brain.ML.Gazetteer
**File:** ml/gazetteer.ex
**Purpose:** GenServer-managed ETS entity gazetteer — exact/prefix/span lookups, contextual type ranking, admin CRUD, per-world overlay namespacing.
**Mismatches:** The documented `:table_prefix` test-isolation feature is broken — ~25 of ~29 read/write call sites reference hardcoded global table-name attributes instead of the per-instance `state.tables`, so a prefixed GenServer still mutates shared global state. This is a live, exercised path (used by `test/support/genserver_sandbox.ex`), not theoretical. `clear_by_type/1` has zero callers.

### Brain.ML.KnowledgeGraph.* (Embedder / EntityVectorCache / PredicateNormalizer / TripleScorer)
**Files:** ml/knowledge_graph/{embedder,entity_vector_cache,predicate_normalizer,triple_scorer}.ex
**Purpose:** A real trained BiLSTM (`TripleScorer`) scoring KG triple validity; entity-embedding extraction meant to reuse its truncated architecture; an ETS cache for those embeddings; a static predicate-vocabulary alignment table.
**Mismatches:** `Embedder.build_embedding_model/1` doesn't actually truncate the model as its own doc claims (it's an identity wrapper around the full scorer including the sigmoid head) — but this is dormant since `EntityVectorCache` (the only real consumer) correctly calls the separately-implemented, correctly-truncated `build_extraction_model/2` instead. `PredicateNormalizer` and `TripleScorer` themselves have no mismatches.

### Brain.ML.MicroClassifiers
**File:** ml/micro_classifiers.ex
**Purpose:** GenServer loading ~17 pre-trained TF-IDF/feature-vector classifiers from `.term` files at startup; missing files degrade to "absent," not a crash.
**Mismatches:** Moduledoc's classifier list omits 2 real, wired-up classifiers (`:event_argument_role`, `:framing_class`) — stale documentation, not a functional gap.

### Brain.ML.ModelPreflight / Brain.ML.ModelStore
**Files:** ml/model_preflight.ex, ml/model_store.ex
**Purpose:** `ModelPreflight` validates every stored model's structural integrity (real deserialization + key checks, no silent skips). `ModelStore` is a real S3/MinIO client, entirely gated behind `enabled?/0` (default `false`, genuine no-op when disabled).
**Mismatches:** `ModelPreflight`'s classifier-name list is a hand-maintained duplicate of `MicroClassifiers`' list (flagged in-code as a sync risk, currently in sync). No functional mismatches in either module.

### Brain.ML.Trainer
**File:** ml/trainer.ex
**Purpose:** Orchestrates the full classical-NLP training pipeline (intent classifier → entity model → gazetteer data → sentiment classifier).
**Mismatches:** See Executive Summary #3 — `train_intent_classifier/2` writes a constant `1.0` IDF for every word into the saved embedder model, discarding real computed IDF, undisclosed. `train_svm_classifier/2`'s doc promises "simple nearest neighbor," but the function does no classification at all — legacy/test-only scaffolding. `save_models/2` is a no-op logging only a deprecation warning, despite its doc.

### Brain.ML.TrainingData.* (Catalog / Diagnostics / RevisionLog / Schemas / SourceDescriptors)
**Files:** ml/training_data/{catalog,diagnostics,revision_log,schemas,source_descriptors}.ex
**Purpose:** Single gateway for every registered training-data source file — validated, atomically-written, revision-logged, PubSub-broadcasting reads/writes; cross-source health diagnostics; append-only edit audit log.
**Mismatches:** `Catalog` has a stale `# Phase 2 — stub for now` comment sitting above fully-implemented write logic — misrepresents working code as incomplete. `Diagnostics.orphan_report/0`'s doc promises a `:registry_without_templates` key that was never implemented.

### Remaining `Brain.ML.*` modules

- **CorpusManager, Evaluation, EvaluationStore, FeatureVectorClassifier, InformalExpansions, TrainingExampleBuffer, TrainingServer, WeightOptimizer, WeightOptimizer.Tracker** — all verified accurate against their docs (a couple of minor doc-completeness notes, no functional mismatches): `EvaluationStore.save/1` silently swallows Atlas write failures via a trailing rescue; `WeightOptimizer.Tracker`'s "mirrors Mix.Tasks.TrainMicro" claim is a duplicated literal, not a real reference (currently in sync, latent drift risk).
- **DataLoaders** — `load_smalltalk_responses/1` and `load_negative_examples/1`/`load_negative_file/2` are fully implemented but dead (Executive Summary #2); `TemplateStore` re-implements the smalltalk load independently rather than reusing this loader.
- **EntityExtractor** — moduledoc implies BIO-model extraction is part of the unified default path; it's actually only reached via the separately-named `extract_entities_with_model/2`, called from one place.
- **EntityTrainer** — `finalize_entity/1` hardcodes `confidence: 0.8` for every BIO-extracted entity regardless of actual model probability, undisclosed.
- **NLPPipeline** — `classify_intent/1` silently converts any internal exception into a generic low-confidence result, indistinguishable from genuine low confidence.
- **Tokenizer** — `expand_contractions/1`'s doc describes suffix-pattern rules that are never actually used by that function (it delegates to `InformalExpansions`' lookup table instead); the suffix rules are used by a *different* function (`check_contraction/1`) for a different purpose.
- **POSTagger, SentimentClassifierSimple, SimpleClassifier, SpeechActClassifierSimple** — verified accurate; `POSTagger.blend_graph_transitions/1` has no callers (the real training path calls `update_weights/3` directly); `SentimentClassifierSimple.classify/2` lacks the `try/catch` its sibling `SpeechActClassifierSimple` has, an asymmetry worth a look if the two are meant to behave identically.

---

## Brain.Response

30 modules — discourse planning, content specification, surface realization (lattice/Ouro/template), response quality.

### Brain.Response.Generator
**File:** response/generator.ex
**Purpose:** Main entry point with two largely independent paths: a legacy pipeline (`generate/3` and friends — template/memory/synthesizer cascades) and `generate_via_synthesis/5`, the actual production path (delegates to `RefinementLoop.generate/2`).
**Mismatches:** The moduledoc's 4-step "Retrieve → Synthesize → Compose → Refine" framing only describes the legacy path; `generate_via_synthesis/5` — the function actually driving live responses — isn't described at all. `generate_with_path/3`, `generate_with_context/4`, `build_template_context/2`, `format_code_snippet/2` have no callers outside their own tests.

### Brain.Response.RefinementLoop
**File:** response/refinement_loop.ex
**Purpose:** Orchestrates plan → specify → realize → evaluate, re-running the weakest-scoring stage up to 3 times until score converges (≥0.7) or the cap is hit.
**Mismatches:** None significant — some return-value variants (`:ouro_dry_run`, `:silence_preferred`) aren't in the `@doc` but are accurate implementation details.

### Brain.Response.SurfaceRealizer
**File:** response/surface_realizer.ex
**Purpose:** Routes primitive realization to lattice/Ouro/synthesizer/template with cascading fallback.
**Mismatches:** **Significant** — the moduledoc states "If Ouro is not loaded, the system raises — there is no template fallback." This is false: every branch (lattice, Ouro, template) has a fallback cascade to the synthesizer or an enriched-placeholder template; nothing raises except a specific non-`:null` backend-unavailable case, which returns an error tuple rather than raising. The moduledoc describes a superseded, earlier design.

### Brain.Response.OuroRealizer
**File:** response/ouro_realizer.ex
**Purpose:** Builds a ChatML packet and generates via the configured backend, then structurally validates with `ConstraintEnforcer`.
**Mismatches:** Moduledoc is written entirely in terms of "the Ouro LoopLM model," but the implementation explicitly routes through the backend-agnostic `Generation.generate/2` — it may not be talking to Ouro at all depending on config. Compounding this, `validate_and_return/2` hardcodes `metadata.model: "ouro-1.4b"` and `metadata.source: :ouro` unconditionally, so the returned metadata falsely claims Ouro produced output that may have come from OpenAI-compatible or another backend.

### Brain.Response.Synthesizer
**File:** response/synthesizer.ex
**Purpose:** Composes responses from domain-knowledge JSON frames + slot filling, plus a self-knowledge-assessment response path using real primitive composition (prefaces/hedges/uncertainty markers).
**Mismatches:** Moduledoc opens with "Instead of template-based responses, this module composes responses from primitives" — but the dominant code path (`synthesize_from_domain/5`) is itself template selection + slot filling from static JSON arrays, architecturally the same technique the doc claims to move away from. Real primitive composition is real, but confined to the self-knowledge-response path.

### Brain.Response.ResponseQuality
**File:** response/response_quality.ex
**Purpose:** Heuristic quality checker (length, topic mismatch, repetition, awkward starts, missing entity reference, generic-fallback) independent of the lattice pipeline; can "improve" low scorers via template swap.
**Mismatches:** Moduledoc's usage examples show a stale 2-tuple return for `improve/2`; the real implementation always returns a 3-tuple. `select_best_response/4`'s "diverse candidates" really means "existing templates for the intent," not diverse synthesis strategies.

### Brain.Response.TemplateStore
**File:** response/template_store.ex
**Purpose:** GenServer loading response templates from a consolidated file or per-intent legacy files, merging custom smalltalk responses, TF-IDF/embedding similarity ranking, condition-based selection, admin CRUD with periodic sync.
**Mismatches:** `data/customSmalltalkResponses_en.json` (which the moduledoc advertises as a merged source) **does not exist anywhere in the repo** — the load always fails and silently returns `%{}`, disabling the feature without surfacing an error (notable given the project's stated no-graceful-degradation policy). Also hardcodes plain-relative paths (unlike sibling modules that use a `priv_dir`-aware helper) — works only because the umbrella happens to be started with cwd at its root; would silently load zero templates under a different cwd (e.g. a release).

### Brain.Response.ChunkSegmenter
**File:** response/chunk_segmenter.ex
**Purpose:** Splits templates into 6 chunk types for blending.
**Mismatches:** Moduledoc says classification uses "sentence boundaries and embedding-based classification"; the primary mechanism is actually keyword/heuristic matching, with embeddings only a secondary tiebreaker for otherwise-unclassified text. `tag_tone/1` is dead code, duplicated (differently) by an unrelated `auto_tag_tone/2` in a mix task.

### Brain.Response.ChunkCompatibility
**File:** response/chunk_compatibility.ex
**Purpose:** Learns chunk co-occurrence statistics for blend scoring, with a static fallback table when unlearned.
**Mismatches:** `find_compatible/2`, `get_cooccurrence_matrix/0` are dead code. Moduledoc's example score (0.85) doesn't reflect the common pre-`learn/1` case, which uses the static fallback (default 0.3) — undisclosed.

### Brain.Response.ContentSpecifier
**File:** response/content_specifier.ex
**Purpose:** Fills primitive content maps from facts/beliefs/analysis based on `{type, variant}`.
**Mismatches:** `check_action_capability/2` is a hardcoded `:unknown` stub (Executive Summary #3), consistent with an independent hardcode in `DiscoursePlanner`, undisclosed in either.

### Brain.Response.DecompressorCollector
**File:** response/decompressor_collector.ex
**Purpose:** Collects live generation examples for offline training, periodically flushing to JSONL and feeding `PhraseInventory`.
**Mismatches:** `feed_to_phrase_inventory/1` fabricates placeholder tone metadata (`"neutral"`, zero vector) for every fragment fed in — undisclosed, meaning runtime-collected fragments carry synthetic rather than real tone data compared to authored ones.

### Brain.Response.FactRetriever
**File:** response/fact_retriever.ex
**Purpose:** Entity + keyword fact lookup layer over `FactDatabase`, exception-safe throughout.
**Mismatches:** `get_relevant_facts/1` is effectively an internal helper exposed as public API with no external consumer. `format_single_fact/1` isn't reused by `Synthesizer`, which reimplements its own private equivalent despite the shared-sounding name.

### Brain.Response.LatticeScorer / PhraseInventory / PhraseLatticeRealizer / TransitionModel
**Files:** response/{lattice_scorer,phrase_inventory,phrase_lattice_realizer,transition_model}.ex
**Purpose:** The lattice-based realization path — fragment scoring (feature vector + tone + intent bonus), fragment storage/indexing, beam search across fragment positions, and inter-fragment transition smoothness scoring.
**Mismatches:** `LatticeScorer.load_weights/0` is dead — production loads weights via a bypassed private function instead. `TransitionModel.build_scores/1`/`save_scores/1` have no discoverable automated entry point (offline-training-only, no mix task exists for them currently).

### Brain.Response.MemoryAugmented
**File:** response/memory_augmented.ex
**Purpose:** Generates responses by reusing outcome patterns from similar past episodes.
**Mismatches:** `find_similar_episodes/3` (documented as a debug/inspection API) has zero callers anywhere.

### Remaining `Brain.Response.*` modules

- **ChunkCompatibility, ConditionEvaluator, ConstraintEnforcer, ContextBuilder, DiscoursePlanner, Enricher, Formatting, RealizationPacket, ResponseEvaluator, ResponseSystemRouter, SemanticFactRetriever, TemplateBlender** — verified accurate against their docs, with minor notes: `Enricher.can_enrich?/2`/`substitute_all/3`/`get_enrichment_metadata/1` are test-only despite being framed as normal-flow helpers; `ConditionEvaluator` has 2 undocumented condition types (`service_missing`, `service_error`); `RealizationPacket` writes an undisclosed prompt-text debug dump in addition to its documented JSON dumps.
- **Primitive, PrimitiveTypes, RealizationPacket** — struct/schema modules, no mismatches.

---

## Brain.Knowledge

14 modules — the research/learning subsystem (academic paper ingestion, contradiction detection, review queue, source reliability).

### Brain.Knowledge.Academic.{Arxiv, OpenAlex, SemanticScholar}
**Purpose:** Real HTTP clients for three academic APIs (arXiv Atom feed, OpenAlex REST, Semantic Scholar Graph API) — genuine network calls, XML/JSON parsing, rate limiting.
**Mismatches:** `OpenAlex.get_references/2`'s doc says it returns papers the given work references; the implementation actually filters by `cited_by:`, i.e. it returns the same relationship as `get_citations/2` — the function does not implement what its name/doc claim (the code even comments that OpenAlex lacks a real references endpoint). `SemanticScholar`'s equivalent correctly distinguishes citations vs. references. `Arxiv`'s doc cites a 3-second rate limit; the code uses 1 second.

### Brain.Knowledge.LearningCenter
**File:** knowledge/learning_center.ex
**Purpose:** GenServer orchestrating research sessions — dispatches `ResearchAgent` tasks, routes findings through hypothesis testing or the review queue.
**Mismatches:** `test_predictions/1`'s `apply_prediction_evaluation/1` computes a prediction's expected outcome but never lets it influence the result — both branches of each conditional call the identical generic evaluator regardless, so "structured prediction evaluation" degrades to plain evidence-count evaluation.

### Brain.Knowledge.ReviewQueue
**File:** knowledge/review_queue.ex
**Purpose:** ETS-backed review queue with Atlas write-through persistence and an auto-approval path.
**Mismatches:** Moduledoc's "Persistence to disk" claim is actually Atlas persistence, not a disk file — misleading wording, though functionally durable. Contains a changelog-flavored warning log referencing a past bug fix, undisclosed context for future readers.

### Brain.Knowledge.SourceReliability
**File:** knowledge/source_reliability.ex
**Purpose:** Per-domain reliability/bias/trust-tier tracking, bootstrapped from JSON, merged with Atlas-learned data.
**Mismatches:** `persist/0` is a documented-but-no-op stub (Executive Summary #1) — real persistence already happens inside `record_feedback/3`.

### Brain.Knowledge.ResearchAgent
**File:** knowledge/research_agent.ex
**Purpose:** Expands a research goal into queries, fetches from web/academic/task/mock sources, cleans HTML, runs the comprehension-gated NLP pipeline.
**Mismatches:** Moduledoc's "fetch from web sources" doesn't disclose that `:web` only ever queries two fixed domains (Wikipedia, Britannica) via URL templates — not general web search. `:academic` fetching has a side effect (ingesting papers into JTMS/BeliefStore) inside what looks like a pure fetch step, undisclosed.

### Remaining `Brain.Knowledge.*` modules

- **ContradictionDetector, Corroborator, HtmlProcessor, LearningTriggers, Types** — verified accurate; `Corroborator.blend_with_kg_similarity/3` is a no-op despite its name implying real KG-aware blending (both branches return the same value).

---

## Brain.Epistemic

10 modules — the JTMS belief-maintenance subsystem.

### Brain.Epistemic.JTMS
**File:** epistemic/jtms.ex
**Purpose:** A genuine Forbus & de Kleer-style JTMS with per-world process isolation (Registry + DynamicSupervisor), label propagation, and contradiction-triggered callbacks.
**Mismatches:** None — one of the most solidly implemented modules in the entire audit.

### Brain.Epistemic.BeliefStore
**File:** epistemic/belief_store.ex
**Purpose:** GenServer storing beliefs with subject/predicate/user/world indices, write-through Atlas persistence, confidence decay, and JTMS-node creation per belief.
**Mismatches:** `persist/0` is a documented-but-no-op stub (Executive Summary #1). A doc comment claims `add_belief_with_authority/5` "replaces the old `add_admin_belief/4`," a function that doesn't exist anywhere — a dangling historical reference.

### Brain.Epistemic.UserModelStore
**File:** epistemic/user_model.ex
**Purpose:** Per-user fact/confidence/provenance model, Atlas write-through, bidirectional sync with `BeliefStore`.
**Mismatches:** `persist/0` is a documented-but-no-op stub (Executive Summary #1).

### Brain.Epistemic.ContradictionHandler
**File:** epistemic/contradiction_handler.ex
**Purpose:** Registers as the JTMS contradiction callback; resolves via built-in rules or configurable strategies (`:auto_least_confident`, `:auto_most_recent`, `:manual`, `:hybrid`).
**Mismatches:** The `assumption_metadata` state field is never populated by any code path — `:auto_least_confident`/`:auto_most_recent` always read empty metadata and fall back to defaults, so they can never actually differentiate assumptions by real confidence/recency in practice. Undisclosed.

### Brain.Epistemic.StanceTracker
**File:** epistemic/stance_tracker.ex
**Purpose:** Tracks per-conversation stance drift, warning when the system's position seems to be shifting toward the user's.
**Mismatches:** `check_drift/3`'s direction logic compares the user's own average stance against the system's *initial* position, not the system's current position against the user's — a metric that may not actually measure "is the system drifting toward the user" despite that being the stated purpose.

### Remaining `Brain.Epistemic.*` modules

- **ConsolidationBridge, DisclosurePolicy, JTMS.Supervisor, SourceAuthority** — verified accurate; `SourceAuthority.persist/0` is a positive counter-example (genuinely persists), unlike its `BeliefStore`/`UserModelStore` siblings.

---

## Brain.Services

12 modules — external/physical-world integrations, dispatched by intent.

### Brain.Services.Dispatcher
**File:** services/dispatcher.ex
**Purpose:** Routes classified intents to a registered service by exact/domain/prefix match, gated on credentials.
**Mismatches:** `@registered_services` omits `Brain.Services.LocalScheduler` entirely (Executive Summary #2) — a fully-implemented, 10-intent service that can never be reached.

### Brain.Services.HomeAssistant (+ CapabilityRegistry, Discovery, EntityMapper, StateFormatter)
**File:** services/home_assistant.ex and services/home_assistant/*.ex
**Purpose:** Home Assistant REST integration — dynamic entity/service resolution, state reads, and (formerly) actuation.
**Mismatches:** See Executive Summary #4 — actuation is deliberately, completely disabled (`call_service/5` always returns `{:error, :actuation_disabled}`) as of a 2026-07-03 remediation, but the moduledoc still describes a live, working integration. Read paths remain fully live and ungated (60s cache only), an asymmetry the docs don't call out. `CapabilityRegistry.refresh/0` has no callers.

### Brain.Services.LocalScheduler
**File:** services/local_scheduler.ex
**Purpose:** In-process alarm/reminder/timer/calendar fallback to Home Assistant.
**Mismatches:** Currently dead code app-wide — see Dispatcher above. Intent matching uses `String.contains?` rather than equality (latent looseness, not currently exploitable since nothing routes to it).

### Brain.Services.CredentialVault
**File:** services/credential_vault.ex
**Purpose:** GenServer holding encrypted, world-scoped credentials in ETS, backed by Atlas.
**Mismatches:** Every single store/delete re-persists the **entire** ETS table to Postgres rather than just the changed row — a scalability footgun not mentioned in the docs.

### Remaining `Brain.Services.*` modules

- **Cache, Service, SystemStatus, Weather** — verified accurate; `Service.implements?/1` has no callers; `SystemStatus`'s rescue-to-"unavailable" pattern is the one place in this subsystem that *does* degrade gracefully, an inconsistency worth a maintainer decision given the project's general no-graceful-degradation stance (arguably fine here since the stakes are lower than physical actuation).

---

## Brain.Code

9 modules — source-code understanding (parsing, symbol extraction, relationship mapping, NL summarization).

### Brain.Code.LanguageGrammar & Brain.Code.Parser
**Files:** code/language_grammar.ex, code/parser.ex
**Purpose:** Nominally tree-sitter-based parsing; in the current build, tree-sitter isn't a compiled dependency, so both always fall back to regex/line-based extraction.
**Mismatches:** `LanguageGrammar` discloses this prominently (a full "Degraded Mode" section in its moduledoc). `Parser` — which does the actual parsing callers use — does **not** disclose it anywhere in its own moduledoc, despite depending on the same fallback (Executive Summary #4).

### Brain.Code.RelationMapper
**File:** code/relation_mapper.ex
**Purpose:** Detects call/extends/instantiates relationships from an AST via per-language pattern tables.
**Mismatches:** Moduledoc describes "two passes" performed by this module; the actual implementation is a single recursive walk (the two-pass structure exists at the pipeline level — extract symbols, then map relations — not within this module). The `:contains`/`:uses` relation types are documented but never actually produced — only `:calls`/`:called_by`, `:extends`, `:instantiates` are emitted.

### Brain.Code.Tokenizer
**File:** code/tokenizer.ex
**Purpose:** Hand-written recursive-descent tokenizer for 9 languages.
**Mismatches:** Fully standalone — zero external callers anywhere; `Parser`/`SymbolExtractor`/`Pipeline` each do their own regex-based extraction instead of using this tokenizer, undisclosed.

### Remaining `Brain.Code.*` modules

- **CodeGazetteer, Pipeline, QueryHandler, Summarizer, SymbolExtractor** — verified accurate; `CodeGazetteer`'s moduledoc documents only 10 of the 14 entity types `entity_types/0` actually returns. `QueryHandler` and `Summarizer` have overlapping-but-distinct explanation-generation responsibilities that neither moduledoc cross-references.

---

## Small subsystems

Lexicon, Memory, Graph, Subprocesses, FactDatabase, Lattice, and standalone singletons — 35 modules total.

### lexicon/

- **ConceptNet, ConceptNetParser, Loader, UserDefined** — verified accurate.
- **SenseDrift** — fully implemented, entirely unused anywhere in the codebase (Executive Summary #2), despite `Brain.Lexicon`'s facade moduledoc advertising it as a supported capability.
- **UserDefined**'s doc claims its ETS table is created by `Loader`; `Loader.init/1` only actually creates the sense-key and ConceptNet tables — the user-defined table must be created elsewhere, an inaccurate cross-reference.

### memory/

- **Consolidation** — clusters episodes into semantic facts; also bridges into `BeliefStore` and writes to the knowledge graph, neither disclosed in the moduledoc.
- **Store** — the primary memory facade. `persist/0` is a no-op stub (Executive Summary #1, zero callers). `clear/1` only actually clears everything when `world_id` is `nil`; passing a specific world silently no-ops despite the doc implying world-scoped clearing works.
- **Embedder, Types, VectorIndex** — verified accurate; `Types.Procedure` is honestly documented as unused future scaffolding.

### graph/

- **ContextCache, Reader, Training, Writer** — all verified accurate against their docs; this is one of the cleanest subsystems audited. `Writer`'s SRL-triple-scoring detail lives only in a function-level `@doc`, not the terse moduledoc, but is itself accurate.

### subprocesses/

- **ConversationSubprocess, HttpSubprocess, Supervisor** — verified accurate.
- **CliSubprocess** — the `conversations` CLI command is a hardcoded stub string ("This would list active conversations") despite `get_help/1` advertising it as functional; `interrupt`/`emergency` commands return canned success strings without actually invoking the real interrupt mechanism (a separate cast API a caller must invoke directly).

### fact_database/

- **Fact, Integration** — verified accurate, including the legacy `"entity_type"`-as-subject-name quirk handling.

### lattice/

- **Candidate, FragmentVectorizer** — verified accurate.

### Misc singletons

- **Brain.HTTP.Retry** (http/retry.ex) — verified accurate.
- **Brain.Metrics.Aggregator** (metrics/aggregator.ex) — verified accurate; a wide set of `handle_cast` clauses have no public wrapper function, presumably cast directly by telemetry handlers (consistent with its documented fire-and-forget design, not a mismatch).
- **Brain.Test.AtlasSandbox** (test/atlas_sandbox.ex) — lives under `lib/`, not `test/support/`, which looks like a stray file at first glance; a code comment explains this is deliberate (so other umbrella apps' test suites can depend on `:brain` normally and reach it) — correctly placed, but worth noting explicitly here since the location is non-idiomatic and the justification lives only in a code comment.

---

## Coverage note

All 209 `.ex` files under `apps/brain/lib/brain` were read and cross-checked.
No silent caps were applied — every module assigned to a review batch was
covered. Where a module isn't called out individually above with its own
mismatches, it means the review found its documentation accurate against its
implementation.
