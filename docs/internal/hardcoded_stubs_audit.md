# Hardcoded Stubs Masquerading as Real Signals

> **Status:** point-in-time audit, generated 2026-07-04, derived from the
> module-by-module review behind [`docs/BRAIN.md`](../BRAIN.md). Re-run after
> significant refactors rather than hand-editing entries piecemeal.

## Why this document exists

`.cursorrules` states project doctrine explicitly:

> **Rule 2 — No String Matching for Classification:** Intent classification,
> entity extraction, speech act classification, and sentiment analysis must be
> data-driven (TF-IDF, feature-vector classifiers, trained models, or
> MicroClassifiers). Never hardcode string-matching rules for NLP decisions.
>
> **Rule 4 — Hard Failure on Missing Models:** If a required model or corpus
> is missing, the system must fail immediately with a clear error message...
> Do not silently degrade.
>
> **Rule 5 — No Silent Fallbacks:** Never introduce a "graceful fallback" that
> masks a missing dependency, model, service, or configuration... If
> something is required, it must be present; if it's absent, fail loudly.
>
> **Rule 8 — No hardcoded intents, entities, or any other data:** If we need a
> "new" system, we need to train a new model for it... This enables us to
> easily add new systems and models to the system.

The `docs/BRAIN.md` audit — a deep read of all 209 modules under
`apps/brain/lib/brain`, cross-checking each module's own documentation
against its implementation — turned up **28 distinct instances** where a
function's real behavior is a fixed, hardcoded value or fabricated data,
while its name, its own doc, its callers, or the surrounding code all treat
its output as if it were a genuine, data-derived signal. None of these carry
a `TODO`/`FIXME` disclosing the gap. This document exists to make them
un-missable in one place, since they're exactly the failure mode Rules 2, 4,
5, and 8 exist to prevent — they don't crash and they don't obviously look
wrong in isolation, so they're the kind of defect that survives code review
and keeps producing quietly-wrong output indefinitely.

**Severity key:**
- 🔴 **High** — feeds a decision, a persisted artifact, or a user-facing
  claim; wrong in a way that's hard to detect after the fact.
- 🟡 **Medium** — inert logic/dead branch; degrades a feature to something
  narrower than documented, but doesn't fabricate false data.
- ⚪ **Low** — currently unreachable in production, so the defect is real but
  dormant.

---

## Summary table

| # | Module.function | File:line | Presents as | Actually does | Severity |
|---|---|---|---|---|---|
| 1 | `Brain.ML.Trainer.train_intent_classifier/2` | ml/trainer.ex:107-111 | Real per-word IDF weighting in the saved embedder model | Writes a constant `1.0` for every vocabulary word, discarding the real IDF `SimpleClassifier.train/1` computed | 🔴 High |
| 2 | `Brain.Response.OuroRealizer.validate_and_return/2` | response/ouro_realizer.ex:77-91 | Provenance metadata (`model`, `source`) reflecting which backend actually generated the response | Hardcodes `"ouro-1.4b"` / `:ouro` unconditionally, even when OpenAI-compatible or another backend produced the text | 🔴 High |
| 3 | `Brain.ML.EntityTrainer.finalize_entity/1` (→`extract_entities_from_bio/1`) | ml/entity_trainer.ex:554-561 | Model-derived confidence for each extracted entity | Hardcodes `confidence: 0.8` for every entity regardless of actual probabilities; `EntityExtractor` treats this as real model confidence downstream | 🔴 High |
| 4 | `Brain.Response.DecompressorCollector.feed_to_phrase_inventory/1` | response/decompressor_collector.ex:302-343 | Real analyzed tone/slot/enrichment metadata for runtime-collected phrase fragments | Fabricates `"tone" => "neutral"`, a zero tone-vector, and empty slots/conditions for every fragment, regardless of the fragment's actual content | 🔴 High |
| 5 | `Brain.Learner` (`learn_from_input/3`, `learn_from_classical_extraction/3`, `learn_from_conversation/4`) | learner.ex:65,96,138 | Relationship extraction (moduledoc explicitly claims this) | Hardcodes `"relationships" => []` in all three entry points; `KnowledgeStore.add_relationship/5`'s only caller is therefore always called with nothing | 🔴 High |
| 6 | `Brain.Epistemic.ContradictionHandler` (`assumption_metadata` state) | epistemic/contradiction_handler.ex:72,310-338,388-400 | Confidence/recency-based contradiction-resolution strategies (`:auto_least_confident`, `:auto_most_recent`) | `assumption_metadata` is never populated by any handler; both strategies always read empty per-id data and fall back to identical defaults — they can never actually differentiate assumptions | 🔴 High |
| 7 | `Brain.Response.ContentSpecifier.check_action_capability/2` + independent copy in `Brain.Response.DiscoursePlanner` | response/content_specifier.ex:615; response/discourse_planner.ex:246 | A real capability lookup deciding whether the agent can perform a requested action | Two separately-hardcoded `:unknown` stubs — capability-gated primitives always collapse to `:incapable`, never `:pending`, app-wide | 🔴 High |
| 8 | `Brain.Knowledge.Corroborator.blend_with_kg_similarity/3` | knowledge/corroborator.ex:290-294 | KG-aware similarity blending, gated by `kg_clustering_enabled?/0` | Both branches of the `if` return `tfidf_sim` unchanged — the blend is a complete no-op whether or not the gate is enabled | 🔴 High |
| 9 | `Brain.Knowledge.LearningCenter.apply_prediction_evaluation/1` (`test_predictions/1`) | knowledge/learning_center.ex:1037-1067 | Structured-prediction evaluation (checking a hypothesis's specific predicted outcome) | For both prediction types, both branches of the conditional call the identical generic `Hypothesis.evaluate(h)` — the prediction's expected outcome is computed and then never used | 🔴 High |
| 10 | 4× `persist/0` — `Brain.Epistemic.BeliefStore`, `Brain.Epistemic.UserModelStore`, `Brain.Knowledge.SourceReliability`, `Brain.Memory.Store` | epistemic/belief_store.ex:473-475; epistemic/user_model.ex:302-304; knowledge/source_reliability.ex:144-146; memory/store.ex:538-540 | "Persists the store to disk" (own `@doc`) | Bare `{:reply, :ok, state}` — reports success, writes nothing | 🔴 High |
| 11 | `Brain.Subprocesses.CliSubprocess` (`conversations`, `interrupt`, `emergency` commands) | subprocesses/cli_subprocess.ex:232,296,298 | Working CLI commands (advertised by the CLI's own `help` text) | Canned strings with no effect; `interrupt`/`emergency` don't call the real `handle_urgent_interrupt/3` at all | 🔴 High |
| 12 | `Brain.ML.Trainer.train_svm_classifier/2` | ml/trainer.ex:434-458 | "A simple nearest neighbor approach" (own `@doc`) | Vectorizes texts and packages raw tensors into a map — no classification logic of any kind runs | 🟡 Medium |
| 13 | `Brain.ML.Trainer.save_models/2` | ml/trainer.ex | "Save trained models to disk" (own `@doc`) | Logs a deprecation warning; saves nothing | 🟡 Medium |
| 14 | `Brain.Analysis.SemanticChunker` (learned `discourse_markers`) | analysis/semantic_chunker.ex:76-94 vs. hardcoded `@default_discourse_markers` | Learnable discourse-marker configuration (fetched from `LearningStore` alongside genuinely-learnable chunk-size thresholds) | The learned list is fetched and then never read — `find_leading_markers/1` always uses the hardcoded module attribute | 🟡 Medium |
| 15 | `Brain.Analysis.FramingDetector` (`causal_attribution`, `dominant_lexical_domains`, `secondary_frames`) | analysis/framing_detector.ex:203-212,248-275 | Real computed evidence fields, per the module's own typespec | Hardcoded to `%{agent_bias: 0.0, patient_bias: 0.0}` / `[]` / `[]` unconditionally — never derived from input, unlike sibling fields (`sentiment_skew`, `modality_skew`) that genuinely are | 🟡 Medium |
| 16 | `Brain.Analysis.FollowupDetector` (`looks_like_slot_filler?/2`) | analysis/followup_detector.ex:136-149 | POS-based grammatical detection ("uses POS tagging... rather than keyword lists," per moduledoc) | Any message ≤5 words is accepted as a slot filler when the missing slot isn't "location" — pure word-count heuristic, no grammatical check | 🟡 Medium |
| 17 | `Brain.Analysis.RacingAnalyzer.analyze_keywords/1` | analysis/racing_analyzer.ex:343-345 | A keyword-pattern voter ("Pattern **and keyword** triggers are loaded from `data/pattern_triggers.json`," per moduledoc) | Ignores its `text` argument entirely, unconditionally returns a 0.0-confidence stub. The `"keywords"` JSON section is parsed and cached but nothing ever reads it | 🟡 Medium |
| 18 | `Brain.Analysis.RacingAnalyzer.apply_intent_safeguards/2` | analysis/racing_analyzer.ex:523-525 | A safeguard/correction pass ("corrected results," per the variable name and what's reported to the UI) | Pure identity function — corrects nothing, but is broadcast to the UI via `Progress.report/3` labeled as corrected output | 🟡 Medium |
| 19 | `Brain.Analysis.BacktrackController.check_entity_mismatch/1` | analysis/backtrack_controller.ex:119-121 | One of three live contradiction signals gating backtracking | `defp check_entity_mismatch(_interp), do: :ok` — ignores its argument, always `:ok`; can never trigger a backtrack | 🟡 Medium |
| 20 | `Brain.Analysis.EventLinker.temporal_contains?/2` | analysis/event_linker.ex:210-212 | Sub-event/temporal-containment detection ("Creates sub-event links when one event's temporal span contains another's," per moduledoc) | `defp temporal_contains?(_outer, _inner), do: false` — `sub_events` is always empty for every event frame | 🟡 Medium |
| 21 | `Brain.Analysis.EventLinker.infer_temporal_order/3` | analysis/event_linker.ex:179-191 | Temporal-value-based ordering between events ("Detects temporal relations... using EntityExtractor's temporal entity detection") | Only compares token position (`trigger_index`); temporal entities merely gate *whether* a relation is emitted, never determine its direction | 🟡 Medium |
| 22 | `Brain.Analysis.AnalyzerCalibration.is_dominating?/2` | analysis/analyzer_calibration.ex:122-127,252 | Win-rate-based analyzer dominance tracking ("Useful for stability self-reflection," per moduledoc) | Reads a `{:win_rate, analyzer}` ETS entry that's only ever zero-initialized at startup — no handler updates it, so this is permanently `false` | 🟡 Medium |
| 23 | `Brain.ML.Gazetteer` (`:table_prefix` isolation) | ml/gazetteer.ex — ~25 call sites vs. 4 | Test-isolated per-instance ETS tables ("useful for test isolation," per `start_link/1` doc) | Nearly every read/write function hardcodes the global `@table_name`-family attributes instead of `state.tables`; a "prefixed" instance still mutates shared global state. Live path — used by `test/support/genserver_sandbox.ex` | 🟡 Medium |
| 24 | `Brain.ML.KnowledgeGraph.Embedder.build_embedding_model/1` | ml/knowledge_graph/embedder.ex:52-58 | A truncated model outputting 128-dim `dense1` activations, per its own `@doc` | `Axon.nx(scorer_model, fn output -> output end)` — an identity wrapper around the *full* scorer including the sigmoid head; would produce a scalar, not a 128-dim vector, if ever called | ⚪ Low (no callers — real consumer bypasses it for the correctly-truncated `build_extraction_model/2`) |
| 25 | `Brain.Analysis.EventExtractor.compute_event_confidence/3` | analysis/event_extractor.ex:315-324 | The confidence-scoring logic actually used during extraction (it's the only `defn` for this purpose) | Dead — the real path (`extract_event_for_verb/5` → `calculate_confidence/3`) uses an independently-implemented, non-tensor equivalent instead | ⚪ Low |
| 26 | `Brain.Analysis.EventExtractor.match_pattern/2` + `Brain.Analysis.EventPatterns`' pattern/tensor API | analysis/event_extractor.ex:281-291; analysis/event_patterns.ex (most of its ~20 functions) | Pattern-based event extraction ("extracting actor-verb-object relationships" via loaded JSON patterns, per both moduledocs) | Real extraction is pure nearest-verb-neighbor search; the JSON-defined pattern/weight/tense-hint system is loaded, cached, and never consulted by the live path | ⚪ Low |
| 27 | `Brain.Analysis.SelfKnowledgeAnalyzer.meta_intents/0` | analysis/self_knowledge_analyzer.ex:151 | — | Always returns `[]` — **but this one discloses it's a stub in its own doc** ("graceful degradation"). Listed here only for contrast; not a violation. | — (counter-example) |
| 28 | `Brain.Services.HomeAssistant.call_service/5` | services/home_assistant.ex:263-280 | — | Deliberately, loudly refuses every actuation call (`{:error, :actuation_disabled}`) post-remediation. **Also listed only for contrast** — this is Rule 4/5 done *right* (fail loud instead of faking success), just under-disclosed in its own moduledoc (see `docs/BRAIN.md` §Services for that specific gap). | — (counter-example) |

---

## Why these two "counter-examples" (27, 28) belong in this document

Rows 27 and 28 are not violations — they're the two clearest existing examples
of the correct pattern (disclosed stub; loud refusal instead of a fake
success), and they're worth citing precisely *because* the surrounding
codebase already knows how to do this. Every 🔴/🟡 row above could be brought
into compliance by following one of these two templates:

- If the feature is genuinely not implemented yet: return an explicit error
  (`{:error, :not_implemented}`) or mark the function `@doc false` with a one-line
  comment stating it's a stub — the way `meta_intents/0` and
  `Brain.ML.Generation.Null` do.
- If the feature was deliberately removed/disabled: say so in the
  `@moduledoc`/`@doc`, not just an inline comment three call-frames away from
  where the doc reader would look — the way `Brain.Code.LanguageGrammar`
  discloses its degraded mode (contrast with its sibling `Brain.Code.Parser`,
  which has the identical fallback but doesn't disclose it — see
  `docs/BRAIN.md`).

Neither template requires deleting the surrounding architecture (the
capability-gating cascade, the win-rate tracking scaffold, the sub-event
linking hooks can all stay) — it only requires the constant-return function
to say what it is, instead of pretending to be a live signal.

---

## Recommended triage order

1. **Rows 1–11 (🔴 High)** first — these either corrupt a persisted artifact
   (row 1: fake IDF baked into a saved model; row 3: fake confidence baked
   into training data), falsify provenance/audit metadata (row 2), silently
   drop a whole class of extracted data (row 5), collapse a resolution
   strategy to a no-op (row 6, 9), fabricate fake analysis output (row 4),
   report false success (row 10), or present broken commands as working
   (row 11). Each of these can propagate wrong data forward into training
   sets, review queues, or user-facing claims before anyone notices.
2. **Rows 12–23 (🟡 Medium)** next — inert branches and narrower-than-documented
   heuristics. Lower urgency because they degrade a feature rather than
   fabricate wrong data, but each is a direct Rule 2/8 violation once traced
   to its root (a hardcoded constant standing in for a signal that's supposed
   to be trained/data-derived).
3. **Rows 24–26 (⚪ Low)** — real bugs, currently dormant. Worth fixing before
   anything wires them up, not urgent today.
4. **Rows 27–28** — no action needed; cited as the fix template.

## Coverage note

This list is drawn from the same ten-batch, all-209-module review behind
`docs/BRAIN.md` — nothing new was searched for beyond what that audit already
surfaced. If `docs/BRAIN.md` is regenerated after a refactor, re-derive this
list from its "Mismatches" entries rather than hand-appending to either
document independently, or the two will drift against each other.

**See also:** [`hardcoded_stubs_remediation.md`](hardcoded_stubs_remediation.md)
— for 8 of the highest-priority rows above, a verified, concrete fix using an
existing data-driven mechanism already in the codebase (exact function
signatures, no new model/system required for most of them).
