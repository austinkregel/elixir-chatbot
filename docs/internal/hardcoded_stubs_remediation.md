# Making the Hardcoded Stubs Data-Driven: Remediation Plan

> **Status:** research/design document, generated 2026-07-04. Companion to
> [`hardcoded_stubs_audit.md`](hardcoded_stubs_audit.md) (what's wrong) — this
> document answers *how to fix it using systems that already exist in this
> codebase*, per `.cursorrules` Rule 8 ("a new system needs a trained model,
> not hardcoded rules") and the general instinct to reuse before inventing.

**Method:** for each stub, I checked whether a real, already-built mechanism
elsewhere in `apps/brain` already computes (or could trivially expose) the
signal the stub fakes. Eight of these were independently re-verified against
the current code this round (exact function signatures, line numbers, and
call-site data availability confirmed) — those are marked **Verified** and
are ready to implement from this document alone. The rest carry the
characterization from the original audit but weren't re-read line-by-line in
this pass — marked **Candidate**, and should get the same verification before
anyone starts implementing.

---

> **Implemented 2026-07-19.** All eight Tier 1 items are done (commits
> `f64afe8` items 1/2/5, `60fc430` items 4/6/7, `a1bed48` item 3, `99d1b28`
> item 8). Each was re-verified against the current code before the change;
> item 8 got the extra line-level pass the doc asked for. Tier 2, the Not-a-
> data-driven-problem fixes, and the deletions remain open.

## Tier 1 — Verified: real signal already exists, this is plumbing, not new ML

These four all have the same shape: the codebase already computes the real
value somewhere nearby, and the stub exists only because nobody threaded it
through. No new training data, no new model, no new system — just wiring.

### 1. `OuroRealizer` response metadata (was: hardcoded `"ouro-1.4b"`/`:ouro`)

**Real signal:** every backend module already implements `name/0`
(`Brain.ML.Generation.Backend` callback, `backend.ex:27`) — `Null.name/0` →
`:null`, `OpenAICompatible.name/0` → `:openai_compatible`,
`OuroSidecar.name/0` → `:ouro_sidecar`. `Generation.name/0`
(`generation.ex:58`) already exposes the *currently configured* backend's
name, and `OuroRealizer.do_generate/3` **already calls `Generation.name()`**
in its own error paths (`ouro_realizer.ex:69,72`) — it's simply not captured
on the success path.

**Fix:** in `do_generate/3` (`ouro_realizer.ex:63-64`), capture
`backend = Generation.name()` alongside the `generate/2` call, pass it into
`validate_and_return(text, primitives, backend)`, and in
`validate_and_return/2` (`:77-92`) set `source: backend, model: to_string(backend)`
instead of the hardcoded literals. Zero new modules.

### 2. `DecompressorCollector` fabricated tone data (was: `"neutral"` + zero vector)

**Real signal:** `Brain.Response.ChunkSegmenter.tag_tone/1`
(`chunk_segmenter.ex:127-167`) already computes a real 10-element tone vector
(`[pos, neg, neu, conf, polarity, arousal, warmth, formality, playfulness,
directness]`) plus a tone label, from the chunk's actual text — and its
output dimensionality matches `PhraseInventory`'s expected/default tone-vector
length (10) exactly.

**Fix:** `feed_to_phrase_inventory/1` (`decompressor_collector.ex:302-343`)
already builds `chunks = ChunkSegmenter.segment(text)` from the same rendered
text a few lines before fabricating tone (line 310 vs. 324-325) — replace
`"tone" => "neutral", "tone_vector" => List.duplicate(0.0, 10)` with
`"tone" => to_string(chunk.tone), "tone_vector" => chunk.tone_vector` by
calling `ChunkSegmenter.tag_tone(chunk)` right there. This is a drop-in
replacement of two lines; `ChunkSegmenter` is already a dependency of this
part of the response pipeline.

### 3. `Learner`'s hardcoded `"relationships" => []`

**Real signal:** `Brain.Analysis.SemanticRoleLabeler.to_triples/1` already
converts SRL frames into real `{subject, predicate, object}` triples, and
this exact conversion is **already used in production** elsewhere —
`Brain.Graph.Writer.write_srl_triples/1` (`graph/writer.ex:375-377`) calls
`SemanticRoleLabeler.to_triples(srl_frames)` on frames pulled from
`ChunkAnalysis.srl_frames`, which the pipeline already populates
(`pipeline.ex:341,358`). `KnowledgeStore.add_relationship/5`
(`knowledge_store.ex:100`) takes `(persona_name, subject, relation, object,
confidence \\ 1.0)` — a direct match for a triple plus a default confidence.

**The gap:** `Learner.learn_from_conversation/4` is called from
`brain.ex:1468-1474` with a hand-trimmed `analysis_for_learning` map
(`%{entities:, speech_act:, intent:}`) that drops `srl_frames` even though
the full `ChunkAnalysis` structs (which have it) are in scope a few lines
earlier (`brain.ex:1463-1465`).

**Fix:** two small changes, no new system —
1. In `brain.ex`, add `srl_frames: Enum.flat_map(analysis_model.analyses, &Map.get(&1, :srl_frames, []))` to `analysis_for_learning`.
2. In `Learner.learn_from_conversation/4` (and the other two entry points, if SRL frames are threaded to them too), replace the hardcoded `[]` with `SemanticRoleLabeler.to_triples(srl_frames) |> Enum.map(fn {s, p, o} -> {s, p, o, 1.0} end)` fed into the existing (already-implemented, currently-starved) `process_relationships/2`.

### 4. `ContentSpecifier.check_action_capability/2` (was: hardcoded `:unknown`)

**Real signal:** `Brain.Services.Dispatcher.find_service/1`
(`dispatcher.ex:199`) already resolves an intent string to a service module,
and `Dispatcher.service_available?/2` (`dispatcher.ex:99`) already checks
that service's credentials/registration — together they already answer
exactly the question this stub is faking. (`CapabilityRegistry.can_handle?/2`,
`home_assistant/capability_registry.ex:59`, is a second, narrower real check
specific to Home Assistant domains, usable as a refinement inside the
Home-Assistant branch.)

**Fix — concrete, from the verification pass:**
```elixir
defp check_action_capability(intent, opts) when is_binary(intent) do
  case Brain.Services.Dispatcher.find_service(intent) do
    nil -> :incapable
    service ->
      if Brain.Services.Dispatcher.service_available?(service.name(), opts),
        do: :capable, else: :incapable
  end
end
defp check_action_capability(_intent, _opts), do: :incapable
```
Same fix applies to `DiscoursePlanner`'s independent hardcoded `:unknown` copy
(`discourse_planner.ex:246`) — both should call the same helper rather than
each keeping its own hardcode.

### 5. `CliSubprocess` canned command strings

**Real signal:** `Brain.Subprocesses.Supervisor.list_subprocesses/0`
(`supervisor.ex:125-127`, `DynamicSupervisor.which_children/1`) already
returns real live subprocess data. `handle_urgent_interrupt/3`
(`cli_subprocess.ex:42`) already exists as a real cast that sets
`is_interrupted: true` and logs to `cli_memory`.

**Fix:** replace the `conversations` canned string (`:232`) with a call to
`Supervisor.list_subprocesses/0` formatted for display; replace the
`interrupt`/`emergency` canned strings (`:296,298`) with an actual call to
`handle_urgent_interrupt(state.subprocess_id, :interrupt, %{})` /
`(..., :emergency, %{})` instead of returning a success string that does
nothing.

### 6. `Corroborator.blend_with_kg_similarity/3` (was: no-op both branches)

**Real signal:** `EntityVectorCache.get_or_compute/3`
(`entity_vector_cache.ex:32`) already returns a real 128-dim KG-aware entity
embedding (via the correctly-truncated `Embedder.build_extraction_model/2`),
and `Corroborator` already has its own `cosine_similarity/2`
(`corroborator.ex:345-359`) sitting unused for this purpose.

**Two bugs, not one:** the plumbing is broken upstream too —
`find_matching_cluster/3` is called with a hardcoded `nil` for the
current-finding argument (`corroborator.ex:271`), and
`cluster_by_similarity/2`'s reduce never forwards the actual `finding` struct
that's available at that point (`:243-246`).

**Fix:** (a) thread the real `finding` through `cluster_by_similarity/2` →
`find_matching_cluster/3` instead of `nil`; (b) in
`blend_with_kg_similarity/3`, call `EntityVectorCache.get_or_compute("default", entity)`
for both entities, convert the returned tensors with `Nx.to_flat_list/1`, and
blend with the existing `cosine_similarity/2` — e.g.
`tfidf_sim * 0.6 + kg_sim * 0.4` — instead of returning `tfidf_sim` unchanged.

### 7. `AnalyzerCalibration.is_dominating?/2` (was: permanently `false`)

**Real signal:** the "did this analyzer win" fact already exists at the
caller — `Interpretation.from_analyzer_results/2` already sorts and picks
`interp.source` as the winner (`interpretation.ex:76,95`), and
`OutcomeLearner` (`outcome_learner.ex:175`) already calls
`track_outcome(interp.source, ...)` for the winner specifically, separately
from the per-participant loop (`:181-187`) — the information just isn't
passed into `track_outcome/3`'s 3-arg signature, so the cast handler can't
tell winner from participant.

**Fix:** add a `won?` boolean to the call (either a `track_outcome/4` arity or
a separate `track_win(analyzer)` cast), have that handler update the same
`{:win_rate, analyzer}` ETS key `is_dominating?/2` already reads (currently
only zero-initialized once in `initialize_calibration_data/0`, never
touched again). No new table, no new system — one new cast arm updating an
existing key.

### 8. `Brain.ML.Trainer.train_intent_classifier/2` (was: hardcoded IDF `1.0`)

**Real signal (from the original audit, not re-verified this round but
straightforward):** `SimpleClassifier.train/1` already computes real IDF
weights as part of training the intent classifier in the same function;
`build_idf_weights_from_model/1` throws that away and substitutes a constant
`1.0` for every vocabulary word when building the *embedder's* saved model.

**Fix:** thread the real IDF map already computed by `SimpleClassifier.train/1`
into the embedder's persisted model instead of a constant map. Recommend a
quick direct-read verification pass (same rigor as items 1-7 above) before
implementing, to confirm the exact variable that's being discarded.

---

## Tier 2 — Candidate: plausible existing-system reuse, not re-verified this round

> **Verified + partially implemented 2026-07-19** (commit `076fe94`). All eight
> candidates below were re-read against current code. Three were genuine
> reuse-existing-signal fixes and are DONE; five turned out to need a new
> dataset/model or cross-module plumbing (not the drop-in the table implied) and
> are DEFERRED. Status per item:
>
> - **DONE — `SemanticChunker` discourse markers** (MODERATE, not "one-line"):
>   the learned value is threaded through `detect_discourse_markers`/`find_leading_markers`.
> - **DONE — `ContradictionHandler` assumption_metadata** (MODERATE): built live
>   from the beliefs backing the JTMS nodes. Correction: confidence/created_at
>   come from the `Belief`, **not** `JTMS.why_node` (the `Node` struct has neither).
> - **DONE — `LearningCenter.apply_prediction_evaluation/1`** (DROP_IN): the
>   `:belief_query` branch already *called* `query_beliefs/1` but discarded the
>   result (both `{:ok, …}` clauses were identical); the result now drives status.
>   The sibling `:corroboration_count` branch has the same dead-`if` and is a
>   not-yet-listed follow-up.
> The five deferred items were then **intentionally designed with exhaustive
> reuse** (2026-07-19) after three wide reuse-discovery passes — all now built:
>
> - **DONE — `BacktrackController.check_entity_mismatch/1`** (`57e3afc`):
>   self-consistency (entities vs intent-domain expected types, reusing
>   `TypeHierarchy.compatible?/2` + `expected_entity_types_from_domain/1`) **plus**
>   cross-turn (thread `conversation_history` in), guarded against OOV/PROPN/
>   low-confidence false positives. Also fixed a latent `humanize_intent(nil)`
>   crash the newly-active path exposed.
> - **DONE — `RacingAnalyzer.analyze_keywords/1`** (`6fe85f4`): retired the dead
>   voter (chosen over reviving the hand-authored keyword table, which would
>   violate Rule 8). Intent stays covered by `:model`/`:intent_full` +
>   `:pattern_recognition`.
> - **DONE — `FollowupDetector`** (`6073c55`): word count is now only a gate;
>   the decision reuses the trained SpeechActClassifier (reject new
>   questions/commands) + `:clarification_response` on the prior bot prompt. No
>   new model.
> - **DONE — `FramingDetector`** (`71ed7e9`): reads real evidence from signals
>   already in the mean vector — lexical domains (group 10) + agent/patient bias
>   (group-12 SRL flags) — via the new `ChunkFeatures.group_offsets/0` source of
>   truth (`247e667`), which also fixed a stale sentiment offset.
> - **DONE — `EventLinker` temporal ordering** (`4ca1143`): new stdlib
>   `TemporalResolver` (relative + absolute dates, granularity-aware containment)
>   drives ordering (resolved dates → grammatical tense → position) and real
>   `temporal_contains?/2`; fixed the atom/string bug that meant temporal args
>   were never extracted from live data.

Same "reuse before inventing" instinct, but these need a verification pass
like Tier 1 got before anyone writes code against them.

| Stub | Candidate existing system to reuse | Why it's plausible |
|---|---|---|
| `SemanticChunker` discarding learned `discourse_markers` | *(no reuse needed)* | The value is already fetched from `LearningStore` right next to the chunk-size thresholds that **do** get used — this is a one-line fix (use the fetched value instead of the hardcoded `@default_discourse_markers`), not a new-system question. |
| `ContradictionHandler`'s always-empty `assumption_metadata` | `BeliefStore`'s existing per-belief confidence + timestamp tracking; `JTMS.why_node/1-2` | `BeliefStore` already tracks confidence and provenance per belief with decay; the resolution handler could look up the real belief/assumption's stored confidence and creation time instead of a separate, never-populated metadata map. |
| `LearningCenter.apply_prediction_evaluation/1` | `BeliefStore.query_beliefs/1` | For `:belief_query`-type predictions specifically, the store already has a real query function that could check whether the predicted belief actually exists, instead of ignoring the prediction and calling the same generic `Hypothesis.evaluate/1` regardless. |
| `BacktrackController.check_entity_mismatch/1` | `TypeHierarchy.compatible?/2` (already used by `EntityDisambiguator`, `ContextualEntityInferrer`) | A real mismatch check could compare the interpretation's entity types against recent context/history entity types via the hierarchy-compatibility function that already exists and is already used for exactly this kind of comparison elsewhere. |
| `RacingAnalyzer.analyze_keywords/1` (ignores its `text` arg) | `MicroClassifiers` (the same mechanism behind `:directed_at_bot`, `:modal_directive`, `:event_argument_role`) | Rather than reviving the currently-parsed-but-unread JSON keyword-pattern cache (which is itself a step away from data-driven), training a proper micro-classifier the way sibling voters already do would be more consistent with Rule 2. |
| `FollowupDetector`'s ≤5-word heuristic | `DiscourseAnalyzer`'s existing hybrid pattern (structural POS check + MicroClassifier) | `DiscourseAnalyzer` already solves a structurally similar problem (is this directed at the bot?) by combining a real POS/structural check with a trained classifier — the same shape could replace the word-count guess here. |
| `FramingDetector`'s hardcoded `causal_attribution`/`dominant_lexical_domains` | `SemanticRoleLabeler` argument roles (agent/patient already extracted); `Brain.Lexicon.domain_histogram/1` | Agent/patient bias could come from real ARG0/ARG1 role counts already produced by SRL; lexical-domain distribution has a real, already-implemented (if currently uncalled) function sitting right there — `Lexicon.domain_histogram/1`. |
| `EventLinker.temporal_contains?/2` and `infer_temporal_order/3`'s position-only ordering | `Brain.ML.Tokenizer.extract_dates/1` (already extracts real date/duration values) | Currently ordering uses token position as a proxy for time; real extracted date/duration values exist elsewhere in the pipeline and could give an actual temporal comparison instead of a position proxy — likely needs new glue code connecting two existing pieces rather than a drop-in fix. |

---

## Not a "make it data-driven" problem — different fix category

A few stubs from the audit aren't really Rule 2/8 violations to solve with a
trained model or an existing signal — they're either straightforward bugs or
dead legacy code better deleted than fixed.

> **Deletions done 2026-07-20** (commits `69a9116`, `d505b55`, `df44331`).
> Each was re-verified as caller-free before removal; only the `Gazetteer`
> `:table_prefix` *fix* remains open (it's a repair, not a deletion).

- **`Brain.ML.Gazetteer`'s broken `:table_prefix` isolation** — *(still open —
  a fix, not a deletion.)* A direct, already-correct template exists in the same
  codebase: `Brain.ML.Lexicon` implements per-instance table isolation
  correctly. Copy that pattern (use `state.tables` consistently) rather than
  inventing anything.
- **DONE — the `persist/0` no-op cluster** (`df44331`): removed the no-op
  `persist/0` + its `handle_call(:persist, …)` handler from `BeliefStore`,
  `SourceReliability`, and `Memory.Store` (no callers; write-through already
  happens on mutation). `SourceAuthority.persist/0` remains as the genuine
  flush template. (The audit's `UserModelStore` entry was stale — no such
  module exists.)
- **DONE — `KnowledgeGraph.Embedder.build_embedding_model/1`** (`69a9116`):
  deleted the broken function and its sole (also caller-free) consumer
  `extract_embeddings/4` + the orphaned `ensure_model_state/1`. The real path
  (`build_extraction_model/2` + `encode_entity/4`) is untouched.
- **DONE — `Trainer.train_svm_classifier/2` and `Trainer.save_models/2`**
  (`d505b55`): deleted both (no production caller), their orphaned private
  helpers, and their tests. The real training path stays
  `SimpleClassifier.train/1` + `Trainer.train_and_save/1`.

---

## Suggested sequencing

1. **Tier 1, items 1, 2, 5** first — smallest, most mechanical, zero risk of
   behavior change beyond "now reports the truth" (metadata, tone, CLI
   commands).
2. **Tier 1, items 4, 6, 7** next — slightly larger diffs (a helper function
   rewrite, threading one extra argument), still no new systems.
3. **Tier 1, item 3** (`Learner` relationships) — touches a call site in
   `brain.ex` plus `Learner` itself; worth its own small PR given it crosses
   a module boundary.
4. **Tier 1, item 8** (`Trainer` IDF) — do the same line-level verification
   the other seven got before touching it; it may be as simple as the others
   once confirmed.
5. **Tier 2 items** — each needs its own verification pass (mirroring how
   items 1-7 got verified this round) before implementation; don't implement
   from the "candidate" table without re-reading the exact code first.
6. **Deletions** — low-risk, can happen anytime, ideally bundled with
   whichever Tier 1 fix touches the same file.
