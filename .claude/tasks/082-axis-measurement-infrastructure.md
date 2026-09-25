# Axis measurement infrastructure: populate, snapshot, tag, bisect

- **ID:** 082
- **Status:** Phase 0 complete (2026-09-12); **Phase 1 complete (2026-09-25)** — Phase 2 (coverage,
  corpora, synthesis) next
- **Area:** measurement
- **Branch:** `feat/lattice-home-assistant-and-memory`
- **Recorded:** 2026-09-12 · revised same day with Austin's storage and coverage decisions

## The requirement

Austin, 2026-09-12:

> "I expressly must require that we absolutely actually populate these regardless of where else it
> is read from. Just because it doesn't change behavior now doesn't mean it won't represent
> important signals in the future... part of this is adding at least enough testing data and
> actually plugged in systems to be able to validate all 18 axes deterministically and being able
> to *prove* that it is 'correct' to the best of the system's present understanding."
>
> "Which means we could snapshot the entire vector and use it to help measure progress towards
> somewhere more accurate... depending on how it's built, we'd be able to trace back our steps to
> know when a given calculation might have started to go wrong."

**Withdrawn:** my earlier criterion that "an axis is only done when something reads it". An axis that
is correct but unconsumed is infrastructure; an axis that is consumed but wrong is a liability. The
proof is this session — four axes have been dead for months and it took hand-written throwaway
probes to notice. A snapshot store would have surfaced it immediately **and dated it**.

## Decisions (Austin, 2026-09-12)

### Snapshots are in flux until tagged

> "I was honestly thinking something like 'snapshots' that are treated as 'in flux' for the most
> part until a given version is tagged. That means we can put off having a 'final' golden set until
> at least 81 and 82 are done and shipped."

So the model is **git-tag-like**, not golden-set-first:

- Runs accumulate freely and cheaply. An untagged run is disposable and may be pruned.
- **Tagging** promotes a run to a durable reference point, with a note saying what it represents.
- Comparison is always *run vs run* or *run vs tag*. No run is privileged by default.
- A "golden set" is not authored up front — it **emerges** from tagged snapshots once an axis looks
  right. Correctness claims are deferred until there is something stable to claim them against.

This resolves the storage question cleanly: **commit a summary to git only for tagged runs.**
Untagged runs live in Postgres and are prunable; tagged runs get a few-KB summary committed
(provenance block, per-axis distribution, bits-per-axis, coverage) so the long-term trend survives
a database reset and ports to the other fork. No per-run ceremony.

| data | home | why |
|---|---|---|
| untagged run observations | Postgres (`atlas_axis_*`) | high volume, disposable, queryable |
| tagged run observations | Postgres, retained | the drill-down for a reference point |
| tagged run summary | git | durable trend line, reviewable, portable |
| per-axis expectations | git, accumulated per PR | hand-ruled, small, must survive a reset |

### The current 238 utterances are controlled test input, not a standard

> "For the moment, we should treat our data as controlled input and output for use when testing the
> systems we're designing now. Once we have a stable north star to build against, we can worry
> about generating the golden standard."

So the existing sample is a **fixture** for building and testing the measurement machinery. It is
not a claim about correctness and must not be described as one.

### Uncovered axis values get a synthesis step, one PR per axis

> "Anytime we don't have something to cover for a given axis' needs, it needs to be added in a
> synthesis step. Where we can put up a dedicated PR for improving the given axis in snapshots."

Per axis: compute `declared value domain − values observed in the controlled set`, then author
utterances that exercise the uncovered values. One PR per axis.

**Why this structure avoids the task 079 failure mode.** A synthesised utterance asserts only the
thing it was constructed to demonstrate. "Do not turn off the lights" asserts `polarity: :negative`
— a ruling a human makes with near-certainty — and asserts *nothing* about the other 17 axes. So
each PR contributes high-confidence expectations for exactly one axis and stays silent elsewhere.
Expectations accumulate as a by-product of coverage work rather than being bulk-authored, which is
precisely what went wrong when `rebuild_gold_standard` wrote model output as truth.

**Caveat to carry:** synthesised utterances are not natural distribution. Tag every utterance with
its origin (`controlled` | `synthesised:<axis>`) so bits-per-axis can be computed over either set
separately. Otherwise the north-star numbers drift for reasons unrelated to correctness — a
synthesis-heavy set would inflate the axes it was built to exercise.

## Two blockers found while scoping — Phase 0, not optional

### The feature vector has no schema

`ChunkFeatures.vector_dimension/0` returns only a **count**, as a hand-maintained sum of 23 group
sizes. The dimension *names* in `.cursor/notes/feature_dimension_audit.md` (`mem_novelty`,
`pos_PROPN`, `char_count`) **exist nowhere in the codebase** — `grep` finds no source, so they were
hand-written or produced by a script that is gone.

This is exactly why task 072 was invisible: dimension 151 has no name, so a change in *what
dimension 151 means* is undetectable by construction. The only way to notice is to recompute a
stored vector and diff it, which is what had to happen.

And the width is not statically determined:

```elixir
# group 10 — lexical-semantic fingerprint over domains
length(Lexicon.domain_atoms()) +
```

**The vector's width depends on runtime WordNet state** — a second silent-drift mechanism,
independent of the AGE-graph one behind 072. `vector_dimension/0` is also a separate source of truth
from the vector actually emitted, and nothing checks that they agree.

Snapshotting numbers whose meaning can shift underneath the record reproduces 072 at a larger scale.
**Name things, then measure them.**

### The 18 axes have no declared value domains

Legal values are implicit in each `derive_*` function or its micro-classifier's label set. Nothing
says "`polarity` is one of `[:affirmative, :negative]`". Without that, "populate all 18 axes" has no
completion criterion, and coverage cannot be computed at all.

## Plan

### Phase 0 — make the system self-describing — **DONE 2026-09-12**

1. ~~**`ChunkFeatures.dimension_manifest/0`**~~ **DONE.** Ordered `[{group, name}]`, 343 entries over
   25 groups, every group's names derived from the same module attribute or `EnrichmentFeatures`
   function the emission walks. `vector_dimension/0` is now `length(dimension_manifest())`.
   `group_widths/0` localises a width change to one group; `schema_fingerprint/0` is the
   **extractor schema fingerprint** (`fb3ce7f1e4739cb3` at this commit), a digest over names *and*
   order.
   - Correction to this task's original framing: the width was *already* guarded —
     `chunk_features_test.exs` asserted emitted length against the hand-summed constant. The manifest
     does not fix drift; it supplies **identity**, without which a snapshot records numbers that
     cannot be attributed to a feature.
2. ~~**`ChunkProfile.axis_manifest/0`**~~ **DONE.** 18 axes as `{axis, kind, domain, source}`, plus
   `axes/0`, `axis_domain/1` and `axis_default/1`.
   - Classifier-backed axes declare `{:model, classifier}` and resolve through the new
     `MicroClassifiers.labels/1`, so the value domain is read off the trained model rather than
     restated. `{:open, :speech_act_subtype}` is declared honestly — no canonical list exists yet
     (task 077).
   - A `{:model, _}` axis whose classifier is unloaded returns `{:error, reason}`, never a guessed
     set: validating against an invented domain would make every value appear to pass.
3. ~~Tests~~ **DONE.** 20 tests on `ChunkProfile`, 8 more on the dimension manifest (`mix test
   apps/brain/test/brain/analysis/chunk_profile_test.exs
   apps/brain/test/brain/analysis/feature_extractor/`). Mutation-tested, not assumed: one bogus
   dimension name fails two tests with the group and offset named.
4. **Provenance now distinguishes `:computed` from `:defaulted`** (the addition this task committed to
   for task 083's selection rule). Every axis records
   `%{source:, status:, reason:, evidence:, confidence:, depends_on:, parents_defaulted:}` as
   applicable. The asserted invariant: **a `:defaulted` axis holds exactly its declared default** —
   without which "off its default AND computed" is not a sound predicate.
   - `:wrong_kind` from a classifier now **raises**. It means a text classifier is wired into a
     vector axis, which no input can fix and which would otherwise present as a permanently
     defaulted axis.
   - Three fallbacks deleted as dead: `safe_classify_vector/3`, `safe_float/3`,
     `compute_slot_completeness/1`.

**What Phase 0 immediately measured** — see the 081 addendum for the full census. Three axes are
*never* computed (`novelty_score`, `polarity`, `self_disclosure_level`); three `cond` branches test
for labels their model cannot emit; and two derived axes have values that are unreachable because a
parent is dead (`temporal_framing` can never return `:negated_past`; `response_posture` can never
return `:tentative_confirm`).

On resetting axes to zero: unnecessary. Mark an axis unvalidated in the manifest until a tagged
snapshot covers it. Same discipline, no code change, no signal discarded, reversible.

### Phase 1 — snapshot + tag — **DONE 2026-09-25**

4. ~~**Run provenance**~~ **DONE.** `Brain.Analysis.RunProvenance.capture!/0`
   (`apps/brain/lib/brain/analysis/run_provenance.ex`): git sha + dirty flag, Elixir/OTP, extractor
   `schema_fingerprint` and `group_widths`, **AGE-graph digest**, lexicon digest, SHA-256 per
   `.term` and per dataset, timestamp. Every accessor raises rather than recording `nil` — a hole
   in a provenance record is invisible at comparison time, so the run looks comparable when it is
   not.
   - **`ml_models/manifest.json` is gone rather than hashed.** One writer, zero readers, and only
     `mix train` wrote it, so `mix train_micro` drifted it further every run. By 2026-09-25 it was
     stale on 19 of 26 models and 7 of 17 datasets while reporting a 2026-04-30 timestamp.
     `Brain.ML.MicroProvenance` replaces it, stamping the record into each model — task 072.
   - **Correction to this task's own Phase 0 framing.** The claim above that the vector's width
     depends on *runtime WordNet state* via `Lexicon.domain_atoms/0` is **false**: that is a
     compile-time `@domain_atoms` of 45 hardcoded lexicographer files, and the type checker rejects
     an emptiness guard against it as unreachable. The genuinely runtime-variable group is **23**,
     `length(TypeHierarchy.parent_types()) + 2`, read from ETS populated **out of the AGE graph** —
     the same mutable store whose emptying inverted the vector in 072. So the graph controls both
     the vector's values and its dimension names. Recorded as `age_graph`.
5. ~~`atlas_axis_runs` / `atlas_axis_observations`~~ **DONE.** `Atlas.Schemas.AxisRun`,
   `Atlas.Schemas.AxisObservation`, `Atlas.Axes`, migration `20260925000001_create_axis_runs.exs`.
   A tag requires a note; a defaulted observation requires a reason; a run and its observations are
   written in one transaction or not at all. 22 tests.
6. ~~Store the full 343-vector for tagged runs~~ **DONE.** `{:array, :float}`, written once per
   utterance rather than once per axis — repeating 343 floats eighteen times multiplies storage by
   the axis count for no extra information.
7. ~~`mix axes.snapshot [--tag NAME --note "..."]`~~ **DONE.** `--corpus` is required and has no
   default: whether `.claude/corpus/` is the right home is task 067, still open, and defaulting
   there would bake an undecided location into a production code path.

**First tagged run — `baseline-2026-09-25`**, fingerprint `bc289842ba5ccb3d`, 5,274 utterances,
**94,932 observations**:

| axis | computed | defaulted | % |
|---|---:|---:|---:|
| `addressee`, `aspect`, `certainty`, `domain`, `polarity`, `speech_act_category`, `speech_act_subtype`, `tense`, `urgency` | 5274 | 0 | 100.0 |
| `modality` | 2704 | 2570 | 51.3 |
| `target` | 2670 | 2604 | 50.6 |
| `engagement_level` | 2101 | 3173 | 39.8 |
| `temporal_framing` | 1330 | 3944 | 25.2 |
| `slot_completeness` | 1291 | 3983 | 24.5 |
| `response_posture` | 635 | 4639 | 12.0 |
| `sentiment_alignment` | 107 | 5167 | 2.0 |
| `novelty_score` | **0** | 5274 | 0.0 |
| `self_disclosure_level` | **0** | 5274 | 0.0 |

**Two axes never computed, down from the four this task recorded in September** — `polarity` and
`sentiment_alignment` both now fire. `sentiment_alignment` at 2.0% is the thinnest real signal in
the set and is gated by speech-act category coverage, per 081 cause 2.

This supersedes `.claude/corpus/axis_sample.json` as the reference point: that baseline was taken
under fingerprint `fb3ce7f1e4739cb3` and is not comparable — see
`.claude/findings/2026-09-25-extractor-schema-drift.md`.

### Phase 2 — coverage, corpora, synthesis

8. `mix axes.coverage` — per axis: declared values, values observed, **uncovered values**, split by
   utterance origin. This is the worklist that drives the per-axis PRs.
9. Per-axis synthesis PR: add utterances covering the gaps, with the expectation for that axis only.
   Baseline gaps already known from the controlled set — no negation at all, 3 interrogatives of 238,
   no first-person disclosure, no expressive/commissive speech acts. Those four gaps are why
   `polarity`, `modality`, `self_disclosure_level` and `sentiment_alignment` cannot currently be
   measured regardless of how good the infrastructure is.

#### Two validation regimes — Austin's corpus idea splits the axes

> "I also have some creative-commons books (and licensed books for research purposes) that could be
> used if we break them up into single sentences, and then we can judge it more like a well
> understood english project instead and still have axis to watch."

This is a better framing than synthesis-only, and it partitions the 18 axes:

| regime | axes | validated against |
|---|---|---|
| **general English** | `tense`, `aspect`, `polarity`, `modality`, `speech_act_category`, `speech_act_subtype`, `target`, `addressee`, `temporal_framing`, `certainty` | natural prose/dialogue, against established linguistic convention |
| **application-specific** | `domain`, `urgency`, `novelty_score`, `slot_completeness`, `engagement_level`, `self_disclosure_level`, `response_posture`, `sentiment_alignment` | in-domain chatbot data; no external convention exists |

The linguistic axes can be validated against **external convention** rather than internal
consensus, which is a far stronger correctness claim — "our `tense` agrees with treebank convention
on real prose" beats "our `tense` agrees with itself".

#### Sources, by register

- **Dialogue — `data/scripts/`, already in the repo.** 834 files, ~28 MB across TOS, TNG, DS9,
  Voyager, Enterprise, Discovery, SNW, TAS and the films. Format is speaker-attributed and trivially
  parseable:

  ```
  BERKELEY: Ready to beam down. Energise. Energise.
  KIRK: Having trouble, gentlemen?
  BERKELEY: I just don't understand the problem, sir.
  WOMAN [OC]: Rehab colony. Come in.
  ```

  Conversational register, which is much closer to the chatbot's traffic than narrative prose. It
  carries real questions, imperatives, negations and first-person disclosure — the four gaps above.
  Critically, **`addressee` and `target` are grounded by the speaker attribution** rather than
  inferred, so they get external ground truth for free. `[OC]` marks off-camera speech; `[Scene]`
  headers give context. (`TOS_processed/` holds only an extracted character list — no sentence work
  to reuse.)
- **Narrative prose — the CC books.** Longer clauses and richer tense/aspect than dialogue. Better
  for `tense`, `aspect`, `polarity`, `temporal_framing`.
- **Synthesis** — remains the fallback for values neither corpus covers.

#### Licensing constraint — must be designed in, not bolted on

The Trek transcripts are **fan transcriptions of copyrighted scripts**, not creative commons. The CC
books are redistributable; the "licensed for research purposes" books are not. These are three
different permission levels and they cannot be conflated.

`data/*` is already gitignored, so the sources stay local. The risk is **derived** artifacts: a
tagged-run summary committed to git must contain only aggregate statistics, hashes and axis values —
**never utterance text** from a non-redistributable source. Design requirements:

- every utterance carries `source_id` and a `license` (`cc-by` / `cc-by-sa` / `research-only` /
  `authored` / `controlled`)
- committed summaries are filtered by redistributability, and the filter is enforced in code, not by
  convention
- `mix axes.coverage` and the reports can cite counts from research-only sources but not quote them

#### Two methodological cautions

- **Sentence splitting must not use the system under test.** Using `ChunkSegmenter` to prepare its
  own evaluation data is circular — and it is one of the components being measured (task 040).
  Use an independent splitter and record which one in the run provenance.
- **Register mismatch cuts both ways.** Narrative prose and Starfleet dialogue are both unlike
  "turn off the kitchen lights". Origin tagging (`controlled` / `corpus:<source>` /
  `synthesised:<axis>`) is what keeps bits-per-axis comparable; a corpus-heavy set will move the
  numbers for reasons unrelated to correctness. Always report per-origin as well as pooled.

### Phase 3 — reports

10. `mix axes.diff <a> <b>` — which axes changed, on which utterances, alongside the provenance
    delta. This is "trace back to when a calculation started going wrong": the diff names the axis,
    the rows, and what differed in the environment.
11. Bits-per-axis (task 081) recomputed per run, keyed on tags, so the north star becomes a time
    series rather than a one-off.
12. `mix axes.validate` once expectations exist — per-axis agreement, with an explicit
    "not yet measurable" state. An axis is never "passing" merely because nothing contradicts it.

## Sequencing

Phase 0 -> 1 -> 2 -> 3. Phase 0 gates everything: measuring an unnamed vector records numbers whose
meaning can change silently. Phase 2 gates meaningful reports: coverage before validation, because a
report over uncovered axes says "no disagreements" and means nothing.

## Acceptance criteria

- [x] Every feature dimension has a name; `vector_dimension/0` derives from the manifest
- [x] Every axis has a declared value domain; out-of-domain values fail loudly
- [x] A run records enough provenance to identify what changed between any two runs — **with one
      known gap**: the fingerprint says *that* the schema moved, and `group_widths` localises it to
      a group, but a group-23 rename at constant width is only attributable from the
      `age_graph.parent_types_digest`, which no run before 2026-09-25 recorded. The 2026-09-12
      drift is therefore detectable and permanently unattributable.
- [ ] `mix axes.diff` attributes an axis change to a code, model, lexicon or dataset delta
- [ ] `mix axes.coverage` names the uncovered values per axis, split by utterance origin
- [ ] All 18 axes take more than one value on a tagged snapshot — **including the four now dead**
- [ ] Tagged-run summaries are committed; the trend survives a database reset
- [ ] Utterance origin (`controlled` / `corpus:<source>` / `synthesised:<axis>`) is recorded and filterable
- [ ] Every utterance carries a `license`; committed artifacts are filtered by redistributability in code
- [ ] Sentence splitting uses a splitter independent of the system under test, recorded in provenance
- [ ] Linguistic axes are validated against external convention, not internal agreement

## Related

081 (the baseline this operationalises, and the metric it turns into a time series), 072 (the
inversion this would have caught and dated), 073 (novelty never populated), 074 (POS tagger —
blocks `polarity`; also broader than first filed: "Turn on the lights" tags `Turn` as NOUN with no
unknown words present), 077 (canonical vocabularies), 078 (workbench, for curation later),
080 (factored intents — these axes are the factors).
