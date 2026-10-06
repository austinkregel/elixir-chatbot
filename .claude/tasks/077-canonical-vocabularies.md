# FOUNDATION: canonical vocabularies with enforced derivation

- **ID:** 077
- **Status:** **in progress — domain 1 (intent labels) underway.** Scoping questions 1 and 3 answered
  2026-09-27; see "Decisions taken". Domain 5's lexical layer **measured 2026-10-06, not reconciled**
  — see "Measured: domain 5's lexical vocabularies". Domains 2, 3, 4 and 6 untouched.
- **Area:** data-architecture
- **Branch:** `feat/lattice-home-assistant-and-memory`
- **Recorded:** 2026-09-12

## The thesis

Austin, 2026-09-12:

> "In order to deliver the functionality I want without creating a massive amount of future tech
> debt, we need to address the inadequacies in our data because our data is being merged from
> disparate sources and purposes. Reviewing all the code we've started work on patching out,
> it's clear that the previous agent was trying to solve a data problem by adding catches,
> checks, and fallbacks."

This task exists because that is correct, and because every other open task is downstream of it.

## Measured: the intent label space is defined six times (PROVEN)

| source | labels | not in registry | registry labels missing | origin / purpose |
|---|---|---|---|---|
| `priv/analysis/intent_registry.json` | **141** | — | — | hand-curated structural metadata |
| `priv/evaluation/intent/gold_standard.json` | 138 | 0 | 3 | evaluation, normalised Apr 2026 |
| `priv/response/templates.json` | **242** | **112** | 11 | response side, grown separately |
| `data/intents/` | 236 | 119 | 24 | **Dialogflow export** |
| `data/training/intents/` | **274** | **155** | 22 | POS/entity-annotated corpus |
| `data/legacy/intents/` | 236 | 119 | 24 | **byte-identical duplicate of `data/intents/`** (measured) |

**Union: 339 labels. Present in all six: 117. Agreement: 34.5%.**

Consequences already measured elsewhere:

- `templates.json` can emit responses for **112 intents the registry does not define**.
- The taxonomy was renamed 170 -> 138 around 2026-04-26..30 (`music_player_control.*` ->
  `music.player.*`; seven per-domain `*.greeting` classes plus `Default Welcome Intent` and
  `message` collapsed into `smalltalk.greet`) with **no recorded mapping**, so pre- and
  post-rename numbers are incomparable (task 008).
- The live pipeline emits composed labels (`weather.statement`, `account.greeting`) that exist
  in **none** of the six (task 038).

## The same defect in five other data domains (all PROVEN)

1. **Primitive types** — `PrimitiveTypes` declares 29 `type:variant` keys; the inventory builder
   copies names verbatim from `primitives.json`, which uses a pluralised vocabulary
   (`hedges`/`hedging`, `transition_phrases`/`transition`, `reflective_patterns`/`content:reflective`).
   21 of 29 planner keys have zero fragments; 36 inventory keys are unaddressable. This is why
   the lattice realizes 2 of 30 patterns (task 031).
2. **Feature dimensions** — `mem_novelty` and `mem_graph_known` are `1 - familiarity` and
   `familiarity`: one signal occupying two dimensions, with identical Fisher/between/within
   variance confirming perfect anti-correlation. Two further memory dims are constant zero
   (task 073). **A second layer under this one was measured 2026-10-06** — the word lists the
   dimensions are computed from are defined 2 to 6 times each; see "Measured: domain 5's lexical
   vocabularies" below.
3. **Graph node labels** — `DateTime`, `Datetime` and `Date` coexist as distinct labels, as do
   `SlotMappings` and `Slotmappings`. Same concept, different casing, separate nodes.
4. **POS annotations** — ~~the trainer reads the gold standard (no `pos_tags`), silently fabricates
   labels from a hardcoded `@prepositions` list, while 26,380 annotated tokens sit unread in
   `data/training/intents/`~~ (task 076). **Two of these three clauses are false as of
   2026-09-28**; see "Correction: the POS clause" below.
5. **Derived artifacts** — `data/classifiers/*.json` are derived from the gold standard via the
   feature extractor, but nothing records or verifies which extractor version produced them.
   The manifest stores hashes that nothing checks; six of them currently mismatch and the
   memory dimensions are inverted relative to what the models learned (task 072).

## Correction: the POS clause (verified 2026-09-28)

Domain 4 above was checked against the code and is mostly wrong. Measured twice by different
methods, agreeing exactly: **26,374 tokens**, not 26,380, across 5,416 rows, plus 3,650 entity spans
across 2,369 rows that this task does not mention.

| clause | verdict |
|---|---|
| "the trainer reads the gold standard (no `pos_tags`)" | **FALSE** — it reads UD EWT |
| "silently fabricates labels from a hardcoded `@prepositions` list" | **FALSE** — no such code exists in `apps/` |
| "26,380 annotated tokens sit unread" | count off by 6; **unread *by POS training* is true** |

- POS trains from `apps/brain/priv/training/pos/ud_ewt.{train,dev,test}.json` (`pos.ex:126-129`),
  imported by `mix pos.import_ud_ewt`, which verifies SHA-256 against a pinned manifest and
  hard-fails on an unmapped tag. Train split: 12,544 sentences / **204,578 tokens** — **7.7× the
  whole intents corpus**, so the "unread" tokens are not a lost opportunity of any size.
- `@prepositions`, `auto_enrich_pos` and `rule_based_pos` appear only in `.claude/tasks/*.md`.
- **Task 076 already struck its own first acceptance criterion** (`done-076-…md:112-115`):
  *"superseded. Those records' provenance was never verified."* That instinct was right — see the
  provenance finding below.
- They are not unread in general: `GoldStandardMigrator` consumes `text`, `tokens`, `pos_tags`, `id`
  and `entities`, and `scripts/{pos_tag_coverage,validate_pos_annotations}.py` read them.
- A live dead read: `gold_standard.json` has **zero** rows with a `pos_tags` key, yet
  `data_loaders.ex:201` reads `item["pos_tags"] || []` from it and so always gets `[]`.

## The corpus was 17.8% synthetic and said it was not (fixed 2026-09-28)

`scripts/materialize_orphan_intents.exs` writes gold-standard texts that have no source backing back
*into* `data/intents/*_usersays_en.json`, tagging them `orphan-<hash>` or `augmented-<hash>`;
`rebuild_gold_standard` then reads that directory as canonical. Every row was stamped
`labeled_by: "dialogflow"`, all 4,868 of them, including the 864 that were not.

| | `data/intents` | `gold_standard.json` | `held_out.json` |
|---|---:|---:|---:|
| genuine Dialogflow | 82.6% | 82.3% | 82.2% |
| `orphan-` (materialized) | 12.0% | 12.2% | 12.3% |
| `augmented-` | 5.4% | 5.5% | 5.5% |

55 of the 1,000 held-out rows are word-order scrambles — *"is bright it enough everywhere"*, *"my
laptop screen is way too"*. Every accuracy figure in this task was measured partly against those.

`labeled_by` is now derived from the row's id, and both `rebuild_gold_standard` and
`split.held_out` print the composition. Nothing was removed: what to do about the 864 rows is a
separate decision. This is the same defect the task's own thesis describes — a vocabulary asserting
something the data does not support — one layer below where the task was looking.

## Measured: domain 5's lexical vocabularies are defined 2 to 6 times (PROVEN 2026-10-06)

Domain 5 in the list above describes two *dimensions* carrying one signal. Below that sits a second
layer the task had not looked at: the word lists those dimensions are computed from.

Surveyed across `apps/brain/lib` — **204 `~w(…)` literals in 53 files**. Excluded as legitimate:
`lib/mix/tasks/` (Austin's stated exception — "Mix tasks, or data mutation asks are the primary
exception"), `code/tokenizer.ex`'s 9 per-language keyword tables and digit/hex alphabets, and
`html_processor.ex`'s HTML tag names. What remains:

| concept | independent definitions | union | shared by all | declared source |
|---|---:|---:|---:|---|
| question words | **6** | 16 | **1** (`what`) | none |
| hedges | **5** | 20 | 4 | `LinguisticData.hedges()` — 10 |
| entity / slot types | **5** | — | — | `entity_types.json` (15), `entity_slot_mappings.json` (13) |
| agreement / backchannel | **4** | 22 | **1** (`ok`) | none |
| pronoun, 1st person | **4** | 10 | 10 | `closed_class.json` `PRON` — 55 |
| imperative verbs | **3** | 29 | 18 | none |
| politeness markers | **3** | 4 | **1** (`please`) | `discourse_config.json` `polite_markers` — 4 |
| greeting tokens | **3** | 9 | 3 | `discourse_config.json` `address_prefixes` — 6 |
| stopwords | **3** | — | — | none |
| pronoun, 2nd person | 3 | 5 | 4 | `PRON` |
| conditional markers | 3 | — | — | `closed_class.json` `SCONJ` — 15 |
| modals | 2 whole + 2 partial | — | — | `closed_class.json` `AUX` — 23 |
| content POS tags | 2 (exact copies) | 6 | 6 | `POSTagger.valid_tags/0` — 16 |
| pronoun, 3rd person | 2 | 16 | 12 | `PRON` |
| days / months | 2 | — | — | none |
| intensifiers | 2 | — | — | `LinguisticData.intensifiers()` — 10 |

### Four pairs are byte-identical copies

| copy A | copy B |
|---|---|
| `speech_act_classifier.ex:844` `@seed_imperative_verbs` (29) | `discourse_analyzer.ex:287` `imperative_verbs` (29) |
| `chunk_features.ex:49-50` 1st-person pronouns (10) | `discourse_analyzer.ex:33` `@first_person_pronouns` (10) |
| `chunk_features.ex:51` 2nd-person (5) | `chunk_profile.ex:829` inline (5) |
| `racing_analyzer.ex:305` question words (9) | `ml/tokenizer.ex:345` `@question_words` (9) |

**The first pair has already diverged, today.** Task 088 expanded `speech_act_classifier`'s seeds
through `Brain.Lexicon.dominant_sense_synonyms/3`. Its identical copy in `discourse_analyzer` was
not touched. Two private functions *of the same name* — `has_imperative_start?/1`, at
`speech_act_classifier.ex:828` and `discourse_analyzer.ex:286` — now answer the same question
differently from what was the same list. This is precisely the failure mode the task's thesis
describes, created by this project rather than inherited from it.

### The disagreements are not stylistic

- **Question words: 6 definitions, and only `what` appears in all six.** Two of them
  (`knowledge/types.ex:867`, `research_agent.ex:118`) fold auxiliaries (`is are was were does did`)
  into the same list as interrogatives; the other four do not, and they disagree about
  `why`/`which`/`whom`/`whose`.
- **Agreement tokens: 4 definitions, only `ok` in all four — and two of them are in the same
  module.** `speech_act_classifier.ex:285` `@backchannel_tokens` (14) and `:287` `@ack_tokens` (7)
  overlap on `ok okay` and otherwise diverge; `chunk_segmenter.ex:235` `@acknowledgment_tokens` (5)
  does not even contain `ok`.
- **Pronoun lists lose members silently.** `discourse_analyzer.ex:30` omits `yourselves`; `:36`
  omits all four 3rd-person reflexives; `learner.ex:610` omits `us`, `ours`, `ourselves`, `myself`.
  Each is a word the system then cannot see as a pronoun on that path.
- **The declared sources are consistently richer than the code**, so reconciling *adds* coverage
  rather than trading it: `PRON` has 55 against the extractor's 40; `AUX` 23 against 10; `SCONJ` 15
  against 5. The exception is hedges, where the union of code usage (20) exceeds the declared list
  (10) — there, the declaration has to grow.

### A live defect this located

`has_imperative_start?/1` at `speech_act_classifier.ex:828-842` skips one leading politeness token
before looking for a verb, using **string literals in a `case`** — `"please"` and `"kindly"` — while
`@please_tokens` in the same module declares four: `please pls plz kindly`. So *"please turn on the
lights"* is read as imperative and *"pls turn on the lights"* is not. The declared source
(`discourse_config.json` `polite_markers`) lists only `please`, and is phrase-shaped
(`"could you"`, `"would you"`), so a one-position token lookahead cannot consume it. Reconciling
this vocabulary is what makes the defect fixable without a third hardcoded list;
`discourse_analyzer.ex:286` has no politeness handling at all.

### How this reaches the feature vector

The extractor family references only `Brain.Analysis.TypeHierarchy` and `Brain.ML.Tokenizer`
directly. Everything else arrives as fields on the `chunk` map that `ChunkFeatures` then reads into
dimensions — `@speech_act_categories`, `@question_subtypes`, `@addressee_values`, `@entity_types`.
So the vector's dimensions are populated by upstream modules whose vocabularies disagree with the
extractor's own *and* with each other, and the disagreement is invisible at the extractor boundary
because the one-hot slot is filled either way.

### Applying this task's four criteria to domain 5

1. **Canonical source** — `priv/knowledge/closed_class.json` for closed-class words (pronouns,
   modals, conditional markers, determiners), `priv/knowledge/linguistic.json` for hedges,
   intensifiers and negation, `POSTagger.valid_tags/0` for POS tags,
   `priv/analysis/entity_types.json` for entity types (shared with domain 3),
   `priv/analysis/discourse_config.json` for politeness and address markers. Four concepts —
   question words, imperative verbs, agreement tokens, stopwords — have **no declared source and
   need one created**.
2. **Declared relationship for every other source** — every list in the table above is `delete`,
   resolving to a read of the canonical source. None of them is a legitimate local variant; the
   survey found no case where a module needs a different pronoun list than its neighbour.
3. **Explicit mapping** — needed where code and declaration differ in *kind*, not just extent:
   `polite_markers` is phrase-shaped where the code is token-shaped, and the two
   question-word definitions that fold in auxiliaries are answering a different question
   (`is this interrogative?` vs `does this open with a wh-word?`) and must become two named
   vocabularies rather than one merged list.
4. **Enforcement at the boundary** — the declared-vocabulary pattern already in use
   (`@external_resource` + compile-time load + `raise` on invariant violation) applied per module,
   so a module reading a vocabulary that no longer contains a word it depends on fails to compile.
   Per Austin's ruling on fingerprints: correctness goes in **affirmative tests** asserting what
   each dimension should contain, not in runtime hashes that only detect that something moved.

## Why the fallbacks existed

Each silent fallback catalogued in `.claude/SILENT_FALLBACKS.md` sits at a junction between two
of these vocabularies. A few examples:

| fallback | disagreement it absorbed |
|---|---|
| `@slot_schemas` -> `%{}` | slot schemas declared in `intent_registry.json` **and** a `slot_schemas.json` that was never generated |
| `if is_list(phrases), do: phrases, else: []` | `primitives.json` nests `attunement.empathy` one level deeper than its siblings |
| `transition` default `0.3` | builder emits boundary bigrams; runtime queries boundary + within-tail + within-head |
| `@memory_context_default [0.5, ...]` | `ContextAccumulator` may or may not have graph/memory context depending on entry point |
| `rescue _ -> %{}` in `Graph.Reader` x12 | graph label casing and shape variance |
| `default_propn_type: "person"` | no canonical answer for "unknown proper noun type" |

Removing the fallback without reconciling the vocabulary just moves the failure earlier. That is
the correct direction — fail loudly at the junction — but it is only half the work.

## What "addressing the data" means concretely

For each data domain, four things must exist:

1. **One canonical source**, named, with its purpose stated.
2. **A declared relationship for every other source** — `derived_from`, `retired_in_favour_of`,
   `foreign_origin_requiring_mapping`, or `delete`.
3. **An explicit mapping** wherever a rename or consolidation happened, so historical artifacts
   remain interpretable (the 170 -> 138 intent rename is the urgent one).
4. **Enforcement at the boundary** — a value outside the canonical vocabulary cannot enter
   training data, cannot be written to the graph, cannot be emitted at runtime. Hard failure,
   not coercion.

## The hook that already exists — do not build a parallel system

`Brain.ML.TrainingData.SourceDescriptors` already models most of this. Each descriptor carries
`id`, `label`, `category`, `tag`, `record_kind`, `path`, `description`, `upstream_of`,
`generated_by`. That is a provenance graph. What is missing:

- It is **descriptive, not enforced** — nothing validates that a source's contents conform to the
  vocabulary its `record_kind` implies.
- It is **incomplete** — `data/intents/`, `data/legacy/intents/`, `data/training/intents/` and
  `templates.json`'s label space are not described.
- `upstream_of` is populated inconsistently, and there is no version/hash link between a derived
  artifact and its inputs (the gap task 072 fell through).

Extending `SourceDescriptors` into the canonical registry is the right move. It also makes the
Training Studio the natural UI for reconciliation work, rather than a new tool.

## Domains to reconcile, in dependency order

| # | domain | canonical candidate | blocks |
|---|---|---|---|
| 1 | intent labels | reconciled `intent_registry.json` | 013-015 measurement, 038 phantom labels, 072 retraining |
| 2 | POS tags | `data/training/intents/` annotations | 076, 074, and every POS-keyed feature |
| 3 | entity types | `priv/analysis/entity_types.json` + graph label casing | 042, gazetteer pollution |
| 4 | primitive types | `PrimitiveTypes` | 031, 032 — the lattice |
| 5 | feature dimensions | `ChunkFeatures` group declarations, over the declared lexicons in `priv/knowledge` and `priv/analysis` | 072, 073, 088 |
| 6 | slot names | `intent_registry.json` slot schemas + service `slot_schema/0` | 047, 061 |

## Decisions taken — 2026-09-27

**Question 1 (canonical source for intent labels): the Dialogflow export, reconciled UP.**

`data/intents/*_usersays_en.json` is canonical. The registry was reconciled up to match it, 138 ->
**207** entries, rather than the corpus being reconciled down to the registry.

Reasoning, in the order it was established:

- Task 079's premise held but its attribution was wrong. The collapse it blamed on a classifier is
  96.2% reproduced by `cleanup_gold_standard`'s 123 hardcoded rename rules — a deliberate
  consolidation, not ML damage. `@smarthome_consolidation` alone has 62 rules, and five of its six
  targets are exactly one axis value at 100%.
- Dialogflow covers the corpus completely: 5,274 distinct normalised texts, matching **100%** of both
  the current and the pre-rebuild gold standards. Only 108 texts mapped to more than one intent, and
  the lights-to-device fold reduced that to 50.
- The test suite already expected the fine taxonomy — 114 references to `smalltalk.greetings.*` /
  `smarthome.lights.*` / `music.player.*` against 14 to the collapsed forms.
- Reconciling up cost nothing statistically: median examples per label 15 against the collapsed
  taxonomy's 14.

`gold_standard.pre-rebuild.json` (5,299 rows, 208 labels, git-tracked since `c6845c8`) is an
independently derived witness of the pre-collapse state; the rebuild agrees with it **96.9%**.

**Question 2 (`data/legacy/intents/`): answered previously, and acted on.** The directory is gone.
`data/legacy/entities/` was **not** deleted — it is genuine divergence, not duplication.

**Question 3 (is the Dialogflow lineage worth keeping): keep it.** It is the canonical source per
question 1. Cleaned rather than discarded:

| removed | why |
|---|---|
| `Default Welcome Intent` | all 16 phrases were already present in `smalltalk.greetings.hello` — a pure duplicate |
| `message` | merged into `communication.text`, which the registry already declared |
| `weather` (bare) | renamed `weather.query`, which the registry already declared |
| `smarthome` (bare) | 13 room-slot fragments ("living room") with no base intent — not an intent |
| `weather.query_usersays_en.json` | contained only "What is the test intent?" twice |
| `test.intent.promotion_usersays_en.json` | contained only "What is the test intent?" |
| 18 device-scheduling intents, 356 phrases | scheduling a device belongs in the app, not in conversation |

**Question 4 (reconciliation before or alongside the verification pages): reconciliation first**, by
consequence rather than by decision — domain 1 was reconciled before any page work began.

## The 170 -> 138 rename mapping this task asks for

The acceptance criterion *"the 170 -> 138 intent rename has a recorded mapping"* is now satisfiable
from material in the repo, and this is where it is recorded:

- **The rules** are `cleanup_gold_standard.ex`'s module attributes: `@smarthome_consolidation` (62),
  `@consolidation_renames` (17), `@smalltalk_user_emotion_renames` (15), `@music_map` (10),
  `@nav_map` (9), `@phantom_renames` (7), `@device_to_lights_map` (15), `@extra_renames` (3) — 123
  rules in total after excluding the device/lights map, which runs opposite to the current fold.
- **The before state** is `apps/brain/priv/evaluation/intent/gold_standard.pre-rebuild.json`.
- **The verification** is that applying those rules to that file reproduces 4,699 of 4,887 shared
  labels (96.2%). The 188 unexplained are `navigation.*` routed to `news.search` /
  `music.player.skip_forward`, which the maps do not cover.

Note that `@device_to_lights_map` maps device -> lights, the **opposite** direction to the ruling that
lights are taxonomically a device. It must not be treated as precedent.

## Remaining open scoping questions

1. **`data/legacy/entities/` still needs a merge decision.** Not a duplicate: 39 files in
   `data/entities/` vs 70 in legacy; of 37 shared filenames only 4 are byte-identical and 33 differ;
   legacy holds 33 files the current set lacks. This belongs to domain 3 (entity types).
2. **`data/training/intents/` (273 labels, 98 not in the corpus).** Unclassified. Are they pre-fold
   spellings, removed scheduling intents, or genuinely different? Measure before deciding.
3. **The ~29 residual orphaned template labels** (of the 66 not in the registry) with no obvious
   counterpart — retire or remap. The other 37 are decided; see "Domain 1 progress".

## Domain 1 progress — 2026-09-27

| gap | before | now |
|---|---:|---:|
| corpus intents not in the registry | 69 | **0** |
| held-out rows hitting the `registered_intent?/1` gate | 475 / 1000 | **0** |
| held-out intent accuracy (pipeline) | 15.5% | **29.1%** |
| corpus intents with no response template | 23 | **3** |
| template labels not in the registry | 66 | **2** (both keepers) |
| annotated-corpus labels not in the corpus | 98 | 98 (unmeasured) |

The registry gap was not a completeness problem — it was a correctness one. `determine_intent/6`
tests `registered_intent?/1` and, when it fails, discards the classifier's label and infers from the
speech act. Measured: the pipeline turned **256 correct predictions into wrong ones** while rescuing
22. The classifier itself was at 38.9% throughout; closing the gate recovered 13.6 of those points.

That is this task's thesis with a number attached, and the reason it is upstream of everything else.

### Template reconciliation — `mix templates.reconcile` (c82ab2f)

The task applies only the two transformations the corpus rebuild applies, and refuses to invent a
rename for anything else. 24 of the 66 orphans resolved: 20 `smarthome.lights.*` keys moved onto
`smarthome.device.*`, 4 `account.* - context: …` spellings merged into their base intent. 656
templates before, 656 after, verified as an exact set match on `{key, text, condition}` against the
renames computed independently of the task.

The 3 corpus intents still without a template are `smarthome.device.switch`,
`smarthome.device.volume` and `smarthome.heating` — the context continuations with no usersays file
either. Synthesizer serves them, which is the decision recorded above.

### Retirement — `mix templates.reconcile --retire` (da5d9b4)

Austin ruled delete. The other 42 were graded by reachability rather than registry membership — the
registry cannot answer reachability, since `question.factual` is a literal default in the pipeline
and `unknown` is what most of `speech_act_intent_map.json` routes to, and both look dead to a
registry check. 40 retired, 60 templates, leaving 2:

| grade | disposition | labels |
|---|---|---|
| reachable via `speech_act_intent_map` | keep | 1 — `unknown` |
| reachable on a live path | keep | 1 — `question.factual` (`pipeline.ex`, `racing_analyzer.ex`) |
| namespace, not an intent | retire | 2 — `music.player`, `weather` |
| cited only by a rename table | retire | 34 — incl. all 13 device-scheduling labels |
| no reference anywhere | retire | 4 — `Default Welcome Intent`, `smalltalk.confirmation.{yes,no,cancel}` |

The **namespace** grade replaced a hardcoded list of four "too generic to grep for" names. A key that
is a dotted prefix of registry keys without being registered itself names a namespace: `weather` is
the parent of `weather.query` and four siblings, `music.player` of ten. The test runs *before* the
name search, because a search cannot separate an intent label from a domain of the same spelling —
`"weather"` occurs in 24 test files as the domain — while the registry's structure settles it. The
rule found `music.player` on its own, which had been sitting in the no-reference bucket on a guess.

`--retire` refuses while any key is undecided, so a key the grader cannot place is never swept into a
deletion. It is idempotent: a second run finds nothing and leaves the file byte-identical.

**Safety, verified rather than argued:** every retired key was absent from the registry, so no
registered intent could lose its text. Measured — registry intents without a template stays at 11, no
surviving entry was mutated, 0 keys added, and no test looks up a retired key as a template key. The
3 failures in `test/brain/response` are identical before and after and are intent-classification
failures (`knowledge.capital` for a greeting), not template ones.

### A defect this surfaced, not fixed

`Jason.encode!` emits a map in Erlang iteration order, which is a function of the map rather than of
its content: adding one key reshuffles the file. `templates.json` was written that way, so a 24-key
rename produced a 3,100-line diff. `templates.reconcile` now sorts every object before encoding and
repeated runs are byte-identical — but it is the only writer that does. **`template_store.ex:687`
writes the same file with a plain `Jason.encode!`**, so the first admin-UI save reshuffles it again,
and 51 files across the umbrella encode independently. A shared deterministic writer is the fix; 51
call sites is not a change to make in this pass.

Separately, `registry_derive.ex:272` writes a timestamped `.bak` beside a git-tracked file, which is
where the stray `intent_registry.json.2026-09-27T01-34-39.033862Z.bak` came from.

## The remaining accuracy gap, attributed — `mix intent.attribute` (4bddc77)

Classifier 37.3%, pipeline 29.1%. The decomposition closes arithmetically, which is the check that
it is one measurement and not three loosely-related ones:

| component | rows |
|---|---|
| classifier top-1 on the pipeline's own vector | 373 |
| lattice rerank (`refine_with_intent` + `apply_profile_rerank`) | 17 destroyed, 38 rescued — **+21** |
| `:domain_conflict` branch (`pipeline.ex:1167`) | 117 destroyed, 14 rescued — **−104** |
| `:unregistered_label` | 0 |
| `:low_confidence_floor` | 0 |
| `ContextualEntityInferrer.infer/5` | 0 |
| **pipeline** | **373 + 21 − 104 = 291** |

**All three of the paths this plan suspected are innocent, and the method it proposed would have
hidden that.** The plan named the domain-conflict branch, `disambiguate_with_atlas/6` and
`@low_confidence_floor 0.3`, and proposed reconstructing the branch from the thresholds. That cannot
work: `determine_intent/6` returns `:speech_act_fallback` from three different branches, and every
one of the 117 destructions came from a single one of them. Measured:

- **`@low_confidence_floor 0.3` guards a branch no correct answer reaches.** Zero of the 373 correct
  classifier predictions score below 0.3. Moving that threshold cannot recover a row.
- **`disambiguate_with_atlas/6` neither destroys nor rescues anything** on this split.
- **`ContextualEntityInferrer` is a fourth override path this plan did not list** — it can rewrite
  `intent` after `determine_intent/6` returns (`pipeline.ex:598`) — and it costs nothing. Worth
  recording because a reconstruction from thresholds would have charged its mistakes to the
  domain-conflict branch.
- **The lattice rerank is net positive (+21).** It moves 124 labels, 38 wrong→right against 17
  right→wrong. An easy thing to blame and remove; the numbers say don't.

### The domain-conflict branch's premise is false where it is applied

The branch discards a registered classifier label whenever `:intent_domain` disagrees, on the stated
grounds that the domain model "is far more reliable than the 232-class intent_full centroid". Tested
against the gold label's own prefix:

| | |
|---|---|
| `:intent_domain` correct, all 1000 rows | **52.7%** |
| `:intent_domain` correct, on the 405 rows where it overrode | **16.8%** |
| replacement landed in gold's domain, on the 134 destroyed rows | **17.2%** |

The disagreement selects for the domain model being wrong. On those same 405 rows the classifier's
label was right 118 times (29.1%) and what the branch produced instead was right 14 times (3.5%) —
the 8.4:1 destruction ratio, from the other direction.

**A second, structural defect at the same site.** Even where the domain model is right, the branch
never uses it: on disagreement it discards *both* labels and calls `infer_intent_from_speech_act`,
which re-scores all 207 registry intents from scratch and ignores the lattice `top_k` it already
has. That is why 83% of replacements land outside the gold domain. A prior fix dated 2026-09-23
(recorded in the comment at `pipeline.ex:1255-1261`) caught a *vocabulary* mismatch in this same
comparison that was discarding every correct prediction for 22 intents; both sides are consolidated
now, so this is a different defect, not a regression of that one.

**Arithmetic implication, a prediction and not a result:** neutralising the branch should move
held-out accuracy from 29.1% to about **39.5%**, above the raw classifier's 37.3%, because the
rerank's +21 survives. Untested — no change has been made.

### Two hypotheses tested and dropped

Recorded so they are not inherited as fact. The concentration of replacements on four labels
(`knowledge.capital` 14, `weather.activity` 12, `device.control` 11, `weather.condition` 10) is **not**
explained by the alphabetical tie-break at `pipeline.ex:1522`: only 63 of 134 replacements are the
alphabetically-first intent in their domain. And the fallback does not get the domain right and the
leaf wrong — it is the reverse, 17.2% domain agreement.

### Incidental, unfixed

`mix evaluate.classifiers` calls `EvaluationStore.load_gold_standard("intent")` with no `:held_out`
(`evaluate_classifiers.ex:27`), so it reports training-set accuracy — the defect `evaluate.intent`
was fixed for and documents in its own comments.

The pipeline is not bit-deterministic: three runs of the same split gave 373, 374, 373 correct. ±1
row, worth knowing before treating a one-row delta as signal.

## Acceptance criteria

- [ ] Every data domain above has exactly one declared canonical source — **1 of 6 done** (intent
      labels: the Dialogflow export)
- [ ] Every other source declares its relationship to the canonical one
- [x] The 170 -> 138 intent rename has a recorded mapping — see the section above
- [ ] A value outside a canonical vocabulary cannot enter training data, the graph, or runtime
      output — enforced, with a loud failure
- [x] Derived artifacts record the version/hash of every input, and loading verifies it — done for
      micro-classifiers via `Brain.ML.MicroProvenance`; not yet for the lattice or POS artifacts
- [ ] `SourceDescriptors` covers all six intent sources and is validated, not just descriptive —
      surveyed 2026-09-27: `upstream_of` is populated **once in 79 descriptors**, there is no
      `derived_from`/hash/version field, `Schemas.validate_all/2` returns `:ok` unconditionally for
      every map-shaped source (`schemas.ex:145`), and the whole subsystem has zero tests

## Verify (current state)

```sh
python3 - <<'EOF'
import json,glob,os
def s(f):
    try: return json.load(open(f))
    except Exception: return None
srcs={}
r=s("apps/brain/priv/analysis/intent_registry.json"); srcs["registry"]=set(r) if isinstance(r,dict) else set()
g=s("apps/brain/priv/evaluation/intent/gold_standard.json"); srcs["gold"]={e.get("intent") for e in g}
t=s("apps/brain/priv/response/templates.json"); srcs["templates"]=set(t)
srcs["dialogflow"]={os.path.basename(f)[:-5].split(" - context_")[0] for f in glob.glob("data/intents/*.json") if "_usersays_" not in f}
srcs["legacy"]={os.path.basename(f)[:-5].split(" - context_")[0] for f in glob.glob("data/legacy/intents/*.json") if "_usersays_" not in f}
ann=set()
for f in glob.glob("data/training/intents/*.json"):
    d=s(f) or []
    for rec in d:
        if isinstance(rec,dict) and rec.get("intent"): ann.add(rec["intent"])
srcs["annotated"]=ann
for k,v in srcs.items(): print(f"{k:12s} {len(v)}")
u=set().union(*srcs.values()); i=set.intersection(*srcs.values())
print(f"union {len(u)}  in-all {len(i)}  agreement {100*len(i)/len(u):.1f}%")
EOF
```

## Related

Upstream of: 072, 073, 074, 076, 031, 032, 038, 013-015, 042, 047, 061.
Evidence: `.claude/SILENT_FALLBACKS.md`, `.claude/determinism.exs`, `.claude/pos_gate.exs`.
