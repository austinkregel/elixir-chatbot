# Intent architecture: factored intents as compressed concepts

- **ID:** 080
- **Status:** proposal — for discussion, nothing implemented
- **Area:** data-architecture
- **Branch:** `feat/lattice-home-assistant-and-memory`
- **Recorded:** 2026-09-12

## The question Austin asked

> "This actually begs a bit of a deeper question about how these intents *should* be formed for
> what this system is capable of analyzing... I like the idea of having human readable general
> intents at a baseline, but I also like the idea of intents being able to be more specific as
> the system learns more different ways to express the same things. I definitely think there
> should be a limit, or an intentional design decision around naming conventions."
>
> "In an earlier session, I had expressed the idea of text compression in the form of simplified
> concepts being shorthand for more complex expressions of the idea. Those three part small talks
> were a remnant of that exploration. The idea being we use the intent to represent a compressed
> version of what we want said/interpreted in."

## The core defect: a flat string hiding orthogonal axes

`smarthome.lights.brightness.schedule.down` is one opaque token. It is actually five
independent facts:

| axis | value |
|---|---|
| domain | smarthome |
| object | lights |
| property | brightness |
| operation | decrease |
| modality | schedule |

Flattening them into a string has three consequences, all measured:

1. **The classifier must learn N opaque classes instead of a handful of small axes.** 141
   registry intents, many with ~10 examples. `smarthome.device.brightness.schedule.down` has 4.
2. **A new combination requires minting a new label.** There is no way to say "same operation,
   different object", so every cross-product cell becomes its own class.
3. **It duplicates axes the system already computes and then throws away.**

### What the system already analyses (PROVEN)

`Brain.Analysis.ChunkProfile` computes 18 axes per chunk:

```
domain · speech_act_category · speech_act_subtype · target · modality · polarity
tense · aspect · addressee · urgency · certainty · sentiment_alignment
slot_completeness · novelty_score · response_posture · engagement_level
self_disclosure_level · temporal_framing
```

Six of these are trained feature-vector micro-classifiers (`intent_domain`, `tense_class`,
`aspect_class`, `urgency`, `certainty_level`, plus `intent_full`); ten more are text
classifiers. The machinery is axis-oriented already. Only the *label* is not.

`ChunkProfile.derived_label/1` even composes `"#{domain}.#{speech_act_subtype}"` — a factored
label — but because the surrounding system expects flat registry strings, that composition
produces unusable phantoms (task 038).

## Addendum 2026-09-26 — the factorisation was written by hand, and the axes are measured

Three things measured while rebuilding the gold standard. The first corrects this task; the other
two are the numbers it asked for.

### 1. It was not the model. A human wrote the factorisation in code.

The section below reads the collapse as "the model recovering the shared operation". **That
attribution is wrong.** `mix cleanup_gold_standard` contains 123 hardcoded rename rules, and its
`@smarthome_consolidation` block alone has 62. Applying those rules to
`gold_standard.pre-rebuild.json` reproduces **96.2%** of the current labels (4,699 of 4,887 shared
texts). Task 079's headline example — "locking your front door and unmuting your speakers are now
the same intent" — is two literal lines in that map.

**This makes the argument stronger, not weaker.** Checking what each consolidation target's sources
share:

| target | sources | shared axis value |
|---|---:|---|
| `smarthome.schedule_create` | 13 | `modality=schedule` — **13/13, 100%** |
| `smarthome.device_check` | 8 | `operation=check` — **8/8, 100%** |
| `smarthome.device_down` | 8 | `operation=down` — **8/8, 100%** |
| `smarthome.device_up` | 8 | `operation=up` — **8/8, 100%** |
| `smarthome.device_set` | 5 | `operation=set` — **5/5, 100%** |
| `smarthome.switch` | 20 | binary-toggle operations: on/off, lock/unlock, open/close, mute/unmute, plus their checks |

Five of six targets are exactly one axis value at 100%. The sixth groups the binary toggles, which
is a coherent super-category of the same axis. **`@smarthome_consolidation` is a factored classifier
implemented by hand.** A person reached for this factorisation deliberately, which is better
evidence than a model stumbling into it — and task 079's residual ~3.8%, including the genuine
direction errors, is where the classifier separately did damage.

### 2. The positional parse works

231 flat labels over 5,394 deduplicated utterances parse into axes as this task predicts:

| axis | values | median examples | values with <10 |
|---|---:|---:|---:|
| `domain` | 28 | 35 | 2 |
| `object` | 46 | 15 | 1 |
| `operation` | 26 | **50** | **0** |
| `modality` | 3 | 359 | 0 |
| *flat intent* | *231* | *15* | *15* |

### 3. The conditioning argument, measured against the real classifier

`FeatureVectorClassifier` gives each class `k = min(4, n div 10)` prototypes:

| scheme | classes | k=0 (unservable) | k<=1 | k=4 | % at full capacity |
|---|---:|---:|---:|---:|---:|
| flat intent | 231 | 15 | **156** | 30 | **13.0%** |
| axis: `domain` | 28 | 2 | 8 | 13 | 46.4% |
| axis: `object` | 46 | 1 | 25 | 13 | 28.3% |
| axis: `operation` | 26 | **0** | 7 | 14 | **53.8%** |
| axis: `modality` | 3 | 0 | 0 | 3 | **100.0%** |

Flat leaves **156 of 231 classes on a single prototype and 15 unservable**. The operation axis has
none unservable and a majority at full capacity.

**What this does not show.** It measures *conditioning*, not joint accuracy. A factored predictor
must get every axis right to reproduce a flat label, so per-axis gains do not automatically
compound. This task's own recommendation — train the axis classifiers and compare the number
against `intent_full` — remains the thing that settles it. `intent_full` is currently at **29.6%**
on a 500-entry gold sample.

### 4. It dissolves four "unplaceable" labels

The Dialogflow-only rebuild raises on 4 labels that are not dotted intent paths. Under factoring,
two of them are not intents at all:

| label | phrases | what it actually is |
|---|---:|---|
| `smarthome` | 25 | **axis bindings.** No base intent exists; the phrases come from `- context_room` and `- context_schedule` follow-ups: "living room", "do it everyday", "weekly" — `room` and `recurrence` values |
| `weather` | 129 | a real base intent plus 6 context files whose phrases are `activity` ("skiing"), `location` and `horizon` bindings |
| `Default Welcome Intent` | 16 | a greeting carrying Dialogflow's default display name |
| `message` | 4 | a real intent with a bare domain name |

The first two are unplaceable *because the taxonomy is flat*. This task predicted exactly that:
the context follow-ups are the axis values, and there is nowhere in a dotted string to put them.

---

## The collapse was the model finding the factorisation (superseded — see addendum 1 above)

Task 079 found the gold standard was overwritten with classifier predictions. Look at *what* it
collapsed:

| gold label | absorbed | what they share |
|---|---|---|
| `smarthome.switch` | 24 labels, 576 utts | `operation = switch` |
| `smarthome.device_down` | 12 labels, 373 utts | `operation = decrease` |
| `smarthome.schedule_create` | 17 labels, 344 utts | `modality = schedule` |
| `smarthome.device_up` | 11 labels, 308 utts | `operation = increase` |
| `smarthome.device_check` | 11 labels, 183 utts | `modality = check` |
| `smarthome.device_set` | 7 labels, 150 utts | `operation = set` |

Every collapse is **one axis held constant while the others vary**. The model was not confusing
unrelated things; it was recovering the shared operation because that is the strongest signal
available and the label space gave it no way to express "same operation, different object".

The genuine errors — `saturation.up -> device_down`, `weather.temperature -> device_check` —
are direction and domain mistakes on top of a correct factorisation instinct.

**This is the strongest argument for factoring: the model already votes for it.**

## The compression idea, already prototyped

`data/decompressor/training_pairs.json` (4,899 lines) holds exactly the scheme Austin
described — one compressed concept, many surface forms:

```json
{"primitive": {"type": "acknowledgment", "variant": "social",
               "content": {"speech_act_sub_type": "greeting"}},
 "output": "Hello!"}
{"primitive": {"type": "acknowledgment", "variant": "social", ...}, "output": "Hi there!"}
{"primitive": {"type": "acknowledgment", "variant": "social", ...}, "output": "Hey!"}
{"primitive": {"type": "acknowledgment", "variant": "social", ...}, "output": "Good to see you!"}
```

Note the shape: **`{type, variant, content}` separates the concept from its parameters.** The
response side has had this from the start. The intent side never did.

`smalltalk.agent.beautiful` / `smalltalk.agent.clever` / `smalltalk.agent.funny` are the
remnant Austin identified: three-part labels trying to encode `(act, target, quality)` in a
string because there was nowhere else to put it.

## Proposal: intent = base concept + axis bindings

```
smarthome.switch { object: lights, state: on,     modality: command  }
smarthome.switch { object: locks,  state: locked, modality: schedule }
smalltalk.greeting { register: flirtatious }
weather.query { property: temperature, horizon: tomorrow }
```

Three parts:

**1. A small closed base vocabulary — the human-readable general intent.**
`domain.act`. Roughly 30-40 entries. Notably, the model already discovered eight of them:
`switch`, `device_down`, `device_up`, `device_set`, `device_check`, `schedule_create`,
`greet`, `compliment`. That is not a coincidence; it is the factorisation asserting itself.

**2. Axis bindings — the growable specificity.**
Optional refinements drawn from a declared axis registry. Adding a new `object` value does not
mint a new intent class; the classifier learns the *axis*, with every example of that axis
contributing, instead of 24 classes with ten examples apiece.

**3. The limit Austin asked for, made self-enforcing.**

> **An axis may exist only if the system can populate it.**

A refinement axis requires a classifier, extractor or slot that actually produces it. No
aspirational distinctions. This is the "intentional design decision around naming conventions",
and it is a rule that cannot rot, because a taxonomy entry with no analyser fails loudly
instead of quietly accumulating ten-example classes.

Corollary for growth: a refinement value below a minimum example count is a **candidate**, not a
label. It sits in a promotion queue until it earns its place — which is what
`IntentAutoPromoter` was already designed to do (criteria: 3+ similar candidates, same domain).

## Why this answers the compression framing

The symmetry becomes exact:

```
  intent  = compressed concept + axis bindings      (what was meant)
  response = primitive type:variant + content       (what to say)
  decompressor : (concept, bindings) -> many surface forms
```

`smalltalk.greeting {register: flirtatious}` compresses a concept that has many reliable
expressions, exactly as `acknowledgment:social {speech_act_sub_type: greeting}` already expands
to "Hello!" / "Hi there!" / "Hey!". Interpretation and generation become the same operation run
in opposite directions over one vocabulary.

It also dissolves task 031 as a side effect: the lattice realizer fails today because
`PrimitiveTypes` uses `type:variant` while `primitives.json` uses a pluralised flat vocabulary.
If both sides share one factored vocabulary, the namespace split cannot recur.

## What it costs — honest accounting

This is a large change and I have not tested the central claim.

- **Unverified:** that a factored classifier scores better than 141 flat classes. The evidence
  (the model's own collapse behaviour, examples-per-class counts) is strong but circumstantial.
  **This should be an experiment before a migration** — train an `operation` axis classifier and
  an `object` axis classifier on the existing corpus and compare against `intent_full`. That is
  a day's work and it settles the question with a number.
- **Every consumer assumes flat strings:** `templates.json` keys, `intent_registry.json` keys,
  slot schemas keyed by intent, `ResponseSystemRouter`'s domain-prefix dispatch,
  `Dispatcher.supported_intents`, `registered_intent?/1`. A factored intent can *render* to a
  canonical string for compatibility, but that is a migration with a long tail.
- **The axis registry becomes the thing that must not drift** — it inherits the whole problem of
  task 077, just in a better-shaped container. It needs the same enforcement.

## Recommended next step

Do not migrate yet. Run the experiment:

1. Derive `operation` and `object` axis labels for the 5,274-utterance corpus by parsing the
   existing dotted labels (they encode the axes positionally; that is why they are parseable).
2. Train two axis classifiers; compare accuracy against the 141-class `intent_full` baseline.
3. If the axes win, the factored scheme is justified by measurement rather than by argument, and
   the migration has a number behind it.

The corpus for this already exists and is clean where it matters: Dialogflow and the annotated
corpus preserve every fine-grained label and agree with each other (task 079).

## Open questions for Austin

1. **How many base concepts feels right?** The model found 8 for smarthome; the full space is
   probably 30-40. Fewer means more axis work; more means drifting back toward flat labels.
2. **Which axes are first-class?** `object`, `property`, `operation`, `modality`, `register`,
   `horizon` all appear in the current labels. `register` (the flirt example) has no classifier
   today — so by the rule above it would start as a candidate axis, not a real one.
3. **Does `smalltalk.compliment` keep its subtypes?** Under factoring the question dissolves:
   `smalltalk.compliment {about: appearance | intellect | humour}`. The subtype survives as an
   axis binding instead of four separate classes.

## Related

079 (the collapse this reinterprets), 077 (canonical vocabularies), 031 (the primitive namespace
split this would dissolve), 038 (`derived_label` phantoms), 072, 078 (the workbench, whose
decision unit this changes again).
