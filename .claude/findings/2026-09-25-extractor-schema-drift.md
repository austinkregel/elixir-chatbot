# The feature vector's schema drifted since 2026-09-12, invisibly

- **Found:** 2026-09-25, on the first real use of `ChunkFeatures.schema_fingerprint/0`
- **Branch:** `feat/lattice-home-assistant-and-memory`
- **Confidence:** PROVEN for the drift and its mechanism; the specific changed value is
  **unrecoverable**, and that is itself the point.

## What was measured

`Brain.Analysis.RunProvenance.capture!/0`, run in dev:

```
extractor   : fp=bc289842ba5ccb3d dim=343 groups=25
age_graph   : 15 parent types, digest b4d47953e17cf3e9
```

`.claude/corpus/axis_sample.json`, `generated: 2026-09-12`, records:

```
schema_fingerprint = fb3ce7f1e4739cb3
```

**The fingerprint moved. The width did not** — 343 both times, 25 groups both times.

## Why this is the task 072 mechanism, one level up

072 is about feature *values* changing meaning while the vector length stayed 343, so no
dimension-mismatch error ever fired. This is the same failure at the *schema* level: the
dimensions were renamed, every stored vector's index 328 now means something else, and every
length-based check still passes.

Any comparison of a vector taken today against one taken on or before 2026-09-12 — including
task 081's mutual-information table and the 238-row `axis_sample.json` baseline — is a comparison
across two different schemas.

## The cause is runtime state, not code

Neither extractor file has changed since the baseline was recorded:

```
apps/brain/lib/brain/analysis/feature_extractor/chunk_features.ex
  724415f 2026-09-12 name all 343 feature dimensions and derive the width from that list
apps/brain/lib/brain/analysis/feature_extractor/enrichment_features.ex
  6dbaaee6 2026-09-12 name every dimension EnrichmentFeatures emits, beside the code that emits it
```

Nothing after 2026-09-12. Since `schema_fingerprint/0` is a pure function of
`dimension_manifest/0`, and 24 of the 25 groups derive their names from module attributes, the
only way the digest can move with the code frozen is through the one group that reads runtime
state.

That group is 23, `entity_type_semantics`. Measured today:

```
parent_types (15): ["action", "artist", "clothing", "commerce", "communication", "date_time",
                    "device", "information", "location", "measurement", "media", "music_meta",
                    "number", "person", "weather"]

group 23 names (17): [:etype_parent_action, ..., :etype_parent_weather,
                      :etype_coherence, :etype_coverage]
```

`EnrichmentFeatures.entity_type_semantics_names/0` emits `:"etype_parent_#{type}"` straight from
`TypeHierarchy.parent_types/0`, which reads an ETS table populated **from the AGE graph**. A
parent type renamed, added, or removed changes the dimension names. At 15 types the width is
`15 + 2 = 17` either way, so a swap of one type for another moves the fingerprint and leaves the
width untouched — which is exactly the observed signature.

## What cannot be determined, and why that matters

**Which type changed.** The 2026-09-12 baseline recorded the fingerprint but not the parent-type
list, so there is nothing to diff against. The drift is detectable but not attributable.

That gap is the argument for `RunProvenance`'s `age_graph` field. A fingerprint alone says only
*that* something moved; `parent_type_count` plus `parent_types_digest`, recorded per run, is what
makes the next occurrence attributable. `group_widths/0` does the same job for the other 24
groups.

## Consequences

1. **Task 081's baseline is not comparable to any measurement taken today** — add this to the
   four stale rows already recorded in that task's 2026-09-25 addendum.
2. **The AGE graph needs versioning, not just recording.** It controls the vector's values
   (072) and its dimension names (this finding). `RunProvenance` records it; nothing yet
   prevents it drifting between a model's training and its use. That is what the Stage B load
   gate is for.
3. **Re-baseline before comparing anything.** The first tagged `mix axes.snapshot` run supersedes
   `axis_sample.json` as the reference point.

## Related

072 (same mechanism, value level), 081 (the baseline this invalidates), 082 Phase 1 (the
provenance this justifies), 049 (the Poincaré/type-hierarchy page, where a parent-type change
would be visible to a human).
