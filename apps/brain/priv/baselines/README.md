# Axis baselines — the pre-rebuild reference point

**Frozen 2026-09-12 at `0d86a526536ab174bb75afde3b7e2cff75c17a68` (`0d86a52`), tracked tree clean.**

These files capture the measured state of the 18 chunk axes **before** the POS training
rebuild. They exist because the task 081 baseline currently reproduces **byte-for-byte**, and
retraining POS destroys that property permanently. Once the rebuild lands, this directory is
the only remaining record of what the system did beforehand.

Do not regenerate these files. If a rebuild needs a new reference point, add a new dated set
alongside them.

## Provenance

| | |
|---|---|
| git sha | `0d86a526536ab174bb75afde3b7e2cff75c17a68` |
| tracked tree | clean (`git status --porcelain -uno` empty) |
| extractor schema fingerprint | `fb3ce7f1e4739cb3` |
| Elixir / OTP | Elixir 1.19.5, Erlang/OTP 28 (erts-16.2.2) |
| WordNet | 3.1 (Kafe Prolog port, Apache-2.0), 23 files, `apps/brain/priv/wordnet/` |
| `pos_model.term` | `5b81e107856af28ba5b3f70d8e31be2d8860e79880df57b1d8a26574c81c7a25` |
| `micro/intent_domain.term` | `0ad5ea9440644be8363baef0a8743a7aa737cc762ec46cf159d99a2f8a25b703` |
| `micro/tense_class.term` | `f72b19d02eac7212befc528c80a59de7875b1372ac099ce433ab05b4c6b9460d` |
| `micro/aspect_class.term` | `5fc7745a2037ec068c4a70ca3cae2e9ae036b1d0c9c90486b73ab26c71c09958` |
| `micro/urgency.term` | `dff2cceaa09d3b490d4f6559cfc2e490994bc0d3811b3b338d90e7c7ef41acc2` |
| `micro/certainty_level.term` | `0b8b4f974f30560bfc58bb54469b3102915fe0bdfe4d21b9f28c2c98785524ec` |

The five `micro/*.term` files above back the only five classifier-backed axes (`domain`,
`tense`, `aspect`, `urgency`, `certainty`). They are the ones expected to move unpredictably
when the feature vector changes underneath them.

Full hashes for all 26 `.term` files:

```sh
find apps/brain/priv/ml_models -name "*.term" -type f -print0 | xargs -0 shasum -a 256 | sort -k2
```

## Files

| file | what it is | reproduce with |
|---|---|---|
| `axis_sample.2026-09-12.baseline.json` | The task 081 sample: 238 utterances x 18 axes, with per-axis provenance and the MI table. **sha256 `174b3b443cb55d4ae40bda3ac0c526af3db28f31ce67579bde6661e7dbc27be1`** | `mix run .claude/corpus/axis_sample.exs` |
| `provenance_report.2026-09-12.baseline.txt` | Computed/defaulted census over 16 hand-written sentences, plus value-domain conformance and defaulted-parent counts | `mix run .claude/corpus/provenance_report.exs` |
| `distortions_probe.2026-09-12.baseline.txt` | All 18 axes over 450 cognitive-distortion sentences (97% first-person, 45% negated) | `mix run apps/brain/priv/baselines/distortions_probe.exs` |
| `pos_tag_distribution.2026-09-12.baseline.txt` | What the POS tagger emits over 3,952 tokens, beside the vocabulary the model can emit | `mix run apps/brain/priv/baselines/pos_tag_distribution.exs` |

`mix run apps/brain/priv/baselines/polarity_evidence.exs` reads the polarity instrumentation directly;
its result is recorded below rather than as a separate file.

## What the baseline says

The numbers the rebuild has to move:

- **POS tagger emits 5 distinct tags over 3,952 tokens — 91.4% NOUN, 0 VERB, 0 AUX, 0 PART.**
  The model's vocabulary holds 13 tags including VERB and PART; the decoder never selects them.
- **`polarity` 0/450 computed**, and over the 204 negation-bearing sentences the instrumentation
  reports `atom_part: 0, string_part: 0` — neither spelling of the tag appears, so the
  atom/string type bug is not the blocker and task 074 is.
- **`target` never reaches `:self`** and **`self_disclosure_level` is 430/430 `:none`** across
  430 first-person sentences. `target` returns `:agent` 158 times on first-person self-talk,
  which is wrong rather than merely defaulted.
- **`urgency` is 450/450 computed and 93% `:critical`** on calm ruminative self-talk. An axis
  can be fully computed and still be wrong: `computed` is necessary, not sufficient.
- `novelty_score` 0/450 computed; `sentiment_alignment` 0/450; `modality` 3/450.

## Where the generators live

Four of the scripts sit beside these files in `apps/brain/priv/baselines/`. Two do not:
`axis_sample.exs` and `provenance_report.exs` predate this directory and remain in
`.claude/corpus/`, which is **untracked**. So `axis_sample.2026-09-12.baseline.json` — the
largest and most important artifact here — is committed while the script that regenerates it
is not. Anyone reproducing it from a fresh clone needs that script recovered first.

`distortions_probe.exs`, `pos_tag_distribution.exs` and `polarity_evidence.exs` additionally
read `.cursor/research/cognitive_distortions_dataset.csv`, which is also untracked. They raise
with the expected path rather than degrading if it is absent.

## Comparison rule

A later run is comparable value-for-value **only** under the same `schema_fingerprint`. If the
fingerprint differs, the feature vector's shape changed and only per-axis distributions are
comparable, not individual rows.
