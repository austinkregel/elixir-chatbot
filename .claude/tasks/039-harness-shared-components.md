# Build the verification harness shared components

- **ID:** 039
- **Status:** ready — do before any page
- **Area:** verification-pages
- **Branch:** `feat/lattice-home-assistant-and-memory`
- **Recorded:** 2026-09-11

## Why this matters

Austin's requirement: every feature gets its own page so a human can exercise it in
isolation and verify it works, plus pages for testing interactions. His explicit constraint:
*"that does not mean everything has to be bespoke. We are able to create components as
needed to help simplify repeated functionality on each page."*

Build these once, before page 1, or 18 pages become 18 bespoke pages.

## What already exists — reuse, do not reinvent

The repo already has the pattern twice:

- `/code` — paste code, pick language, `analyze_code` -> `Brain.Code.Pipeline.process`,
  render result (`code_analysis_live.ex`).
- `/training-studio` Trace tab — text -> `Diagnostics.trace_prediction/2`, render
  (`training_studio_live.ex`).

And the chrome is all there:

- `ChatWeb.AppShell.app_shell/1` — sidebar, world selector, status footer, flash. Adopted by
  8 of 9 pages. Add a nav item to expose a new page.
- `ChatWeb.WorldContext` on_mount — supplies `@current_world_id`, `@available_worlds`,
  `@system_ready`, `@current_path`, and owns `switch_world` / `refresh_worlds`. Every page
  pattern-matches `{:world_context_changed, world_id}` to reload.
- `ChatWeb.UI` — 16 components (`card`, `badge`, `btn`, `tabs`, `stat_kpi`, `toggle`,
  `alert`, `status_dot`, `text_input`, `section_header`, ...) plus `ChatWeb.CoreComponents`
  (`table`, `input`, `header`, `icon`). Both auto-imported by `use ChatWeb, :live_view`.
- `Brain.ML.TrainingServer.start_training/2` + the `training:progress` PubSub topic — the
  sanctioned way to kick a long job and stream status.

## To build

1. **`ChatWeb.Harness.Runner`** — a function component wrapping the proven shape: input form
   -> subsystem call -> rendered output, with timing and the raw term. Takes the subsystem
   module/function and a result renderer.
2. **`ChatWeb.Harness.CaseStore`** — save an input with its expected output, re-run, show
   pass/fail. Austin chose **persist expected-vs-actual** over transient inspection.
   Storage: a new Atlas schema `atlas_verification_cases` — `subsystem`, `input` (jsonb),
   `expected` (jsonb), `last_actual` (jsonb), `status`, `world_id`, timestamps. Atlas is
   already the persistence app with ~19 `atlas_*` tables, so this is consistent rather than
   a parallel store.
3. **`ChatWeb.Harness.Diff`** — expected-vs-actual rendering that works for maps, lists and
   floats with a tolerance.
4. **`/verify` index page** — lists all pages with live pass/fail counts per subsystem.

## Design note

The saved cases are **not** a gold standard. Gold standards are bulk corpora for
training/evaluation (tasks 013-015); these are curated human-verified scenarios, most of
which have no gold-standard analogue. For the one place they overlap (the response page),
add an explicit export into `priv/evaluation/response/gold_standard.json` rather than
maintaining two copies.


## Added requirement: surface decision provenance

Austin, during the token-novelty investigation:

> "*That* is why I want semi-isolated patches to test each of these feature types in isolation
> and hand-in-hand. It should help surface things like hardcoded rules like what you found for
> the OOV path."

So a page must not only show *what* a subsystem produced, but **which rule produced it**. The
investigation that prompted this found, in one sitting:

- `default_propn_type: "person"` — every unknown proper noun is typed `person` by a static
  config default, regardless of sentence frame (task 075)
- the `PROPN` gate in `extract_proper_noun_hints/2` that silently never opens because the POS
  tagger tags OOV tokens `NOUN` (task 074)
- `@memory_context_default [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]` standing in for real memory context
- three different silent fallbacks for entity familiarity with three different biases
  (0.0, 0.5, 0.5) — task 073

None of these are visible in output alone; all are visible if the page reports its provenance.

**Concretely, `Harness.Runner` should render, alongside the result:**

1. **Which branch/rule fired** — named `defp`, config key, or model, and its value.
   `Brain.Analysis.ProcessingTrace` already exists and may be the right vehicle.
2. **Which inputs were defaults rather than real data** — a value that came from
   `@memory_context_default` or a `rescue` must be visually distinct from a computed one.
   This is the single highest-value display decision on the page.
3. **Config values in play**, read live, not documented — e.g. `default_propn_type`,
   thresholds, `@valid_types`.
4. **Where a value crossed a module boundary**, and in what shape — this is what would have
   caught task 010's `{:ok, list}` treated as a list.

Design consequence: subsystems need to *return* provenance, not just values. Rather than
changing every signature, prefer a per-request trace collector the harness can read
(`ProcessingTrace` is the existing candidate — note it has 6 silent rescues, task 024).

**Acceptance criterion to add:** a value that came from a fallback default is visually
distinguishable from a computed value on every page.

## Acceptance criteria

- [ ] `atlas_verification_cases` migration written and applied
- [ ] A case can be saved, re-run, and shown pass/fail
- [ ] `/verify` lists subsystems with counts
- [ ] Page 040 built on these components with no bespoke chrome
- [ ] No new page duplicates `AppShell` or `WorldContext` behaviour

## Open question for Austin — ANSWERED 2026-10-06: render it

Failure display convention — this interacts with task 026. If a subsystem call raises, should
the harness page crash (loud, consistent with no-fallbacks) or render the exception in the
result panel (visible, page survives)? For a verification tool specifically, I lean toward
rendering it: seeing the stacktrace *is* the verification result.

**Resolved by the evidence rather than by preference.** A crashed LiveView shows the person
*nothing*: the stacktrace goes to the server log, the page disconnects, and the human who came to
verify a subsystem learns less than if the failure were on screen. "Crash = loud" is false here —
crashing *hides* the verification result, which is the opposite of what the no-fallbacks rule is
for. The rule forbids swallowing a failure and continuing as though it did not happen; this does
the reverse.

Three things keep it from becoming the pattern task 026 is deciding about:

- the catch wraps exactly one expression, the subsystem call — not a render, not an event handler;
- a raised outcome carries **no `:value`**, so nothing downstream can mistake a failure for a result;
- the call is resolved to a callable *before* the `try`, so a page passing a typo'd function name
  raises at the caller instead of being reported as the subsystem failing.

`Atlas.Verification.record_error/3` stores it under its own `"error"` status, distinct from
`"fail"`, because a subsystem that raised produced no answer while a failing one produced a wrong
answer.

## Built — 2026-10-06

| file | what it is |
|---|---|
| `apps/atlas/priv/repo/migrations/20261006000001_create_verification_cases.exs` | applied |
| `Atlas.Verification.Subsystems` | the 18 subsystems, each naming the task that builds its page |
| `Atlas.Verification.Comparison` | the verdict, and jsonb normalisation |
| `Atlas.Schemas.VerificationCase` | schema and invariants |
| `Atlas.Verification` | context; computes pass/fail itself rather than trusting a caller |
| `Brain.Provenance` | the per-request trace collector this task asks for |
| `ChatWeb.Harness.Diff` | renders a comparison; marks stand-in values |
| `ChatWeb.Harness.Runner` | `run/2` plus the result and case panels |
| `ChatWeb.VerifyLive` + route + nav | `/verify` |

Two departures from the sketch above, both deliberate. **The verdict moved out of `Harness.Diff`
into `Atlas.Verification.Comparison`** — leaving it in the web app would mean `record_result/3`
trusting a caller's pass/fail and a mix task re-running cases being unable to reach the same answer.
**`Runner` stayed functions and function components rather than becoming a LiveComponent** — a
LiveComponent would absorb each page's event handlers and remove more duplication, at the cost of
making every page's behaviour invisible at the page.

### The provenance criterion is wired, not stubbed

`run/2` turns `Brain.Provenance` on around the subsystem call, so a page gets provenance without
gathering anything and without any subsystem signature changing. Five origins, closed:
`:computed`, `:declared`, `:default`, `:absent`, `:unavailable`. The last is its own origin because
a source that could not be read means something is *broken*, which is not the same as a value being
unset.

Instrumented so far: `TypeHierarchy.config/2` (the chokepoint for every config read in the brain —
one change covers them all) and `ChunkFeatures.entity_features/1`'s entity familiarity, one of task
073's three silent fallbacks. `ChunkFeatures.schema_fingerprint/0` is unchanged at
`ab0903fe86037864`, so the feature vector and every model trained on it are untouched.

## A defect the harness found on its first live run (PROVEN 2026-10-06)

**`domain_lemmas` is read 207 times per utterance and returns `%{}` every time.**

`pipeline.ex:1591` reads it as `TypeHierarchy.config("domain_lemmas", %{})`, which looks inside
`entity_types.json`'s **`config`** section. `domain_lemmas` is a **top-level** key of that file,
with 13 entries (`smarthome` → `["home", "automation", "device"]`, and so on). It is not in
`config`, so every read falls through to the caller's `%{}`.

Measured on one analysis of *"turn on the kitchen lights"*: 277 provenance entries over 11 distinct
facts, of which **207 reads were this single key**, all `:default`, all `%{}`. The other 67 config
reads resolved correctly against 8 declared keys.

This is exactly the class of defect the provenance criterion was added for — invisible in output,
obvious the moment a value says where it came from — and it surfaced without anyone looking for it.

**Not fixed.** Pointing the read at the right level changes what the pipeline does with domain
lemmas on every utterance, which is a behaviour change needing a decision and a measurement, not a
one-line edit folded into a harness commit.

### Repro

```sh
mix run -e '
alias Brain.Provenance
{_r, entries} = Provenance.collect(fn -> Brain.Analysis.Pipeline.analyze_chunk("turn on the kitchen lights") end)
entries |> Enum.group_by(& &1.origin) |> Enum.each(fn {origin, es} ->
  IO.puts("#{origin}: #{length(es)} reads over #{es |> Enum.map(& &1.path) |> Enum.uniq() |> length()} paths")
end)
'
```

## Remaining

- **Page 040 built on these components** is the last acceptance criterion, and it is task 040.
  "A case can be saved, re-run, and shown pass/fail" is built and tested at the store level with no
  UI to exercise it until that page lands.
- More subsystems reporting provenance. Two sites are instrumented; the rest report nothing, and
  the panel says so for a path with no entries rather than rendering empty.
