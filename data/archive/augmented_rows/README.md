# Augmented rows removed from the intent corpus

591 synthetic rows removed on 2026-09-29 by `mix corpus.prune --provenance augmented --save`:
268 from `data/intents/*_usersays_en.json`, 268 from `data/training/intents/*.json` (the same
utterances, in their POS-annotated form), and 55 from
`apps/brain/priv/evaluation/intent/held_out.json`.

## What these rows are

They are not Dialogflow utterances. `scripts/materialize_orphan_intents.exs` writes texts
that have no source backing *back into* the export, tagging them `orphan-<hash>` or
`augmented-<hash>`. The augmented ones are mutations of real utterances — words dropped
and prefixes added, per `augment_training_data.ex`'s `@droppable_pos` and `@add_prefixes` —
and the mutation frequently destroys the grammar:

    is bright it enough everywhere
    I'd like to it's dark in here
    my laptop screen is way too

`rebuild_gold_standard` had been stamping every corpus row `labeled_by: "dialogflow"`,
including these, so accuracy measured on the corpus was reported as accuracy on user input.
That claim was corrected first; this removal followed from it.

## Why they were removed

55 of the 1,000 held-out rows were mutations of this kind, so 5.5% of every reported intent
figure was the model being scored on text no user would type. The training half is the same
material.

Only the augmented rows were removed. The 596 `materialized-` rows remain: they read as
plausible phrasings, and removing all 864 synthetic rows would destroy 33 of 194 intents
entirely — `alarm.set`, `timer.set`, `calendar.schedule`, `communication.call`,
`reminder.create`, `todo.add`, the `code.*` family and more. Whether their labels are sound
is a separate, unanswered question.

## Layout

    manifest.json                  provenance, totals, per-file counts
    <intent>_usersays_en.json      the rows removed from that export file (68 files, 268 rows)
    held_out.json                  the rows removed from the held-out split (55 rows)
    training_intents/<intent>.json the POS-annotated form of the same rows (68 files, 268 rows)

Each row is kept verbatim, as the exact source text it had in the file it came from, not as
a re-encoding of it. The export mixes two escaping conventions and two key orders, so
re-encoding it is not byte-faithful; restoring a row means putting its text back.

## Why this directory is tracked

`/data/*` is gitignored with `!/data/archive/` as an exception, so these rows are under
version control although the corpus they came from is not. Before this archive existed the
removal would have been unrecoverable.

## Restoring them

Insert a row's text back into the array in the matching `data/intents/` file, then run
`mix rebuild_gold_standard --save`. The corpus is derived from the export alone, so the row
returns with its `labeled_by: "augmented"` intact. Put the `training_intents/` copy back in the
matching `data/training/intents/` file to restore its tokens and POS tags — they are the original
bytes, so no re-tagging is needed.

A restored row does **not** return to the held-out split: that file is not regenerated from the
corpus, and re-carving it would draw a different split.
