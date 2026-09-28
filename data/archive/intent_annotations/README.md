# Archived intent annotations

38 files, 651 rows, moved out of `data/training/intents/` on 2026-09-28.

Everything else in `data/training/intents/` is **derived output**: it is generated
from `data/intents/*_usersays_en.json` by `scripts/generate_pos_annotations.py`,
which builds each filename by taking the Dialogflow intent name and applying
`name.replace(' - ', '.').replace(' ', '.')`, then adds `tokens` and `pos_tags`
with NLTK. Re-running that script reproduces those files.

**These 38 cannot be reproduced.** No file in `data/intents/` mangles to any of
their names any more, so the source they were generated from is gone. `data/` is
ignored by git (`.gitignore`, `/data/*`), so they had no history either — deleting
them would have been final. That is why they are here, and why `!/data/archive/`
was added to `.gitignore`: without the exception, moving them aside would have left
them exactly as losable as they were.

Selection was by the canonical join — a file qualifies only when no current
`data/intents/` name mangles to it. Selecting by matching `schedule` in the name
would have caught `calendar.schedule`, a live registry intent.

## device_scheduling/ — 17 files, 344 rows

The device-scheduling taxonomy, removed from the chat pipeline deliberately:
asking an assistant to manage a schedule belongs in the application, not in the
conversation. The intents were dropped from the corpus and their response
templates retired, but these annotated utterances are the only copy of the
phrasings — 331 of the texts appear nowhere else in the repo.

Kept for the app-level scheduling feature.

## context_continuations/ — 8 files, 82 rows

Follow-up utterances that only make sense after a previous turn: *"living room
too"*, *"what about bathroom"*, *"at 7 pm"*, *"do it everyday"*. Dialogflow modelled
these as context variants of a base intent; the corpus rebuild folds a
` - context: …` suffix into its base intent, so they have no standalone label.

62 of their texts appear nowhere else. Kept because context handling is unsolved,
not because these labels should come back — three corpus intents
(`smarthome.device.switch`, `smarthome.device.volume`, `smarthome.heating`) are
continuations of the same kind and are served by the Synthesizer rather than by a
template.

## superseded_spellings/ — 13 files, 225 rows

Older spellings that the canonical export has since renamed. Every text in this
group already exists in the corpus under the current label, so nothing here is
unique; they are kept only so the rename is legible later.

| archived | current |
|---|---|
| `music_player_control.*` (10) | `music.player.*` |
| `weather` | `weather.query` — the corpus assigns all 89 of its texts there |
| `Default.Welcome.Intent` | `smalltalk.greetings.hello` |
| `message` | removed; it was a bare Dialogflow display name |

## Provenance of the rows

Row `id` values carry their origin, and the prefixes are not interchangeable:

- a UUID — a genuine Dialogflow usersays entry
- `orphan-<hash>` — a gold-standard text with no source backing, written back into
  `data/intents/` by `scripts/materialize_orphan_intents.exs`
- `augmented-<hash>` — synthetic text derived by mutating a real utterance, and
  frequently ungrammatical (*"want I to open an account"*)

`pos_tags` were produced by NLTK over whatever text the row carries, so tags on
synthetic rows describe synthetic word order. Treat them accordingly.
