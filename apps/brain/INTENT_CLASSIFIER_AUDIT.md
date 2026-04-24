# Intent Classifier Audit: `Can you help me understand neural networks?`

This is the Workstream E deliverable from the multi-chunk understanding plan.
It explains, with cited evidence from training data and the intent registry,
why the trained intent classifier predicts `meta.privacy_probe` for the third
chunk of:

> "My name is Austin Kregel. I live in Owosso Michigan. Can you help me
> understand neural networks?"

It also documents a secondary defect (entity drop) and proposes a candidate
seed-data fix. **No production code change is included in this workstream.**

## 1. Why `meta.privacy_probe` wins

The classifier is data-driven: TF-IDF / LSTM trained on
`apps/brain/priv/evaluation/intent/gold_standard.json`. The `meta.privacy_probe`
class has 15 examples; every one is a "can you ... me/my ..." or
"are you ... me ..." question framed against the agent itself, e.g.:

- `"can you delete what you know about me"`
- `"how do I delete my data"`
- `"are you logging our conversation"`
- `"is my data being stored"`
- `"are you tracking me"`
- `"are you spying on me"`
- `"are you watching me"`

Lexical overlap with `Can you help me understand neural networks?`:

- `can you` (exact bigram)
- `me` (token)
- question-form (interrogative AUX-PRON-VERB pattern)

The combined `can you ... me ...` shape is **the** identifying signature for
`meta.privacy_probe` in this dataset, so an unrelated `can you ... me ...`
question lands in that cluster.

## 2. Why no other intent rescues it

Looked at every "knowledge / learning / help" intent in
`gold_standard.json`:

| Intent | Count | Sample texts | Verdict |
|---|---|---|---|
| `knowledge.define` | 15 | `Define X`, `What does X mean`, `Definition of X`, `What is the meaning of X` | All imperative or "what is/does"; no "can you ... understand" pattern. |
| `knowledge.translate` | 15 | Translation prompts | Off-domain. |
| `knowledge.capital` | 15 | Capital city facts | Off-domain. |
| `meta.self_knowledge` | 20 | "what do you know about me", "what's my name" | About the user, not topics. |
| `meta.memory_check` | 15 | "do you remember me" | Off-domain. |
| `code.explain` | 40 | Code-specific explanation requests | Off-domain. |
| `smalltalk.agent.can_you_help` | 41 | `can you help me`, `please help me`, `assist me`, `I need your help`, `can you help me out` | Closest in surface form, but **no domain object** in any example -- it is registered as `category: expressive`, `speech_act: general`, and so is filtered out as non-substantive in `Brain.try_nlp_with_analysis/5`. |

There is **no class** in this dataset whose training examples mean
"the user wants to learn about a topic the agent might know about."
The phrase "Can you help me understand X" is a real intent gap.

## 3. Secondary defect: entity drop on `meta.privacy_probe`

From `apps/brain/priv/analysis/intent_registry.json`:

```json
"meta.privacy_probe": {
  "category": "directive",
  "domain": "meta",
  "entity_mappings": {},
  "speech_act": "request_information",
  ...
}
```

`entity_mappings: {}` is what `Brain.Analysis.Pipeline.filter_entities_by_intent/2`
sees. The function logic:

```
schema = SlotDetector.get_schema(intent)
entity_mappings = Map.get(schema, "entity_mappings", %{})
valid_types = entity_mappings |> Map.values() |> List.flatten() |> MapSet.new()
if MapSet.size(valid_types) == 0, do: []
```

So any chunk classified as `meta.privacy_probe` has its extracted entities
**filtered to `[]` at the per-chunk stage**, before any cross-chunk merge.
Workstream A's union saves entities from *other* chunks (name, location), but
"neural networks" lives in the same chunk as the misclassification, so it is
the casualty.

`knowledge.define` has the same problem -- `entity_mappings: {}` -- so even
the "least bad" knowledge intent in the current registry would drop the
extracted topic entity.

## 4. Recommended seed-data fix (candidate, separate PR)

Two changes in two files, both data-only:

### 4a. Add a new intent class for "explain a topic" requests

In `apps/brain/priv/evaluation/intent/gold_standard.json`, add ~15-25 examples
under a new label `knowledge.explain_topic` (or reuse `knowledge.define` if
the team prefers consolidation -- but `define` is currently lexicon-narrow).
Suggested examples:

- `Can you help me understand neural networks?`
- `Help me understand recursion`
- `Explain quantum entanglement to me`
- `Can you explain how a transformer works?`
- `I want to understand how kidneys work`
- `Teach me about TCP`
- `Walk me through how compilers work`
- `Could you explain photosynthesis?`
- `What can you tell me about Bayesian inference?`
- `Help me learn about Elixir GenServers`
- `Can you help me understand event sourcing?`
- `Explain the difference between TCP and UDP`
- `I'd like to learn about string theory`
- `Tell me about how DNS resolution works`
- `Can you describe how an LSTM works?`

This volume (~15) matches the existing knowledge.* class sizes (15 each).

### 4b. Register the new intent and give it a topic entity mapping

In `apps/brain/priv/analysis/intent_registry.json`, add:

```json
"knowledge.explain_topic": {
  "category": "directive",
  "clarification_templates": {
    "topic": "What topic would you like me to explain?"
  },
  "defaults": {},
  "description": "Request to explain or teach about a topic",
  "domain": "knowledge",
  "entity_mappings": {
    "topic": ["topic", "concept", "subject", "noun_phrase"]
  },
  "input_contexts": [],
  "optional": [],
  "output_contexts": [],
  "query_type": "explain_topic",
  "required": ["topic"],
  "speech_act": "request_information"
}
```

The `entity_mappings` matters: it is what makes `filter_entities_by_intent/2`
keep the extracted topic entity instead of dropping it.

While there, also consider adding a non-empty `entity_mappings` to
`knowledge.define` (term: `["term", "word", "noun_phrase"]`) so existing
"Define X" classifications keep their entity. That is independent of the
neural-networks bug.

### 4c. Retrain and verify

After 4a + 4b, retrain via the existing TF-IDF / LSTM pipeline (no new code
required) and re-evaluate the three sample chunks:

1. `My name is Austin Kregel.` -- should remain in the introduction class.
2. `I live in Owosso Michigan.` -- should remain in the location-statement class.
3. `Can you help me understand neural networks?` -- should now classify as
   `knowledge.explain_topic` with `topic: "neural networks"` extracted and
   preserved through `filter_entities_by_intent/2`.

Capture the top-k for chunk 3 before and after, and attach to the seed-data PR.

## 5. Interaction with the just-shipped multi-chunk workstreams

Even *without* this seed-data fix, the multi-chunk plan (Workstreams A-D)
already mitigates the user-visible problem:

- Workstream A unions entities across chunks, so name + location survive even
  when chunk 3 misclassifies and drops its own entities.
- Workstream B prefers the question chunk as `primary` regardless of
  classifier confidence, so the response will be aimed at "neural networks?"
  rather than at the misclassified meta intent.
- Workstreams C-D ensure Ouro sees all three chunks and that fact lookup is
  routed to the question chunk.

The seed-data fix in Section 4 is the proper long-term fix; the multi-chunk
plan is the runtime safety net while training data catches up.

## 6. Out of scope for this audit

- `SelfKnowledgeAnalyzer` heuristics. It returns `meta.self_query`, not
  `meta.privacy_probe`, and is not the source of the misclassification.
- Any regex/keyword bandaid in `Pipeline` or selection logic. Per
  `.cursorrules`, classifier behavior is fixed by training data, not by
  hardcoded rules.
