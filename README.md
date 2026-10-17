## :bangbang: Disclaimer :bangbang:
This application was vibe coded with Claude and Cursor over the course of several months. I did try rather hard to ensure that the built systems do what they describe, but the _whole_ time was a constant battle with every model trying to take shortcuts like string matching, regex, or hardcoding intents, or switching to python while writing some of the pipelines.

While this is my pet-project, I cannot guarantee that it works as intended or that it is free of bugs. It has significantly ballooned in scope since I started it, and I have left it unattended sometimes while implementing different parts. No one has permission to use this in a production environment, and no I wouldn't recommend it.

## Retroduct (Elixir/Phoenix)

The project is **Retroduct** — Peirce's word for abduction, reasoning backward from what is observed to the explanation that best accounts for it. The code is not yet renamed; some paths and module docs still say `chat_bot_llm` or ChatBot.

This is a Phoenix LiveView application built around a **pipeline of specialized classical and local-neural analyzers** rather than a single hosted LLM. The analysis layer is TF-IDF and dense-centroid micro-classifiers (17 of them, feature-vector weights evolved by a genetic algorithm against a composite 50/50 raw+balanced-accuracy fitness). Surface text in the response layer is produced per-domain by one of a phrase-lattice beam-search realizer, the local **Ouro LoopLM 1.4B** (4 recurrence passes), or a template/domain-knowledge synthesizer — routed by `priv/response/system_config.json`. Candidate readings stay alive with their margins in a `Brain.Lattice`; every value's origin is recorded in `Brain.Provenance`'s closed vocabulary; a 9-dimension `ResponseEvaluator` scores each pass and drives up-to-3-iteration refinement. Memory is world-scoped episodic/semantic over Atlas; the epistemic layer is a per-world JTMS of beliefs with authority-aware confidence decay.

For a thorough navigation map of the brain module — file paths, decision thresholds, hard-fail invariants, world-scoping, and the gotchas that bite — see `CLAUDE.md`, which loads at the start of every session.

### What you get

- **Web UI**: LiveView chat at `/chat` with a processing inspector, plus dashboards for system monitoring (`/dashboard`), ML accuracy metrics (`/accuracy`), training world exploration (`/explorer`), settings management (`/settings`), and admin review panels for intents and knowledge.
- **Analysis pipeline (two-pass per chunk)**: semantic chunking → pass 1 (discourse / speech act / sentiment in parallel Tasks, basic entity extraction, anaphora, POS tagging, event extraction, event linking, SRL, preliminary intent) → cross-chunk aggregation (`ChunkPriority`, `ContextAccumulator`) → pass 2 for substantive chunks (`:intent_full` micro-classifier, speech-act refinement, `ChunkProfile` materialization, lattice rerank, intent arbitration, entity retyping with profile, intent-aware span resolution, entity-type rescoring, contextual IS-A narrowing, slot detection, context resolution, fact verification, novelty detection).
- **17 micro-classifiers**: TF-IDF centroid (`SimpleClassifier`) and dense-centroid (`FeatureVectorClassifier`) — see `Brain.ML.MicroClassifiers` for the full list. Per-dimension weights for feature-vector models evolved by `Brain.ML.WeightOptimizer`, a GA with EXLA JIT, Fisher seeding, adaptive mutation, and a composite fitness that refuses to trade majority accuracy for minority recall.
- **Candidate lattice**: `Brain.Lattice` keeps N-best candidates with raw score, normalized confidence, margin, entropy, source and stage through reranks; empty lattices tombstone a reason (`:unknown_intent`, `:not_loaded`) rather than returning silently.
- **Provenance**: `Brain.Provenance` closed vocabulary (`:computed | :declared | :default | :absent | :unavailable`); every recorded value names its `:source`.
- **Response generation (4-stage iterative)**: `RefinementLoop` → `DiscoursePlanner` (data-driven backbone from `discourse_patterns.json`, with a 20+ reason→reply table for service errors) → `ContentSpecifier` → `SurfaceRealizer` (per-domain routed by `ResponseSystemRouter` to `:lattice | :ouro | :synthesizer | :template`) → `ResponseEvaluator` (9 dimensions, including belief grounding against JTMS and epistemic consistency; the realized text is analyzed with the full pipeline again to score it). Up to 3 iterations; the weakest dimension feeds back into the stage that owns it. Returns `nil` + `:silence_preferred` if even the best pass is below the silence threshold.
- **Surface realizers**: a phrase-lattice beam search over typed fragments with cosine scoring (`PhraseLatticeRealizer`); the Ouro LoopLM 1.4B with 4 recurrence passes (`OuroRealizer`); and the domain-knowledge synthesizer with primitive composition and sense-aware entity substitution (`Synthesizer`).
- **Memory**: world-scoped episodic/semantic store (`Brain.Memory.Store`) backed by Atlas as the primary store, VectorIndex as the similarity cache; KG-entity re-rank blends `tfidf * 0.7 + entity_cosine * 0.3`; 15-min consolidation clusters at cosine ≥ 0.8 into semantic facts that bridge to beliefs.
- **Epistemic layer**: `Brain.Epistemic.JTMS` (classic Forbus/de Kleer, per-world networks, label propagation, contradiction handler), `BeliefStore` with authority-aware confidence decay and auto-retraction below 0.1, `UserModelStore` (per web user, cross-world), `StanceTracker` (bidirectional opinion drift from Jiang et al. CHI 2026), `DisclosurePolicy`, `ConsolidationBridge`.
- **Owned lexicon with sense drift**: `Brain.Lexicon.UserDefined` holds sense facts in ETS, persisted in Atlas, EMA-blended centroids with weekly decay and archival; `SenseDrift` measures canonical-vs-observed cosine distance to flag words that may have shifted sense.
- **Knowledge expansion**: Academic research integration (arXiv, Semantic Scholar, OpenAlex), source reliability tracking, multi-source corroboration, and human review workflows.
- **Code analysis**: AST-based code parsing, symbol extraction, code gazetteer, and relationship mapping across multiple languages.
- **Training & data tooling**: `mix` tasks for training, evaluation (precision/recall/F1/confusion matrices), entity dataset generation, experiment tracking with A/B comparisons, and gold standard management.
- **Services**: External dispatch (`Brain.Services.Dispatcher`) to Weather, SystemStatus, HomeAssistant, and `LocalScheduler` (the only scheduling driver today); credentials and chosen scheduling driver are world-scoped in Atlas. `side_effects: false` refuses `writes?/1` intents with `{:error, :side_effects_off}`. Every service error maps to a planned reply in `DiscoursePlanner.enrichment_error_primitive/2`.
- **Response templates**: Runtime-manageable via Settings UI (`/settings?section=templates`) with ETS caching and file persistence.
- **Self-learning training worlds**: Isolated environments for entity discovery from large corpora (e.g., scripts, documents) with A/B testing, metrics comparison, and human review workflows.
- **Autonomous learning**: Comprehension-gated knowledge acquisition, conversation-driven research triggers, auto-approval of high-confidence corroborated findings, runtime intent/entity learning, incremental TF-IDF updates. Training examples only buffer on **explicit user feedback**, never on the model's own high-activation output — a deliberate guard against self-confirmation.
- **Comprehension assessment**: 8-dimension self-interrogation system that gates knowledge acquisition on text understanding. Dimension weights evolve via EMA based on approval/rejection outcomes.
- **Fast paths**: `Brain.Analysis.RacingAnalyzer` runs six parallel analyzers (`:model, :structural, :keyword, :pattern_recognition, :memory_similarity, :self_knowledge`); early-exit at raw_score ≥ 0.9, pipeline-bypass fast-path at ≥ 0.85 for heuristic or memory match.
- **Telemetry & metrics**: Instrumented telemetry across the pipeline with metrics aggregation and a monitoring dashboard.
- **Code maintenance tooling**: Automated Credo fixers with AST-based transformations and corruption detection (`mix credo_fix`).

---

## Quick start

### Prerequisites

**Required:**

- **Elixir** >= 1.18 (`.tool-versions` pins 1.19.5)
- **Erlang/OTP** >= 27 (`.tool-versions` pins 28.3.3)
- **PostgreSQL** — local install or the `apache/age` container via Docker Compose
- **Node.js with npm** — `mix assets.setup` installs the web fonts from npm (`apps/chat_web/assets/package.json`)

**Optional:**

- **asdf** — version manager; install Elixir/Erlang/Python from `.tool-versions`
- **Python** >= 3.12 — for tokenizer data generation scripts
- **gcc / g++** — for tree-sitter grammar compilation
- **Docker** — for containerized setup or the database service

### Automated setup (recommended)

The setup script auto-detects your hardware (CPU/CUDA/ROCm), scans for missing
artifacts (models, corpora, data files), and runs only the steps needed:

```bash
./scripts/setup.sh
```

Check what's missing without changing anything:

```bash
./scripts/setup.sh --check
```

Enable all optional steps (Python data, WordNet, Ouro model, tree-sitter grammars):

```bash
./scripts/setup.sh --all
```

### Docker Compose setup

The setup script auto-detects GPU hardware and selects the right compose config
(base, CUDA overlay, or ROCm overlay):

```bash
./scripts/setup.sh --docker
```

Or manually:

```bash
docker compose up -d db          # Start the database
docker compose up app            # Start the app (CPU)

# With NVIDIA GPU:
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up app

# With AMD GPU:
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up app
```

### Manual setup

If you prefer manual control:

```bash
cp .env.example .env             # Create env file, adjust as needed
mix setup                        # Deps, DB, corpora, training
mix phx.server                   # Start the server
```

### Open the app

- **Chat UI**: `http://localhost:4000/chat`
- **Dashboard**: `http://localhost:4000/dashboard`
- **Accuracy metrics**: `http://localhost:4000/accuracy`

For a REPL while running Phoenix:

```bash
iex -S mix phx.server
```

---

## How it works (high level)

### Core flow

1. **Input** arrives at `Brain.evaluate/3` (GenServer in the brain app). Requires `:time_zone` (a `Brain.Temporal.Zone`); takes `:language`, `:user_id`. `:side_effects` is set once per conversation in `create_conversation/1`.
2. `FollowupDetector.is_followup?` decides whether this turn resolves a pending clarification; if so it threads through `handle_followup_message` and either completes the request, re-asks for the one slot it narrowed, or (if the user didn't name a candidate) falls back to `process_new_message`.
3. `SelfKnowledgeAnalyzer.is_self_knowledge_query?` short-circuits meta-cognitive queries about the bot's own knowledge.
4. The **analysis pipeline** (`Brain.Analysis.Pipeline.process/2`) runs two passes per chunk. Pass 1 (every chunk, parallel): discourse / speech act / sentiment classification, basic entity extraction, anaphora resolution, POS tagging, event extraction and linking, SRL, preliminary intent via an empty lattice. Cross-chunk aggregation picks the primary and question chunks, folds speech-act / addressee / entity-familiarity into a `ContextAccumulator`. Pass 2 (substantive chunks only — primary, question, or assertive/directive with ≥1 entity): the `:intent_full` micro-classifier produces a `Brain.Lattice`, `SpeechActClassifier.refine_with_intent` splices the result into the speech-act voter, `ChunkProfile.materialize` builds the profile that reranks the lattice (+0.15 exact match, +0.05 domain prefix, registered intents only), `determine_intent` arbitrates (`@disambiguation_threshold = 0.5`, `@low_confidence_floor = 0.3`, trust the classifier only if its label is registered AND confidence ≥ 0.5 AND `:intent_domain` micro-classifier agrees — otherwise route through `infer_intent_from_speech_act` and tag `:speech_act_fallback`), entities retype with the profile, spans that were previously undecided get a second chance with `resolve_with_intent`, `EntityTypeScorer.rescore` then `ContextualEntityInferrer.infer` (multi-hypothesis IS-A narrowing with 4 signals), slot detection and context resolution, fact verification, novel-intent candidate recording.
5. `ResponseGate.evaluate` decides `:respond | :optional | :defer` from speech-act history (gratitude loops, backchannels, continuations, rapid-fire detection, echo).
6. The **response** is iteratively refined (up to 3 passes) by `Brain.Response.RefinementLoop`: `DiscoursePlanner.plan` picks a backbone of primitives from `discourse_patterns.json` (or collapses to one `enrichment_error_primitive` on a failed service enrichment), `ContentSpecifier.specify` fills each primitive from facts / beliefs / enrichment / entities / sentiment, `SurfaceRealizer.realize` routes per-domain to the phrase-lattice beam-search realizer, the Ouro LoopLM 1.4B, or the domain-knowledge synthesizer, and `ResponseEvaluator.evaluate` scores the realized text on 9 dimensions (speech-act alignment, confidence alignment, content coverage, content completeness, slot coverage, naturalness, echo avoidance, belief grounding against JTMS at weight 0.16, epistemic consistency). The weakest dimension drives which stage re-runs. If the best pass is below the silence threshold (<0.35), the response is withheld — the one exception is a service-error reply, which is never withheld.
7. Results are stored into:
   - conversation memory
   - cognitive memory store (episodes + 15-min consolidation at cosine ≥ 0.8 into semantic facts; `ConsolidationBridge` bridges them to beliefs)
   - epistemic layer: `BeliefStore` (per world, each belief with a JTMS node — premise for `:explicit`, retractable assumption for `:inferred/:learned/:consolidated`, confidence decayed on a timer with authority-aware rate), `UserModelStore` (per web user, cross-world)
8. **Outcome learning** (`Brain.Analysis.OutcomeLearner`) feeds back into racing-analyzer calibration, heuristic promotion (user 5 / cohort 4 / global 3 successes), and training example buffering — but only when `user_feedback == :positive`, not from the model's own prediction.
9. **Autonomous learning loops** run in the background:
   - Conversation events flow to BeliefStore (via `extract_beliefs_from_event/2`) and JTMS nodes
   - Memory consolidation bridges semantic facts into beliefs (ConsolidationBridge)
   - NoveltyDetector publishes gaps to LearningTriggers, which auto-starts research sessions
   - ResearchAgent gates findings through ComprehensionAssessor (8-dimension scoring)
   - High-confidence corroborated findings auto-approve into FactDatabase + BeliefStore
   - EntityPromoter auto-promotes discovered entities to the Gazetteer after sufficient occurrences
   - TrainingExampleBuffer collects high-activation outcomes and triggers incremental TF-IDF updates

### Umbrella Apps

This is an umbrella project with five apps:

| App | Location | Purpose |
|-----|----------|---------|
| **brain** | `apps/brain/` | Core NLP, ML (TF-IDF + feature-vector classifiers), Memory, Epistemic, Response, Knowledge, Code Analysis, Telemetry |
| **world** | `apps/world/` | Training worlds, entity discovery, type inference, per-world model registry |
| **tasks** | `apps/tasks/` | NLP benchmark task data curation and transformation |
| **chat_web** | `apps/chat_web/` | Phoenix 1.8 web layer, LiveView dashboards, WebSocket channels |
| **fourth_wall** | `apps/fourth_wall/` | Code maintenance tooling, automated Credo fixers, AST transformations |

### Notable subsystems

- **ML/NLP** (`apps/brain/lib/brain/ml/`): TF-IDF classifiers (`SimpleClassifier`), dense-centroid classifiers (`FeatureVectorClassifier` with `WeightOptimizer` GA), `MicroClassifiers` GenServer (17 named classifiers; `ready?/0` is false until all load; stale models surface in `stale/0` and raise if `Pipeline.process/2` is called), gazetteer (ETS-backed), tokenizer, entity extractor (gazetteer + system parsers + length readings + location hints + proper-noun hints + world BIO entity model + `Brain.Code.EntityReadings`, decided by `SpanReliability`), POS tagger, training server, corpus manager, training example buffer, KnowledgeGraph triple scorer and entity vector cache.
- **Analysis** (`apps/brain/lib/brain/analysis/`): two-pass pipeline orchestration, `FeatureExtractor` (25-group feature vector whose shape is sized at runtime by `dimension_manifest/0`), `ChunkProfile` (profile domain, posture, derived label), slot detection (`intent_registry.json` + service schemas), `ContextualEntityInferrer` (multi-hypothesis IS-A narrowing), `ContextResolver` and `AnaphoraResolver`, `RacingAnalyzer` (fast path with heuristic / memory match / parallel analyzers), `AnalyzerCalibration` (10-bucket histogram per racing analyzer — note: does **not** calibrate the pipeline's micro-classifiers), novelty detection, comprehension assessment, outcome learning.
- **Memory** (`apps/brain/lib/brain/memory/`): TF-IDF embedder (global), vector index, episodic/semantic store (world-scoped, Atlas-primary with VectorIndex cache), consolidation (`ConsolidationBridge` to beliefs), recurring-pattern aggregation.
- **Epistemic** (`apps/brain/lib/brain/epistemic/`): JTMS (per-world networks, label propagation, contradiction handler), BeliefStore (confidence decay with authority-aware rate, auto-retraction below 0.1, Atlas write-through), UserModelStore (per web user, cross-world), SourceAuthority, ContradictionHandler, DisclosurePolicy, ConsolidationBridge, StanceTracker.
- **Response** (`apps/brain/lib/brain/response/`): `RefinementLoop` (iterative, up to 3 passes), `DiscoursePlanner` (data-driven backbone + service-error reply table), `ContentSpecifier`, `SurfaceRealizer` (routed per domain by `ResponseSystemRouter`), `PhraseLatticeRealizer` + `LatticeScorer` (beam search over typed fragments), `OuroRealizer` (Ouro LoopLM 1.4B), `Synthesizer` (domain-knowledge frames + enriched frames + service replies), `ResponseEvaluator` (9-dimension scoring), `TemplateStore` + `TemplateBlender`, `SemanticFactRetriever`, `DecompressorCollector` (training data for Ouro's decompressor), `MemoryAugmented`, `ResponseQuality`, `Enricher`.
- **Lexicon** (`apps/brain/lib/brain/lexicon/`): `UserDefined` (owned sense facts in ETS + Atlas, EMA centroid blending, weekly decay), `SenseDrift` (canonical vs observed cosine distance), `IntentDomains` (compile-time validated consolidation), `TypeAnchors`, `ConceptNet`, `Clitics`, `ClosedClass`, `Seeder`, `Loader`.
- **Knowledge** (`apps/brain/lib/brain/knowledge/`): learning center, review queue (with auto-approval rules), source reliability, corroboration, research agent (with comprehension gate), learning triggers (conversation-driven auto-research), academic APIs (arXiv, Semantic Scholar, OpenAlex).
- **Code** (`apps/brain/lib/brain/code/`): code analysis pipeline, AST parsing, symbol extraction, code gazetteer, relationship mapping, multi-language grammar support.
- **Services** (`apps/brain/lib/brain/services/`): `Dispatcher` (routes to Weather, SystemStatus, HomeAssistant, LocalScheduler), `CredentialVault` (world-scoped credentials), `Cache`, `SchedulingDriver` behavior, `LocalScheduler` (the one scheduling driver today — a world picks one via `Atlas.WorldServiceDrivers`), `HomeAssistant` with capability registry / entity mapper / discovery / state formatter.
- **Temporal** (`apps/brain/lib/brain/temporal/`): `Zone`, `Vocabulary`, `Resolver` (reads durations and resolves temporal references against the turn's time), `Resolution`.
- **Metrics** (`apps/brain/lib/brain/metrics/`): telemetry instrumentation and metrics aggregation.
- **Fact Database** (`apps/brain/lib/brain/fact_database/`): structured fact storage and retrieval.
- **World** (`apps/world/lib/world/`): training worlds, entity discovery (POS-based), type inference, document ingestion, per-world model registry, entity auto-promoter, persistence, world metrics.

---

## Documentation

For detailed documentation, see the `docs/` folder:

- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)**: Comprehensive contributor guide with project philosophy, module APIs, and development workflow
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Visual architecture diagrams and end-to-end flow documentation
- **[PIPELINE_ORDER.md](docs/PIPELINE_ORDER.md)**: Detailed pipeline execution order reference
- **[SUBSYSTEM_INTEGRATION_REVIEW.md](docs/SUBSYSTEM_INTEGRATION_REVIEW.md)**: Disconnected subsystems, scoping evolution, and integration recommendations
- **[WRITING_TESTS.md](docs/WRITING_TESTS.md)**: Testing guidelines and best practices
- **[SCIENTIFIC_METHOD.md](docs/SCIENTIFIC_METHOD.md)**: Knowledge expansion system methodology
- **[EXTERNAL_SERVICES_INTEGRATION.md](docs/EXTERNAL_SERVICES_INTEGRATION.md)**: External services integration guide

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Two-pass pipeline** | Pass 1 runs on every chunk (cheap: discourse, speech act, sentiment, basic entities, POS, events, SRL, preliminary intent); pass 2 only on substantive chunks (full `:intent_full` classification, retyping, slot detection, fact verification) |
| **Fast path vs deep analysis** | `RacingAnalyzer` short-circuits on a heuristic or memory match at ≥ 0.85; otherwise the full `Pipeline` runs |
| **Candidate lattice** | `Brain.Lattice` keeps every candidate alive with margins and sources through reranks; abstention is an empty lattice with an error tombstone |
| **Multi-signal intent arbitration** | `determine_intent` trusts the classifier only when the label is registered AND confidence ≥ 0.5 AND the `:intent_domain` micro-classifier agrees; otherwise routes through `infer_intent_from_speech_act` (not simple ensemble voting) |
| **Provenance as a closed vocabulary** | `:computed | :declared | :default | :absent | :unavailable`; per-process collection, off by default |
| **Iterative response refinement** | `RefinementLoop` runs plan → specify → realize → evaluate up to 3 times; the 9-dim `ResponseEvaluator` analyzes the realized text through the full pipeline again to score belief grounding and epistemic consistency |
| **Hard-fail invariants** | Pipelines refuse to run with stale classifiers, missing POS model, or POS/SRL tag drift; `Brain.evaluate/3` raises without `:time_zone`; `DiscoursePlanner` raises on an unmapped service-error reason |
| **Comprehension Gating** | 8-dimension self-interrogation gates knowledge acquisition; text must be understood before it can be learned |
| **Autonomous Learning** | Conversation gaps trigger research sessions; high-confidence findings auto-approve; intents/entities learn at runtime; training examples buffer only on explicit user feedback (never the model's own output) |
| **World-Scoping** | Per-world isolated models, embeddings, entities, beliefs, JTMS networks, lexicon facts, service credentials, and scheduling driver choice (via `World.Context` + `Atlas`) |
| **Owned lexicon with sense drift** | `Brain.Lexicon.UserDefined` learns sense facts (seeded from WordNet, extended from conversation) with EMA centroid blending; `SenseDrift` flags words whose canonical and observed centroids diverge |
| **Memory Types** | `MemoryStore` (persona) vs `Memory.Store` (world-scoped episodic/semantic) vs semantic (consolidated at cosine ≥ 0.8) |
| **No Regex for NLP** | All NLP uses classical ML / trained models, not string matching |

---

## Web routes

| Route | LiveView | Purpose |
|-------|----------|---------|
| `/` | PageController | Landing page |
| `/chat` | ChatLive | Main chat with processing inspector |
| `/chat/:conversation_id` | ChatLive | Chat with specific conversation |
| `/dashboard` | DashboardLive | System monitoring, GenServer status, ML model health |
| `/accuracy` | AccuracyLive | ML accuracy metrics, confusion matrices, experiment comparison |
| `/explorer` | ExplorerLive | Training world data browser (entities, episodes, knowledge) |
| `/settings` | SettingsLive | World/entity management, ML training controls, templates, services |
| `/code` | CodeAnalysisLive | Code symbol browser, parsing, relationship visualization |
| `/knowledge-review` | Admin.KnowledgeReviewLive | Knowledge expansion review with source reliability |

---

## Data & model files

### Inputs (training/data)

- **Intents**: `data/intents/*.json`
- **Entities**: `data/entities/*.json` (plus generated `*_entries_en.json` files)
- **Intent registry (slot schemas + clarification templates)**: `apps/brain/priv/analysis/intent_registry.json` — this is the single source of truth. Service modules (`HomeAssistant`, `LocalScheduler`, …) contribute additional schemas via `Dispatcher.service_schemas/0` which `SlotDetector` merges in at runtime. A `slot_schemas.json` reference you may find in older notes is stale; the file was removed because nothing generated it.
- **Intent domain consolidation**: `apps/brain/priv/knowledge/intent_domains.json` (compile-time validated for idempotence and registry coverage; see `Brain.Lexicon.IntentDomains`)
- **Discourse patterns**: `apps/brain/priv/response/discourse_patterns.json` (backbone selection for `DiscoursePlanner`)
- **Response system config**: `apps/brain/priv/response/system_config.json` (per-domain `:lattice | :ouro | :synthesizer | :template` routing, tone vectors, lattice scorer weights)
- **Response primitives**: `apps/brain/priv/knowledge/domains/primitives.json` (hedges, acknowledgments, correction invites, fallback responses, service reply templates, service item kinds)
- **Entity type hierarchy and type mappings**: `apps/brain/priv/analysis/entity_types.json`, `apps/brain/priv/evaluation/ner/type_mappings.json`
- **Response templates**: `apps/brain/priv/response/templates.json`
- **Domain knowledge frames**: `apps/brain/priv/knowledge/domains/*.json`
- **Pattern triggers** (fast-path heuristics): `data/pattern_triggers.json`
- **Seeded heuristics**: `data/heuristics/seeded_heuristics.json`
- **Tokenizer expansions**: `data/informal_expansions.json` (generated by Python script)
- **Gold standard evaluation data**: `apps/brain/priv/evaluation/{intent,ner,sentiment,speech_act}/gold_standard.json`

### Outputs (generated / trained)

- **Classical models**: `apps/brain/priv/ml_models/*.term` (intent classifier, entity model, embedder, POS tagger, sentiment classifier)
- **Feature-vector micro-classifiers**: `apps/brain/priv/ml_models/micro/*.term` (intent_full, intent_domain, framing_class, tense_class, aspect_class, urgency, certainty_level, framing_neutral_centroid, ...)
- **Knowledge-graph triple scorer**: `apps/brain/priv/ml_models/kg_lstm/<world_id>/*.term` (BiLSTM, trained per world via `mix train.kg_lstm`)
- **Comprehension weights**: `apps/brain/priv/analysis/comprehension_weights.json` (EMA-evolved dimension weights)
- **Persisted conversation memory** (runtime): `apps/brain/priv/memory/`
- **Persisted knowledge** (runtime): `apps/brain/priv/knowledge/`
- **Training worlds** (persistent mode): `apps/world/priv/training_worlds/{world_id}/`

---

## Configuration (env vars)

Most configuration is in `config/config.exs` and can be overridden by environment variables:

- **`ML_ENABLED`**: enable/disable ML/NLP pipeline (`true`/`false`)
- **`ML_CONFIDENCE_THRESHOLD`**: default `0.75`
- **`ML_ENTITY_CONFIDENCE_THRESHOLD`**: default `0.51`
- **`ML_MODELS_PATH`**: default `priv/ml_models` (resolved via `Application.app_dir`)
- **`ML_TRAINING_DATA_PATH`**: default `data`
- **`ML_USE_GPU`**: default `true` (only relevant if Nx/EXLA backend supports it)
- **`ML_BATCH_SIZE`**: default `1000`
- **`ML_MAX_FEATURES`**: default `5000`
- **`KNOWLEDGE_DIR`**: default `priv/knowledge` (resolved via `Application.app_dir`)
- **`MEMORY_DIR`**: default `priv/memory` (resolved via `Application.app_dir`)
- **`TRAINING_WORLDS_PATH`**: default `priv/training_worlds` (resolved via `Application.app_dir`)
- **`INTENT_PROMOTION_ENABLED`**: enable novel intent promotion (`true`/`false`, default `false`)
- **`WEBSOCKET_HUB_URL`**: default `http://localhost:3001`

Autonomous learning:

- **`auto_extraction_enabled`**: enable automatic belief extraction from conversation events (config flag, default `false`)
- **`auto_approval_enabled`**: enable auto-approval of high-confidence corroborated findings in ReviewQueue (config flag, default `false`)

XLA/GPU acceleration:

- **`XLA_TARGET`**: `cpu` (default), `cuda` (NVIDIA GPU), `rocm` (AMD GPU), `tpu` (Google TPU)
- **macOS / Apple Silicon**: use **`XLA_TARGET=cpu`**. EXLA uses the XLA CPU backend there (no Metal GPU). Set this in your shell before **`mix deps.compile xla exla --force`** so the native `xla` / `exla` artifacts match; loading `.env` at runtime does not recompile those deps.
- **Docker Compose**: the `app` service defaults to **`GPU_DOCKERFILE=Dockerfile`** (CPU) and sets **`XLA_TARGET`/`XLA_BUILD`** in `environment` so a GPU-oriented `.env` does not break the CPU image. On **Apple Silicon**, use the default stack; for **Linux + NVIDIA** + a CUDA image (e.g. `Dockerfile.test.cuda`), set `GPU_DOCKERFILE` and add a small compose override with `deploy.resources.reservations.devices` / `gpus: all` as appropriate for your Docker engine.
- **`XLA_FLAGS`**: e.g., `--xla_gpu_cuda_data_dir=/path/to/cuda` for CUDA

Release/runtime:

- **`PHX_SERVER`**: set to `true` to start the endpoint in releases (see `config/runtime.exs`)
- **`SECRET_KEY_BASE`**: required in prod
- **`PHX_HOST`**, **`PORT`**: prod endpoint config

---

## Commands (mix tasks + scripts)

This section lists the **project-specific** tasks/scripts, plus the **most useful built-in** Phoenix/Mix commands for day-to-day work.

### Setup scripts

| Script | Purpose |
|--------|---------|
| `./scripts/setup.sh` | Smart local/Docker setup — detects hardware, scans artifacts, runs only what's needed |
| `./scripts/setup_production.sh` | Production setup — downloads pre-trained models from S3, runs migrations, starts server |
| `./scripts/setup_and_deploy.sh` | Full pipeline — trains all models, evaluates, builds assets (production) |

#### `./scripts/setup.sh` flags

| Flag | Effect |
|------|--------|
| `--check` | Dry-run: detect hardware and report missing artifacts only |
| `--docker` | Setup via Docker Compose (auto-detects GPU config) |
| `--skip-training` | Skip model training |
| `--full-training` | Full training pipeline (no `--quick` shortcut) |
| `--skip-db` | Skip database create/migrate/seed |
| `--with-python` | Set up Python venv and generate data files |
| `--with-grammars` | Compile tree-sitter grammars |
| `--with-ouro` | Download Ouro model files |
| `--with-wordnet` | Download WordNet Prolog files |
| `--all` | Enable all optional steps |

#### Scripts architecture (`scripts/lib/`)

Shared shell helpers live in `scripts/lib/` and are sourced by all setup scripts.
When adding new setup logic, put reusable functions here instead of duplicating
across scripts:

| File | Contents |
|------|----------|
| `lib/ui.sh` | Colors, `step()`, `ok()`, `warn()`, `fail()`, `banner()`, `print_elapsed()` |
| `lib/env.sh` | `resolve_root_dir()`, `load_dotenv()`, `install_interrupt_trap()` |
| `lib/xla.sh` | `detect_xla_target()`, `configure_xla_cache()` |
| `lib/preflight.sh` | `source_exla_preflight()` wrapper |
| `lib/artifacts.sh` | `check_term_models()`, `check_corpora()`, `check_ouro()`, `check_grammars()`, etc. |
| `lib/docker.sh` | `detect_docker_compose_files()`, `suggest_gpu_dockerfile()` |

### Core app / dev workflow

- **`mix setup`**: installs deps + sets up/builds assets (alias: `deps.get`, `assets.setup`, `assets.build`)
- **`mix phx.server`**: run the web server on `http://localhost:4000`
- **`iex -S mix phx.server`**: run server with an interactive shell
- **`mix test`**: run fast tests (excludes slow/integration/training/benchmark/gpu by default)
- **`mix test --include slow`**: include slow tests
- **`mix test --include integration`**: include integration tests
- **`mix test --include training`**: include model training tests
- **`mix test --only benchmark`**: run accuracy threshold tests
- **`mix test --exclude requires_lstm`**: skip LSTM-dependent tests
- **`mix format`**: format code
- **`mix precommit`**: runs `compile --warning-as-errors`, `deps.unlock --unused`, `format`, `test`

#### Test Tags

Tests are tagged for CI optimization:

| Tag | Purpose |
|-----|---------|
| `:slow` | Tests that take >5 seconds |
| `:integration` | Tests requiring full application stack |
| `:training` | Tests that train ML models |
| `:benchmark` | Accuracy threshold tests (excluded by default) |
| `:requires_lstm` | Tests needing LSTM models loaded |
| `:requires_pos_model` | Tests needing POS tagger model |
| `:gpu` | GPU-specific tests |
| `:smoke` | Quick smoke tests for route rendering |
| `:wip` | Work-in-progress tests (excluded) |

Assets (aliases defined in `mix.exs`):

- **`mix assets.setup`**: install Tailwind + esbuild if missing, and the npm packages in `assets/package.json` (`npm ci`)
- **`mix assets.build`**: compile + build Tailwind + esbuild, and publish the IBM Plex fonts to `priv/static/fonts` (`mix chat_web.fonts`)
- **`mix assets.deploy`**: publish the fonts, minify assets + `phx.digest` (for production)

### Train ML models

Models are loaded on app boot; train them when you change intent/entity data.

#### Master training pipeline

- **`mix train`**: full training pipeline (all classical + LSTM models)
- **`mix train --quick`**: skip slow models
- **`mix train --name "experiment_v1"`**: track experiment with a name
- **`mix train --compare`**: show experiment comparison after training

#### Classical models (TF-IDF)

- **`mix train_models`**: train TF-IDF models; saves to `priv/ml_models/`
- **`mix train_models --intent-only`**: train only intent classifier
- **`mix train_models --entity-only`**: train only entity recognition model

The gazetteer is not a trained model. `Brain.ML.Gazetteer` builds it in memory
from `data/entities/` and the city, artist and emoji CSVs every time the app
starts, and the app will not start if any of those sources is missing.

#### LSTM models (Axon/Nx)

- **`mix train_unified`**: train unified LSTM (intent + NER + sentiment + speech act)
- **`mix train_response`**: train response scorer LSTM
- **`mix train_lstm`**: train standalone LSTM intent classifier

#### Micro-classifiers (TF-IDF) — pragmatic + ChunkProfile axes

Small `SimpleClassifier` models used for lightweight decisions and for projecting
**domain, tense, aspect, urgency, certainty,** and **coarse semantic class**. Training
data lives under `data/classifiers/*.json`; axis JSON is generated from the intent
gold standard.

- **`mix gen_micro_data --world default`**: (re)build the six axis JSON files from `apps/brain/priv/evaluation/intent/gold_standard.json`
- **`mix train_micro`**: train every micro-classifier listed in `Mix.Tasks.TrainMicro` → `priv/ml_models/micro/<name>.term`
- **`mix train_micro --only intent_domain`**: train a single model
- **`mix train_micro --list`**: show which JSON/model files exist

Stage **9** of **`mix train`** runs `mix train_micro` unless you pass **`--skip-micro`**.
Run **`mix gen_micro_data --world default`** when you change gold data or axis heuristics **before** retraining micros.

### Evaluate models

- **`mix evaluate --world default`**: evaluate all tasks
- **`mix evaluate.intent --world default [--save] [--verbose]`**: intent classification evaluation
- **`mix evaluate.sentiment [--verbose]`**: sentiment analysis evaluation
- **`mix evaluate.speech_act`**: speech act classification evaluation
- **`mix evaluate.ner --world default`**: named entity recognition evaluation

### Utility tasks

- **`mix regenerate_test_models --check`**: check Nx/Axon model compatibility
- **`mix regenerate_test_models --all`**: regenerate all test models
- **`mix validate_intent`**: validate intent training data
- **`mix snapshot.record`**: record HTTP snapshots for test fixtures
- **`mix fact_database`**: fact database operations

### Code maintenance (fourth_wall)

- **`mix credo_fix`**: dry-run all Credo auto-fixers
- **`mix credo_fix --apply`**: apply fixes to files
- **`mix credo_fix --only trailing_whitespace,large_numbers`**: run specific fixers
- **`mix credo_fix --exclude alias_usage`**: exclude specific fixers

Available fixers: `trailing_whitespace`, `large_numbers`, `length_check`, `map_join`, `alias_usage`, `unused_alias`

### Clear runtime "knowledge" (without touching training data)

- **`mix clear_knowledge`**: clear all runtime knowledge (memory, learned facts, brain memory, admin entities)
- **`mix clear_knowledge --memory`**: clear only cognitive memory store
- **`mix clear_knowledge --learned`**: clear only learned facts (knowledge store)
- **`mix clear_knowledge --brain`**: clear only brain conversation memory
- **`mix clear_knowledge --entities`**: clear only admin-added gazetteer entries
- **`mix clear_knowledge --reload`**: clear and reload gazetteer from data files

### Generate / refresh entity datasets (then restart)

These tasks download bounded datasets, write JSON into `data/entities/`, and cache remote responses in `priv/data_cache/`. After running any of them, restart the app; the gazetteer is rebuilt from `data/entities/` at startup.

- **`mix generate_person_names`**
  - Source: US SSA baby names dataset (`names.zip`)
  - Outputs: `data/entities/person_entries_en.json`
  - Useful flags: `--min-count`, `--min-length`, `--years`, `--download`, `--use-builtin`, `--output`

- **`mix generate_countries_capitals`**
  - Source: REST Countries v3.1
  - Outputs: `data/entities/country_entries_en.json`, `data/entities/capital_entries_en.json`
  - Useful flags: `--download`, `--country-output`, `--capital-output`

- **`mix generate_fortune500`**
  - Source: Wikipedia "Fortune 500" page wikitext (MediaWiki API)
  - Outputs: `data/entities/company_entries_en.json`
  - Useful flags: `--year`, `--download`, `--output`

- **`mix generate_news_sources`**
  - Source: Wikidata SPARQL endpoint
  - Outputs: `data/entities/news-source_entries_en.json`
  - Useful flags: `--limit`, `--download`, `--output`

### Training worlds (self-learning)

Training worlds provide isolated environments for entity discovery from large text corpora. Use them to process scripts, documents, or other text sources to discover new entities without affecting production data.

- **`mix training_world.create "name"`**: create a new training world
  - `--mode=ephemeral` (default): in-memory only
  - `--mode=persistent`: saved to disk for later use

- **`mix training_world.ingest "world_id" "path/to/*.txt"`**: ingest files into a world
  - Discovers proper nouns using POS tagging
  - Tracks entity candidates, ambiguities, and co-occurrences
  - `--chunk-size=N`: characters per processing chunk

- **`mix training_world.metrics "world_id"`**: view discovery metrics (entities, types, confidence distribution)

- **`mix training_world.entities "world_id"`**: view discovered entity candidates
  - `--sort=confidence|occurrences`: sort order
  - `--limit=N`: max results

- **`mix training_world.ambiguous "world_id"`**: view entities with multiple possible types (need human review)

- **`mix training_world.events "world_id"`**: view event log
  - `--type=event_type`: filter by event type
  - `--limit=N`: max results

- **`mix training_world.compare "world_1" "world_2"`**: A/B comparison of two worlds

- **`mix training_world.export "world_id"`**: export world data for review
  - `--output=file.json`: output file path

- **`mix training_world.merge "source" "target"`**: merge learned entities from source to target
  - `--require-review=true` (default): returns entities for review instead of merging
  - `--min-confidence=0.7`: minimum confidence threshold

- **`mix training_world.list`**: list all active and persisted worlds
- **`mix training_world.checkpoint "world_id"`**: save persistent world to disk
- **`mix training_world.load "world_id"`**: load persisted world from disk
- **`mix training_world.destroy "world_id"`**: destroy a world

### Python data scripts

There is currently one Python script used to generate tokenizer normalization data.

- **Install script deps**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

- **Generate informal contraction/colloquial expansions**

```bash
python scripts/generate_informal_expansions.py
```

This writes/overwrites: `data/informal_expansions.json`

---

## ML Framework Stack

| Library | Version | Purpose |
|---------|---------|---------|
| `Nx` | ~> 0.10 | Numerical computing (tensors) |
| `Axon` | ~> 0.8 | Neural network framework |
| `EXLA` | ~> 0.10 | XLA compiler backend (GPU/CPU acceleration) |
| `Scholar` | ~> 0.3 | Statistical/ML functions |

Models are serialized via Erlang `:erlang.binary_to_term` / `term_to_binary` (`.term` files).

---

## Common workflows

### I updated intent/entity JSON -- what next?

```bash
mix train_models
mix phx.server
```

### I want to retrain everything (classical + LSTM) and track it

```bash
mix train --name "v2" --compare
```

### I generated new entity lists (SSA/countries/Wikidata/etc.)

Restart the app; the gazetteer is rebuilt from `data/entities/` at startup.

```bash
mix phx.server
```

### I want a "clean slate" for local testing

```bash
mix clear_knowledge
```

### I want to evaluate model accuracy

```bash
mix evaluate --world default --save --compare
mix test --only benchmark
```

### I want to discover entities from a corpus (e.g., scripts)

```bash
# Create an ephemeral training world
mix training_world.create "my_corpus"
# Note the world_id printed

# Ingest your text files
mix training_world.ingest "WORLD_ID" "path/to/scripts/*.txt"

# View what was discovered
mix training_world.metrics "WORLD_ID"
mix training_world.entities "WORLD_ID" --sort=occurrences

# Review ambiguous entities (need human decision)
mix training_world.ambiguous "WORLD_ID"

# Export for review before merging
mix training_world.export "WORLD_ID" --output=review.json

# When ready, merge approved entities (or use --require-review=false)
mix training_world.merge "WORLD_ID" "production_world"

# Clean up
mix training_world.destroy "WORLD_ID"
```

### I want to A/B test two learning approaches

```bash
# Create two worlds with different configurations
mix training_world.create "approach_a"
mix training_world.create "approach_b"

# Ingest the same data into both
mix training_world.ingest "WORLD_A_ID" "data/*.txt"
mix training_world.ingest "WORLD_B_ID" "data/*.txt"

# Compare results
mix training_world.compare "WORLD_A_ID" "WORLD_B_ID"
```

---

## Autonomous Learning System

The system can learn autonomously from conversations, research sessions, and its own analysis outcomes. All autonomous features are opt-in and gated by safety mechanisms.

### How it works

1. **Comprehension gating**: Before any text enters the learning pipeline, the ComprehensionAssessor scores it across 8 dimensions (referential clarity, actor identification, propositional content, temporal grounding, contextual sufficiency, epistemic grounding, structural coherence, illocutionary clarity). Text must score >= 0.4 composite to be learnable; structural coherence < 0.2 is an immediate reject.

2. **Conversation-driven research**: When NoveltyDetector flags 3+ novel inputs on the same topic within 24 hours, LearningTriggers auto-starts a LearningCenter research session. Rate-limited to 2 sessions/day.

3. **Knowledge auto-approval**: Findings that meet all criteria (confidence >= 0.85, 3+ independent sources, all sources reliability >= 0.6, hypothesis `:supported`, no conflicts, comprehension verdict `:comprehended`) are auto-approved. Capped at 10/day.

4. **Entity promotion**: EntityPromoter scans entity candidates every 10 minutes, promoting those with >= 3 occurrences and confidence >= 0.6 to the Gazetteer.

5. **Incremental model updates**: TrainingExampleBuffer collects high-activation outcomes (>= 0.8) and flushes at 50+ examples to incrementally update the TF-IDF classifier. Full retrain triggers after 200 incremental updates.

6. **Belief pipeline**: Conversation events flow to BeliefStore and create JTMS nodes (premises for explicit, retractable assumptions for inferred). Memory consolidation bridges semantic facts into beliefs. Inferred beliefs decay 5%/tick if unconfirmed for 24+ hours, auto-retracting below 0.1.

### Safety mechanisms

| Mechanism | Details |
|-----------|---------|
| Comprehension gate | Composite score < 0.4 blocks learning; structural coherence < 0.2 = immediate `:garbled` reject |
| Partial verdict penalty | Findings from `:partial` comprehension get `confidence * composite_score` |
| Cold-start protection | Dimension weights don't evolve until 10+ outcomes recorded |
| Weight rollback | Last 5 weight snapshots persisted; `reset_weights/0` reverts to equal weights |
| Auto-approval cap | 10/day (ReviewQueue) |
| Auto-trigger cap | 2 LearningCenter sessions per day |
| Config kill switches | `auto_extraction_enabled`, `auto_approval_enabled` |
| Confidence decay | Inferred beliefs decay; auto-retracted below 0.1 |
| Incremental drift guard | Full retrain after 200 incremental TF-IDF updates |
| World isolation | Entity promotions use world-scoped `Gazetteer.add_to_world/4` |
| Backpressure | Belief extraction via `Task.Supervisor` with max_children limits |

### Quick reference

```elixir
# Check comprehension assessment
Brain.Analysis.ComprehensionAssessor.assess(chunk_analyses)
Brain.Analysis.ComprehensionAssessor.stats()
Brain.Analysis.ComprehensionAssessor.reset_weights()

# Check autonomous learning status
Brain.Knowledge.ReviewQueue.auto_approval_enabled?()
```

---

## Troubleshooting

### Models not found / low quality responses

- Run `mix train` (all 6 stages) and restart the app.
- Run `mix setup` for a full setup including corpus downloads and data generation.
- Confirm `ML_TRAINING_DATA_PATH` points at the folder containing `intents/` + `entities/`.
- Confirm `ML_MODELS_PATH` is where you expect `*.term` files to be written/read.
- Check `Brain.ML.MicroClassifiers.ready?()` in an IEx session.

### Large data generation tasks timing out

- Re-run with the task's `--download` flag (forces cache refresh).
- Check `priv/data_cache/` to confirm cached downloads exist.

### How do I check what's missing?

Run the setup script in check mode — it scans for all expected artifacts and
prints a report without changing anything:

```bash
./scripts/setup.sh --check
```

### Docker GPU detection

The setup script probes for ROCm (`/opt/rocm`), CUDA (`nvidia-smi` / `/usr/local/cuda`),
and Apple Silicon (`sysctl`). Override the auto-detected backend by setting `XLA_TARGET`
in your `.env` or shell before running:

```bash
XLA_TARGET=cuda12 ./scripts/setup.sh --docker
```

### Setup script failing on database

If PostgreSQL isn't running yet, skip the database step and start the container first:

```bash
docker compose up -d db
./scripts/setup.sh           # will now connect to the container DB
```

Or skip the database entirely and set it up later:

```bash
./scripts/setup.sh --skip-db
```

### GenServer timeout errors

- Check `Brain.SystemStatus.get_all()` for component health.
- Ensure you call `ready?()` before interacting with GenServers that may still be initializing.
- Use short timeouts (100ms) with graceful fallbacks.
