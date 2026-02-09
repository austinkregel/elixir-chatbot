## :bangbang: Disclaimer :bangbang:
This application was vibe coded with Claude and Cursor over the course of several months. I did try rather hard to ensure that the built systems do what they describe, but the _whole_ time was a constant battle with every model trying to take shortcuts like string matching, regex, or hardcoding intents, or switching to python while writing some of the pipelines.

While this is my pet-project, I cannot guarantee that it works as intended or that . It has sigificantly ballooned in scope since I started it, and I have left it unattended sometimes while implementing different parts. No one has permission to use this in a production environment, and no I wouldn't recommend it.

## ChatBot (Elixir/Phoenix)

This is a Phoenix LiveView chatbot application built around **classical NLP techniques** (no LLMs): TF‑IDF vectorization, intent classification, entity extraction (gazetteer + learned maps), slot filling + clarification, and a cognitive memory + “epistemic” user-model system.

### What you get

- **Web UI**: LiveView chat UI at `/chat`, plus an entity admin panel at `/admin`.
- **Classical NLP pipeline**: chunking → discourse analysis → speech act classification → entity extraction → intent determination → slot detection → context resolution.
- **Cognitive memory**: TF‑IDF embeddings and similarity search used to augment reasoning/classification.
- **Epistemic user model**: stores user “facts” (beliefs) extracted from conversation when enabled.
- **Training & data tooling**: project-specific `mix` tasks to generate entity datasets, migrate training data, and train/load models.
- **Response Templates**: Runtime-manageable response templates via Settings UI (`/settings?section=templates`) with ETS caching and file persistence.
- **Self-learning training worlds**: isolated learning environments for entity discovery from large corpora (e.g., scripts, documents) with A/B testing, metrics comparison, and human review workflows.

---

## Quick start

### Prerequisites

- **Elixir**: `~> 1.15` (see `mix.exs`)
- **Node**: only needed for asset tooling (Phoenix uses esbuild + Tailwind)

### Setup + run

```bash
mix setup
mix phx.server
```

Open:

- **Chat UI**: `http://localhost:4000/chat`
- **Admin (gazetteer)**: `http://localhost:4000/admin`
- **Ops dashboard**: `http://localhost:4000/ops/dashboard`

For a REPL while running Phoenix:

```bash
iex -S mix phx.server
```

---

## How it works (high level)

### Core flow

1. **Input** arrives at `Brain` (GenServer in the brain app).
2. The **analysis pipeline** (`Brain.Analysis.Pipeline`) runs:
   - semantic chunking (multi-sentence handling)
   - discourse + speech act classification (parallel)
   - anaphora resolution (uses recent conversation history)
   - entity extraction (resolved text)
   - intent selection (entities → keywords → speech-act fallback)
   - slot detection + context resolution (history + user model + user profile)
   - overall strategy decision (`:can_respond`, `:needs_clarification`, etc.)
3. The **response** is generated (domain handlers, templates/smalltalk, memory-augmented fallback), then stored into:
   - conversation memory
   - global memory
   - cognitive memory store (episodes + consolidation into semantic facts)
   - epistemic user model (optional, based on config)

### Umbrella Apps

This is an umbrella project with five apps:

| App | Location | Purpose |
|-----|----------|---------|
| **nerve** | `apps/nerve/` | Telemetry, metrics, system status |
| **brain** | `apps/brain/` | Core NLP, ML, Memory, Epistemic, Response |
| **world** | `apps/world/` | Training worlds, entity discovery |
| **tasks** | `apps/tasks/` | Task data curation system |
| **chat_web** | `apps/chat_web/` | Phoenix web layer, LiveViews |

### Notable subsystems

- **ML/NLP** (`apps/brain/lib/brain/ml/`): intent classifier, gazetteer, tokenizer, entity extractor, POS tagger.
- **Analysis** (`apps/brain/lib/brain/analysis/`): pipeline orchestration, slot schemas, context/anaphora resolution, heuristics.
- **Memory** (`apps/brain/lib/brain/memory/`): TF-IDF embedder, vector index, store, consolidation, high-level API (`Think`).
- **Epistemic** (`apps/brain/lib/brain/epistemic/`): user facts/beliefs + contradiction handling.
- **World** (`apps/world/lib/world/`): training worlds, entity discovery, type inference, document ingestion, world metrics.
- **Responses** (`apps/brain/lib/brain/response/`): templates, response composition/connectors, memory augmentation.

---

## Documentation

For detailed documentation, see the `docs/` folder:

- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)**: Comprehensive contributor guide with project philosophy, module APIs, and development workflow
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Visual architecture diagrams and end-to-end flow documentation
- **[PIPELINE_ORDER.md](docs/PIPELINE_ORDER.md)**: Detailed pipeline execution order reference
- **[SUBSYSTEM_INTEGRATION_REVIEW.md](docs/SUBSYSTEM_INTEGRATION_REVIEW.md)**: Disconnected subsystems, scoping evolution, and integration recommendations
- **[WRITING_TESTS.md](docs/WRITING_TESTS.md)**: Testing guidelines and best practices
- **[SCIENTIFIC_METHOD.md](docs/SCIENTIFIC_METHOD.md)**: Knowledge expansion system methodology

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Two-Tier Analysis** | RacingAnalyzer (fast path) vs Pipeline (deep analysis) |
| **World-Scoping** | Directory-scoped embeddings for specialized domains |
| **Memory Types** | MemoryStore (persona) vs Memory.Store (episodic) |
| **No Regex** | All NLP uses classical ML, not string matching |

---

## Web routes

- **`/`**: home page
- **`/chat`**: main chat LiveView (streams analysis progress + status)
- **`/admin`**: manage gazetteer entries (add/remove/search; admin-added vs data-sourced)
- **`/ops/dashboard`**: operational dashboard (app-level)
- **`/api/*`**: test endpoints for learning/knowledge (see `ChatBotWeb.Router`)

---

## Data & model files

### Inputs (training/data)

- **Intents**: `data/intents/*.json`
- **Entities**: `data/entities/*.json` (plus generated `*_entries_en.json` files)
- **Slot schemas**: `priv/analysis/slot_schemas.json`
- **Response connectors**: `data/response_connectors.json`
- **Tokenizer expansions**: `data/informal_expansions.json` (generated by Python script)

### Outputs (generated / trained)

- **Trained models**: `priv/ml_models/*.term` (intent classifier, entity model, gazetteer, vectorizer)
- **Caches for external downloads**: `priv/data_cache/*`
- **Persisted conversation memory** (runtime): `priv/memory/`
- **Persisted knowledge** (runtime): `priv/knowledge/`
- **Training worlds** (persistent mode): `priv/training_worlds/{world_id}/`

---

## Configuration (env vars)

Most configuration is in `config/config.exs` and can be overridden by environment variables:

- **`ML_ENABLED`**: enable/disable ML/NLP pipeline (`true`/`false`)
- **`ML_CONFIDENCE_THRESHOLD`**: default `0.75`
- **`ML_MODELS_PATH`**: default `priv/ml_models`
- **`ML_TRAINING_DATA_PATH`**: default `data`
- **`ML_USE_GPU`**: default `true` (only relevant if Nx backend supports it)
- **`ML_BATCH_SIZE`**: default `1000`
- **`ML_MAX_FEATURES`**: default `5000`
- **`KNOWLEDGE_DIR`**: default `priv/knowledge`
- **`MEMORY_DIR`**: default `priv/memory`
- **`WEBSOCKET_HUB_URL`**: default `http://localhost:3001`

Release/runtime:

- **`PHX_SERVER`**: set to `true` to start the endpoint in releases (see `config/runtime.exs`)
- **`SECRET_KEY_BASE`**: required in prod
- **`PHX_HOST`**, **`PORT`**: prod endpoint config

---

## Commands (mix tasks + scripts)

This section lists the **project-specific** tasks/scripts, plus the **most useful built-in** Phoenix/Mix commands for day-to-day work.

### Core app / dev workflow

- **`mix setup`**: installs deps + sets up/builds assets (alias: `deps.get`, `assets.setup`, `assets.build`)
- **`mix phx.server`**: run the web server on `http://localhost:4000`
- **`iex -S mix phx.server`**: run server with an interactive shell
- **`mix test`**: run fast tests (excludes slow/integration/training by default)
- **`mix test --include slow`**: include slow tests
- **`mix test --include integration`**: include integration tests
- **`mix test --include training`**: include model training tests
- **`mix test --include slow --include integration --include training`**: run full suite
- **`mix format`**: format code
- **`mix precommit`**: runs `compile --warning-as-errors`, `deps.unlock --unused`, `format`, `test`

#### Test Tags

Tests are tagged for CI optimization:

| Tag | Purpose |
|-----|---------|
| `:slow` | Tests that take >5 seconds |
| `:integration` | Tests requiring full application stack |
| `:training` | Tests that train ML models |
| `:requires_pos_model` | Tests needing POS tagger model |
| `:smoke` | Quick smoke tests for route rendering |
| `:wip` | Work-in-progress tests (excluded) |

Assets (aliases defined in `mix.exs`):

- **`mix assets.setup`**: install Tailwind + esbuild if missing
- **`mix assets.build`**: compile + build Tailwind + esbuild
- **`mix assets.deploy`**: minify assets + `phx.digest` (for production)

### Train ML models

Models are loaded on app boot; train them when you change intent/entity data.

- **`mix train_models`**: full training pipeline; saves models to `priv/ml_models/`
- **`mix train_models --intent-only`**: train only intent classifier
- **`mix train_models --entity-only`**: train only entity recognition model
- **`mix train_models --gazetteer-only`**: build only gazetteer lookup tables
- **`mix train_models --skip-gazetteer`**: train models but skip gazetteer build (faster / lower memory)

### Clear runtime “knowledge” (without touching training data)

- **`mix clear_knowledge`**: clear all runtime knowledge (memory, learned facts, brain memory, admin entities)
- **`mix clear_knowledge --memory`**: clear only cognitive memory store
- **`mix clear_knowledge --learned`**: clear only learned facts (knowledge store)
- **`mix clear_knowledge --brain`**: clear only brain conversation memory
- **`mix clear_knowledge --entities`**: clear only admin-added gazetteer entries
- **`mix clear_knowledge --reload`**: clear and reload gazetteer from data files

### Generate / refresh entity datasets (then rebuild gazetteer)

These tasks download bounded datasets, write JSON into `data/entities/`, and cache remote responses in `priv/data_cache/`. After running any of them, typically run `mix train_models --gazetteer-only` and restart the app.

- **`mix generate_person_names`**
  - Source: US SSA baby names dataset (`names.zip`)
  - Outputs: `data/entities/person_entries_en.json`
  - Useful flags: `--min-count`, `--min-length`, `--years`, `--download`, `--use-builtin`, `--output`

- **`mix generate_countries_capitals`**
  - Source: REST Countries v3.1
  - Outputs: `data/entities/country_entries_en.json`, `data/entities/capital_entries_en.json`
  - Useful flags: `--download`, `--country-output`, `--capital-output`

- **`mix generate_fortune500`**
  - Source: Wikipedia “Fortune 500” page wikitext (MediaWiki API)
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

## Common workflows

### I updated intent/entity JSON — what next?

```bash
mix train_models
mix phx.server
```

### I generated new entity lists (SSA/countries/Wikidata/etc.)

```bash
mix train_models --gazetteer-only
mix phx.server
```

### I want a “clean slate” for local testing

```bash
mix clear_knowledge
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

## Troubleshooting

### Models not found / low quality responses

- Run `mix train_models` and restart the app.
- Confirm `ML_TRAINING_DATA_PATH` points at the folder containing `intents/` + `entities/`.
- Confirm `ML_MODELS_PATH` is where you expect `*.term` files to be written/read.

### Large data generation tasks timing out

- Re-run with the task’s `--download` flag (forces cache refresh).
- Check `priv/data_cache/` to confirm cached downloads exist.
