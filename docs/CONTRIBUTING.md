# Contributing to ChatBot

Welcome to the ChatBot project! This guide will help you understand our architecture, design philosophy, and how to contribute effectively.

## Table of Contents

- [Project Philosophy](#project-philosophy)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Two-Tier Analysis System](#two-tier-analysis-system)
- [Module API Reference](#module-api-reference)
  - [Core Processing (Brain)](#core-processing-brain)
  - [Analysis Pipeline](#analysis-pipeline)
  - [Response Generation](#response-generation)
  - [Memory System](#memory-system)
  - [Learning System](#learning-system)
  - [Knowledge Expansion](#knowledge-expansion)
  - [Epistemic System](#epistemic-system)
  - [ML/NLP Components](#mlnlp-components)
  - [Data Stores](#data-stores)
  - [Subprocesses](#subprocesses)
  - [Telemetry & Monitoring](#telemetry--monitoring)
- [Common Use Cases](#common-use-cases)
- [Pipeline Flows](#pipeline-flows)
- [Subsystem Interactions](#subsystem-interactions)
- [Testing & Debugging](#testing--debugging)
- [Development Workflow](#development-workflow)

---

## Project Philosophy

### Core Principles

1. **No Regex/String Matching**: We use classical ML, classification, and analysis instead of regex or string matching for NLP operations. This is a fundamental design constraint - use `Tokenizer` functions instead of regex patterns.

2. **LLM-Inspired Without LLMs**: We learn from modern LLMs but avoid resource-intensive models. The goal is novel responses from systems that don't require expensive GPUs or hundreds of gigabytes of training data.

3. **Human Thought Process Modeling**: Our systems model how humans analyze text. The different modules reflect personal thought processes for analyzing text received or said.

4. **Novel Responses**: Generate context-aware responses without repetition or parroting input. Responses don't need to be novel in that they've never been said before, but they should not just repeat words from the input.

5. **World-Based Learning**: Directory-scoped embeddings for specialized domains. Each "world" is an isolated training environment with its own embeddings, models, and learned data.

6. **Test-Driven Development**: Reproducible A/B/C testing for pattern recognition balance. Finding the balance between pattern recognition and avoiding training data drowning out signals requires systematic testing.

### Design Intentions

The overarching goal is to build a text analysis, review, and response system with the learnings of modern day LLMs but without the actual LLM. LLMs are much too resource intensive for the vast majority of use cases.

The different modules have come as the result of thinking through how humans personally analyze text - either sent to them or said to them. The systems are all in various states, but developed with the idea that regex, string replacement, or string matching of any kind for app code is unacceptable unless there is truly no alternative.

The closest we get to an LLM is the response generator, but we believe there are ways to build responses intelligently and concisely that allow users to experience non-repetitive responses when interacting with our system.

---

## Quick Start

### Prerequisites

- **Elixir**: `~> 1.15` (see `mix.exs`)
- **Node**: Only needed for asset tooling (Phoenix uses esbuild + Tailwind)

### Setup

```bash
# Install dependencies and setup assets
mix setup

# Start the development server
mix phx.server

# Or with interactive shell
iex -S mix phx.server
```

### Key URLs

- **Chat UI**: `http://localhost:4000/chat`
- **Admin (gazetteer)**: `http://localhost:4000/admin`
- **Ops Dashboard**: `http://localhost:4000/ops/dashboard`

### First Contribution Checklist

1. Read this document and understand the philosophy
2. Run `mix test` to ensure everything passes
3. Understand the two-tier analysis system (RacingAnalyzer vs Pipeline)
4. Familiarize yourself with world-scoping concepts
5. Add telemetry for any new subsystems you create

---

## Architecture Overview

```
User Input
    │
    ▼
┌────────────────┐
│ Brain.evaluate │ ─── Entry point, manages conversation state
└───────┬────────┘
        │
        ▼
    ┌───────────┐
    │ Fast Path │ ─── Check: is_followup?, is_self_knowledge?, heuristics, memory
    └─────┬─────┘
          │ (if no fast path hit)
          ▼
┌─────────────────────┐
│ Pipeline.process    │ ─── Main NLP analysis pipeline
├─────────────────────┤
│ 1. SemanticChunker  │ → Split into utterances
│ 2a. DiscourseAnalyzer│ ┐ (parallel)
│ 2b. SpeechActClassifier│ ┘
│ 2c. AnaphoraResolver │ → Resolve pronouns
│ 3a. EntityExtractor │ → Extract entities
│ 3b. determine_intent │ → Map to intent
│ 3c. SlotDetector    │ → Fill slot schema
│ 3d. ContextResolver │ → Resolve from history
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ ResponseGate.evaluate│ ─── Should we respond?
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Generator.generate  │ ─── Create response
└─────────────────────┘
```

### Subsystem Overview

| Subsystem | Purpose |
|-----------|---------|
| **Analysis Pipeline** | Text understanding (chunking, discourse, speech acts, slots) |
| **Epistemic System** | Truth maintenance, beliefs, user models |
| **Memory System** | Episodic/semantic memory with consolidation |
| **Learning System** | Training worlds, entity discovery, heuristic learning |
| **Knowledge Expansion** | Scientific method research, hypothesis testing, admin review |
| **Response Generation** | Template-based, semantic fact retrieval, memory-augmented responses |

---

## Two-Tier Analysis System

We have two types of analysis for input, designed to mirror human cognition:

### RacingAnalyzer (Fast Path) - Autonomic Nervous System

The fast path is like an autonomic nervous system - responding quickly to intents that need immediate responses.

```elixir
# Fast path check in Brain
case RacingAnalyzer.check_fast_path(text, world_id, user_id, cohort_id) do
  {:fast_path, interpretation} ->
    # Quick response - heuristic or memory match
    handle_fast_path_response(persona, interpretation, memory, opts)
    
  :no_match ->
    # Continue to deep analysis
    process_standard_message(persona, input, memory, opts)
end
```

**Characteristics:**
- Pre-trained on support documents
- Heuristic-based fast matching via `HeuristicStore`
- Memory similarity shortcuts via `Memory.Store`
- Designed for "help" style systems or pre-trained domains
- Uses `ActivationPool` for normalization

### Pipeline (Deep Analysis) - Personal Thought Process

The deep analysis path is meant to be your personal normal thought process for analyzing things - finding new nouns, slang, or deeper entity extraction.

```elixir
# Deep analysis via Pipeline
analysis_model = Pipeline.process(input, opts)
```

**Characteristics:**
- Finds new nouns, slang
- Deeper entity extraction
- Context-aware disambiguation
- Multi-pass analysis (discourse, speech act, entity, slot, context)
- Used when fast path has no confident match

### When Each Path is Used

| Situation | Path Used |
|-----------|-----------|
| High-confidence heuristic match (>= 0.85) | Fast Path |
| Similar memory episode found (>= 0.85) | Fast Path |
| Follow-up to previous message | Special handling |
| Self-knowledge query ("What do you know about me?") | Meta-cognitive handler |
| Everything else | Deep Analysis Pipeline |

---

## Module API Reference

### Core Processing (Brain)

The `Brain` module is the main orchestrator that manages conversations, processes input, and coordinates all subsystems.

**Location**: `lib/chat_bot/brain.ex`

#### Key Functions

```elixir
# Start a new conversation
{:ok, conversation_id} = Brain.create_conversation(world_id: "default")

# Evaluate user input
{:ok, response} = Brain.evaluate(conversation_id, "Hello! What's the weather?")

# Get conversation status
{:ok, conversation} = Brain.get_conversation(conversation_id)

# End a conversation
:ok = Brain.end_conversation(conversation_id)

# Get Brain status
status = Brain.get_status()
```

#### Options for `evaluate/3`

| Option | Description |
|--------|-------------|
| `:server` | The server to call (default: `Brain`) |
| `:timeout` | Call timeout in ms (default: 90,000) |
| `:world_id` | Training world to use (default: "default") |
| `:user_id` | User ID for epistemic tracking |
| `:defer_threshold` | Threshold for optional response deferral |

#### Subprocess Coordination

The Brain coordinates subprocesses which act as domain specialists - different "parts of the brain":

```elixir
# Start HTTP subprocess
{:ok, subprocess_id} = Brain.start_http_subprocess()

# Start conversation subprocess
{:ok, subprocess_id} = Brain.start_conversation_subprocess(conversation_id)

# List all subprocesses
subprocesses = Brain.list_subprocesses()

# Stop a subprocess
:ok = Brain.stop_subprocess(subprocess_id)
```

---

### Analysis Pipeline

The analysis pipeline orchestrates text understanding through multiple stages.

#### Pipeline (`Brain.Analysis.Pipeline`)

**Location**: `lib/chat_bot/analysis/pipeline.ex`

```elixir
# Process text through full pipeline
analysis_model = Pipeline.process("Hello! What's the weather in NYC?", 
  participants: [:user, :bot],
  conversation_history: history,
  user_profile: %{},
  world_id: "default"
)

# Analyze a single chunk
chunk_analysis = Pipeline.analyze_chunk("What's the weather?", opts)
```

**Returns**: `%InternalModel{}` with:
- `chunks` - List of semantic chunks
- `analyses` - Per-chunk analysis results
- `overall_strategy` - `:can_respond`, `:needs_clarification`, `:defer_to_user`, `:cannot_respond`
- `suggested_prompts` - Clarification prompts if needed

#### RacingAnalyzer (`Brain.Analysis.RacingAnalyzer`)

**Location**: `lib/chat_bot/analysis/racing_analyzer.ex`

```elixir
# Check fast path (world-scoped)
case RacingAnalyzer.check_fast_path(text, world_id, user_id, cohort_id) do
  {:fast_path, interpretation} -> # Use fast path
  :no_match -> # Continue to pipeline
end

# Full racing analysis
interpretation = RacingAnalyzer.race(text, 
  user_id: user_id,
  world_id: world_id
)
```

**Note**: `check_fast_path/3` is deprecated - use `check_fast_path/4` with `world_id`.

#### ResponseGate (`Brain.Analysis.ResponseGate`)

Evaluates whether a response is appropriate based on speech act sequences.

```elixir
case ResponseGate.evaluate(analysis_model, memory, opts) do
  {:respond, reason} -> # Generate response
  {:optional, confidence, reason} -> # Check threshold
  {:defer, reason} -> # Don't respond (gratitude loop, etc.)
end
```

#### Other Analysis Modules

| Module | Purpose |
|--------|---------|
| `SemanticChunker` | Splits input into utterances |
| `DiscourseAnalyzer` | Determines addressee (bot/user/third-party) |
| `SpeechActClassifier` | Classifies intent type (greeting, question, etc.) |
| `SlotDetector` | Detects required/missing slots for intents |
| `ContextResolver` | Resolves slots from history/user model |
| `AnaphoraResolver` | Resolves pronouns to entities |
| `EntityDisambiguator` | Resolves entity type ambiguity (used with AnaphoraResolver) |
| `FollowupDetector` | Detects follow-up messages |
| `SelfKnowledgeAnalyzer` | Handles meta-cognitive queries |
| `BacktrackController` | Alternative analysis paths |
| `ActivationPool` | Activation normalization (prevents runaway confidence) |
| `ProcessingTrace` | Debugging tool only |

---

### Response Generation

The response system assembles context-aware responses from all subsystems.

#### Generator (`Brain.Response.Generator`)

**Location**: `lib/chat_bot/response/generator.ex`

```elixir
# Generate response for intent and entities
{:ok, response, type} = Generator.generate(intent, entities, query_text)

# Generate with full path tracking (for debugging)
{:ok, response, type, path} = Generator.generate_with_path(intent, entities, query_text)
```

**Response types**: `:domain`, `:memory_augmented`, `:template`, `:fallback`

**Priority order**:
1. Domain-specific handlers (weather, music, device, etc.)
2. Memory-augmented (similar episodes found)
3. Template-based (from TemplateStore)
4. Fallback (generic acknowledgment)

#### Other Response Modules

| Module | Purpose |
|--------|---------|
| `TemplateStore` | Template management, slot substitution, runtime CRUD via UI |
| `MemoryAugmented` | Memory-based responses using Memory.Store |
| `FactRetriever` | Factual query handling |
| `SemanticFactRetriever` | TF-IDF fact search (wraps FactDatabase) |
| `Composer` | Multi-part response composition |
| `TemplateBlender` | Novel response generation from template chunks |
| `Synthesizer` | Epistemic responses |

---

### Memory System

The memory system uses a human analogy:

| Module | Human Analogy | Purpose |
|--------|---------------|---------|
| `MemoryStore` | Personality traits | Persona-scoped JSON storage |
| `Memory.Store` | Long-term memory | Day-over-day, month-over-month episodic memory |
| `Memory.Consolidation` | Memory compression | Clusters episodes into semantic facts |
| `Memory.Think` | Conscious memory access | High-level API for memory operations |

#### Memory.Think (`Brain.Memory.Think`)

**Location**: `lib/chat_bot/memory/think.ex`

```elixir
# Add an episode
{:ok, {:episode_added, id}} = Think.think(:add_episode, %{
  state: "User asked about weather",
  action: "conversation",
  outcome: "Provided weather info",
  tags: ["weather", "query"],
  world_id: "default"
})

# Query similar episodes
{:ok, {:chat_results, results}} = Think.think(:query_chat, %{
  input: "What's the weather?",
  k: 5,
  world_id: "default"
})

# Query semantic facts
{:ok, {:semantic_results, results}} = Think.think(:query_semantic, %{
  input: "weather patterns",
  k: 5
})

# Consolidate episodes into semantic facts
{:ok, {:consolidated, count}} = Think.think(:consolidate, %{
  threshold: 0.8,
  min_size: 2
})
```

#### MemoryStore vs Memory.Store

**Important**: These are different systems!

- **MemoryStore** (`Brain.MemoryStore`): Persona-scoped JSON storage - like personality traits
- **Memory.Store** (`Brain.Memory.Store`): Cognitive episodic memory with TF-IDF embeddings - long-term memory

```elixir
# MemoryStore - persona traits
MemoryStore.load_all("Echo")  # Load persona memory
MemoryStore.append_thought("Echo", "user", "Some thought", ["tag"])

# Memory.Store - episodic memory
Memory.Store.add_episode(state, action, outcome, tags, world_id: "default")
Memory.Store.query_similar(text, k, world_id: "default")
```

---

### Learning System

The learning system is world-based - directory-scoped embeddings for specialized domains.

#### WorldContext (`World.Context`)

**Location**: `lib/chat_bot/learning/world_context.ex`

Provides unified API for world-scoped data access with inheritance.

```elixir
# Get inheritance chain
chain = WorldContext.get_inheritance_chain("my_world")
# => ["my_world", "base_world", "default"]

# Lookup entity in world (with inheritance)
{:ok, result, source_world} = WorldContext.lookup_entity("my_world", "Austin")

# Add episode to world
WorldContext.add_episode(world_id, input, action, outcome, tags)
```

#### WorldManager (`World.Manager`)

**Location**: `lib/chat_bot/learning/world_manager.ex`

```elixir
# Create a new training world
{:ok, world} = WorldManager.create("star_trek_world",
  mode: :persistent,
  base_world: "default",
  metadata: %{description: "Star Trek scripts training"}
)

# Get a world
{:ok, world} = WorldManager.get("star_trek_world")

# List all worlds
worlds = WorldManager.list()
```

#### Other Learning Modules

| Module | Purpose |
|--------|---------|
| `EntityDiscoverer` | POS-based entity discovery from text |
| `DocumentIngestor` | Bulk document processing (works with EntityDiscoverer) |
| `TaskAnalyzer` | Analyzes NLP benchmark tasks |
| `TaskTransformer` | Transforms tasks to training format |
| `OutcomeLearner` | Learns from conversation outcomes to create heuristics |
| `TypeInferrer` | Entity type patterns (world-scoped) |
| `WorldModelRegistry` | Per-world ML models |
| `WorldEmbedder` | Per-world embeddings |

---

### Knowledge Expansion

The Knowledge Expansion system implements a teacher/student research model.

**Status**: Experimental - needs summarization system to work properly.

#### LearningCenter (`Brain.Knowledge.LearningCenter`)

**Location**: `lib/chat_bot/knowledge/learning_center.ex`

```elixir
# Start a learning session (admin-triggered)
{:ok, session} = LearningCenter.start_session("European capitals",
  questions: ["What is the capital of France?"],
  sources: [:task]  # Use local task files instead of internet
)

# Start task-based training
{:ok, session} = LearningCenter.start_task_training(:question_answering,
  max_tasks: 10,
  max_instances: 20
)

# Get session status
{:ok, session} = LearningCenter.get_session(session_id)
```

**Teacher/Student Concept**:
- **Students** (ResearchAgent): Gather evidence, research topics
- **Teacher** (Corroborator): Verify factual consistency, test hypotheses
- **Admin** (ReviewQueue): Final review before facts are stored

**Recommendation**: Use local directories/digital copies instead of internet sources for more control and reproducibility.

#### Other Knowledge Modules

| Module | Purpose |
|--------|---------|
| `ResearchAgent` | Student agents for evidence gathering |
| `TaskSource` | Provides tasks to ResearchAgent |
| `Corroborator` | Teacher role - verification, hypothesis testing |
| `ReviewQueue` | Admin review queue |
| `SourceReliability` | Trust scores for sources |
| `HTMLProcessor` | Web content processing |

---

### Epistemic System

The epistemic system provides truth maintenance and belief management. **Always enabled**.

#### BeliefStore (`Brain.Epistemic.BeliefStore`)

**Location**: `lib/chat_bot/epistemic/belief_store.ex`

```elixir
# Add a belief
{:ok, belief_id} = BeliefStore.add_belief(:user, :location, "Austin",
  source: :explicit,
  confidence: 0.85,
  user_id: user_id
)

# Query beliefs
beliefs = BeliefStore.query_beliefs(
  user_id: user_id,
  predicate: :location,
  min_confidence: 0.5
)

# Retract a belief
BeliefStore.retract_belief(belief_id)
```

#### Other Epistemic Modules

| Module | Purpose |
|--------|---------|
| `UserModelStore` | Per-user facts and preferences |
| `JTMS` | Justification-based Truth Maintenance System |
| `ContradictionHandler` | Handles contradictions when detected |
| `DisclosurePolicy` | Decides what to disclose to users |

---

### ML/NLP Components

All ML/NLP components use classical ML - **no regex or string matching**.

#### IntentClassifierSimple (`Brain.ML.IntentClassifierSimple`)

**Location**: `lib/chat_bot/ml/intent_classifier_simple.ex`

Active, world-scoped with fallback to default.

```elixir
# Classify intent
{:ok, %{intent: intent, confidence: confidence}} = 
  IntentClassifierSimple.classify("What's the weather?")

# Load models for a world
IntentClassifierSimple.load_world_model(world_id)
```

**Note**: `IntentClassifier` (without "Simple") is a legacy global-scoped version - use `IntentClassifierSimple` instead.

#### Other ML Modules

| Module | Purpose |
|--------|---------|
| `EntityExtractor` | Entity extraction (pattern recognition, no regex) |
| `Gazetteer` | Fast entity lookup via ETS |
| `Tokenizer` | Text tokenization (**use this instead of regex!**) |
| `POSTagger` | Part-of-speech tagging |
| `NLPPipeline` | Additional NLP processing |
| `InformalExpansions` | Contraction expansion |

#### Tokenizer Best Practices

```elixir
# CORRECT - Use Tokenizer functions
tokens = Tokenizer.tokenize_words(text)
normalized = Tokenizer.tokenize_normalized(text)
is_question = Tokenizer.ends_with_question?(text)

# WRONG - Don't use regex
String.split(text, ~r/\s+/)  # NO!
Regex.match?(~r/\?$/, text)  # NO!
```

---

### Data Stores

| Store | Purpose | Access Pattern |
|-------|---------|----------------|
| `FactDatabase` | Verified facts | Use SemanticFactRetriever for queries |
| `KnowledgeStore` | World knowledge | Direct CRUD |
| `HeuristicStore` | Fast-path patterns | RacingAnalyzer uses this |
| `TemplateStore` | Response templates | Generator uses this |

---

### Subprocesses

Subprocesses act as domain specialists - different "parts of the brain".

**Location**: `lib/chat_bot/subprocesses/`

| Subprocess | Purpose |
|------------|---------|
| `HttpSubprocess` | HTTP API specialist |
| `ConversationSubprocess` | Conversation specialist |
| `CliSubprocess` | CLI specialist |
| `Subprocesses.Supervisor` | Subprocess management |

Brain coordinates but subprocesses operate independently.

---

### Telemetry & Monitoring

#### Dashboard Integration Requirements

**Primary Purpose**: Debugging and insight into which subsystems handle given input.

**Requirements**:
1. Every new subsystem **MUST** emit telemetry
2. Every new telemetry event should have dashboard visibility
3. Dashboard should "light up" when subsystems are used

```elixir
# Emit telemetry span
Brain.Telemetry.span(:my_subsystem_operation, %{input: input}, fn ->
  # Your operation here
  result
end)
```

#### SystemStatus

```elixir
# Get all system statuses
statuses = SystemStatus.get_all()

# Check if all systems are ready
ready? = SystemStatus.all_ready?()
```

---

## Common Use Cases

### Processing User Input

```elixir
# 1. Create a conversation
{:ok, conv_id} = Brain.create_conversation(world_id: "default")

# 2. Evaluate input
{:ok, response} = Brain.evaluate(conv_id, "Hello! What's the weather in NYC?")

# 3. Handle response
IO.puts(response)

# 4. End conversation when done
Brain.end_conversation(conv_id)
```

### Fast Path vs Deep Analysis

The system automatically chooses:
- **Fast Path**: When heuristics or memory have high-confidence match
- **Deep Analysis**: When no confident fast-path match

### Adding New Intents

1. Create intent file in `data/intents/`:
```json
{
  "name": "my_domain.my_intent",
  "userDefined": true,
  "utterances": ["example phrase", "another example"],
  "responses": [{"messages": [{"speech": ["Response here"]}]}]
}
```

2. Define slot schema in `priv/analysis/slot_schemas.json`:
```json
{
  "my_domain.my_intent": {
    "required": ["entity_type"],
    "optional": ["optional_entity"],
    "clarification_templates": {
      "entity_type": "What entity would you like?"
    }
  }
}
```

3. Train models:
```bash
mix train_models
```

### Data Migration & Template Management

The project includes tools for migrating training data and managing response templates.

#### Intent Data Migration

```bash
# List available intents and their sources
mix migrate_gold_standard --list

# Preview what would be migrated
mix migrate_gold_standard --preview

# Run migration (migrate training data to gold_standard.json)
mix migrate_gold_standard

# Destructive migration (also deletes source files after migrating)
mix migrate_gold_standard --destructive

# Extract intent metadata to intent_registry.json
mix migrate_gold_standard --extract-metadata

# Extract response templates to templates.json
mix migrate_gold_standard --extract-templates

# Clean up source directories (after migration is complete)
mix migrate_gold_standard --cleanup-sources
```

#### Response Templates (Runtime CRUD)

Response templates can be managed at runtime via the Settings UI or programmatically:

```elixir
alias Brain.Response.TemplateStore

# Add a new template for an intent
TemplateStore.add_template("smalltalk.greeting", "Hello there!")

# List templates with metadata
TemplateStore.list_templates_with_metadata("smalltalk.greeting")

# Remove a template
TemplateStore.remove_template("smalltalk.greeting", "Hello there!")

# Save changes to file
TemplateStore.sync_to_file()

# Check for unsaved changes
TemplateStore.has_unsaved_changes?()
```

Templates are stored in memory (ETS) for fast access, with periodic syncing to `priv/response/templates.json`.

### Memory Operations

```elixir
# Add episode to memory
Think.think(:add_episode, %{
  state: "User discussed weather",
  action: "query",
  outcome: "Provided forecast",
  tags: ["weather", "forecast"],
  world_id: "default"
})

# Query similar conversations
Think.think(:query_chat, %{input: "weather forecast", k: 5})
```

### World Management

```elixir
# Create specialized world
{:ok, world} = WorldManager.create("support_docs",
  mode: :persistent,
  metadata: %{description: "Customer support training"}
)

# Ingest documents
DocumentIngestor.ingest(world.id, "path/to/docs/*.txt")

# Discover entities
EntityDiscoverer.discover_entities(text, world.id)
```

---

## Pipeline Flows

### Brain.evaluate Flow

```
Brain.evaluate(conv_id, input, opts)
    │
    ├─→ Check: FollowupDetector.is_followup?
    │     └─→ If YES: handle_followup_message()
    │
    ├─→ Check: SelfKnowledgeAnalyzer.is_self_knowledge_query?
    │     └─→ If YES: handle_meta_cognitive_query()
    │
    ├─→ RacingAnalyzer.check_fast_path(text, world_id, user_id, cohort_id)
    │     ├─→ {:fast_path, interpretation} → handle_fast_path_response()
    │     └─→ :no_match → process_standard_message()
    │
    ├─→ Pipeline.process(input, opts)
    │
    ├─→ ResponseGate.evaluate(analysis_model, memory, opts)
    │     ├─→ {:defer, _} → Return nil
    │     └─→ {:respond, _} → proceed_with_standard_response()
    │
    ├─→ Generator.generate(intent, entities, query)
    │
    └─→ Post-processing:
          ├─→ Update conversation memory
          ├─→ Add to learning queue
          ├─→ Extract beliefs (epistemic)
          └─→ OutcomeLearner.learn_from_outcome()
```

### Analysis Pipeline Stages

| Stage | Component | Output |
|-------|-----------|--------|
| 1 | SemanticChunker | List of chunks |
| 2a | DiscourseAnalyzer | Addressee detection (parallel) |
| 2b | SpeechActClassifier | Speech act type (parallel) |
| 2c | AnaphoraResolver | Resolved pronouns |
| 3a | EntityExtractor | Extracted entities |
| 3b | determine_intent | Intent classification |
| 3c | SlotDetector | Filled/missing slots |
| 3d | ContextResolver | Resolved context |
| 4 | calculate_confidence | Confidence score |
| 5 | determine_strategy | Response strategy |

---

## Subsystem Interactions

### Memory System Interactions

```
MemoryStore (persona traits)
    │
    └─→ Brain loads on init for persona personality

Memory.Store (episodic)
    │
    ├─→ RacingAnalyzer queries for fast-path matches
    ├─→ MemoryAugmented queries for response generation
    └─→ Brain stores episodes after conversations

Memory.Consolidation
    │
    └─→ Clusters episodes into semantic facts (background)
```

### World Scoping

```
WorldContext.resolve(world_id, :entity, lookup_fn)
    │
    └─→ Traverses: my_world → base_world → default
          │
          └─→ Returns first successful match
```

---

## Testing & Debugging

### Running Tests

```bash
# Run all tests
mix test

# Run specific file
mix test test/chat_bot/feature_test.exs

# Run failed tests
mix test --failed

# Full validation
mix precommit
```

### Debugging Pipeline Execution

Enable debug logging in `config/dev.exs`:
```elixir
config :logger, level: :debug
```

Look for these log messages:
- `"Starting analysis pipeline"` - Pipeline entry
- `"Chunking complete"` - After Stage 1
- `"Pipeline complete"` - All stages done

### Telemetry Spans

The system emits these telemetry spans:
- `:pipeline_process`
- `:racing_analysis`
- `:brain_evaluate`
- `:belief_operation`
- `:jtms_justify`
- `:code_pipeline` (code analysis)
- `:code_parse` (code parsing)
- `:code_extract` (symbol extraction)
- `:code_gazetteer_lookup` (code symbol lookups)
- `:code_gazetteer_add` (code symbol additions)

### Dashboard Visibility

All subsystems should be visible in the dashboard when used. If you add a new subsystem:
1. Add telemetry spans
2. Update dashboard to show the new subsystem
3. The dashboard should "light up" when your subsystem is used

---

## Development Workflow

### Setup

```bash
mix setup
mix phx.server
```

### Training Models

```bash
# Full training
mix train_models

# Intent only
mix train_models --intent-only

# Gazetteer only
mix train_models --gazetteer-only
```

### World Creation

```bash
# Create training world
mix training_world.create "my_world"

# Ingest documents
mix training_world.ingest "WORLD_ID" "path/to/*.txt"

# View metrics
mix training_world.metrics "WORLD_ID"
```

### Adding Telemetry

When adding new subsystems:

```elixir
defmodule MySubsystem do
  def process(input) do
    Brain.Telemetry.span(:my_subsystem, %{input_length: String.length(input)}, fn ->
      # Your processing here
      result
    end)
  end
end
```

### Common Tasks

| Task | Command |
|------|---------|
| Format code | `mix format` |
| Run tests | `mix test` |
| Full validation | `mix precommit` |
| Clear knowledge | `mix clear_knowledge` |
| Train models | `mix train_models` |
| Create world | `mix training_world.create "name"` |

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Visual architecture guide
- [PIPELINE_ORDER.md](PIPELINE_ORDER.md) - Detailed pipeline execution order
- [SUBSYSTEM_INTEGRATION_REVIEW.md](SUBSYSTEM_INTEGRATION_REVIEW.md) - Disconnected subsystems and scoping evolution
- [WRITING_TESTS.md](WRITING_TESTS.md) - Testing guide
