# Pipeline Execution Order - Quick Reference

This document shows the exact execution order of the ChatBot pipeline, verified from the source code.

## Executive Summary

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
│ 3b'. filter_entities │ → Keep only relevant
│ 3c. SlotDetector    │ → Fill slot schema
│ 3d. ContextResolver │ → Resolve from history
│ 4. calculate_confidence │
│ 5. determine_response_strategy │
│ 6. InternalModel.determine_strategy │
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│ ResponseGate.evaluate│ ─── Should we respond?
└─────────┬───────────┘
          │ (if :respond or :optional < threshold)
          ▼
┌─────────────────────┐
│ Generator.generate  │ ─── Create response
└─────────┬───────────┘
          │
          ▼
     Response
          │
          ▼
┌─────────────────────┐
│ Post-Processing     │ ─── Learning & Storage (async)
├─────────────────────┤
│ • build_context_snapshot │
│ • Update conversation memory │
│ • Add to learning_queue │
│ • extract_and_store_beliefs │ → BeliefStore, UserModelStore
│ • OutcomeLearner.learn │ → HeuristicStore, AnalyzerCalibration
│ • Store in cognitive memory │ → WorldContext.add_episode
│ • Learner.learn_from_extraction │ → KnowledgeStore
└─────────────────────┘
```

## All Subsystems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CHATBOT SYSTEM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        CORE PROCESSING                               │    │
│  │  ┌─────────┐    ┌──────────┐    ┌────────────┐    ┌───────────┐     │    │
│  │  │  Brain  │───▶│ Pipeline │───▶│ResponseGate│───▶│ Generator │     │    │
│  │  └────┬────┘    └────┬─────┘    └────────────┘    └───────────┘     │    │
│  │       │              │                                               │    │
│  │       │    ┌─────────┴──────────────────────────────┐               │    │
│  │       │    │ SemanticChunker → DiscourseAnalyzer    │               │    │
│  │       │    │ SpeechActClassifier → AnaphoraResolver │               │    │
│  │       │    │ EntityExtractor → SlotDetector         │               │    │
│  │       │    │ ContextResolver → determine_strategy   │               │    │
│  │       │    └────────────────────────────────────────┘               │    │
│  │       │                                                              │    │
│  │       ▼                                                              │    │
│  │  ┌─────────────────────────────────┐                                │    │
│  │  │        RACING ANALYZER          │                                │    │
│  │  │  HeuristicStore │ MemoryStore  │                                │    │
│  │  │  PatternMatch   │ ModelAnalyzer │                                │    │
│  │  └─────────────────────────────────┘                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│  │  EPISTEMIC       │  │  MEMORY          │  │  LEARNING              │    │
│  │  ───────────     │  │  ─────────       │  │  ────────              │    │
│  │  JTMS            │  │  Embedder        │  │  WorldManager          │    │
│  │  BeliefStore     │  │  Store           │  │  WorldContext          │    │
│  │  UserModelStore  │  │  VectorIndex     │  │  EntityDiscoverer      │    │
│  │  Contradiction   │  │  Consolidation   │  │  OutcomeLearner        │    │
│  │  Handler         │  │  Think API       │  │  HeuristicStore        │    │
│  │  DisclosurePolicy│  │                  │  │  AnalyzerCalibration   │    │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘    │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│  │  ML COMPONENTS   │  │  KNOWLEDGE       │  │  RESPONSE              │    │
│  │  ─────────────   │  │  EXPANSION       │  │  ─────────             │    │
│  │  Tokenizer       │  │  ────────────    │  │  TemplateStore         │    │
│  │  POSTagger       │  │  LearningCenter  │  │  FactRetriever         │    │
│  │  Gazetteer       │  │  ResearchAgent   │  │  MemoryAugmented       │    │
│  │  IntentClassifier│  │  Corroborator    │  │  Composer              │    │
│  │  Simple (Active) │  │                  │  │                        │    │
│  │  EntityExtractor │  │  SourceReliability│ │  Synthesizer           │    │
│  │  NLPPipeline     │  │  ReviewQueue     │  │                        │    │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA STORES                                  │    │
│  │  KnowledgeStore │ FactDatabase │ MemoryStore │ IntentRegistry       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Stage Breakdown

### Stage 0: Brain Entry Point

```
Brain.evaluate(conversation_id, input, opts)
  │
  ├─→ Find conversation in state.active_conversations
  │
  ├─→ Check: FollowupDetector.is_followup?(input, previous_context)
  │     └─→ If YES: handle_followup_message() → skip pipeline
  │
  ├─→ Check: SelfKnowledgeAnalyzer.is_self_knowledge_query?(input)
  │     └─→ If YES: handle_meta_cognitive_query() → skip pipeline
  │
  ├─→ Check: RacingAnalyzer.check_fast_path(input, world_id, user_id, cohort_id)
  │     ├─→ HeuristicStore.match_best() → if confidence >= 0.85 → fast path
  │     └─→ MemoryStore.query_similar() → if similarity >= 0.85 → fast path
  │     NOTE: check_fast_path/3 (without world_id) is deprecated
  │
  └─→ If no fast path: process_standard_message()
```

### Stage 1: Semantic Chunking

```elixir
# SemanticChunker.chunk(text)
# Location: lib/chat_bot/analysis/semantic_chunker.ex

text
  |> normalize_whitespace()           # Collapse whitespace
  |> extract_quoted_sections()        # Preserve quoted text
  |> split_into_sentences()           # Split on . ! ?
  |> merge_short_sentences(min_words) # Merge < 3 word sentences
  |> detect_discourse_markers()       # Find "but", "and", "also"
  |> apply_max_chunk_size(max_words)  # Split > 50 word chunks
  |> build_chunks(original_text)      # Create Chunk structs
```

**Output**: List of `%Chunk{text, index, start_pos, end_pos, is_quoted, discourse_markers}`

### Stage 2a & 2b: Parallel Analysis

```elixir
# These run in PARALLEL via Task.async
# Location: lib/chat_bot/analysis/pipeline.ex:192-224

discourse_task = Task.async(fn ->
  DiscourseAnalyzer.analyze(chunk.text, participants: [:user, :bot])
end)

speech_act_task = Task.async(fn ->
  SpeechActClassifier.classify(chunk.text)
end)

discourse_result = Task.await(discourse_task, 3000)
speech_act_result = Task.await(speech_act_task, 3000)
```

**DiscourseAnalyzer Output**: `%DiscourseResult{addressee: :bot/:user/:third_party, confidence, indicators}`

**SpeechActClassifier Output**: `%SpeechActResult{category, sub_type, confidence, is_question, is_imperative, indicators}`

### Stage 2c: Anaphora Resolution

```elixir
# AnaphoraResolver.resolve_and_substitute(text, history)
# Location: lib/chat_bot/analysis/anaphora_resolver.ex

# Resolves pronouns like "it", "there", "that" to entities from conversation history
{resolved_text, anaphora_entities} = resolve_anaphora(chunk.text, history)
```

**Output**: Tuple of `{resolved_text, list_of_resolved_entities}`

### Stage 3a: Entity Extraction

```elixir
# EntityExtractor.extract_entities(text, opts)
# Location: lib/chat_bot/ml/entity_extractor.ex

# Opts include discourse and speech_act context for disambiguation
entities = EntityExtractor.extract_entities(resolved_text, [
  discourse: discourse_result,
  speech_act: speech_act_result
])

# Merge with anaphora-resolved entities
entities = merge_anaphora_entities(entities, anaphora_entities)
```

**Output**: List of `%{entity_type, value, confidence, source}`

### Stage 3b: Intent Determination

```elixir
# Location: lib/chat_bot/analysis/pipeline.ex:374-398

{intent, intent_method, intent_confidence} = determine_intent(speech_act_result, entities, chunk.text)

# Intent sources (in order of preference):
# 1. Classifier intent from speech_act.indicators (e.g., "intent:weather.query")
# 2. Inferred from speech_act category/sub_type (e.g., :greeting → "smalltalk.greeting")
```

### Stage 3b': Entity Filtering

```elixir
# filter_entities_by_intent(entities, intent)
# Location: lib/chat_bot/analysis/pipeline.ex:591-623

# CRITICAL: Filters entities to only those relevant to the intent's slot schema
# Prevents cross-intent contamination (e.g., "Austin" as person in greeting
# leaking to "Austin" as location for weather query)

relevant_entities = filter_entities_by_intent(entities, intent)
```

### Stage 3c: Slot Detection

```elixir
# SlotDetector.detect(intent, entities)
# Location: lib/chat_bot/analysis/slot_detector.ex

slot_result = SlotDetector.detect(intent, relevant_entities)

# Returns: %SlotResult{
#   intent: "weather.query",
#   filled_slots: %{"location" => %{value: "NYC", source: :explicit}},
#   missing_required: [],
#   missing_optional: ["date"],
#   all_required_filled: true
# }
```

### Stage 3d: Context Resolution

```elixir
# ContextResolver.resolve(slot_result, opts)
# Location: lib/chat_bot/analysis/context_resolver.ex

# Tries to fill missing slots from:
# 1. Conversation history (recent mentions)
# 2. User model (epistemic system)
# 3. User profile (stored preferences)

resolved_slots = ContextResolver.resolve(slot_result,
  conversation_history: history,
  user_profile: profile,
  user_id: user_id
)
```

### Stage 4 & 5: Confidence & Strategy

```elixir
# calculate_confidence(analysis)
# Location: lib/chat_bot/analysis/pipeline.ex:466-504

# Weighted average: discourse (0.3) + speech_act (0.4) + slots (0.3)
confidence = (discourse_conf * 0.3 + speech_act_conf * 0.4 + slot_conf * 0.3)

# ChunkAnalysis.determine_response_strategy(analysis)
# Location: lib/chat_bot/analysis/internal_model.ex

# Returns one of:
# - :can_respond (all required slots filled)
# - :needs_clarification (missing required slots)
# - :defer_to_user (not addressed to bot)
# - :cannot_respond (unknown intent)
```

### Stage 6: Overall Strategy

```elixir
# InternalModel.determine_strategy(model)
# Location: lib/chat_bot/analysis/internal_model.ex

# Combines per-chunk strategies into overall strategy:
# - All :can_respond → :can_respond
# - Mix of :can_respond + :needs_clarification → :partial_response_with_clarification
# - All :defer_to_user → :defer_to_user
# - All :cannot_respond → :cannot_respond
# - Any missing slots → :needs_clarification
```

## Post-Pipeline Processing

### ResponseGate Evaluation

```elixir
# ResponseGate.evaluate(analysis_model, memory, opts)
# Location: lib/chat_bot/analysis/response_gate.ex

case ResponseGate.evaluate(analysis_model, memory, opts) do
  {:respond, reason} ->
    # Normal response generation
    proceed_with_standard_response(...)

  {:optional, confidence, reason} ->
    # Check threshold, may defer
    if confidence >= defer_threshold, do: return nil

  {:defer, reason} ->
    # Don't respond (gratitude loop, continuation, etc.)
    return {nil, :response_deferred, context}
end
```

### Response Generation Priority

```elixir
# Generator.generate(intent, entities, query_text)
# Location: lib/chat_bot/response/generator.ex

# Priority order:
# 1. Domain-specific handlers (weather, music, device, news, reminder, factual)
# 2. Memory-augmented (MemoryAugmented.generate)
# 3. Template-based (TemplateStore.get_random_template)
# 4. Fallback (generic acknowledgment)
```

## Verification Points

To verify the pipeline is running in the expected order, check these logs:

```elixir
# Enable debug logging in config/dev.exs:
config :logger, level: :debug

# Look for these log messages in order:
# "Starting analysis pipeline"       → Pipeline.process entry
# "Chunking complete"                → After Stage 1
# Progress.report :chunk_start       → Per-chunk analysis starts
# Progress.report :discourse_complete → Stage 2a done
# Progress.report :speech_act_complete → Stage 2b done
# Progress.report :anaphora_resolved → Stage 2c done (if any)
# Progress.report :entities_extracted → Stage 3a done
# Progress.report :intent_determined → Stage 3b done
# Progress.report :entities_filtered → Stage 3b' done
# Progress.report :slots_detected    → Stage 3c done
# Progress.report :context_resolved  → Stage 3d done
# Progress.report :chunk_complete    → Chunk analysis done
# "Pipeline complete"                → All stages done
```

## Common Execution Paths

### Simple Greeting ("Hello!")

```
1. SemanticChunker → 1 chunk: "Hello!"
2a. DiscourseAnalyzer → addressee: :bot (in 1-on-1)
2b. SpeechActClassifier → category: :expressive, sub_type: :greeting
3a. EntityExtractor → [] (no entities)
3b. determine_intent → "smalltalk.greeting"
3c. SlotDetector → no required slots for greeting
4. confidence → high
5. strategy → :can_respond
6. overall → :can_respond
→ ResponseGate → :respond
→ Generator → expressive greeting template
```

### Weather Query ("What's the weather in NYC?")

```
1. SemanticChunker → 1 chunk
2a. DiscourseAnalyzer → addressee: :bot
2b. SpeechActClassifier → category: :directive, sub_type: :request_information
3a. EntityExtractor → [{entity_type: "location", value: "NYC"}]
3b. determine_intent → "weather.query"
3b'. filter_entities → keep location (matches slot schema)
3c. SlotDetector → location filled, all_required_filled: true
4. confidence → high
5. strategy → :can_respond
→ ResponseGate → :respond (directive)
→ Generator → domain handler → weather response
```

### Multi-Chunk ("Hello! What's the weather in Dallas?")

```
1. SemanticChunker → 2 chunks: ["Hello!", "What's the weather in Dallas?"]

Chunk 0:
2a/2b. → expressive, greeting
3b. → "smalltalk.greeting"
3b'. → no entities (person names filtered out for greeting)
5. → :can_respond

Chunk 1:
2a/2b. → directive, request_information
3a. → [{location: "Dallas"}]
3b. → "weather.query"
3b'. → keep location
3c. → location filled
5. → :can_respond

6. overall → :can_respond (both chunks can respond)
→ ResponseGate → :respond
→ Generator → combines greeting + weather response
```

### Weather Without Location ("What's the weather?")

```
1. SemanticChunker → 1 chunk
2b. SpeechActClassifier → :directive
3a. EntityExtractor → [] (no location)
3b. determine_intent → "weather.query"
3c. SlotDetector → missing_required: ["location"]
3d. ContextResolver → check history, user model, profile (no match)
4. confidence → medium (missing slot)
5. strategy → :needs_clarification
→ ResponseGate → :respond (directive)
→ Brain → build_clarification_response
→ "What location would you like the weather for?"
```

## Post-Processing: Learning & Storage

After a response is generated, the Brain performs these async operations:

### 1. Context Storage
```elixir
# Build context snapshot for conversation memory
context_snapshot = %{
  intent: intent,
  entities: entities,
  slots: filled_slots,
  missing_slots: missing,
  speech_act: %{category, sub_type, confidence, is_question}
}

# Update conversation memory
updated_conversation = Map.put(conversation, :memory, memory ++ [user_msg, assistant_msg])
```

### 2. Learning Queue
```elixir
# Add to learning queue (processed async)
learning_entry = %{
  conversation_id: conv_id,
  world_id: world_id,
  timestamp: now,
  input: input,
  response: response
}

# Processed by handle_info(:process_learning_queue, ...)
```

### 3. Epistemic Integration
```elixir
# Extract beliefs from entities
Brain.extract_and_store_beliefs(input, entities, user_id, conversation_id)

# For each entity of user-fact type (location, name, preference, etc.):
#   1. Create Belief struct
#   2. Add to BeliefStore
#   3. Update UserModelStore

# Also extracts self-referential facts from patterns like:
#   "I'm from X" → location fact
#   "My name is X" → name fact
#   "I work at X" → workplace fact
#   "I like X" → preference fact
```

### 4. Outcome Learning
```elixir
# Learn from this interaction (Task.start - non-blocking)
Task.start(fn ->
  interpretation = build_interpretation_from_context(input, context)
  
  OutcomeLearner.learn_from_outcome(interpretation, response,
    user_id: user_id,
    cohort_id: cohort_id
  )
end)

# OutcomeLearner:
#   1. assess_outcome → :success/:failure/:uncertain
#   2. Update heuristic stats (if triggered by heuristic)
#   3. Update AnalyzerCalibration
#   4. Maybe create new heuristic from successful slow-path
#   5. Track pattern for future heuristic creation
```

### 5. Cognitive Memory
```elixir
# Store in cognitive memory system
WorldContext.add_episode(
  world_id,
  input,           # state
  "conversation",  # action
  response,        # outcome
  ["conversation", conv_id | tags]
)

# Episode gets embedded (TF-IDF vector) for similarity search
```

### 6. Learner Module
```elixir
# Extract and store entities in KnowledgeStore
Learner.learn_from_classical_extraction(persona.name, entities, input)

# For each entity type:
#   person → KnowledgeStore.add_person
#   room → KnowledgeStore.add_room
#   device → KnowledgeStore.add_device
#   etc.
```

## Background Processes

These processes run continuously in the background:

### Memory Consolidation
```elixir
# Periodic consolidation of episodic memory
Consolidation.consolidate(threshold: 0.8, min_cluster_size: 2)

# Clusters similar episodes into SemanticFacts
# Reduces memory size while preserving knowledge
```

### Knowledge Expansion (On-Demand)
```elixir
# Start a learning session (admin-triggered or scheduled)
LearningCenter.start_session("European capitals")

# Flow:
#   1. Decompose topic into research goals
#   2. Dispatch ResearchAgent tasks
#   3. Fetch web content
#   4. Extract claims via Pipeline
#   5. Corroborate across sources
#   6. Check for contradictions with BeliefStore
#   7. Add to ReviewQueue for admin approval
#   8. Approved → BeliefStore + FactDatabase
```

### Entity Discovery (Per-World)
```elixir
# Discover unknown entities in text
EntityDiscoverer.discover_entities(text, world_id)

# Uses POS tagger to find proper nouns (PROPN)
# Checks Gazetteer for known types
# Unknown → candidates pool
# Admin promotes → Gazetteer overlay
```

## JTMS (Truth Maintenance)

The epistemic system uses a JTMS for belief management:

```
Belief Added
    │
    ▼
BeliefStore.add_belief(belief)
    │
    ▼
JTMS.create_node(belief.id, :assumption)
    │
    ▼
JTMS.justify_node([premise_ids], conclusion_id, "inference_rule")
    │
    ▼
Label Propagation (IN/OUT)
    │
    ├─→ Contradiction Node becomes IN?
    │       └─→ ContradictionHandler triggered
    │           └─→ Find supporting assumptions
    │               └─→ Retract weakest assumption
    │
    └─→ Normal operation continues
```

### JTMS Node Types:
- **Premise**: Always IN (base facts)
- **Assumption**: Can be enabled/retracted
- **Derived**: IN if any valid justification
- **Contradiction**: Triggers handler when IN

---

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Main contributor guide with module API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) - Visual architecture diagrams
- [SUBSYSTEM_INTEGRATION_REVIEW.md](SUBSYSTEM_INTEGRATION_REVIEW.md) - Scoping evolution and disconnected subsystems
- [WRITING_TESTS.md](WRITING_TESTS.md) - Testing guide
