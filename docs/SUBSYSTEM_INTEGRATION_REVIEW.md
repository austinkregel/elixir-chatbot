# Subsystem Integration Review

This document identifies subsystems that may be disconnected, experimental, or need integration review. It also documents the scoping evolution of the codebase.

## Table of Contents

- [Scoping Evolution](#scoping-evolution)
- [Disconnected Subsystems](#disconnected-subsystems)
  - [Summarization Subsystem](#summarization-subsystem)
  - [IntentClassifier (Legacy)](#intentclassifier-legacy)
  - [BacktrackController](#backtrackcontroller)
  - [Mailer](#mailer)
  - [ProcessingTrace](#processingtrace)
  - [Knowledge Expansion System](#knowledge-expansion-system)
  - [ParallelPipeline](#parallelpipeline)
- [Migration Guide](#migration-guide)
- [Recommendations](#recommendations)

---

## Scoping Evolution

The codebase has evolved through three scoping paradigms, with world-scoping being the current standard.

### Three Scoping Types

#### 1. Global (Pre-Scoping Era)

Single model/data for all users.

| Aspect | Details |
|--------|---------|
| **Examples** | Original `IntentClassifier` (legacy, appears unused) |
| **Status** | Deprecated or replaced with scoped versions |
| **Migration** | These were replaced with scoped versions |

**Characteristics**:
- One model serves all users
- No isolation between conversations or domains
- Simpler but less flexible

#### 2. Persona-Scoped

Per-character/bot personality.

| Aspect | Details |
|--------|---------|
| **Example** | `MemoryStore` (persona-scoped JSON storage) |
| **Purpose** | Personality traits, character-specific memory |
| **Note** | Different from world-scoped - serves different purpose |

**Characteristics**:
- Data isolated per persona/character
- Used for personality traits
- Persisted as JSON files per persona

#### 3. World-Scoped (Current Standard)

Per-training-world isolation with directory-scoped embeddings.

| Aspect | Details |
|--------|---------|
| **Learning System** | Fully world-scoped: `WorldContext`, `WorldManager`, `WorldEmbedder`, `WorldModelRegistry` |
| **Memory.Store** | Episodes/semantics stored per-world |
| **IntentClassifierSimple** | World-scoped with fallback to default |
| **HeuristicStore** | World-scoped, used by RacingAnalyzer |
| **RacingAnalyzer.check_fast_path/4** | Accepts world_id parameter |
| **Purpose** | Specialized domains, subject matter experts, isolated training environments |

**Characteristics**:
- Directory-scoped embeddings
- World activation/isolation
- Enables specialized training (e.g., Star Trek scripts → fantasy world)
- Subject matter experts per world
- Inheritance chain: child → base → default

### Migration Path

```
Global → User-Scoped → World-Scoped
```

World-scoping was added systematically to enable:
- Directory-scoped embeddings
- World activation/isolation
- Specialized training
- Subject matter experts per world

### World-Scoping Implementation Status

**Fully World-Scoped:**
- Learning System (`WorldContext`, `WorldManager`, `WorldEmbedder`, `WorldModelRegistry`)
- `Memory.Store` (episodes/semantics)
- `IntentClassifierSimple` (with fallback to default)
- `HeuristicStore` (used by RacingAnalyzer)
- `RacingAnalyzer.check_fast_path/4`

**Partially World-Scoped:**
- Some components may have world_id parameters but fallback to default
- Document fallback behavior when adding new features

**Not Yet World-Scoped (Different Purpose):**
- `MemoryStore` (persona-scoped, different purpose - personality traits)
- Some legacy components

### Deprecated APIs

These APIs still exist for backward compatibility but log deprecation warnings:

#### RacingAnalyzer.check_fast_path/3

| Aspect | Details |
|--------|---------|
| **Location** | `lib/chat_bot/analysis/racing_analyzer.ex:189` |
| **Deprecation** | Logs warning, calls `check_fast_path/4` with "default" world_id |
| **Migration** | Use `check_fast_path(text, world_id, user_id, cohort_id)` |

```elixir
# Deprecated
RacingAnalyzer.check_fast_path(text, user_id, cohort_id)

# Current
RacingAnalyzer.check_fast_path(text, world_id, user_id, cohort_id)
```

#### Other Deprecated APIs

When adding new features, check for similar patterns:
- Look for `/3` functions that have `/4` counterparts with world_id
- Check for deprecation warnings in logs
- Document migration path for each

### Scoping Best Practices

1. **Always pass world_id** when available in new code
2. **Use "default" world_id** for backward compatibility in deprecated APIs
3. **Document world isolation benefits** in module docs
4. **Explain directory-scoped embeddings** in world management docs
5. **Clarify persona-scoped vs world-scoped** when documenting memory systems

---

## Disconnected Subsystems

### Summarization Subsystem

| Aspect | Details |
|--------|---------|
| **Status** | Offline until ML-based generation ready |
| **Location** | `lib/chat_bot/summarization/` |
| **Reason** | Current system doesn't produce coherent summaries |

**Modules:**
- `Summarization.Pipeline` - Main summarization pipeline
- `EventExtractor` - Extracts events from text
- `FactRanker` - Ranks facts by importance
- `ReferenceResolver` - Resolves references
- `SpeakerTracker` - Tracks speakers in dialogue
- `SummaryBuilder` - Builds summary output
- `TurnSegmenter` - Segments conversation turns
- `Types` - Type definitions

**Current State:**
- Not integrated with Brain or conversation flow
- Needs ML-based generative summaries
- Research needed into ML-based generation approaches

**Dependencies:**
- Knowledge Expansion system depends on this for A/B experimentation
- Without summarization, Knowledge Expansion cannot properly condense research findings

**Recommendation:**
- Document as future feature
- Research ML-based summarization approaches that don't require LLMs
- Consider keyword/association-based compression (similar to Memory.Consolidation)

---

### IntentClassifier (Legacy)

| Aspect | Details |
|--------|---------|
| **Status** | Appears unused, likely global-scoped version |
| **Location** | `lib/chat_bot/ml/intent_classifier.ex` |
| **Current State** | `IntentClassifierSimple` is active (world-scoped with fallback) |

**Scoping Evolution:**
This is likely the pre-world-scoping version (global-scoped). It was replaced by `IntentClassifierSimple` which supports world-scoping.

**Key Differences:**

| Aspect | IntentClassifier | IntentClassifierSimple |
|--------|------------------|------------------------|
| Scoping | Global | World-scoped with fallback |
| Status | Legacy/Unused | Active |
| World Support | No | Yes |

**Recommendation:**
- Verify if deprecated or if it serves a specific purpose
- Document as legacy/global-scoped version
- Note that `IntentClassifierSimple` is the active, world-scoped implementation
- Consider removal if confirmed unused

---

### BacktrackController

| Aspect | Details |
|--------|---------|
| **Status** | Unclear usage |
| **Location** | `lib/chat_bot/analysis/backtrack_controller.ex` |
| **Purpose** | Alternative analysis paths |

**Intended Purpose:**
The BacktrackController is designed to handle alternative analysis paths when the primary path fails or produces low-confidence results.

**Current State:**
- Unclear if actively used in production flow
- May be experimental or for future use

**Recommendation:**
- Verify usage in the codebase
- If used, document the triggering conditions
- If unused, mark as experimental or remove

---

### Mailer

| Aspect | Details |
|--------|---------|
| **Status** | Unused |
| **Location** | `lib/chat_bot/mailer.ex` |

**Current State:**
- Phoenix-generated module
- Not integrated with any subsystem
- No email sending functionality implemented

**Recommendation:**
- Remove if email functionality not planned
- Or document as future feature placeholder

---

### ProcessingTrace

| Aspect | Details |
|--------|---------|
| **Status** | Debugging tool only |
| **Location** | `lib/chat_bot/analysis/processing_trace.ex` |
| **Purpose** | Trace processing steps for debugging |

**Current State:**
- Used for debugging pipeline execution
- Not a production feature

**Recommendation:**
- Document as debugging tool
- Clarify that it's not for production use
- Consider integration with dashboard for visual debugging

---

### Knowledge Expansion System

| Aspect | Details |
|--------|---------|
| **Status** | Too ambitious, needs summarization |
| **Current State** | Teacher/student system, admin-triggered |
| **Location** | `lib/chat_bot/knowledge/` |

**Original Concept:**
- **Students** (ResearchAgent): Research internet sources
- **Teacher** (Corroborator): Verify factual consistency, run through full analysis
- **Corroboration**: Multiple students must agree on findings
- **Admin Review**: Final approval before facts are stored

**Current Limitations:**

1. **Summarization Dependency**: Needs summarization to work properly
   - System depends on A/B experimentation capability
   - A/B experimentation requires summarization for comparing results

2. **Internet Source Reliability**: Original internet-based approach may be too unreliable
   - Web content varies in quality
   - Rate limiting and network issues
   - Difficult to reproduce

3. **Better Approach**: Limit to local directories/digital copies on disk
   - More control over content quality
   - Reproducible training
   - No network dependencies

**Recommendation:**
- Document current limitations and experimental status
- Emphasize that local directory approach is preferred over internet sources
- Note dependency on summarization system for A/B experimentation
- Clarify that it's too ambitious for current state but has potential when summarization is ready

---

### ParallelPipeline

| Aspect | Details |
|--------|---------|
| **Status** | Future replacement |
| **Location** | `lib/chat_bot/analysis/parallel_pipeline.ex` |
| **Current State** | Not active (Pipeline is used) |

**Intended Purpose:**
- Parallel execution of analysis stages
- Performance optimization
- May replace `Pipeline` in the future

**Current State:**
- Not integrated with Brain
- `Pipeline` is the active implementation
- May be experimental or in development

**Recommendation:**
- Document as future/experimental feature
- Note migration plan if applicable
- Clarify relationship with current `Pipeline`

---

## Migration Guide

### Migrating to World-Scoped APIs

When migrating from global to world-scoped APIs:

1. **Identify the global API:**
   ```elixir
   # Old global API
   SomeModule.some_function(text, user_id)
   ```

2. **Find the world-scoped version:**
   ```elixir
   # New world-scoped API
   SomeModule.some_function(text, world_id, user_id)
   ```

3. **Use "default" for backward compatibility:**
   ```elixir
   # If world_id not available
   SomeModule.some_function(text, "default", user_id)
   ```

4. **Update callers to pass world_id:**
   ```elixir
   # Pass world_id from conversation context
   world_id = conversation.world_id || "default"
   SomeModule.some_function(text, world_id, user_id)
   ```

### Deprecation Pattern

When deprecating a global API:

```elixir
# In your module
def some_function(text, user_id) do
  Logger.warning("some_function/2 is deprecated, use some_function/3 with world_id")
  some_function(text, "default", user_id)
end

def some_function(text, world_id, user_id) do
  # World-scoped implementation
end
```

---

## Recommendations

### Immediate Actions

1. **Verify IntentClassifier usage** - Confirm if it's deprecated and can be removed
2. **Document BacktrackController** - Clarify if it's used and under what conditions
3. **Remove or document Mailer** - Either remove or note as future feature
4. **Add telemetry to ProcessingTrace** - Integrate with dashboard for visual debugging

### Medium-Term Actions

1. **Research ML-based summarization** - Required for Knowledge Expansion to work
2. **Evaluate ParallelPipeline** - Decide on migration timeline if performance benefits are significant
3. **Document all deprecated APIs** - Create a deprecation list with migration paths

### Long-Term Actions

1. **Complete summarization system** - Enable Knowledge Expansion A/B experimentation
2. **Migrate Knowledge Expansion to local directories** - More reliable than internet sources
3. **Complete world-scoping migration** - Ensure all relevant modules are world-scoped

### Best Practices for New Subsystems

1. **Always add world_id parameter** when data is world-specific
2. **Emit telemetry** for dashboard visibility
3. **Document scoping** - Is it global, persona-scoped, or world-scoped?
4. **Add deprecation warnings** when replacing old APIs
5. **Provide migration examples** in documentation

---

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Main contributor guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Visual architecture guide
- [PIPELINE_ORDER.md](PIPELINE_ORDER.md) - Detailed pipeline execution order
