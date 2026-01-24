# Fact Database System

## Overview

The Fact Database system provides a curated collection of verifiable, general knowledge facts that can be used for testing and knowledge building. The system has been enhanced to:

1. **Expand and grow dynamically** as the system learns
2. **Integrate with the epistemic user model** for unified knowledge management
3. **Support truth maintenance verification** through the JTMS system

## Architecture

### Core Components

- **FactDatabase** (`lib/chat_bot/fact_database.ex`): Main GenServer managing fact storage and queries
- **FactDatabase.Integration** (`lib/chat_bot/fact_database/integration.ex`): Bridge to epistemic system
- **FactRetriever** (`lib/chat_bot/response/fact_retriever.ex`): Helper for response generation

### Fact Storage

Facts are stored in JSON files under `data/facts/`:
- `geography.json` - Geographic facts (capitals, landmarks, etc.)
- `science.json` - Scientific facts (physics, chemistry, biology)
- `history.json` - Historical facts and dates
- `general.json` - General knowledge facts
- `learned.json` - Dynamically learned facts (auto-generated)

## Features

### 1. Dynamic Expansion

The fact database can grow as the system learns:

```elixir
# Add a new learned fact
ChatBot.FactDatabase.Integration.add_fact(
  "water",
  "Water is composed of two hydrogen atoms and one oxygen atom",
  category: "science",
  confidence: 0.9,
  verification_source: "chemistry_textbook"
)
```

**Integration with Learner:**
- When the `Learner` module extracts facts from conversations, it automatically:
  - Verifies facts against existing beliefs
  - Adds verified facts to the database
  - Registers them with the truth maintenance system
  - Creates beliefs for tracking

### 2. Epistemic User Model Integration

Facts are integrated with the epistemic system:

```elixir
# Sync all facts to beliefs
ChatBot.FactDatabase.Integration.sync_facts_to_beliefs()

# Verify a fact against existing beliefs
case ChatBot.FactDatabase.Integration.verify_fact("France", "The capital is Paris") do
  {:verified, confidence} -> # Fact is consistent
  {:contradicted, conflicts} -> # Fact contradicts existing beliefs
  {:uncertain, reason} -> # Cannot verify
end
```

**Belief Creation:**
- High-confidence facts (≥0.9) are registered as **premises** in JTMS
- Lower-confidence facts are registered as **assumptions**
- Facts are linked to beliefs for tracking and verification

### 3. Truth Maintenance Verification

Facts can be verified and checked for contradictions:

```elixir
# Check for contradictions
case ChatBot.FactDatabase.Integration.check_contradiction("France", "The capital is London") do
  {:contradiction, conflicting_beliefs} -> # Contradiction detected
  :consistent -> # No contradiction
  :no_data -> # No existing data to check against
end
```

**JTMS Integration:**
- Facts are registered as JTMS nodes
- Contradictions are automatically detected
- The `ContradictionHandler` can resolve conflicts
- Facts can be retracted if contradicted

## Usage Examples

### Querying Facts

```elixir
# Get facts about an entity
ChatBot.FactDatabase.get_entity_facts("France")

# Search facts by keyword
ChatBot.FactDatabase.query(search: "capital", limit: 5)

# Get facts by category
ChatBot.FactDatabase.get_category_facts("geography")
```

### Adding Learned Facts

```elixir
# The Learner automatically adds facts when learning from conversations
# But you can also add facts manually:

ChatBot.FactDatabase.Integration.add_fact(
  "Einstein",
  "Albert Einstein developed the theory of relativity",
  category: "science",
  confidence: 0.95,
  verification_source: "historical_records",
  register_with_jtms: true,
  create_belief: true
)
```

### Verifying Facts

```elixir
# Before adding a fact, verify it
case ChatBot.FactDatabase.Integration.verify_fact("water", "Water boils at 100C") do
  {:verified, conf} when conf >= 0.8 ->
    # Safe to add
    ChatBot.FactDatabase.Integration.add_fact(...)
  
  {:contradicted, conflicts} ->
    # Handle contradiction
    Logger.warning("Fact contradicts existing beliefs", conflicts: conflicts)
  
  {:uncertain, reason} ->
    # Low confidence or no data
    Logger.debug("Cannot verify fact", reason: reason)
end
```

## Fact Format

Each fact follows this structure:

```json
{
  "id": "geo_001",
  "entity": "France",
  "fact": "The capital of France is Paris",
  "category": "geography",
  "verification_source": "World Atlas, CIA World Factbook",
  "confidence": 1.0
}
```

## Integration Points

### With Learner Module
- Automatically extracts general knowledge facts from conversations
- Verifies facts before adding to database
- Distinguishes between user-specific and general knowledge

### With Epistemic System
- Facts become beliefs in the BeliefStore
- Facts are tracked in UserModelStore when relevant
- Facts can be queried alongside user-specific knowledge

### With Truth Maintenance
- Facts registered as JTMS nodes
- Contradictions automatically detected
- Facts can be retracted if proven false

## Best Practices

1. **Verification Sources**: Always include verification sources for facts
2. **Confidence Levels**: Use appropriate confidence (1.0 for verified facts, lower for inferred)
3. **Categories**: Organize facts by category for better querying
4. **Contradiction Checking**: Always verify facts before adding to avoid contradictions
5. **User vs General**: Distinguish between user-specific facts (go to KnowledgeStore) and general facts (go to FactDatabase)

## Future Enhancements

- Fact versioning (track changes over time)
- Fact confidence decay (reduce confidence if not confirmed)
- Multi-source verification (require multiple sources for high confidence)
- Fact relationships (link related facts)
- Fact expiration (mark time-sensitive facts)
