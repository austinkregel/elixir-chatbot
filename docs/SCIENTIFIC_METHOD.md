# Scientific Method Integration

This document describes how the ChatBot system implements the scientific method for knowledge acquisition and capability testing.

## Overview

The Knowledge Expansion System follows the scientific method as outlined in "An Introduction to Scientific Investigation":

1. **Observation** → User inputs, training data, benchmark tasks
2. **Hypothesis** → Testable claims derived from questions
3. **Investigation** → Systematic evidence gathering
4. **Evidence** → Findings from sources (web, tasks, tests)
5. **Evaluation** → Test hypotheses against evidence
6. **Conclusion** → Supported, falsified, or inconclusive

**Key Principle**: We cannot prove a hypothesis true, only support it with evidence or falsify it with contradicting evidence.

## Core Concepts

### Hypothesis

A hypothesis is a tentative explanation that can be tested and potentially falsified.

```elixir
alias Brain.Knowledge.Types.Hypothesis

hypothesis = Hypothesis.new(
  "Paris is the capital of France",
  entity: "France",
  derived_from: "What is the capital of France?",
  prediction: "If true, independent sources will confirm this claim."
)
```

**Properties:**
- `claim` - The testable statement
- `derived_from` - The original question
- `prediction` - "If/Then" statement for expected results
- `status` - `:untested`, `:testing`, `:supported`, `:falsified`, `:inconclusive`
- `supporting_evidence` - Findings that agree with the claim
- `contradicting_evidence` - Findings that disagree
- `confidence` - 0.0-1.0 based on evidence quality and quantity
- `replication_count` - Evidence from same sources (replication)

### Investigation

An investigation tests one or more hypotheses using the scientific method.

```elixir
alias Brain.Knowledge.Types.Investigation

investigation = Investigation.new("European Capitals",
  hypotheses: [hypothesis1, hypothesis2]
)
```

**Experimental Variables:**
- **Independent Variable** - What we vary (sources queried)
- **Dependent Variable** - What we measure (claims extracted)
- **Constants** - What we hold fixed (NLP pipeline, thresholds)
- **Control Treatment** - Baseline facts for comparison

### Falsifiability

A hypothesis is **falsified** when:
- Reliable contradicting evidence exists (reliability >= 0.6)
- The contradicting source is independent

A hypothesis is **supported** when:
- 2+ independent sources provide agreeing evidence
- No reliable contradicting evidence exists

```elixir
# After gathering evidence
hypothesis = Hypothesis.evaluate(hypothesis)

case hypothesis.status do
  :supported -> "Evidence supports the hypothesis"
  :falsified -> "Contradicting evidence disproves the hypothesis"
  :inconclusive -> "Mixed or insufficient evidence"
end
```

## Confidence Calculation

Confidence reflects the quality and quantity of evidence:

```
confidence = pass_rate * 0.6 +        # Primary: what % of evidence supports
             reliability * 0.2 +       # Source quality
             sample_size_factor * 0.1 + # Need 5+ samples for reliability
             diversity_bonus * 0.1     # Multiple independent sources
```

**Confidence Levels:**
- `very_high` - >= 85%
- `high` - >= 70%
- `moderate` - >= 50%
- `low` - >= 25%
- `none` - < 25%

## Task-Based Training

The system uses domain-specific benchmark tasks via `LearningCenter` for training:

### Available Task Categories

| Category | Description |
|----------|-------------|
| `question_answering` | Factual Q&A pairs |
| `commonsense` | Reasoning with explanations |
| `sentiment` | Emotion detection |
| `explanation` | Reasoning patterns |

### Starting Task Training

```elixir
alias Brain.Knowledge.LearningCenter

# Start task-based training
{:ok, session} = LearningCenter.start_task_training(:question_answering,
  max_tasks: 10,
  max_instances: 20
)

# Train on all categories
{:ok, session} = LearningCenter.start_task_training(:all)
```

### Training Flow

1. **Load task files** from `data/domain_specific_tasks/`
2. **Transform tasks** to training format via `TaskTransformer`
3. **Process each instance:**
   - Extract question/answer pairs
   - Add to fact database
   - Update knowledge store
4. **Track metrics** for the training session

### Task Source Benefits

| Aspect | Web Sources | Task Sources |
|--------|-------------|--------------|
| Quality | Variable | Human-verified |
| Speed | Network-bound | Local files |
| Reproducibility | Varies | Consistent |
| Coverage | Broad | Focused domains |

## Knowledge Expansion with Scientific Method

The `LearningCenter` uses scientific investigations for knowledge acquisition:

```elixir
# Start a learning session
{:ok, session} = LearningCenter.start_session("France",
  questions: ["What is the capital of France?"]
)

# System automatically:
# 1. Creates investigation from research goal
# 2. Generates hypotheses from questions
# 3. Dispatches agents to gather evidence
# 4. Tests hypotheses against findings
# 5. Promotes supported hypotheses to review queue
# 6. Logs falsified hypotheses for learning
```

### Session Tracking

Learning sessions now track scientific outcomes:

```elixir
LearningSession.scientific_summary(session)
# => %{
#      investigations_completed: 3,
#      hypotheses_tested: 15,
#      hypotheses_supported: 10,
#      hypotheses_falsified: 3,
#      support_rate: 0.67
#    }
```

## UI Integration

### Settings Page (`/settings`)

The Settings page includes training configuration options.

### Training Sessions

Training sessions display scientific outcomes:
- Number of hypotheses tested
- Count of supported hypotheses (green)
- Count of falsified hypotheses (red)
- Session progress and metrics

### Dashboard (`/ops/dashboard`)

The operations dashboard shows:
- Active training sessions
- Knowledge expansion progress
- Review queue status

## Data Sources

### Domain-Specific Tasks

Benchmark tasks are stored in `data/domain_specific_tasks/`:

```elixir
TaskSource.list_categories()
# => ["Question Answering", "Text Categorization", "Sentiment Analysis", ...]

TaskSource.list_tasks(:question_answering)
# => [%{path: "...", domains: ["Wikipedia"], ...}, ...]
```

### Task Structure

```json
{
  "Categories": ["Question Answering"],
  "Domains": ["Wikipedia"],
  "Definition": ["...task description..."],
  "Positive Examples": [
    {"input": "...", "output": "...", "explanation": "..."}
  ],
  "Instances": [
    {"id": "task001-abc", "input": "...", "output": ["..."]}
  ]
}
```

## Principles from Research

Based on "An Introduction to Scientific Investigation":

| Principle | Implementation |
|-----------|---------------|
| Falsifiability | Hypotheses can be proven false with contradicting evidence |
| Cannot Prove True | Only "supported", never "proven" |
| Independent Verification | Requires 2+ independent sources |
| Replication | Multiple tests increase confidence |
| Control Treatment | Existing facts as baseline |
| Variables | Track independent (sources), dependent (claims), constants (pipeline) |

## Module Reference

| Module | Purpose |
|--------|---------|
| `Brain.Knowledge.Types.Hypothesis` | Testable claim with evidence tracking |
| `Brain.Knowledge.Types.Investigation` | Scientific investigation container |
| `Brain.Knowledge.Corroborator` | Hypothesis testing and evaluation |
| `Brain.Knowledge.LearningCenter` | Orchestrates scientific investigations |
| `Tasks.Source` | Provides NLP benchmark tasks |
| `Tasks.Analyzer` | Analyzes task file structure |
| `Tasks.Transformer` | Transforms tasks to training format |

## Future Enhancements

- [ ] Track hypothesis evolution over time
- [ ] Implement A/B testing for pipeline improvements (requires summarization system)
- [ ] Add statistical significance testing
- [ ] Create improvement recommendations from falsified hypotheses
- [ ] Build learning curves from repeated tests

---

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Main contributor guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture with knowledge expansion diagrams
- [SUBSYSTEM_INTEGRATION_REVIEW.md](SUBSYSTEM_INTEGRATION_REVIEW.md) - Knowledge expansion limitations and status
