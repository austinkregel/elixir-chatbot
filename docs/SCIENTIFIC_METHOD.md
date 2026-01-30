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
alias ChatBot.Knowledge.Types.Hypothesis

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
alias ChatBot.Knowledge.Types.Investigation

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

## Capability Testing

The `ChatBot.Testing.CapabilityTest` module uses the scientific method to evaluate NLP capabilities.

### Available Capabilities

| Capability | Description |
|------------|-------------|
| `question_answering` | Extract answers from passages |
| `entity_recognition` | Identify named entities |
| `sentiment` | Detect emotional tone |
| `classification` | Categorize text by intent |
| `temporal_reasoning` | Understand time relationships |
| `coreference` | Resolve pronouns to entities |
| `commonsense` | Apply common knowledge reasoning |

### Running Tests

```elixir
alias ChatBot.Testing.CapabilityTest

# Test a single capability
{:ok, investigation} = CapabilityTest.test_capability(:question_answering,
  limit: 10,      # instances per task
  max_tasks: 3,   # task files to use
  verbose: true   # log each test result
)

# View results
Investigation.summary(investigation)
# => %{
#      topic: "Capability: question_answering",
#      total_hypotheses: 3,
#      supported: 0,
#      falsified: 3,
#      inconclusive: 0,
#      conclusion: :hypotheses_falsified
#    }

# Run full benchmark
{:ok, results} = CapabilityTest.run_benchmark(
  capabilities: [:sentiment, :entity_recognition],
  limit: 10,
  max_tasks: 3
)
```

### Test Flow

1. **Create Investigation** for the capability
2. **For each task file:**
   - Create hypothesis: "System can perform {capability} on {task}"
   - Run test instances
   - Each passed test → supporting evidence
   - Each failed test → contradicting evidence
   - Evaluate hypothesis
3. **Conclude Investigation** with overall result

### Interpreting Results

| Conclusion | Meaning | Action |
|------------|---------|--------|
| `:hypotheses_supported` | All tests passed sufficiently | Capability is working |
| `:hypotheses_falsified` | Tests failed significantly | Needs improvement |
| `:inconclusive` | Mixed or insufficient results | More testing needed |
| `:mixed` | Some supported, some falsified | Partial capability |

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

### Testing Tab (`/settings?section=testing`)

The Settings page includes a Testing tab with:

1. **Scientific Method Overview** - Visual explanation of the process
2. **Run Capability Test** - Select capability and run tests
3. **Test Results Panel** - Shows:
   - Summary stats (hypotheses, supported, falsified)
   - Conclusion badge (color-coded)
   - Individual hypothesis details with confidence
4. **Task Category Statistics** - Available benchmark tasks

### Training Sessions

Training sessions display scientific outcomes:
- Number of hypotheses tested
- Count of supported hypotheses (green)
- Count of falsified hypotheses (red)

## Data Sources

### Domain-Specific Tasks

1600+ benchmark tasks in `data/domain_specific_tasks/`:

```elixir
CapabilityTest.task_stats()
# => %{
#      "Question Answering" => 206,
#      "Text Categorization" => 46,
#      "Sentiment Analysis" => 22,
#      ...
#    }
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
| `ChatBot.Knowledge.Types.Hypothesis` | Testable claim with evidence tracking |
| `ChatBot.Knowledge.Types.Investigation` | Scientific investigation container |
| `ChatBot.Knowledge.Corroborator` | Hypothesis testing and evaluation |
| `ChatBot.Testing.CapabilityTest` | NLP capability benchmarking |
| `ChatBot.Knowledge.LearningCenter` | Orchestrates scientific investigations |

## Future Enhancements

- [ ] Track hypothesis evolution over time
- [ ] Implement A/B testing for pipeline improvements
- [ ] Add statistical significance testing
- [ ] Create improvement recommendations from falsified hypotheses
- [ ] Build learning curves from repeated tests
