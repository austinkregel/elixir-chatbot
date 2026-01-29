# ChatBot Architecture

This document provides a comprehensive visual guide to the ChatBot application's architecture, subsystems, and pipeline execution order.

## System Overview

The ChatBot is a **classical NLP chatbot** (no LLMs) built on several interconnected subsystems:

| Subsystem | Purpose |
|-----------|---------|
| **Analysis Pipeline** | Text understanding (chunking, discourse, speech acts, slots) |
| **Epistemic System** | Truth maintenance, beliefs, user models |
| **Memory System** | Episodic/semantic memory with consolidation |
| **Learning System** | Training worlds, entity discovery, heuristic learning |
| **Knowledge Expansion** | Web research, fact verification, admin review |
| **Response Generation** | Template-based, fact-based, memory-augmented responses |

## High-Level Overview

```mermaid
flowchart TB
    subgraph Input["User Input"]
        UI[("User types message")]
    end

    subgraph Brain["Brain GenServer"]
        EVAL[evaluate/3]
        CONV[Conversation State]
    end

    subgraph FastPath["Fast Path Check"]
        RACE[RacingAnalyzer.check_fast_path]
        HEUR[HeuristicStore.match_best]
        MEM_SIM[MemoryStore.query_similar]
    end

    subgraph Pipeline["Analysis Pipeline"]
        PIPE[Pipeline.process]
    end

    subgraph Response["Response Generation"]
        GATE[ResponseGate.evaluate]
        GEN[Generator.generate]
    end

    subgraph Output["Output"]
        OUT[("Response to User")]
    end

    UI --> Brain
    EVAL --> FastPath
    FastPath -->|"fast_path hit"| GEN
    FastPath -->|"no match"| Pipeline
    Pipeline --> GATE
    GATE -->|":respond"| GEN
    GATE -->|":defer"| OUT
    GATE -->|":optional"| GEN
    GEN --> OUT
```

## Complete System Interaction Map

```mermaid
flowchart TB
    subgraph User["User Interface"]
        INPUT[("User Input")]
        OUTPUT[("Response")]
    end

    subgraph Core["Core Processing"]
        BRAIN["Brain<br/>(Orchestrator)"]
        PIPE["Pipeline<br/>(Analysis)"]
        RACE["RacingAnalyzer<br/>(Fast Path)"]
        GEN["Generator<br/>(Response)"]
    end

    subgraph Analysis["Analysis Components"]
        CHUNK["SemanticChunker"]
        DISC["DiscourseAnalyzer"]
        SPEECH["SpeechActClassifier"]
        SLOT["SlotDetector"]
        GATE["ResponseGate"]
        ANAPH["AnaphoraResolver"]
        FOLLOW["FollowupDetector"]
    end

    subgraph ML["ML Layer"]
        TOK["Tokenizer"]
        POS["POSTagger"]
        IC["IntentClassifier"]
        EE["EntityExtractor"]
        GAZ["Gazetteer"]
    end

    subgraph Memory["Memory System"]
        EMBED["Embedder"]
        MSTORE["Memory.Store"]
        CONSOL["Consolidation"]
        THINK["Think API"]
    end

    subgraph Epistemic["Epistemic System"]
        JTMS["JTMS"]
        BELIEF["BeliefStore"]
        UMODEL["UserModelStore"]
        CONTRA["ContradictionHandler"]
    end

    subgraph Learning["Learning System"]
        WORLD["WorldManager"]
        WCTX["WorldContext"]
        ENTDISC["EntityDiscoverer"]
        OUTCOME["OutcomeLearner"]
        HEUR["HeuristicStore"]
    end

    subgraph Knowledge["Knowledge Expansion"]
        LC["LearningCenter"]
        RA["ResearchAgent"]
        RQ["ReviewQueue"]
        SR["SourceReliability"]
    end

    subgraph Stores["Data Stores"]
        KS["KnowledgeStore"]
        FD["FactDatabase"]
        MS["MemoryStore"]
        TS["TemplateStore"]
    end

    %% Main flow
    INPUT --> BRAIN
    BRAIN --> RACE
    RACE -->|fast path| GEN
    RACE -->|no match| PIPE
    PIPE --> GATE
    GATE --> GEN
    GEN --> OUTPUT

    %% Analysis dependencies
    PIPE --> CHUNK
    PIPE --> DISC
    PIPE --> SPEECH
    PIPE --> SLOT
    PIPE --> ANAPH
    BRAIN --> FOLLOW
    SPEECH --> IC
    SLOT --> GAZ

    %% ML dependencies
    DISC --> TOK
    SPEECH --> TOK
    SPEECH --> POS
    EE --> TOK
    EE --> GAZ
    EE --> POS

    %% Memory usage
    RACE --> MSTORE
    SPEECH --> MSTORE
    BRAIN --> THINK
    THINK --> MSTORE
    MSTORE --> EMBED
    CONSOL --> MSTORE

    %% Epistemic usage
    BRAIN --> BELIEF
    BRAIN --> UMODEL
    JTMS --> CONTRA
    BELIEF --> JTMS
    SLOT --> UMODEL

    %% Learning usage
    BRAIN --> OUTCOME
    OUTCOME --> HEUR
    RACE --> HEUR
    WCTX --> WORLD
    ENTDISC --> WORLD
    WORLD --> GAZ

    %% Knowledge expansion
    LC --> RA
    RA --> RQ
    RQ --> BELIEF
    SR --> RA

    %% Response generation
    GEN --> TS
    GEN --> FD
    GEN --> MSTORE

    %% Learner
    BRAIN --> KS
    BRAIN --> MS

    style BRAIN fill:#f96,stroke:#333,stroke-width:4px
    style PIPE fill:#bbf,stroke:#333,stroke-width:2px
    style JTMS fill:#f9f,stroke:#333,stroke-width:2px
    style WORLD fill:#9f9,stroke:#333,stroke-width:2px
    style LC fill:#ff9,stroke:#333,stroke-width:2px
```

## Application Startup Order

The application starts processes in this order (defined in `application.ex`):

```mermaid
flowchart TD
    subgraph Core["Core Infrastructure"]
        T[ChatBotWeb.Telemetry]
        DNS[DNSCluster]
        PS[Phoenix.PubSub]
        REG[Registry - SubprocessRegistry]
    end

    subgraph ML["ML & Memory Layer"]
        MA[Metrics.Aggregator]
        IE[ML.InformalExpansions]
        GAZ[ML.Gazetteer]
        LS[Analysis.LearningStore]
        KS[KnowledgeStore]
        FD[FactDatabase]
        MS[MemoryStore]
        EMB[Memory.Embedder]
        STORE[Memory.Store]
    end

    subgraph Epistemic["Epistemic System"]
        JTMS[Epistemic.JTMS]
        BS[Epistemic.BeliefStore]
        UMS[Epistemic.UserModelStore]
        CH[Epistemic.ContradictionHandler]
    end

    subgraph Adaptive["Adaptive Processing"]
        AC[Analysis.AnalyzerCalibration]
        HS[Analysis.HeuristicStore]
        IC[ML.IntentClassifierSimple]
        WM[Learning.WorldManager]
        WMR[Learning.WorldModelRegistry]
        TS[Response.TemplateStore]
    end

    subgraph Knowledge["Knowledge Expansion"]
        ASV[Task.Supervisor - AgentSupervisor]
        SR[Knowledge.SourceReliability]
        RQ[Knowledge.ReviewQueue]
        LC[Knowledge.LearningCenter]
    end

    subgraph Main["Main Components"]
        BRAIN[ChatBot.Brain]
        EP[ChatBotWeb.Endpoint]
    end

    Core --> ML --> Epistemic --> Adaptive --> Knowledge --> Main

    style BRAIN fill:#f9f,stroke:#333,stroke-width:4px
    style PIPE fill:#bbf,stroke:#333,stroke-width:2px
```

## Brain.evaluate() Flow

This is the main entry point for processing user input:

```mermaid
flowchart TD
    START[["Brain.evaluate(conv_id, input, opts)"]]

    subgraph Lookup["1. Conversation Lookup"]
        FIND[Find conversation by ID]
        NOTFOUND{Found?}
        ERR[Return error]
    end

    subgraph FastCheck["2. Fast Path Check"]
        FOLLOW{is_followup?}
        FOLLOWUP[handle_followup_message]
        NEWMSG[process_new_message]
    end

    subgraph NewMsgFlow["3. New Message Processing"]
        META{is_self_knowledge_query?}
        METACOG[handle_meta_cognitive_query]
        RACING[RacingAnalyzer.check_fast_path]
        FASTPATH{fast_path hit?}
        FASTGEN[handle_fast_path_response]
        STANDARD[process_standard_message]
    end

    subgraph StandardFlow["4. Standard Processing"]
        PIPE[run_analysis_pipeline]
        GATECHECK[ResponseGate.evaluate]
        DEFER{response decision}
        DEFERNIL[Return nil response]
        PROCEED[proceed_with_standard_response]
    end

    subgraph ResponseGen["5. Response Generation"]
        STRAT{overall_strategy}
        CLARIFY[build_clarification_response]
        PARTIAL[partial + clarification]
        DEFERUSER[simple_acknowledgment]
        CANNOT[simple_fallback_response]
        CANRESP[try_nlp_with_analysis]
    end

    subgraph Storage["6. Storage & Learning"]
        BUILDCTX[build_context_snapshot]
        ADDMEM[Update conversation memory]
        ADDQUEUE[Add to learning_queue]
        BELIEFS[extract_and_store_beliefs]
        OUTCOME[OutcomeLearner.learn_from_outcome]
    end

    START --> FIND --> NOTFOUND
    NOTFOUND -->|No| ERR
    NOTFOUND -->|Yes| FOLLOW
    FOLLOW -->|Yes| FOLLOWUP --> Storage
    FOLLOW -->|No| NEWMSG

    NEWMSG --> META
    META -->|Yes| METACOG --> Storage
    META -->|No| RACING
    RACING --> FASTPATH
    FASTPATH -->|Yes| FASTGEN --> Storage
    FASTPATH -->|No| STANDARD

    STANDARD --> PIPE --> GATECHECK --> DEFER
    DEFER -->|:defer| DEFERNIL --> Storage
    DEFER -->|:optional >= threshold| DEFERNIL
    DEFER -->|:respond or :optional < threshold| PROCEED

    PROCEED --> STRAT
    STRAT -->|:needs_clarification| CLARIFY --> Storage
    STRAT -->|:partial_response_with_clarification| PARTIAL --> Storage
    STRAT -->|:defer_to_user| DEFERUSER --> Storage
    STRAT -->|:cannot_respond| CANNOT --> Storage
    STRAT -->|:can_respond| CANRESP --> Storage

    Storage --> BUILDCTX --> ADDMEM --> ADDQUEUE --> BELIEFS --> OUTCOME

    style START fill:#f96,stroke:#333,stroke-width:4px
    style PIPE fill:#bbf,stroke:#333,stroke-width:2px
    style GATECHECK fill:#bfb,stroke:#333,stroke-width:2px
```

## Analysis Pipeline (Pipeline.process)

This is the core NLP processing pipeline:

```mermaid
flowchart TD
    START[["Pipeline.process(text, opts)"]]

    subgraph Stage1["Stage 1: Chunking"]
        CHUNK[SemanticChunker.chunk]
        CHUNKS[("List of Chunks")]
    end

    subgraph Stage2["Stage 2 & 3: Per-Chunk Analysis"]
        LOOP[For each chunk...]
        PARALLEL["Parallel Tasks"]
        DISC[DiscourseAnalyzer.analyze]
        SPEECH[SpeechActClassifier.classify]
        ANAPH[AnaphoraResolver.resolve_and_substitute]
        ENTITY[EntityExtractor.extract_entities]
        MERGE[Merge anaphora entities]
        INTENT[determine_intent]
        FILTER[filter_entities_by_intent]
        SLOT[SlotDetector.detect]
        CTX[ContextResolver.resolve]
        CONF[calculate_confidence]
        STRAT[ChunkAnalysis.determine_response_strategy]
    end

    subgraph Stage4["Stage 4: Overall Strategy"]
        OVERALL[InternalModel.determine_strategy]
        MODEL[("InternalModel with analyses")]
    end

    START --> CHUNK --> CHUNKS --> LOOP

    LOOP --> PARALLEL
    PARALLEL --> DISC
    PARALLEL --> SPEECH

    DISC --> ANAPH
    SPEECH --> ANAPH

    ANAPH --> ENTITY --> MERGE --> INTENT --> FILTER --> SLOT --> CTX --> CONF --> STRAT

    STRAT --> LOOP
    LOOP -->|all chunks done| OVERALL --> MODEL

    style START fill:#f96,stroke:#333,stroke-width:4px
    style PARALLEL fill:#ff9,stroke:#333,stroke-width:2px
    style MODEL fill:#9f9,stroke:#333,stroke-width:2px
```

## Per-Chunk Analysis Detail

Each chunk goes through this detailed analysis:

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant DA as DiscourseAnalyzer
    participant SAC as SpeechActClassifier
    participant AR as AnaphoraResolver
    participant EE as EntityExtractor
    participant SD as SlotDetector
    participant CR as ContextResolver
    participant CA as ChunkAnalysis

    Note over P: Stage 2a + 2b (Parallel)
    par Parallel Analysis
        P->>+DA: analyze(chunk.text, opts)
        DA-->>-P: discourse_result (addressee, confidence)
    and
        P->>+SAC: classify(chunk.text)
        SAC-->>-P: speech_act_result (category, sub_type)
    end

    Note over P: Stage 2c: Anaphora Resolution
    P->>+AR: resolve_and_substitute(text, history)
    AR-->>-P: {resolved_text, anaphora_entities}

    Note over P: Stage 3a: Entity Extraction
    P->>+EE: extract_entities(resolved_text, opts)
    EE-->>-P: entities[]
    P->>P: merge_anaphora_entities

    Note over P: Stage 3b: Intent Determination
    P->>P: determine_intent(speech_act, entities, text)
    P->>P: filter_entities_by_intent(entities, intent)

    Note over P: Stage 3c: Slot Detection
    P->>+SD: detect(intent, filtered_entities)
    SD-->>-P: slot_result (filled, missing)

    Note over P: Stage 3d: Context Resolution
    P->>+CR: resolve(slot_result, opts)
    CR-->>-P: resolved_slots

    Note over P: Build Chunk Analysis
    P->>+CA: new(index, text)
    P->>CA: with discourse, speech_act, intent, entities, slots
    P->>CA: calculate_confidence
    P->>CA: determine_response_strategy
    CA-->>-P: ChunkAnalysis
```

## RacingAnalyzer Fast Path

The racing analyzer provides early-exit optimization:

```mermaid
flowchart TD
    START[["RacingAnalyzer.race(text, opts)"]]

    subgraph FastPath["Fast Path Check"]
        HEUR[HeuristicStore.match_best]
        HEUR_CONF{confidence >= 0.85?}
        MEM[MemoryStore.query_similar]
        MEM_CONF{similarity >= 0.85?}
        FAST_YES[Return fast_path interpretation]
        FAST_NO[Continue to racing]
    end

    subgraph Racing["Racing Analyzers (Parallel)"]
        TASKS[Launch parallel tasks]
        MODEL[analyze_with_model]
        STRUCT[analyze_structure]
        KEYWORD[analyze_keywords]
        PATTERN[analyze_patterns]
        MEMSIM[analyze_memory]
        SELF[analyze_self_knowledge]
    end

    subgraph Collect["Collection with Early Exit"]
        WAIT[Wait for any task]
        EARLYCHECK{result >= 0.90?}
        KILL[Kill remaining tasks]
        CONTINUE[Wait for more]
        TIMEOUT{Timeout?}
    end

    subgraph Calibrate["Post-Processing"]
        CALIB[calibrate_results]
        SAFE[apply_intent_safeguards]
        INTERP[Build Interpretation]
        POOL[ActivationPool.normalize_with_alternatives]
    end

    START --> HEUR --> HEUR_CONF
    HEUR_CONF -->|Yes| FAST_YES
    HEUR_CONF -->|No| MEM --> MEM_CONF
    MEM_CONF -->|Yes| FAST_YES
    MEM_CONF -->|No| FAST_NO

    FAST_NO --> TASKS
    TASKS --> MODEL
    TASKS --> STRUCT
    TASKS --> KEYWORD
    TASKS --> PATTERN
    TASKS --> MEMSIM
    TASKS --> SELF

    MODEL --> WAIT
    STRUCT --> WAIT
    KEYWORD --> WAIT
    PATTERN --> WAIT
    MEMSIM --> WAIT
    SELF --> WAIT

    WAIT --> EARLYCHECK
    EARLYCHECK -->|Yes| KILL --> CALIB
    EARLYCHECK -->|No| CONTINUE --> TIMEOUT
    TIMEOUT -->|No| WAIT
    TIMEOUT -->|Yes| CALIB

    CALIB --> SAFE --> INTERP --> POOL

    style START fill:#f96,stroke:#333,stroke-width:4px
    style FAST_YES fill:#9f9,stroke:#333,stroke-width:2px
    style KILL fill:#f99,stroke:#333,stroke-width:2px
```

## SpeechActClassifier Multi-Pass Analysis

The speech act classifier uses multiple analysis passes:

```mermaid
flowchart LR
    subgraph Input
        TEXT[Text Input]
    end

    subgraph Passes["Analysis Passes"]
        P1["Pass 1: Intent Model"]
        P2["Pass 2: Structural"]
        P3["Pass 3: Keywords (disabled)"]
        P4["Pass 4: Pragmatics"]
        P5["Pass 5: Memory"]
    end

    subgraph Combination["Vote Combination"]
        VOTES[Collect votes]
        WEIGHT[Apply source weights]
        WINNER[Determine winner]
    end

    subgraph Output
        RESULT[SpeechActResult]
    end

    TEXT --> P1 & P2 & P3 & P4 & P5
    P1 & P2 & P3 & P4 & P5 --> VOTES --> WEIGHT --> WINNER --> RESULT

    P1 -.->|"weight: 1.5"| VOTES
    P5 -.->|"weight: 1.4"| VOTES
    P2 -.->|"weight: 1.0"| VOTES
    P4 -.->|"weight: 0.8"| VOTES
```

## Response Generation Flow

```mermaid
flowchart TD
    START[["Generator.generate(intent, entities, query_text)"]]

    subgraph Priority["Response Priority Order"]
        D1{Domain Handler?}
        D2{Memory Augmented?}
        D3{Template Available?}
        FALL[Fallback Response]
    end

    subgraph Domain["Domain-Specific"]
        WEATHER[weather.query handler]
        MUSIC[music.play handler]
        DEVICE[device.control handler]
        NEWS[news.query handler]
        REMINDER[reminder.create handler]
        FACTUAL[question.factual + FactRetriever]
    end

    subgraph Memory["Memory-Based"]
        MEMAUG[MemoryAugmented.generate]
    end

    subgraph Template["Template-Based"]
        TSTORE[TemplateStore.get_random_template]
        SUBST[substitute_slots]
    end

    START --> D1
    D1 -->|Yes| Domain --> OUTPUT
    D1 -->|No| D2
    D2 -->|Yes| Memory --> OUTPUT
    D2 -->|No| D3
    D3 -->|Yes| Template --> OUTPUT
    D3 -->|No| FALL --> OUTPUT

    OUTPUT[("{:ok, response, type}")]

    style START fill:#f96,stroke:#333,stroke-width:4px
    style OUTPUT fill:#9f9,stroke:#333,stroke-width:2px
```

## ResponseGate Decision Tree

```mermaid
flowchart TD
    START[["ResponseGate.evaluate(analysis_model, memory, opts)"]]

    SA{speech_act.category?}
    Q{is_question?}
    GRAT{gratitude_loop?}
    BACK{sub_type == :backchannel?}
    ACK{sub_type == :acknowledgment?}
    THANKS{recent_thanks?}
    COMP{sub_type == :compliment?}
    CONT{sub_type == :continuation?}
    EXP{category == :expressive?}
    EXPECT{expects_response?}

    RESPOND[":respond"]
    DEFER[":defer"]
    OPT[":optional"]

    START --> SA
    SA -->|:directive| RESPOND
    SA -->|other| Q
    Q -->|Yes| RESPOND
    Q -->|No| GRAT
    GRAT -->|Yes| DEFER
    GRAT -->|No| BACK
    BACK -->|Yes| OPT
    BACK -->|No| ACK
    ACK -->|Yes| THANKS
    THANKS -->|Yes| OPT
    THANKS -->|No| COMP
    COMP -->|Yes| OPT
    COMP -->|No| CONT
    CONT -->|Yes| DEFER
    CONT -->|No| EXP
    EXP -->|Yes| EXPECT
    EXPECT -->|No| OPT
    EXPECT -->|Yes| RESPOND
    EXP -->|No| RESPOND

    style START fill:#f96,stroke:#333,stroke-width:4px
    style RESPOND fill:#9f9,stroke:#333,stroke-width:2px
    style DEFER fill:#f99,stroke:#333,stroke-width:2px
    style OPT fill:#ff9,stroke:#333,stroke-width:2px
```

## Data Flow Summary

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Brain.evaluate                                │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Check if follow-up (FollowupDetector)                               │
│  2. Check for self-knowledge query (SelfKnowledgeAnalyzer)              │
│  3. Check fast-path (RacingAnalyzer → HeuristicStore, MemoryStore)      │
│  4. Run Pipeline.process if no fast-path                                 │
│  5. Check ResponseGate.evaluate for response optionality                │
│  6. Generate response via Generator                                      │
│  7. Store in conversation memory                                         │
│  8. Add to learning queue                                                │
│  9. Extract beliefs (Epistemic system)                                   │
│ 10. Learn from outcome (OutcomeLearner)                                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Response to User
```

## Pipeline Stage Execution Order (Verified)

| Stage | Component | Parallel? | Input | Output |
|-------|-----------|-----------|-------|--------|
| 1 | SemanticChunker.chunk | No | raw text | List<Chunk> |
| 2a | DiscourseAnalyzer.analyze | **Yes** (with 2b) | chunk.text | DiscourseResult |
| 2b | SpeechActClassifier.classify | **Yes** (with 2a) | chunk.text | SpeechActResult |
| 2c | AnaphoraResolver.resolve_and_substitute | No | text, history | resolved_text, entities |
| 3a | EntityExtractor.extract_entities | No | resolved_text | entities |
| 3a' | merge_anaphora_entities | No | entities | merged_entities |
| 3b | determine_intent | No | speech_act, entities | intent |
| 3b' | filter_entities_by_intent | No | entities, intent | filtered_entities |
| 3c | SlotDetector.detect | No | intent, entities | SlotResult |
| 3d | ContextResolver.resolve | No | slot_result, opts | resolved_slots |
| 4 | calculate_confidence | No | analysis | confidence |
| 5 | ChunkAnalysis.determine_response_strategy | No | analysis | strategy |
| 6 | InternalModel.determine_strategy | No | all analyses | overall_strategy |

## Epistemic System

The epistemic system provides truth maintenance and belief management:

```mermaid
flowchart TB
    subgraph Epistemic["Epistemic System"]
        JTMS["JTMS<br/>(Justification-Based<br/>Truth Maintenance)"]
        BS["BeliefStore<br/>(Belief CRUD)"]
        UMS["UserModelStore<br/>(Per-User Facts)"]
        CH["ContradictionHandler"]
        DP["DisclosurePolicy"]
    end

    subgraph Operations
        ADD[Add Belief]
        JUSTIFY[Justify Node]
        RETRACT[Retract Assumption]
        QUERY[Query Beliefs]
        DISCLOSE[Decide Disclosure]
    end

    ADD --> BS --> JTMS
    JUSTIFY --> JTMS
    RETRACT --> JTMS --> CH
    QUERY --> BS
    QUERY --> UMS
    DISCLOSE --> DP --> UMS

    style JTMS fill:#f9f,stroke:#333,stroke-width:2px
```

### Key Components:

- **JTMS**: Maintains a dependency network where nodes represent beliefs and justifications link premises to conclusions. Labels (IN/OUT) propagate automatically.
- **BeliefStore**: CRUD operations for beliefs with indexing by subject, predicate, user.
- **UserModelStore**: Per-user knowledge models with confidence tracking and disclosure history.
- **ContradictionHandler**: Triggered when contradiction nodes become IN.

## Memory System (Cognitive Memory)

Ported from a Rust implementation, provides episodic and semantic memory:

```mermaid
flowchart TB
    subgraph Memory["Cognitive Memory System"]
        EMB["Embedder<br/>(TF-IDF Vectorization)"]
        STORE["Store<br/>(Episodes & Semantics)"]
        VI["VectorIndex<br/>(ETS-based Search)"]
        CON["Consolidation<br/>(Cluster → Semantic)"]
        THINK["Think<br/>(High-Level API)"]
    end

    subgraph Operations
        ADD_EP[Add Episode]
        QUERY_SIM[Query Similar]
        CONSOLIDATE[Consolidate Episodes]
    end

    ADD_EP --> THINK --> STORE
    STORE --> EMB --> VI
    QUERY_SIM --> THINK --> VI
    CONSOLIDATE --> CON --> STORE

    style THINK fill:#bbf,stroke:#333,stroke-width:2px
```

### Memory Types:

| Type | Description |
|------|-------------|
| **Episode** | Individual experience: state, action, outcome, tags, embedding |
| **SemanticFact** | Aggregated knowledge from clustered episodes |
| **Procedure** | Learned action sequences (future) |

### Consolidation:
Episodes with embeddings whose cosine similarity exceeds threshold are clustered. Each cluster becomes a `SemanticFact`.

## Learning System (Training Worlds)

Provides isolated training environments with inheritance:

```mermaid
flowchart TB
    subgraph Learning["Learning System"]
        WM["WorldManager<br/>(Lifecycle)"]
        WC["WorldContext<br/>(Inheritance API)"]
        WMR["WorldModelRegistry<br/>(Per-World Models)"]
        ED["EntityDiscoverer<br/>(POS-based Discovery)"]
        TI["TypeInferrer"]
        OL["OutcomeLearner<br/>(Heuristic Creation)"]
        WE["WorldEmbedder<br/>(Per-World Embeddings)"]
    end

    subgraph World["Training World"]
        CONFIG[Config]
        EPISODES[Episodes]
        KNOWLEDGE[Knowledge]
        OVERLAY[Gazetteer Overlay]
        METRICS[Metrics]
        EVENTS[Events]
        CANDIDATES[Entity Candidates]
    end

    WM --> World
    WC --> WM
    ED --> CANDIDATES
    OL --> WM

    style WM fill:#f9f,stroke:#333,stroke-width:2px
    style WC fill:#bbf,stroke:#333,stroke-width:2px
```

### World Inheritance:
```
Child World → Base World → Default World
```

Data lookups traverse the inheritance chain. Child overrides parent.

### Entity Discovery Flow:
1. `EntityDiscoverer` uses POS tagger to find proper nouns (PROPN)
2. Checks Gazetteer for known types
3. Unknown entities → candidates pool
4. Admin promotes candidates → Gazetteer overlay

### Outcome Learning:
`OutcomeLearner` monitors conversation outcomes:
- Successful patterns → may become heuristics
- Failed heuristics → deprecated
- Calibration data → `AnalyzerCalibration`

## Knowledge Expansion System

Autonomous knowledge gathering with human review:

```mermaid
flowchart TB
    subgraph Knowledge["Knowledge Expansion"]
        LC["LearningCenter<br/>(Orchestrator)"]
        RA["ResearchAgent<br/>(Web Fetcher)"]
        COR["Corroborator<br/>(Source Cross-Check)"]
        SR["SourceReliability<br/>(Trust Scores)"]
        RQ["ReviewQueue<br/>(Admin Review)"]
        HP["HtmlProcessor"]
    end

    subgraph Session["Learning Session"]
        GOAL[Research Goal]
        FINDINGS[Findings]
        CANDIDATES[Candidates]
    end

    LC -->|dispatch| RA
    RA -->|fetch| HP
    RA -->|findings| COR
    COR -->|candidates| RQ
    SR -->|trust| COR
    RQ -->|approved| BS["BeliefStore"]

    style LC fill:#f9f,stroke:#333,stroke-width:2px
    style RQ fill:#ff9,stroke:#333,stroke-width:2px
```

### Flow:
1. `LearningCenter.start_session(topic)` creates goals
2. `ResearchAgent` tasks fetch web content
3. Content cleaned by `HtmlProcessor`
4. Claims extracted via `Pipeline.process`
5. `Corroborator` cross-checks sources
6. Candidates checked for contradictions with `BeliefStore`
7. `ReviewQueue` holds for admin approval
8. Approved facts → `BeliefStore` + `FactDatabase`

## Classical NLP Components

```mermaid
flowchart LR
    subgraph ML["ML Components"]
        TOK["Tokenizer<br/>(No Regex!)"]
        POS["POSTagger<br/>(HMM-based)"]
        GAZ["Gazetteer<br/>(Entity Lookup)"]
        IC["IntentClassifier<br/>(TF-IDF + Cosine)"]
        EE["EntityExtractor"]
        NLP["NLPPipeline"]
    end

    subgraph Data
        INTENTS[data/intents/*.json]
        ENTITIES[data/entities/*.json]
        MODELS[priv/ml_models/*.term]
    end

    INTENTS --> IC
    ENTITIES --> GAZ
    MODELS --> IC
    MODELS --> POS
    MODELS --> EE

    TOK --> POS --> EE
    TOK --> IC
    GAZ --> EE
```

### Key Design Principles:
- **No Regex in NLP**: All text processing via `Tokenizer` functions
- **TF-IDF Vectorization**: Intent classification and memory similarity
- **ETS for Speed**: Gazetteer lookups use ETS tables
- **World-Scoped Models**: Each training world can have its own classifier

## Learner Module

Extracts knowledge from conversations:

```mermaid
flowchart TD
    INPUT[User Input]
    LEARNER["Learner.learn_from_input"]
    EE["EntityExtractor"]
    
    subgraph Storage
        KS["KnowledgeStore<br/>(Persona-scoped)"]
        MS["MemoryStore"]
        FD["FactDatabase<br/>(General Knowledge)"]
        BS["BeliefStore"]
    end

    INPUT --> LEARNER --> EE
    LEARNER --> KS
    LEARNER --> MS
    LEARNER --> FD
    FD --> BS
```

Entity types processed:
- person, pet, room, device, place, task, event, preference

## Key Component Descriptions

### Brain (GenServer)
The central orchestrator that manages conversations, processes input, and coordinates all subsystems. Entry point: `Brain.evaluate/3`.

### Analysis Pipeline
Orchestrates text analysis through multiple stages. Each chunk goes through discourse analysis, speech act classification, entity extraction, slot detection, and context resolution.

### RacingAnalyzer
Runs multiple interpretation paths in parallel with early-exit optimization. Fast paths: heuristics, memory similarity.

### SpeechActClassifier
5 analysis passes (intent model, structural, keywords, pragmatics, memory) with weighted voting to classify pragmatic function (Searle's taxonomy).

### ResponseGate
Determines if a response is appropriate based on speech act sequences. Handles gratitude loops, backchannels, compliments, and continuations.

### Generator
Unified response generation: domain handlers → memory-augmented → templates → fallback.

### JTMS (Justification-Based Truth Maintenance)
Classic JTMS implementation. Nodes = beliefs, Justifications = dependencies. Labels propagate automatically on changes.

### WorldManager
Manages training world lifecycle. ETS-backed for performance. Supports ephemeral and persistent modes.

### LearningCenter
Orchestrates autonomous knowledge gathering. Dispatches ResearchAgents as supervised Tasks.

### OutcomeLearner
Learns from conversation outcomes to create/update heuristics. Scoped by user/cohort/global.

## Confirming Pipeline Order

To verify the actual execution order at runtime, you can:

1. **Enable debug logging**: The Pipeline module logs at each stage with `Progress.report/3`

2. **Check telemetry events**: The system emits telemetry spans for:
   - `:pipeline_process`
   - `:racing_analysis`
   - `:brain_evaluate`
   - `:knowledge_research`
   - `:belief_operation`
   - `:jtms_justify`

3. **Review the code execution**:
   - `Pipeline.process/2` → `do_process/2` runs stages sequentially
   - `analyze_single_chunk/2` runs stages 2a/2b in parallel, then 2c-5 sequentially
   - `InternalModel.determine_strategy/1` runs after all chunks are analyzed
