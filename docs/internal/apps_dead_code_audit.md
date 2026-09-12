# Dead-Code and Disconnected-Feature Audit

**Date:** 2026-04-27
**Umbrella version:** 0.1.0
**Method:** `mix xref graph` (329 tracked files, 992 runtime edges), compiler warnings, exhaustive ripgrep across 571 `.ex`/`.exs` files under `apps/`, plus module inventory across `apps/*/lib/**/*.ex` (383 lib modules, including 81 `Mix.Tasks.*` modules).

**Limitations:** Static xref + grep can miss **runtime dispatch** (`apply/3`, config-driven module atoms, Phoenix behaviour callbacks, dynamic supervision). Treat “high confidence” as **static** confidence unless verified at runtime.

---

## Category legend

| Tag | Meaning |
|-----|---------|
| **dead** | Zero xref incoming edges AND no grep callers in lib code. Truly unreachable. |
| **test_only** | Referenced exclusively from test files (no lib caller). |
| **self_only** | Only references are within its own file (defmodule line). |
| **not_wired_ui** | Server-side code exists but no client/route/JS uses it. |
| **config_gated_off** | Behaviour depends on config/env (may default on or off per subsystem). |
| **deprecated** | Marked deprecated in docs/comments or superseded by newer API. |
| **legacy** | Named `*_legacy` or documented as legacy compatibility path. |
| **mix_task_only** | Mix task with zero incoming xref edges (normal for tasks—standalone by design). |
| **orphan_lib** | Library module with no incoming xref edges AND no test references. |

---

## 1. Dead / orphan library code (highest confidence)

These modules have **zero incoming xref edges** and **zero grep references** from any lib code. Some have test references.

| Module | File | Evidence | Confidence |
|--------|------|----------|------------|
| `Brain.Code.Tokenizer` | `apps/brain/lib/brain/code/tokenizer.ex` | Zero xref incoming. Only referenced from own file + 1 test file (`code_modules_test.exs`). | **high** |
| `Brain.ML.KnowledgeGraph.Embedder` | `apps/brain/lib/brain/ml/knowledge_graph/embedder.ex` | **ALIVE as of KG Signal Strengthening (Phase 4).** Used by `EntityVectorCache.get_or_compute/3` for memory retrieval re-ranking and by `StanceTracker`, `NoveltyDetector`, `Corroborator` for entity vector lookups. No longer dead code. | **low** (actively used) |
| `Brain.Lexicon.SenseDrift` | `apps/brain/lib/brain/lexicon/sense_drift.ex` | Zero xref incoming. Only self-reference. Not called from any lib or test file. | **high** |
| `Brain.Lexicon.ConceptNetParser` | `apps/brain/lib/brain/lexicon/conceptnet_parser.ex` | Zero xref incoming from lib by module name; live call from `mix ingest_lexicon` only. `enrichment_features.ex` **line ~360** mentions the name in a **comment** only (not a call—can look like a false positive in grep). | **high** (mix-task-only caller) |
| `Brain.Code.Summarizer` | `apps/brain/lib/brain/code/summarizer.ex` | Zero xref incoming from lib. Only `code_modules_test.exs`. Same pattern as `Brain.Code.Tokenizer`. | **high** |
| `Brain.Analysis.BacktrackController` | `apps/brain/lib/brain/analysis/backtrack_controller.ex` | Lib: only `ProcessingTrace` (not in main pipeline). Tests: `adaptive_processing_test.exs` calls `BacktrackController` **directly** (not via `ProcessingTrace`). | **high** |
| `Brain.Analysis.ProcessingTrace` | `apps/brain/lib/brain/analysis/processing_trace.ex` | Zero xref incoming from lib. **Test:** only `processing_trace_test.exs` references `ProcessingTrace` (not `adaptive_processing_test.exs`—that file exercises `BacktrackController` only). | **high** |
| `Brain.Analysis.ChunkProfile.Projections` | `apps/brain/lib/brain/analysis/chunk_profile/projections.ex` | Zero xref incoming from lib. Only `Projections.` calls in test (`projections_test.exs`). | **high** |
| `Brain.ML.ModelPreflight` | `apps/brain/lib/brain/ml/model_preflight.ex` | Zero xref incoming from lib. Live call site: `Brain.ML.ModelPreflight.validate_all!()` in `apps/brain/test/test_helper.exs` only (`train_framing.ex` mentions it in a comment only). Treat as **test boot–only**, not mix-wired. | **high** (test-only) |
| `FourthWall` (root module) | `apps/fourth_wall/lib/fourth_wall.ex` | Zero xref incoming. Only defines `version/0`. No callers in any app. | **high** |
| `Atlas` (root module) | `apps/atlas/lib/atlas.ex` | Zero xref incoming. Only defines module—no functions are called by other modules. | **high** |
| `Tasks` (root module) | `apps/tasks/lib/tasks.ex` | Zero xref incoming. Shell module only. | **high** |
| `World` (root module) | `apps/world/lib/world.ex` | Only referenced from `World.Persistence`. Single caller. | **low** |

**Not dead — naming fix:** `Brain.Lexicon.ConceptNet` is used from `Brain.Lexicon`, `analysis/feature_extractor/enrichment_features.ex`, and `ml/gazetteer.ex`; static xref may not show every edge. Do not treat it as orphan code.

---

## 2. Not wired in shipped UI

| Module | File | Evidence | Confidence |
|--------|------|----------|------------|
| `ChatWeb.PageHTML` | `apps/chat_web/lib/chat_web/controllers/page_html.ex` | `PageController` only calls `redirect/2`—never renders a template. `PageHTML` and `home.html.heex` are unreachable. | **high** |
| `ChatWeb.Mailer` | `apps/chat_web/lib/chat_web/mailer.ex` | Zero references from any lib code. Only self-defines. No email sending anywhere in the app. | **high** |
| `ChatWeb.BrainChannel` + `ChatWeb.UserSocket` | `apps/chat_web/lib/chat_web/channels/brain_channel.ex`, `user_socket.ex` | Endpoint mounts `/socket` with `UserSocket`, but `app.js` line 3 comments out `user_socket.js`. No JS client connects. Chat uses LiveView PubSub instead. BrainChannel is only exercised from tests. | **high** |
| `ChatWeb.ErrorJSON` | `apps/chat_web/lib/chat_web/controllers/error_json.ex` | Phoenix’s default JSON error path when JSON routes exist; low use today. If you add JSON APIs later, it becomes the default error encoder—keep unless you intentionally replace error handling. | **low** |

### 2a. Routed LiveViews (wired — not dead)

These `live_session :world_context` routes in [`apps/chat_web/lib/chat_web/router.ex`](../../apps/chat_web/lib/chat_web/router.ex) are **shipped and connected** (each has a LiveView module under `lib/chat_web/live/`):

| Route(s) | LiveView module |
|----------|-----------------|
| `/chat`, `/chat/:conversation_id` | `ChatWeb.ChatLive` |
| `/explorer` | `ChatWeb.ExplorerLive` |
| `/dashboard` | `ChatWeb.DashboardLive` |
| `/settings` | `ChatWeb.SettingsLive` |
| `/code` | `ChatWeb.CodeAnalysisLive` |
| `/sessions`, `/sessions/:session_id` | `ChatWeb.SessionsLive` |
| `/knowledge-review` | `ChatWeb.Admin.KnowledgeReviewLive` |
| `/accuracy` | `ChatWeb.AccuracyLive` |

Also: `GET /` → `PageController.home` → redirect to `/chat`; legacy redirects under `/admin`, `/worlds`, `/memories`, `/ops/dashboard`; `/api/*` → `TestController`; `/dev/*` LiveDashboard/mailbox when `dev_routes`.

---

## 3. Supervised / dynamic systems needing explicit disposition

These modules were missed by the initial static summary because they are started by supervision, invoked through callbacks/PubSub/timers, or only reachable through public API wrappers. They are not all dead, but each deserves an explicit disposition in a dead-code audit.

| Module(s) | File(s) | Evidence | Confidence |
|-----------|---------|----------|------------|
| `Brain.Analysis.OutcomeLearner.Store` | `apps/brain/lib/brain/analysis/outcome_learner/store.ex` | Supervised from `Brain.Application`; **never wired internally.** Parent `Brain.Analysis.OutcomeLearner.track_pattern/4` → `increment_pattern_success/1` uses **`Process.get` / `Process.put`** on `{:pattern_success, key}` (`outcome_learner.ex` ~L300–309). The Store’s moduledoc says it *replaces* that approach for cross-process persistence, but **`OutcomeLearner` never calls `Store.increment/1` or `Store.get_count/1`.** No external production callers either—so the GenServer is a **stub relative to the learner**. **Possible integration:** replace `increment_pattern_success` / `get_pattern_success_count` bodies with `Store.increment/1` and `Store.get_count/1` (and decide lifecycle for process-local vs ETS semantics). | **high** disconnected / unfinished |
| `Brain.Analysis.FramingDetector` | `apps/brain/lib/brain/analysis/framing_detector.ex` | Supervised and tested, but no production caller of `assess_document/1`, `assess_entity_framing/2`, or `detect_drift/2`. Moduledoc says JTMS integration will "eventually" call it. | **high** disconnected |
| `World.CodeContext` | `apps/world/lib/world/code_context.ex` | Only lib caller is `Brain.Code.Summarizer`, which is already classified as test-only/dead. No UI, pipeline, or Mix task caller found. | **high** orphan chain |
| `World.DocumentIngestor` | `apps/world/lib/world/document_ingestor.ex` | Used by `mix training_world` only; no product runtime caller. `Brain.Code.QueryHandler` only mentions it in an error/remediation string. **`code_analysis_live.html.heex` and `dashboard_live.html.heex`** show `World.DocumentIngestor.ingest_codebase(...)` as **copy-paste documentation snippets** for operators—not executed calls. | **high** mix-task-only subsystem |
| `Brain.Subprocesses.Supervisor`, `HttpSubprocess`, `ConversationSubprocess`, `CliSubprocess` | `apps/brain/lib/brain/subprocesses/*.ex` | Supervisor is started by `Brain.Application`; child subprocesses are only started through `Brain.start_*_subprocess` API wrappers and tests. No `chat_web` or Mix task caller found for those public APIs. | **medium** optional/unwired API |
| `Brain.Epistemic.ContradictionHandler` | `apps/brain/lib/brain/epistemic/contradiction_handler.ex` | No direct production caller of public contradiction APIs, but it registers a callback with `JTMS.set_contradiction_handler/1` during init. This is dynamically alive, not dead. | **medium** callback-driven |
| `Brain.Knowledge.LearningTriggers` | `apps/brain/lib/brain/knowledge/learning_triggers.ex` | Supervised and subscribes to PubSub topics `learning:novel_input` and `learning:investigation`; publishers exist in `Analysis.Pipeline` and `Knowledge.LearningCenter`. Alive through PubSub, not direct calls. | **medium** PubSub-driven |
| `World.EntityPromoter` | `apps/world/lib/world/entity_promoter.ex` | Supervised by `World.Application` and timer-driven (`:scan` every 10 minutes). No regular product caller except tests/manual `scan_now/1`. | **medium** timer-driven |
| `Atlas.Importer` | `apps/atlas/lib/atlas/importer.ex` | Used by `mix atlas.import` and optionally by `Atlas.Application` when `:atlas, :auto_import` is enabled. Not a normal runtime feature unless config turns it on. | **medium** config/mix gated |
| `Atlas.Stats`, `Atlas.Stats.Collector` | `apps/atlas/lib/atlas/stats.ex`, `apps/atlas/lib/atlas/stats/collector.ex` | `Stats.Collector` is supervised by `Atlas.Application`; `ChatWeb.DashboardLive` calls `Atlas.Stats.get_overview/0`. Alive, but missed from framework/OTP-alive inventory. | **high** alive |
| `ChatWeb.TestController` | `apps/chat_web/lib/chat_web/controllers/test_controller.ex` | Wired under `/api` routes for test-learning/test-knowledge endpoints. This is not dead, but it is an odd shipped surface area that should be explicitly retained or removed. | **medium** wired test API |
| `FourthWall.Math`, `FourthWall.ID`, `FourthWall.AST`, `FourthWall.Safety` | `apps/fourth_wall/lib/fourth_wall/{math,id,ast,safety}.ex` | Root `FourthWall` is an unused shell, but these library modules are alive: `Math` is used by Brain similarity code, `ID` by JTMS IDs, and `AST`/`Safety` by Credo fixer Mix tasks. | **high** alive |

---

## 4. Config-gated / optional stacks

These are supervised and compiled; behaviour depends on **application env**. Split “feature flag defaults” from “lazy start” below.

| Module(s) | Config gate | Default | Evidence |
|-----------|------------|---------|----------|
| `Brain.ML.Ouro.Model`, `Client`, `Spec`, tokenizer helpers, `ModelDownloader` | `:ouro_enabled`, `:skip_ml_init` | **`ouro_enabled` defaults `true`** (`Application.get_env(:brain, :ouro_enabled, true)`). Init still skips heavy work when `:skip_ml_init` or not `ouro_enabled`. | Feature is **on** unless env disables it; not “off by default.” |
| `Brain.ML.Ouro.SidecarLauncher` | `:ouro_auto_start` in `:ml` config | **`false`** (`ml_config[:ouro_auto_start] \|\| false`) | Starts in `:disabled` / idle until **`ensure_ready!`** or explicit start—**lazy sidecar**, not the same as `ouro_enabled: false`. |
| `Brain.ML.ModelStore` | `MODEL_STORE_ENABLED` / app env | **`false`** in non-test (`config/runtime.exs`) | Returns `{:error, :disabled}` when off. Used by `MicroClassifiers`, `Embedder`, `PoincaréEmbeddings`, `TripleScorer`, training tasks. |
| `Brain.Epistemic.*` (BeliefStore, JTMS, UserModel, StanceTracker, ContradictionHandler, SourceAuthority) | `Brain.Epistemic.Types.Config` `:enabled` | **`true`** (`Keyword.get(config, :enabled, true)`). No `config :brain, :epistemic` in checked `config/*.exs` → **enabled unless overridden at runtime.** | **Configurable kill switch**, not disabled by default. When disabled: early returns / no registration; `ContextResolver` skips user-model slot resolution. |
| `Brain.Response.OuroRealizer` | Only caller: `SurfaceRealizer` | N/A | Depends on `Ouro.Model.ready?()` / sidecar health; `SurfaceRealizer` falls back when Ouro is not ready. |
| ML init pipeline | `:ml` `enabled` + `:skip_ml_init` | Varies by env | `Brain.Application.init_ml_pipeline/0` runs only when ML is enabled and `:skip_ml_init` is not set. |
| `Atlas.Importer` auto-import path | `:atlas, :auto_import` | **`false`** unless configured | `Atlas.Application` calls `Atlas.Importer.import_all/1` only when `auto_import` is enabled and selected stores are empty. Manual `mix atlas.import` remains available. |
| Atlas auto-migrations | `:atlas, :auto_migrate` | **`false`** unless configured | `Atlas.Application` runs Ecto migrations on boot only when `auto_migrate` is enabled. |

---

## 5. Deprecated APIs and legacy code

| Module / Function | File | Evidence | Status |
|-------------------|------|----------|--------|
| `Brain.SystemStatus.get_lstm_status/0` | `apps/brain/lib/brain/system_status.ex` | Returns hardcoded "LSTM models removed" map. Doc says deprecated. | **deprecated** |
| `Brain.SystemStatus.get_all/0` | same | Doc: "legacy format for compatibility". | **legacy** |
| `Brain.Knowledge.SourceReliability` stats helper | `apps/brain/lib/brain/knowledge/source_reliability.ex` | `@doc` says "Deprecated: Use `stats/0` instead." | **deprecated** |
| `Brain.Analysis.RacingAnalyzer.check_fast_path/3` | `apps/brain/lib/brain/analysis/racing_analyzer.ex` | Deprecated in favor of `/4` arity. | **deprecated** |
| `Brain.ML.Trainer.load_training_data_legacy/0` | `apps/brain/lib/brain/ml/trainer.ex` | Explicit `_legacy` suffix. | **legacy** |
| `Brain.ML.Trainer.save_models/2` | same | Deprecated warning in docs. | **deprecated** |
| `Brain.ML.DataLoaders.load_all_intents_legacy/0` | `apps/brain/lib/brain/ml/data_loaders.ex` | Explicit `_legacy` suffix. Falls back to `data/intents` directory. | **legacy** |
| `Brain.ML.GoldStandardMigrator` | `apps/brain/lib/brain/ml/gold_standard_migrator.ex` | One-shot migration tool from `data/legacy/intents`. | **legacy** |
| `Brain.MemoryStore` | `apps/brain/lib/brain/memory_store.ex` | `SystemStatus` labels it "Memory Store (Legacy)". Superseded by `Brain.Memory.Store`. | **legacy** |
| `Brain.KnowledgeStore` | `apps/brain/lib/brain/knowledge_store.ex` | Moduledoc: "legacy persona storage". Still used from `Brain.ex`, `ChatLive`, `ExplorerLive`, `TestController`, `Learner`. | **legacy (actively called)** |
| `Brain.ML.Ouro.Model` Bumblebee backend | `apps/brain/lib/brain/ml/ouro/model.ex` | Comment: "legacy" backend. Sidecar is the intended path. | **legacy** |
| `Brain.FactDatabase.Fact.from_map/1` legacy keys | `apps/brain/lib/brain/fact_database/fact.ex` | Handles `entity_type` vs `entity` legacy map shape. | **legacy compat** |
| `Brain.ML.Gazetteer` single-entity format | `apps/brain/lib/brain/ml/gazetteer.ex` | Comment: "legacy single-entity format". | **legacy compat** |

---

## 6. Mix tasks — exhaustive standalone vs alias-driven inventory

All mix task modules have **zero incoming xref edges** (normal—they are CLI entry points).

**Alias / setup wiring is broader than a short list:** Root `mix.exs` pulls many tasks indirectly—e.g. `setup` chains `atlas.setup`, `atlas.seed`, corpus downloads, `gen_framing_data`, `gen_micro_data`, `train`, etc. Other aliases include `train.quick`, `train.tfidf`, `models.upload` / `models.download`, `world.*`, `atlas.*`. So tasks are **not** split cleanly into “6 aliased” vs “53 orphan”; many are reachable from **umbrella aliases or nested `setup`**, others only from manual `mix …` invocation.

There are **81** `Mix.Tasks.*` modules under `apps/*/lib`. This is the exhaustive inventory:

| Category | Task modules |
|----------|--------------|
| **Evaluation** | `Mix.Tasks.Evaluate`, `Mix.Tasks.Evaluate.Classifiers`, `Mix.Tasks.Evaluate.Gate`, `Mix.Tasks.Evaluate.Intent`, `Mix.Tasks.Evaluate.Ner`, `Mix.Tasks.Evaluate.Sentiment`, `Mix.Tasks.Evaluate.SpeechAct` |
| **Data generation** | `Mix.Tasks.GenFramingData`, `Mix.Tasks.GenMicroData`, `Mix.Tasks.GenerateCountriesCapitals`, `Mix.Tasks.GenerateFortune500`, `Mix.Tasks.GenerateGoldStandard`, `Mix.Tasks.GenerateIntentData`, `Mix.Tasks.GenerateNewsSources`, `Mix.Tasks.GeneratePersonNames` |
| **Migration / cleanup** | `Mix.Tasks.CleanupGoldStandard`, `Mix.Tasks.CleanupWorlds`, `Mix.Tasks.ClearKnowledge`, `Mix.Tasks.MigrateFacts`, `Mix.Tasks.MigrateGoldStandard`, `Mix.Tasks.MigrateTrainingData`, `Mix.Tasks.NormalizeGoldStandard` |
| **Training** | `Mix.Tasks.RegenerateTestModels`, `Mix.Tasks.RebuildGoldStandard`, `Mix.Tasks.Rebuild.SpeechActGold`, `Mix.Tasks.Train`, `Mix.Tasks.TrainFraming`, `Mix.Tasks.TrainFromGraph`, `Mix.Tasks.TrainKgLstm`, `Mix.Tasks.TrainMicro`, `Mix.Tasks.TrainModels`, `Mix.Tasks.TrainPoincare` |
| **Ouro / models** | `Mix.Tasks.BenchmarkOuroPipeline`, `Mix.Tasks.Models.Download`, `Mix.Tasks.Models.Upload`, `Mix.Tasks.Ouro.Download`, `Mix.Tasks.Ouro.Verify` |
| **Utility / corpus / interactive** | `Mix.Tasks.AugmentTrainingData`, `Mix.Tasks.Chat`, `Mix.Tasks.DownloadSentimentCorpus`, `Mix.Tasks.DownloadSpeechActCorpus`, `Mix.Tasks.DownloadWordnet`, `Mix.Tasks.FactDatabase`, `Mix.Tasks.IngestFramingCorpus`, `Mix.Tasks.IngestLexicon`, `Mix.Tasks.ReconcileTemplates`, `Mix.Tasks.ReportTrainingBalance`, `Mix.Tasks.SetupLexicon`, `Mix.Tasks.Snapshot.Record`, `Mix.Tasks.Split.HeldOut`, `Mix.Tasks.ValidateIntent` |
| **Audit** | `Mix.Tasks.Audit.NerGold`, `Mix.Tasks.Audit.SpeechActGold` |
| **Credo fixers** | `Mix.Tasks.CredoFix`, `Mix.Tasks.CredoFix.AliasUsage`, `Mix.Tasks.CredoFix.LargeNumbers`, `Mix.Tasks.CredoFix.LengthCheck`, `Mix.Tasks.CredoFix.MapJoin`, `Mix.Tasks.CredoFix.TrailingWhitespace`, `Mix.Tasks.CredoFix.UnusedAlias` |
| **Atlas** | `Mix.Tasks.Atlas.BootstrapAge`, `Mix.Tasks.Atlas.Import`, `Mix.Tasks.Atlas.Seed` |
| **World** | `Mix.Tasks.TrainingWorld`, `Mix.Tasks.TrainingWorld.Ambiguous`, `Mix.Tasks.TrainingWorld.Checkpoint`, `Mix.Tasks.TrainingWorld.Compare`, `Mix.Tasks.TrainingWorld.Create`, `Mix.Tasks.TrainingWorld.Destroy`, `Mix.Tasks.TrainingWorld.Entities`, `Mix.Tasks.TrainingWorld.Events`, `Mix.Tasks.TrainingWorld.Export`, `Mix.Tasks.TrainingWorld.Import`, `Mix.Tasks.TrainingWorld.Ingest`, `Mix.Tasks.TrainingWorld.List`, `Mix.Tasks.TrainingWorld.Load`, `Mix.Tasks.TrainingWorld.Merge`, `Mix.Tasks.TrainingWorld.Metrics` |
| **Domain / tasks app** | `Mix.Tasks.DomainTasks.Analyze`, `Mix.Tasks.DomainTasks.List`, `Mix.Tasks.DomainTasks.Transform` |

---

## 7. Test-only references (module exists in lib but only called from tests)

**Not listed here—production callers exist:** `Brain.Response.ChunkSegmenter` is used from `Brain.Response.TemplateBlender` (`segment_all/1`, `segment/2`). `Brain.Response.ConstraintEnforcer` is used from `Brain.Response.OuroRealizer` (`validate/2`). They have dedicated tests but are **not** test-only modules.

| Module | File | Callers |
|--------|------|---------|
| `Brain.Analysis.ProcessingTrace` | `apps/brain/lib/brain/analysis/processing_trace.ex` | Only **`processing_trace_test.exs`** references this module (`adaptive_processing_test.exs` does **not**). |
| `Brain.Analysis.BacktrackController` | `apps/brain/lib/brain/analysis/backtrack_controller.ex` | Lib: `ProcessingTrace` only. Tests: **`adaptive_processing_test.exs`** exercises `BacktrackController` directly (see describe `"BacktrackController"`). |
| `Brain.Analysis.ChunkProfile.Projections` | `apps/brain/lib/brain/analysis/chunk_profile/projections.ex` | Only `projections_test.exs` |
| `Brain.Code.Summarizer` | `apps/brain/lib/brain/code/summarizer.ex` | Only `code_modules_test.exs` (see also §1) |
| `Brain.Code.Tokenizer` | `apps/brain/lib/brain/code/tokenizer.ex` | Only `code_modules_test.exs` (see also §1) |
| `Brain.Test.AtlasSandbox` | `apps/brain/lib/brain/test/atlas_sandbox.ex` | Test support only (`brain_case.ex`, `graph_case.ex`, `conn_case.ex`) |

---

## 8. Modules with zero incoming xref edges but alive via OTP / application / framework

These appear as "zero incoming" because they are started by the supervision tree or invoked by Phoenix framework conventions, not by direct module calls:

| Module | Reason it's alive |
|--------|-------------------|
| `Brain.Application` | OTP application callback |
| `ChatWeb.Application` | OTP application callback |
| `World.Application` | OTP application callback |
| `Atlas.Application` | OTP application callback |
| `Atlas.PostgrexTypes` | Compile-time extension registration |
| `ChatWeb.ErrorHTML` | Phoenix fallback error view |

---

## 9. Release vs umbrella applications

Root `mix.exs` `releases/0` lists **`atlas`, `brain`, `world`, `tasks`, `chat_web`** only. **`:fourth_wall`** has no `mod:` in its `mix.exs`, is **not** a top-level release application, and ships as a **library** dependency of `:brain` and `:chat_web`. That explains empty-looking “shell” modules (e.g. `FourthWall` root) versus runtime OTP apps—they are tooling/libs, not a separate supervised app in the release.

---

## 10. Summary statistics

| Metric | Count |
|--------|-------|
| Total lib modules inventoried (`apps/*/lib/**/*.ex`) | 383 |
| Mix task modules inventoried | 81 |
| Total xref-tracked files | 329 |
| Files with zero incoming edges | 74 |
| Of those: Mix task files (expected) | 53 |
| Of those: OTP/framework entry points | 6 |
| Of those: **genuine dead/orphan lib code** | **~7–8** if counting only **high-confidence, non-shell** rows in §1 (Embedder, SenseDrift, Summarizer/Tokenizer, ProcessingTrace chain, Projections, ModelPreflight test-only). **~10–12** if §1 table rows are counted literally (**12 rows**, including umbrella root shells `FourthWall` / `Atlas` / `Tasks`, low-confidence `World`, mix-task-only `ConceptNetParser`, test-only `ModelPreflight`). |
| Of those: **test-only lib modules** | **~5–6** (§7; excludes ChunkSegmenter / ConstraintEnforcer) |
| Not-wired UI modules | **4** |
| Supervised/dynamic systems needing explicit disposition | **12 rows** (§3) |
| Config-optional stacks | **ModelStore off by default; Atlas auto-import/migrate off by default; epistemic on by default; Ouro flag defaults true with lazy sidecar** |
| Deprecated/legacy APIs | **12+** |
| Compiler warnings | **1** (FourthWall `&Mix.shell().info/1` syntax) |

### 10a. Cross-check: modules confirmed not dead (additional review)

Spot-checks that did **not** surface new orphan modules:

- **`Brain.Analysis.Progress`** — Alive via alias as `Progress.report/3` from `pipeline.ex`, `racing_analyzer.ex`, and `brain.ex` (searching only the string `Brain.Analysis.Progress` understates usage).
- **`Brain.ML.NLPPipeline`** — Defined in `nlp_pipeline.ex` as `Brain.ML.NLPPipeline` (capitalization matters); actively used from `brain.ex` and consistency checker—**not** `Brain.ML.NlpPipeline`.

---

## Recommended actions (for discussion)

1. **Remove dead code:** `SenseDrift`, `Code.Tokenizer`, `Code.Summarizer`, `BacktrackController`, `ProcessingTrace`, `ChunkProfile.Projections` — unless there are planned features that need them. (`KnowledgeGraph.Embedder` is now actively used by the KG Signal Strengthening feature.)

2. **Remove unwired UI:** `PageHTML` + `home.html.heex`, `Mailer`. Either reconnect `BrainChannel` to the JS client or remove the channel + socket.

3. **Resolve disconnected supervised systems:** **`OutcomeLearner.Store`:** wire `OutcomeLearner`’s `increment_pattern_success` / `get_pattern_success_count` to `Store.increment/1` and `Store.get_count/1` (see §3), or remove the GenServer if keeping process-local counts. Also decide on `FramingDetector`, the subprocess API stack, and the `World.CodeContext` chain. Keep `LearningTriggers`, `EntityPromoter`, and `ContradictionHandler` only if their PubSub/timer/callback behaviour is intentionally part of runtime.

4. **Decide on Ouro:** Defaults: **`ouro_enabled` is `true`** unless env sets otherwise; **sidecar is lazy** (`ouro_auto_start` defaults `false`). Clarify ops docs (lazy start vs kill-switch) before removing anything.

5. **Epistemic stack:** Defaults to **enabled** (`Config` `:enabled` defaults `true`). Review whether a **kill switch** in prod is desired; modules no-op only when `:epistemic` is explicitly disabled at runtime—not “off by default.”

6. **Clean up deprecated APIs:** Remove `_legacy` functions, `get_lstm_status`, deprecated `save_models/2`, old `check_fast_path/3`.

7. **Consider `ModelStore`:** It's `enabled: false` by default but used by 10+ modules that all gracefully handle `:disabled`. Decide if S3 model storage is needed or if local-only is the path.

8. **Root modules:** `Atlas` / `Tasks` shells are optional API polish. **`FourthWall`** is a **library app** (no release `mod:`); its root module is mostly branding/version—removal is low priority unless you want a smaller public surface.
