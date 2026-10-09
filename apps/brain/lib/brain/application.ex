defmodule Brain.Application do
  @moduledoc false

  alias Brain.Telemetry
  use Application

  @impl true
  def start(_type, _args) do
    Brain.Graph.ContextCache.init()

    children = [
      {Phoenix.PubSub, [name: Brain.PubSub]},
      {Task.Supervisor, [name: Brain.AtlasTaskSupervisor]},
      Brain.Metrics.Aggregator,
      {Registry, [keys: :unique, name: Brain.SubprocessRegistry]},
      Brain.Services.CredentialVault,
      Brain.Services.Cache,
      Brain.Services.LocalScheduler,
      Brain.ML.Lexicon,
      Brain.ML.POSTagger.LexicalFeatures,
      Brain.Analysis.LexicalEvidence.Loader,
      Brain.Analysis.ActiveContexts,
      Brain.Lexicon.Loader,
      Brain.Lexicon.UserDefined,
      Brain.ML.InformalExpansions,
      Brain.ML.Gazetteer,
      Brain.Analysis.LearningStore,
      Brain.Analysis.OutcomeLearner.Store,
      Brain.Analysis.AnalyzerCalibration,
      {Brain.Analysis.HeuristicStore, [seeded_path: "data/heuristics/seeded_heuristics.json"]},
      Brain.Analysis.ComprehensionAssessor,
      Brain.KnowledgeStore,
      Brain.FactDatabase,
      Brain.MemoryStore,
      Brain.Memory.Embedder,
      Brain.Memory.Store,
      Brain.Epistemic.SourceAuthority,
      Brain.Epistemic.JTMS,
      Brain.Epistemic.BeliefStore,
      Brain.Epistemic.UserModelStore,
      Brain.Epistemic.ContradictionHandler,
      Brain.Epistemic.StanceTracker,
      Brain.ML.KnowledgeGraph.EntityVectorCache,
      Brain.ML.MicroClassifiers,
      Brain.Analysis.FramingDetector,
      Brain.ML.SentimentClassifierSimple,
      Brain.ML.SpeechActClassifierSimple,
      Brain.Code.LanguageGrammar,
      Brain.Code.CodeGazetteer,
      Brain.Response.TemplateStore,
      Brain.Response.ChunkCompatibility,
      Brain.Response.TemplateBlender,
      Brain.Response.SemanticFactRetriever,
      Brain.Response.DecompressorCollector,
      Brain.Response.PhraseInventory,
      Brain.Services.HomeAssistant.CapabilityRegistry,
      Brain.ML.Ouro.Model,
      Brain.ML.Ouro.SidecarLauncher,
      {Task.Supervisor, [name: Brain.ML.TrainingServer.TaskSupervisor]},
      Brain.ML.TrainingServer,
      Brain.ML.TrainingExampleBuffer,
      {Task.Supervisor, [name: Brain.ML.WeightOptimizer.TaskSupervisor]},
      Brain.ML.WeightOptimizer.Tracker,
      {Task.Supervisor, [name: Brain.Knowledge.AgentSupervisor]},
      Brain.Knowledge.SourceReliability,
      Brain.Knowledge.ReviewQueue,
      Brain.Knowledge.LearningCenter,
      Brain.Knowledge.LearningTriggers,
      Brain.Analysis.TypeHierarchy,
      {Brain, Application.get_env(:brain, :artifact_path, "priv/static/demo.echo.json")},
      # After Brain, so it stops before Brain does (children stop in reverse
      # order): a CLI session ends its Brain conversation as it stops, which
      # needs Brain still running.
      Brain.Subprocesses.Supervisor
    ]

    # Models load in the children that own them: Brain.ML.Gazetteer reads its
    # sources in init/1, Brain.Memory.Embedder reads the trained vocabulary when
    # it starts, and a world's models are read the first time the world uses
    # them. No world is activated at boot.
    opts = [strategy: :one_for_one, name: Brain.Supervisor]
    result = Supervisor.start_link(children, opts)
    Telemetry.attach_handlers()

    result
  end
end
