defmodule Brain.Application do
  @moduledoc false

  # These modules are in sibling umbrella apps that depend on :brain.
  # They're available at runtime but not at compile time.
  @compile {:no_warn_undefined, [World.Embedder, World.ModelRegistry]}

  require Logger

  alias World.Embedder
  alias Brain.ML.EntityExtractor
  alias World.ModelRegistry
  alias Brain.ML.Gazetteer
  alias Brain.Telemetry
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Phoenix.PubSub, [name: Brain.PubSub]},
      {Task.Supervisor, [name: Brain.AtlasTaskSupervisor]},
      Brain.Metrics.Aggregator,
      {Registry, [keys: :unique, name: Brain.SubprocessRegistry]},
      Brain.Services.CredentialVault,
      Brain.Services.Cache,
      Brain.ML.Lexicon,
      Brain.Lexicon.Loader,
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
      Brain.ML.Poincare.Embeddings,
      Brain.ML.KnowledgeGraph.TripleScorer,
      Brain.ML.MicroClassifiers,
      Brain.Analysis.FramingDetector,
      Brain.ML.SentimentClassifierSimple,
      Brain.ML.SpeechActClassifierSimple,
      Brain.ML.EntityExtractor,
      Brain.Code.LanguageGrammar,
      Brain.Code.CodeGazetteer,
      Brain.Response.TemplateStore,
      Brain.Response.ChunkCompatibility,
      Brain.Response.TemplateBlender,
      Brain.Response.SemanticFactRetriever,
      Brain.Response.DecompressorCollector,
      Brain.ML.Ouro.Model,
      Brain.ML.Ouro.SidecarLauncher,
      Brain.ML.TrainingServer,
      Brain.ML.TrainingExampleBuffer,
      Brain.Subprocesses.Supervisor,
      {Task.Supervisor, [name: Brain.Knowledge.AgentSupervisor]},
      Brain.Knowledge.SourceReliability,
      Brain.Knowledge.ReviewQueue,
      Brain.Knowledge.LearningCenter,
      Brain.Knowledge.LearningTriggers,
      Brain.Analysis.TypeHierarchy,
      {Brain, Application.get_env(:brain, :artifact_path, "priv/static/demo.echo.json")}
    ]

    opts = [strategy: :one_for_one, name: Brain.Supervisor]
    result = Supervisor.start_link(children, opts)
    Telemetry.attach_handlers()
    ml_config = Application.get_env(:brain, :ml, [])
    skip_init = Application.get_env(:brain, :skip_ml_init, false)

    if ml_config[:enabled] and not skip_init do
      init_ml_pipeline()
    end

    result
  end

  defp init_ml_pipeline do
    require Logger

    Task.start(fn ->
      Process.sleep(100)

      Logger.info("Initializing NLP pipeline...")

      case Gazetteer.load_all() do
        {:ok, stats} ->
          Logger.info("Gazetteer loaded", stats)

        {:error, reason} ->
          Logger.warning("Gazetteer loading failed: #{inspect(reason)}")
      end

      if Code.ensure_loaded?(World.ModelRegistry) and Process.whereis(World.ModelRegistry) do
        case ModelRegistry.activate_world("default") do
          {:ok, status} ->
            Logger.info("Default world models activated", status)

          {:error, reason} ->
            Logger.warning("Default world model activation failed: #{inspect(reason)}")
            load_classifier_fallback()
        end
      else
        Logger.debug("World.ModelRegistry not available, using fallback classifier loading")
        load_classifier_fallback()
      end

      if EntityExtractor.is_loaded?() do
        status = EntityExtractor.get_status()
        Logger.info("Entity extractor ready", %{entities_count: status.entities_count})
      else
        Logger.debug("Entity extractor still loading...")
      end

      # Ensure the embedder is loaded (may not be loaded by ModelRegistry)
      unless Brain.Memory.Embedder.ready?() do
        load_embedder_fallback()
      end

      if Code.ensure_loaded?(World.Embedder) do
        Embedder.init()
        Logger.info("World embedder system initialized")
      end

      Logger.info("NLP pipeline initialization complete")
    end)
  end

  defp load_classifier_fallback do
    load_embedder_fallback()
  end

  defp load_embedder_fallback do
    embedder_path =
      case Application.get_env(:brain, :ml, [])[:models_path] do
        nil -> Path.join(:code.priv_dir(:brain), "ml_models/embedder.term")
        path -> Path.join(path, "embedder.term")
      end

    if File.exists?(embedder_path) do
      case File.read(embedder_path) do
        {:ok, binary} ->
          model = :erlang.binary_to_term(binary)
          Brain.Memory.Embedder.load_model(model)
          Logger.info("Embedder vocabulary loaded via fallback")

        {:error, reason} ->
          Logger.warning("Failed to read embedder model: #{inspect(reason)}")
      end
    else
      Logger.debug("No embedder model found at #{embedder_path}")
    end
  end
end
