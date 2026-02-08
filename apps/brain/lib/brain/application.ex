defmodule Brain.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # PubSub for inter-app communication - must start first
      {Phoenix.PubSub, name: Brain.PubSub},

      # Metrics aggregator for telemetry
      Brain.Metrics.Aggregator,

      # Registry for subprocesses
      {Registry, keys: :unique, name: Brain.SubprocessRegistry},

      # ML components
      Brain.ML.InformalExpansions,
      Brain.ML.Gazetteer,

      # Analysis components
      Brain.Analysis.LearningStore,
      Brain.Analysis.AnalyzerCalibration,
      {Brain.Analysis.HeuristicStore, seeded_path: "data/heuristics/seeded_heuristics.json"},

      # Knowledge stores
      Brain.KnowledgeStore,
      Brain.FactDatabase,
      Brain.MemoryStore,

      # Memory system
      Brain.Memory.Embedder,
      Brain.Memory.Store,

      # Epistemic system
      Brain.Epistemic.JTMS,
      Brain.Epistemic.BeliefStore,
      Brain.Epistemic.UserModelStore,
      Brain.Epistemic.ContradictionHandler,

      # ML classifiers
      Brain.ML.IntentClassifierSimple,
      Brain.ML.EntityExtractor,

      # Code analysis components
      Brain.Code.LanguageGrammar,
      Brain.Code.CodeGazetteer,

      # LSTM unified model (NER, intent, sentiment, speech act)
      Brain.ML.LSTM.UnifiedModel,

      # LSTM multi-task model (intent, NER, POS via shared encoder)
      Brain.ML.LSTM.MultiTaskModel,

      # Response system
      Brain.Response.TemplateStore,
      Brain.Response.ChunkCompatibility,
      Brain.Response.TemplateBlender,
      Brain.Response.SemanticFactRetriever,

      # LSTM response scorer and refinement
      Brain.Response.LSTMResponse,

      # ML training job server
      Brain.ML.TrainingServer,

      # Subprocess supervisor
      Brain.Subprocesses.Supervisor,

      # Knowledge system
      {Task.Supervisor, name: Brain.Knowledge.AgentSupervisor},
      Brain.Knowledge.SourceReliability,
      Brain.Knowledge.ReviewQueue,
      Brain.Knowledge.LearningCenter,

      # Intent review system
      Brain.Analysis.IntentReviewQueue,

      # Main Brain GenServer
      {Brain, Application.get_env(:brain, :artifact_path, "priv/static/demo.echo.json")}
    ]

    opts = [strategy: :one_for_one, name: Brain.Supervisor]
    result = Supervisor.start_link(children, opts)

    # Attach telemetry handlers
    Brain.Telemetry.attach_handlers()

    # Initialize ML pipeline if enabled AND not in training mode
    # Training mode skips async init to avoid conflicts with training process
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
      # Give GenServers time to start
      Process.sleep(100)

      Logger.info("Initializing NLP pipeline...")

      # Load gazetteer data
      case Brain.ML.Gazetteer.load_all() do
        {:ok, stats} ->
          Logger.info("Gazetteer loaded", stats)

        {:error, reason} ->
          Logger.warning("Gazetteer loading failed: #{inspect(reason)}")
      end

      # Load default world models via WorldModelRegistry (if available)
      if Code.ensure_loaded?(World.ModelRegistry) and Process.whereis(World.ModelRegistry) do
        case World.ModelRegistry.activate_world("default") do
          {:ok, status} ->
            Logger.info("Default world models activated", status)

          {:error, reason} ->
            Logger.warning("Default world model activation failed: #{inspect(reason)}")
            load_classifier_fallback()
        end
      else
        # World.ModelRegistry not running, use fallback
        Logger.debug("World.ModelRegistry not available, using fallback classifier loading")
        load_classifier_fallback()
      end

      # EntityExtractor now loads maps automatically as a GenServer
      # Just log the status
      if Brain.ML.EntityExtractor.is_loaded?() do
        status = Brain.ML.EntityExtractor.get_status()
        Logger.info("Entity extractor ready", %{entities_count: status.entities_count})
      else
        Logger.debug("Entity extractor still loading...")
      end

      # Initialize world embedder
      if Code.ensure_loaded?(World.Embedder) do
        World.Embedder.init()
        Logger.info("World embedder system initialized")
      end

      Logger.info("NLP pipeline initialization complete")
    end)
  end

  defp load_classifier_fallback do
    case Brain.ML.IntentClassifierSimple.load_models() do
      {:ok, _models} ->
        Logger.info("Intent classifier loaded via fallback")

      {:error, _} ->
        :ok
    end
  end
end
