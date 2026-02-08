defmodule Brain.Application do
  @moduledoc false
  alias Brain.ML.IntentClassifierSimple
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
      Brain.Metrics.Aggregator,
      {Registry, [keys: :unique, name: Brain.SubprocessRegistry]},
      Brain.Services.CredentialVault,
      Brain.Services.Cache,
      Brain.ML.InformalExpansions,
      Brain.ML.Gazetteer,
      Brain.Analysis.LearningStore,
      Brain.Analysis.AnalyzerCalibration,
      {Brain.Analysis.HeuristicStore, [seeded_path: "data/heuristics/seeded_heuristics.json"]},
      Brain.KnowledgeStore,
      Brain.FactDatabase,
      Brain.MemoryStore,
      Brain.Memory.Embedder,
      Brain.Memory.Store,
      Brain.Epistemic.JTMS,
      Brain.Epistemic.BeliefStore,
      Brain.Epistemic.UserModelStore,
      Brain.Epistemic.ContradictionHandler,
      Brain.ML.IntentClassifierSimple,
      Brain.ML.EntityExtractor,
      Brain.Code.LanguageGrammar,
      Brain.Code.CodeGazetteer,
      Brain.ML.LSTM.UnifiedModel,
      Brain.ML.LSTM.MultiTaskModel,
      Brain.Response.TemplateStore,
      Brain.Response.ChunkCompatibility,
      Brain.Response.TemplateBlender,
      Brain.Response.SemanticFactRetriever,
      Brain.Response.LSTMResponse,
      Brain.ML.TrainingServer,
      Brain.Subprocesses.Supervisor,
      {Task.Supervisor, [name: Brain.Knowledge.AgentSupervisor]},
      Brain.Knowledge.SourceReliability,
      Brain.Knowledge.ReviewQueue,
      Brain.Knowledge.LearningCenter,
      Brain.Analysis.IntentReviewQueue,
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

      if Code.ensure_loaded?(World.Embedder) do
        Embedder.init()
        Logger.info("World embedder system initialized")
      end

      Logger.info("NLP pipeline initialization complete")
    end)
  end

  defp load_classifier_fallback do
    case IntentClassifierSimple.load_models() do
      {:ok, _models} ->
        Logger.info("Intent classifier loaded via fallback")

      {:error, _} ->
        :ok
    end
  end
end