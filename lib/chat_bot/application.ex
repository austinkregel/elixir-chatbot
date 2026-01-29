defmodule ChatBot.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      ChatBotWeb.Telemetry,
      {DNSCluster, query: Application.get_env(:chat_bot, :dns_cluster_query) || :ignore},
      {Phoenix.PubSub, name: ChatBot.PubSub},
      # Start the Registry for subprocesses
      {Registry, keys: :unique, name: ChatBot.SubprocessRegistry},
      # Start the Metrics Aggregator (for telemetry collection)
      ChatBot.Metrics.Aggregator,
      # Start the Informal Expansions data loader (for contraction expansion)
      ChatBot.ML.InformalExpansions,
      # Start the Gazetteer GenServer for entity lookups
      ChatBot.ML.Gazetteer,
      # Start the Analysis Learning Store
      ChatBot.Analysis.LearningStore,
      # Start the Knowledge Store
      ChatBot.KnowledgeStore,
      # Start the Fact Database
      ChatBot.FactDatabase,
      # Start the Memory Store
      ChatBot.MemoryStore,
      # Start the Cognitive Memory System (Embedder and Store)
      ChatBot.Memory.Embedder,
      ChatBot.Memory.Store,
      # Start the Epistemic System (JTMS, BeliefStore, UserModelStore, ContradictionHandler)
      ChatBot.Epistemic.JTMS,
      ChatBot.Epistemic.BeliefStore,
      ChatBot.Epistemic.UserModelStore,
      ChatBot.Epistemic.ContradictionHandler,
      # Start the Adaptive Processing System
      ChatBot.Analysis.AnalyzerCalibration,
      {ChatBot.Analysis.HeuristicStore, seeded_path: "priv/heuristics/seeded.json"},
      # Start the Intent Classifier
      ChatBot.ML.IntentClassifierSimple,
      # Start the Training World Manager
      ChatBot.Learning.WorldManager,
      # Start the World Model Registry (manages per-world ML models)
      ChatBot.Learning.WorldModelRegistry,
      # Start the Response Template Store (loads templates from intent files)
      ChatBot.Response.TemplateStore,
      # Start the Subprocess Supervisor
      ChatBot.Subprocesses.Supervisor,
      # Start the Knowledge Expansion System
      {Task.Supervisor, name: ChatBot.Knowledge.AgentSupervisor},
      ChatBot.Knowledge.SourceReliability,
      ChatBot.Knowledge.ReviewQueue,
      ChatBot.Knowledge.LearningCenter,
      # Start the Brain GenServer
      {ChatBot.Brain, "priv/static/demo.echo.json"},
      # Start to serve requests, typically the last entry
      ChatBotWeb.Endpoint
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: ChatBot.Supervisor]
    result = Supervisor.start_link(children, opts)

    # Attach telemetry handlers after supervisor is started
    ChatBot.Telemetry.attach_handlers()

    # Initialize ML models after supervisor is started
    if Application.get_env(:chat_bot, :ml)[:enabled] do
      init_ml_pipeline()
    end

    result
  end

  defp init_ml_pipeline do
    require Logger
    alias ChatBot.Learning.WorldModelRegistry

    # Use the new NLP pipeline initialization
    Task.start(fn ->
      # Give GenServers time to start
      Process.sleep(100)

      Logger.info("Initializing NLP pipeline...")

      # Load gazetteer data (global, not per-world)
      case ChatBot.ML.Gazetteer.load_all() do
        {:ok, stats} ->
          Logger.info("Gazetteer loaded", stats)

        {:error, reason} ->
          Logger.warning("Gazetteer loading failed: #{inspect(reason)}")
      end

      # Load default world models via WorldModelRegistry
      # This handles classifier, embedder, and other per-world models
      case WorldModelRegistry.activate_world("default") do
        {:ok, status} ->
          Logger.info("Default world models activated", status)

        {:error, reason} ->
          Logger.warning("Default world model activation failed: #{inspect(reason)}")
          Logger.warning("Run 'mix train_models' to train models.")

          # Fall back to loading classifier directly
          case ChatBot.ML.IntentClassifierSimple.load_models() do
            {:ok, _models} ->
              Logger.info("Intent classifier loaded via fallback")

            {:error, _} ->
              :ok
          end
      end

      # Load entity maps as fallback
      case ChatBot.ML.EntityExtractor.load_entity_maps() do
        {:ok, maps} ->
          Logger.info("Entity maps loaded", %{count: map_size(maps)})

        {:error, reason} ->
          Logger.warning("Entity maps loading failed: #{inspect(reason)}")
      end

      # Initialize cognitive memory system
      init_cognitive_memory()

      # Initialize default training world
      init_default_world()

      Logger.info("NLP pipeline initialization complete")
    end)
  end

  defp init_default_world do
    require Logger
    alias ChatBot.Learning.WorldManager

    # Wait for WorldManager to be ready
    unless Process.whereis(WorldManager) do
      Logger.debug("Waiting for WorldManager to start...")
      Process.sleep(100)
    end

    unless Process.whereis(WorldManager) do
      Logger.warning("WorldManager not available, skipping default world init")
      :ok
    else
      # Ensure default world exists
      case WorldManager.get("default") do
        {:ok, _world} ->
          Logger.debug("Default world already exists")
          :ok

        {:error, :not_found} ->
          Logger.info("Creating default training world...")

          # Create the default world with persistent mode
          # Use "default" as both name and ID for easy reference
          case WorldManager.create("default",
                 id: "default",
                 mode: :persistent,
                 base_world: nil,
                 metadata: %{description: "Default training world containing base data"}
               ) do
            {:ok, world} ->
              Logger.info("Default world created", %{id: world.id})
              :ok

            {:error, reason} ->
              Logger.warning("Failed to create default world: #{inspect(reason)}")
              :error
          end
      end
    end
  end

  defp init_cognitive_memory do
    require Logger

    # Wait briefly for Memory.Store to be available
    unless Process.whereis(ChatBot.Memory.Store) do
      Logger.debug("Waiting for Memory.Store to start...")
      Process.sleep(100)
    end

    # Skip if Store still not available (e.g., in test environment)
    unless Process.whereis(ChatBot.Memory.Store) do
      Logger.info("Memory.Store not available, skipping cognitive memory init")
      :ok
    else
      Logger.info("Initializing cognitive memory system...")

      # Initialize world-specific embedder ETS table
      ChatBot.Learning.WorldEmbedder.init()
      Logger.info("World embedder system initialized")

      # Note: We no longer load intent training data into episodic memory.
      # The intent classifier has its own TF-IDF model (classifier.term).
      # Episodic memory should only contain real user interactions,
      # which are added by the Brain during conversations.
      #
      # Embeddings are now world-specific and built lazily when needed.

      Logger.info("Cognitive memory system initialized")
    end
  end

  # Tell Phoenix to update the endpoint configuration
  # whenever the application is updated.
  @impl true
  def config_change(changed, _new, removed) do
    ChatBotWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
