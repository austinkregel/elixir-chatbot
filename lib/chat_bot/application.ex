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
      # Start the Response Template Store (loads templates from intent files)
      ChatBot.Response.TemplateStore,
      # Start the Subprocess Supervisor
      ChatBot.Subprocesses.Supervisor,
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

    # Use the new NLP pipeline initialization
    Task.start(fn ->
      # Give the Gazetteer GenServer time to start
      Process.sleep(100)

      Logger.info("Initializing NLP pipeline...")

      # Load gazetteer data
      case ChatBot.ML.Gazetteer.load_all() do
        {:ok, stats} ->
          Logger.info("Gazetteer loaded", stats)

        {:error, reason} ->
          Logger.warning("Gazetteer loading failed: #{inspect(reason)}")
      end

      # Load intent classifier
      case ChatBot.ML.IntentClassifierSimple.load_models() do
        {:ok, _models} ->
          Logger.info("Intent classifier loaded successfully")

        {:error, reason} ->
          Logger.warning("Intent classifier not found: #{inspect(reason)}")
          Logger.warning("Run 'mix train_models' to train models.")
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

      Logger.info("NLP pipeline initialization complete")
    end)
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

      # Load training data into memory system for classification
      case ChatBot.Memory.Think.load_training_data() do
        {:ok, count} ->
          Logger.info("Cognitive memory loaded #{count} episodes from training data")

          # Run initial consolidation to create semantic facts
          case ChatBot.Memory.Think.think(:consolidate, %{threshold: 0.7, min_size: 3}) do
            {:ok, {:consolidated, new_facts}} ->
              Logger.info("Consolidated #{new_facts} semantic facts")

            _ ->
              :ok
          end

        {:error, reason} ->
          Logger.warning("Failed to load cognitive memory: #{inspect(reason)}")
      end

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
