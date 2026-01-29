defmodule ChatBot.DataDrivenTestSetup do
  @moduledoc """
  Shared setup for data-driven tests that require ML models and GenServer dependencies.
  """

  import ChatBot.TestHelpers

  @doc """
  Sets up all required services for ML-dependent tests.
  Call this in your test's setup block.
  """
  def setup_ml_services do
    ensure_pubsub_started()

    # Configure ML settings
    Application.put_env(:chat_bot, :ml,
      enabled: true,
      confidence_threshold: 0.5,
      models_path: "priv/ml_models",
      training_data_path: "data"
    )

    # Start core services
    ensure_started(ChatBot.ML.Gazetteer)
    ensure_started(ChatBot.ML.IntentClassifierSimple)
    ensure_started(ChatBot.KnowledgeStore)
    ensure_started(ChatBot.MemoryStore)
    ensure_started(ChatBot.FactDatabase)
    ensure_started(ChatBot.Memory.Embedder)
    ensure_started(ChatBot.Memory.Store)
    ensure_started(ChatBot.Analysis.LearningStore)

    # Load models
    load_ml_models()

    :ok
  end

  @doc """
  Loads ML models. Returns :ok even if some models fail to load.
  """
  def load_ml_models do
    # Load intent classifier models
    try do
      ChatBot.ML.IntentClassifierSimple.load_models()
    catch
      _, _ -> :ok
    end

    # Load entity maps
    try do
      ChatBot.ML.EntityExtractor.load_entity_maps()
    catch
      _, _ -> :ok
    end

    # Load Gazetteer data
    try do
      ChatBot.ML.Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  @doc """
  Sets up services for epistemic/belief-related tests.
  """
  def setup_epistemic_services do
    ensure_pubsub_started()

    ensure_started(ChatBot.Epistemic.JTMS)
    ensure_started(ChatBot.Epistemic.BeliefStore)
    ensure_started(ChatBot.Epistemic.ContradictionHandler)

    :ok
  end

  @doc """
  Sets up services for memory-related tests.
  """
  def setup_memory_services do
    ensure_pubsub_started()

    ensure_started(ChatBot.Memory.Embedder)
    ensure_started(ChatBot.Memory.Store)

    :ok
  end

  @doc """
  Sets up services for learning-related tests.
  """
  def setup_learning_services do
    ensure_pubsub_started()

    ensure_started(ChatBot.ML.Gazetteer)
    ensure_started(ChatBot.Learning.WorldManager)

    # Load gazetteer data for entity discovery
    try do
      ChatBot.ML.Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  @doc """
  Sets up all services - use for comprehensive integration tests.
  """
  def setup_all_services do
    setup_ml_services()
    setup_epistemic_services()
    setup_learning_services()
    :ok
  end

  @doc """
  Checks if ML models are available.
  """
  def ml_models_available? do
    File.exists?("priv/ml_models/classifier.term")
  end

  @doc """
  Checks if POS model is available.
  """
  def pos_model_available? do
    File.exists?("priv/ml_models/pos_model.term")
  end
end
