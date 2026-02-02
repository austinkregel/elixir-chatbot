defmodule Brain.DataDrivenTestSetup do
  @moduledoc """
  Shared setup for data-driven tests that require ML models and GenServer dependencies.
  """

  import Brain.TestHelpers

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
    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.ML.IntentClassifierSimple)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.MemoryStore)
    ensure_started(Brain.FactDatabase)
    ensure_started(Brain.Memory.Embedder)
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.Analysis.LearningStore)

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
      Brain.ML.IntentClassifierSimple.load_models()
    catch
      _, _ -> :ok
    end

    # Load entity maps
    try do
      Brain.ML.EntityExtractor.load_entity_maps()
    catch
      _, _ -> :ok
    end

    # Load Gazetteer data
    try do
      Brain.ML.Gazetteer.load_all()
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

    ensure_started(Brain.Epistemic.JTMS)
    ensure_started(Brain.Epistemic.BeliefStore)
    ensure_started(Brain.Epistemic.ContradictionHandler)

    :ok
  end

  @doc """
  Sets up services for memory-related tests.
  """
  def setup_memory_services do
    ensure_pubsub_started()

    ensure_started(Brain.Memory.Embedder)
    ensure_started(Brain.Memory.Store)

    :ok
  end

  @doc """
  Sets up services for learning-related tests.
  """
  def setup_learning_services do
    ensure_pubsub_started()

    ensure_started(Brain.ML.Gazetteer)
    ensure_started(World.Manager)

    # Load gazetteer data for entity discovery
    try do
      Brain.ML.Gazetteer.load_all()
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
