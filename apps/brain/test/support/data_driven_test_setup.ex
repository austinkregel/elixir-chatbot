defmodule Brain.DataDrivenTestSetup do
  @moduledoc "Shared setup for data-driven tests that require ML models and GenServer dependencies.\n"

  alias Brain.ML.Gazetteer
  alias Brain.ML.EntityExtractor
  alias Brain.ML.IntentClassifierSimple
  import Brain.TestHelpers

  @doc "Sets up all required services for ML-dependent tests.\nCall this in your test's setup block.\n"
  def setup_ml_services do
    ensure_pubsub_started()

    Application.put_env(:chat_bot, :ml,
      enabled: true,
      confidence_threshold: 0.5,
      models_path: "priv/ml_models",
      training_data_path: "data"
    )

    ensure_started(Brain.ML.Gazetteer)
    ensure_started(Brain.ML.IntentClassifierSimple)
    ensure_started(Brain.KnowledgeStore)
    ensure_started(Brain.MemoryStore)
    ensure_started(Brain.FactDatabase)
    ensure_started(Brain.Memory.Embedder)
    ensure_started(Brain.Memory.Store)
    ensure_started(Brain.Analysis.LearningStore)
    load_ml_models()

    :ok
  end

  @doc "Loads ML models via ModelFactory. Returns :ok even if some models fail to load.\n"
  def load_ml_models do
    Brain.Test.ModelFactory.train_and_load_test_models()

    try do
      EntityExtractor.load_entity_maps()
    catch
      _, _ -> :ok
    end

    try do
      Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  @doc "Sets up services for epistemic/belief-related tests.\n"
  def setup_epistemic_services do
    ensure_pubsub_started()

    ensure_started(Brain.Epistemic.JTMS)
    ensure_started(Brain.Epistemic.BeliefStore)
    ensure_started(Brain.Epistemic.ContradictionHandler)

    :ok
  end

  @doc "Sets up services for memory-related tests.\n"
  def setup_memory_services do
    ensure_pubsub_started()

    ensure_started(Brain.Memory.Embedder)
    ensure_started(Brain.Memory.Store)

    :ok
  end

  @doc "Sets up services for learning-related tests.\n"
  def setup_learning_services do
    ensure_pubsub_started()

    ensure_started(Brain.ML.Gazetteer)
    ensure_started(World.Manager)

    try do
      Gazetteer.load_all()
    catch
      _, _ -> :ok
    end

    :ok
  end

  @doc "Sets up all services - use for comprehensive integration tests.\n"
  def setup_all_services do
    setup_ml_services()
    setup_epistemic_services()
    setup_learning_services()
    :ok
  end

  @doc "Checks if ML models are loaded and ready.\n"
  def ml_models_available? do
    IntentClassifierSimple.ready?()
  end

  @doc "Checks if POS model is available.\n"
  def pos_model_available? do
    Brain.ML.POSTagger.model_exists?()
  end
end
