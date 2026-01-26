defmodule ChatBot.SystemStatus do
  @moduledoc """
  Reports the status of various background systems for UI display.
  Provides comprehensive monitoring of all GenServers in the application.
  """

  alias ChatBot.Memory.{Embedder, Store}
  alias ChatBot.Learning.WorldModelRegistry

  # All GenServers organized by category
  @genserver_categories %{
    core: [
      {ChatBot.Brain, "Brain", :has_status},
      {ChatBot.Memory.Embedder, "Memory Embedder", :has_ready},
      {ChatBot.Memory.Store, "Memory Store", :has_stats}
    ],
    epistemic: [
      {ChatBot.Epistemic.JTMS, "JTMS", :has_ready_and_stats},
      {ChatBot.Epistemic.BeliefStore, "Belief Store", :has_ready_and_stats},
      {ChatBot.Epistemic.UserModelStore, "User Model Store", :has_ready_and_stats},
      {ChatBot.Epistemic.ContradictionHandler, "Contradiction Handler", :has_ready}
    ],
    analysis: [
      {ChatBot.Analysis.LearningStore, "Learning Store", :basic},
      {ChatBot.Analysis.AnalyzerCalibration, "Analyzer Calibration", :has_stats},
      {ChatBot.Analysis.HeuristicStore, "Heuristic Store", :has_stats}
    ],
    ml: [
      {ChatBot.ML.Gazetteer, "Gazetteer", :has_stats},
      {ChatBot.ML.InformalExpansions, "Informal Expansions", :has_ready},
      {ChatBot.Response.TemplateStore, "Template Store", :has_ready}
    ],
    learning: [
      {ChatBot.Learning.WorldManager, "World Manager", :has_ready},
      {ChatBot.Learning.WorldModelRegistry, "World Model Registry", :has_ready}
    ],
    storage: [
      {ChatBot.KnowledgeStore, "Knowledge Store", :basic},
      {ChatBot.MemoryStore, "Memory Store (Legacy)", :basic},
      {ChatBot.FactDatabase, "Fact Database", :has_stats}
    ],
    metrics: [
      {ChatBot.Metrics.Aggregator, "Metrics Aggregator", :basic}
    ]
  }

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Returns a map of all system statuses (legacy format for compatibility).
  """
  def get_all do
    %{
      embedder: get_embedder_status(),
      memory_store: get_memory_store_status(),
      brain: get_brain_status(),
      nlp_pipeline: get_nlp_pipeline_status()
    }
  end

  @doc """
  Returns comprehensive status of all GenServers organized by category.
  """
  def get_all_genservers_status do
    started_at = System.monotonic_time(:millisecond)

    categories =
      @genserver_categories
      |> Enum.map(fn {category, servers} ->
        statuses =
          servers
          |> Enum.map(fn {module, name, type} ->
            {module, get_genserver_status(module, name, type)}
          end)
          |> Map.new()

        {category, statuses}
      end)
      |> Map.new()

    # Add subprocess supervisor status
    subprocess_status = get_subprocess_supervisor_status()

    elapsed = System.monotonic_time(:millisecond) - started_at

    %{
      categories: categories,
      subprocess_supervisor: subprocess_status,
      checked_at: DateTime.utc_now(),
      check_duration_ms: elapsed
    }
  end

  @doc """
  Returns performance metrics from the Metrics Aggregator.
  """
  def get_performance_metrics do
    if Code.ensure_loaded?(ChatBot.Metrics.Aggregator) do
      try do
        ChatBot.Metrics.Aggregator.get_metrics()
      catch
        :exit, _ -> default_metrics()
      end
    else
      default_metrics()
    end
  end

  @doc """
  Returns health indicators for the system.
  """
  def get_health_indicators do
    genserver_status = get_all_genservers_status()
    supervisor_info = get_supervisor_info()

    # Count running/total GenServers
    {running, total} = count_genserver_health(genserver_status.categories)

    # Calculate health score (0-100)
    health_score = if total > 0, do: round(running / total * 100), else: 0

    %{
      genservers_running: running,
      genservers_total: total,
      health_score: health_score,
      health_status: health_status_label(health_score),
      supervisor: supervisor_info,
      uptime_seconds: get_uptime_seconds(),
      checked_at: DateTime.utc_now()
    }
  end

  @doc """
  Returns the embedder status with detailed initialization progress.
  """
  def get_embedder_status do
    if Process.whereis(Embedder) do
      # Get detailed status which includes progress info
      detailed = Embedder.get_status()

      %{
        running: true,
        ready: detailed.ready,
        status: phase_to_status(detailed.phase),
        label: build_embedder_label(detailed),
        # Detailed progress info
        phase: detailed.phase,
        phase_label: detailed.phase_label,
        progress: detailed.progress,
        vocabulary_size: detailed.vocabulary_size,
        elapsed_ms: detailed.elapsed_ms
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started",
        phase: :not_started,
        phase_label: "Not started",
        progress: nil,
        vocabulary_size: 0,
        elapsed_ms: nil
      }
    end
  end

  defp phase_to_status(:ready), do: :ready
  defp phase_to_status(:idle), do: :idle
  defp phase_to_status(:not_started), do: :not_started
  defp phase_to_status(:busy), do: :building_vocabulary
  defp phase_to_status(_), do: :building_vocabulary

  defp build_embedder_label(%{ready: true, vocabulary_size: size}) do
    "Ready (#{size} terms)"
  end

  defp build_embedder_label(%{phase: :idle}) do
    "Idle (on-demand)"
  end

  defp build_embedder_label(%{phase: :busy}) do
    "Processing (busy)..."
  end

  defp build_embedder_label(%{
         phase: phase,
         phase_label: label,
         progress: progress,
         elapsed_ms: elapsed
       }) do
    base = label || phase_label(phase)

    progress_str =
      if progress && progress.percent do
        " (#{progress.percent}%)"
      else
        ""
      end

    elapsed_str =
      if elapsed && elapsed > 1000 do
        " - #{Float.round(elapsed / 1000, 1)}s"
      else
        ""
      end

    "#{base}#{progress_str}#{elapsed_str}"
  end

  defp phase_label(:tokenizing), do: "Tokenizing texts"
  defp phase_label(:building_frequencies), do: "Building frequencies"
  defp phase_label(:calculating_idf), do: "Calculating IDF weights"
  defp phase_label(_), do: "Initializing"

  @doc """
  Returns the memory store status.
  """
  def get_memory_store_status do
    if Process.whereis(Store) do
      stats =
        try do
          Store.stats()
        catch
          :exit, _ -> %{episode_count: 0, semantic_count: 0}
        end

      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready",
        episodes: Map.get(stats, :episode_count, 0),
        semantics: Map.get(stats, :semantic_count, 0)
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started",
        episodes: 0,
        semantics: 0
      }
    end
  end

  @doc """
  Returns the brain status.
  """
  def get_brain_status do
    if Process.whereis(ChatBot.Brain) do
      %{
        running: true,
        ready: true,
        status: :ready,
        label: "Ready"
      }
    else
      %{
        running: false,
        ready: false,
        status: :not_started,
        label: "Not started"
      }
    end
  end

  @doc """
  Returns the NLP pipeline status.
  """
  def get_nlp_pipeline_status do
    # Check if models are loaded
    classifier_ready =
      try do
        ChatBot.ML.IntentClassifierSimple.is_loaded?()
      catch
        :exit, _ -> false
      end

    gazetteer_ready =
      try do
        ChatBot.ML.Gazetteer.is_loaded?()
      catch
        :exit, _ -> false
      end

    all_ready = classifier_ready and gazetteer_ready

    %{
      running: true,
      ready: all_ready,
      status: if(all_ready, do: :ready, else: :loading),
      label: if(all_ready, do: "Ready", else: "Loading models..."),
      components: %{
        intent_classifier: classifier_ready,
        gazetteer: gazetteer_ready
      }
    }
  end

  @doc """
  Returns true if all systems are ready.
  Checks core systems, NLP pipeline, and ML models.

  Note: The embedder is optional - it's initialized on-demand when
  episodic memory is used. The system works without it (graceful degradation).
  """
  def all_ready? do
    status = get_all()
    models = get_ml_models_status()

    # Core systems (embedder is optional - it's on-demand)
    # The embedder is only required if it's actively building vocabulary
    # Idle state means it's not needed yet, ready means it's trained
    embedder_ok = status.embedder.ready or status.embedder.phase == :idle

    core_ready =
      embedder_ok and
        status.memory_store.ready and
        status.brain.ready

    # NLP pipeline
    nlp_ready = status.nlp_pipeline.ready

    # ML models (agents must be loaded)
    models_ready =
      models.intent_classifier.loaded and
        models.entity_extractor.loaded

    # Template store
    template_ready = safe_call_ready(ChatBot.Response.TemplateStore)

    core_ready and nlp_ready and models_ready and template_ready
  end

  @doc """
  Returns detailed readiness status for all subsystems.
  Useful for debugging what's still initializing.

  Options:
    - world_id: Get world-specific embedder status (default: "default")
  """
  def get_readiness_details(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, "default")

    status = get_all()
    models = get_ml_models_status()
    embedder_status = get_embedder_status()
    world_embedder_status = get_world_embedder_status(world_id)
    world_models_status = get_world_models_status(world_id)

    %{
      core: %{
        embedder: status.embedder.ready or status.embedder.phase == :idle,
        memory_store: status.memory_store.ready,
        brain: status.brain.ready
      },
      # Global embedder status (legacy, for backward compatibility)
      embedder_details: %{
        ready: embedder_status.ready,
        phase: embedder_status.phase,
        phase_label: embedder_status.phase_label,
        label: embedder_status.label,
        progress: embedder_status.progress,
        vocabulary_size: embedder_status.vocabulary_size,
        elapsed_ms: embedder_status.elapsed_ms
      },
      # World-specific embedder status
      world_embedder: world_embedder_status,
      # World-specific ML models status
      world_models: world_models_status,
      nlp_pipeline: %{
        ready: status.nlp_pipeline.ready,
        components: status.nlp_pipeline.components
      },
      ml_models: %{
        intent_classifier: models.intent_classifier.loaded,
        entity_extractor: models.entity_extractor.loaded,
        pos_model_exists: models.pos_model.exists,
        entity_model_exists: models.entity_model.exists,
        classifier_exists: models.classifier.exists
      },
      template_store: safe_call_ready(ChatBot.Response.TemplateStore),
      all_ready: all_ready?()
    }
  end

  @doc """
  Returns the status of the world-specific embedder.
  """
  def get_world_embedder_status(world_id) do
    alias ChatBot.Learning.WorldEmbedder

    status = WorldEmbedder.get_status(world_id)

    %{
      world_id: world_id,
      ready: status.ready,
      phase: status.phase,
      phase_label: status.phase_label,
      vocabulary_size: status.vocabulary_size,
      episode_count: status.episode_count,
      built_at: status.built_at,
      label: build_world_embedder_label(status)
    }
  end

  defp build_world_embedder_label(%{ready: true, vocabulary_size: size, episode_count: count}) do
    "Ready (#{size} terms from #{count} episodes)"
  end

  defp build_world_embedder_label(%{phase: :not_initialized}) do
    "Not initialized (will build on first use)"
  end

  defp build_world_embedder_label(%{phase: :no_data}) do
    "No training data"
  end

  defp build_world_embedder_label(%{phase: phase, phase_label: label}) do
    label || world_embedder_phase_label(phase)
  end

  # ============================================================================
  # World Model Status
  # ============================================================================

  @doc """
  Returns the status of world models from the WorldModelRegistry.
  """
  def get_world_models_status(world_id \\ nil) do
    target_world_id = world_id || get_active_world_id()

    if Process.whereis(WorldModelRegistry) do
      try do
        WorldModelRegistry.get_world_status(target_world_id)
      catch
        :exit, _ ->
          default_world_models_status(target_world_id)
      end
    else
      default_world_models_status(target_world_id)
    end
  end

  @doc """
  Returns the status of all loaded world models.
  """
  def get_all_world_models_status do
    if Process.whereis(WorldModelRegistry) do
      try do
        WorldModelRegistry.get_all_status()
      catch
        :exit, _ ->
          %{active_world_id: "default", loaded_worlds: [], loading: [], ready: false, worlds: %{}}
      end
    else
      %{active_world_id: "default", loaded_worlds: [], loading: [], ready: false, worlds: %{}}
    end
  end

  @doc """
  Returns the currently active world ID from the WorldModelRegistry.
  """
  def get_active_world_id do
    if Process.whereis(WorldModelRegistry) do
      try do
        WorldModelRegistry.get_active_world()
      catch
        :exit, _ -> "default"
      end
    else
      "default"
    end
  end

  defp default_world_models_status(world_id) do
    %{
      world_id: world_id,
      is_active: false,
      is_loaded: false,
      is_loading: false,
      has_classifier: false,
      has_embedder: false,
      has_pos_model: false,
      has_entity_model: false,
      classifier_vocab_size: 0,
      embedder_vocab_size: 0
    }
  end

  defp world_embedder_phase_label(:loading_episodes), do: "Loading episodes"
  defp world_embedder_phase_label(:tokenizing), do: "Tokenizing texts"
  defp world_embedder_phase_label(:building_frequencies), do: "Building frequencies"
  defp world_embedder_phase_label(:calculating_idf), do: "Calculating IDF weights"
  defp world_embedder_phase_label(:ready), do: "Ready"
  defp world_embedder_phase_label(_), do: "Initializing"

  @doc """
  Returns the status of all ML models (file-based and agent-based).
  """
  def get_ml_models_status do
    models_path = get_models_path()

    %{
      # File-based models
      pos_model: get_model_file_status(models_path, "pos_model.term"),
      entity_model: get_model_file_status(models_path, "entity_model.term"),
      classifier: get_model_file_status(models_path, "classifier.term"),
      gazetteer: get_model_file_status(models_path, "gazetteer.term"),
      # Agent-based models (runtime loaded)
      intent_classifier: get_agent_status(ChatBot.ML.IntentClassifierSimple),
      entity_extractor: get_agent_status(ChatBot.ML.EntityExtractor),
      checked_at: DateTime.utc_now()
    }
  end

  @doc """
  Returns training history/stats if available.
  """
  def get_training_stats do
    try do
      ChatBot.Metrics.Aggregator.get_metrics()
      |> Map.take([:pos_train, :entity_train, :classifier_train, :model_load])
      |> Enum.filter(fn {_k, v} -> v != nil end)
      |> Map.new()
    catch
      :exit, _ -> %{}
    end
  end

  @doc """
  Returns status of training worlds (self-learning system).
  """
  def get_training_worlds_status do
    alias ChatBot.Learning.{WorldManager, WorldMetrics}

    if Process.whereis(WorldManager) do
      try do
        worlds = WorldManager.list_worlds()

        world_summaries =
          Enum.map(worlds, fn world ->
            metrics =
              case WorldManager.get_metrics(world.id) do
                {:ok, m} -> WorldMetrics.summary(m)
                _ -> nil
              end

            candidates_count = length(WorldManager.get_candidates(world.id, limit: 1000))

            %{
              id: world.id,
              name: world.name,
              mode: world.mode,
              created_at: world.created_at,
              metrics: metrics,
              candidates_count: candidates_count
            }
          end)

        # Get persisted worlds count
        persisted_count =
          try do
            ChatBot.Learning.WorldPersistence.list_persisted_worlds() |> length()
          catch
            _, _ -> 0
          end

        %{
          manager_ready: WorldManager.ready?(),
          active_worlds: length(worlds),
          persisted_worlds: persisted_count,
          worlds: world_summaries,
          checked_at: DateTime.utc_now()
        }
      catch
        :exit, _ ->
          %{
            manager_ready: false,
            active_worlds: 0,
            persisted_worlds: 0,
            worlds: [],
            checked_at: DateTime.utc_now()
          }
      end
    else
      %{
        manager_ready: false,
        active_worlds: 0,
        persisted_worlds: 0,
        worlds: [],
        checked_at: DateTime.utc_now()
      }
    end
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp get_models_path do
    Application.get_env(:chat_bot, :ml)[:models_path] || "priv/ml_models"
  end

  defp get_model_file_status(models_path, filename) do
    path = Path.join(models_path, filename)

    case File.stat(path) do
      {:ok, stat} ->
        %{
          exists: true,
          loaded: check_model_loaded(filename),
          size_bytes: stat.size,
          modified_at: stat.mtime |> NaiveDateTime.from_erl!() |> DateTime.from_naive!("Etc/UTC"),
          path: path
        }

      {:error, _} ->
        %{
          exists: false,
          loaded: false,
          size_bytes: 0,
          modified_at: nil,
          path: path
        }
    end
  end

  defp check_model_loaded(filename) do
    case filename do
      "classifier.term" ->
        try do
          ChatBot.ML.IntentClassifierSimple.is_loaded?()
        catch
          _, _ -> false
        end

      "gazetteer.term" ->
        try do
          ChatBot.ML.Gazetteer.is_loaded?()
        catch
          _, _ -> false
        end

      "entity_model.term" ->
        # Entity model is loaded into EntityExtractor agent
        Process.whereis(ChatBot.ML.EntityExtractor) != nil

      "pos_model.term" ->
        # POS model exists on disk but loaded on-demand
        ChatBot.ML.POSTagger.model_exists?()

      _ ->
        false
    end
  end

  defp get_agent_status(module) do
    pid = Process.whereis(module)

    if pid do
      process_info = get_process_info(pid)

      %{
        loaded: true,
        pid: pid,
        memory_bytes: process_info[:memory],
        message_queue_len: process_info[:message_queue_len]
      }
    else
      %{
        loaded: false,
        pid: nil,
        memory_bytes: nil,
        message_queue_len: nil
      }
    end
  end

  defp get_genserver_status(module, name, type) do
    pid = Process.whereis(module)

    base_status = %{
      name: name,
      module: module,
      running: pid != nil,
      pid: pid,
      ready: false,
      status: :not_started,
      label: "Not started",
      stats: nil,
      memory_bytes: nil,
      message_queue_len: nil
    }

    if pid do
      # Get process info (non-blocking)
      process_info = get_process_info(pid)

      status_with_info = %{
        base_status
        | running: true,
          status: :running,
          label: "Running",
          memory_bytes: process_info[:memory],
          message_queue_len: process_info[:message_queue_len]
      }

      # Add type-specific information
      enhance_status(status_with_info, module, type)
    else
      base_status
    end
  end

  defp enhance_status(status, _module, :basic) do
    %{status | ready: true, status: :ready, label: "Ready"}
  end

  defp enhance_status(status, ChatBot.Memory.Embedder = _module, :has_ready) do
    # Special handling for Embedder to show detailed progress
    embedder_status = get_embedder_status()

    %{
      status
      | ready: embedder_status.ready,
        status: embedder_status.status,
        label: embedder_status.label,
        stats: %{
          phase: embedder_status.phase,
          vocabulary: embedder_status.vocabulary_size,
          progress: embedder_status.progress
        }
    }
  end

  defp enhance_status(status, module, :has_ready) do
    ready = safe_call_ready(module)

    %{
      status
      | ready: ready,
        status: if(ready, do: :ready, else: :initializing),
        label: if(ready, do: "Ready", else: "Initializing...")
    }
  end

  defp enhance_status(status, module, :has_stats) do
    stats = safe_call_stats(module)

    %{
      status
      | ready: true,
        status: :ready,
        label: "Ready",
        stats: stats
    }
  end

  defp enhance_status(status, module, :has_ready_and_stats) do
    ready = safe_call_ready(module)
    stats = if ready, do: safe_call_stats(module), else: nil

    %{
      status
      | ready: ready,
        status: if(ready, do: :ready, else: :initializing),
        label: if(ready, do: "Ready", else: "Initializing..."),
        stats: stats
    }
  end

  defp enhance_status(status, module, :has_status) do
    # For Brain which has get_status
    brain_status = safe_call_brain_status(module)

    %{
      status
      | ready: true,
        status: :ready,
        label: "Ready",
        stats: brain_status
    }
  end

  defp safe_call_ready(module) do
    try do
      module.ready?()
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
      :exit, _ -> false
    end
  end

  defp safe_call_stats(module) do
    try do
      module.stats()
    catch
      :exit, {:timeout, _} -> nil
      :exit, {:noproc, _} -> nil
      :exit, _ -> nil
    end
  end

  defp safe_call_brain_status(module) do
    try do
      module.get_status()
    catch
      :exit, {:timeout, _} -> nil
      :exit, {:noproc, _} -> nil
      :exit, _ -> nil
    end
  end

  defp get_process_info(pid) do
    try do
      Process.info(pid, [:memory, :message_queue_len]) || []
    catch
      _, _ -> []
    end
  end

  defp get_subprocess_supervisor_status do
    pid = Process.whereis(ChatBot.Subprocesses.Supervisor)

    if pid do
      children =
        try do
          DynamicSupervisor.count_children(ChatBot.Subprocesses.Supervisor)
        catch
          :exit, _ -> %{active: 0, specs: 0, supervisors: 0, workers: 0}
        end

      %{
        running: true,
        pid: pid,
        active_children: children[:active] || 0,
        specs: children[:specs] || 0,
        workers: children[:workers] || 0,
        supervisors: children[:supervisors] || 0
      }
    else
      %{
        running: false,
        pid: nil,
        active_children: 0,
        specs: 0,
        workers: 0,
        supervisors: 0
      }
    end
  end

  defp get_supervisor_info do
    try do
      children = Supervisor.count_children(ChatBot.Supervisor)

      %{
        active: children[:active] || 0,
        specs: children[:specs] || 0,
        supervisors: children[:supervisors] || 0,
        workers: children[:workers] || 0
      }
    catch
      :exit, _ ->
        %{active: 0, specs: 0, supervisors: 0, workers: 0}
    end
  end

  defp count_genserver_health(categories) do
    Enum.reduce(categories, {0, 0}, fn {_category, servers}, {running, total} ->
      category_stats =
        Enum.reduce(servers, {0, 0}, fn {_module, status}, {r, t} ->
          {r + if(status.running, do: 1, else: 0), t + 1}
        end)

      {running + elem(category_stats, 0), total + elem(category_stats, 1)}
    end)
  end

  defp health_status_label(score) when score >= 90, do: :healthy
  defp health_status_label(score) when score >= 70, do: :degraded
  defp health_status_label(score) when score >= 50, do: :warning
  defp health_status_label(_score), do: :critical

  defp get_uptime_seconds do
    case :erlang.statistics(:wall_clock) do
      {uptime_ms, _} -> div(uptime_ms, 1000)
      _ -> 0
    end
  end

  defp default_metrics do
    %{
      brain_evaluate: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      pipeline_process: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      memory_query: %{count: 0, avg_ms: 0, min_ms: 0, max_ms: 0},
      errors: %{count: 0, rate_per_minute: 0.0}
    }
  end
end
