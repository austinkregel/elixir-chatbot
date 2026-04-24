defmodule Brain.SystemStatus do
  @moduledoc "Reports the status of various background systems for UI display.\nProvides comprehensive monitoring of all GenServers in the application.\n"

  # These modules are in sibling umbrella apps that depend on :brain.
  # They're available at runtime but not at compile time.
  @compile {:no_warn_undefined,
            [World.ModelRegistry, World.Embedder, World.Manager, World.Persistence, World.Metrics]}

  alias Brain.Metrics.Aggregator
  alias Brain.ML.EntityExtractor
  alias Brain.ML.EntityTrainer
  alias Brain.ML.MicroClassifiers
  alias Brain.ML.POSTagger
  alias Brain.ML.Gazetteer
  alias World.ModelRegistry
  alias World.Embedder
  alias World.Manager
  alias World.Persistence
  alias World.Metrics
  alias Brain.Code.LanguageGrammar
  alias Brain.Code.CodeGazetteer
  alias Brain.ML.EvaluationStore
  alias Brain.Memory.Store

  @genserver_categories %{
    core: [
      {Brain, "Brain", :has_status},
      {Brain.Memory.Embedder, "Memory Embedder", :has_ready},
      {Brain.Memory.Store, "Memory Store", :has_stats}
    ],
    epistemic: [
      {Brain.Epistemic.JTMS, "JTMS", :has_ready_and_stats},
      {Brain.Epistemic.BeliefStore, "Belief Store", :has_ready_and_stats},
      {Brain.Epistemic.UserModelStore, "User Model Store", :has_ready_and_stats},
      {Brain.Epistemic.ContradictionHandler, "Contradiction Handler", :has_ready_and_stats}
    ],
    analysis: [
      {Brain.Analysis.LearningStore, "Learning Store", :has_ready},
      {Brain.Analysis.AnalyzerCalibration, "Analyzer Calibration", :has_ready_and_stats},
      {Brain.Analysis.HeuristicStore, "Heuristic Store", :has_stats}
    ],
    ml: [
      {Brain.ML.Gazetteer, "Gazetteer", :has_stats},
      {Brain.ML.InformalExpansions, "Informal Expansions", :has_ready},
      {Brain.ML.MicroClassifiers, "Micro Classifiers", :has_ready},
      {Brain.Response.TemplateStore, "Template Store", :has_ready}
    ],
    knowledge: [
      {Brain.Knowledge.LearningCenter, "Learning Center", :has_ready_and_stats},
      {Brain.Knowledge.ReviewQueue, "Review Queue", :has_ready_and_stats},
      {Brain.Knowledge.SourceReliability, "Source Reliability", :has_ready_and_stats}
    ],
    learning: [
      {World.Manager, "World Manager", :has_ready},
      {World.ModelRegistry, "World Model Registry", :has_ready}
    ],
    code_analysis: [
      {Brain.Code.CodeGazetteer, "Code Gazetteer", :has_stats},
      {Brain.Code.LanguageGrammar, "Language Grammar", :has_ready}
    ],
    services: [
      {Brain.Services.CredentialVault, "Credential Vault", :has_ready},
      {Brain.Services.Cache, "Service Cache", :has_ready}
    ],
    storage: [
      {Brain.KnowledgeStore, "Knowledge Store", :has_ready},
      {Brain.MemoryStore, "Memory Store (Legacy)", :has_ready},
      {Brain.FactDatabase, "Fact Database", :has_ready_and_stats}
    ],
    metrics: [{Brain.Metrics.Aggregator, "Metrics Aggregator", :basic}]
  }
  @module_metric_map %{
    Brain => :brain_evaluate,
    Brain.Analysis.Pipeline => :pipeline_process,
    Brain.Memory.Embedder => :memory_embed,
    Brain.Memory.Store => :memory_query,
    Brain.ML.Gazetteer => :gazetteer_lookup,
    Brain.ML.EntityExtractor => :entity_extract,
    Brain.Knowledge.LearningCenter => :knowledge_research,
    Brain.Knowledge.ReviewQueue => :knowledge_review,
    Brain.Epistemic.JTMS => :jtms_justify,
    Brain.Epistemic.BeliefStore => :belief_operation,
    Brain.Analysis.RacingAnalyzer => :racing_analysis,
    Brain.Code.CodeGazetteer => :code_gazetteer_lookup,
    Brain.Code.Pipeline => :code_pipeline,
    Brain.Code.Parser => :code_parse,
    Brain.Code.SymbolExtractor => :code_extract,
    Brain.Services.Dispatcher => :service_dispatch
  }

  @doc "Returns a map of all system statuses (legacy format for compatibility).\n"
  def get_all do
    %{
      embedder: get_embedder_status(),
      memory_store: get_memory_store_status(),
      brain: get_brain_status(),
      nlp_pipeline: get_nlp_pipeline_status(),
      micro_classifiers: get_micro_classifiers_status()
    }
  end

  @doc "Returns comprehensive status of all GenServers organized by category.\n"
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

    subprocess_status = get_subprocess_supervisor_status()

    elapsed = System.monotonic_time(:millisecond) - started_at

    %{
      categories: categories,
      subprocess_supervisor: subprocess_status,
      checked_at: DateTime.utc_now(),
      check_duration_ms: elapsed
    }
  end

  @doc "Returns performance metrics from the Metrics Aggregator.\n"
  def get_performance_metrics do
    if Code.ensure_loaded?(Brain.Metrics.Aggregator) do
      try do
        Aggregator.get_metrics()
      catch
        :exit, _ -> default_metrics()
      end
    else
      default_metrics()
    end
  end

  @doc "Returns health indicators for the system.\n"
  def get_health_indicators do
    genserver_status = get_all_genservers_status()
    supervisor_info = get_supervisor_info()
    {running, total} = count_genserver_health(genserver_status.categories)

    health_score =
      if total > 0 do
        round(running / total * 100)
      else
        0
      end

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

  @doc "Returns a utilization report identifying idle, low-usage, and high-cost systems.\n"
  def get_utilization_report do
    genserver_status = get_all_genservers_status()

    all_statuses =
      genserver_status.categories
      |> Enum.flat_map(fn {category, servers} ->
        Enum.map(servers, fn {_module, status} ->
          Map.put(status, :category, category)
        end)
      end)

    idle_systems =
      all_statuses
      |> Enum.filter(&(&1.utilization_status == :idle))
      |> Enum.map(&summarize_system/1)

    never_used_systems =
      all_statuses
      |> Enum.filter(&(&1.utilization_status == :never_used))
      |> Enum.map(&summarize_system/1)

    low_usage_systems =
      all_statuses
      |> Enum.filter(&(&1.utilization_status == :low_usage))
      |> Enum.map(&summarize_system/1)

    high_cost_systems =
      all_statuses
      |> Enum.filter(&(&1.utilization_status == :high_cost))
      |> Enum.map(&summarize_system/1)

    normal_systems =
      all_statuses
      |> Enum.filter(&(&1.utilization_status == :normal))
      |> Enum.map(&summarize_system/1)

    %{
      idle: idle_systems,
      never_used: never_used_systems,
      low_usage: low_usage_systems,
      high_cost: high_cost_systems,
      normal: normal_systems,
      summary: %{
        total: length(all_statuses),
        idle_count: length(idle_systems),
        never_used_count: length(never_used_systems),
        low_usage_count: length(low_usage_systems),
        high_cost_count: length(high_cost_systems),
        normal_count: length(normal_systems)
      },
      checked_at: DateTime.utc_now()
    }
  end

  @doc "Returns the embedder status with detailed initialization progress.\n"
  def get_embedder_status do
    if Code.ensure_loaded?(Brain.Memory.Embedder) and Process.whereis(Brain.Memory.Embedder) do
      detailed = Brain.Memory.Embedder.get_status()

      %{
        running: true,
        ready: detailed.ready,
        status: phase_to_status(detailed.phase),
        label: build_embedder_label(detailed),
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

  @doc "Returns the memory store status.\n"
  def get_memory_store_status do
    if Code.ensure_loaded?(Brain.Memory.Store) and Process.whereis(Brain.Memory.Store) do
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

  @doc "Returns the brain status.\n"
  def get_brain_status do
    if Code.ensure_loaded?(Brain) and Process.whereis(Brain) do
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

  @doc "Returns the NLP pipeline status.\n"
  def get_nlp_pipeline_status do
    gazetteer_ready =
      if Code.ensure_loaded?(Brain.ML.Gazetteer) do
        try do
          Gazetteer.is_loaded?()
        catch
          :exit, _ -> false
        end
      else
        false
      end

    intent_ready = micro_intent_ready?()

    %{
      running: true,
      ready: gazetteer_ready and intent_ready,
      status:
        if(gazetteer_ready) do
          :ready
        else
          :loading
        end,
      label:
        if(gazetteer_ready and intent_ready) do
          "Ready"
        else
          "Loading models..."
        end,
      components: %{
        gazetteer: gazetteer_ready,
        intent_classifier: intent_ready
      }
    }
  end

  @doc "Returns LSTM status (deprecated — LSTM modules have been removed).\n"
  def get_lstm_status do
    %{
      running: false,
      ready: false,
      status: :removed,
      label: "LSTM models removed",
      components: %{}
    }
  end

  @doc "Returns true if all systems are ready.\n"
  def all_ready? do
    status = get_all()
    models = get_ml_models_status()

    embedder_ok = status.embedder.ready or status.embedder.phase == :idle

    core_ready =
      embedder_ok and
        status.memory_store.ready and
        status.brain.ready

    nlp_ready = status.nlp_pipeline.ready

    models_ready = models.entity_extractor.loaded

    template_ready = safe_call_ready(Brain.Response.TemplateStore)

    core_ready and nlp_ready and models_ready and template_ready
  end

  @doc "Returns the status of all ML models.\n"
  def get_ml_models_status do
    models_path = get_models_path()

    %{
      pos_model: get_model_file_status(models_path, "pos_model.term"),
      entity_model: get_model_file_status(models_path, "entity_model.term"),
      gazetteer: get_model_file_status(models_path, "gazetteer.term"),
      intent_classifier: get_intent_classifier_status(models_path),
      entity_extractor: get_agent_status(Brain.ML.EntityExtractor),
      last_evaluation: get_last_evaluation(),
      checked_at: DateTime.utc_now()
    }
  end

  defp micro_intent_ready? do
    if Code.ensure_loaded?(MicroClassifiers) and Process.whereis(MicroClassifiers) do
      try do
        MicroClassifiers.ready?()
      catch
        :exit, _ -> false
      end
    else
      false
    end
  end

  defp get_intent_classifier_status(models_path) do
    rel = Path.join(["micro", "intent_full.term"])
    file_st = get_model_file_status(models_path, rel)
    loaded = file_st.exists and micro_intent_ready?()

    Map.merge(file_st, %{loaded: loaded, label: "intent_full (MicroClassifiers)"})
  end

  defp get_last_evaluation do
    try do
      EvaluationStore.latest("intent")
    rescue
      _ -> nil
    end
  end

  @doc "Returns code analysis system status.\n"
  def get_code_analysis_status do
    gazetteer_status = get_code_gazetteer_status()
    grammar_status = get_language_grammar_status()

    %{
      code_gazetteer: gazetteer_status,
      language_grammar: grammar_status,
      ready: gazetteer_status.ready and grammar_status.ready,
      checked_at: DateTime.utc_now()
    }
  end

  @doc """
  Returns comprehensive status of external services.

  Includes:
  - GenServer status (CredentialVault, Cache)
  - Registered services and their configuration status
  - Dispatch metrics (success rate, avg latency)
  - Cache performance (hit rate)
  - Health check results
  """
  def get_services_status do
    alias Brain.Services.{Dispatcher, CredentialVault, Cache}

    # GenServer statuses
    vault_ready = CredentialVault.ready?()
    cache_ready = Cache.ready?()

    # Get registered services info
    services =
      Dispatcher.list_services()
      |> Enum.map(fn service_info ->
        service_name = service_info.name

        # Get metrics for this service from aggregator
        metrics = Aggregator.get_service_metrics()
        service_metrics = get_in(metrics, [:by_service, service_name]) || %{}

        Map.merge(service_info, %{
          total_dispatches: Map.get(service_metrics, :total_dispatches, 0),
          success_count: Map.get(service_metrics, :success_count, 0),
          error_count: Map.get(service_metrics, :error_count, 0),
          success_rate: Map.get(service_metrics, :success_rate, 0.0),
          last_dispatch: Map.get(service_metrics, :last_dispatch),
          health_status: Map.get(service_metrics, :health_status)
        })
      end)

    # Overall dispatch metrics
    all_metrics = Aggregator.get_service_metrics()
    dispatch_metrics = Map.get(all_metrics, :dispatch, %{})
    cache_metrics = Map.get(all_metrics, :cache, %{})

    %{
      genservers: %{
        credential_vault: %{ready: vault_ready, name: "Credential Vault"},
        cache: %{ready: cache_ready, name: "Service Cache"}
      },
      services: services,
      dispatch_metrics: %{
        total_count: Map.get(dispatch_metrics, :count, 0),
        avg_ms: Map.get(dispatch_metrics, :avg_ms, 0.0),
        error_count: Map.get(dispatch_metrics, :error_count, 0)
      },
      cache_metrics: %{
        hits: Map.get(cache_metrics, :hits, 0),
        misses: Map.get(cache_metrics, :misses, 0),
        hit_rate: Map.get(cache_metrics, :hit_rate, 0.0)
      },
      ready: vault_ready and cache_ready,
      checked_at: DateTime.utc_now()
    }
  rescue
    _ ->
      %{
        genservers: %{
          credential_vault: %{ready: false, name: "Credential Vault"},
          cache: %{ready: false, name: "Service Cache"}
        },
        services: [],
        dispatch_metrics: %{total_count: 0, avg_ms: 0.0, error_count: 0},
        cache_metrics: %{hits: 0, misses: 0, hit_rate: 0.0},
        ready: false,
        checked_at: DateTime.utc_now()
      }
  end

  @doc "Returns CodeGazetteer status.\n"
  def get_code_gazetteer_status do
    if Code.ensure_loaded?(Brain.Code.CodeGazetteer) and Process.whereis(Brain.Code.CodeGazetteer) do
      try do
        stats = CodeGazetteer.stats()
        ready = CodeGazetteer.ready?()

        %{
          running: true,
          ready: ready,
          status:
            if(ready) do
              :ready
            else
              :initializing
            end,
          label:
            if(ready) do
              "Ready"
            else
              "Initializing"
            end,
          stats: %{
            total_symbols: Map.get(stats, :symbols, 0),
            total_relations: Map.get(stats, :relations, 0),
            total_files: Map.get(stats, :files, 0),
            total_languages: Map.get(stats, :languages, 0),
            language_list: Map.get(stats, :language_list, []),
            worlds_tracked: Map.get(stats, :worlds, 0)
          }
        }
      catch
        :exit, _ -> default_code_gazetteer_status()
      end
    else
      default_code_gazetteer_status()
    end
  end

  defp default_code_gazetteer_status do
    %{
      running: false,
      ready: false,
      status: :not_started,
      label: "Not started",
      stats: %{
        total_symbols: 0,
        total_relations: 0,
        total_files: 0,
        total_languages: 0,
        language_list: [],
        worlds_tracked: 0
      }
    }
  end

  @doc "Returns LanguageGrammar status.\n"
  def get_language_grammar_status do
    if Code.ensure_loaded?(Brain.Code.LanguageGrammar) and
         Process.whereis(Brain.Code.LanguageGrammar) do
      try do
        ready = LanguageGrammar.ready?()
        languages = LanguageGrammar.list_languages()

        available_count = Enum.count(languages, & &1.available)
        total_count = length(languages)

        %{
          running: true,
          ready: ready,
          status:
            if(ready) do
              :ready
            else
              :initializing
            end,
          label:
            if(ready) do
              "Ready (#{available_count}/#{total_count} grammars)"
            else
              "Initializing"
            end,
          stats: %{
            available_grammars: available_count,
            total_grammars: total_count,
            languages: Enum.map(languages, & &1.language)
          }
        }
      catch
        :exit, _ -> default_language_grammar_status()
      end
    else
      default_language_grammar_status()
    end
  end

  defp default_language_grammar_status do
    %{
      running: false,
      ready: false,
      status: :not_started,
      label: "Not started",
      stats: %{available_grammars: 0, total_grammars: 0, languages: []}
    }
  end

  @doc "Returns training worlds status.\n"
  def get_training_worlds_status do
    if Code.ensure_loaded?(World.Manager) and Process.whereis(World.Manager) do
      try do
        worlds = Manager.list_worlds()

        world_summaries =
          Enum.map(worlds, fn world ->
            metrics =
              case Manager.get_metrics(world.id) do
                {:ok, m} -> Metrics.summary(m)
                _ -> nil
              end

            candidates_count = length(Manager.get_candidates(world.id, limit: 1000))

            %{
              id: world.id,
              name: world.name,
              mode: world.mode,
              created_at: world.created_at,
              metrics: metrics,
              candidates_count: candidates_count
            }
          end)

        persisted_count =
          try do
            Persistence.list_persisted_worlds() |> length()
          catch
            _, _ -> 0
          end

        %{
          manager_ready: Manager.ready?(),
          active_worlds: length(worlds),
          persisted_worlds: persisted_count,
          worlds: world_summaries,
          checked_at: DateTime.utc_now()
        }
      catch
        :exit, _ ->
          default_training_worlds_status()
      end
    else
      default_training_worlds_status()
    end
  end

  @doc "Returns readiness details for a specific world.\n\nOptions:\n  - `:world_id` - The world ID to check (default: \"default\")\n"
  def get_readiness_details(opts \\ []) do
    world_id = Keyword.get(opts, :world_id, "default")

    status = get_all()
    models = get_ml_models_status()
    embedder_status = get_embedder_status()
    world_embedder_status = get_world_embedder_status(world_id)
    world_models_status = get_world_models_status(world_id)

    %{
      core: %{
        embedder:
          get_in(status, [:embedder, :ready]) || get_in(status, [:embedder, :phase]) == :idle,
        memory_store: get_in(status, [:memory_store, :ready]) || false,
        brain: get_in(status, [:brain, :ready]) || false
      },
      embedder_details: %{
        ready: embedder_status.ready,
        phase: embedder_status.phase,
        phase_label: embedder_status.phase_label,
        label: embedder_status.label,
        progress: embedder_status.progress,
        vocabulary_size: embedder_status.vocabulary_size,
        elapsed_ms: embedder_status.elapsed_ms
      },
      world_embedder: world_embedder_status,
      world_models: world_models_status,
      nlp_pipeline: %{
        ready: get_in(status, [:nlp_pipeline, :ready]) || false,
        components: get_in(status, [:nlp_pipeline, :components]) || %{}
      },
      ml_models: %{
        entity_extractor: get_in(models, [:entity_extractor, :loaded]) || false,
        pos_model_exists: get_in(models, [:pos_model, :exists]) || false,
        entity_model_exists: get_in(models, [:entity_model, :exists]) || false
      },
      template_store: safe_call_ready(Brain.Response.TemplateStore),
      all_ready: all_ready?()
    }
  end

  @doc "Returns the status of the world-specific embedder.\n"
  def get_world_embedder_status(world_id) do
    if Code.ensure_loaded?(World.Embedder) and function_exported?(World.Embedder, :get_status, 1) do
      try do
        status = Embedder.get_status(world_id)

        %{
          world_id: world_id,
          ready: status[:ready] || false,
          phase: status[:phase] || :not_initialized,
          phase_label: status[:phase_label] || "Not initialized",
          vocabulary_size: status[:vocabulary_size] || 0,
          episode_count: status[:episode_count] || 0,
          built_at: status[:built_at],
          label: build_world_embedder_label(status)
        }
      catch
        _, _ -> default_world_embedder_status(world_id)
      end
    else
      default_world_embedder_status(world_id)
    end
  end

  defp default_world_embedder_status(world_id) do
    %{
      world_id: world_id,
      ready: false,
      phase: :not_initialized,
      phase_label: "Not initialized",
      vocabulary_size: 0,
      episode_count: 0,
      built_at: nil,
      label: "Not initialized (will build on first use)"
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

  defp build_world_embedder_label(%{phase: :table_not_ready}) do
    "Table not ready"
  end

  defp build_world_embedder_label(%{phase: phase, phase_label: label}) when is_binary(label) do
    phase_str = phase |> to_string() |> String.replace("_", " ") |> String.capitalize()
    "#{phase_str}: #{label}"
  end

  defp build_world_embedder_label(_) do
    "Unknown status"
  end

  @doc "Returns the ML model status for a specific training world.\n"
  def get_world_models_status(world_id) when is_binary(world_id) do
    if Code.ensure_loaded?(World.ModelRegistry) and Process.whereis(World.ModelRegistry) do
      try do
        has_models = ModelRegistry.world_has_models?(world_id)

        models =
          if has_models do
            case ModelRegistry.get_world_models(world_id) do
              {:ok, m} -> m
              _ -> nil
            end
          else
            nil
          end

        embedder = models[:embedder]
        entity_model = models[:entity_model]
        pos_model = models[:pos_model]
        classifier = models[:classifier] || models[:intent_full]
        embedder_vocab_size = extract_vocab_size(embedder, :vocabulary)
        pos_model_tags = extract_vocab_size(pos_model, :tag_vocabulary)
        entity_model_tags = extract_vocab_size(entity_model, :tag_vocabulary)
        classifier_vocab_size = extract_vocab_size(classifier, :vocabulary)

        %{
          world_id: world_id,
          has_models: has_models,
          models: models,
          status:
            if(has_models) do
              :ready
            else
              :not_loaded
            end,
          is_loading: false,
          is_loaded: has_models,
          has_classifier: classifier != nil and is_map(classifier) and map_size(classifier) > 0,
          has_embedder: embedder != nil and is_map(embedder) and map_size(embedder) > 0,
          has_entity_model: entity_model != nil and map_size(entity_model) > 0,
          has_pos_model: pos_model != nil and map_size(pos_model) > 0,
          classifier_vocab_size: classifier_vocab_size,
          embedder_vocab_size: embedder_vocab_size,
          entity_model_size: entity_model_tags,
          pos_model_size: pos_model_tags,
          checked_at: DateTime.utc_now()
        }
      catch
        :exit, _ ->
          default_world_models_status(world_id, :error)
      end
    else
      default_world_models_status(world_id, :not_available)
    end
  end

  defp default_world_models_status(world_id, status) do
    %{
      world_id: world_id,
      has_models: false,
      models: nil,
      status: status,
      is_loading: false,
      is_loaded: false,
      has_classifier: false,
      has_embedder: false,
      has_entity_model: false,
      has_pos_model: false,
      classifier_vocab_size: 0,
      embedder_vocab_size: 0,
      entity_model_size: 0,
      pos_model_size: 0,
      checked_at: DateTime.utc_now()
    }
  end

  defp extract_vocab_size(nil, _key) do
    0
  end

  defp extract_vocab_size(model, key) when is_map(model) do
    case Map.get(model, key) do
      vocab when is_map(vocab) -> map_size(vocab)
      _ -> 0
    end
  end

  defp extract_vocab_size(_, _) do
    0
  end

  defp summarize_system(status) do
    %{
      name: status.name,
      module: status.module,
      category: status.category,
      memory_bytes: status.memory_bytes,
      rate_per_minute: status.rate_per_minute,
      call_count_total: status.call_count_total,
      last_activity_at: status.last_activity_at,
      utilization_status: status.utilization_status
    }
  end

  defp phase_to_status(:ready) do
    :ready
  end

  defp phase_to_status(:idle) do
    :idle
  end

  defp phase_to_status(:not_started) do
    :not_started
  end

  defp phase_to_status(:busy) do
    :building_vocabulary
  end

  defp phase_to_status(_) do
    :building_vocabulary
  end

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

  defp phase_label(:tokenizing) do
    "Tokenizing texts"
  end

  defp phase_label(:building_frequencies) do
    "Building frequencies"
  end

  defp phase_label(:calculating_idf) do
    "Calculating IDF weights"
  end

  defp phase_label(_) do
    "Initializing"
  end

  defp get_models_path do
    Application.get_env(:brain, :ml, [])[:models_path] ||
      Application.get_env(:chat_bot, :ml, [])[:models_path] ||
      Brain.priv_path("ml_models")
  end

  defp get_model_file_status(models_path, "gazetteer.term" = filename) do
    path = Path.join(models_path, filename)

    is_loaded =
      try do
        Gazetteer.loaded?()
      catch
        :exit, _ -> false
      end

    build_model_file_status(path, is_loaded)
  end

  defp get_model_file_status(models_path, "pos_model.term" = filename) do
    path = Path.join(models_path, filename)

    is_loaded =
      try do
        case POSTagger.load_model(path) do
          {:ok, _model} -> true
          _ -> false
        end
      catch
        :exit, _ -> false
      end

    build_model_file_status(path, is_loaded)
  end

  defp get_model_file_status(models_path, "entity_model.term" = filename) do
    path = Path.join(models_path, filename)

    is_loaded =
      try do
        case EntityTrainer.load_model() do
          {:ok, _model} -> true
          _ -> false
        end
      catch
        :exit, _ -> false
      end

    build_model_file_status(path, is_loaded)
  end

  defp get_model_file_status(models_path, filename) do
    path = Path.join(models_path, filename)
    build_model_file_status(path, false)
  end

  defp build_model_file_status(path, is_loaded) do
    case File.stat(path) do
      {:ok, stat} ->
        %{
          exists: true,
          loaded: is_loaded,
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

  defp get_agent_status(Brain.ML.EntityExtractor = module) do
    pid = Process.whereis(module)

    if pid do
      process_info = get_process_info(pid)

      is_loaded =
        try do
          EntityExtractor.is_loaded?()
        catch
          :exit, _ -> false
        end

      %{
        loaded: is_loaded,
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
      message_queue_len: nil,
      last_activity_at: nil,
      call_count_total: 0,
      call_count_window: 0,
      rate_per_minute: 0.0,
      utilization_status: :not_started
    }

    if pid do
      process_info = get_process_info(pid)

      status_with_info = %{
        base_status
        | running: true,
          status: :running,
          label: "Running",
          memory_bytes: process_info[:memory],
          message_queue_len: process_info[:message_queue_len]
      }

      status_with_info
      |> enhance_status(module, type)
      |> add_utilization_metrics(module)
    else
      base_status
    end
  end

  defp enhance_status(status, _module, :basic) do
    %{status | ready: true, status: :ready, label: "Ready"}
  end

  defp enhance_status(status, Brain.Memory.Embedder, :has_ready) do
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
        status:
          if(ready) do
            :ready
          else
            :initializing
          end,
        label:
          if(ready) do
            "Ready"
          else
            "Initializing..."
          end
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

    stats =
      if ready do
        safe_call_stats(module)
      else
        nil
      end

    %{
      status
      | ready: ready,
        status:
          if(ready) do
            :ready
          else
            :initializing
          end,
        label:
          if(ready) do
            "Ready"
          else
            "Initializing..."
          end,
        stats: stats
    }
  end

  defp enhance_status(status, module, :has_status) do
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
    if Code.ensure_loaded?(module) do
      try do
        module.ready?()
      catch
        :exit, _ -> false
      end
    else
      false
    end
  end

  defp safe_call_stats(module) do
    if Code.ensure_loaded?(module) do
      try do
        module.stats()
      catch
        :exit, _ -> nil
      end
    else
      nil
    end
  end

  defp safe_call_brain_status(module) do
    if Code.ensure_loaded?(module) do
      try do
        module.get_status()
      catch
        :exit, _ -> nil
      end
    else
      nil
    end
  end

  defp get_process_info(pid) do
    try do
      Process.info(pid, [:memory, :message_queue_len]) || []
    catch
      _, _ -> []
    end
  end

  defp add_utilization_metrics(status, module) do
    metric_name = Map.get(@module_metric_map, module)

    if metric_name do
      metrics = get_metric_for_module(metric_name)

      utilization_status =
        classify_utilization(
          status.memory_bytes,
          metrics.count,
          metrics.rate_per_minute,
          metrics.last_updated
        )

      %{
        status
        | last_activity_at: metrics.last_updated,
          call_count_total: metrics.count,
          call_count_window: estimate_window_count(metrics.rate_per_minute),
          rate_per_minute: metrics.rate_per_minute,
          utilization_status: utilization_status
      }
    else
      %{
        status
        | utilization_status:
            if(status.running) do
              :normal
            else
              :not_started
            end
      }
    end
  end

  defp get_metric_for_module(metric_name) do
    try do
      case Aggregator.get_metric(metric_name) do
        nil ->
          %{count: 0, rate_per_minute: 0.0, last_updated: nil}

        data ->
          %{
            count: Map.get(data, :count, 0),
            rate_per_minute: Map.get(data, :rate_per_minute, 0.0),
            last_updated: Map.get(data, :last_updated)
          }
      end
    catch
      _, _ -> %{count: 0, rate_per_minute: 0.0, last_updated: nil}
    end
  end

  defp classify_utilization(memory_bytes, count, rate_per_minute, last_updated) do
    now = System.monotonic_time(:millisecond)

    minutes_idle =
      if last_updated && last_updated > 0 do
        (now - last_updated) / 60_000
      else
        nil
      end

    cond do
      count == 0 -> :never_used
      minutes_idle && minutes_idle > 5 && rate_per_minute < 0.1 -> :idle
      memory_bytes && memory_bytes > 10_000_000 && rate_per_minute < 1.0 -> :high_cost
      rate_per_minute < 0.1 && count > 0 -> :low_usage
      true -> :normal
    end
  end

  defp estimate_window_count(rate_per_minute) do
    round(rate_per_minute * 5)
  end

  defp get_subprocess_supervisor_status do
    supervisor = Brain.Subprocesses.Supervisor
    pid = Process.whereis(supervisor)

    if pid do
      children =
        try do
          DynamicSupervisor.count_children(supervisor)
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
      children = Supervisor.count_children(Brain.Supervisor)

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
          {r +
             if(status.running) do
               1
             else
               0
             end, t + 1}
        end)

      {running + elem(category_stats, 0), total + elem(category_stats, 1)}
    end)
  end

  defp health_status_label(score) when score >= 90 do
    :healthy
  end

  defp health_status_label(score) when score >= 70 do
    :degraded
  end

  defp health_status_label(score) when score >= 50 do
    :warning
  end

  defp health_status_label(_score) do
    :critical
  end

  defp get_uptime_seconds do
    case :erlang.statistics(:wall_clock) do
      {uptime_ms, _} -> div(uptime_ms, 1000)
      _ -> 0
    end
  end

  @doc "Returns the status of micro-classifiers.\n"
  def get_micro_classifiers_status do
    alias Brain.ML.MicroClassifiers

    if Code.ensure_loaded?(MicroClassifiers) do
      MicroClassifiers.status()
    else
      %{ready: false, classifiers: %{}}
    end
  end

  @doc "Returns response timing metrics from the aggregator.\n"
  def get_response_timing do
    try do
      metrics = Aggregator.get_metrics()
      brain_eval = Map.get(metrics, :brain_evaluate, %{})
      pipeline = Map.get(metrics, :pipeline_process, %{})

      %{
        brain_avg_ms: Map.get(brain_eval, :avg_ms, 0),
        brain_count: Map.get(brain_eval, :count, 0),
        pipeline_avg_ms: Map.get(pipeline, :avg_ms, 0),
        pipeline_count: Map.get(pipeline, :count, 0)
      }
    catch
      _, _ -> %{brain_avg_ms: 0, brain_count: 0, pipeline_avg_ms: 0, pipeline_count: 0}
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

  defp default_training_worlds_status do
    %{
      manager_ready: false,
      active_worlds: 0,
      persisted_worlds: 0,
      worlds: [],
      checked_at: DateTime.utc_now()
    }
  end
end
