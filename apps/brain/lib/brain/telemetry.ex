defmodule Brain.Telemetry do
  @moduledoc """
  Telemetry event definitions and handlers for the ChatBot application.

  All telemetry is designed to be async and non-blocking:
  - Handlers use GenServer.cast (fire-and-forget) to send metrics
  - Heavy processing happens in the Metrics.Aggregator, not in handlers
  - ETS is used for fast concurrent reads

  ## Events

  - `[:chat_bot, :brain, :evaluate, :start | :stop | :exception]` - Brain evaluation
  - `[:chat_bot, :pipeline, :process, :start | :stop | :exception]` - Pipeline processing
  - `[:chat_bot, :memory, :query, :start | :stop]` - Memory queries
  - `[:chat_bot, :memory, :embed, :start | :stop]` - Embedding operations
  - `[:chat_bot, :gazetteer, :lookup, :start | :stop]` - Gazetteer lookups
  - `[:chat_bot, :ml, :train, :start | :stop | :exception]` - ML model training
  - `[:chat_bot, :ml, :load, :stop]` - ML model loading
  - `[:chat_bot, :genserver, :message_queue]` - Periodic queue size sampling
  - `[:chat_bot, :error]` - Error events

  ## Learning/Training World Events

  - `[:chat_bot, :learning, :entity_candidate_detected]` - New proper noun discovered
  - `[:chat_bot, :learning, :entity_promoted_to_gazetteer]` - Entity added to gazetteer
  - `[:chat_bot, :learning, :entity_ambiguity_detected]` - Entity with multiple types found
  - `[:chat_bot, :learning, :document_processed]` - Document ingestion complete
  - `[:chat_bot, :learning, :batch_complete]` - Batch ingestion complete
  - `[:chat_bot, :learning, :world_created]` - Training world created
  - `[:chat_bot, :learning, :world_destroyed]` - Training world destroyed

  ## Knowledge Expansion Events

  - `[:chat_bot, :knowledge, :research, :start | :stop | :exception]` - Research agent operations
  - `[:chat_bot, :knowledge, :corroborate, :start | :stop | :exception]` - Cross-source corroboration
  - `[:chat_bot, :knowledge, :review, :start | :stop]` - Review queue operations (approve/reject)

  ## Epistemic System Events

  - `[:chat_bot, :epistemic, :jtms_justify, :start | :stop | :exception]` - JTMS justification operations
  - `[:chat_bot, :epistemic, :belief_operation, :start | :stop | :exception]` - BeliefStore operations
  - `[:chat_bot, :epistemic, :fact_verification, :stop]` - Fact verification during analysis

  ## Evaluation Events

  - `[:chat_bot, :evaluation, :complete]` - Evaluation task completed (accuracy, F1, per-task results)

  ## Analysis Events

  - `[:chat_bot, :analysis, :racing, :start | :stop | :exception]` - Racing analyzer parallel processing
  - `[:chat_bot, :analysis, :racing, :early_exit]` - Racing analyzer early exit (fast path)
  - `[:chat_bot, :analysis, :event_extraction]` - Event extraction with GPU/tensor verification

  ## Code Analysis Events

  - `[:chat_bot, :code, :parse, :start | :stop | :exception]` - Code parsing operations
  - `[:chat_bot, :code, :extract, :start | :stop | :exception]` - Symbol extraction operations
  - `[:chat_bot, :code, :pipeline, :start | :stop | :exception]` - Full code analysis pipeline
  - `[:chat_bot, :code, :gazetteer, :lookup, :stop]` - Code gazetteer lookups
  - `[:chat_bot, :code, :gazetteer, :add, :stop]` - Code gazetteer additions
  - `[:chat_bot, :code, :file_processed]` - Code file processed event

  ## External Services Events

  - `[:chat_bot, :services, :dispatch, :start | :stop | :exception]` - Service dispatch operations
  - `[:chat_bot, :services, :enrichment, :start | :stop | :exception]` - Response enrichment
  - `[:chat_bot, :services, :cache, :hit]` - Cache hit events
  - `[:chat_bot, :services, :cache, :miss]` - Cache miss events
  - `[:chat_bot, :services, :health_check, :stop]` - Service health check results
  - `[:chat_bot, :services, :credential]` - Credential operations (store/delete)
  """

  require Logger

  # ============================================================================
  # Event Names
  # ============================================================================

  @brain_evaluate [:chat_bot, :brain, :evaluate]
  @pipeline_process [:chat_bot, :pipeline, :process]
  @memory_query [:chat_bot, :memory, :query]
  @memory_embed [:chat_bot, :memory, :embed]
  @gazetteer_lookup [:chat_bot, :gazetteer, :lookup]

  # Response and Fact Database
  @response_generate [:chat_bot, :response, :generate]
  @response_template_lookup [:chat_bot, :response, :template_lookup]
  @fact_database_query [:chat_bot, :fact_database, :query]
  @entity_extract [:chat_bot, :entity, :extract]
  @ml_train [:chat_bot, :ml, :train]
  @model_load [:chat_bot, :ml, :load]
  @message_queue [:chat_bot, :genserver, :message_queue]
  @error_event [:chat_bot, :error]

  # Learning/Training World Events
  @learning_entity_discovered [:chat_bot, :learning, :entity_candidate_detected]
  @learning_entity_promoted [:chat_bot, :learning, :entity_promoted_to_gazetteer]
  @learning_ambiguity [:chat_bot, :learning, :entity_ambiguity_detected]
  @learning_document_processed [:chat_bot, :learning, :document_processed]
  @learning_batch_complete [:chat_bot, :learning, :batch_complete]
  @learning_world_created [:chat_bot, :learning, :world_created]
  @learning_world_destroyed [:chat_bot, :learning, :world_destroyed]

  # Knowledge Expansion Events
  @knowledge_research [:chat_bot, :knowledge, :research]
  @knowledge_corroborate [:chat_bot, :knowledge, :corroborate]
  @knowledge_review [:chat_bot, :knowledge, :review]

  # Epistemic System Events
  @jtms_justify [:chat_bot, :epistemic, :jtms_justify]
  @belief_operation [:chat_bot, :epistemic, :belief_operation]
  @epistemic_fact_verification [:chat_bot, :epistemic, :fact_verification]

  # Analysis Events
  @racing_analysis [:chat_bot, :analysis, :racing]
  @racing_early_exit [:chat_bot, :analysis, :racing, :early_exit]
  @event_extraction [:chat_bot, :analysis, :event_extraction]

  # Code Analysis Events
  @code_parse [:chat_bot, :code, :parse]
  @code_extract [:chat_bot, :code, :extract]
  @code_pipeline [:chat_bot, :code, :pipeline]
  @code_gazetteer_lookup [:chat_bot, :code, :gazetteer, :lookup]
  @code_gazetteer_add [:chat_bot, :code, :gazetteer, :add]
  @code_file_processed [:chat_bot, :code, :file_processed]

  # Evaluation Events
  @evaluation_complete [:chat_bot, :evaluation, :complete]

  # Consistency Events
  @consistency_disagreement [:chat_bot, :analysis, :consistency, :disagreement]

  # External Services Events
  @service_dispatch [:chat_bot, :services, :dispatch]
  @service_enrichment [:chat_bot, :services, :enrichment]
  @service_cache_hit [:chat_bot, :services, :cache, :hit]
  @service_cache_miss [:chat_bot, :services, :cache, :miss]
  @service_health_check [:chat_bot, :services, :health_check]
  @service_credential_operation [:chat_bot, :services, :credential]

  # ============================================================================
  # Public API - Attach Handlers
  # ============================================================================

  @doc """
  Attaches all telemetry handlers. Call this during application startup.
  """
  def attach_handlers do
    handlers = [
      # Brain evaluate handlers
      {"chatbot-brain-evaluate-stop", @brain_evaluate ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :brain_evaluate}},
      {"chatbot-brain-evaluate-exception", @brain_evaluate ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :brain_evaluate}},

      # Pipeline process handlers
      {"chatbot-pipeline-process-stop", @pipeline_process ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :pipeline_process}},
      {"chatbot-pipeline-process-exception", @pipeline_process ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :pipeline_process}},

      # Memory query handlers
      {"chatbot-memory-query-stop", @memory_query ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :memory_query}},

      # Response generation
      {"chatbot-response-generate-stop", @response_generate ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :response_generate}},
      {"chatbot-response-template-lookup-stop", @response_template_lookup ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :response_template_lookup}},

      # Fact database
      {"chatbot-fact-database-query-stop", @fact_database_query ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :fact_database_query}},

      # Memory embed handlers
      {"chatbot-memory-embed-stop", @memory_embed ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :memory_embed}},

      # Gazetteer lookup handlers
      {"chatbot-gazetteer-lookup-stop", @gazetteer_lookup ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :gazetteer_lookup}},

      # Entity extraction handlers
      {"chatbot-entity-extract-stop", @entity_extract ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :entity_extract}},

      # ML Training handlers
      {"chatbot-ml-train-start", @ml_train ++ [:start], &__MODULE__.handle_training_start/4, %{}},
      {"chatbot-ml-train-stop", @ml_train ++ [:stop], &__MODULE__.handle_training_stop/4, %{}},
      {"chatbot-ml-train-exception", @ml_train ++ [:exception],
       &__MODULE__.handle_training_exception/4, %{}},

      # Model load handlers
      {"chatbot-model-load-stop", @model_load ++ [:stop], &__MODULE__.handle_model_load/4, %{}},

      # Message queue sampling
      {"chatbot-message-queue", @message_queue, &__MODULE__.handle_message_queue/4, %{}},

      # Error events
      {"chatbot-error", @error_event, &__MODULE__.handle_error/4, %{}},

      # Learning/Training World events
      {"chatbot-learning-entity-discovered", @learning_entity_discovered,
       &__MODULE__.handle_learning_event/4, %{event: :entity_discovered}},
      {"chatbot-learning-entity-promoted", @learning_entity_promoted,
       &__MODULE__.handle_learning_event/4, %{event: :entity_promoted}},
      {"chatbot-learning-ambiguity", @learning_ambiguity, &__MODULE__.handle_learning_event/4,
       %{event: :ambiguity_detected}},
      {"chatbot-learning-document", @learning_document_processed,
       &__MODULE__.handle_learning_event/4, %{event: :document_processed}},
      {"chatbot-learning-batch", @learning_batch_complete, &__MODULE__.handle_learning_event/4,
       %{event: :batch_complete}},
      {"chatbot-learning-world-created", @learning_world_created,
       &__MODULE__.handle_learning_event/4, %{event: :world_created}},
      {"chatbot-learning-world-destroyed", @learning_world_destroyed,
       &__MODULE__.handle_learning_event/4, %{event: :world_destroyed}},

      # Knowledge Expansion handlers
      {"chatbot-knowledge-research-stop", @knowledge_research ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :knowledge_research}},
      {"chatbot-knowledge-research-exception", @knowledge_research ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :knowledge_research}},
      {"chatbot-knowledge-corroborate-stop", @knowledge_corroborate ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :knowledge_corroborate}},
      {"chatbot-knowledge-corroborate-exception", @knowledge_corroborate ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :knowledge_corroborate}},
      {"chatbot-knowledge-review-stop", @knowledge_review ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :knowledge_review}},

      # Epistemic System handlers
      {"chatbot-jtms-justify-stop", @jtms_justify ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :jtms_justify}},
      {"chatbot-jtms-justify-exception", @jtms_justify ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :jtms_justify}},
      {"chatbot-belief-operation-stop", @belief_operation ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :belief_operation}},
      {"chatbot-belief-operation-exception", @belief_operation ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :belief_operation}},
      {"chatbot-fact-verification-stop", @epistemic_fact_verification ++ [:stop],
       &__MODULE__.handle_fact_verification/4, %{}},

      # Racing Analyzer handlers
      {"chatbot-racing-analysis-stop", @racing_analysis ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :racing_analysis}},
      {"chatbot-racing-analysis-exception", @racing_analysis ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :racing_analysis}},
      {"chatbot-racing-early-exit", @racing_early_exit, &__MODULE__.handle_racing_early_exit/4, %{}},

      # Code Analysis handlers
      {"chatbot-code-parse-stop", @code_parse ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :code_parse}},
      {"chatbot-code-parse-exception", @code_parse ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :code_parse}},
      {"chatbot-code-extract-stop", @code_extract ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :code_extract}},
      {"chatbot-code-extract-exception", @code_extract ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :code_extract}},
      {"chatbot-code-pipeline-stop", @code_pipeline ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :code_pipeline}},
      {"chatbot-code-pipeline-exception", @code_pipeline ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :code_pipeline}},
      {"chatbot-code-gazetteer-lookup-stop", @code_gazetteer_lookup ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :code_gazetteer_lookup}},
      {"chatbot-code-gazetteer-add-stop", @code_gazetteer_add ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :code_gazetteer_add}},
      {"chatbot-code-file-processed", @code_file_processed,
       &__MODULE__.handle_code_file_processed/4, %{}},

      # Evaluation handlers
      {"chatbot-evaluation-complete", @evaluation_complete,
       &__MODULE__.handle_evaluation_complete/4, %{}},

      # Consistency handlers
      {"chatbot-consistency-disagreement", @consistency_disagreement,
       &__MODULE__.handle_consistency_disagreement/4, %{}},

      # External Services handlers
      {"chatbot-service-dispatch-stop", @service_dispatch ++ [:stop],
       &__MODULE__.handle_service_dispatch/4, %{}},
      {"chatbot-service-dispatch-exception", @service_dispatch ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :service_dispatch}},
      {"chatbot-service-enrichment-stop", @service_enrichment ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :service_enrichment}},
      {"chatbot-service-enrichment-exception", @service_enrichment ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :service_enrichment}},
      {"chatbot-service-cache-hit", @service_cache_hit,
       &__MODULE__.handle_service_cache/4, %{type: :hit}},
      {"chatbot-service-cache-miss", @service_cache_miss,
       &__MODULE__.handle_service_cache/4, %{type: :miss}},
      {"chatbot-service-health-check-stop", @service_health_check ++ [:stop],
       &__MODULE__.handle_service_health_check/4, %{}},
      {"chatbot-service-credential", @service_credential_operation,
       &__MODULE__.handle_service_credential/4, %{}}
    ]

    Enum.each(handlers, fn {id, event, handler, config} ->
      :telemetry.attach(id, event, handler, config)
    end)

    :ok
  end

  @doc """
  Detaches all telemetry handlers. Useful for testing.
  """
  def detach_handlers do
    handler_ids = [
      "chatbot-brain-evaluate-stop",
      "chatbot-brain-evaluate-exception",
      "chatbot-pipeline-process-stop",
      "chatbot-pipeline-process-exception",
      "chatbot-memory-query-stop",
      "chatbot-memory-embed-stop",
      "chatbot-response-generate-stop",
      "chatbot-response-template-lookup-stop",
      "chatbot-fact-database-query-stop",
      "chatbot-gazetteer-lookup-stop",
      "chatbot-entity-extract-stop",
      "chatbot-ml-train-start",
      "chatbot-ml-train-stop",
      "chatbot-ml-train-exception",
      "chatbot-model-load-stop",
      "chatbot-message-queue",
      "chatbot-error",
      # Learning events
      "chatbot-learning-entity-discovered",
      "chatbot-learning-entity-promoted",
      "chatbot-learning-ambiguity",
      "chatbot-learning-document",
      "chatbot-learning-batch",
      "chatbot-learning-world-created",
      "chatbot-learning-world-destroyed",
      # Knowledge Expansion events
      "chatbot-knowledge-research-stop",
      "chatbot-knowledge-research-exception",
      "chatbot-knowledge-corroborate-stop",
      "chatbot-knowledge-corroborate-exception",
      "chatbot-knowledge-review-stop",
      # Epistemic events
      "chatbot-fact-verification-stop",
      "chatbot-jtms-justify-stop",
      "chatbot-jtms-justify-exception",
      "chatbot-belief-operation-stop",
      "chatbot-belief-operation-exception",
      # Racing analyzer events
      "chatbot-racing-analysis-stop",
      "chatbot-racing-analysis-exception",
      "chatbot-racing-early-exit",
      # Code analysis events
      "chatbot-code-parse-stop",
      "chatbot-code-parse-exception",
      "chatbot-code-extract-stop",
      "chatbot-code-extract-exception",
      "chatbot-code-pipeline-stop",
      "chatbot-code-pipeline-exception",
      "chatbot-code-gazetteer-lookup-stop",
      "chatbot-code-gazetteer-add-stop",
      "chatbot-code-file-processed",
      # Evaluation events
      "chatbot-evaluation-complete",
      # Consistency events
      "chatbot-consistency-disagreement",
      # External services events
      "chatbot-service-dispatch-stop",
      "chatbot-service-dispatch-exception",
      "chatbot-service-enrichment-stop",
      "chatbot-service-enrichment-exception",
      "chatbot-service-cache-hit",
      "chatbot-service-cache-miss",
      "chatbot-service-health-check-stop",
      "chatbot-service-credential"
    ]

    Enum.each(handler_ids, fn id ->
      :telemetry.detach(id)
    end)

    :ok
  end

  # ============================================================================
  # Convenience Functions for Emitting Events
  # ============================================================================

  @doc """
  Wraps a function with telemetry span measurement.
  Returns the result of the function.

  ## Example

      Brain.Telemetry.span(:brain_evaluate, %{conversation_id: id}, fn ->
        do_evaluate(input)
      end)
  """
  def span(:brain_evaluate, metadata, fun) do
    :telemetry.span(@brain_evaluate, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:pipeline_process, metadata, fun) do
    :telemetry.span(@pipeline_process, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:memory_query, metadata, fun) do
    :telemetry.span(@memory_query, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:memory_embed, metadata, fun) do
    :telemetry.span(@memory_embed, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:gazetteer_lookup, metadata, fun) do
    :telemetry.span(@gazetteer_lookup, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:response_generate, metadata, fun) do
    :telemetry.span(@response_generate, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:response_template_lookup, metadata, fun) do
    :telemetry.span(@response_template_lookup, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:fact_database_query, metadata, fun) do
    :telemetry.span(@fact_database_query, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:entity_extract, metadata, fun) do
    :telemetry.span(@entity_extract, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  # Knowledge Expansion spans

  def span(:knowledge_research, metadata, fun) do
    :telemetry.span(@knowledge_research, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:knowledge_corroborate, metadata, fun) do
    :telemetry.span(@knowledge_corroborate, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:knowledge_review, metadata, fun) do
    :telemetry.span(@knowledge_review, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  # Epistemic System spans

  def span(:jtms_justify, metadata, fun) do
    :telemetry.span(@jtms_justify, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:belief_operation, metadata, fun) do
    :telemetry.span(@belief_operation, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  # Analysis spans

  def span(:racing_analysis, metadata, fun) do
    :telemetry.span(@racing_analysis, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  # Code Analysis spans

  def span(:code_parse, metadata, fun) do
    :telemetry.span(@code_parse, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:code_extract, metadata, fun) do
    :telemetry.span(@code_extract, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:code_pipeline, metadata, fun) do
    :telemetry.span(@code_pipeline, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:code_gazetteer_lookup, metadata, fun) do
    :telemetry.span(@code_gazetteer_lookup, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:code_gazetteer_add, metadata, fun) do
    :telemetry.span(@code_gazetteer_add, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  # External Services spans

  def span(:service_dispatch, metadata, fun) do
    :telemetry.span(@service_dispatch, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:service_enrichment, metadata, fun) do
    :telemetry.span(@service_enrichment, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  def span(:service_health_check, metadata, fun) do
    :telemetry.span(@service_health_check, metadata, fn ->
      result = fun.()
      {result, %{}}
    end)
  end

  @doc """
  Emits a service cache hit event.
  """
  def emit_service_cache_hit(service, key) do
    :telemetry.execute(
      @service_cache_hit,
      %{count: 1},
      %{service: service, key: key, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits a service cache miss event.
  """
  def emit_service_cache_miss(service, key) do
    :telemetry.execute(
      @service_cache_miss,
      %{count: 1},
      %{service: service, key: key, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits a service dispatch event with detailed metrics.
  """
  def emit_service_dispatch(service, intent, status, duration_ms) do
    :telemetry.execute(
      @service_dispatch ++ [:stop],
      %{duration: duration_ms},
      %{
        service: service,
        intent: intent,
        status: status,
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  @doc """
  Emits a credential operation event.
  """
  def emit_credential_operation(operation, service, world) do
    :telemetry.execute(
      @service_credential_operation,
      %{count: 1},
      %{
        operation: operation,
        service: service,
        world: world,
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  @doc """
  Emits a fact verification event from the epistemic system.

  ## Parameters
  - `status` - Verification status (:verified, :contradicted, :uncertain, :unchecked)
  - `subject` - The subject being verified
  - `duration_ms` - Verification duration in milliseconds
  - `metadata` - Additional metadata (beliefs_count, verification_result)
  """
  def emit_fact_verification(status, subject, duration_ms, metadata \\ %{}) do
    :telemetry.execute(
      @epistemic_fact_verification ++ [:stop],
      %{duration: duration_ms},
      Map.merge(metadata, %{
        status: status,
        subject: subject,
        timestamp: System.monotonic_time(:millisecond)
      })
    )
  end

  @doc """
  Emits a code file processed event.
  """
  def emit_code_file_processed(file_path, language, symbols_count, relations_count, duration_ms) do
    :telemetry.execute(
      @code_file_processed,
      %{symbols_count: symbols_count, relations_count: relations_count, duration_ms: duration_ms},
      %{file_path: file_path, language: language, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits an evaluation completion event with accuracy metrics.

  ## Parameters
  - `task` - The evaluation task (e.g., "intent", "ner", "sentiment", "speech_act")
  - `metrics` - Map with :accuracy, :macro_f1, :weighted_f1, :total_examples, :duration_ms
  """
  def emit_evaluation_complete(task, metrics) when is_map(metrics) do
    :telemetry.execute(
      @evaluation_complete,
      %{
        accuracy: Map.get(metrics, :accuracy, 0.0),
        macro_f1: Map.get(metrics, :macro_f1, 0.0),
        weighted_f1: Map.get(metrics, :weighted_f1, 0.0),
        total_examples: Map.get(metrics, :total_examples, 0),
        duration_ms: Map.get(metrics, :duration_ms, 0)
      },
      %{
        task: task,
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  @doc """
  Emits a racing analyzer early exit event when a fast path is taken.
  """
  def emit_racing_early_exit(analyzer, confidence, duration_ms) do
    :telemetry.execute(
      @racing_early_exit,
      %{confidence: confidence, duration_ms: duration_ms},
      %{analyzer: analyzer, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits an event extraction telemetry event.

  Used to verify tensor operations and GPU backend usage.

  ## Measurements
  - `:duration` - Extraction duration in native time units
  - `:event_count` - Number of events extracted
  - `:token_count` - Number of tokens processed

  ## Metadata
  - `:backend` - Nx backend in use (e.g., "EXLA.Backend")
  - `:tensor_ops` - Whether tensor operations were used
  - `:string_ops` - Whether string operations were used (should be false)
  """
  def emit_event_extraction(measurements) when is_map(measurements) do
    :telemetry.execute(
      @event_extraction,
      %{
        duration: Map.get(measurements, :duration, 0),
        event_count: Map.get(measurements, :event_count, 0),
        token_count: Map.get(measurements, :token_count, 0)
      },
      %{
        backend: Map.get(measurements, :backend, "unknown"),
        tensor_ops: Map.get(measurements, :tensor_ops, false),
        string_ops: Map.get(measurements, :string_ops, false),
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  @doc """
  Emits a message queue size event. Used for periodic sampling.
  """
  def emit_message_queue(genserver_name, queue_length) do
    :telemetry.execute(
      @message_queue,
      %{queue_length: queue_length},
      %{genserver: genserver_name, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits an error event. Non-blocking.
  """
  def emit_error(error_type, details \\ %{}) do
    :telemetry.execute(
      @error_event,
      %{count: 1},
      %{error_type: error_type, details: details, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  # ============================================================================
  # Handler Functions (Must be fast - use cast only)
  # These are public so telemetry can call them efficiently as module functions
  # ============================================================================

  @doc false
  def handle_span_stop(_event, measurements, metadata, config) do
    duration_ms = native_to_ms(measurements[:duration])
    metric = config[:metric]

    # Fire-and-forget cast to aggregator
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_duration, metric, duration_ms, metadata}
      )
    end
  end

  @doc false
  def handle_span_exception(_event, measurements, metadata, config) do
    duration_ms = native_to_ms(measurements[:duration])
    metric = config[:metric]

    # Fire-and-forget cast to aggregator
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_error, metric, duration_ms, metadata}
      )
    end
  end

  @doc false
  def handle_message_queue(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_queue_size, metadata[:genserver], measurements[:queue_length]}
      )
    end
  end

  @doc false
  def handle_error(_event, _measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_error_event, metadata[:error_type], metadata[:details]}
      )
    end
  end

  @doc false
  def handle_training_start(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_training_start, metadata[:model], measurements[:sequence_count], metadata}
      )
    end
  end

  @doc false
  def handle_training_stop(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_training_stop, metadata[:model], measurements, metadata}
      )
    end
  end

  @doc false
  def handle_training_exception(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_training_exception, metadata[:model], measurements, metadata}
      )
    end
  end

  @doc false
  def handle_model_load(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_model_load, metadata[:model], measurements[:duration_ms], metadata}
      )
    end
  end

  @doc false
  def handle_learning_event(_event, measurements, metadata, config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_learning_event, config[:event], measurements, metadata}
      )
    end
  end

  @doc false
  def handle_racing_early_exit(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_racing_early_exit, metadata[:analyzer], measurements[:confidence],
         measurements[:duration_ms]}
      )
    end
  end

  @doc false
  def handle_code_file_processed(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_code_file_processed, metadata[:file_path], metadata[:language],
         measurements[:symbols_count], measurements[:relations_count], measurements[:duration_ms]}
      )
    end
  end

  # ============================================================================
  # Evaluation Handlers
  # ============================================================================

  @doc false
  def handle_evaluation_complete(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_evaluation_complete, metadata[:task], measurements}
      )
    end
  end

  # ============================================================================
  # External Services Handlers
  # ============================================================================

  @doc false
  def handle_service_dispatch(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      # Duration is already in milliseconds from emit_service_dispatch
      duration_ms = measurements[:duration] || 0

      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_service_dispatch, metadata[:service], metadata[:intent], metadata[:status],
         duration_ms}
      )
    end
  end

  @doc false
  def handle_service_cache(_event, measurements, metadata, config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_service_cache, config[:type], metadata[:service], measurements[:count]}
      )
    end
  end

  @doc false
  def handle_service_health_check(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      # Duration is already in milliseconds from the emit function
      duration_ms = measurements[:duration] || 0

      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_service_health_check, metadata[:service], metadata[:status], duration_ms}
      )
    end
  end

  @doc false
  def handle_service_credential(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_service_credential, metadata[:operation], metadata[:service],
         measurements[:count]}
      )
    end
  end

  @doc false
  def handle_consistency_disagreement(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_consistency_disagreement, Map.merge(measurements, metadata)}
      )
    end
  end

  # ============================================================================
  # Epistemic System Handlers
  # ============================================================================

  @doc false
  def handle_fact_verification(_event, measurements, metadata, _config) do
    if Process.whereis(Brain.Metrics.Aggregator) do
      duration_ms = measurements[:duration] || 0

      GenServer.cast(
        Brain.Metrics.Aggregator,
        {:record_fact_verification, metadata[:status], metadata[:subject],
         metadata[:beliefs_count] || 0, duration_ms}
      )
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp native_to_ms(duration) when is_integer(duration) do
    System.convert_time_unit(duration, :native, :millisecond)
  end

  defp native_to_ms(_), do: 0
end
