defmodule Brain.Telemetry do
  @moduledoc """
  Telemetry event definitions and handlers for the ChatBot application.

  Every handler `attach_handlers/0` attaches does one thing: it hands the
  event to `Brain.Metrics.Aggregator.submit/2`, which sends it to the
  aggregator without waiting. The aggregator reads the event and counts it,
  in Atlas where the count must survive a restart.

  A handler never raises, so `:telemetry` never detaches it (a handler that
  raises is detached for good). When the aggregator is not running, as
  while its supervisor restarts it, the event cannot be counted: the
  handler logs it as an error and counts it as missed
  (`Brain.Metrics.Aggregator.get_missed_events/0`), and counts the next
  event as usual once the aggregator is back.

  ## Events

  - `[:chat_bot, :brain, :evaluate, :start | :stop | :exception]` - Brain evaluation
  - `[:chat_bot, :pipeline, :process, :start | :stop | :exception]` - Pipeline processing
  - `[:chat_bot, :memory, :query, :start | :stop]` - Memory queries
  - `[:chat_bot, :memory, :semantic_fact, :retrieved]` - Semantic facts retrieved for a turn, and the recurring pattern found among them (`Brain.Memory.RecurringPattern`)
  - `[:chat_bot, :memory, :semantic_fact, :used]` - A turn's response built from its recurring pattern, by the synthesizer or the realization packet; once per turn
    (both counted per world by `Brain.Metrics.Aggregator.get_semantic_fact_metrics/1`)
  - `[:chat_bot, :memory, :embed, :start | :stop]` - Embedding operations
  - `[:chat_bot, :gazetteer, :lookup, :start | :stop]` - Gazetteer lookups
  - `[:chat_bot, :ml, :train, :start | :stop | :exception]` - ML model training
  - `[:chat_bot, :ml, :load, :stop]` - A model read from disk, as it was loaded (`emit_model_load/3`)

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

  - `[:chat_bot, :evaluation, :complete]` - Evaluation task completed (accuracy, F1, per-task results).
    No handler counts it: `Brain.ML.EvaluationStore` keeps each result in Atlas.

  ## Analysis Events

  - `[:chat_bot, :brain, :lattice, :resolved]` - Multi-hypothesis lattice built (intent / scientific ranking)

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
  - `[:chat_bot, :services, :health_check, :stop]` - Service health check results (`emit_service_health_check/3`)
  - `[:chat_bot, :services, :credential]` - Credential operations (store/delete)
  """

  require Logger

  # ============================================================================
  # Event Names
  # ============================================================================

  @brain_evaluate [:chat_bot, :brain, :evaluate]
  @pipeline_process [:chat_bot, :pipeline, :process]
  @memory_query [:chat_bot, :memory, :query]
  @semantic_fact_retrieved [:chat_bot, :memory, :semantic_fact, :retrieved]
  @semantic_fact_used [:chat_bot, :memory, :semantic_fact, :used]
  @memory_embed [:chat_bot, :memory, :embed]
  @gazetteer_lookup [:chat_bot, :gazetteer, :lookup]

  # Response and Fact Database
  @response_generate [:chat_bot, :response, :generate]
  @response_template_lookup [:chat_bot, :response, :template_lookup]
  @fact_database_query [:chat_bot, :fact_database, :query]
  @entity_extract [:chat_bot, :entity, :extract]
  @ml_train [:chat_bot, :ml, :train]
  @model_load [:chat_bot, :ml, :load]

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

  # Every handler `attach_handlers/0` attaches: its id, the event it is
  # attached to, and the kind of event `Brain.Metrics.Aggregator` reads it as.
  # `detach_handlers/0` detaches the same list, so no handler is left behind.
  @handlers [
    # Spans: how long each took, and the ones that raised
    {"chatbot-brain-evaluate-stop", @brain_evaluate ++ [:stop], {:span_stop, :brain_evaluate}},
    {"chatbot-brain-evaluate-exception", @brain_evaluate ++ [:exception], {:span_exception, :brain_evaluate}},
    {"chatbot-pipeline-process-stop", @pipeline_process ++ [:stop], {:span_stop, :pipeline_process}},
    {"chatbot-pipeline-process-exception", @pipeline_process ++ [:exception], {:span_exception, :pipeline_process}},
    {"chatbot-memory-query-stop", @memory_query ++ [:stop], {:span_stop, :memory_query}},
    {"chatbot-memory-embed-stop", @memory_embed ++ [:stop], {:span_stop, :memory_embed}},
    {"chatbot-response-generate-stop", @response_generate ++ [:stop], {:span_stop, :response_generate}},
    {"chatbot-response-template-lookup-stop", @response_template_lookup ++ [:stop],
     {:span_stop, :response_template_lookup}},
    {"chatbot-fact-database-query-stop", @fact_database_query ++ [:stop], {:span_stop, :fact_database_query}},
    {"chatbot-gazetteer-lookup-stop", @gazetteer_lookup ++ [:stop], {:span_stop, :gazetteer_lookup}},
    {"chatbot-entity-extract-stop", @entity_extract ++ [:stop], {:span_stop, :entity_extract}},
    {"chatbot-knowledge-research-stop", @knowledge_research ++ [:stop], {:span_stop, :knowledge_research}},
    {"chatbot-knowledge-research-exception", @knowledge_research ++ [:exception],
     {:span_exception, :knowledge_research}},
    {"chatbot-knowledge-corroborate-stop", @knowledge_corroborate ++ [:stop], {:span_stop, :knowledge_corroborate}},
    {"chatbot-knowledge-corroborate-exception", @knowledge_corroborate ++ [:exception],
     {:span_exception, :knowledge_corroborate}},
    {"chatbot-knowledge-review-stop", @knowledge_review ++ [:stop], {:span_stop, :knowledge_review}},
    {"chatbot-jtms-justify-stop", @jtms_justify ++ [:stop], {:span_stop, :jtms_justify}},
    {"chatbot-jtms-justify-exception", @jtms_justify ++ [:exception], {:span_exception, :jtms_justify}},
    {"chatbot-belief-operation-stop", @belief_operation ++ [:stop], {:span_stop, :belief_operation}},
    {"chatbot-belief-operation-exception", @belief_operation ++ [:exception], {:span_exception, :belief_operation}},
    {"chatbot-racing-analysis-stop", @racing_analysis ++ [:stop], {:span_stop, :racing_analysis}},
    {"chatbot-racing-analysis-exception", @racing_analysis ++ [:exception], {:span_exception, :racing_analysis}},
    {"chatbot-code-parse-stop", @code_parse ++ [:stop], {:span_stop, :code_parse}},
    {"chatbot-code-parse-exception", @code_parse ++ [:exception], {:span_exception, :code_parse}},
    {"chatbot-code-extract-stop", @code_extract ++ [:stop], {:span_stop, :code_extract}},
    {"chatbot-code-extract-exception", @code_extract ++ [:exception], {:span_exception, :code_extract}},
    {"chatbot-code-pipeline-stop", @code_pipeline ++ [:stop], {:span_stop, :code_pipeline}},
    {"chatbot-code-pipeline-exception", @code_pipeline ++ [:exception], {:span_exception, :code_pipeline}},
    {"chatbot-code-gazetteer-lookup-stop", @code_gazetteer_lookup ++ [:stop], {:span_stop, :code_gazetteer_lookup}},
    {"chatbot-code-gazetteer-add-stop", @code_gazetteer_add ++ [:stop], {:span_stop, :code_gazetteer_add}},
    {"chatbot-service-dispatch-exception", @service_dispatch ++ [:exception], {:span_exception, :service_dispatch}},
    {"chatbot-service-enrichment-stop", @service_enrichment ++ [:stop], {:span_stop, :service_enrichment}},
    {"chatbot-service-enrichment-exception", @service_enrichment ++ [:exception],
     {:span_exception, :service_enrichment}},

    # Semantic facts: retrieved for a turn, and the recurring pattern used
    {"chatbot-semantic-fact-retrieved", @semantic_fact_retrieved, :semantic_facts_retrieved},
    {"chatbot-semantic-fact-used", @semantic_fact_used, :semantic_fact_used},

    # ML training and model loading
    {"chatbot-ml-train-start", @ml_train ++ [:start], :training_start},
    {"chatbot-ml-train-stop", @ml_train ++ [:stop], :training_stop},
    {"chatbot-ml-train-exception", @ml_train ++ [:exception], :training_exception},
    {"chatbot-model-load-stop", @model_load ++ [:stop], :model_load},

    # Learning/Training World events
    {"chatbot-learning-entity-discovered", @learning_entity_discovered, {:learning, :entity_discovered}},
    {"chatbot-learning-entity-promoted", @learning_entity_promoted, {:learning, :entity_promoted}},
    {"chatbot-learning-ambiguity", @learning_ambiguity, {:learning, :ambiguity_detected}},
    {"chatbot-learning-document", @learning_document_processed, {:learning, :document_processed}},
    {"chatbot-learning-batch", @learning_batch_complete, {:learning, :batch_complete}},
    {"chatbot-learning-world-created", @learning_world_created, {:learning, :world_created}},
    {"chatbot-learning-world-destroyed", @learning_world_destroyed, {:learning, :world_destroyed}},

    # Epistemic, racing analyzer, code analysis and consistency events
    {"chatbot-fact-verification-stop", @epistemic_fact_verification ++ [:stop], :fact_verification},
    {"chatbot-racing-early-exit", @racing_early_exit, :racing_early_exit},
    {"chatbot-code-file-processed", @code_file_processed, :code_file_processed},
    {"chatbot-consistency-disagreement", @consistency_disagreement, :consistency_disagreement},

    # External services
    {"chatbot-service-dispatch-stop", @service_dispatch ++ [:stop], :service_dispatch},
    {"chatbot-service-cache-hit", @service_cache_hit, {:service_cache, :hit}},
    {"chatbot-service-cache-miss", @service_cache_miss, {:service_cache, :miss}},
    {"chatbot-service-health-check-stop", @service_health_check ++ [:stop], :service_health_check},
    {"chatbot-service-credential", @service_credential_operation, :service_credential},

    # KG signals and novelty
    {"chatbot-kg-signal-scored", [:brain, :kg_signal, :scored], {:kg_signal, :scored}},
    {"chatbot-kg-signal-cache-hit", [:brain, :kg_signal, :cache_hit], {:kg_signal, :cache_hit}},
    {"chatbot-kg-signal-cache-miss", [:brain, :kg_signal, :cache_miss], {:kg_signal, :cache_miss}},
    {"chatbot-kg-signal-cache-invalidated", [:brain, :kg_signal, :cache_invalidated],
     {:kg_signal, :cache_invalidated}},
    {"chatbot-kg-signal-srl-batch", [:brain, :kg_signal, :srl_batch], {:kg_signal, :srl_batch}},
    {"chatbot-novelty-downweighted", [:brain, :novelty, :downweighted], {:kg_signal, :novelty_downweighted}},
    {"chatbot-novelty-state-change", [:brain, :novelty, :state_change_detected], {:kg_signal, :novelty_state_change}}
  ]

  @handler_ids Enum.map(@handlers, &elem(&1, 0))

  @doc "The id of every handler `attach_handlers/0` attaches."
  @spec handler_ids() :: [String.t()]
  def handler_ids, do: @handler_ids

  @doc """
  Attaches all telemetry handlers. Call this during application startup.
  A handler attached already is left as it is, so calling it again attaches
  only the ones that are not.
  """
  def attach_handlers do
    :ok = Brain.Metrics.Aggregator.ensure_missed_counters()

    Enum.each(@handlers, fn {id, event, kind} ->
      case :telemetry.attach(id, event, &__MODULE__.handle_event/4, %{handler_id: id, kind: kind}) do
        :ok -> :ok
        {:error, :already_exists} -> :ok
      end
    end)

    :ok
  end

  @doc """
  Detaches every handler `attach_handlers/0` attaches. Useful for testing.
  """
  def detach_handlers do
    Enum.each(@handler_ids, &:telemetry.detach/1)
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

  @doc """
  Emits the result of a service health check: `status` is what the
  service's `health_check/1` returned, `duration_ms` how long it took.
  """
  def emit_service_health_check(service, status, duration_ms) do
    :telemetry.execute(
      @service_health_check ++ [:stop],
      %{duration: duration_ms},
      %{service: service, status: status, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Emits a model read from disk as it was loaded: `model` names it,
  `duration_ms` is how long the read took and `success` whether it loaded.
  Emitted where the model is read, not where a loaded copy is served.
  """
  def emit_model_load(model, duration_ms, success) when is_boolean(success) do
    :telemetry.execute(
      @model_load ++ [:stop],
      %{duration_ms: duration_ms},
      %{model: model, success: success, timestamp: System.monotonic_time(:millisecond)}
    )
  end

  @doc """
  Runs `load`, a function that reads `model` from disk, and emits the load
  (`emit_model_load/3`): a failure when it returns `:error` or
  `{:error, _}`, or raises (raised again), a success when it returns
  anything else. Returns what `load` returns.
  """
  def measure_model_load(model, load) when is_function(load, 0) do
    started = System.monotonic_time()

    result =
      try do
        load.()
      rescue
        exception ->
          emit_model_load(model, elapsed_ms(started), false)
          reraise exception, __STACKTRACE__
      end

    success =
      case result do
        :error -> false
        {:error, _reason} -> false
        _loaded -> true
      end

    emit_model_load(model, elapsed_ms(started), success)
    result
  end

  defp elapsed_ms(started) do
    System.convert_time_unit(System.monotonic_time() - started, :native, :microsecond) / 1000
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
  Emits the semantic facts retrieved for a turn in `world_id`.

  Measurements: `:count`, the number retrieved. Metadata: `:world_id`,
  `:semantic_fact_ids` and `:similarities` (in retrieval order), and
  `:recurring_pattern_id`, the fact found as the turn's recurring pattern
  (`Brain.Memory.RecurringPattern.find/2`), or `nil`.
  """
  def emit_semantic_facts_retrieved(world_id, semantic_facts, recurring_pattern_id) when is_list(semantic_facts) do
    :telemetry.execute(
      @semantic_fact_retrieved,
      %{count: length(semantic_facts)},
      %{
        world_id: world_id,
        semantic_fact_ids: Enum.map(semantic_facts, fn {fact, _similarity} -> fact.id end),
        similarities: Enum.map(semantic_facts, fn {_fact, similarity} -> similarity end),
        recurring_pattern_id: recurring_pattern_id,
        timestamp: System.monotonic_time(:millisecond)
      }
    )
  end

  @doc """
  Emits a response built from a recurring pattern
  (`Brain.Memory.RecurringPattern`), by `used_by`: `:synthesizer` or
  `:realization_packet`.

  Measurements: `:similarity`, `:evidence_count`, `:reply_count`. Metadata:
  `:world_id`, `:semantic_fact_id`, `:used_by`.
  """
  def emit_semantic_fact_used(%Brain.Memory.RecurringPattern{} = pattern, used_by) do
    :telemetry.execute(
      @semantic_fact_used,
      %{
        similarity: pattern.similarity,
        evidence_count: pattern.evidence_count,
        reply_count: pattern.reply_count
      },
      %{
        world_id: pattern.world_id,
        semantic_fact_id: pattern.fact.id,
        used_by: used_by,
        timestamp: System.monotonic_time(:millisecond)
      }
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

  # ============================================================================
  # The handler
  # ============================================================================

  @doc false
  # The one handler every event is attached to. It reads nothing out of the
  # event, so a malformed event cannot make it raise; the aggregator reads
  # the event and records a malformed one as missed. When the aggregator is
  # not running, `submit/2` logs the event and counts it as missed.
  def handle_event(_event, measurements, metadata, %{handler_id: handler_id, kind: kind}) do
    Brain.Metrics.Aggregator.submit(handler_id, {:telemetry, kind, measurements, metadata, DateTime.utc_now()})
  end
end
