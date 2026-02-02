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

  ## Analysis Events

  - `[:chat_bot, :analysis, :racing, :start | :stop | :exception]` - Racing analyzer parallel processing
  - `[:chat_bot, :analysis, :racing, :early_exit]` - Racing analyzer early exit (fast path)
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

  # Analysis Events
  @racing_analysis [:chat_bot, :analysis, :racing]
  @racing_early_exit [:chat_bot, :analysis, :racing, :early_exit]

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

      # Memory embed handlers
      {"chatbot-memory-embed-stop", @memory_embed ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :memory_embed}},

      # Gazetteer lookup handlers
      {"chatbot-gazetteer-lookup-stop", @gazetteer_lookup ++ [:stop],
       &__MODULE__.handle_span_stop/4, %{metric: :gazetteer_lookup}},

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

      # Racing Analyzer handlers
      {"chatbot-racing-analysis-stop", @racing_analysis ++ [:stop], &__MODULE__.handle_span_stop/4,
       %{metric: :racing_analysis}},
      {"chatbot-racing-analysis-exception", @racing_analysis ++ [:exception],
       &__MODULE__.handle_span_exception/4, %{metric: :racing_analysis}},
      {"chatbot-racing-early-exit", @racing_early_exit, &__MODULE__.handle_racing_early_exit/4, %{}}
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
      "chatbot-gazetteer-lookup-stop",
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
      "chatbot-jtms-justify-stop",
      "chatbot-jtms-justify-exception",
      "chatbot-belief-operation-stop",
      "chatbot-belief-operation-exception",
      # Racing analyzer events
      "chatbot-racing-analysis-stop",
      "chatbot-racing-analysis-exception",
      "chatbot-racing-early-exit"
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

  # ============================================================================
  # Helpers
  # ============================================================================

  defp native_to_ms(duration) when is_integer(duration) do
    System.convert_time_unit(duration, :native, :millisecond)
  end

  defp native_to_ms(_), do: 0
end
