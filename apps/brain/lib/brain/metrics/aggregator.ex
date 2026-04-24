defmodule Brain.Metrics.Aggregator do
  @moduledoc "Non-blocking metrics aggregator for telemetry events.\n\nDesign principles for zero latency impact:\n1. All writes via `cast` - never blocks the caller\n2. ETS for reads - dashboard reads directly from ETS, never calls GenServer\n3. Atomic counters - uses `:ets.update_counter/3` for thread-safe increments\n4. Periodic aggregation - heavy computation happens on a timer, not per-event\n5. Bounded memory - sliding window with automatic expiration\n\n## Usage\n\n    # Recording metrics (fire-and-forget via telemetry handlers)\n    GenServer.cast(Aggregator, {:record_duration, :brain_evaluate, 150, %{}})\n\n    # Reading metrics (direct ETS read, non-blocking)\n    Brain.Metrics.Aggregator.get_metrics()\n"

  use GenServer
  require Logger

  @metrics_table :chatbot_metrics
  @raw_data_table :chatbot_metrics_raw
  @window_duration_ms 5 * 60 * 1000
  @aggregation_interval_ms 10_000
  @max_raw_points 1000

  @doc """
  Starts the Metrics.Aggregator GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Gets aggregated metrics. Reads directly from ETS - non-blocking.\n"
  def get_metrics do
    try do
      @metrics_table
      |> :ets.tab2list()
      |> Enum.reduce(%{}, fn
        {{:metric, name}, data}, acc ->
          Map.put(acc, name, data)

        {{:error, name}, data}, acc ->
          errors = Map.get(acc, :errors, %{})
          Map.put(acc, :errors, Map.put(errors, name, data))

        {{:queue, name}, data}, acc ->
          queues = Map.get(acc, :queues, %{})
          Map.put(acc, :queues, Map.put(queues, name, data))

        {{:training, name}, data}, acc ->
          training = Map.get(acc, :training, %{})
          Map.put(acc, :training, Map.put(training, name, data))

        {{:model_load, name}, data}, acc ->
          model_loads = Map.get(acc, :model_loads, %{})
          Map.put(acc, :model_loads, Map.put(model_loads, name, data))

        {{:readiness, name}, data}, acc ->
          readiness = Map.get(acc, :readiness, %{})
          Map.put(acc, :readiness, Map.put(readiness, name, data))

        {{:evaluation, task}, data}, acc ->
          evaluation = Map.get(acc, :evaluation, %{})
          Map.put(acc, :evaluation, Map.put(evaluation, task, data))

        _, acc ->
          acc
      end)
    catch
      :error, :badarg -> %{}
    end
  end

  @doc "Gets a specific metric. Reads directly from ETS - non-blocking.\n"
  def get_metric(name) do
    try do
      case :ets.lookup(@metrics_table, {:metric, name}) do
        [{{:metric, ^name}, data}] -> data
        [] -> nil
      end
    catch
      :error, :badarg -> nil
    end
  end

  @doc "Gets error metrics. Reads directly from ETS - non-blocking.\n"
  def get_errors do
    try do
      @metrics_table
      |> :ets.match({{:error, :"$1"}, :"$2"})
      |> Enum.map(fn [name, data] -> {name, data} end)
      |> Map.new()
    catch
      :error, :badarg -> %{}
    end
  end

  @doc "Gets queue size metrics. Reads directly from ETS - non-blocking.\n"
  def get_queue_sizes do
    try do
      @metrics_table
      |> :ets.match({{:queue, :"$1"}, :"$2"})
      |> Enum.map(fn [name, data] -> {name, data} end)
      |> Map.new()
    catch
      :error, :badarg -> %{}
    end
  end

  @doc """
  Gets external service metrics. Reads directly from ETS - non-blocking.

  Returns a map with:
  - `:dispatch` - Overall dispatch metrics (count, avg_ms, etc.)
  - `:by_service` - Per-service breakdown
  - `:cache` - Cache hit/miss ratios
  - `:health` - Last health check status per service
  """
  def get_service_metrics do
    try do
      dispatch_metrics = get_metric(:service_dispatch) || %{count: 0}
      enrichment_metrics = get_metric(:service_enrichment) || %{count: 0}

      # Get per-service metrics by finding services in raw data counters
      # Look for {:counter, {:service, name}, :count} entries in raw_data_table
      by_service =
        @raw_data_table
        |> :ets.match({{:counter, {:service, :"$1"}, :count}, :"$2"})
        |> Enum.map(fn [name, _count] -> name end)
        |> Enum.uniq()
        |> Enum.map(fn service ->
          count = get_counter_for_service({:service, service}, :count)
          success = get_counter_for_service({:service, service}, :success_count)
          errors = get_counter_for_service({:service, service}, :error_count)
          last_dispatch = get_last_event({:service_dispatch_last, service})
          health = get_last_event({:service_health, service})

          {service,
           %{
             total_dispatches: count,
             success_count: success,
             error_count: errors,
             success_rate: if(count > 0, do: Float.round(success / count * 100, 1), else: 0.0),
             last_dispatch: last_dispatch,
             health_status: health
           }}
        end)
        |> Map.new()

      # Get cache metrics
      cache_hits = get_counter_for_service({:service_cache, :hit}, :count)
      cache_misses = get_counter_for_service({:service_cache, :miss}, :count)
      total_cache = cache_hits + cache_misses

      cache_metrics = %{
        hits: cache_hits,
        misses: cache_misses,
        hit_rate: if(total_cache > 0, do: Float.round(cache_hits / total_cache * 100, 1), else: 0.0)
      }

      %{
        dispatch: dispatch_metrics,
        enrichment: enrichment_metrics,
        by_service: by_service,
        cache: cache_metrics
      }
    catch
      :error, :badarg -> %{dispatch: %{count: 0}, enrichment: %{count: 0}, by_service: %{}, cache: %{}}
    end
  end

  defp get_counter_for_service(key, field) do
    case :ets.lookup(@raw_data_table, {:counter, key, field}) do
      [{{:counter, ^key, ^field}, count}] -> count
      [] -> 0
    end
  rescue
    _ -> 0
  end

  defp get_last_event(key) do
    case :ets.lookup(@metrics_table, key) do
      [{^key, data}] -> data
      [] -> nil
    end
  rescue
    _ -> nil
  end

  @doc """
  Gets epistemic system metrics. Reads directly from ETS - non-blocking.

  Returns a map with:
  - `:total_verifications` - Total fact verification count
  - `:by_status` - Count by status (verified, contradicted, uncertain, unchecked)
  - `:contradiction_count` - Total contradictions detected
  - `:last_verifications` - Last verification per status type
  """
  def get_epistemic_metrics do
    try do
      total = get_counter_value(:fact_verification, :count)
      contradictions = get_counter_value(:fact_verification, :contradiction_count)

      # Get counts by status
      by_status =
        [:verified, :contradicted, :uncertain, :unchecked]
        |> Enum.map(fn status ->
          count = get_counter_value({:fact_verification_status, status}, :count)
          {status, count}
        end)
        |> Map.new()

      # Get last verification per status
      last_verifications =
        [:verified, :contradicted, :uncertain]
        |> Enum.map(fn status ->
          last = get_last_event({:fact_verification_last, status})
          {status, last}
        end)
        |> Enum.filter(fn {_, v} -> v != nil end)
        |> Map.new()

      %{
        total_verifications: total,
        contradiction_count: contradictions,
        by_status: by_status,
        last_verifications: last_verifications
      }
    catch
      :error, :badarg ->
        %{total_verifications: 0, contradiction_count: 0, by_status: %{}, last_verifications: %{}}
    end
  end

  @doc """
  Gets response enrichment metrics (fact/semantic retrieval hit rates).
  Reads directly from ETS - non-blocking.
  """
  def get_enrichment_metrics do
    try do
      fact_total = get_counter_value(:fact_retrieval, :total)
      fact_hits = get_counter_value(:fact_retrieval, :hits)
      fact_results = get_counter_value(:fact_retrieval, :total_results)

      semantic_total = get_counter_value(:semantic_retrieval, :total)
      semantic_hits = get_counter_value(:semantic_retrieval, :hits)

      fact_hit_rate =
        if fact_total > 0, do: Float.round(fact_hits / fact_total * 100, 1), else: 0.0

      semantic_hit_rate =
        if semantic_total > 0, do: Float.round(semantic_hits / semantic_total * 100, 1), else: 0.0

      avg_facts =
        if fact_total > 0, do: Float.round(fact_results / fact_total, 1), else: 0.0

      %{
        fact_hit_rate: fact_hit_rate,
        semantic_hit_rate: semantic_hit_rate,
        avg_facts_per_response: avg_facts,
        fact_queries: fact_total,
        semantic_queries: semantic_total
      }
    catch
      :error, :badarg ->
        %{fact_hit_rate: 0.0, semantic_hit_rate: 0.0, avg_facts_per_response: 0.0,
          fact_queries: 0, semantic_queries: 0}
    end
  end

  @doc """
  Gets fact verification miss reason breakdown.
  Reads directly from ETS - non-blocking.
  """
  def get_verification_miss_reasons do
    try do
      reasons = [:no_subject, :no_facts, :no_beliefs, :low_confidence]

      reason_counts =
        Enum.map(reasons, fn reason ->
          {reason, get_counter_value({:verification_miss, reason}, :count)}
        end)
        |> Map.new()

      total = Enum.sum(Map.values(reason_counts))

      reason_pcts =
        if total > 0 do
          Enum.map(reason_counts, fn {k, v} ->
            {k, Float.round(v / total * 100, 1)}
          end)
          |> Map.new()
        else
          Map.new(reasons, &{&1, 0.0})
        end

      %{counts: reason_counts, percentages: reason_pcts, total: total}
    catch
      :error, :badarg ->
        %{counts: %{}, percentages: %{}, total: 0}
    end
  end

  defp get_counter_value(key, field) do
    case :ets.lookup(@raw_data_table, {:counter, key, field}) do
      [{{:counter, ^key, ^field}, count}] -> count
      [] -> 0
    end
  rescue
    _ -> 0
  end

  @doc "Records a duration metric. Use cast for non-blocking.\n"
  def record_duration(metric_name, duration_ms, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:record_duration, metric_name, duration_ms, metadata})
  end

  @doc "Records an error. Use cast for non-blocking.\n"
  def record_error(metric_name, duration_ms, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:record_error, metric_name, duration_ms, metadata})
  end

  @doc "Records a queue size. Use cast for non-blocking.\n"
  def record_queue_size(genserver_name, queue_length) do
    GenServer.cast(__MODULE__, {:record_queue_size, genserver_name, queue_length})
  end

  @doc "Records readiness state for a subsystem. Use cast for non-blocking.\n"
  def record_readiness(system_name, ready?) when is_atom(system_name) or is_binary(system_name) do
    GenServer.cast(__MODULE__, {:record_readiness, system_name, ready?})
  end

  @doc "Returns true if the Aggregator is ready to accept requests."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Resets all metrics. Useful for testing.\n"
  def reset do
    GenServer.call(__MODULE__, :reset, 5_000)
  end


  @doc "Gets training metrics. Reads directly from ETS - non-blocking.\nReturns a map of model name to training stats.\n"
  def get_training_stats do
    try do
      @metrics_table
      |> :ets.match({{:training, :"$1"}, :"$2"})
      |> Enum.map(fn [model, data] -> {model, data} end)
      |> Map.new()
    catch
      :error, :badarg -> %{}
    end
  end

  @doc "Gets model load metrics. Reads directly from ETS - non-blocking.\n"
  def get_model_load_stats do
    try do
      @metrics_table
      |> :ets.match({{:model_load, :"$1"}, :"$2"})
      |> Enum.map(fn [model, data] -> {model, data} end)
      |> Map.new()
    catch
      :error, :badarg -> %{}
    end
  end

  @doc "Gets latest evaluation results per task. Reads directly from ETS.\n"
  def get_evaluation_metrics do
    try do
      @metrics_table
      |> :ets.match({{:evaluation, :"$1"}, :"$2"})
      |> Enum.map(fn [task, data] -> {task, data} end)
      |> Map.new()
    catch
      :error, :badarg -> %{}
    end
  end

  @impl true
  def init(_opts) do
    :ets.new(@metrics_table, [:named_table, :public, :set, read_concurrency: true])

    :ets.new(@raw_data_table, [:named_table, :public, :set, read_concurrency: true])
    initialize_metrics()
    schedule_aggregation()

    Logger.info("Metrics.Aggregator started")

    {:ok, %{started_at: System.monotonic_time(:millisecond)}}
  end

  @impl true
  def handle_cast({:record_duration, metric_name, duration_ms, _metadata}, state) do
    now = System.monotonic_time(:millisecond)
    add_raw_data_point(metric_name, duration_ms, now)
    increment_counter(metric_name, :count)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_error, metric_name, duration_ms, metadata}, state) do
    now = System.monotonic_time(:millisecond)
    error_type = Map.get(metadata, :kind, :unknown)
    add_raw_data_point(metric_name, duration_ms, now)
    increment_counter(metric_name, :count)
    increment_counter(metric_name, :error_count)
    increment_error_counter(error_type)

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_error_event, error_type, _details}, state) do
    increment_error_counter(error_type)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_readiness, system_name, ready?}, state) do
    try do
      :ets.insert(
        @metrics_table,
        {{:readiness, system_name},
         %{
           ready?: ready?,
           timestamp: System.monotonic_time(:millisecond)
         }}
      )
    catch
      :error, :badarg -> :ok
    end

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_queue_size, genserver_name, queue_length}, state) do
    :ets.insert(
      @metrics_table,
      {{:queue, genserver_name},
       %{
         length: queue_length,
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_training_start, model, sequence_count, metadata}, state) do
    now = System.monotonic_time(:millisecond)

    :ets.insert(
      @metrics_table,
      {{:training, model},
       %{
         status: :in_progress,
         started_at: metadata[:started_at] || DateTime.utc_now(),
         sequence_count: sequence_count,
         duration_ms: nil,
         tag_count: nil,
         feature_count: nil,
         last_updated: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_training_stop, model, measurements, _metadata}, state) do
    now = System.monotonic_time(:millisecond)

    existing =
      case :ets.lookup(@metrics_table, {:training, model}) do
        [{{:training, ^model}, data}] -> data
        [] -> %{}
      end

    updated = %{
      status: :completed,
      started_at: existing[:started_at],
      completed_at: DateTime.utc_now(),
      sequence_count: measurements[:sequence_count] || existing[:sequence_count],
      duration_ms: measurements[:duration_ms],
      tag_count: measurements[:tag_count],
      feature_count: measurements[:feature_count],
      success: true,
      last_updated: now
    }

    :ets.insert(@metrics_table, {{:training, model}, updated})
    add_raw_data_point({:train, model}, measurements[:duration_ms] || 0, now)
    increment_counter({:train, model}, :count)

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_training_exception, model, measurements, metadata}, state) do
    now = System.monotonic_time(:millisecond)

    existing =
      case :ets.lookup(@metrics_table, {:training, model}) do
        [{{:training, ^model}, data}] -> data
        [] -> %{}
      end

    updated = %{
      status: :failed,
      started_at: existing[:started_at],
      completed_at: DateTime.utc_now(),
      sequence_count: measurements[:sequence_count] || existing[:sequence_count],
      duration_ms: measurements[:duration_ms],
      reason: metadata[:reason],
      success: false,
      last_updated: now
    }

    :ets.insert(@metrics_table, {{:training, model}, updated})
    increment_error_counter({:training_failed, model})

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_model_load, model, duration_ms, metadata}, state) do
    now = System.monotonic_time(:millisecond)

    :ets.insert(
      @metrics_table,
      {{:model_load, model},
       %{
         loaded_at: DateTime.utc_now(),
         duration_ms: duration_ms,
         success: metadata[:success] != false,
         last_updated: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_evaluation_complete, task, measurements}, state) do
    now = System.monotonic_time(:millisecond)

    :ets.insert(
      @metrics_table,
      {{:evaluation, task},
       %{
         accuracy: measurements[:accuracy] || 0.0,
         macro_f1: measurements[:macro_f1] || 0.0,
         weighted_f1: measurements[:weighted_f1] || 0.0,
         total_examples: measurements[:total_examples] || 0,
         duration_ms: measurements[:duration_ms] || 0,
         completed_at: DateTime.utc_now(),
         last_updated: now
       }}
    )

    {:noreply, state}
  end


  @impl true
  def handle_cast({:record_consistency_disagreement, metadata}, state) do
    severity = Map.get(metadata, :severity, :low)
    increment_counter(:consistency, :total)
    increment_counter(:consistency, :"#{severity}_count")

    :ets.insert(
      @metrics_table,
      {{:consistency, :last_disagreement},
       %{
         final_intent: Map.get(metadata, :final_intent),
         consensus_intent: Map.get(metadata, :consensus_intent),
         severity: severity,
         agreeing: Map.get(metadata, :agreeing, []),
         dissenting: Map.get(metadata, :dissenting, []),
         signals: Map.get(metadata, :signals, []),
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_learning_event, event_type, _measurements, metadata}, state) do
    now = System.monotonic_time(:millisecond)
    world_id = Map.get(metadata, :world_id, "unknown")
    metric_key = {:learning, event_type}
    increment_counter(metric_key, :count)
    world_key = {:learning_world, world_id, event_type}
    increment_counter(world_key, :count)

    :ets.insert(
      @metrics_table,
      {{:learning_last, event_type},
       %{
         timestamp: now,
         world_id: world_id
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_racing_early_exit, analyzer, confidence, duration_ms}, state) do
    now = System.monotonic_time(:millisecond)
    key = {:racing_early_exit, analyzer}
    increment_counter(key, :count)

    :ets.insert(
      @metrics_table,
      {{:racing_early_exit_last, analyzer},
       %{
         timestamp: now,
         confidence: confidence,
         duration_ms: duration_ms
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast(
        {:record_code_file_processed, file_path, language, symbols_count, relations_count,
         duration_ms},
        state
      ) do
    now = System.monotonic_time(:millisecond)
    add_raw_data_point(:code_pipeline, duration_ms, now)
    increment_counter(:code_pipeline, :count)
    lang_key = {:code_language, language}
    increment_counter(lang_key, :count)

    :ets.insert(
      @metrics_table,
      {{:code_file_last, language},
       %{
         file_path: file_path,
         symbols_count: symbols_count,
         relations_count: relations_count,
         duration_ms: duration_ms,
         timestamp: now
       }}
    )

    {:noreply, state}
  end

  # ============================================================================
  # External Services Handlers
  # ============================================================================

  @impl true
  def handle_cast({:record_service_dispatch, service, intent, status, duration_ms}, state) do
    now = System.monotonic_time(:millisecond)
    add_raw_data_point(:service_dispatch, duration_ms, now)
    increment_counter(:service_dispatch, :count)

    # Track per-service metrics
    service_key = {:service, service}
    increment_counter(service_key, :count)

    # Track success/failure
    case status do
      :success ->
        increment_counter(service_key, :success_count)

      {:error, _} ->
        increment_counter(service_key, :error_count)

      _ ->
        :ok
    end

    # Store last dispatch info
    :ets.insert(
      @metrics_table,
      {{:service_dispatch_last, service},
       %{
         intent: intent,
         status: status,
         duration_ms: duration_ms,
         timestamp: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_service_cache, type, service, count}, state) do
    cache_key = {:service_cache, type}
    increment_counter(cache_key, :count, count || 1)

    service_cache_key = {:service_cache, service, type}
    increment_counter(service_cache_key, :count, count || 1)

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_service_health_check, service, status, duration_ms}, state) do
    now = System.monotonic_time(:millisecond)

    :ets.insert(
      @metrics_table,
      {{:service_health, service},
       %{
         status: status,
         duration_ms: duration_ms,
         timestamp: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_service_credential, operation, service, _count}, state) do
    now = System.monotonic_time(:millisecond)
    cred_key = {:service_credential, operation}
    increment_counter(cred_key, :count)

    :ets.insert(
      @metrics_table,
      {{:service_credential_last, service},
       %{
         operation: operation,
         timestamp: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_fact_verification, status, subject, beliefs_count, duration_ms}, state) do
    now = System.monotonic_time(:millisecond)

    # Track overall fact verification metrics
    add_raw_data_point(:fact_verification, duration_ms, now)
    increment_counter(:fact_verification, :count)

    # Track by status (verified, contradicted, uncertain, unchecked)
    status_key = {:fact_verification_status, status}
    increment_counter(status_key, :count)

    # Track contradictions specifically
    if status == :contradicted do
      increment_counter(:fact_verification, :contradiction_count)
    end

    # Store last verification info
    :ets.insert(
      @metrics_table,
      {{:fact_verification_last, status},
       %{
         subject: subject,
         beliefs_count: beliefs_count,
         duration_ms: duration_ms,
         timestamp: now
       }}
    )

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_fact_retrieval, result_count}, state) do
    increment_counter(:fact_retrieval, :total)
    if result_count > 0, do: increment_counter(:fact_retrieval, :hits)
    increment_counter(:fact_retrieval, :total_results, result_count)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_semantic_retrieval, result_count}, state) do
    increment_counter(:semantic_retrieval, :total)
    if result_count > 0, do: increment_counter(:semantic_retrieval, :hits)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_verification_miss_reason, reason}, state) do
    increment_counter({:verification_miss, reason}, :count)
    {:noreply, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    :ets.delete_all_objects(@metrics_table)
    :ets.delete_all_objects(@raw_data_table)
    initialize_metrics()
    {:reply, :ok, state}
  end

  @impl true
  def handle_info(:aggregate, state) do
    perform_aggregation()
    schedule_aggregation()
    {:noreply, state}
  end

  defp initialize_metrics do
    default_metrics = [
      :brain_evaluate,
      :pipeline_process,
      :memory_query,
      :memory_embed,
      :gazetteer_lookup,
      :response_generate,
      :response_template_lookup,
      :fact_database_query,
      :knowledge_research,
      :knowledge_corroborate,
      :knowledge_review,
      :jtms_justify,
      :belief_operation,
      :racing_analysis,
      :code_pipeline,
      :code_parse,
      :code_extract,
      :code_gazetteer_lookup,
      # External services
      :service_dispatch,
      :service_enrichment
    ]

    Enum.each(default_metrics, fn name ->
      :ets.insert(
        @metrics_table,
        {{:metric, name},
         %{
           count: 0,
           error_count: 0,
           avg_ms: 0.0,
           min_ms: 0,
           max_ms: 0,
           p95_ms: 0,
           rate_per_minute: 0.0,
           last_updated: System.monotonic_time(:millisecond)
         }}
      )

      :ets.insert(@raw_data_table, {{:raw, name}, []})
      :ets.insert(@raw_data_table, {{:counter, name, :count}, 0})
      :ets.insert(@raw_data_table, {{:counter, name, :error_count}, 0})
    end)
  end

  defp add_raw_data_point(metric_name, duration_ms, timestamp) do
    key = {:raw, metric_name}
    cutoff = timestamp - @window_duration_ms

    existing =
      case :ets.lookup(@raw_data_table, key) do
        [{^key, data}] -> data
        [] -> []
      end

    new_data =
      [{duration_ms, timestamp} | existing]
      |> Enum.filter(fn {_val, ts} -> ts > cutoff end)
      |> Enum.take(@max_raw_points)

    :ets.insert(@raw_data_table, {key, new_data})
  end

  defp increment_counter(metric_name, counter_type, amount \\ 1)

  defp increment_counter(metric_name, counter_type, amount) when is_integer(amount) do
    key = {:counter, metric_name, counter_type}

    try do
      :ets.update_counter(@raw_data_table, key, {2, amount})
    catch
      :error, :badarg ->
        :ets.insert(@raw_data_table, {key, amount})
    end
  end

  defp increment_error_counter(error_type) do
    key = {:error, error_type}

    try do
      case :ets.lookup(@metrics_table, key) do
        [{^key, data}] ->
          :ets.insert(
            @metrics_table,
            {key, %{data | count: data.count + 1, last_seen: System.monotonic_time(:millisecond)}}
          )

        [] ->
          :ets.insert(
            @metrics_table,
            {key,
             %{
               count: 1,
               first_seen: System.monotonic_time(:millisecond),
               last_seen: System.monotonic_time(:millisecond)
             }}
          )
      end
    catch
      :error, :badarg -> :ok
    end
  end

  defp schedule_aggregation do
    Process.send_after(self(), :aggregate, @aggregation_interval_ms)
  end

  defp perform_aggregation do
    now = System.monotonic_time(:millisecond)
    cutoff = now - @window_duration_ms

    metrics = [
      :brain_evaluate,
      :pipeline_process,
      :memory_query,
      :memory_embed,
      :gazetteer_lookup,
      :response_generate,
      :response_template_lookup,
      :fact_database_query,
      :code_pipeline,
      :code_parse,
      :code_extract,
      :code_gazetteer_lookup,
      # External services
      :service_dispatch,
      :service_enrichment,
      # Knowledge expansion
      :knowledge_research,
      :knowledge_corroborate,
      :knowledge_review,
      # Epistemic
      :jtms_justify,
      :belief_operation,
      # Racing analyzer
      :racing_analysis
    ]

    Enum.each(metrics, fn metric_name ->
      aggregate_metric(metric_name, now, cutoff)
    end)
  end

  defp aggregate_metric(metric_name, now, cutoff) do
    raw_key = {:raw, metric_name}

    data_points =
      case :ets.lookup(@raw_data_table, raw_key) do
        [{^raw_key, points}] ->
          points
          |> Enum.filter(fn {_val, ts} -> ts > cutoff end)

        [] ->
          []
      end

    count = get_counter(metric_name, :count)
    error_count = get_counter(metric_name, :error_count)
    values = Enum.map(data_points, fn {val, _ts} -> val end)

    stats =
      if values != [] do
        sorted = Enum.sort(values)
        sum = Enum.sum(values)
        len = length(values)
        avg = sum / len
        min_val = List.first(sorted, 0)
        max_val = List.last(sorted, 0)
        p95_idx = round(len * 0.95) - 1
        p95 = Enum.at(sorted, max(p95_idx, 0), 0)
        window_seconds = @window_duration_ms / 1000
        rate = len / window_seconds * 60

        %{
          count: count,
          error_count: error_count,
          avg_ms: Float.round(avg, 2),
          min_ms: min_val,
          max_ms: max_val,
          p95_ms: p95,
          rate_per_minute: Float.round(rate, 2),
          sample_count: len,
          last_updated: now
        }
      else
        %{
          count: count,
          error_count: error_count,
          avg_ms: 0.0,
          min_ms: 0,
          max_ms: 0,
          p95_ms: 0,
          rate_per_minute: 0.0,
          sample_count: 0,
          last_updated: now
        }
      end

    :ets.insert(@metrics_table, {{:metric, metric_name}, stats})
  end

  defp get_counter(metric_name, counter_type) do
    key = {:counter, metric_name, counter_type}

    case :ets.lookup(@raw_data_table, key) do
      [{^key, count}] -> count
      [] -> 0
    end
  end
end
