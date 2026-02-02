defmodule Brain.Metrics.Aggregator do
  @moduledoc """
  Non-blocking metrics aggregator for telemetry events.

  Design principles for zero latency impact:
  1. All writes via `cast` - never blocks the caller
  2. ETS for reads - dashboard reads directly from ETS, never calls GenServer
  3. Atomic counters - uses `:ets.update_counter/3` for thread-safe increments
  4. Periodic aggregation - heavy computation happens on a timer, not per-event
  5. Bounded memory - sliding window with automatic expiration

  ## Usage

      # Recording metrics (fire-and-forget via telemetry handlers)
      GenServer.cast(Aggregator, {:record_duration, :brain_evaluate, 150, %{}})

      # Reading metrics (direct ETS read, non-blocking)
      Brain.Metrics.Aggregator.get_metrics()
  """

  use GenServer
  require Logger

  @metrics_table :chatbot_metrics
  @raw_data_table :chatbot_metrics_raw

  # Sliding window duration in milliseconds (5 minutes)
  @window_duration_ms 5 * 60 * 1000

  # Aggregation interval in milliseconds (10 seconds)
  @aggregation_interval_ms 10_000

  # Maximum raw data points to keep per metric
  @max_raw_points 1000

  # ============================================================================
  # Client API
  # ============================================================================

  @doc """
  Starts the Metrics.Aggregator GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Gets aggregated metrics. Reads directly from ETS - non-blocking.
  """
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

        _, acc ->
          acc
      end)
    catch
      :error, :badarg -> %{}
    end
  end

  @doc """
  Gets a specific metric. Reads directly from ETS - non-blocking.
  """
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

  @doc """
  Gets error metrics. Reads directly from ETS - non-blocking.
  """
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

  @doc """
  Gets queue size metrics. Reads directly from ETS - non-blocking.
  """
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
  Records a duration metric. Use cast for non-blocking.
  """
  def record_duration(metric_name, duration_ms, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:record_duration, metric_name, duration_ms, metadata})
  end

  @doc """
  Records an error. Use cast for non-blocking.
  """
  def record_error(metric_name, duration_ms, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:record_error, metric_name, duration_ms, metadata})
  end

  @doc """
  Records a queue size. Use cast for non-blocking.
  """
  def record_queue_size(genserver_name, queue_length) do
    GenServer.cast(__MODULE__, {:record_queue_size, genserver_name, queue_length})
  end

  @doc """
  Resets all metrics. Useful for testing.
  """
  def reset do
    GenServer.call(__MODULE__, :reset)
  end

  @doc """
  Gets training metrics. Reads directly from ETS - non-blocking.
  Returns a map of model name to training stats.
  """
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

  @doc """
  Gets model load metrics. Reads directly from ETS - non-blocking.
  """
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

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Create ETS tables with concurrent read access
    :ets.new(@metrics_table, [
      :named_table,
      :public,
      :set,
      read_concurrency: true
    ])

    :ets.new(@raw_data_table, [
      :named_table,
      :public,
      :set,
      read_concurrency: true
    ])

    # Initialize default metrics
    initialize_metrics()

    # Schedule periodic aggregation
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

    # Record the duration as well
    add_raw_data_point(metric_name, duration_ms, now)
    increment_counter(metric_name, :count)
    increment_counter(metric_name, :error_count)

    # Track error by type
    increment_error_counter(error_type)

    {:noreply, state}
  end

  @impl true
  def handle_cast({:record_error_event, error_type, _details}, state) do
    increment_error_counter(error_type)
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

    # Get existing training record or create new one
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

    # Also record as a duration metric for aggregation
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

    # Track error
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
  def handle_cast({:record_learning_event, event_type, _measurements, metadata}, state) do
    now = System.monotonic_time(:millisecond)
    world_id = Map.get(metadata, :world_id, "unknown")

    # Increment counter for this event type
    metric_key = {:learning, event_type}
    increment_counter(metric_key, :count)

    # Track per-world counts
    world_key = {:learning_world, world_id, event_type}
    increment_counter(world_key, :count)

    # Record timestamp
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

    # Track early exit counts by analyzer
    key = {:racing_early_exit, analyzer}
    increment_counter(key, :count)

    # Update aggregate metrics
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

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp initialize_metrics do
    # Initialize default metrics with zero values
    default_metrics = [
      # Core operations
      :brain_evaluate,
      :pipeline_process,
      :memory_query,
      :memory_embed,
      :gazetteer_lookup,
      # Knowledge Expansion operations
      :knowledge_research,
      :knowledge_corroborate,
      :knowledge_review,
      # Epistemic System operations
      :jtms_justify,
      :belief_operation,
      # Analysis operations
      :racing_analysis
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

    # Get existing data and add new point
    existing =
      case :ets.lookup(@raw_data_table, key) do
        [{^key, data}] -> data
        [] -> []
      end

    # Add new point and trim old ones, keeping max points
    new_data =
      [{duration_ms, timestamp} | existing]
      |> Enum.filter(fn {_val, ts} -> ts > cutoff end)
      |> Enum.take(@max_raw_points)

    :ets.insert(@raw_data_table, {key, new_data})
  end

  defp increment_counter(metric_name, counter_type) do
    key = {:counter, metric_name, counter_type}

    try do
      :ets.update_counter(@raw_data_table, key, {2, 1})
    catch
      :error, :badarg ->
        # Counter doesn't exist, create it
        :ets.insert(@raw_data_table, {key, 1})
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
      :gazetteer_lookup
    ]

    Enum.each(metrics, fn metric_name ->
      aggregate_metric(metric_name, now, cutoff)
    end)
  end

  defp aggregate_metric(metric_name, now, cutoff) do
    raw_key = {:raw, metric_name}

    # Get raw data points within the window
    data_points =
      case :ets.lookup(@raw_data_table, raw_key) do
        [{^raw_key, points}] ->
          points
          |> Enum.filter(fn {_val, ts} -> ts > cutoff end)

        [] ->
          []
      end

    # Get counters
    count = get_counter(metric_name, :count)
    error_count = get_counter(metric_name, :error_count)

    # Calculate statistics
    values = Enum.map(data_points, fn {val, _ts} -> val end)

    stats =
      if length(values) > 0 do
        sorted = Enum.sort(values)
        sum = Enum.sum(values)
        len = length(values)
        avg = sum / len
        min_val = List.first(sorted, 0)
        max_val = List.last(sorted, 0)
        p95_idx = round(len * 0.95) - 1
        p95 = Enum.at(sorted, max(p95_idx, 0), 0)

        # Calculate rate per minute
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

    # Update aggregated metrics
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
