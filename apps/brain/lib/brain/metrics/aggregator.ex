defmodule Brain.Metrics.Aggregator do
  # How long its supervisor gives it to stop, and what it does in that time.
  @shutdown_ms 15_000
  @drain_ms 2_000
  @final_write_ms 10_000
  # How often, and after how many events, the batch is written.
  @batch_interval_ms 1_000
  @batch_max_events 500
  # How long after a logged miss from a sender for a reason the next ones are
  # logged together, and how often the aggregator logs what is not logged yet.
  @missed_log_interval_ms 10_000

  @moduledoc """
  Counts the events `Brain.Telemetry` hands it, off the emitting process's
  path, and answers what was counted.

  ## Where each metric lives

  What must survive a restart is written to Atlas (`Atlas.Metrics`), in
  measurement windows, and every reader reads it from Atlas; there is no
  copy in memory to diverge. Durations are kept as aggregates, never as raw
  events: count, sum, min, max and a histogram.

  Kept in Atlas, for the whole node (the `"telemetry"` family, `:node`):

  - span durations (`get_metrics/0`, `get_metric/1`): count, errors, sum,
    min, max and histogram buckets per span; the events carry no world, or
    not on every span
  - errors (`get_errors/0`): count and last seen per error type
  - training runs (`get_metrics/0` `:training`): the last run of each model
  - fact verification (`get_epistemic_metrics/0`): count by status,
    contradictions, and its duration
  - response enrichment (`get_enrichment_metrics/0`) and verification miss
    reasons (`get_verification_miss_reasons/0`)
  - service dispatches, per service and as a duration, and service cache
    hits and misses (`get_service_metrics/0`)
  - KG signal, learning, consistency, racing early exit and code language
    counts (`get_activity_metrics/0`)
  - events the aggregator missed (`get_missed_events/0`)

  Kept in Atlas per world:

  - semantic facts (the `"semantic_facts"` family,
    `get_semantic_fact_metrics/1`): the events name their world
  - learning events and credential operations (the `"telemetry"` family):
    the events name their world

  Kept in memory only, because each describes the live process or the
  moment, not history (`get_metrics/0`):

  - readiness: a gauge of the running processes
  - model loads: what this node loaded last, per model
  - the rate per minute and when each duration was last seen: how busy this
    process is now
  - the last service health check, and the last dispatch, verification,
    learning event, racing early exit, code file, credential operation and
    consistency disagreement: snapshots of the latest event

  Evaluation results are kept by `Brain.ML.EvaluationStore`, in Atlas, and
  read from there; the aggregator does not keep a copy.

  ## Writes are batched

  The aggregator combines what the events it handles add, as Atlas would
  (sums added, the smaller minimum and the larger maximum kept, the later
  record kept), and writes them in one transaction every
  #{@batch_interval_ms} milliseconds, as soon as #{@batch_max_events} events are waiting, before it
  starts a window (`reset/0`, `start_semantic_fact_window/1`), when asked
  (`flush/0`) and when it stops. What it answers is read from Atlas, so an
  event shows there once its batch is written. The missed counts held in
  memory are written in the same transaction.

  ## Nothing is missed silently

  An event the aggregator cannot count is counted as missed, by who sent it
  and why (`get_missed_events/0`), and logged as an error:

  - `:aggregator_not_running` - it was sent while the aggregator was not
    running, as while its supervisor restarts it (`submit/2`)
  - `:atlas_write_failed` - Atlas refused or could not take the batch it was
    in; the aggregator logs it and goes on, so an unavailable Atlas never
    crashes it into a restart loop
  - `:malformed_event` - the event lacked what it is counted by
  - `:aggregator_stopped` - it was still waiting in the mailbox when the
    aggregator, stopping, had used the time it takes to count what waits
    (`:drain_ms`, #{@drain_ms} milliseconds unless started with another)

  The first miss from a sender for a reason is logged as it happens; the
  misses after it from the same sender for the same reason, within
  #{@missed_log_interval_ms} milliseconds, are counted and logged together:
  by the sender's next miss after that time, or by the aggregator, which
  every #{@missed_log_interval_ms} milliseconds and when it stops logs one
  line with what was missed and not yet logged, by sender and reason. So an
  Atlas outage logs a line per sender and reason in that time, not one per
  event, and every miss is in a line once the aggregator runs.

  Missed counts are held in memory, outside the aggregator's process, until
  the aggregator writes them to Atlas. Any other exception crashes the
  aggregator, loudly; its supervisor restarts it.

  ## Stopping

  Its supervisor gives it #{@shutdown_ms} milliseconds to stop. Stopping, it
  stops taking events (a sender from then on counts its event as missed),
  counts what waits in its mailbox for up to `:drain_ms`, counts the rest
  as missed, and writes the batch in at most #{@final_write_ms} milliseconds; a batch
  that cannot be written in that time is counted as missed. So it always
  finishes within the time its supervisor gives it, and no event in its
  mailbox goes uncounted.
  """

  use GenServer, shutdown: @shutdown_ms
  require Logger

  @gauges_table :chatbot_metrics
  @telemetry_family "telemetry"
  @semantic_fact_family "semantic_facts"
  @rate_window_ms 5 * 60 * 1000
  @max_rate_points 1000

  # The spans `Brain.Telemetry` measures and the durations the aggregator
  # records itself; each is in `get_metrics/0`, counted or not.
  @duration_metrics [
    :brain_evaluate,
    :pipeline_process,
    :memory_query,
    :memory_embed,
    :gazetteer_lookup,
    :response_generate,
    :response_template_lookup,
    :fact_database_query,
    :entity_extract,
    :knowledge_research,
    :knowledge_corroborate,
    :knowledge_review,
    :jtms_justify,
    :belief_operation,
    :racing_analysis,
    :code_parse,
    :code_extract,
    :code_pipeline,
    :code_gazetteer_lookup,
    :code_gazetteer_add,
    :service_dispatch,
    :service_enrichment,
    :fact_verification
  ]

  # Upper bounds of the duration histogram's buckets, in milliseconds; a
  # duration above the last is in the `:inf` bucket.
  @duration_buckets_ms [1, 2, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 30_000, 60_000]

  # Who builds a response from a recurring pattern (`Brain.Memory.RecurringPattern.used_by/0`).
  @semantic_fact_users [:synthesizer, :realization_packet]
  @verification_statuses [:verified, :contradicted, :uncertain, :unchecked]
  @verification_miss_reasons [:no_subject, :no_facts, :no_beliefs, :low_confidence]
  @training_statuses %{"in_progress" => :in_progress, "completed" => :completed, "failed" => :failed}

  # Who sends the aggregator events besides the telemetry handlers: its own
  # record functions.
  @direct_sources ~w(record_duration record_error record_readiness record_fact_retrieval
                     record_semantic_retrieval record_verification_miss_reason)
  @miss_reasons [:aggregator_not_running, :atlas_write_failed, :malformed_event, :aggregator_stopped]
  @missed_key {__MODULE__, :missed}

  @typedoc """
  The node-wide telemetry window as read from Atlas once (`read_telemetry/0`),
  for readers that answer several questions from one read.
  """
  @type telemetry :: %{window: Atlas.Schemas.MetricWindow.t() | nil, values: map(), records: map()}

  defmodule MalformedEvent do
    @moduledoc false
    defexception [:message]
  end

  @doc """
  Starts the Metrics.Aggregator GenServer.

  ## Options
    - `:name` - The name to register under (default: `#{__MODULE__}`)
    - `:drain_ms` - how long, stopping, it counts what waits in its mailbox
      before it counts the rest as missed (default: #{@drain_ms})
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  # ============================================================================
  # Sending events
  # ============================================================================

  @doc """
  Sends `message` from `source` (a telemetry handler id, or one of the
  aggregator's own record functions) to the aggregator, without waiting.
  Never raises: when the aggregator is not running, the event is counted as
  missed (`get_missed_events/0`) and logged.
  """
  @spec submit(String.t(), tuple()) :: :ok
  def submit(source, message) when is_binary(source) and is_tuple(message) do
    send(__MODULE__, {:metric_event, source, message})
    :ok
  rescue
    # Sending to a name nothing is registered under raises.
    ArgumentError -> missed(source, :aggregator_not_running, 1, describe(message))
  end

  @doc "Records a duration metric, in milliseconds, without waiting.\n"
  def record_duration(metric_name, duration_ms, _metadata \\ %{}) do
    submit("record_duration", {:record_duration, metric_name, duration_ms})
  end

  @doc "Records a duration metric that raised; `metadata` names the `:kind` of error.\n"
  def record_error(metric_name, duration_ms, metadata) do
    submit("record_error", {:record_error, metric_name, duration_ms, metadata})
  end

  @doc "Records readiness state for a subsystem, without waiting.\n"
  def record_readiness(system_name, ready?) when is_atom(system_name) or is_binary(system_name) do
    submit("record_readiness", {:record_readiness, system_name, ready?})
  end

  @doc """
  Records a fact retrieval for a response that found `result_count` facts
  (`get_enrichment_metrics/0`), without waiting.
  """
  def record_fact_retrieval(result_count) do
    submit("record_fact_retrieval", {:record_fact_retrieval, result_count})
  end

  @doc """
  Records a semantic retrieval for a response that found `result_count`
  facts (`get_enrichment_metrics/0`), without waiting.
  """
  def record_semantic_retrieval(result_count) do
    submit("record_semantic_retrieval", {:record_semantic_retrieval, result_count})
  end

  @doc """
  Records why a fact verification missed, one of
  `#{inspect(@verification_miss_reasons)}` (`get_verification_miss_reasons/0`),
  without waiting.
  """
  def record_verification_miss_reason(reason) do
    submit("record_verification_miss_reason", {:record_verification_miss_reason, reason})
  end

  @doc "Returns true if the Aggregator is ready to accept requests."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc """
  Writes every event sent before this call to Atlas, in one batch, and
  returns once it is written or, when Atlas cannot take it, counted as
  missed.
  """
  @spec flush() :: :ok
  def flush do
    GenServer.call(__MODULE__, :flush, 30_000)
  end

  @doc """
  Clears the gauges held in memory and starts a new node-wide telemetry
  window in Atlas, so what is counted from now on starts at zero; the ended
  window keeps its counts. Made by the aggregator after it wrote every event
  sent before it. Raises when the window cannot be started. Useful for
  testing. The semantic-fact windows are left as they are
  (`start_semantic_fact_window/1`).
  """
  def reset do
    case GenServer.call(__MODULE__, :reset, 15_000) do
      :ok -> :ok
      {:error, exception} -> raise exception
    end
  end

  # ============================================================================
  # Missed events
  # ============================================================================

  @doc false
  # Creates the counters of missed events, once per node, where they outlive
  # the aggregator: a handler counts a miss while the aggregator is down, and
  # the next aggregator writes it to Atlas. Beside each count: the misses not
  # logged yet, and when a miss of its sender and reason was last logged.
  def ensure_missed_counters do
    if :persistent_term.get(@missed_key, nil) == nil do
      :global.trans(
        {@missed_key, self()},
        fn ->
          if :persistent_term.get(@missed_key, nil) == nil do
            keys = for source <- sources(), reason <- @miss_reasons, do: {source, reason}
            index = keys |> Enum.with_index(1) |> Map.new()
            size = map_size(index)

            :persistent_term.put(
              @missed_key,
              {index, :counters.new(size, [:write_concurrency]), :counters.new(size, [:write_concurrency]),
               :atomics.new(size, signed: true)}
            )
          end
        end,
        [node()]
      )
    end

    :ok
  end

  defp sources, do: Brain.Telemetry.handler_ids() ++ @direct_sources

  # Counts `count` events from `source` as missed for `reason`, and logs them
  # unless a miss from `source` for `reason` was logged in the last
  # `@missed_log_interval_ms`: then they are logged with the next one, or by
  # the aggregator's report (`report_unlogged_misses/0`).
  defp missed(source, reason, count, detail) do
    :ok = ensure_missed_counters()
    {index, counters, unlogged, logged_at} = :persistent_term.get(@missed_key)
    i = Map.fetch!(index, {source, reason})
    :counters.add(counters, i, count)
    :counters.add(unlogged, i, count)

    now = System.system_time(:millisecond)
    last = :atomics.get(logged_at, i)

    if now - last >= @missed_log_interval_ms and :atomics.compare_exchange(logged_at, i, last, now) == :ok do
      case :counters.get(unlogged, i) do
        # The aggregator's report logged them meanwhile.
        0 ->
          :ok

        n ->
          :counters.sub(unlogged, i, n)

          Logger.error(
            "Brain.Metrics.Aggregator: missed #{n} event(s) from #{source} (#{reason}), the latest #{detail}; " <>
              "further misses from #{source} (#{reason}) in the next #{div(@missed_log_interval_ms, 1000)} seconds " <>
              "are counted and logged together. Counted in Brain.Metrics.Aggregator.get_missed_events/0"
          )
      end
    end

    :ok
  end

  # Logs, in one line, the misses counted and not logged yet, by sender and
  # reason.
  defp report_unlogged_misses do
    :ok = ensure_missed_counters()
    {index, _counters, unlogged, _logged_at} = :persistent_term.get(@missed_key)

    reported =
      index
      |> Enum.sort_by(&elem(&1, 1))
      |> Enum.flat_map(fn {key, i} ->
        case :counters.get(unlogged, i) do
          0 ->
            []

          n ->
            :counters.sub(unlogged, i, n)
            [{key, n}]
        end
      end)

    if reported != [] do
      total = reported |> Enum.map(&elem(&1, 1)) |> Enum.sum()

      Logger.error(
        "Brain.Metrics.Aggregator: #{total} more missed event(s) since the last report: " <>
          Enum.map_join(reported, ", ", fn {{source, reason}, n} -> "#{n} from #{source} (#{reason})" end) <>
          ". Counted in Brain.Metrics.Aggregator.get_missed_events/0"
      )
    end

    :ok
  end

  # Missed events not yet written to Atlas.
  defp pending_missed do
    :ok = ensure_missed_counters()
    {index, counters, _unlogged, _logged_at} = :persistent_term.get(@missed_key)

    Enum.reduce(index, %{}, fn {key, i}, acc ->
      case :counters.get(counters, i) do
        0 -> acc
        count -> Map.put(acc, key, count)
      end
    end)
  end

  # Takes the missed events written to Atlas off the counters; a miss counted
  # meanwhile stays counted.
  defp missed_written(pending) do
    {index, counters, _unlogged, _logged_at} = :persistent_term.get(@missed_key)
    Enum.each(pending, fn {key, count} -> :counters.sub(counters, Map.fetch!(index, key), count) end)
  end

  defp missed_write(pending) when pending == %{}, do: []

  defp missed_write(pending) do
    [
      %{
        scope: :node,
        family: @telemetry_family,
        values: Enum.map(pending, fn {{source, reason}, count} -> {"missed:#{source}:#{reason}", :sum, count} end)
      }
    ]
  end

  @doc """
  The events the aggregator was sent and did not count, in the current
  telemetry window: those written to Atlas and those still held in memory.

  Returns `%{total: count, by_source: %{source => %{reason => count}}}`,
  where a source is a telemetry handler id or one of the aggregator's own
  record functions, and a reason is one of `#{inspect(@miss_reasons)}`.
  Raises when Atlas cannot be read.
  """
  @spec get_missed_events(telemetry()) :: %{total: non_neg_integer(), by_source: %{String.t() => %{atom() => pos_integer()}}}
  def get_missed_events(telemetry \\ read_telemetry()) do
    persisted =
      for {"missed:" <> missed, count} <- telemetry.values do
        [source, reason] = String.split(missed, ":", parts: 2)
        {{source, decode_miss_reason(reason)}, count}
      end

    by_source =
      Enum.reduce(persisted ++ Map.to_list(pending_missed()), %{}, fn {{source, reason}, count}, acc ->
        Map.update(acc, source, %{reason => count}, &Map.update(&1, reason, count, fn held -> held + count end))
      end)

    %{
      total: by_source |> Map.values() |> Enum.flat_map(&Map.values/1) |> Enum.sum(),
      by_source: by_source
    }
  end

  for reason <- @miss_reasons do
    defp decode_miss_reason(unquote(Atom.to_string(reason))), do: unquote(reason)
  end

  defp decode_miss_reason(reason), do: raise_unknown_value("missed:<source>:#{reason}")

  defp describe({:telemetry, kind, _measurements, _metadata, _at}), do: "a telemetry event #{inspect(kind)}"
  defp describe(message), do: "#{inspect(elem(message, 0))}"

  # ============================================================================
  # Reading what was counted
  # ============================================================================

  @doc """
  Reads the current node-wide telemetry window from Atlas, once: every
  reader below answers from it when given it, so a caller with several
  questions (the dashboard) reads Atlas once. Each reader reads it itself
  when not given it. Raises when Atlas cannot be read.
  """
  @spec read_telemetry() :: telemetry()
  def read_telemetry do
    case Atlas.Metrics.windows(:node, @telemetry_family, 1) do
      [%{window: window, values: values, records: records}] -> %{window: window, values: values, records: records}
      [] -> %{window: nil, values: %{}, records: %{}}
    end
  end

  @doc """
  The current node-wide telemetry window, `%{started_at: DateTime.t(),
  ended_at: nil}`: what the Atlas-kept metrics count since; `nil` before
  anything was counted. Raises when Atlas cannot be read.
  """
  @spec get_telemetry_window(telemetry()) :: %{started_at: DateTime.t(), ended_at: nil} | nil
  def get_telemetry_window(telemetry \\ read_telemetry()) do
    window = telemetry.window
    window && %{started_at: window.started_at, ended_at: window.ended_at}
  end

  @doc """
  The metrics of the node: each duration metric in its current telemetry
  window (`get_metric/1`), and under `:errors` (`get_errors/0`), `:training`
  (`get_training_stats/0`), `:model_loads` (`get_model_load_stats/0`) and
  `:readiness` the rest. Raises when Atlas cannot be read.
  """
  def get_metrics(telemetry \\ read_telemetry()) do
    telemetry.values
    |> duration_metrics()
    |> Map.merge(%{
      errors: errors(telemetry.values),
      training: training(telemetry.records),
      model_loads: gauges(:model_load),
      readiness: gauges(:readiness)
    })
  end

  @doc """
  One duration metric in the current telemetry window, `nil` for a name that
  is neither one of the known duration metrics nor counted. Read from Atlas:

  - `:count`, `:error_count` - spans counted, and those that raised
  - `:avg_ms`, `:min_ms`, `:max_ms` - their mean, shortest and longest;
    `nil` before the first
  - `:histogram` - `[%{le_ms: bound | :inf, count: count}]`, the spans at
    most `bound` milliseconds long and longer than the bucket before
  - `:p95_ms` - the upper bound of the histogram bucket the 95th percentile
    falls in (the longest span when that is the `:inf` bucket); an upper
    bound, not the percentile itself; `nil` before the first span

  Read from memory, about this process now: `:rate_per_minute` and
  `:sample_count`, the spans in the last five minutes, and `:last_updated`,
  the monotonic time in milliseconds of the last one (`nil` before one).
  """
  def get_metric(name, telemetry \\ read_telemetry()) do
    Map.get(duration_metrics(telemetry.values), name)
  end

  @doc """
  The errors counted in the current telemetry window, by error type:
  `%{type => %{count: count, last_seen_at: DateTime.t()}}`. A span that
  raised counts under its kind (`"error"`, `"exit"`, `"throw"`); a failed
  training run under `"training_failed.<model>"`.
  """
  def get_errors(telemetry \\ read_telemetry()) do
    errors(telemetry.values)
  end

  @doc """
  The last training run of each model in the current telemetry window, by
  model name: `%{model => %{status:, started_at:, completed_at:,
  sequence_count:, duration_ms:, tag_count:, feature_count:, success:,
  reason:, recorded_at:}}`. `started_at` is `nil` when the run started
  before the aggregator last started.
  """
  def get_training_stats(telemetry \\ read_telemetry()) do
    training(telemetry.records)
  end

  @doc """
  The last load of each model this node made, in memory, by model name:
  `%{model => %{loaded_at:, duration_ms:, success:, last_updated:}}`.
  """
  def get_model_load_stats, do: gauges(:model_load)

  @doc """
  External service metrics in the current telemetry window:

  - `:dispatch`, `:enrichment` - their duration metrics (`get_metric/1`)
  - `:by_service` - per service name: `:total_dispatches`, `:success_count`,
    `:error_count`, `:success_rate`, and from memory the `:last_dispatch` and
    `:health_status` this process saw
  - `:cache` - `:hits`, `:misses` and `:hit_rate`
  """
  def get_service_metrics(telemetry \\ read_telemetry()) do
    values = telemetry.values
    durations = duration_metrics(values)

    by_service =
      values
      |> Enum.flat_map(fn
        {"service:" <> counted, count} ->
          {service, field} = split_last(counted)
          [{service, field, count}]

        _other ->
          []
      end)
      |> Enum.group_by(&elem(&1, 0), fn {_service, field, count} -> {field, count} end)
      |> Map.new(fn {service, fields} ->
        fields = Map.new(fields)
        count = Map.get(fields, "count", 0)
        success = Map.get(fields, "success", 0)

        {service,
         %{
           total_dispatches: count,
           success_count: success,
           error_count: Map.get(fields, "error", 0),
           success_rate: if(count > 0, do: Float.round(success / count * 100, 1), else: 0.0),
           last_dispatch: gauge({:service_dispatch_last, service}),
           health_status: gauge({:service_health, service})
         }}
      end)

    hits = Map.get(values, "service_cache:hit", 0)
    misses = Map.get(values, "service_cache:miss", 0)

    %{
      dispatch: Map.fetch!(durations, :service_dispatch),
      enrichment: Map.fetch!(durations, :service_enrichment),
      by_service: by_service,
      cache: %{
        hits: hits,
        misses: misses,
        hit_rate: if(hits + misses > 0, do: Float.round(hits / (hits + misses) * 100, 1), else: 0.0)
      }
    }
  end

  @doc """
  Fact verification in the current telemetry window:

  - `:total_verifications`, `:contradiction_count`
  - `:by_status` - `%{status => count}` for #{inspect(@verification_statuses)}
  - `:last_verifications` - from memory, the last verification this process
    saw per status
  """
  def get_epistemic_metrics(telemetry \\ read_telemetry()) do
    values = telemetry.values

    %{
      total_verifications: Map.get(values, "duration:fact_verification:count", 0),
      contradiction_count: Map.get(values, "fact_verification:contradictions", 0),
      by_status: Map.new(@verification_statuses, &{&1, Map.get(values, "fact_verification:status:#{&1}", 0)}),
      last_verifications:
        [:verified, :contradicted, :uncertain]
        |> Enum.flat_map(fn status ->
          case gauge({:fact_verification_last, Atom.to_string(status)}) do
            nil -> []
            last -> [{status, last}]
          end
        end)
        |> Map.new()
    }
  end

  @doc """
  Response enrichment in the current telemetry window: fact and semantic
  retrieval hit rates, and facts retrieved per response.
  """
  def get_enrichment_metrics(telemetry \\ read_telemetry()) do
    values = telemetry.values
    fact_total = Map.get(values, "fact_retrieval:total", 0)
    fact_hits = Map.get(values, "fact_retrieval:hits", 0)
    fact_results = Map.get(values, "fact_retrieval:results", 0)
    semantic_total = Map.get(values, "semantic_retrieval:total", 0)
    semantic_hits = Map.get(values, "semantic_retrieval:hits", 0)

    %{
      fact_hit_rate: percent(fact_hits, fact_total),
      semantic_hit_rate: percent(semantic_hits, semantic_total),
      avg_facts_per_response: if(fact_total > 0, do: Float.round(fact_results / fact_total, 1), else: 0.0),
      fact_queries: fact_total,
      semantic_queries: semantic_total
    }
  end

  @doc "Why fact verifications missed, in the current telemetry window: counts, percentages and total.\n"
  def get_verification_miss_reasons(telemetry \\ read_telemetry()) do
    counts = Map.new(@verification_miss_reasons, &{&1, Map.get(telemetry.values, "verification_miss:#{&1}", 0)})
    total = counts |> Map.values() |> Enum.sum()

    %{counts: counts, percentages: Map.new(counts, fn {reason, count} -> {reason, percent(count, total)} end), total: total}
  end

  @doc """
  Counts of the node's activity in the current telemetry window, each by
  the name its events carry:

  - `:kg_signals` - `%{event => %{count:, items:, duration:}}`, `:items`
    summing the events' `:count` measurement and `:duration` the event's
    duration metric (`get_metric/1`) where it carries one, else `nil`
  - `:learning` - learning events by event
  - `:consistency` - `%{total:, by_severity: %{severity => count}}`
  - `:racing_early_exits` - by analyzer
  - `:code_languages` - code files processed by language
  """
  def get_activity_metrics(telemetry \\ read_telemetry()) do
    values = telemetry.values
    durations = duration_metrics(values)

    kg_signal_fields =
      for {"kg_signal:" <> counted, count} <- values, reduce: %{} do
        acc ->
          {event, field} = split_last(counted)
          Map.update(acc, event, %{field => count}, &Map.put(&1, field, count))
      end

    kg_signals =
      Map.new(kg_signal_fields, fn {event, fields} ->
        {event,
         %{
           count: Map.get(fields, "count", 0),
           items: Map.get(fields, "items", 0),
           duration: Map.get(durations, "kg_signal.#{event}")
         }}
      end)

    %{
      kg_signals: kg_signals,
      learning: prefixed(values, "learning:"),
      consistency: %{
        total: Map.get(values, "consistency:total", 0),
        by_severity: values |> prefixed("consistency:") |> Map.delete("total")
      },
      racing_early_exits: prefixed(values, "racing_early_exit:"),
      code_languages: prefixed(values, "code_language:")
    }
  end

  defp prefixed(values, prefix) do
    for {name, count} <- values, String.starts_with?(name, prefix), into: %{} do
      {String.replace_prefix(name, prefix, ""), count}
    end
  end

  defp percent(_part, 0), do: 0.0
  defp percent(part, total), do: Float.round(part / total * 100, 1)

  # "a:b:c" -> {"a:b", "c"}
  defp split_last(name) do
    [field | rest] = name |> String.split(":") |> Enum.reverse()

    if rest == [] do
      raise_unknown_value(name)
    end

    {rest |> Enum.reverse() |> Enum.join(":"), field}
  end

  defp raise_unknown_value(name) do
    raise ArgumentError,
          "Brain.Metrics.Aggregator: Atlas holds a metric #{inspect(name)} that is not one the aggregator writes"
  end

  defp duration_metrics(values) do
    counted =
      for {"duration:" <> counted, value} <- values, reduce: %{} do
        acc ->
          [metric, field] = String.split(counted, ":", parts: 2)
          Map.update(acc, metric, %{field => value}, &Map.put(&1, field, value))
      end

    known = Map.new(@duration_metrics, &{Atom.to_string(&1), &1})

    (Map.keys(known) ++ Map.keys(counted))
    |> Enum.uniq()
    |> Map.new(fn metric -> {Map.get(known, metric, metric), duration_stats(metric, Map.get(counted, metric, %{}))} end)
  end

  defp duration_stats(metric, fields) do
    count = Map.get(fields, "count", 0)
    max_ms = us_to_ms(Map.get(fields, "max_us"))
    histogram = for bound <- @duration_buckets_ms ++ [:inf], do: %{le_ms: bound, count: Map.get(fields, "le_ms:#{bound}", 0)}
    {rate_per_minute, sample_count} = live_rate(metric)

    %{
      count: count,
      error_count: Map.get(fields, "error_count", 0),
      avg_ms: if(count > 0, do: Float.round(Map.get(fields, "sum_us", 0) / count / 1000, 3)),
      min_ms: us_to_ms(Map.get(fields, "min_us")),
      max_ms: max_ms,
      p95_ms: p95(metric, histogram, count, max_ms),
      histogram: histogram,
      rate_per_minute: rate_per_minute,
      sample_count: sample_count,
      last_updated: gauge({:last_seen, metric})
    }
  end

  defp us_to_ms(nil), do: nil
  defp us_to_ms(us), do: Float.round(us / 1000, 3)

  defp p95(_metric, _histogram, 0, _max_ms), do: nil

  defp p95(metric, histogram, count, max_ms) do
    target = ceil(count * 0.95)

    histogram
    |> Enum.reduce_while(0, fn %{le_ms: bound, count: in_bucket}, below ->
      if below + in_bucket >= target, do: {:halt, {:found, bound}}, else: {:cont, below + in_bucket}
    end)
    |> case do
      {:found, :inf} -> max_ms
      {:found, bound} -> bound * 1.0
      # Every duration is counted in one bucket, in the same write as its count.
      held -> raise "Brain.Metrics.Aggregator: #{metric} counts #{count} durations, its histogram #{held}"
    end
  end

  defp errors(values) do
    for {"error:" <> counted, value} <- values, reduce: %{} do
      acc ->
        {type, field} = split_last(counted)

        entry =
          case field do
            "count" -> %{count: value}
            "last_at_ms" -> %{last_seen_at: DateTime.from_unix!(value, :millisecond)}
            _other -> raise_unknown_value("error:" <> counted)
          end

        Map.update(acc, type, entry, &Map.merge(&1, entry))
    end
  end

  defp training(records) do
    for {"training:" <> model, %{record: record, recorded_at: recorded_at}} <- records, into: %{} do
      {model,
       %{
         status: Map.fetch!(@training_statuses, Map.fetch!(record, "status")),
         started_at: parse_time(Map.get(record, "started_at")),
         completed_at: parse_time(Map.get(record, "completed_at")),
         sequence_count: Map.get(record, "sequence_count"),
         duration_ms: Map.get(record, "duration_ms"),
         tag_count: Map.get(record, "tag_count"),
         feature_count: Map.get(record, "feature_count"),
         success: Map.get(record, "success"),
         reason: Map.get(record, "reason"),
         recorded_at: recorded_at
       }}
    end
  end

  defp parse_time(nil), do: nil

  defp parse_time(iso) do
    {:ok, time, 0} = DateTime.from_iso8601(iso)
    time
  end

  # The gauges a running aggregator holds; a stopped one holds none.
  defp gauges(kind) do
    if :ets.whereis(@gauges_table) == :undefined do
      %{}
    else
      @gauges_table |> :ets.match({{kind, :"$1"}, :"$2"}) |> Map.new(fn [name, data] -> {name, data} end)
    end
  end

  defp gauge(key) do
    if :ets.whereis(@gauges_table) == :undefined do
      nil
    else
      case :ets.lookup(@gauges_table, key) do
        [{^key, data}] -> data
        [] -> nil
      end
    end
  end

  defp live_rate(metric) do
    now = System.monotonic_time(:millisecond)
    recent = (gauge({:rate, metric}) || []) |> Enum.filter(&(&1 > now - @rate_window_ms))
    {Float.round(length(recent) / (@rate_window_ms / 60_000), 2), length(recent)}
  end

  # ============================================================================
  # Semantic facts
  # ============================================================================

  @doc """
  The semantic-fact metrics of `world_id` in its current measurement window:
  what the `[:chat_bot, :memory, :semantic_fact, :retrieved | :used]` events
  (`Brain.Memory.RecurringPattern`) counted for that world since the window
  started.

  Kept in Atlas (`Atlas.Metrics`, the `"semantic_facts"` family), so they
  survive a restart and can be compared across a retrain:
  `start_semantic_fact_window/1` ends the window and starts a new one, and
  `get_previous_semantic_fact_metrics/1` reads the window it ended. Read from
  Atlas on every call; there is no copy in memory to diverge. Raises when
  Atlas cannot be read.

  Returns a map with:
  - `:window` - `%{started_at: DateTime.t(), ended_at: DateTime.t() | nil}`,
    the window counted; `nil` for a world nothing was counted for and no
    window was started for
  - `:retrievals` - turns that retrieved the world's semantic facts
  - `:facts_retrieved` - semantic facts retrieved across those turns
  - `:patterns_found` - those turns with a recurring pattern among the facts
  - `:pattern_found_rate` - `:patterns_found` as a percentage of
    `:retrievals`; `nil` before the first retrieval
  - `:used` - turns whose response was built from the recurring pattern, by
    who built it: `%{synthesizer: count, realization_packet: count}`
  - `:used_total` - those turns, by either
  - `:retrieved_similarity` - how similar the retrieved facts were to their
    turns, as `%{from: float, to: float, count: count}` buckets a tenth wide,
    ascending, the buckets that counted any only; the top bucket, from 0.9,
    holds a similarity of 1.0
  - `:used_similarity` - the same for the patterns responses were built from
  """
  @spec get_semantic_fact_metrics(String.t()) :: map()
  def get_semantic_fact_metrics(world_id) when is_binary(world_id) do
    get_semantic_fact_windows(world_id).current
  end

  @doc """
  The semantic-fact metrics of `world_id` in the window before its current
  one, the window `start_semantic_fact_window/1` last ended, in the shape
  `get_semantic_fact_metrics/1` returns; `nil` when no window was ended.
  """
  @spec get_previous_semantic_fact_metrics(String.t()) :: map() | nil
  def get_previous_semantic_fact_metrics(world_id) when is_binary(world_id) do
    get_semantic_fact_windows(world_id).previous
  end

  @doc """
  `%{current: get_semantic_fact_metrics(world_id), previous:
  get_previous_semantic_fact_metrics(world_id)}`, read from Atlas once.
  """
  @spec get_semantic_fact_windows(String.t()) :: %{current: map(), previous: map() | nil}
  def get_semantic_fact_windows(world_id) when is_binary(world_id) do
    case Atlas.Metrics.windows({:world, world_id}, @semantic_fact_family, 2) do
      [current, previous] -> %{current: semantic_fact_metrics(current), previous: semantic_fact_metrics(previous)}
      [current] -> %{current: semantic_fact_metrics(current), previous: nil}
      [] -> %{current: semantic_fact_metrics(nil, %{}), previous: nil}
    end
  end

  @doc """
  Ends the current semantic-fact measurement window of `world_id`, keeping its
  counts, and starts a new one; returns the new window's
  `%{started_at: DateTime.t(), ended_at: nil}`.

  Made by the aggregator, after it wrote every event it was sent before this
  call, so those events count in the window they happened in. Raises when
  the window cannot be started.
  """
  @spec start_semantic_fact_window(String.t()) :: %{started_at: DateTime.t(), ended_at: nil}
  def start_semantic_fact_window(world_id) when is_binary(world_id) do
    case GenServer.call(__MODULE__, {:start_semantic_fact_window, world_id}, 15_000) do
      {:ok, window} -> window
      {:error, exception} -> raise exception
    end
  end

  defp semantic_fact_metrics(%{window: window, values: values}) do
    semantic_fact_metrics(window, Map.new(values, fn {counter, count} -> {decode_semantic_fact_counter(counter), count} end))
  end

  defp semantic_fact_metrics(window, counters) do
    retrievals = Map.get(counters, :retrievals, 0)
    patterns_found = Map.get(counters, :patterns_found, 0)
    used = Map.new(@semantic_fact_users, &{&1, Map.get(counters, {:used, &1}, 0)})

    %{
      window: window && %{started_at: window.started_at, ended_at: window.ended_at},
      retrievals: retrievals,
      facts_retrieved: Map.get(counters, :facts_retrieved, 0),
      patterns_found: patterns_found,
      pattern_found_rate: if(retrievals > 0, do: Float.round(patterns_found / retrievals * 100, 1)),
      used: used,
      used_total: used |> Map.values() |> Enum.sum(),
      retrieved_similarity: similarity_buckets(counters, :retrieved_similarity),
      used_similarity: similarity_buckets(counters, :used_similarity)
    }
  end

  defp similarity_buckets(counters, distribution) do
    for {{^distribution, bucket}, count} <- Enum.sort(counters) do
      %{from: bucket / 10, to: (bucket + 1) / 10, count: count}
    end
  end

  # The name each semantic-fact counter is stored under in Atlas, and back.
  # `"used:synthesizer"` counts the uses by the synthesizer,
  # `"retrieved_similarity:8"` the retrieved facts in bucket 8.
  @semantic_fact_scalar_counters [:retrievals, :facts_retrieved, :patterns_found]
  @semantic_fact_distributions [:retrieved_similarity, :used_similarity]

  defp encode_semantic_fact_counter(counter) when counter in @semantic_fact_scalar_counters,
    do: Atom.to_string(counter)

  defp encode_semantic_fact_counter({:used, used_by}) when used_by in @semantic_fact_users,
    do: "used:#{used_by}"

  defp encode_semantic_fact_counter({distribution, bucket})
       when distribution in @semantic_fact_distributions and is_integer(bucket),
       do: "#{distribution}:#{bucket}"

  for counter <- @semantic_fact_scalar_counters do
    defp decode_semantic_fact_counter(unquote(Atom.to_string(counter))), do: unquote(counter)
  end

  for used_by <- @semantic_fact_users do
    defp decode_semantic_fact_counter(unquote("used:#{used_by}")), do: {:used, unquote(used_by)}
  end

  for distribution <- @semantic_fact_distributions do
    defp decode_semantic_fact_counter(unquote("#{distribution}:") <> bucket) do
      case Integer.parse(bucket) do
        {bucket, ""} -> {unquote(distribution), bucket}
        _ -> raise_unknown_semantic_fact_counter(unquote("#{distribution}:") <> bucket)
      end
    end
  end

  defp decode_semantic_fact_counter(counter), do: raise_unknown_semantic_fact_counter(counter)

  defp raise_unknown_semantic_fact_counter(counter) do
    raise ArgumentError,
          "Brain.Metrics.Aggregator: Atlas holds a semantic fact counter #{inspect(counter)} " <>
            "that is not one the aggregator counts"
  end

  # The write of one semantic-fact event to the world's current window.
  # Several counters of an event can be the same one (two retrieved facts in
  # one similarity bucket), so they are summed first; an amount of zero (a
  # turn that retrieved no facts) adds nothing.
  defp semantic_fact_write(world_id, counters) do
    values =
      counters
      |> Enum.reduce(%{}, fn {counter, amount}, acc ->
        Map.update(acc, encode_semantic_fact_counter(counter), amount, &(&1 + amount))
      end)
      |> Enum.reject(fn {_counter, amount} -> amount == 0 end)
      |> Enum.map(fn {counter, amount} -> {counter, :sum, amount} end)

    %{scope: {:world, world_id}, family: @semantic_fact_family, values: values}
  end

  # The tenth of the similarity range a similarity falls in: 0 for [0.0, 0.1),
  # 9 for [0.9, 1.0]. A cosine similarity can be negative, down to -10 for
  # [-1.0, -0.9).
  defp similarity_bucket(similarity) when is_number(similarity) do
    (similarity * 10) |> :math.floor() |> trunc() |> min(9)
  end

  defp similarity_bucket(other), do: malformed!("a similarity must be a number, got #{inspect(other)}")

  # ============================================================================
  # GenServer
  # ============================================================================

  @impl true
  def init(opts) do
    drain_ms = Keyword.get(opts, :drain_ms, @drain_ms)

    unless is_integer(drain_ms) and drain_ms >= 0 do
      raise ArgumentError, "Brain.Metrics.Aggregator: :drain_ms must be a non-negative integer, got #{inspect(drain_ms)}"
    end

    if drain_ms + @final_write_ms >= @shutdown_ms do
      raise ArgumentError,
            "Brain.Metrics.Aggregator: :drain_ms (#{drain_ms}) and the final write (#{@final_write_ms}) " <>
              "must fit in the #{@shutdown_ms} ms its supervisor gives it to stop"
    end

    # Stopped by its supervisor, it still runs terminate/2, and counts what it
    # was sent and had not counted.
    Process.flag(:trap_exit, true)
    :ok = ensure_missed_counters()
    :ets.new(@gauges_table, [:named_table, :public, :set, read_concurrency: true])
    schedule(:flush, @batch_interval_ms)
    schedule(:report_missed, @missed_log_interval_ms)

    Logger.info("Metrics.Aggregator started")

    {:ok, %{batch: %{}, batch_sources: %{}, batch_events: 0, drain_ms: drain_ms}, {:continue, :flush}}
  end

  @impl true
  def handle_continue(:flush, state) do
    {:noreply, flush_batch(state)}
  end

  @impl true
  def handle_info({:metric_event, source, message}, state) do
    state = count(state, source, message)

    if state.batch_events >= @batch_max_events do
      {:noreply, flush_batch(state)}
    else
      {:noreply, state}
    end
  end

  def handle_info(:flush, state) do
    schedule(:flush, @batch_interval_ms)
    {:noreply, flush_batch(state)}
  end

  def handle_info(:report_missed, state) do
    report_unlogged_misses()
    schedule(:report_missed, @missed_log_interval_ms)
    {:noreply, state}
  end

  def handle_info({:EXIT, _pid, reason}, state) do
    {:stop, reason, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  def handle_call(:flush, _from, state) do
    {:reply, :ok, flush_batch(state)}
  end

  # A window Atlas cannot start is raised in the caller; the aggregator goes on.
  def handle_call({:start_semantic_fact_window, world_id}, _from, state) do
    state = flush_batch(state)

    reply =
      with {:ok, window} <- atlas_start_window({:world, world_id}, @semantic_fact_family) do
        {:ok, %{started_at: window.started_at, ended_at: window.ended_at}}
      end

    {:reply, reply, state}
  end

  def handle_call(:reset, _from, state) do
    state = flush_batch(state)
    :ets.delete_all_objects(@gauges_table)

    reply =
      with {:ok, _window} <- atlas_start_window(:node, @telemetry_family) do
        :ok
      end

    {:reply, reply, state}
  end

  @impl true
  def terminate(_reason, state) do
    # Senders that find no aggregator from here on count their events as
    # missed themselves; what already waits in the mailbox is counted here,
    # within the time its supervisor gives it.
    if Process.whereis(__MODULE__) == self(), do: Process.unregister(__MODULE__)
    state = drain_mailbox(state, System.monotonic_time(:millisecond) + state.drain_ms)
    final_flush(state)
    report_unlogged_misses()
  end

  # Counts the events waiting in the mailbox into the batch until `deadline`,
  # and those still waiting after it as missed.
  defp drain_mailbox(state, deadline) do
    receive do
      {:metric_event, source, message} ->
        if System.monotonic_time(:millisecond) < deadline do
          drain_mailbox(count(state, source, message), deadline)
        else
          :ok = missed(source, :aggregator_stopped, 1, describe(message))
          drain_mailbox(state, deadline)
        end
    after
      0 -> state
    end
  end

  # The last batch, written in a process of its own so that a write Atlas
  # does not finish within `@final_write_ms` is stopped there and counted as
  # missed, not left to its supervisor's kill.
  defp final_flush(state) do
    pending = pending_missed()
    writes = batch_writes(state.batch) ++ missed_write(pending)

    if writes != [] do
      task = Task.async(fn -> atlas_write(writes) end)

      result =
        case Task.yield(task, @final_write_ms) || Task.shutdown(task, :brutal_kill) do
          {:ok, result} -> result
          {:exit, reason} -> {:error, "the write exited: #{inspect(reason)}"}
          nil -> {:error, "the write did not finish within #{@final_write_ms} ms"}
        end

      batch_written(state, pending, result)
    end

    :ok
  end

  defp schedule(message, after_ms) do
    Process.send_after(self(), message, after_ms)
  end

  # ============================================================================
  # Counting one event
  # ============================================================================

  # Adds what the event counts to the batch; a malformed event is missed.
  defp count(state, source, message) do
    case interpret(message) do
      [] -> state
      writes -> add_to_batch(state, source, writes)
    end
  rescue
    exception in MalformedEvent ->
      :ok = missed(source, :malformed_event, 1, "#{describe(message)}: #{exception.message}")
      state
  end

  defp add_to_batch(state, source, writes) do
    batch =
      Enum.reduce(writes, state.batch, fn %{scope: scope, family: family} = write, batch ->
        held = Map.get(batch, {scope, family}, %{values: %{}, records: %{}})

        values =
          Enum.reduce(Map.get(write, :values, []), held.values, fn {name, aggregate, value}, values ->
            Map.update(values, name, {aggregate, value}, fn {held_aggregate, held_value} ->
              {held_aggregate, combine(name, held_aggregate, held_value, aggregate, value)}
            end)
          end)

        records = Enum.reduce(Map.get(write, :records, []), held.records, fn {name, record}, records -> Map.put(records, name, record) end)
        Map.put(batch, {scope, family}, %{values: values, records: records})
      end)

    %{
      state
      | batch: batch,
        batch_sources: Map.update(state.batch_sources, source, 1, &(&1 + 1)),
        batch_events: state.batch_events + 1
    }
  end

  # How Atlas combines a value with the one its window holds.
  defp combine(_name, :sum, held, :sum, value), do: held + value
  defp combine(_name, :min, held, :min, value), do: min(held, value)
  defp combine(_name, :max, held, :max, value), do: max(held, value)

  defp combine(name, held_aggregate, _held, aggregate, _value) do
    raise "Brain.Metrics.Aggregator: #{name} is counted as #{held_aggregate} and as #{aggregate}"
  end

  defp batch_writes(batch) do
    Enum.map(batch, fn {{scope, family}, %{values: values, records: records}} ->
      %{
        scope: scope,
        family: family,
        values: Enum.map(values, fn {name, {aggregate, value}} -> {name, aggregate, value} end),
        records: Map.to_list(records)
      }
    end)
  end

  # Writes the batch and the missed counts held in memory to Atlas, in one
  # transaction.
  defp flush_batch(state) do
    pending = pending_missed()
    writes = batch_writes(state.batch) ++ missed_write(pending)

    if writes == [] do
      state
    else
      batch_written(state, pending, atlas_write(writes))
    end
  end

  # What a write of the batch and the `pending` missed counts did: written,
  # the missed counts come off the counters; not written, every event in the
  # batch is missed, and the missed counts stay held for the next write.
  defp batch_written(state, pending, :ok) do
    missed_written(pending)
    %{state | batch: %{}, batch_sources: %{}, batch_events: 0}
  end

  defp batch_written(state, pending, {:error, error}) do
    message = if is_exception(error), do: Exception.message(error), else: error

    Enum.each(state.batch_sources, fn {source, events} ->
      :ok = missed(source, :atlas_write_failed, events, "in a batch of #{state.batch_events} Atlas did not write: #{message}")
    end)

    if pending != %{} do
      Logger.error(
        "Brain.Metrics.Aggregator: could not write #{pending |> Map.values() |> Enum.sum()} missed event(s) " <>
          "to Atlas; they stay counted here and are written on the next try: #{message}"
      )
    end

    %{state | batch: %{}, batch_sources: %{}, batch_events: 0}
  end

  # Atlas refusing a write (Postgrex.Error), being unreachable
  # (DBConnection.ConnectionError) or giving this process no connection
  # (DBConnection.OwnershipError) is returned; anything else raises.
  defp atlas_write(writes) do
    Atlas.Metrics.write!(writes)
  rescue
    exception in [DBConnection.ConnectionError, DBConnection.OwnershipError, Postgrex.Error] -> {:error, exception}
  end

  defp atlas_start_window(scope, family) do
    {:ok, Atlas.Metrics.start_window!(scope, family)}
  rescue
    exception in [DBConnection.ConnectionError, DBConnection.OwnershipError, Postgrex.Error] -> {:error, exception}
  end

  # What one event adds: the gauges it sets in memory are set here, and the
  # writes to Atlas are returned.
  defp interpret({:telemetry, {:span_stop, metric}, measurements, _metadata, _at}) do
    [node_write(duration(metric, native_us!(measurements)))]
  end

  defp interpret({:telemetry, {:span_exception, metric}, measurements, metadata, at}) do
    [node_write(duration(metric, native_us!(measurements), :error) ++ error(fetch!(metadata, :kind), at))]
  end

  defp interpret({:telemetry, :semantic_facts_retrieved, _measurements, metadata, _at}) do
    world_id = world!(metadata, :world_id)
    similarities = fetch!(metadata, :similarities)

    unless is_list(similarities), do: malformed!("similarities must be a list, got #{inspect(similarities)}")

    [
      semantic_fact_write(
        world_id,
        [
          {:retrievals, 1},
          {:facts_retrieved, length(similarities)},
          {:patterns_found, if(fetch!(metadata, :recurring_pattern_id) != nil, do: 1, else: 0)}
        ] ++ Enum.map(similarities, &{{:retrieved_similarity, similarity_bucket(&1)}, 1})
      )
    ]
  end

  defp interpret({:telemetry, :semantic_fact_used, measurements, metadata, _at}) do
    world_id = world!(metadata, :world_id)
    used_by = fetch!(metadata, :used_by)

    unless used_by in @semantic_fact_users, do: malformed!("no semantic fact user #{inspect(used_by)}")

    [
      semantic_fact_write(world_id, [
        {{:used, used_by}, 1},
        {{:used_similarity, similarity_bucket(fetch!(measurements, :similarity))}, 1}
      ])
    ]
  end

  defp interpret({:telemetry, :training_start, measurements, metadata, at}) do
    model = segment!(fetch!(metadata, :model))
    sequence_count = number_or_nil!(measurements, :sequence_count)
    :ets.insert(@gauges_table, {{:training_started, model}, %{started_at: at, sequence_count: sequence_count}})

    [
      node_write([], [
        {"training:#{model}", %{"status" => "in_progress", "started_at" => DateTime.to_iso8601(at), "sequence_count" => sequence_count}}
      ])
    ]
  end

  defp interpret({:telemetry, :training_stop, measurements, metadata, at}) do
    model = segment!(fetch!(metadata, :model))
    started = take_training_start(model)

    record = %{
      "status" => "completed",
      "started_at" => started && DateTime.to_iso8601(started.started_at),
      "completed_at" => DateTime.to_iso8601(at),
      # The stop's count when it reports one, else the one its start reported.
      "sequence_count" => number_or_nil!(measurements, :sequence_count) || (started && started.sequence_count),
      "duration_ms" => number_or_nil!(measurements, :duration_ms),
      "tag_count" => number_or_nil!(measurements, :tag_count),
      "feature_count" => number_or_nil!(measurements, :feature_count),
      "success" => true
    }

    [node_write([], [{"training:#{model}", record}])]
  end

  defp interpret({:telemetry, :training_exception, measurements, metadata, at}) do
    model = segment!(fetch!(metadata, :model))
    started = take_training_start(model)

    record = %{
      "status" => "failed",
      "started_at" => started && DateTime.to_iso8601(started.started_at),
      "completed_at" => DateTime.to_iso8601(at),
      "sequence_count" => number_or_nil!(measurements, :sequence_count) || (started && started.sequence_count),
      "duration_ms" => number_or_nil!(measurements, :duration_ms),
      "reason" => inspect(Map.get(metadata, :reason)),
      "success" => false
    }

    [node_write(error({:training_failed, model}, at), [{"training:#{model}", record}])]
  end

  defp interpret({:telemetry, :model_load, measurements, metadata, at}) do
    success = fetch!(metadata, :success)

    unless is_boolean(success), do: malformed!("a model load's success must be a boolean, got #{inspect(success)}")

    :ets.insert(
      @gauges_table,
      {{:model_load, segment!(fetch!(metadata, :model))},
       %{
         loaded_at: at,
         duration_ms: number_or_nil!(measurements, :duration_ms),
         success: success,
         last_updated: System.monotonic_time(:millisecond)
       }}
    )

    []
  end

  # A destroyed world is counted for the node only: what the world held in
  # Atlas may be purged with it, and a count written after would bring it back.
  defp interpret({:telemetry, {:learning, :world_destroyed}, _measurements, metadata, _at}) do
    world_id = world!(metadata, :world_id)
    :ets.insert(@gauges_table, {{:learning_last, :world_destroyed}, %{timestamp: System.monotonic_time(:millisecond), world_id: world_id}})
    [node_write([{"learning:world_destroyed", :sum, 1}])]
  end

  defp interpret({:telemetry, {:learning, event}, _measurements, metadata, _at}) do
    world_id = world!(metadata, :world_id)
    :ets.insert(@gauges_table, {{:learning_last, event}, %{timestamp: System.monotonic_time(:millisecond), world_id: world_id}})
    name = "learning:#{event}"

    [
      node_write([{name, :sum, 1}]),
      %{scope: {:world, world_id}, family: @telemetry_family, values: [{name, :sum, 1}]}
    ]
  end

  defp interpret({:telemetry, :fact_verification, measurements, metadata, _at}) do
    status = segment!(fetch!(metadata, :status))
    duration_ms = fetch!(measurements, :duration)

    :ets.insert(
      @gauges_table,
      {{:fact_verification_last, status},
       %{
         subject: Map.get(metadata, :subject),
         beliefs_count: Map.get(metadata, :beliefs_count),
         duration_ms: duration_ms,
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    contradictions = if status == "contradicted", do: [{"fact_verification:contradictions", :sum, 1}], else: []

    [
      node_write(
        duration(:fact_verification, ms_to_us!(duration_ms)) ++
          [{"fact_verification:status:#{status}", :sum, 1}] ++ contradictions
      )
    ]
  end

  defp interpret({:telemetry, :racing_early_exit, measurements, metadata, _at}) do
    analyzer = segment!(fetch!(metadata, :analyzer))

    :ets.insert(
      @gauges_table,
      {{:racing_early_exit_last, analyzer},
       %{
         timestamp: System.monotonic_time(:millisecond),
         confidence: Map.get(measurements, :confidence),
         duration_ms: Map.get(measurements, :duration_ms)
       }}
    )

    [node_write([{"racing_early_exit:#{analyzer}", :sum, 1}])]
  end

  # The file's duration is the `:code_pipeline` span it was processed in,
  # counted by that span's handler; this event counts its language.
  defp interpret({:telemetry, :code_file_processed, measurements, metadata, _at}) do
    language = segment!(fetch!(metadata, :language))

    :ets.insert(
      @gauges_table,
      {{:code_file_last, language},
       %{
         file_path: Map.get(metadata, :file_path),
         symbols_count: Map.get(measurements, :symbols_count),
         relations_count: Map.get(measurements, :relations_count),
         duration_ms: Map.get(measurements, :duration_ms),
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    [node_write([{"code_language:#{language}", :sum, 1}])]
  end

  defp interpret({:telemetry, :consistency_disagreement, _measurements, metadata, _at}) do
    severity = segment!(fetch!(metadata, :severity))

    :ets.insert(
      @gauges_table,
      {{:consistency, :last_disagreement},
       %{
         final_intent: Map.get(metadata, :final_intent),
         consensus_intent: Map.get(metadata, :consensus_intent),
         severity: severity,
         agreeing: Map.get(metadata, :agreeing),
         dissenting: Map.get(metadata, :dissenting),
         signals: Map.get(metadata, :signals),
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    [node_write([{"consistency:total", :sum, 1}, {"consistency:#{severity}", :sum, 1}])]
  end

  defp interpret({:telemetry, :service_dispatch, measurements, metadata, _at}) do
    service = segment!(fetch!(metadata, :service))
    status = fetch!(metadata, :status)
    duration_ms = fetch!(measurements, :duration)

    :ets.insert(
      @gauges_table,
      {{:service_dispatch_last, service},
       %{
         intent: Map.get(metadata, :intent),
         status: status,
         duration_ms: duration_ms,
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    outcome =
      case status do
        :success -> [{"service:#{service}:success", :sum, 1}]
        {:error, _reason} -> [{"service:#{service}:error", :sum, 1}]
        _neither -> []
      end

    [node_write(duration(:service_dispatch, ms_to_us!(duration_ms)) ++ [{"service:#{service}:count", :sum, 1}] ++ outcome)]
  end

  defp interpret({:telemetry, {:service_cache, type}, measurements, metadata, _at}) do
    service = segment!(fetch!(metadata, :service))
    count = fetch!(measurements, :count)

    unless is_integer(count) and count > 0, do: malformed!("a cache count must be a positive integer, got #{inspect(count)}")

    [node_write([{"service_cache:#{type}", :sum, count}, {"service_cache:#{service}:#{type}", :sum, count}])]
  end

  defp interpret({:telemetry, :service_health_check, measurements, metadata, _at}) do
    :ets.insert(
      @gauges_table,
      {{:service_health, segment!(fetch!(metadata, :service))},
       %{
         status: fetch!(metadata, :status),
         duration_ms: fetch!(measurements, :duration),
         timestamp: System.monotonic_time(:millisecond)
       }}
    )

    []
  end

  defp interpret({:telemetry, :service_credential, _measurements, metadata, _at}) do
    world_id = world!(metadata, :world)
    operation = segment!(fetch!(metadata, :operation))

    :ets.insert(
      @gauges_table,
      {{:service_credential_last, Map.get(metadata, :service)},
       %{operation: operation, timestamp: System.monotonic_time(:millisecond)}}
    )

    [%{scope: {:world, world_id}, family: @telemetry_family, values: [{"service_credential:#{operation}", :sum, 1}]}]
  end

  defp interpret({:telemetry, {:kg_signal, event}, measurements, _metadata, _at}) do
    # A signal that carries no `:count` (a cache hit) counts no items.
    items =
      case Map.get(measurements, :count) do
        nil -> []
        0 -> []
        count when is_integer(count) and count > 0 -> [{"kg_signal:#{event}:items", :sum, count}]
        other -> malformed!("a KG signal's count must be a non-negative integer, got #{inspect(other)}")
      end

    timed =
      case Map.get(measurements, :duration_ms) do
        nil -> []
        duration_ms -> duration({:kg_signal, event}, ms_to_us!(duration_ms))
      end

    [node_write([{"kg_signal:#{event}:count", :sum, 1}] ++ items ++ timed)]
  end

  defp interpret({:record_duration, metric, duration_ms}) do
    [node_write(duration(metric, ms_to_us!(duration_ms)))]
  end

  defp interpret({:record_error, metric, duration_ms, metadata}) do
    [node_write(duration(metric, ms_to_us!(duration_ms), :error) ++ error(fetch!(metadata, :kind), DateTime.utc_now()))]
  end

  defp interpret({:record_readiness, system_name, ready?}) do
    :ets.insert(@gauges_table, {{:readiness, system_name}, %{ready?: ready?, timestamp: System.monotonic_time(:millisecond)}})
    []
  end

  defp interpret({:record_fact_retrieval, result_count}) do
    count = non_negative_integer!(result_count)

    [
      node_write(
        [{"fact_retrieval:total", :sum, 1}] ++
          if(count > 0, do: [{"fact_retrieval:hits", :sum, 1}, {"fact_retrieval:results", :sum, count}], else: [])
      )
    ]
  end

  defp interpret({:record_semantic_retrieval, result_count}) do
    count = non_negative_integer!(result_count)

    [
      node_write(
        [{"semantic_retrieval:total", :sum, 1}] ++ if(count > 0, do: [{"semantic_retrieval:hits", :sum, 1}], else: [])
      )
    ]
  end

  defp interpret({:record_verification_miss_reason, reason}) do
    unless reason in @verification_miss_reasons do
      malformed!("no verification miss reason #{inspect(reason)}, expected one of #{inspect(@verification_miss_reasons)}")
    end

    [node_write([{"verification_miss:#{reason}", :sum, 1}])]
  end

  defp interpret(message), do: malformed!("not an event the aggregator counts: #{inspect(message)}")

  defp node_write(values, records \\ []) do
    %{scope: :node, family: @telemetry_family, values: values, records: records}
  end

  defp take_training_start(model) do
    case :ets.take(@gauges_table, {:training_started, model}) do
      [{_key, started}] -> started
      [] -> nil
    end
  end

  # The values one duration adds to its metric, and its live rate. `:error`
  # counts it as a span that raised.
  defp duration(metric, duration_us, outcome \\ :ok) do
    name = "duration:#{duration_metric_name!(metric)}"
    track_rate(duration_metric_name!(metric))

    [
      {"#{name}:count", :sum, 1},
      {"#{name}:min_us", :min, duration_us},
      {"#{name}:max_us", :max, duration_us},
      {"#{name}:le_ms:#{duration_bucket(duration_us)}", :sum, 1}
    ] ++
      if(duration_us > 0, do: [{"#{name}:sum_us", :sum, duration_us}], else: []) ++
      if(outcome == :error, do: [{"#{name}:error_count", :sum, 1}], else: [])
  end

  defp duration_metric_name!({:kg_signal, event}), do: "kg_signal.#{segment!(event)}"
  defp duration_metric_name!(metric), do: segment!(metric)

  defp duration_bucket(duration_us) do
    Enum.find(@duration_buckets_ms, :inf, &(duration_us <= &1 * 1000))
  end

  defp track_rate(metric) do
    now = System.monotonic_time(:millisecond)

    recent =
      [now | gauge({:rate, metric}) || []]
      |> Enum.filter(&(&1 > now - @rate_window_ms))
      |> Enum.take(@max_rate_points)

    :ets.insert(@gauges_table, [{{:rate, metric}, recent}, {{:last_seen, metric}, now}])
  end

  defp error(type, %DateTime{} = at) do
    name = "error:#{error_type!(type)}"
    [{"#{name}:count", :sum, 1}, {"#{name}:last_at_ms", :max, DateTime.to_unix(at, :millisecond)}]
  end

  defp error_type!({:training_failed, model}), do: "training_failed.#{segment!(model)}"
  defp error_type!(type), do: segment!(type)

  defp native_us!(measurements) do
    case fetch!(measurements, :duration) do
      native when is_integer(native) and native >= 0 -> System.convert_time_unit(native, :native, :microsecond)
      other -> malformed!("a span duration must be a non-negative integer, got #{inspect(other)}")
    end
  end

  defp ms_to_us!(ms) when is_number(ms) and ms >= 0, do: round(ms * 1000)
  defp ms_to_us!(other), do: malformed!("a duration in milliseconds must be a non-negative number, got #{inspect(other)}")

  defp non_negative_integer!(count) when is_integer(count) and count >= 0, do: count
  defp non_negative_integer!(other), do: malformed!("a count must be a non-negative integer, got #{inspect(other)}")

  defp number_or_nil!(map, key) do
    case Map.get(map, key) do
      value when is_number(value) or is_nil(value) -> value
      other -> malformed!("#{key} must be a number, got #{inspect(other)}")
    end
  end

  # A name an event carries, as one segment of a metric name.
  defp segment!(value) when is_atom(value) and value not in [nil, true, false], do: segment!(Atom.to_string(value))

  defp segment!(value) when is_binary(value) do
    if String.trim(value) == "" or String.contains?(value, ":") do
      malformed!("a name to count by must be non-blank and without a colon, got #{inspect(value)}")
    end

    value
  end

  defp segment!(other), do: malformed!("a name to count by must be an atom or a string, got #{inspect(other)}")

  defp world!(metadata, key) do
    case fetch!(metadata, key) do
      world_id when is_binary(world_id) ->
        if String.trim(world_id) == "", do: malformed!("#{key} must name a world, got #{inspect(world_id)}")
        world_id

      other ->
        malformed!("#{key} must name a world, got #{inspect(other)}")
    end
  end

  defp fetch!(map, key) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> value
      :error -> malformed!("missing #{inspect(key)} in #{inspect(Map.keys(map))}")
    end
  end

  defp fetch!(other, key), do: malformed!("cannot read #{inspect(key)} from #{inspect(other)}")

  defp malformed!(message), do: raise(MalformedEvent, message: message)
end
