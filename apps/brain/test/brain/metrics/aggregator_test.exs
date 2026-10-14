defmodule Brain.Metrics.AggregatorTest do
  @moduledoc """
  `Brain.Metrics.Aggregator` counts the node's telemetry in Atlas, so the
  counts survive it restarting: durations as count, sum, min, max and
  histogram, errors, training runs, fact verification, enrichment, service
  dispatches and cache use, and the node's activity. What describes the
  live process (readiness, model loads, the rate per minute, the last event
  seen) is held in memory and starts over with it.

  Events reach it through the handlers `Brain.Telemetry.attach_handlers/0`
  attaches, as in the running application, or through its own record
  functions. It writes them to Atlas in batches, one transaction each.
  """
  use Brain.MetricsCase, async: false

  alias Brain.Telemetry

  describe "process status" do
    test "the aggregator is running and ready" do
      assert Aggregator.ready?() == true
    end
  end

  describe "durations" do
    test "are counted with their mean, shortest, longest, histogram and the upper bound of the 95th percentile" do
      Aggregator.record_duration(:knowledge_research, 4)
      Aggregator.record_duration(:knowledge_research, 40)
      Aggregator.record_duration(:knowledge_research, 400)
      :ok = counted()

      metric = Aggregator.get_metric(:knowledge_research)

      assert metric.count == 3
      assert metric.error_count == 0
      assert metric.avg_ms == 148.0
      assert metric.min_ms == 4.0
      assert metric.max_ms == 400.0
      assert metric.p95_ms == 500.0
      assert %{le_ms: 5, count: 1} in metric.histogram
      assert %{le_ms: 50, count: 1} in metric.histogram
      assert %{le_ms: 500, count: 1} in metric.histogram
      assert metric.histogram |> Enum.map(& &1.count) |> Enum.sum() == 3
      assert metric.sample_count == 3
      assert metric.rate_per_minute == 0.6
      assert is_integer(metric.last_updated)
    end

    test "a duration past the last bucket puts the 95th percentile at the longest one" do
      Aggregator.record_duration(:code_parse, 90_000)
      :ok = counted()

      metric = Aggregator.get_metric(:code_parse)
      assert %{le_ms: :inf, count: 1} in metric.histogram
      assert metric.p95_ms == 90_000.0
    end

    test "a span emitted through telemetry is counted, and one that raised counts as an error of its kind" do
      :ok = Telemetry.span(:jtms_justify, %{}, fn -> :ok end)
      assert_raise RuntimeError, fn -> Telemetry.span(:jtms_justify, %{}, fn -> raise "boom" end) end
      :ok = counted()

      metric = Aggregator.get_metric(:jtms_justify)
      assert metric.count == 2
      assert metric.error_count == 1
      assert %{"error" => %{count: 1, last_seen_at: %DateTime{}}} = Aggregator.get_errors()
    end

    test "every known span is listed before anything is counted, with no mean, and an unknown one is not" do
      metrics = Aggregator.get_metrics()

      for name <- [:brain_evaluate, :pipeline_process, :entity_extract, :code_gazetteer_add, :fact_verification] do
        assert %{count: 0, avg_ms: nil, min_ms: nil, p95_ms: nil} = metrics[name]
      end

      assert Aggregator.get_metric(:nonexistent_metric) == nil
    end
  end

  describe "service metrics" do
    test "dispatches are counted per service, with their outcome, and the last one is held in memory" do
      for _ <- 1..5, do: Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      Telemetry.emit_service_dispatch(:weather, "weather.forecast", {:error, :timeout}, 500)
      :ok = counted()

      metrics = Aggregator.get_service_metrics()
      weather = metrics.by_service["weather"]

      assert weather.total_dispatches == 6
      assert weather.success_count == 5
      assert weather.error_count == 1
      assert weather.success_rate == 83.3
      assert %{intent: "weather.forecast", status: {:error, :timeout}, duration_ms: 500} = weather.last_dispatch
      assert metrics.dispatch.count == 6
      assert metrics.dispatch.max_ms == 500.0
    end

    test "cache hits and misses are counted with their hit rate" do
      Telemetry.emit_service_cache_hit(:weather, "k1")
      Telemetry.emit_service_cache_hit(:weather, "k2")
      Telemetry.emit_service_cache_hit(:weather, "k3")
      Telemetry.emit_service_cache_miss(:weather, "k4")
      :ok = counted()

      assert Aggregator.get_service_metrics().cache == %{hits: 3, misses: 1, hit_rate: 75.0}
    end

    test "a health check is held in memory for the service" do
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      Telemetry.emit_service_health_check(:weather, :ok, 50)
      :ok = counted()

      assert %{status: :ok, duration_ms: 50} = Aggregator.get_service_metrics().by_service["weather"].health_status
    end
  end

  describe "epistemic, enrichment and verification misses" do
    test "fact verifications are counted by status, with contradictions and their duration" do
      Telemetry.emit_fact_verification(:verified, "paris", 5)
      Telemetry.emit_fact_verification(:verified, "rome", 7)
      Telemetry.emit_fact_verification(:contradicted, "the moon", 3)
      :ok = counted()

      metrics = Aggregator.get_epistemic_metrics()
      assert metrics.total_verifications == 3
      assert metrics.contradiction_count == 1
      assert metrics.by_status == %{verified: 2, contradicted: 1, uncertain: 0, unchecked: 0}
      assert %{verified: %{subject: "rome"}, contradicted: %{subject: "the moon"}} = metrics.last_verifications
      assert Aggregator.get_metric(:fact_verification).count == 3
    end

    test "fact and semantic retrievals give their hit rates, and verification misses their reasons" do
      :ok = Aggregator.record_fact_retrieval(3)
      :ok = Aggregator.record_fact_retrieval(0)
      :ok = Aggregator.record_semantic_retrieval(2)
      :ok = Aggregator.record_verification_miss_reason(:no_facts)
      :ok = Aggregator.record_verification_miss_reason(:no_facts)
      :ok = Aggregator.record_verification_miss_reason(:low_confidence)
      :ok = counted()

      assert Aggregator.get_enrichment_metrics() == %{
               fact_hit_rate: 50.0,
               semantic_hit_rate: 100.0,
               avg_facts_per_response: 1.5,
               fact_queries: 2,
               semantic_queries: 1
             }

      assert %{total: 3, counts: %{no_facts: 2, low_confidence: 1, no_subject: 0}, percentages: %{no_facts: 66.7}} =
               Aggregator.get_verification_miss_reasons()
    end
  end

  describe "training" do
    test "the last run of each model is kept, and a failed one counts as an error" do
      :ok = :telemetry.execute([:chat_bot, :ml, :train, :start], %{sequence_count: 10}, %{model: :pos_tagger})

      :ok =
        :telemetry.execute(
          [:chat_bot, :ml, :train, :stop],
          %{duration_ms: 1200, tag_count: 17, feature_count: 400},
          %{model: :pos_tagger, success: true}
        )

      :ok = :telemetry.execute([:chat_bot, :ml, :train, :exception], %{duration_ms: 3}, %{model: :entity_trainer, reason: :no_rows})
      :ok = counted()

      assert %{
               "pos_tagger" => %{
                 status: :completed,
                 started_at: %DateTime{},
                 completed_at: %DateTime{},
                 sequence_count: 10,
                 duration_ms: 1200,
                 tag_count: 17,
                 feature_count: 400,
                 success: true
               },
               "entity_trainer" => %{status: :failed, started_at: nil, reason: ":no_rows", success: false}
             } = Aggregator.get_training_stats()

      assert Aggregator.get_metrics().training == Aggregator.get_training_stats()
      assert %{"training_failed.entity_trainer" => %{count: 1}} = Aggregator.get_errors()
    end
  end

  describe "activity" do
    test "KG signals, learning, consistency, racing early exits and code languages are counted" do
      world_id = "test_aggregator_learning_#{System.unique_integer([:positive])}"

      :ok = :telemetry.execute([:brain, :kg_signal, :srl_batch], %{duration_ms: 12, count: 4}, %{})
      :ok = :telemetry.execute([:brain, :kg_signal, :cache_hit], %{}, %{})
      :ok = :telemetry.execute([:chat_bot, :learning, :document_processed], %{}, %{world_id: world_id})
      :ok = :telemetry.execute([:chat_bot, :analysis, :consistency, :disagreement], %{}, %{severity: :high})
      Telemetry.emit_racing_early_exit(:fast_path, 0.9, 3)

      # As Brain.Code.Pipeline.process/3 emits a processed file: inside its span.
      :ok =
        Telemetry.span(:code_pipeline, %{language: :elixir}, fn ->
          Telemetry.emit_code_file_processed("lib/a.ex", :elixir, 3, 2, 25)
        end)

      :ok = counted()

      activity = Aggregator.get_activity_metrics()

      assert %{"srl_batch" => %{count: 1, items: 4, duration: %{count: 1, max_ms: 12.0}}, "cache_hit" => %{count: 1, items: 0, duration: nil}} =
               activity.kg_signals

      assert activity.learning == %{"document_processed" => 1}
      assert activity.consistency == %{total: 1, by_severity: %{"high" => 1}}
      assert activity.racing_early_exits == %{"fast_path" => 1}
      assert activity.code_languages == %{"elixir" => 1}
      # The file is counted once, by its span.
      assert Aggregator.get_metric(:code_pipeline).count == 1

      # A learning event is counted in its world as well.
      assert [%{values: %{"learning:document_processed" => 1}}] = Atlas.Metrics.windows({:world, world_id}, "telemetry", 1)
    end
  end

  describe "gauges" do
    test "readiness and model loads are held in memory, and start over with the aggregator" do
      Aggregator.record_readiness(:test_system, true)
      Telemetry.emit_model_load("pos_tagger", 12.5, true)
      :ok = counted()

      assert %{test_system: %{ready?: true}} = Aggregator.get_metrics().readiness
      assert %{"pos_tagger" => %{duration_ms: 12.5, success: true}} = Aggregator.get_model_load_stats()
      assert Aggregator.get_metrics().model_loads == Aggregator.get_model_load_stats()

      _restarted = restart_aggregator!()

      assert Aggregator.get_metrics().readiness == %{}
      assert Aggregator.get_model_load_stats() == %{}
    end

    test "a model load is measured where the model is read: a returned error or a raise is a failed load" do
      assert {:ok, :model} = Telemetry.measure_model_load("speech_act_classifier", fn -> {:ok, :model} end)
      assert {:error, :model_not_found} = Telemetry.measure_model_load("sentiment_classifier", fn -> {:error, :model_not_found} end)
      assert_raise RuntimeError, fn -> Telemetry.measure_model_load("entity_model", fn -> raise "not a model" end) end
      :ok = counted()

      assert %{
               "speech_act_classifier" => %{success: true, duration_ms: ms},
               "sentiment_classifier" => %{success: false},
               "entity_model" => %{success: false}
             } = Aggregator.get_model_load_stats()

      assert is_float(ms)
    end
  end

  describe "batches" do
    test "the events handled since the last write are written in one transaction" do
      test_pid = self()

      :ok =
        :telemetry.attach(
          "aggregator-test-queries",
          [:atlas, :repo, :query],
          fn _event, _measurements, %{source: source}, _config -> send(test_pid, {:query, source}) end,
          nil
        )

      on_exit(fn -> :telemetry.detach("aggregator-test-queries") end)

      for _ <- 1..100, do: Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      :ok = counted()
      :telemetry.detach("aggregator-test-queries")

      assert Aggregator.get_service_metrics().by_service["weather"].total_dispatches == 100
      # One write per batch: the window opened or found, and its values.
      assert length(received_queries("atlas_metric_values")) <= 2
    end

    test "starting a window writes the events sent before it into the window they happened in" do
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      :ok = Aggregator.reset()

      assert [%{values: new_values}, %{values: %{"service:weather:count" => 1}}] = Atlas.Metrics.windows(:node, "telemetry", 2)
      assert new_values == %{}
    end

    test "a sum, a minimum and a maximum combine in a batch as Atlas combines them" do
      Aggregator.record_duration(:knowledge_research, 40)
      Aggregator.record_duration(:knowledge_research, 4)
      Aggregator.record_duration(:knowledge_research, 400)
      :ok = counted()
      Aggregator.record_duration(:knowledge_research, 2)
      :ok = counted()

      assert %{count: 4, min_ms: 2.0, max_ms: 400.0, avg_ms: 111.5} = Aggregator.get_metric(:knowledge_research)
    end
  end

  describe "read once" do
    test "every reader answers from one read of the telemetry window" do
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      Telemetry.emit_fact_verification(:verified, "paris", 5)
      :ok = counted()

      telemetry = Aggregator.read_telemetry()
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      :ok = counted()

      # What was read stays what is answered; a new read sees the later event.
      assert Aggregator.get_service_metrics(telemetry).by_service["weather"].total_dispatches == 1
      assert Aggregator.get_metric(:service_dispatch, telemetry).count == 1
      assert Aggregator.get_epistemic_metrics(telemetry).total_verifications == 1
      assert Aggregator.get_telemetry_window(telemetry) == Aggregator.get_telemetry_window()
      assert Aggregator.get_service_metrics().by_service["weather"].total_dispatches == 2
    end
  end

  defp received_queries(source) do
    receive do
      {:query, ^source} -> [source | received_queries(source)]
      {:query, _other} -> received_queries(source)
    after
      0 -> []
    end
  end

  describe "a restart" do
    test "keeps every count kept in Atlas, and later events add to them" do
      Aggregator.record_duration(:knowledge_research, 40)
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      Telemetry.emit_service_cache_hit(:weather, "k1")
      Telemetry.emit_fact_verification(:verified, "paris", 5)
      :ok = Aggregator.record_fact_retrieval(3)
      :ok = Aggregator.record_verification_miss_reason(:no_facts)
      :ok = :telemetry.execute([:chat_bot, :ml, :train, :stop], %{duration_ms: 1200}, %{model: :pos_tagger})
      :ok = :telemetry.execute([:brain, :kg_signal, :cache_hit], %{}, %{})
      assert_raise RuntimeError, fn -> Telemetry.span(:jtms_justify, %{}, fn -> raise "boom" end) end
      :ok = counted()

      before = persisted()
      window = Aggregator.get_telemetry_window()
      assert %DateTime{} = window.started_at

      _restarted = restart_aggregator!()

      assert persisted() == before
      assert Aggregator.get_telemetry_window() == window

      # What describes the process before the restart is gone.
      research = Aggregator.get_metric(:knowledge_research)
      assert research.sample_count == 0
      assert research.last_updated == nil
      assert Aggregator.get_service_metrics().by_service["weather"].last_dispatch == nil
      assert Aggregator.get_epistemic_metrics().last_verifications == %{}

      Aggregator.record_duration(:knowledge_research, 4)
      Telemetry.emit_service_dispatch(:weather, "weather.query", :success, 100)
      :ok = counted()

      research = Aggregator.get_metric(:knowledge_research)
      assert research.count == 2
      assert research.min_ms == 4.0
      assert research.max_ms == 40.0
      assert Aggregator.get_service_metrics().by_service["weather"].total_dispatches == 2
    end
  end

  describe "reset/0" do
    test "starts a new telemetry window from zero, and the ended window keeps its counts" do
      Telemetry.emit_service_dispatch(:weather, "test", :success, 100)
      :ok = counted()
      before = Aggregator.get_telemetry_window()

      :ok = Aggregator.reset()

      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service == %{}
      assert metrics.cache.hits == 0
      assert Aggregator.get_telemetry_window().started_at != before.started_at

      assert [%{values: new_values}, %{window: %{ended_at: %DateTime{}}, values: %{"service:weather:count" => 1}}] =
               Atlas.Metrics.windows(:node, "telemetry", 2)

      assert new_values == %{}
    end
  end

  # Everything the aggregator answers from Atlas, without what it holds in
  # memory about the live process.
  defp persisted do
    durations =
      Aggregator.get_metrics()
      |> Enum.filter(fn {_name, metric} -> is_map(metric) and Map.has_key?(metric, :histogram) end)
      |> Map.new(fn {name, metric} -> {name, Map.drop(metric, [:rate_per_minute, :sample_count, :last_updated])} end)

    services = Aggregator.get_service_metrics()

    %{
      durations: durations,
      errors: Aggregator.get_errors(),
      training: Aggregator.get_training_stats(),
      services: %{
        services
        | by_service: Map.new(services.by_service, fn {name, service} -> {name, Map.drop(service, [:last_dispatch, :health_status])} end),
          dispatch: Map.drop(services.dispatch, [:rate_per_minute, :sample_count, :last_updated]),
          enrichment: Map.drop(services.enrichment, [:rate_per_minute, :sample_count, :last_updated])
      },
      epistemic: Map.delete(Aggregator.get_epistemic_metrics(), :last_verifications),
      enrichment: Aggregator.get_enrichment_metrics(),
      verification_misses: Aggregator.get_verification_miss_reasons(),
      activity: Aggregator.get_activity_metrics() |> update_in([:kg_signals], &drop_live_kg_durations/1)
    }
  end

  defp drop_live_kg_durations(kg_signals) do
    Map.new(kg_signals, fn
      {event, %{duration: nil} = signal} -> {event, signal}
      {event, signal} -> {event, update_in(signal.duration, &Map.drop(&1, [:rate_per_minute, :sample_count, :last_updated]))}
    end)
  end
end
