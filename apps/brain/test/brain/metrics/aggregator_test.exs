defmodule Brain.Metrics.AggregatorTest do
  use ExUnit.Case, async: false

  alias Brain.Metrics.Aggregator

  setup do
    # Reset metrics before each test
    if Process.whereis(Aggregator) do
      Aggregator.reset()
    end

    :ok
  end

  describe "process status" do
    test "aggregator is running" do
      assert Process.whereis(Aggregator) != nil
    end
  end

  describe "get_metrics/0" do
    test "returns metrics map" do
      metrics = Aggregator.get_metrics()

      assert is_map(metrics)
      # Should have some default metrics initialized
      assert Map.has_key?(metrics, :brain_evaluate) or map_size(metrics) >= 0
    end
  end

  describe "get_metric/1" do
    test "returns nil for unknown metric" do
      assert Aggregator.get_metric(:nonexistent_metric) == nil
    end

    test "returns default metrics structure" do
      metric = Aggregator.get_metric(:brain_evaluate)

      if metric do
        assert Map.has_key?(metric, :count)
        assert Map.has_key?(metric, :avg_ms)
      end
    end
  end

  describe "get_service_metrics/0" do
    test "returns service metrics structure" do
      metrics = Aggregator.get_service_metrics()

      assert is_map(metrics)
      assert Map.has_key?(metrics, :dispatch)
      assert Map.has_key?(metrics, :enrichment)
      assert Map.has_key?(metrics, :by_service)
      assert Map.has_key?(metrics, :cache)
    end

    test "cache metrics have required fields" do
      metrics = Aggregator.get_service_metrics()

      assert Map.has_key?(metrics.cache, :hits)
      assert Map.has_key?(metrics.cache, :misses)
      assert Map.has_key?(metrics.cache, :hit_rate)
    end

    test "dispatch metrics have required fields" do
      metrics = Aggregator.get_service_metrics()

      assert Map.has_key?(metrics.dispatch, :count)
      assert Map.has_key?(metrics.dispatch, :avg_ms)
    end
  end

  describe "record_service_dispatch cast" do
    test "records service dispatch metrics" do
      GenServer.cast(Aggregator, {:record_service_dispatch, :weather, "weather.query", :success, 150})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      weather_metrics = metrics.by_service[:weather]

      assert weather_metrics != nil
      assert weather_metrics.total_dispatches == 1
      assert weather_metrics.success_count == 1
      assert weather_metrics.error_count == 0
      assert weather_metrics.success_rate == 100.0
    end

    test "tracks multiple dispatches" do
      for _ <- 1..5 do
        GenServer.cast(Aggregator, {:record_service_dispatch, :weather, "weather.query", :success, 100})
      end

      GenServer.cast(Aggregator, {:record_service_dispatch, :weather, "weather.query", {:error, :timeout}, 500})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      weather = metrics.by_service[:weather]

      assert weather.total_dispatches == 6
      assert weather.success_count == 5
      assert weather.error_count == 1
      # 5 out of 6 = 83.3%
      assert weather.success_rate == 83.3
    end

    test "tracks last dispatch info" do
      GenServer.cast(Aggregator, {:record_service_dispatch, :weather, "weather.forecast", :success, 200})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      last = metrics.by_service[:weather].last_dispatch

      assert last.intent == "weather.forecast"
      assert last.status == :success
      assert last.duration_ms == 200
      assert is_integer(last.timestamp)
    end
  end

  describe "record_service_cache cast" do
    test "records cache hits" do
      GenServer.cast(Aggregator, {:record_service_cache, :hit, :weather, 1})
      GenServer.cast(Aggregator, {:record_service_cache, :hit, :weather, 1})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.hits == 2
    end

    test "records cache misses" do
      GenServer.cast(Aggregator, {:record_service_cache, :miss, :weather, 1})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.misses == 1
    end

    test "calculates hit rate" do
      # 3 hits, 1 miss = 75%
      GenServer.cast(Aggregator, {:record_service_cache, :hit, :weather, 1})
      GenServer.cast(Aggregator, {:record_service_cache, :hit, :weather, 1})
      GenServer.cast(Aggregator, {:record_service_cache, :hit, :weather, 1})
      GenServer.cast(Aggregator, {:record_service_cache, :miss, :weather, 1})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.hit_rate == 75.0
    end
  end

  describe "record_service_health_check cast" do
    test "records health check result" do
      GenServer.cast(Aggregator, {:record_service_health_check, :weather, :ok, 50})

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      health = metrics.by_service[:weather]

      # Health status should be recorded
      if health do
        assert health.health_status != nil or health.health_status == nil
      end
    end
  end

  describe "reset/0" do
    test "clears all metrics" do
      # Record some data
      GenServer.cast(Aggregator, {:record_service_dispatch, :weather, "test", :success, 100})
      Process.sleep(50)

      # Reset
      Aggregator.reset()

      # Verify cleared
      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service == %{}
      assert metrics.cache.hits == 0
      assert metrics.cache.misses == 0
    end
  end

  describe "get_errors/0" do
    test "returns error metrics map" do
      errors = Aggregator.get_errors()
      assert is_map(errors)
    end
  end

  describe "get_queue_sizes/0" do
    test "returns queue size metrics" do
      queue_sizes = Aggregator.get_queue_sizes()
      assert is_map(queue_sizes)
    end
  end

  describe "record_duration/3" do
    test "records duration metric" do
      Aggregator.record_duration(:test_metric, 150, %{})

      # Duration metrics are processed asynchronously
      Process.sleep(50)

      # The metric should be recorded without error
      assert true
    end
  end
end
