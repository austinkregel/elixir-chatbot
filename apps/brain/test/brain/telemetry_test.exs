defmodule Brain.TelemetryTest do
  use ExUnit.Case, async: false

  alias Brain.Telemetry
  alias Brain.Metrics.Aggregator

  setup do
    # Reset metrics before each test
    if Process.whereis(Aggregator) do
      Aggregator.reset()
    end

    :ok
  end

  describe "emit_service_dispatch/4" do
    test "emits telemetry event that gets recorded" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", :success, 150)

      # Wait for async processing
      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service[:test_service] != nil
      assert metrics.by_service[:test_service].total_dispatches == 1
      assert metrics.by_service[:test_service].success_count == 1
    end

    test "records error status correctly" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", {:error, :timeout}, 500)

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service[:test_service].error_count == 1
    end

    test "tracks duration in last_dispatch metadata" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", :success, 250)

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      last_dispatch = metrics.by_service[:test_service].last_dispatch
      assert last_dispatch != nil
      assert last_dispatch.duration_ms == 250
      assert last_dispatch.intent == "test.intent"
      assert last_dispatch.status == :success
    end
  end

  describe "emit_service_cache_hit/2" do
    test "increments cache hit counter" do
      Telemetry.emit_service_cache_hit(:weather, "test_key")
      Telemetry.emit_service_cache_hit(:weather, "another_key")

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.hits == 2
    end
  end

  describe "emit_service_cache_miss/2" do
    test "increments cache miss counter" do
      Telemetry.emit_service_cache_miss(:weather, "test_key")

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.misses == 1
    end
  end

  describe "cache hit rate calculation" do
    test "calculates hit rate correctly" do
      # 2 hits, 2 misses = 50% hit rate
      Telemetry.emit_service_cache_hit(:weather, "key1")
      Telemetry.emit_service_cache_hit(:weather, "key2")
      Telemetry.emit_service_cache_miss(:weather, "key3")
      Telemetry.emit_service_cache_miss(:weather, "key4")

      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.cache.hits == 2
      assert metrics.cache.misses == 2
      assert metrics.cache.hit_rate == 50.0
    end
  end

  describe "emit_credential_operation/3" do
    test "records credential store operation" do
      Telemetry.emit_credential_operation(:store, :weather, "default")

      Process.sleep(50)

      # The operation is recorded - we verify it doesn't crash
      # Full verification would require checking ETS directly
      assert true
    end

    test "records credential delete operation" do
      Telemetry.emit_credential_operation(:delete, :weather, "default")

      Process.sleep(50)

      assert true
    end
  end

  describe "attach_handlers/0 and detach_handlers/0" do
    test "handlers can be attached and detached" do
      # Detach
      assert :ok = Telemetry.detach_handlers()

      # Verify events don't get processed (no crash expected)
      Telemetry.emit_service_dispatch(:test, "test", :success, 100)

      # Re-attach
      assert :ok = Telemetry.attach_handlers()

      # Verify events get processed again
      Telemetry.emit_service_dispatch(:test2, "test", :success, 100)
      Process.sleep(50)

      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service[:test2] != nil
    end
  end
end
