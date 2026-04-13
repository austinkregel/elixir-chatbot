defmodule Brain.Services.CacheTest do
  use ExUnit.Case, async: false

  alias Brain.Services.Cache

  setup do
    # Clear test entries after each test
    on_exit(fn ->
      if Cache.ready?() do
        Cache.clear(:test_service)
      end
    end)

    :ok
  end

  describe "ready?/0" do
    test "returns true when cache is running" do
      assert Cache.ready?() == true
    end
  end

  describe "put/4 and get/2" do
    test "stores and retrieves a value" do
      Cache.put(:test_service, "test_key", "test_value")

      assert {:ok, "test_value"} = Cache.get(:test_service, "test_key")
    end

    test "stores complex data structures" do
      data = %{
        temperature: "72°F",
        conditions: "sunny",
        raw: %{temp: 72.5}
      }

      Cache.put(:test_service, "weather:NYC", data)

      assert {:ok, ^data} = Cache.get(:test_service, "weather:NYC")
    end

    test "returns :miss for non-existent key" do
      assert :miss = Cache.get(:test_service, "nonexistent_key")
    end
  end

  describe "TTL expiration" do
    test "returns :miss for expired entries" do
      # Store with 1ms TTL
      Cache.put(:test_service, "expiring_key", "value", ttl: 1)

      # Wait for expiration
      Process.sleep(10)

      assert :miss = Cache.get(:test_service, "expiring_key")
    end

    test "returns value for non-expired entries" do
      # Store with long TTL
      Cache.put(:test_service, "valid_key", "value", ttl: :timer.minutes(10))

      assert {:ok, "value"} = Cache.get(:test_service, "valid_key")
    end
  end

  describe "delete/2" do
    test "removes a specific entry" do
      Cache.put(:test_service, "key_to_delete", "value")
      assert {:ok, "value"} = Cache.get(:test_service, "key_to_delete")

      Cache.delete(:test_service, "key_to_delete")

      assert :miss = Cache.get(:test_service, "key_to_delete")
    end

    test "succeeds for non-existent key" do
      assert :ok = Cache.delete(:test_service, "nonexistent")
    end
  end

  describe "clear/1" do
    test "removes all entries for a service" do
      Cache.put(:test_service, "key1", "value1")
      Cache.put(:test_service, "key2", "value2")
      Cache.put(:other_service, "key3", "value3")

      Cache.clear(:test_service)

      assert :miss = Cache.get(:test_service, "key1")
      assert :miss = Cache.get(:test_service, "key2")
      assert {:ok, "value3"} = Cache.get(:other_service, "key3")

      # Cleanup
      Cache.delete(:other_service, "key3")
    end
  end

  describe "clear_all/0" do
    test "removes all cached entries" do
      Cache.put(:service_a, "key1", "value1")
      Cache.put(:service_b, "key2", "value2")

      Cache.clear_all()

      assert :miss = Cache.get(:service_a, "key1")
      assert :miss = Cache.get(:service_b, "key2")
    end
  end

  describe "stats/0" do
    test "returns cache statistics" do
      Cache.clear_all()

      Cache.put(:test_service, "key1", "value1")
      Cache.put(:test_service, "key2", "value2")

      stats = Cache.stats()

      assert stats.total >= 2
      assert stats.valid >= 2
      assert is_map(stats.by_service)
    end
  end

  describe "telemetry emission" do
    setup do
      # Reset metrics to get clean counts
      if Process.whereis(Brain.Metrics.Aggregator) do
        Brain.Metrics.Aggregator.reset()
      end

      :ok
    end

    test "emits cache hit telemetry on successful get" do
      Cache.put(:telemetry_test, "hit_key", "value")

      # First get should be a hit
      assert {:ok, "value"} = Cache.get(:telemetry_test, "hit_key")

      # Wait for telemetry processing
      Process.sleep(50)

      metrics = Brain.Metrics.Aggregator.get_service_metrics()
      assert metrics.cache.hits >= 1

      # Cleanup
      Cache.delete(:telemetry_test, "hit_key")
    end

    test "emits cache miss telemetry when key not found" do
      # Get non-existent key
      assert :miss = Cache.get(:telemetry_test, "nonexistent_key_#{:rand.uniform(10000)}")

      Process.sleep(50)

      metrics = Brain.Metrics.Aggregator.get_service_metrics()
      assert metrics.cache.misses >= 1
    end

    test "emits cache miss telemetry when entry expired" do
      # Store with very short TTL
      Cache.put(:telemetry_test, "expiring", "value", ttl: 1)

      # Wait for expiration
      Process.sleep(10)

      # Get expired entry
      assert :miss = Cache.get(:telemetry_test, "expiring")

      Process.sleep(50)

      metrics = Brain.Metrics.Aggregator.get_service_metrics()
      assert metrics.cache.misses >= 1
    end
  end
end
