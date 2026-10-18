defmodule Brain.TelemetryTest do
  @moduledoc """
  The events `Brain.Telemetry` emits reach `Brain.Metrics.Aggregator`
  through the handlers `attach_handlers/0` attaches, and are counted;
  `detach_handlers/0` detaches every one of them, and attaching again
  counts again.
  """
  use Brain.MetricsCase, async: false

  alias Brain.Telemetry

  describe "emit_service_dispatch/4" do
    test "emits an event that is counted for its service" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", :success, 150)
      :ok = counted()

      metrics = Aggregator.get_service_metrics()
      assert metrics.by_service["test_service"].total_dispatches == 1
      assert metrics.by_service["test_service"].success_count == 1
    end

    test "counts an error status as an error" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", {:error, :timeout}, 500)
      :ok = counted()

      assert Aggregator.get_service_metrics().by_service["test_service"].error_count == 1
    end

    test "holds the last dispatch's intent, status and duration" do
      Telemetry.emit_service_dispatch(:test_service, "test.intent", :success, 250)
      :ok = counted()

      last_dispatch = Aggregator.get_service_metrics().by_service["test_service"].last_dispatch
      assert last_dispatch.duration_ms == 250
      assert last_dispatch.intent == "test.intent"
      assert last_dispatch.status == :success
    end
  end

  describe "emit_service_cache_hit/2 and emit_service_cache_miss/2" do
    test "count hits and misses, and their hit rate" do
      Telemetry.emit_service_cache_hit(:weather, "key1")
      Telemetry.emit_service_cache_hit(:weather, "key2")
      Telemetry.emit_service_cache_miss(:weather, "key3")
      Telemetry.emit_service_cache_miss(:weather, "key4")
      :ok = counted()

      assert Aggregator.get_service_metrics().cache == %{hits: 2, misses: 2, hit_rate: 50.0}
    end
  end

  describe "emit_credential_operation/3" do
    test "counts each operation in the world it was made in, and no other" do
      world_id = "test_telemetry_credentials_#{System.unique_integer([:positive])}"
      other_world_id = "test_telemetry_credentials_other_#{System.unique_integer([:positive])}"

      Telemetry.emit_credential_operation(:store, :weather, world_id)
      Telemetry.emit_credential_operation(:store, :weather, world_id)
      Telemetry.emit_credential_operation(:delete, :weather, world_id)
      :ok = counted()

      assert [%{values: values}] = Atlas.Metrics.windows({:world, world_id}, "telemetry", 1)
      assert values == %{"service_credential:store" => 2, "service_credential:delete" => 1}
      assert Atlas.Metrics.windows({:world, other_world_id}, "telemetry", 1) == []
    end
  end

  describe "attach_handlers/0 and detach_handlers/0" do
    test "every handler id is attached once" do
      ids = Telemetry.handler_ids()
      assert ids == Enum.uniq(ids)

      attached = [] |> :telemetry.list_handlers() |> Enum.map(& &1.id)
      assert Enum.all?(ids, &(&1 in attached))
    end

    test "detach_handlers/0 detaches every handler, the KG signal and novelty ones too, and attaching again counts again" do
      :ok = Telemetry.detach_handlers()

      attached = [] |> :telemetry.list_handlers() |> Enum.map(& &1.id)
      assert Enum.filter(Telemetry.handler_ids(), &(&1 in attached)) == []

      for id <- ~w(chatbot-kg-signal-scored chatbot-kg-signal-cache-hit chatbot-kg-signal-cache-miss
                   chatbot-kg-signal-cache-invalidated chatbot-kg-signal-srl-batch
                   chatbot-novelty-downweighted chatbot-novelty-state-change) do
        assert id in Telemetry.handler_ids()
        refute id in attached
      end

      # Nothing hears it, so nothing counts it.
      Telemetry.emit_service_dispatch(:detached_service, "test", :success, 100)

      :ok = Telemetry.attach_handlers()

      Telemetry.emit_service_dispatch(:attached_service, "test", :success, 100)
      :ok = counted()

      by_service = Aggregator.get_service_metrics().by_service
      refute Map.has_key?(by_service, "detached_service")
      assert by_service["attached_service"].total_dispatches == 1
    end

    test "attaching again leaves the handlers attached as they are" do
      :ok = Telemetry.attach_handlers()
      :ok = Telemetry.attach_handlers()

      attached = [] |> :telemetry.list_handlers() |> Enum.map(& &1.id)
      assert Enum.filter(attached, &(&1 in Telemetry.handler_ids())) |> Enum.sort() == Enum.sort(Telemetry.handler_ids())
    end
  end
end
