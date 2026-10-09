defmodule Brain.Services.DispatcherTest do
  use ExUnit.Case, async: false

  alias Brain.Services.Dispatcher

  describe "find_service/1" do
    test "returns Weather service for weather.query intent" do
      assert Dispatcher.find_service("weather.query") == Brain.Services.Weather
    end

    test "returns Weather service for weather.forecast intent" do
      assert Dispatcher.find_service("weather.forecast") == Brain.Services.Weather
    end

    test "returns nil for bare weather intent without dot" do
      # "weather" alone is not a dotted intent, so domain-prefix fallback doesn't apply
      assert Dispatcher.find_service("weather") == nil
    end

    test "returns Weather service for weather.current intent" do
      assert Dispatcher.find_service("weather.current") == Brain.Services.Weather
    end

    test "returns nil for unknown intent" do
      assert Dispatcher.find_service("unknown.intent") == nil
    end
  end

  describe "find_service_by_name/1" do
    test "returns Weather service for :weather name" do
      assert Dispatcher.find_service_by_name(:weather) == Brain.Services.Weather
    end

    test "returns nil for unknown service name" do
      assert Dispatcher.find_service_by_name(:unknown_service) == nil
    end
  end

  describe "list_services/1" do
    test "returns list of service info maps" do
      services = Dispatcher.list_services(world: "default")

      assert is_list(services)
      assert length(services) >= 1

      weather = Enum.find(services, &(&1.name == :weather))
      assert weather != nil
      assert weather.display_name == "Weather"
      assert weather.module == Brain.Services.Weather
      assert "weather.query" in weather.supported_intents
    end

    test "includes configuration status" do
      services = Dispatcher.list_services(world: "default")
      weather = Enum.find(services, &(&1.name == :weather))

      assert Map.has_key?(weather, :configured)
      assert is_boolean(weather.configured)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/Dispatcher.list_services needs :world/, fn ->
        Dispatcher.list_services([])
      end
    end
  end

  describe "dispatch/3" do
    test "returns :no_handler for unknown intent" do
      assert Dispatcher.dispatch("unknown.intent", %{}, %{world_id: "default"}) == :no_handler
    end

    test "raises when the context names no world" do
      assert_raise ArgumentError, ~r/Dispatcher.dispatch context needs :world_id/, fn ->
        Dispatcher.dispatch("weather.query", %{location: "NYC"}, %{})
      end
    end

    test "raises before routing, even for an intent no service handles" do
      assert_raise ArgumentError, ~r/Dispatcher.dispatch context needs :world_id/, fn ->
        Dispatcher.dispatch("unknown.intent", %{}, %{})
      end
    end

    test "raises when the context world is not a world id" do
      assert_raise ArgumentError, ~r/Dispatcher.dispatch context :world_id must be a world id/, fn ->
        Dispatcher.dispatch("weather.query", %{location: "NYC"}, %{world_id: ""})
      end
    end

    test "accepts the world as :world in the context" do
      result = Dispatcher.dispatch("weather.query", %{location: "NYC"}, %{world: "default"})

      assert result == {:error, :missing_credentials}
    end

    test "returns error when service has missing credentials" do
      # Weather service without configured credentials
      result = Dispatcher.dispatch("weather.query", %{location: "NYC"}, %{world_id: "default"})

      assert result == {:error, :missing_credentials}
    end
  end

  describe "service_available?/2" do
    test "returns false when credentials not configured" do
      # By default, weather won't have credentials
      refute Dispatcher.service_available?(:weather, world: "default")
    end

    test "returns false for unknown service" do
      refute Dispatcher.service_available?(:unknown_service, world: "default")
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/Dispatcher.service_available\? needs :world/, fn ->
        Dispatcher.service_available?(:weather, [])
      end
    end
  end

  describe "service_schemas/0" do
    test "returns schemas for all services that implement slot_schema/0" do
      schemas = Dispatcher.service_schemas()

      assert is_map(schemas)
      assert Map.has_key?(schemas, "weather.query")
      assert Map.has_key?(schemas, "weather.forecast")
      assert Map.has_key?(schemas, "weather.current")
    end

    test "weather schema includes location in required slots" do
      schemas = Dispatcher.service_schemas()
      weather_schema = schemas["weather.query"]

      assert "location" in weather_schema["required"]
    end

    test "weather schema includes entity_mappings for location" do
      schemas = Dispatcher.service_schemas()
      weather_schema = schemas["weather.query"]
      mappings = weather_schema["entity_mappings"]

      assert "gpe" in mappings["location"]
      assert "city" in mappings["location"]
    end

    test "weather schema includes clarification templates" do
      schemas = Dispatcher.service_schemas()
      weather_schema = schemas["weather.query"]

      assert weather_schema["clarification_templates"]["location"] ==
               "What location would you like the weather for?"
    end
  end

  describe "find_service/1 domain prefix fallback" do
    test "finds Weather service for weather.request_information via domain prefix" do
      assert Dispatcher.find_service("weather.request_information") == Brain.Services.Weather
    end

    test "finds Weather service for weather.some_other_subtype via domain prefix" do
      assert Dispatcher.find_service("weather.some_other_subtype") == Brain.Services.Weather
    end

    test "returns nil for completely unrelated domain prefix" do
      assert Dispatcher.find_service("banking.transfer") == nil
    end

    test "prefers exact match over domain prefix" do
      assert Dispatcher.find_service("weather.query") == Brain.Services.Weather
    end
  end

  describe "service_schemas/0 domain prefix" do
    test "includes domain-level keys for schema lookup" do
      schemas = Dispatcher.service_schemas()
      assert Map.has_key?(schemas, "weather")
    end

    test "domain-level schema has same structure as intent-level schema" do
      schemas = Dispatcher.service_schemas()
      weather_domain = schemas["weather"]
      weather_query = schemas["weather.query"]

      assert weather_domain["required"] == weather_query["required"]
      assert weather_domain["entity_mappings"] == weather_query["entity_mappings"]
    end
  end

  describe "find_service/1 Home Assistant routing" do
    test "routes smarthome.switch to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("smarthome.switch") == Brain.Services.HomeAssistant
    end

    test "routes smarthome.query to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("smarthome.query") == Brain.Services.HomeAssistant
    end

    test "routes device.control to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("device.control") == Brain.Services.HomeAssistant
    end

    test "routes music.play to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("music.play") == Brain.Services.HomeAssistant
    end

    test "routes an alarm intent the local scheduler does not handle to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("alarm.snooze") == Brain.Services.HomeAssistant
    end

    test "routes timer.start to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("timer.start") == Brain.Services.HomeAssistant
    end
  end

  describe "find_service/1 local scheduler routing" do
    test "routes every intent the local scheduler supports to it, ahead of HomeAssistant's domains" do
      for intent <- Brain.Services.LocalScheduler.supported_intents() do
        assert Dispatcher.find_service(intent) == Brain.Services.LocalScheduler, intent
      end
    end

    test "lists the local scheduler among the services" do
      assert Dispatcher.find_service_by_name(:local_scheduler) == Brain.Services.LocalScheduler

      scheduler =
        Dispatcher.list_services(world: "default")
        |> Enum.find(&(&1.module == Brain.Services.LocalScheduler))

      assert scheduler.configured
      assert "alarm.set" in scheduler.supported_intents
    end
  end

  describe "service_schemas/0 Home Assistant domains" do
    test "includes smarthome domain-level schema" do
      schemas = Dispatcher.service_schemas()
      assert Map.has_key?(schemas, "smarthome")
    end

    test "includes device domain-level schema" do
      schemas = Dispatcher.service_schemas()
      assert Map.has_key?(schemas, "device")
    end
  end

  describe "health_check/2" do
    test "returns error for unknown service" do
      assert {:error, :service_not_found} = Dispatcher.health_check(:unknown_service, world: "default")
    end

    test "returns missing_credentials when not configured" do
      result = Dispatcher.health_check(:weather, world: "default")
      assert result == {:error, :missing_credentials}
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/Dispatcher.health_check needs :world/, fn ->
        Dispatcher.health_check(:weather, [])
      end
    end
  end
end
