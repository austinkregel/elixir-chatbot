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
      services = Dispatcher.list_services()

      assert is_list(services)
      assert length(services) >= 1

      weather = Enum.find(services, &(&1.name == :weather))
      assert weather != nil
      assert weather.display_name == "Weather"
      assert weather.module == Brain.Services.Weather
      assert "weather.query" in weather.supported_intents
    end

    test "includes configuration status" do
      services = Dispatcher.list_services()
      weather = Enum.find(services, &(&1.name == :weather))

      assert Map.has_key?(weather, :configured)
      assert is_boolean(weather.configured)
    end
  end

  describe "dispatch/3" do
    test "returns :no_handler for unknown intent" do
      assert Dispatcher.dispatch("unknown.intent", %{}, %{}) == :no_handler
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
      refute Dispatcher.service_available?(:weather)
    end

    test "returns false for unknown service" do
      refute Dispatcher.service_available?(:unknown_service)
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

  describe "health_check/2" do
    test "returns error for unknown service" do
      assert {:error, :service_not_found} = Dispatcher.health_check(:unknown_service)
    end

    test "returns missing_credentials when not configured" do
      result = Dispatcher.health_check(:weather)
      assert result == {:error, :missing_credentials}
    end
  end
end
