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

    test "returns nil for bare weather intent" do
      # Bare "weather" intent was removed; only specific weather.* intents are supported
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
