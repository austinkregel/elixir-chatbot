defmodule Brain.Services.DispatcherTest do
  use ExUnit.Case, async: false

  alias Brain.Services.{CredentialVault, Dispatcher, HomeAssistant, LocalScheduler, SchedulingDriver}

  describe "find_service/2" do
    test "returns Weather service for weather.query intent" do
      assert Dispatcher.find_service("weather.query", "default") == Brain.Services.Weather
    end

    test "returns Weather service for weather.forecast intent" do
      assert Dispatcher.find_service("weather.forecast", "default") == Brain.Services.Weather
    end

    test "returns nil for bare weather intent without dot" do
      # "weather" alone is not a dotted intent, so domain-prefix fallback doesn't apply
      assert Dispatcher.find_service("weather", "default") == nil
    end

    test "returns Weather service for weather.current intent" do
      assert Dispatcher.find_service("weather.current", "default") == Brain.Services.Weather
    end

    test "returns nil for unknown intent" do
      assert Dispatcher.find_service("unknown.intent", "default") == nil
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/Dispatcher.find_service needs the world/, fn ->
        Dispatcher.find_service("weather.query", nil)
      end

      assert_raise ArgumentError, ~r/Dispatcher.find_service needs the world/, fn ->
        Dispatcher.find_service("weather.query", "")
      end
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

  describe "find_service/2 domain prefix fallback" do
    test "finds Weather service for weather.request_information via domain prefix" do
      assert Dispatcher.find_service("weather.request_information", "default") == Brain.Services.Weather
    end

    test "finds Weather service for weather.some_other_subtype via domain prefix" do
      assert Dispatcher.find_service("weather.some_other_subtype", "default") == Brain.Services.Weather
    end

    test "returns nil for completely unrelated domain prefix" do
      assert Dispatcher.find_service("banking.transfer", "default") == nil
    end

    test "prefers exact match over domain prefix" do
      assert Dispatcher.find_service("weather.query", "default") == Brain.Services.Weather
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

  describe "find_service/2 Home Assistant routing" do
    test "routes smarthome.switch to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("smarthome.switch", "default") == HomeAssistant
    end

    test "routes smarthome.query to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("smarthome.query", "default") == HomeAssistant
    end

    test "routes device.control to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("device.control", "default") == HomeAssistant
    end

    test "routes music.play to HomeAssistant via supported_domains" do
      assert Dispatcher.find_service("music.play", "default") == HomeAssistant
    end
  end

  describe "the local scheduler as a service" do
    test "is listed among the services, configured with no credentials, with the scheduling intents" do
      assert Dispatcher.find_service_by_name(:local_scheduler) == LocalScheduler

      scheduler =
        Dispatcher.list_services(world: "default")
        |> Enum.find(&(&1.module == LocalScheduler))

      assert scheduler.configured
      assert "alarm.set" in scheduler.supported_intents
    end
  end

  # The alarm, timer, reminder and calendar domains go to the scheduling
  # driver each world has chosen. Three worlds: one on the local scheduler,
  # one on Home Assistant with its credentials set, and one with no driver.
  # The driver rows and credential rows are written inside this test's
  # sandbox transaction; the credentials the vault holds in memory are
  # deleted when the test ends.
  describe "scheduling routing by the world's driver" do
    setup tags do
      owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
      on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

      suffix = System.unique_integer([:positive])
      local_world = "test_dispatch_local_#{suffix}"
      ha_world = "test_dispatch_home_assistant_#{suffix}"
      no_driver_world = "test_dispatch_no_driver_#{suffix}"

      on_exit(fn ->
        LocalScheduler.clear_world(local_world)
        :ok = CredentialVault.delete(:home_assistant, :url, world: ha_world)
        :ok = CredentialVault.delete(:home_assistant, :access_token, world: ha_world)
      end)

      :ok = Dispatcher.assign_scheduling_driver(local_world, LocalScheduler)

      :ok = CredentialVault.store(:home_assistant, :url, "http://home-assistant.invalid", world: ha_world)
      :ok = CredentialVault.store(:home_assistant, :access_token, "unused", world: ha_world)
      :ok = Dispatcher.assign_scheduling_driver(ha_world, HomeAssistant)

      %{local_world: local_world, ha_world: ha_world, no_driver_world: no_driver_world}
    end

    test "the same intent goes to each world's own driver", %{local_world: local_world, ha_world: ha_world} do
      for intent <- SchedulingDriver.intents() do
        assert Dispatcher.find_service(intent, local_world) == LocalScheduler, intent
        assert Dispatcher.find_service(intent, ha_world) == HomeAssistant, intent
      end
    end

    test "dispatching the same intent reaches a different driver in each world", %{
      local_world: local_world,
      ha_world: ha_world
    } do
      # With side effects off, Home Assistant refuses a snooze, which calls a
      # service; the local scheduler, which never rings an alarm, reports
      # that there is none to snooze.
      context = %{side_effects: false}

      assert {:error, :no_ringing_alarm} =
               Dispatcher.dispatch("alarm.snooze", %{}, Map.put(context, :world_id, local_world))

      assert {:error, :side_effects_off} =
               Dispatcher.dispatch("alarm.snooze", %{}, Map.put(context, :world_id, ha_world))
    end

    test "dispatching to the local driver performs the operation the intent names", %{local_world: local_world} do
      assert {:ok, %{timer_status: "running", duration: "6 minutes"}} =
               Dispatcher.dispatch("timer.set", %{duration: "6 minutes"}, %{world_id: local_world})

      assert {:ok, %{timer_status: "running", duration: "6 minutes"}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: local_world})
    end

    test "a world with no driver raises, naming the world and the capability", %{no_driver_world: world} do
      message = ~r/world "#{world}" has no driver for the "scheduling" capability/

      assert_raise RuntimeError, message, fn -> Dispatcher.find_service("alarm.set", world) end
      assert_raise RuntimeError, message, fn -> Dispatcher.dispatch("timer.check", %{}, %{world_id: world}) end
      assert_raise RuntimeError, message, fn -> Dispatcher.scheduling_driver!(world) end
      assert Dispatcher.scheduling_driver(world) == :none
    end

    test "a world with no driver still routes the intents of other domains", %{no_driver_world: world} do
      assert Dispatcher.find_service("weather.query", world) == Brain.Services.Weather
      assert Dispatcher.find_service("smarthome.switch", world) == HomeAssistant
    end

    test "an intent of a scheduling domain that names no operation has no handler", %{local_world: local_world} do
      assert Dispatcher.find_service("alarm.ring", local_world) == nil
      assert Dispatcher.find_service("reminder.set", local_world) == nil
      assert Dispatcher.dispatch("alarm.set.recurring", %{}, %{world_id: local_world}) == :no_handler
    end

    test "Home Assistant cannot be chosen until its credentials are set for the world, and nothing is stored",
         %{no_driver_world: world} do
      assert {:error, {:missing_credentials, [:url, :access_token]}} =
               Dispatcher.assign_scheduling_driver(world, HomeAssistant)

      assert Dispatcher.scheduling_driver(world) == :none
      assert Dispatcher.missing_credentials(HomeAssistant, world) == [:url, :access_token]
      assert Dispatcher.missing_credentials(LocalScheduler, world) == []
    end

    test "choosing again replaces the world's driver", %{ha_world: ha_world} do
      assert Dispatcher.scheduling_driver(ha_world) == {:ok, HomeAssistant}
      :ok = Dispatcher.assign_scheduling_driver(ha_world, LocalScheduler)
      assert Dispatcher.scheduling_driver(ha_world) == {:ok, LocalScheduler}
      assert Dispatcher.find_service("alarm.set", ha_world) == LocalScheduler
    end

    test "only a scheduling driver can be chosen, and only by a driver's name", %{no_driver_world: world} do
      assert_raise ArgumentError, ~r/is not a scheduling driver/, fn ->
        Dispatcher.assign_scheduling_driver(world, Brain.Services.Weather)
      end

      assert_raise ArgumentError, ~r/no scheduling driver is named "weather"/, fn ->
        Dispatcher.scheduling_driver_named!("weather")
      end

      assert Dispatcher.scheduling_driver_named!("local_scheduler") == LocalScheduler
      assert Dispatcher.scheduling_driver_named!("home_assistant") == HomeAssistant
    end

    test "a stored driver name that is no scheduling driver's raises", %{no_driver_world: world} do
      Atlas.WorldServiceDrivers.put!(world, "scheduling", "pager")

      assert_raise ArgumentError, ~r/no scheduling driver is named "pager"/, fn ->
        Dispatcher.find_service("alarm.set", world)
      end
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

  describe "service_schemas/0 scheduling domains" do
    test "each scheduling domain carries the slot schema of the drivers that declare one" do
      schemas = Dispatcher.service_schemas()

      for domain <- SchedulingDriver.domains() do
        assert schemas[domain] == HomeAssistant.slot_schema(), domain
        assert "time" in schemas[domain]["optional"], domain
        assert "duration" in schemas[domain]["entity_mappings"]["duration"], domain
      end
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
