defmodule Brain.Services.HomeAssistant.EntityMapperTest do
  use ExUnit.Case, async: false

  alias Brain.Services.HomeAssistant.EntityMapper
  alias Brain.Services.HomeAssistant.CapabilityRegistry

  @ets_table :ha_capability_registry

  setup do
    ensure_ets_table()
    seed_entity_states()
    :ok
  end

  describe "resolve_entity_id/1 with direct _ha_entity_id" do
    test "uses _ha_entity_id when present in slot map" do
      slots = %{location: "office", _ha_entity_id: "light.office_2"}
      assert EntityMapper.resolve_entity_id(slots) == "light.office_2"
    end

    test "uses string-keyed _ha_entity_id" do
      slots = %{"location" => "office", "_ha_entity_id" => "switch.office_fan"}
      assert EntityMapper.resolve_entity_id(slots) == "switch.office_fan"
    end
  end

  describe "resolve_entity_id/1 with location slot" do
    test "resolves location to HA entity via registry lookup" do
      slots = %{location: "office"}
      result = EntityMapper.resolve_entity_id(slots)
      assert result == "light.office_2"
    end

    test "resolves room slot as location fallback" do
      slots = %{room: "kitchen"}
      result = EntityMapper.resolve_entity_id(slots)
      assert result == "light.kitchen_ceiling"
    end

    test "returns nil when no matching area found" do
      slots = %{location: "nonexistent_room"}
      assert EntityMapper.resolve_entity_id(slots) == nil
    end
  end

  describe "resolve_entity_id/1 with device slot" do
    test "resolves device name directly" do
      slots = %{device: "garage door"}
      result = EntityMapper.resolve_entity_id(slots)
      assert result == "cover.garage_door"
    end
  end

  describe "resolve_entity_id/1 with device + location" do
    test "tries combined name first then falls back to location" do
      slots = %{device: "lights", location: "kitchen"}
      result = EntityMapper.resolve_entity_id(slots)
      assert is_binary(result)
      assert result =~ "kitchen"
    end
  end

  describe "resolve_entity_id/1 edge cases" do
    test "returns nil for empty slot map" do
      assert EntityMapper.resolve_entity_id(%{}) == nil
    end

    test "returns nil for non-map input" do
      assert EntityMapper.resolve_entity_id(nil) == nil
      assert EntityMapper.resolve_entity_id("string") == nil
    end

    test "returns nil for entity list with no device-type entity" do
      entities = [%{type: "city", name: "Owosso"}]
      assert EntityMapper.resolve_entity_id(entities) == nil
    end
  end

  describe "find_entities_for_area/1 domain priority" do
    test "lights sort before switches and sensors" do
      results = CapabilityRegistry.find_entities_for_area("office")
      assert Enum.count_until(results, 2) >= 2

      domains = Enum.map(results, & &1.ha_domain)
      light_idx = Enum.find_index(domains, &(&1 == "light"))
      switch_idx = Enum.find_index(domains, &(&1 == "switch"))
      sensor_idx = Enum.find_index(domains, &(&1 == "sensor"))

      if light_idx && switch_idx, do: assert(light_idx < switch_idx)
      if light_idx && sensor_idx, do: assert(light_idx < sensor_idx)
    end

    test "returns empty list for unknown area" do
      assert CapabilityRegistry.find_entities_for_area("mars_colony") == []
    end

    test "returns empty list for nil" do
      assert CapabilityRegistry.find_entities_for_area(nil) == []
    end

    test "matching is case-insensitive" do
      results_lower = CapabilityRegistry.find_entities_for_area("office")
      results_upper = CapabilityRegistry.find_entities_for_area("Office")
      assert results_lower == results_upper
    end
  end

  describe "build_service_data/1" do
    test "extracts brightness from slots" do
      data = EntityMapper.build_service_data(%{brightness: 75})
      assert data["brightness_pct"] == 75
    end

    test "extracts color from slots" do
      data = EntityMapper.build_service_data(%{color: "red"})
      assert data["color_name"] == "red"
    end

    test "extracts volume and normalizes to 0-1 range" do
      data = EntityMapper.build_service_data(%{volume: 50})
      assert data["volume_level"] == 0.5
    end

    test "returns empty map when no relevant slots" do
      assert EntityMapper.build_service_data(%{location: "office"}) == %{}
    end

    test "returns empty map for non-map input" do
      assert EntityMapper.build_service_data(nil) == %{}
    end
  end

  describe "ha_entity_to_brain/1" do
    test "converts HA state to Brain entity map" do
      ha_state = %{
        "entity_id" => "light.office_2",
        "state" => "on",
        "attributes" => %{"friendly_name" => "Office Light", "brightness" => 255}
      }

      brain = EntityMapper.ha_entity_to_brain(ha_state)
      assert brain.name == "Office Light"
      assert brain.ha_entity_id == "light.office_2"
      assert brain.ha_domain == "light"
      assert brain.state == "on"
      assert is_map(brain.attributes)
    end
  end

  describe "name_to_entity_id/1" do
    test "converts friendly name to entity_id format" do
      assert EntityMapper.name_to_entity_id("Office Light") == "office_light"
    end

    test "strips special characters" do
      assert EntityMapper.name_to_entity_id("Living Room (Main)") == "living_room_main"
    end

    test "returns nil for non-string" do
      assert EntityMapper.name_to_entity_id(nil) == nil
    end
  end

  # ---- Test infrastructure ----

  defp ensure_ets_table do
    case :ets.whereis(@ets_table) do
      :undefined ->
        :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])
        :ets.insert(@ets_table, {:ready, true})

      _ref ->
        :ets.insert(@ets_table, {:ready, true})
    end
  end

  defp seed_entity_states do
    states = [
      ha_entity("light.office_2", "Office Light", "on"),
      ha_entity("switch.office_outlet", "Office Outlet", "off"),
      ha_entity("sensor.office_temperature", "Office Temperature", "72.4"),
      ha_entity("light.kitchen_ceiling", "Kitchen Ceiling", "on"),
      ha_entity("switch.kitchen_counter", "Kitchen Counter", "on"),
      ha_entity("cover.garage_door", "Garage Door", "closed"),
      ha_entity("fan.bedroom_fan", "Bedroom Fan", "off"),
      ha_entity("climate.living_room", "Living Room Thermostat", "heat"),
      ha_entity("media_player.living_room_tv", "Living Room TV", "idle"),
      ha_entity("zone.home", "Home", "zoning"),
      ha_entity("zone.office", "Office", "zoning")
    ]

    :ets.insert(@ets_table, {:entity_states, states})

    by_domain =
      Enum.group_by(states, fn s ->
        s["entity_id"] |> String.split(".", parts: 2) |> hd()
      end)
      |> Map.new(fn {d, ents} -> {d, Enum.map(ents, & &1["entity_id"])} end)

    :ets.insert(@ets_table, {:entities_by_domain, by_domain})
  end

  defp ha_entity(entity_id, friendly_name, state, extra_attrs \\ %{}) do
    %{
      "entity_id" => entity_id,
      "state" => state,
      "attributes" => Map.merge(%{"friendly_name" => friendly_name}, extra_attrs),
      "last_changed" => "2026-05-01T00:00:00Z",
      "last_updated" => "2026-05-01T00:00:00Z"
    }
  end
end
