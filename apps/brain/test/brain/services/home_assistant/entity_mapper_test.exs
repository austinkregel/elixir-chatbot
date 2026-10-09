defmodule Brain.Services.HomeAssistant.EntityMapperTest do
  use ExUnit.Case, async: false

  alias Brain.Services.HomeAssistant.EntityMapper
  alias Brain.Services.HomeAssistant.CapabilityRegistry

  @ets_table :ha_capability_registry

  setup do
    # `CapabilityRegistry` hardcodes `@ets_table` as :ha_capability_registry,
    # the table the running registry owns, and keys each world's capabilities
    # as `{world_id, key}`. The registry reads no world directory, so these
    # tests seed invented entities under fresh `test_` ids and remove every
    # key those worlds hold when the test ends.
    suffix = System.unique_integer([:positive])
    world = "test_entity_mapper_#{suffix}"
    other_world = "test_entity_mapper_other_#{suffix}"

    on_exit(fn ->
      if :ets.whereis(@ets_table) != :undefined do
        for world_id <- [world, other_world] do
          :ets.match_delete(@ets_table, {{world_id, :_}, :_})
        end
      end
    end)

    ensure_ets_table()
    seed_entity_states(world)
    %{world: world, other_world: other_world}
  end

  describe "resolve_entity_id/2 with direct _ha_entity_id" do
    test "uses _ha_entity_id when present in slot map", %{world: world} do
      slots = %{location: "office", _ha_entity_id: "light.office_2"}
      assert EntityMapper.resolve_entity_id(slots, world) == "light.office_2"
    end

    test "uses string-keyed _ha_entity_id", %{world: world} do
      slots = %{"location" => "office", "_ha_entity_id" => "switch.office_fan"}
      assert EntityMapper.resolve_entity_id(slots, world) == "switch.office_fan"
    end
  end

  describe "resolve_entity_id/2 with location slot" do
    test "resolves location to HA entity via registry lookup", %{world: world} do
      slots = %{location: "office"}
      result = EntityMapper.resolve_entity_id(slots, world)
      assert result == "light.office_2"
    end

    test "resolves room slot as location fallback", %{world: world} do
      slots = %{room: "kitchen"}
      result = EntityMapper.resolve_entity_id(slots, world)
      assert result == "light.kitchen_ceiling"
    end

    test "returns nil when no matching area found", %{world: world} do
      slots = %{location: "nonexistent_room"}
      assert EntityMapper.resolve_entity_id(slots, world) == nil
    end
  end

  describe "resolve_entity_id/2 with device slot" do
    test "resolves device name directly", %{world: world} do
      slots = %{device: "garage door"}
      result = EntityMapper.resolve_entity_id(slots, world)
      assert result == "cover.garage_door"
    end
  end

  describe "resolve_entity_id/2 with device + location" do
    test "tries combined name first then falls back to location", %{world: world} do
      slots = %{device: "lights", location: "kitchen"}
      result = EntityMapper.resolve_entity_id(slots, world)
      assert is_binary(result)
      assert result =~ "kitchen"
    end
  end

  describe "resolve_entity_id/2 edge cases" do
    test "returns nil for empty slot map", %{world: world} do
      assert EntityMapper.resolve_entity_id(%{}, world) == nil
    end

    test "returns nil for non-map input", %{world: world} do
      assert EntityMapper.resolve_entity_id(nil, world) == nil
      assert EntityMapper.resolve_entity_id("string", world) == nil
    end

    test "returns nil for entity list with no device-type entity", %{world: world} do
      entities = [%{type: "city", name: "Owosso"}]
      assert EntityMapper.resolve_entity_id(entities, world) == nil
    end
  end

  describe "resolve_entity_id/2 world scoping" do
    test "does not resolve against another world's entities", %{other_world: other_world} do
      assert EntityMapper.resolve_entity_id(%{location: "office"}, other_world) == nil
      assert EntityMapper.resolve_entity_id(%{device: "garage door"}, other_world) == nil
    end

    test "each world resolves against its own entities", %{world: world, other_world: other_world} do
      seed_states(other_world, [ha_entity("light.office_lamp", "Office Lamp", "off")])

      assert EntityMapper.resolve_entity_id(%{location: "office"}, world) == "light.office_2"
      assert EntityMapper.resolve_entity_id(%{location: "office"}, other_world) == "light.office_lamp"
    end

    test "raises when a registry lookup is needed and no world is named" do
      assert_raise ArgumentError, ~r/needs the world whose Home Assistant capabilities are read, got: nil/, fn ->
        EntityMapper.resolve_entity_id(%{location: "office"}, nil)
      end
    end
  end

  describe "find_entities_for_area/2 domain priority" do
    test "lights sort before switches and sensors", %{world: world} do
      results = CapabilityRegistry.find_entities_for_area(world, "office")
      assert length(results) >= 2

      domains = Enum.map(results, & &1.ha_domain)
      light_idx = Enum.find_index(domains, &(&1 == "light"))
      switch_idx = Enum.find_index(domains, &(&1 == "switch"))
      sensor_idx = Enum.find_index(domains, &(&1 == "sensor"))

      if light_idx && switch_idx, do: assert(light_idx < switch_idx)
      if light_idx && sensor_idx, do: assert(light_idx < sensor_idx)
    end

    test "returns empty list for unknown area", %{world: world} do
      assert CapabilityRegistry.find_entities_for_area(world, "mars_colony") == []
    end

    test "returns empty list for nil", %{world: world} do
      assert CapabilityRegistry.find_entities_for_area(world, nil) == []
    end

    test "matching is case-insensitive", %{world: world} do
      results_lower = CapabilityRegistry.find_entities_for_area(world, "office")
      results_upper = CapabilityRegistry.find_entities_for_area(world, "Office")
      assert results_lower == results_upper
    end
  end

  describe "CapabilityRegistry world isolation" do
    test "one world's entities are not visible from another", %{world: world, other_world: other_world} do
      assert CapabilityRegistry.find_entities_for_area(world, "office") != []
      assert CapabilityRegistry.entity_states(world) != []
      assert CapabilityRegistry.entities_by_domain(world) != %{}

      assert CapabilityRegistry.find_entities_for_area(other_world, "office") == []
      assert CapabilityRegistry.entity_states(other_world) == []
      assert CapabilityRegistry.entities_by_domain(other_world) == %{}
      assert CapabilityRegistry.domain_entities(other_world, "light") == []
    end

    test "domain entities are read from the named world", %{world: world, other_world: other_world} do
      seed_states(other_world, [ha_entity("light.office_lamp", "Office Lamp", "off")])

      assert Enum.sort(CapabilityRegistry.domain_entities(world, "light")) ==
               ["light.kitchen_ceiling", "light.office_2"]

      assert CapabilityRegistry.domain_entities(other_world, "light") == ["light.office_lamp"]
    end

    test "one world's services are not visible from another", %{world: world, other_world: other_world} do
      :ets.insert(
        @ets_table,
        {{world, :services}, %{"light" => %{"turn_on" => %{"fields" => %{"brightness_pct" => %{}}}}}}
      )

      assert CapabilityRegistry.can_handle?(world, "light", "turn_on")
      assert CapabilityRegistry.service_fields(world, "light", "turn_on") == ["brightness_pct"]
      assert Map.has_key?(CapabilityRegistry.available_services(world), "light")

      refute CapabilityRegistry.can_handle?(other_world, "light", "turn_on")
      assert CapabilityRegistry.service_fields(other_world, "light", "turn_on") == []
      assert CapabilityRegistry.available_services(other_world) == %{}
    end

    test "readiness is held per world", %{world: world, other_world: other_world} do
      assert CapabilityRegistry.ready?(world)
      refute CapabilityRegistry.ready?(other_world)
    end
  end

  describe "CapabilityRegistry reads without a world" do
    test "each read raises" do
      for world_id <- [nil, ""] do
        assert_raise ArgumentError, ~r/CapabilityRegistry.ready\? needs the world/, fn ->
          CapabilityRegistry.ready?(world_id)
        end

        assert_raise ArgumentError, ~r/CapabilityRegistry.available_services needs the world/, fn ->
          CapabilityRegistry.available_services(world_id)
        end

        assert_raise ArgumentError, ~r/CapabilityRegistry.entities_by_domain needs the world/, fn ->
          CapabilityRegistry.entities_by_domain(world_id)
        end

        assert_raise ArgumentError, ~r/CapabilityRegistry.entity_states needs the world/, fn ->
          CapabilityRegistry.entity_states(world_id)
        end

        assert_raise ArgumentError, ~r/needs the world whose Home Assistant capabilities are read/, fn ->
          CapabilityRegistry.can_handle?(world_id, "light", "turn_on")
        end

        assert_raise ArgumentError, ~r/needs the world whose Home Assistant capabilities are read/, fn ->
          CapabilityRegistry.service_fields(world_id, "light", "turn_on")
        end

        assert_raise ArgumentError, ~r/needs the world whose Home Assistant capabilities are read/, fn ->
          CapabilityRegistry.domain_entities(world_id, "light")
        end

        assert_raise ArgumentError, ~r/needs the world whose Home Assistant capabilities are read/, fn ->
          CapabilityRegistry.find_entities_for_area(world_id, "office")
        end

        assert_raise ArgumentError, ~r/CapabilityRegistry.find_entities_for_area needs the world/, fn ->
          CapabilityRegistry.find_entities_for_area(world_id, nil)
        end
      end
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

      _ref ->
        :ok
    end
  end

  defp seed_entity_states(world_id) do
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

    seed_states(world_id, states)
  end

  # Writes the keys a completed sync leaves for a world.
  defp seed_states(world_id, states) do
    :ets.insert(@ets_table, {{world_id, :entity_states}, states})

    by_domain =
      Enum.group_by(states, fn s ->
        s["entity_id"] |> String.split(".", parts: 2) |> hd()
      end)
      |> Map.new(fn {d, ents} -> {d, Enum.map(ents, & &1["entity_id"])} end)

    :ets.insert(@ets_table, {{world_id, :entities_by_domain}, by_domain})
    :ets.insert(@ets_table, {{world_id, :ready}, true})
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
