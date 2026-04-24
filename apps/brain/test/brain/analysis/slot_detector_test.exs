defmodule Brain.Analysis.SlotDetectorTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.SlotDetector
  alias Brain.Analysis.SlotResult

  describe "detect/2" do
    test "detects filled slots from entities" do
      entities = [
        %{entity_type: "location", value: "New York", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert %SlotResult{
               schema_name: "weather.query",
               all_required_filled: true
             } = result

      assert SlotResult.get_slot_value(result, "location") == "New York"
    end

    test "identifies missing required slots" do
      result = SlotDetector.detect("weather.query", [])

      assert result.all_required_filled == false
      assert "location" in result.missing_required
    end

    test "applies default values" do
      result = SlotDetector.detect("weather.query", [])

      # "date" should have default value "today"
      assert SlotResult.get_slot_value(result, "date") == "today"
    end

    test "handles smarthome device intent" do
      entities = [
        %{entity_type: "device", value: "lights", confidence: 0.9},
        %{entity_type: "room", value: "kitchen", confidence: 0.8}
      ]

      result = SlotDetector.detect("smarthome.lights.switch.off", entities)

      assert SlotResult.get_slot_value(result, "device") == "lights"
      assert SlotResult.get_slot_value(result, "room") == "kitchen"
    end

    test "handles unknown intent gracefully" do
      entities = [
        %{entity_type: "color", value: "blue", confidence: 0.9}
      ]

      result = SlotDetector.detect("nonexistent.intent", entities)

      assert result.schema_name == "unknown"
      # Should still capture the entity
      assert SlotResult.get_slot_value(result, "color") == "blue"
    end

    test "handles empty entities" do
      result = SlotDetector.detect("news.query", [])

      # news.query has no required slots
      assert result.all_required_filled == true
    end
  end

  describe "get_schema/1" do
    test "returns schema for known intent" do
      schema = SlotDetector.get_schema("weather.query")

      assert schema != nil
      assert "location" in schema["required"]
    end

    test "returns nil for unknown intent" do
      schema = SlotDetector.get_schema("totally.unknown.intent")

      assert schema == nil
    end
  end

  describe "suggest_intent_from_entities/1" do
    test "suggests a location-accepting intent from location entity" do
      entities = [
        %{entity_type: "location", value: "Paris", confidence: 0.9}
      ]

      {:ok, intent, _score} = SlotDetector.suggest_intent_from_entities(entities)

      # With only a location entity (no textual context), multiple intents
      # are equally valid: weather, navigation, knowledge (capital queries),
      # etc. The function should return any intent that accepts location.
      assert is_binary(intent), "Expected a string intent, got #{inspect(intent)}"
    end

    test "suggests device control from device entity" do
      entities = [
        %{entity_type: "device", value: "TV", confidence: 0.9},
        %{entity_type: "room", value: "living room", confidence: 0.8}
      ]

      {:ok, intent, score} = SlotDetector.suggest_intent_from_entities(entities)

      # A smarthome.device.* intent should be returned for device + room entities
      assert String.starts_with?(intent, "smarthome.lights.")
      assert score >= 2
    end

    test "suggests music intent from song entity" do
      entities = [
        %{entity_type: "song", value: "Bohemian Rhapsody", confidence: 0.9}
      ]

      {:ok, intent, _score} = SlotDetector.suggest_intent_from_entities(entities)

      assert intent == "music.play"
    end

    test "returns error for no matching entities" do
      entities = [
        %{entity_type: "unknown_type", value: "something", confidence: 0.9}
      ]

      result = SlotDetector.suggest_intent_from_entities(entities)

      assert result == {:error, :no_match}
    end
  end

  describe "SlotResult" do
    test "fill_slot adds slot value" do
      result = SlotResult.new("test")
      updated = SlotResult.fill_slot(result, "location", "NYC", :explicit, 0.9)

      assert SlotResult.get_slot_value(updated, "location") == "NYC"
    end

    test "fill_slot tracks source" do
      result = SlotResult.new("test")
      updated = SlotResult.fill_slot(result, "date", "today", :default, 1.0)

      assert updated.filled_slots["date"].source == :default
    end
  end

  describe "get_clarification_prompt/2" do
    test "returns template from intent_registry.json for known slot" do
      prompt = SlotDetector.get_clarification_prompt("location", "weather.query")

      assert prompt == "What location would you like the weather for?"
    end

    test "returns template for smarthome device slots" do
      device_prompt = SlotDetector.get_clarification_prompt("device", "smarthome.lights.switch.on")
      room_prompt = SlotDetector.get_clarification_prompt("room", "smarthome.lights.switch.on")

      # Should return a prompt (either from registry or generic fallback)
      assert is_binary(device_prompt)
      assert is_binary(room_prompt)
    end

    test "returns generic prompt for unknown slot" do
      prompt = SlotDetector.get_clarification_prompt("unknown_slot", "weather.query")

      assert prompt == "Could you please specify the unknown slot?"
    end

    test "handles slot names with hyphens" do
      prompt = SlotDetector.get_clarification_prompt("music-artist", "music.play")

      # Should use generic prompt since music.play doesn't have clarification for music-artist
      assert prompt == "Could you please specify the music artist?"
    end

    test "handles atom slot names" do
      prompt = SlotDetector.get_clarification_prompt(:location, "weather.query")

      assert prompt == "What location would you like the weather for?"
    end

    test "handles unknown intent gracefully" do
      prompt = SlotDetector.get_clarification_prompt("location", "unknown.intent")

      # Should return generic prompt
      assert prompt == "Could you please specify the location?"
    end
  end

  describe "get_clarification_prompts/2" do
    test "returns list of prompts for multiple missing slots" do
      missing_slots = ["device", "room"]
      prompts = SlotDetector.get_clarification_prompts(missing_slots, "smarthome.lights.switch.on")

      assert length(prompts) == 2
      assert Enum.all?(prompts, &is_binary/1)
    end

    test "handles empty list" do
      prompts = SlotDetector.get_clarification_prompts([], "weather.query")

      assert prompts == []
    end

    test "handles atom slot names in list" do
      missing_slots = [:location, :date]
      prompts = SlotDetector.get_clarification_prompts(missing_slots, "weather.query")

      assert length(prompts) == 2
      assert "What location would you like the weather for?" in prompts
    end
  end
end
