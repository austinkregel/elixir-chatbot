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

      # "room" is an entity type, not a slot: no schema in
      # priv/analysis/intent_registry.json declares a "room" slot, and every
      # smarthome schema there (smarthome.switch, smarthome.set,
      # smarthome.device_set) maps a "room" entity onto its "location" slot,
      # as does the Home Assistant service schema
      # (Brain.Services.HomeAssistant.slot_schema/0) that answers for the
      # unregistered "smarthome.lights.switch.off" through its "smarthome"
      # parent.
      assert SlotResult.get_slot_value(result, "device") == "lights"
      assert SlotResult.get_slot_value(result, "location") == "kitchen"
    end

    test "web.search's query is text a search is run for: a number does not fill it" do
      # Every gold q of the NER gold standard is a text query ("how to build
      # a house", "Monty Python"), so q is information, not a number.
      number = [%{entity_type: "number", value: "40", confidence: 0.9}]
      refute SlotResult.get_slot_value(SlotDetector.detect("web.search", number, side_effects: false), "q")

      query = [%{entity_type: "q", value: "apple pie recipes", confidence: 0.9}]
      assert SlotResult.get_slot_value(SlotDetector.detect("web.search", query, side_effects: false), "q") == "apple pie recipes"
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

  describe "service schema merge" do
    test "GPE entity fills location slot for weather.query via service schema" do
      entities = [
        %{entity_type: "gpe", value: "Owosso", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert result.schema_name == "weather.query"
      assert result.all_required_filled == true
      assert SlotResult.get_slot_value(result, "location") == "Owosso"
    end

    test "city entity fills location slot for weather.query" do
      entities = [
        %{entity_type: "city", value: "Portland", confidence: 0.85}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert result.all_required_filled == true
      assert SlotResult.get_slot_value(result, "location") == "Portland"
    end

    test "service schema merges with intent_registry defaults" do
      result = SlotDetector.detect("weather.query", [])

      assert result.all_required_filled == false
      assert "location" in result.missing_required
      assert SlotResult.get_slot_value(result, "date") == "today"
    end

    test "service schema adds gpe to entity_mappings for weather.query" do
      schema = SlotDetector.get_schema("weather.query")
      mappings = schema["entity_mappings"]["location"]

      assert "gpe" in mappings
      assert "location" in mappings
      assert "city" in mappings
    end

    test "weather.forecast uses same service schema" do
      entities = [
        %{entity_type: "gpe", value: "Chicago", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.forecast", entities)

      assert result.all_required_filled == true
      assert SlotResult.get_slot_value(result, "location") == "Chicago"
    end

    test "clarification prompt from service schema" do
      prompt = SlotDetector.get_clarification_prompt("location", "weather.forecast")

      assert prompt == "What location would you like the weather for?"
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

    test "reads the intent registry regardless of the working directory" do
      # mix test runs from apps/brain, where a path relative to the working
      # directory happens to resolve. The app and the axis probe run from the
      # umbrella root, where it does not.
      original = File.cwd!()
      on_exit(fn -> File.cd!(original) end)
      File.cd!(Path.expand("../..", original))

      schema = SlotDetector.get_schema("music.play")

      assert "music_artist" in schema["optional"]
    end
  end

  describe "detect/2 for music.play" do
    test "fills the artist slot from a music_artist entity" do
      entities = [%{entity_type: "music_artist", value: "The Beatles", confidence: 0.9}]

      result = SlotDetector.detect("music.play", entities)

      assert SlotResult.get_slot_value(result, "music_artist") == "The Beatles"
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

      # An intent of the registry whose slots take both a device and a room.
      # (The registry holds no smarthome.lights.* intent: the gold standard's
      # rebuild folded them into smarthome.device.*. A service's domain schema,
      # "device" or "smarthome", is no intent.)
      schema = Map.fetch!(SlotDetector.registry(), intent)
      types = schema |> Map.fetch!("entity_mappings") |> Map.values() |> List.flatten()
      assert "device" in types and "room" in types, "#{intent}: #{inspect(schema["entity_mappings"])}"
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

    test "handles compound slot names" do
      prompt = SlotDetector.get_clarification_prompt("music_artist", "music.play")

      # Should use generic prompt since music.play doesn't have clarification for music_artist
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

  describe "domain-prefix fallback" do
    test "weather.request_information fills location slot via domain prefix schema" do
      entities = [
        %{entity_type: "gpe", value: "Owosso", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.request_information", entities)

      assert SlotResult.get_slot_value(result, "location") == "Owosso"
    end

    test "weather.request_information fills location from city entity type" do
      entities = [
        %{entity_type: "city", value: "Detroit", confidence: 0.85}
      ]

      result = SlotDetector.detect("weather.request_information", entities)

      assert SlotResult.get_slot_value(result, "location") == "Detroit"
    end

    test "weather.request_information fills location from location entity type" do
      entities = [
        %{entity_type: "location", value: "Chicago", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.request_information", entities)

      assert SlotResult.get_slot_value(result, "location") == "Chicago"
    end

    test "completely unknown domain falls through to build_unknown_result" do
      entities = [
        %{entity_type: "widget", value: "Foo", confidence: 0.5}
      ]

      result = SlotDetector.detect("banking.transfer", entities)

      assert result.schema_name == "unknown"
    end
  end

  describe "hierarchy-aware slot filling" do
    test "setting entity fills location slot via TypeHierarchy compatibility" do
      entities = [
        %{entity_type: "setting", value: "Owosso", confidence: 0.8}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert SlotResult.get_slot_value(result, "location") == "Owosso"
    end

    test "place_name entity fills location slot via TypeHierarchy compatibility" do
      entities = [
        %{entity_type: "place_name", value: "Grand Rapids", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert SlotResult.get_slot_value(result, "location") == "Grand Rapids"
    end

    test "exact match is still preferred over hierarchy fallback" do
      entities = [
        %{entity_type: "city", value: "Detroit", confidence: 0.85}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert SlotResult.get_slot_value(result, "location") == "Detroit"
    end

    test "unrelated entity type does not fill location slot" do
      entities = [
        %{entity_type: "person", value: "John", confidence: 0.9}
      ]

      result = SlotDetector.detect("weather.query", entities)

      assert SlotResult.get_slot_value(result, "location") == nil
    end
  end

  # Entities are the spans extraction decided; the spans it could not decide
  # never reach SlotDetector (`Brain.ML.EntityExtractor.extract/2` keeps them
  # in `:unresolved_spans`). An entity is typed, or detection raises.
  describe "an entity with no type" do
    test "a decided entity with no type raises" do
      assert_raise ArgumentError, ~r/no type cannot fill a slot/, fn ->
        SlotDetector.detect("weather.query", [%{entity_type: nil, value: "x", confidence: 0.9}])
      end
    end

    test "raises beside typed entities, rather than slotting as nil or \"\"" do
      typeless = %{entity_type: nil, value: "Prince", confidence: 0.9, resolved: true}
      decided = %{entity_type: "location", value: "Ibiza", confidence: 0.9, resolved: true}

      assert_raise ArgumentError, ~r/no type cannot fill a slot/, fn ->
        SlotDetector.detect("weather.query", [decided, typeless])
      end
    end

    test "raises for an intent with no schema too" do
      assert_raise ArgumentError, ~r/no type cannot fill a slot/, fn ->
        SlotDetector.detect("zzz_no_such_domain.intent", [%{entity_type: nil, value: "Prince", confidence: 0.9}])
      end
    end
  end

  # A slot fills from an entity of a type it lists, or of one related to it in
  # the type hierarchy of priv/analysis/entity_types.json. A listed type the
  # hierarchy does not declare is related to nothing, so no entity of another
  # type can fill the slot, and the gold labels cannot be compared with it.
  describe "the intent registry's slot types" do
    test "every type a registry slot lists is declared in the entity type hierarchy" do
      hierarchy =
        "analysis/entity_types.json"
        |> Brain.priv_path()
        |> File.read!()
        |> Jason.decode!()
        |> Map.fetch!("type_hierarchy")

      declared = hierarchy |> Enum.flat_map(fn {parent, children} -> [parent | children] end) |> MapSet.new()

      undeclared =
        for {intent, schema} <- SlotDetector.registry(),
            {slot, types} <- Map.get(schema, "entity_mappings", %{}),
            type <- types,
            not MapSet.member?(declared, type) do
          "#{intent}.#{slot}: #{type}"
        end

      assert undeclared == [], "registry slots list types entity_types.json does not declare:\n" <> Enum.join(Enum.sort(undeclared), "\n")
    end
  end

  # What fills a slot is decided by what the slot's value is used for.
  # Brain.Services.LocalScheduler runs a timer for the length its duration
  # names and raises for a duration that names none, and stores a calendar
  # item's event as the item's text, which a cancellation is matched against.
  describe "slots filled for the scheduler" do
    test "a length fills a timer's and a snooze's duration, with a length the scheduler runs a timer for" do
      alias Brain.Temporal.{Resolution, Resolver, Zone}

      turn = ~U[2026-10-12 09:00:00Z]

      for {text, said, length} <- [
            {"set a timer for 5 minutes", "5 minutes", Duration.new!(minute: 5)},
            {"snooze for 10 minutes", "10 minutes", Duration.new!(minute: 10)},
            {"cancel the 1 hour 30 min timer", "1 hour 30 min", Duration.new!(hour: 1, minute: 30)}
          ] do
        assert [reading] = Brain.ML.EntityExtractor.length_readings(text)
        assert %{entity_type: "duration", value: ^said} = reading

        for intent <- ["timer.set", "timer.cancel", "alarm.snooze"] do
          result = SlotDetector.detect(intent, [reading], side_effects: false)
          assert SlotResult.get_slot_value(result, "duration") == said, "#{intent}: #{text}"
          assert result.filled_slots["duration"].source == :explicit
        end

        # Brain.Services.LocalScheduler resolves the slot's text and runs the
        # timer for the length it names; the canonical value names the same one.
        assert %Resolution{length: ^length} = Resolver.resolve!(said, turn, time_zone: Zone.utc())
        assert %Resolution{length: ^length} = Resolver.resolve!(reading.canonical_value, turn, time_zone: Zone.utc())
      end
    end

    test "a number alone and a point in time are read as no length" do
      for text <- ["set a timer for 10", "turn on the TV in 2 hours", "red light off after 3 hours"] do
        assert Brain.ML.EntityExtractor.length_readings(text) == [], text
      end
    end

    test "a bare number does not fill a timer's or a snooze's duration: it names no length" do
      # Brain.Temporal.Resolver reads "10" as said with no unit.
      number = [%{entity_type: "number", value: "10", confidence: 0.9}]

      for intent <- ["timer.set", "timer.cancel", "alarm.snooze"] do
        refute SlotResult.get_slot_value(SlotDetector.detect(intent, number, side_effects: false), "duration"), intent
      end
    end

    test "a point in time does not fill a timer's duration: \"in 2 hours\" names no length" do
      # The gold standard labels "in 2 hours" and "after 3 hours" date_time,
      # and Brain.Temporal.Resolver resolves them to a moment with no length.
      for type <- ["date_time", "sys_time", "sys_date"] do
        moment = [%{entity_type: type, value: "in 2 hours", confidence: 0.9}]
        refute SlotResult.get_slot_value(SlotDetector.detect("timer.set", moment, side_effects: false), "duration"), type
      end
    end

    test "a recurrence does not fill a calendar item's event: it says when, not what" do
      # The gold standard's recurrences are "every day", "on Mondays",
      # "each morning".
      recurrence = [%{entity_type: "recurrence", value: "every monday", confidence: 0.9}]

      for intent <- ["calendar.schedule", "calendar.cancel"] do
        refute SlotResult.get_slot_value(SlotDetector.detect(intent, recurrence, side_effects: false), "event"), intent
      end
    end
  end

  describe "slots for an appliance" do
    test "an appliance is a device entry: no slot lists appliance, and a device entity fills every slot that lists device" do
      # data/entities/device_entries_en.json holds the dishwasher, the oven
      # and the refrigerator as devices; no entity file holds appliances.
      mappings = for {intent, schema} <- SlotDetector.registry(), {slot, types} <- Map.get(schema, "entity_mappings", %{}), do: {intent, slot, types}

      assert for({intent, slot, types} <- mappings, "appliance" in types, do: "#{intent}.#{slot}") == []

      slots = for {intent, slot, types} <- mappings, "device" in types, do: {intent, slot}

      assert slots != []

      dishwasher = [%{entity_type: "device", value: "dishwasher", confidence: 0.9}]

      for {intent, slot} <- slots do
        result = SlotDetector.detect(intent, dishwasher, side_effects: false)
        assert SlotResult.get_slot_value(result, slot) == "dishwasher", "#{intent}.#{slot}"
        assert result.filled_slots[slot].source == :explicit, "#{intent}.#{slot}"
      end
    end
  end
end
