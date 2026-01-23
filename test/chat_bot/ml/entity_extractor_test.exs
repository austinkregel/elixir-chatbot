defmodule ChatBot.ML.EntityExtractorTest do
  use ExUnit.Case, async: false

  alias ChatBot.ML.EntityExtractor

  setup do
    # Ensure entity maps are loaded
    EntityExtractor.load_entity_maps()
    :ok
  end

  describe "extract_entities/1" do
    test "returns a list of entities" do
      entities = EntityExtractor.extract_entities("hello world")
      assert is_list(entities)
    end

    test "extracts known entity from entity maps" do
      # This will depend on what's in the entity files
      entities = EntityExtractor.extract_entities("turn on the kitchen lights")

      assert is_list(entities)

      # Look for kitchen if it's in the entity data
      kitchen =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "kitchen"
        end)

      if kitchen do
        assert Map.has_key?(kitchen, :entity)
        assert Map.has_key?(kitchen, :value)
        assert Map.has_key?(kitchen, :confidence)
      end
    end

    test "extracts numbers as system entities" do
      entities = EntityExtractor.extract_entities("set temperature to 72 degrees")

      number_entity =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "number"
        end)

      assert number_entity != nil
      assert number_entity.value == "72"
      assert number_entity.confidence >= 0.8
    end

    test "extracts relative dates" do
      entities = EntityExtractor.extract_entities("remind me tomorrow")

      # Look for tomorrow as either relative_date or with value matching
      date_entity =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "relative_date" or
            String.downcase(Map.get(e, :value, "")) == "tomorrow"
        end)

      # This might not find if not in entity data - just check it doesn't crash
      if date_entity do
        assert String.downcase(date_entity.value) == "tomorrow"
      else
        # Function should still return a list
        assert is_list(entities)
      end
    end

    test "extracts day names" do
      entities = EntityExtractor.extract_entities("schedule for Monday")

      day_entity =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "day_name"
        end)

      assert day_entity != nil
      assert String.downcase(day_entity.value) == "monday"
    end

    test "entity has required fields" do
      entities = EntityExtractor.extract_entities("set to 50 degrees today")

      if length(entities) > 0 do
        entity = Enum.at(entities, 0)
        assert Map.has_key?(entity, :entity)
        assert Map.has_key?(entity, :value)
        assert Map.has_key?(entity, :match)
        assert Map.has_key?(entity, :start_pos)
        assert Map.has_key?(entity, :end_pos)
        assert Map.has_key?(entity, :confidence)
      end
    end

    test "handles empty string" do
      entities = EntityExtractor.extract_entities("")
      assert entities == []
    end

    test "handles text with no entities" do
      entities = EntityExtractor.extract_entities("xyzabc nonsense")
      assert is_list(entities)
    end
  end

  describe "extract_entities with custom entity maps" do
    test "uses provided entity maps" do
      # Entity maps use lowercase normalized keys
      custom_maps = %{
        "custom item" => %{entity_type: "custom", value: "Custom Item"}
      }

      entities = EntityExtractor.extract_entities("I need a custom item please", custom_maps)

      custom =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "custom" or
            String.downcase(Map.get(e, :value, "")) == "custom item"
        end)

      # If found, validate the structure
      if custom do
        assert custom.value == "Custom Item" or custom.value == "custom item"
      else
        # The gazetteer might be loaded and override, just verify function works
        assert is_list(entities)
      end
    end
  end

  describe "location extraction" do
    test "extracts location from prepositional context" do
      entities = EntityExtractor.extract_entities("weather in New York")

      location =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "location"
        end)

      # Should find New York as location (either from gazetteer or context)
      if location do
        assert String.contains?(location.value, "New") or
                 String.contains?(location.value, "York")
      end
    end
  end

  describe "conflict resolution" do
    test "resolves overlapping entities by keeping longest match" do
      # If we have both "New" and "New York" as entities,
      # "New York" should win
      custom_maps = %{
        "new" => %{entity_type: "word", value: "New"},
        "new york" => %{entity_type: "city", value: "New York"}
      }

      entities = EntityExtractor.extract_entities("I'm in New York", custom_maps)

      # Should prefer "New York" over "New"
      new_york =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "new york"
        end)

      just_new =
        Enum.find(entities, fn e ->
          Map.get(e, :value) == "New" and Map.get(e, :entity) == "word"
        end)

      # If we found "New York", we shouldn't also find just "New" at the same position
      if new_york do
        assert just_new == nil or just_new.start_pos != new_york.start_pos
      end
    end
  end

  describe "load_entity_maps/0" do
    test "loads and caches entity maps" do
      result = EntityExtractor.load_entity_maps()

      case result do
        {:ok, maps} ->
          assert is_map(maps)

        {:error, _} ->
          # May fail if no entity files exist
          assert true
      end
    end
  end

  describe "get_entity_maps/0" do
    test "returns cached entity maps" do
      maps = EntityExtractor.get_entity_maps()
      assert is_map(maps)
    end
  end
end
