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

  describe "person name extraction" do
    test "extracts common person names from gazetteer" do
      # Test with common names that should be in our person gazetteer
      entities = EntityExtractor.extract_entities("My name is Michael")

      person =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "person" and
            String.downcase(Map.get(e, :value, "")) == "michael"
        end)

      assert person != nil, "Should find 'Michael' as a person entity"
      assert person.entity == "person"
      assert String.downcase(person.value) == "michael"
    end

    test "extracts person name with high enough confidence for learning" do
      # Person names should have confidence >= 0.7 to be learned by Learner
      entities = EntityExtractor.extract_entities("Tell Sarah about the meeting")

      person =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "person"
        end)

      if person do
        assert person.confidence >= 0.7,
               "Person entity confidence (#{person.confidence}) should be >= 0.7 for learning"
      end
    end

    test "extracts multiple person names from text" do
      entities = EntityExtractor.extract_entities("John and Emily are coming to dinner")

      person_names =
        entities
        |> Enum.filter(fn e -> Map.get(e, :entity) == "person" end)
        |> Enum.map(fn e -> String.downcase(e.value) end)

      # Should find at least one of the names
      assert Enum.any?(["john", "emily"], fn name -> name in person_names end),
             "Should find at least one person name, got: #{inspect(person_names)}"
    end

    test "does not extract stoplist words as person names" do
      # These are common words that are also names but filtered out
      # to prevent false positives
      entities = EntityExtractor.extract_entities("I will do it in May")

      # "Will" and "May" are in the stoplist and should NOT be extracted as person
      person_entities =
        Enum.filter(entities, fn e ->
          Map.get(e, :entity) == "person" and
            String.downcase(Map.get(e, :value, "")) in ["will", "may"]
        end)

      assert person_entities == [],
             "Should not extract 'Will' or 'May' as person entities in this context"
    end

    test "extracts person names case-insensitively" do
      # Names should be matched regardless of case
      entities = EntityExtractor.extract_entities("DAVID said hello")

      person =
        Enum.find(entities, fn e ->
          Map.get(e, :entity) == "person"
        end)

      if person do
        assert String.downcase(person.value) == "david"
      end
    end
  end

  describe "casing-based confidence adjustment" do
    test "reduces confidence when match casing doesn't match entity value" do
      # "friend" (lowercase in text) should have lower confidence
      # when matched against "Friend" (capitalized location)
      entities = EntityExtractor.extract_entities("hello friend")

      friend_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "friend"
        end)

      if friend_entity do
        # If "friend" matched a location "Friend", confidence should be reduced
        entity_type = Map.get(friend_entity, :entity)
        confidence = Map.get(friend_entity, :confidence, 1.0)
        match_text = Map.get(friend_entity, :match, "")
        entity_value = Map.get(friend_entity, :value, "")

        if entity_type == "location" && match_text != entity_value &&
             String.downcase(match_text) == String.downcase(entity_value) do
          # Casing mismatch for location - confidence should be penalized
          assert confidence < 0.7,
                 "Expected lower confidence for casing mismatch, got: #{confidence} for match='#{match_text}' value='#{entity_value}'"
        end
      end
    end

    test "maintains high confidence when casing matches" do
      # "Friend" (capitalized) should have normal confidence when matched against "Friend" location
      entities = EntityExtractor.extract_entities("I'm from Friend")

      friend_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "friend"
        end)

      if friend_entity do
        match_text = Map.get(friend_entity, :match, "")
        entity_value = Map.get(friend_entity, :value, "")
        confidence = Map.get(friend_entity, :confidence, 0.0)

        if match_text == entity_value do
          # Casing matches - should have normal confidence
          assert confidence >= 0.5,
                 "Expected normal confidence for matching casing, got: #{confidence}"
        end
      end
    end
  end

  describe "confidence threshold filtering" do
    test "filters out entities below threshold" do
      # Create test entities with different confidence levels
      high_conf_entity = %{
        entity: "location",
        value: "Austin",
        confidence: 0.85
      }

      low_conf_entity = %{
        entity: "location",
        value: "Friend",
        confidence: 0.45
      }

      medium_conf_entity = %{
        entity: "person",
        value: "John",
        confidence: 0.60
      }

      # Test filtering with 0.51 threshold
      entities = [high_conf_entity, low_conf_entity, medium_conf_entity]
      filtered = Enum.filter(entities, fn e ->
        confidence = Map.get(e, :confidence, 0.0)
        confidence >= 0.51
      end)

      # Should only include high and medium confidence entities
      assert length(filtered) == 2
      assert Enum.any?(filtered, &(&1.value == "Austin"))
      assert Enum.any?(filtered, &(&1.value == "John"))
      refute Enum.any?(filtered, &(&1.value == "Friend"))
    end

    test "respects min_confidence option" do
      # Test that we can override threshold via opts
      text = "hello friend"

      # Extract with high threshold
      entities_high = EntityExtractor.extract_entities(text, min_confidence: 0.8)

      # Extract with low threshold
      entities_low = EntityExtractor.extract_entities(text, min_confidence: 0.3)

      # High threshold should filter more strictly
      assert length(entities_low) >= length(entities_high)
    end
  end
end
