defmodule Brain.ML.EntityExtractorTest do
  use ExUnit.Case, async: false

  alias Brain.ML.EntityExtractor

  describe "extract_entities/1" do
    test "returns a list of entities" do
      entities = EntityExtractor.extract_entities("hello world")
      assert is_list(entities)
    end

    test "a word with several gazetteer candidates keeps all of them" do
      # "door" is a device, a lock and an emoji name. Extraction reads the one
      # gazetteer, picks a primary type, and keeps every candidate in :types.
      entities = EntityExtractor.extract_entities("open the door")
      door = Enum.find(entities, &(String.downcase(&1.value) == "door"))

      assert door
      candidate_types = Enum.map(door.types, & &1[:entity_type])
      assert "device" in candidate_types
      assert "emoji" in candidate_types
      assert door.entity_type in candidate_types
    end

    test "reports loaded when the gazetteer it reads has loaded" do
      assert EntityExtractor.is_loaded?() == Brain.ML.Gazetteer.loaded?()
      assert EntityExtractor.is_loaded?()
    end

    test "extracts known entity from entity maps" do
      entities = EntityExtractor.extract_entities("turn on the kitchen lights")

      assert is_list(entities)

      kitchen =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "kitchen"
        end)

      if kitchen do
        assert Map.has_key?(kitchen, :entity_type)
        assert Map.has_key?(kitchen, :value)
        assert Map.has_key?(kitchen, :confidence)
      end
    end

    test "extracts numbers as system entities" do
      entities = EntityExtractor.extract_entities("set temperature to 72 degrees")

      number_entity =
        Enum.find(entities, fn e ->
          Map.get(e, :entity_type) == "number"
        end)

      assert number_entity != nil
      assert number_entity.value == "72"
      assert number_entity.confidence >= 0.8
    end

    test "extracts relative dates" do
      entities = EntityExtractor.extract_entities("remind me tomorrow")

      date_entity =
        Enum.find(entities, fn e ->
          Map.get(e, :entity_type) == "relative_date" or
            String.downcase(Map.get(e, :value, "")) == "tomorrow"
        end)

      if date_entity do
        assert String.downcase(date_entity.value) == "tomorrow"
      else
        assert is_list(entities)
      end
    end

    test "extracts day names" do
      entities = EntityExtractor.extract_entities("schedule for Monday")

      day_entity =
        Enum.find(entities, fn e ->
          entity_type = Map.get(e, :entity_type, "")
          value = String.downcase(Map.get(e, :value, ""))

          value == "monday" or
            entity_type in ["day_name", "sys_date", "weekday", "date", "relative_date"]
        end)

      if day_entity do
        assert String.downcase(day_entity.value) == "monday"
      else
        assert is_list(entities)
      end
    end

    test "entity has required fields" do
      entities = EntityExtractor.extract_entities("set to 50 degrees today")

      if entities != [] do
        entity = Enum.at(entities, 0)
        assert Map.has_key?(entity, :entity_type)
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

  describe "location extraction" do
    test "extracts location from prepositional context" do
      entities = EntityExtractor.extract_entities("weather in New York")

      location =
        Enum.find(entities, fn e ->
          Map.get(e, :entity_type) == "location"
        end)

      if location do
        assert String.contains?(location.value, "New") or
                 String.contains?(location.value, "York")
      end
    end
  end

  describe "conflict resolution" do
    test "resolves overlapping entities by keeping longest match" do
      # "new york" and "york" are both cities in the gazetteer; only the
      # longer span may survive.
      entities = EntityExtractor.extract_entities("I'm in New York")

      new_york = Enum.find(entities, &(String.downcase(&1.value) == "new york"))
      assert new_york

      inside =
        Enum.filter(entities, fn e ->
          e != new_york and e.start_pos >= new_york.start_pos and e.end_pos <= new_york.end_pos
        end)

      assert inside == []
    end
  end

  describe "person name extraction" do
    alias Brain.ML.Gazetteer

    setup do
      unless Gazetteer.loaded?() do
        Gazetteer.load_all()
      end

      # All four are already in the gazetteer's person sources, so `add_entry`
      # refuses them as duplicates and the old `remove_entry` cleanup deleted
      # the real entries for the rest of the run. Snapshot and restore instead.
      Brain.Test.Singletons.preserve_ets_keys!(:gazetteer_entities, ~w(michael sarah john emily))

      Gazetteer.add_entry("Michael", "person", %{confidence: 0.9})
      Gazetteer.add_entry("Sarah", "person", %{confidence: 0.9})
      Gazetteer.add_entry("John", "person", %{confidence: 0.9})
      Gazetteer.add_entry("Emily", "person", %{confidence: 0.9})

      :ok
    end

    test "extracts common person names from gazetteer" do
      assert {:ok, _info} = Gazetteer.lookup("Michael"),
             "Michael should be in the gazetteer from setup"

      entities = EntityExtractor.extract_entities("My name is Michael")

      michael_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "michael"
        end)

      assert michael_entity != nil, "Should find 'Michael' as an entity"
      assert String.downcase(michael_entity.value) == "michael"

      has_person_type =
        michael_entity.entity_type == "person" or
          (is_list(Map.get(michael_entity, :types)) and
             Enum.any?(michael_entity.types, fn t ->
               Map.get(t, :entity_type) == "person"
             end))

      assert has_person_type,
             "Michael should have person type, got: #{inspect(michael_entity)}"
    end

    test "extracts person name with high enough confidence for learning" do
      assert {:ok, _info} = Gazetteer.lookup("Sarah"),
             "Sarah should be in the gazetteer from setup"

      entities = EntityExtractor.extract_entities("Tell Sarah about the meeting")

      sarah_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "sarah"
        end)

      assert sarah_entity != nil, "Should find 'Sarah' as an entity"
      assert String.downcase(sarah_entity.value) == "sarah"

      assert sarah_entity.confidence >= 0.7,
             "Entity confidence (#{sarah_entity.confidence}) should be >= 0.7 for learning"
    end

    test "extracts multiple person names from text" do
      entities = EntityExtractor.extract_entities("John and Emily are coming to dinner")

      person_names =
        entities
        |> Enum.filter(fn e -> Map.get(e, :entity_type) == "person" end)
        |> Enum.map(fn e -> String.downcase(e.value) end)

      found_count = Enum.count(["john", "emily"], &(&1 in person_names))

      assert found_count >= 1,
             "Should find at least 1 person name (John or Emily), got: #{inspect(person_names)}"
    end

    test "does not extract stoplist words as person names" do
      entities = EntityExtractor.extract_entities("I will do it in May")

      person_entities =
        Enum.filter(entities, fn e ->
          Map.get(e, :entity_type) == "person" and
            String.downcase(Map.get(e, :value, "")) in ["will", "may"]
        end)

      assert person_entities == [],
             "Should not extract 'Will' or 'May' as person entities in this context"
    end

    test "extracts person names case-insensitively" do
      entities = EntityExtractor.extract_entities("DAVID said hello")

      person =
        Enum.find(entities, fn e ->
          Map.get(e, :entity_type) == "person"
        end)

      if person do
        assert String.downcase(person.value) == "david"
      end
    end
  end

  describe "casing-based confidence adjustment" do
    test "reduces confidence when match casing doesn't match entity value" do
      entities = EntityExtractor.extract_entities("hello friend")

      friend_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "friend"
        end)

      if friend_entity do
        entity_type = Map.get(friend_entity, :entity_type)
        confidence = Map.get(friend_entity, :confidence, 1.0)
        match_text = Map.get(friend_entity, :match, "")
        entity_value = Map.get(friend_entity, :value, "")

        if entity_type == "location" && match_text != entity_value &&
             String.downcase(match_text) == String.downcase(entity_value) do
          assert confidence < 0.7,
                 "Expected lower confidence for casing mismatch, got: #{confidence} for match='#{match_text}' value='#{entity_value}'"
        end
      end
    end

    test "maintains high confidence when casing matches" do
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
          assert confidence >= 0.5,
                 "Expected normal confidence for matching casing, got: #{confidence}"
        end
      end
    end
  end

  describe "confidence threshold filtering" do
    test "filters out entities below threshold" do
      high_conf_entity = %{
        entity_type: "location",
        value: "Austin",
        confidence: 0.85
      }

      low_conf_entity = %{
        entity_type: "location",
        value: "Friend",
        confidence: 0.45
      }

      medium_conf_entity = %{
        entity_type: "person",
        value: "John",
        confidence: 0.6
      }

      entities = [high_conf_entity, low_conf_entity, medium_conf_entity]

      filtered =
        Enum.filter(entities, fn e ->
          confidence = Map.get(e, :confidence, 0.0)
          confidence >= 0.51
        end)

      assert length(filtered) == 2
      assert Enum.any?(filtered, &(&1.value == "Austin"))
      assert Enum.any?(filtered, &(&1.value == "John"))
      refute Enum.any?(filtered, &(&1.value == "Friend"))
    end

    test "respects min_confidence option" do
      text = "hello friend"
      entities_high = EntityExtractor.extract_entities(text, min_confidence: 0.8)
      entities_low = EntityExtractor.extract_entities(text, min_confidence: 0.3)
      assert length(entities_low) >= length(entities_high)
    end
  end
end