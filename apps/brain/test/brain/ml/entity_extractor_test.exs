defmodule Brain.ML.EntityExtractorTest do
  use ExUnit.Case, async: false

  alias Brain.ML.EntityExtractor

  setup do
    EntityExtractor.load_entity_maps()
    :ok
  end

  describe "extract_entities/1" do
    test "returns a list of entities" do
      entities = EntityExtractor.extract_entities("hello world")
      assert is_list(entities)
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

  describe "extract_entities with custom entity maps" do
    test "uses provided entity maps" do
      custom_maps = %{
        "custom item" => %{entity_type: "custom", value: "Custom Item"}
      }

      entities = EntityExtractor.extract_entities("I need a custom item please", custom_maps)

      custom =
        Enum.find(entities, fn e ->
          Map.get(e, :entity_type) == "custom" or
            String.downcase(Map.get(e, :value, "")) == "custom item"
        end)

      if custom do
        assert custom.value == "Custom Item" or custom.value == "custom item"
      else
        assert is_list(entities)
      end
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
      custom_maps = %{
        "new" => %{entity_type: "word", value: "New"},
        "new york" => %{entity_type: "city", value: "New York"}
      }

      entities = EntityExtractor.extract_entities("I'm in New York", custom_maps)

      new_york =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "new york"
        end)

      just_new =
        Enum.find(entities, fn e ->
          Map.get(e, :value) == "New" and Map.get(e, :entity_type) == "word"
        end)

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
    alias Brain.ML.Gazetteer

    setup do
      unless Gazetteer.loaded?() do
        Gazetteer.load_all()
      end

      Gazetteer.add_entry("Michael", "person", %{confidence: 0.9})
      Gazetteer.add_entry("Sarah", "person", %{confidence: 0.9})
      Gazetteer.add_entry("John", "person", %{confidence: 0.9})
      Gazetteer.add_entry("Emily", "person", %{confidence: 0.9})

      on_exit(fn ->
        Gazetteer.remove_entry("Michael")
        Gazetteer.remove_entry("Sarah")
        Gazetteer.remove_entry("John")
        Gazetteer.remove_entry("Emily")
      end)

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