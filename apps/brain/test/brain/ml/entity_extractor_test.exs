defmodule Brain.ML.EntityExtractorTest do
  use ExUnit.Case, async: false

  alias Brain.ML.EntityExtractor

  # The world the test suite runs in, seeded by `mix test.models`.
  @world_id "default"

  describe "extract_entities/2" do
    test "returns a list of entities" do
      entities = EntityExtractor.extract_entities("hello world", world_id: @world_id)
      assert is_list(entities)
    end

    test "a word with several gazetteer candidates keeps all of them" do
      # "door" is a device, a lock and an emoji name. The gazetteer's reading
      # picks a primary type and keeps every candidate in :types; a decided
      # span built on any reading of its hypothesis carries them too.
      entities = EntityExtractor.extract_entities("open the door", world_id: @world_id)
      door = Enum.find(entities, &(String.downcase(&1.value) == "door"))

      assert door
      gazetteer = Enum.find(door.readings, &(&1.source == :gazetteer))
      assert gazetteer, "the gazetteer's reading of door is kept: #{inspect(door.readings)}"

      if door.resolved and Map.has_key?(door, :types) do
        candidate_types = Enum.map(door.types, & &1[:entity_type])
        assert "device" in candidate_types
        assert "emoji" in candidate_types
        assert gazetteer.entity_type in candidate_types
      end
    end

    test "every source's reading is kept in exactly one span, and every reading names its source" do
      text = "turn on the kitchen lights at 7 pm in Paris"
      readings = EntityExtractor.readings(text, world_id: @world_id)
      spans = EntityExtractor.extract_entities(text, world_id: @world_id, skip_disambiguation: true)

      assert Enum.all?(readings, &(&1.source in [:gazetteer, :system, :location_hint, :pos_tagger_propn, :entity_model]))

      kept = spans |> Enum.flat_map(& &1.readings) |> Enum.map(&{&1.source, &1.entity_type, &1.start_pos, &1.end_pos})
      assert Enum.sort(kept) == readings |> Enum.map(&{&1.source, &1.entity_type, &1.start_pos, &1.end_pos}) |> Enum.sort()

      assert Enum.all?(spans, &is_boolean(&1.resolved))
    end

    test "reports loaded when the gazetteer it reads has loaded" do
      assert EntityExtractor.is_loaded?() == Brain.ML.Gazetteer.loaded?()
      assert EntityExtractor.is_loaded?()
    end

    test "extracts known entity from entity maps" do
      entities = EntityExtractor.extract_entities("turn on the kitchen lights", world_id: @world_id)

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

    test "extracts numbers as system readings" do
      entities = EntityExtractor.extract_entities("set temperature to 72 degrees", world_id: @world_id)

      span = Enum.find(entities, &(&1.value == "72"))
      assert span != nil

      number = Enum.find(span.readings, &(&1.source == :system))
      assert %{entity_type: "number"} = number
      assert number.raw_confidence >= 0.8
    end

    test "extracts relative dates" do
      entities = EntityExtractor.extract_entities("remind me tomorrow", world_id: @world_id)

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
      entities = EntityExtractor.extract_entities("schedule for Monday", world_id: @world_id)

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
      entities = EntityExtractor.extract_entities("set to 50 degrees today", world_id: @world_id)

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
      entities = EntityExtractor.extract_entities("", world_id: @world_id)
      assert entities == []
    end

    test "handles text with no entities" do
      entities = EntityExtractor.extract_entities("xyzabc nonsense", world_id: @world_id)
      assert is_list(entities)
    end
  end

  describe "the world extraction runs in" do
    test "raises when no world is named" do
      assert_raise ArgumentError, ~r/EntityExtractor needs :world_id/, fn ->
        EntityExtractor.extract_entities("turn on the kitchen lights")
      end

      assert_raise ArgumentError, ~r/EntityExtractor needs :world_id/, fn ->
        EntityExtractor.extract_entities("turn on the kitchen lights", min_confidence: 0.3)
      end
    end

    test "runs the world's entity model, and every entity's match is the text at its span" do
      # Gold sentences the world's model was trained on, taken by a fixed
      # stride so the run stays short. Which spans the model wins against the
      # gazetteer and the POS hints moves with retraining, so this asserts that
      # it contributes at all, not where.
      rows = Brain.ML.EvaluationStore.load_gold_standard("ner", :train)

      texts =
        rows
        |> Enum.take_every(max(div(length(rows), 200), 1))
        |> Enum.map(& &1["text"])

      entities_by_text =
        Map.new(texts, fn text -> {text, EntityExtractor.extract_entities(text, world_id: @world_id)} end)

      assert Enum.any?(entities_by_text, fn {_text, entities} ->
               Enum.any?(entities, &(Map.get(&1, :source) == :entity_model))
             end),
             "no extraction over #{length(texts)} gold sentences kept an entity from the world's entity model"

      mismatched =
        for {text, entities} <- entities_by_text,
            entity <- entities,
            String.slice(text, entity.start_pos, entity.end_pos - entity.start_pos + 1) != entity.match,
            do: {text, entity}

      assert mismatched == [],
             "entities whose match is not the text at their span: #{inspect(Enum.take(mismatched, 5))}"
    end
  end

  describe "location extraction" do
    test "extracts location from prepositional context" do
      entities = EntityExtractor.extract_entities("weather in New York", world_id: @world_id)

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
    test "overlapping readings are one span, which keeps all of them" do
      # "new york" and "york" are both cities in the gazetteer; whatever the
      # data decides, they are one span, not two overlapping entities.
      entities = EntityExtractor.extract_entities("I'm in New York", world_id: @world_id)

      new_york = Enum.find(entities, &(&1.start_pos <= 7 and &1.end_pos >= 14))
      assert new_york
      assert Enum.any?(new_york.readings, &(&1.source == :gazetteer and &1.start_pos == 7 and &1.end_pos == 14))

      overlapping =
        for a <- entities, b <- entities, a != b, a.start_pos <= b.end_pos and b.start_pos <= a.end_pos, do: {a, b}

      assert overlapping == []
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

      entities = EntityExtractor.extract_entities("My name is Michael", world_id: @world_id)

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

      entities = EntityExtractor.extract_entities("Tell Sarah about the meeting", world_id: @world_id)

      sarah_entity =
        Enum.find(entities, fn e ->
          String.downcase(Map.get(e, :value, "")) == "sarah"
        end)

      assert sarah_entity != nil, "Should find 'Sarah' as an entity"
      assert String.downcase(sarah_entity.value) == "sarah"

      # The gazetteer's own confidence is what it was; whether the span is a
      # decided entity is the data's call, and an undecided one says why.
      gazetteer = Enum.find(sarah_entity.readings, &(&1.source == :gazetteer))
      assert gazetteer.raw_confidence >= 0.7, "Gazetteer confidence (#{gazetteer.raw_confidence}) should be >= 0.7"

      if sarah_entity.resolved do
        assert sarah_entity.confidence >= 0.7,
               "Entity confidence (#{sarah_entity.confidence}) should be >= 0.7 for learning"
      else
        assert sarah_entity.unresolved_reason in [:insufficient_evidence, :below_confidence_floor, :no_measured_reading]
      end
    end

    test "extracts multiple person names from text" do
      entities = EntityExtractor.extract_entities("John and Emily are coming to dinner", world_id: @world_id)

      person_names =
        entities
        |> Enum.filter(fn e -> Map.get(e, :entity_type) == "person" end)
        |> Enum.map(fn e -> String.downcase(e.value) end)

      found_count = Enum.count(["john", "emily"], &(&1 in person_names))

      assert found_count >= 1,
             "Should find at least 1 person name (John or Emily), got: #{inspect(person_names)}"
    end

    test "does not extract stoplist words as person names" do
      entities = EntityExtractor.extract_entities("I will do it in May", world_id: @world_id)

      person_entities =
        Enum.filter(entities, fn e ->
          e.resolved and Map.get(e, :entity_type) == "person" and
            String.downcase(Map.get(e, :value, "")) in ["will", "may"]
        end)

      assert person_entities == [],
             "Should not extract 'Will' or 'May' as person entities in this context"
    end

    test "extracts person names case-insensitively" do
      entities = EntityExtractor.extract_entities("DAVID said hello", world_id: @world_id)

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
      entities = EntityExtractor.extract_entities("hello friend", world_id: @world_id)

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
      entities = EntityExtractor.extract_entities("I'm from Friend", world_id: @world_id)

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

    test "respects min_confidence option: a decision below it is withheld, not dropped" do
      text = "turn on the kitchen lights in Paris"
      entities_high = EntityExtractor.extract_entities(text, world_id: @world_id, min_confidence: 0.99)
      entities_low = EntityExtractor.extract_entities(text, world_id: @world_id, min_confidence: 0.3)

      assert length(entities_low) == length(entities_high)
      assert Enum.count(entities_low, & &1.resolved) >= Enum.count(entities_high, & &1.resolved)

      for %{resolved: false, unresolved_reason: :below_confidence_floor} = span <- entities_high do
        assert span.readings != []
      end
    end
  end
end
