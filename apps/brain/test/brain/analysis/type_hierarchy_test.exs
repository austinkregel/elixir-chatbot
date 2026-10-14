defmodule Brain.Analysis.TypeHierarchyTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.TypeHierarchy

  setup do
    ensure_started()
    :ok
  end

  defp ensure_started do
    case Process.whereis(TypeHierarchy) do
      nil ->
        start_supervised!({TypeHierarchy, name: TypeHierarchy})

      _pid ->
        :ok
    end
  end

  describe "is_a?/2" do
    test "artist is a person" do
      assert TypeHierarchy.is_a?("artist", "person")
    end

    test "music_artist is a person" do
      assert TypeHierarchy.is_a?("music_artist", "person")
    end

    test "city is a location" do
      assert TypeHierarchy.is_a?("city", "location")
    end

    test "room is a location" do
      assert TypeHierarchy.is_a?("room", "location")
    end

    test "lights is a device" do
      assert TypeHierarchy.is_a?("lights", "device")
    end

    test "person is NOT an artist (reversed)" do
      refute TypeHierarchy.is_a?("person", "artist")
    end

    test "person is NOT a location (unrelated)" do
      refute TypeHierarchy.is_a?("person", "location")
    end

    test "song is NOT a person (unrelated)" do
      refute TypeHierarchy.is_a?("song", "person")
    end
  end

  describe "specializations/1" do
    test "person has artist, name as direct children" do
      specs = TypeHierarchy.specializations("person")
      assert "artist" in specs
      assert "name" in specs
    end

    test "artist has music_artist as child" do
      specs = TypeHierarchy.specializations("artist")
      assert "music_artist" in specs
    end

    test "location has city, room, etc." do
      specs = TypeHierarchy.specializations("location")
      assert "city" in specs
      assert "room" in specs
    end

    test "unknown type returns empty" do
      assert TypeHierarchy.specializations("xyzzy") == []
    end

    test "leaf type returns empty" do
      assert TypeHierarchy.specializations("city") == []
    end
  end

  describe "compatible?/2" do
    test "same type is compatible" do
      assert TypeHierarchy.compatible?("person", "person")
    end

    test "parent is compatible with child (can be narrowed)" do
      assert TypeHierarchy.compatible?("person", "artist")
    end

    test "child is compatible with parent (already specialized)" do
      assert TypeHierarchy.compatible?("artist", "person")
    end

    test "unrelated types are not compatible" do
      refute TypeHierarchy.compatible?("person", "song")
    end

    test "person is compatible with music_artist" do
      assert TypeHierarchy.compatible?("person", "music_artist")
    end
  end

  describe "narrowing_candidates/2" do
    test "person can narrow to artist from [artist, song, album]" do
      candidates = TypeHierarchy.narrowing_candidates("person", ["artist", "song", "album"])
      assert candidates == ["artist"]
    end

    test "person can narrow to artist" do
      candidates = TypeHierarchy.narrowing_candidates("person", ["artist", "music_artist"])
      assert "artist" in candidates
    end

    test "artist can narrow to music_artist" do
      candidates = TypeHierarchy.narrowing_candidates("artist", ["music_artist", "song"])
      assert "music_artist" in candidates
    end

    test "location can narrow to city" do
      candidates = TypeHierarchy.narrowing_candidates("location", ["city", "artist"])
      assert candidates == ["city"]
    end

    test "leaf type returns empty candidates" do
      candidates = TypeHierarchy.narrowing_candidates("city", ["city", "song"])
      assert candidates == []
    end

    test "unrelated expected types return empty" do
      candidates = TypeHierarchy.narrowing_candidates("person", ["song", "album", "city"])
      assert candidates == []
    end
  end

  describe "parent_type?/1" do
    test "person is a parent type" do
      assert TypeHierarchy.parent_type?("person")
    end

    test "location is a parent type" do
      assert TypeHierarchy.parent_type?("location")
    end

    test "artist is a parent type (has music_artist)" do
      assert TypeHierarchy.parent_type?("artist")
    end

    test "music_artist is not a parent type" do
      refute TypeHierarchy.parent_type?("music_artist")
    end
  end

  describe "parents_of/1" do
    test "artist's parent is person" do
      assert TypeHierarchy.parents_of("artist") == ["person"]
    end

    test "city's parent is location" do
      assert TypeHierarchy.parents_of("city") == ["location"]
    end

    test "a type listed under two parents keeps both" do
      assert TypeHierarchy.parents_of("color") == ["device", "information"]
      assert TypeHierarchy.is_a?("color", "device")
      assert TypeHierarchy.is_a?("color", "information")
    end

    test "source and sort are parents of their medium-specific kinds" do
      assert TypeHierarchy.parents_of("news_source") == ["source"]
      assert TypeHierarchy.parents_of("source") == ["information"]
      assert TypeHierarchy.parents_of("music_sort") == ["sort"]
      assert TypeHierarchy.parents_of("sort") == ["selection"]
      assert TypeHierarchy.parents_of("all") == ["selection"]
    end

    test "person has no parent" do
      assert TypeHierarchy.parents_of("person") == []
    end

    test "unknown type has no parent" do
      assert TypeHierarchy.parents_of("xyzzy") == []
    end
  end

  describe "all_types/0" do
    test "is every type entity_types.json declares: each parent, a root with no kinds among them, and each kind" do
      hierarchy =
        Path.join(:code.priv_dir(:brain), "analysis/entity_types.json") |> File.read!() |> Jason.decode!() |> Map.fetch!("type_hierarchy")

      declared = (Map.keys(hierarchy) ++ List.flatten(Map.values(hierarchy))) |> Enum.uniq() |> Enum.sort()

      assert Enum.any?(hierarchy, fn {_root, kinds} -> kinds == [] end), "entity_types.json declares no root without kinds"
      assert TypeHierarchy.all_types() == declared
    end
  end

  # Relations come only from priv/analysis/entity_types.json. These are types
  # registry slots list that the file declares as roots with no kinds, each an
  # English noun whose WordNet hypernyms reach another declared type's name
  # ("duration" reaches "amount", "event" reaches "activity"): the file, not
  # WordNet, relates them, and it relates them to nothing.
  describe "a root entity_types.json declares with no kinds" do
    @childless_roots ~w(duration event)

    test "is declared, and related to itself and to nothing else" do
      declared = TypeHierarchy.all_types()
      assert declared != []

      for type <- @childless_roots do
        assert type in declared, "#{type} is not declared in entity_types.json"
        assert TypeHierarchy.compatible?(type, type)
        assert TypeHierarchy.parents_of(type) == []
        assert TypeHierarchy.specializations(type) == []

        for other <- declared, other != type do
          refute TypeHierarchy.is_a?(type, other), "#{type} is_a? #{other}"
          refute TypeHierarchy.is_a?(other, type), "#{other} is_a? #{type}"
          refute TypeHierarchy.compatible?(type, other), "#{type} compatible? #{other}"
          refute TypeHierarchy.compatible?(other, type), "#{other} compatible? #{type}"
        end

        assert TypeHierarchy.narrowing_candidates(type, declared) == []
        assert TypeHierarchy.narrowing_candidates("amount", [type]) == []
      end
    end

    test "is not filled by an entity of a declared type through SlotDetector" do
      alias Brain.Analysis.{SlotDetector, SlotResult}

      cases = [
        {"timer.set", "duration", "number", "10"},
        {"alarm.snooze", "duration", "number", "10"},
        {"timer.set", "duration", "amount", "10"},
        {"calendar.schedule", "event", "recurrence", "every monday"},
        {"calendar.schedule", "event", "activity", "running"}
      ]

      for {intent, slot, type, value} <- cases do
        entities = [%{entity_type: type, value: value, confidence: 0.9}]
        result = SlotDetector.detect(intent, entities, side_effects: false)
        refute SlotResult.get_slot_value(result, slot), "#{intent}.#{slot} filled by #{type} #{inspect(value)}"
      end
    end
  end
end
