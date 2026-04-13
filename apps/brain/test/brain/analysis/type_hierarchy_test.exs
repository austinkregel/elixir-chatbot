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

    test "music-artist is a person" do
      assert TypeHierarchy.is_a?("music-artist", "person")
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

    test "artist has music-artist as child" do
      specs = TypeHierarchy.specializations("artist")
      assert "music-artist" in specs
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

    test "person is compatible with music-artist" do
      assert TypeHierarchy.compatible?("person", "music-artist")
    end
  end

  describe "narrowing_candidates/2" do
    test "person can narrow to artist from [artist, song, album]" do
      candidates = TypeHierarchy.narrowing_candidates("person", ["artist", "song", "album"])
      assert candidates == ["artist"]
    end

    test "person can narrow to artist" do
      candidates = TypeHierarchy.narrowing_candidates("person", ["artist", "music-artist"])
      assert "artist" in candidates
    end

    test "artist can narrow to music-artist" do
      candidates = TypeHierarchy.narrowing_candidates("artist", ["music-artist", "song"])
      assert "music-artist" in candidates
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

    test "artist is a parent type (has music-artist)" do
      assert TypeHierarchy.parent_type?("artist")
    end

    test "music-artist is not a parent type" do
      refute TypeHierarchy.parent_type?("music-artist")
    end
  end

  describe "parent_of/1" do
    test "artist's parent is person" do
      assert TypeHierarchy.parent_of("artist") == "person"
    end

    test "city's parent is location" do
      assert TypeHierarchy.parent_of("city") == "location"
    end

    test "person has no parent" do
      assert TypeHierarchy.parent_of("person") == nil
    end

    test "unknown type has no parent" do
      assert TypeHierarchy.parent_of("xyzzy") == nil
    end
  end
end
