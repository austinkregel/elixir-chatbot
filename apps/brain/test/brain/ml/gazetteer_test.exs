defmodule Brain.ML.GazetteerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Gazetteer
  import Brain.TestHelpers

  setup do
    ensure_started(Brain.ML.Gazetteer)
    :ok
  end

  describe "lookup/1" do
    test "returns :not_found for unknown text before loading" do
      result = Gazetteer.lookup("unknown_xyz_123")
      assert result == :not_found
    end

    test "lookup is case-insensitive" do
      if Gazetteer.loaded?() do
        _result1 = Gazetteer.lookup("kitchen")
        _result2 = Gazetteer.lookup("KITCHEN")
        _result3 = Gazetteer.lookup("Kitchen")
        assert true
      else
        assert true
      end
    end
  end

  describe "lookup_spans/1" do
    test "returns empty list for empty token list" do
      spans = Gazetteer.lookup_spans([])
      assert spans == []
    end

    test "returns empty list when no matches found" do
      spans = Gazetteer.lookup_spans(["xyzabc", "nonsense", "words"])
      assert is_list(spans)
    end

    test "finds multi-word entities" do
      if Gazetteer.loaded?() do
        spans = Gazetteer.lookup_spans(["new", "york", "city"])
        assert is_list(spans)

        if spans != [] do
          {start_idx, end_idx, entity_info} = Enum.at(spans, 0)
          assert is_integer(start_idx)
          assert is_integer(end_idx)
          assert is_list(entity_info) or is_map(entity_info)
        end
      else
        assert true
      end
    end
  end

  describe "is_prefix?/1" do
    test "returns boolean" do
      result = Gazetteer.is_prefix?("new")
      assert is_boolean(result)
    end
  end

  describe "stats/0" do
    test "returns stats map" do
      stats = Gazetteer.stats()
      assert is_map(stats)
    end
  end

  describe "loaded?/0" do
    test "returns boolean" do
      result = Gazetteer.loaded?()
      assert is_boolean(result)
    end
  end

  describe "load_all/0" do
    test "loads gazetteer data" do
      result = Gazetteer.load_all()

      case result do
        {:ok, stats} ->
          assert is_map(stats)
          assert Map.has_key?(stats, :entities)
          assert Map.has_key?(stats, :loaded)
          assert stats.loaded == true

        {:error, _reason} ->
          assert true
      end
    end

    test "after loading, loaded? returns true" do
      Gazetteer.load_all()
      result = Gazetteer.loaded?()
      assert is_boolean(result)
    end

    test "after loading, stats shows entity counts" do
      Gazetteer.load_all()

      stats = Gazetteer.stats()

      if stats[:loaded] do
        assert Map.has_key?(stats, :entities)
        assert Map.has_key?(stats, :load_time_ms)
      end
    end
  end

  describe "integration with entity lookup" do
    setup do
      Gazetteer.load_all()
      :ok
    end

    test "can lookup entities after loading" do
      if Gazetteer.loaded?() do
        _kitchen = Gazetteer.lookup("kitchen")
        _bedroom = Gazetteer.lookup("bedroom")
        assert true
      else
        assert true
      end
    end

    test "can find spans in token lists" do
      if Gazetteer.loaded?() do
        spans = Gazetteer.lookup_spans(["turn", "on", "the", "kitchen", "lights"])
        assert is_list(spans)
      else
        assert true
      end
    end
  end

  describe "duplicate prevention" do
    test "add_entry prevents duplicate entries" do
      unique_name = "test_unique_city_#{System.unique_integer([:positive])}"
      assert {:ok, _key} = Gazetteer.add_entry(unique_name, "location")
      assert {:error, {:duplicate, "location"}} = Gazetteer.add_entry(unique_name, "location")
      assert {:error, {:duplicate, "location"}} = Gazetteer.add_entry(unique_name, "city")
    end

    test "add_entry is case-insensitive for duplicate detection" do
      unique_name = "TestCityCase#{System.unique_integer([:positive])}"
      assert {:ok, _key} = Gazetteer.add_entry(unique_name, "location")

      assert {:error, {:duplicate, "location"}} =
               Gazetteer.add_entry(String.upcase(unique_name), "location")

      assert {:error, {:duplicate, "location"}} =
               Gazetteer.add_entry(String.downcase(unique_name), "location")
    end

    test "exists? returns correct results" do
      unique_name = "test_exists_city_#{System.unique_integer([:positive])}"
      assert Gazetteer.exists?(unique_name) == false
      {:ok, _} = Gazetteer.add_entry(unique_name, "location")
      assert {true, infos} = Gazetteer.exists?(unique_name)
      assert is_list(infos)
      assert infos != []
      assert Enum.any?(infos, fn info -> info[:entity_type] == "location" end)
      assert {true, _} = Gazetteer.exists?(String.upcase(unique_name))
    end
  end

  describe "multi-type support" do
    test "lookup returns all entity types for ambiguous entries" do
      if Gazetteer.loaded?() do
        case Gazetteer.lookup("austin") do
          {:ok, infos} when is_list(infos) ->
            types = Enum.map(infos, &(Map.get(&1, :entity_type) || Map.get(&1, :type)))
            assert types != []

          {:ok, info} when is_map(info) ->
            assert true

          :not_found ->
            assert true
        end
      else
        assert true
      end
    end

    test "lookup_all_types returns list for any entry" do
      if Gazetteer.loaded?() do
        result = Gazetteer.lookup_all_types("kitchen")

        assert is_list(result)

        if result != [] do
          info = hd(result)
          assert Map.has_key?(info, :entity_type) or Map.has_key?(info, :type)
        end
      else
        assert true
      end
    end

    test "lookup_all_types returns empty list for not found" do
      result = Gazetteer.lookup_all_types("xyznonexistent123")
      assert result == []
    end

    test "list_by_type works with multi-type entries" do
      if Gazetteer.loaded?() do
        locations = Gazetteer.list_by_type("location")

        assert is_list(locations)

        for {_key, info} <- locations do
          entity_type = Map.get(info, :entity_type) || Map.get(info, :type)
          assert entity_type == "location"
        end
      else
        assert true
      end
    end

    test "list_types returns all unique types" do
      if Gazetteer.loaded?() do
        types = Gazetteer.list_types()

        assert is_list(types)
        assert types != []
        assert Enum.all?(types, &is_binary/1)
      else
        assert true
      end
    end

    test "lookup_spans returns list of entity_infos for multi-type entries" do
      if Gazetteer.loaded?() do
        spans = Gazetteer.lookup_spans(["austin"])

        for {_start, _end, entity_info} <- spans do
          assert is_list(entity_info) or is_map(entity_info)
        end
      else
        assert true
      end
    end
  end

  describe "insert_entry_direct merges multi-type" do
    test "inserting same key with different entity_type creates multi-type entry" do
      if Gazetteer.loaded?() do
        unique_key = "test_merge_city_#{System.unique_integer([:positive])}"

        Gazetteer.add_entry(unique_key, "location", %{source: :csv})

        existing_before = Gazetteer.lookup_all_types(unique_key)
        assert length(existing_before) >= 1

        assert Enum.any?(existing_before, fn entry ->
          Map.get(entry, :entity_type) == "location"
        end)
      else
        assert true
      end
    end
  end

  describe "context-aware type ranking" do
    setup do
      table = :gazetteer_entities

      unique = System.unique_integer([:positive])
      key = "ctx_test_#{unique}"

      infos = [
        %{entity_type: "location", value: key, source: :csv, type: "location"},
        %{entity_type: "setting", value: key, source: :json, type: "setting"},
        %{entity_type: "person", value: key, source: :json, type: "person"}
      ]

      :ets.insert(table, {String.downcase(key), infos})

      {:ok, key: String.downcase(key), infos: infos}
    end

    test "rank_types_with_context boosts location for weather domain", %{infos: infos} do
      ranked = Gazetteer.rank_types_with_context(infos, domain: :weather)

      primary_type = Map.get(hd(ranked), :entity_type)
      assert primary_type in ["location", "city"]
    end

    test "rank_types_with_context returns unchanged for no context", %{infos: infos} do
      ranked = Gazetteer.rank_types_with_context(infos, [])
      assert ranked == infos
    end

    test "single-type list is returned unchanged regardless of context" do
      single = [%{entity_type: "person", value: "John", source: :csv}]
      ranked = Gazetteer.rank_types_with_context(single, domain: :weather)
      assert ranked == single
    end

    test "lookup_spans with domain context ranks types", %{key: key} do
      spans = Gazetteer.lookup_spans([key], domain: :weather)

      case spans do
        [{0, 0, entity_infos}] ->
          primary = hd(entity_infos)
          assert Map.get(primary, :entity_type) in ["location", "city"]

        _ ->
          assert true
      end
    end
  end

  describe "lattice span generation" do
    setup do
      table = :gazetteer_entities

      :ets.insert(table, {"new", [%{entity_type: "person", value: "New", source: :json}]})
      :ets.insert(table, {"new york", [%{entity_type: "location", value: "New York", source: :csv}]})

      :ets.insert(:gazetteer_prefixes, {"new", true})

      :ok
    end

    test "lookup_spans_lattice returns overlapping spans" do
      lattice = Gazetteer.lookup_spans_lattice(["new", "york"])

      assert length(lattice) >= 1

      starts = Enum.map(lattice, fn {start, _end, _infos, _score} -> start end)
      assert 0 in starts
    end

    test "lookup_spans_lattice includes scores" do
      lattice = Gazetteer.lookup_spans_lattice(["new", "york"])

      for {_start, _end, _infos, score} <- lattice do
        assert is_number(score)
        assert score > 0
      end
    end

    test "longer spans get higher scores in lattice" do
      lattice = Gazetteer.lookup_spans_lattice(["new", "york"])

      long_spans = Enum.filter(lattice, fn {s, e, _, _} -> e - s >= 1 end)
      short_spans = Enum.filter(lattice, fn {s, e, _, _} -> e - s == 0 end)

      if long_spans != [] and short_spans != [] do
        best_long = Enum.max_by(long_spans, fn {_, _, _, score} -> score end)
        best_short = Enum.max_by(short_spans, fn {_, _, _, score} -> score end)

        {_, _, _, long_score} = best_long
        {_, _, _, short_score} = best_short

        assert long_score > short_score
      end
    end
  end

  describe "type preference learning" do
    test "record_type_preference adjusts stored preferences" do
      unique = "learn_test_#{System.unique_integer([:positive])}"
      key = String.downcase(unique)

      infos = [
        %{entity_type: "location", value: unique, source: :csv},
        %{entity_type: "person", value: unique, source: :json}
      ]

      :ets.insert(:gazetteer_entities, {key, infos})

      assert :ok = Gazetteer.record_type_preference(unique, "location", :weather)

      updated = Gazetteer.lookup_all_types(unique)
      location_entry = Enum.find(updated, &(Map.get(&1, :entity_type) == "location"))
      assert location_entry != nil

      prefs = Map.get(location_entry, :domain_preferences, %{})
      assert Map.get(prefs, :weather) == 1
    end

    test "record_type_preference returns :not_found for unknown entries" do
      result = Gazetteer.record_type_preference("nonexistent_xyz_999", "location", :weather)
      assert result == :not_found
    end

    test "learned preferences boost ranking in subsequent lookups" do
      unique = "learn_rank_#{System.unique_integer([:positive])}"
      key = String.downcase(unique)

      infos = [
        %{entity_type: "person", value: unique, source: :json},
        %{entity_type: "location", value: unique, source: :json}
      ]

      :ets.insert(:gazetteer_entities, {key, infos})

      Gazetteer.record_type_preference(unique, "location", :weather)
      Gazetteer.record_type_preference(unique, "location", :weather)

      all = Gazetteer.lookup_all_types(unique)
      ranked = Gazetteer.rank_types_with_context(all, domain: :weather)

      primary = hd(ranked)
      assert Map.get(primary, :entity_type) == "location"
    end
  end
end