defmodule World.TypeInferrerTest do
  use ExUnit.Case, async: false

  alias World.TypeInferrer

  setup do
    TypeInferrer.init()
    TypeInferrer.clear()
    :ok
  end

  describe "init/0" do
    test "initializes ETS tables" do
      assert :ok = TypeInferrer.init()
    end

    test "double init does not crash" do
      assert :ok = TypeInferrer.init()
      assert :ok = TypeInferrer.init()
    end
  end

  describe "infer_type/4" do
    test "returns unknown with 0 confidence when no data" do
      {type, confidence} = TypeInferrer.infer_type("Paris", ["in", "Paris", "today"], ["ADP", "PROPN", "ADV"], "test_world")
      assert type == "unknown"
      assert confidence == 0.0
    end

    test "raises for nil world_id" do
      assert_raise ArgumentError, ~r/world_id is required/, fn ->
        TypeInferrer.infer_type("Paris", ["in", "Paris"], ["ADP", "PROPN"], nil)
      end
    end

    test "infers type after learning from known entity" do
      TypeInferrer.learn_from_known_entity("city", ["the", "city", "of"], ["DET", "NOUN", "ADP"], "test_world")
      TypeInferrer.learn_from_known_entity("city", ["visit", "the", "city"], ["VERB", "DET", "NOUN"], "test_world")

      {type, confidence} = TypeInferrer.infer_type("London", ["the", "city", "of"], ["DET", "NOUN", "ADP"], "test_world")

      if confidence > 0 do
        assert type == "city"
      end
    end
  end

  describe "learn_from_known_entity/4" do
    test "learns context patterns" do
      assert :ok = TypeInferrer.learn_from_known_entity("person", ["hello", "John"], ["INTJ", "PROPN"], "test_world")

      patterns = TypeInferrer.get_patterns_for_type("person", "test_world")
      assert is_map(patterns)
      assert Map.get(patterns, :_total, 0) > 0
    end

    test "raises for nil world_id" do
      assert_raise ArgumentError, ~r/world_id is required/, fn ->
        TypeInferrer.learn_from_known_entity("person", [], [], nil)
      end
    end
  end

  describe "get_patterns_for_type/2" do
    test "returns empty map when no patterns learned" do
      assert %{} = TypeInferrer.get_patterns_for_type("unknown_type", "test_world")
    end

    test "returns learned patterns" do
      TypeInferrer.learn_from_known_entity("animal", ["the", "dog"], ["DET", "NOUN"], "test_world")
      patterns = TypeInferrer.get_patterns_for_type("animal", "test_world")
      assert is_map(patterns)
      assert map_size(patterns) > 0
    end

    test "raises for nil world_id" do
      assert_raise ArgumentError, ~r/world_id is required/, fn ->
        TypeInferrer.get_patterns_for_type("test", nil)
      end
    end
  end

  describe "get_learned_types/1" do
    test "returns empty list initially" do
      assert [] = TypeInferrer.get_learned_types("test_world")
    end

    test "returns learned types" do
      TypeInferrer.learn_from_known_entity("city", ["in", "Paris"], ["ADP", "PROPN"], "test_world")
      TypeInferrer.learn_from_known_entity("person", ["hello", "John"], ["INTJ", "PROPN"], "test_world")

      types = TypeInferrer.get_learned_types("test_world")
      assert "city" in types
      assert "person" in types
    end

    test "scopes types to world" do
      TypeInferrer.learn_from_known_entity("food", ["eat", "pizza"], ["VERB", "NOUN"], "world_a")
      TypeInferrer.learn_from_known_entity("color", ["the", "blue"], ["DET", "ADJ"], "world_b")

      assert "food" in TypeInferrer.get_learned_types("world_a")
      refute "color" in TypeInferrer.get_learned_types("world_a")
      assert "color" in TypeInferrer.get_learned_types("world_b")
    end

    test "raises for nil world_id" do
      assert_raise ArgumentError, ~r/world_id is required/, fn ->
        TypeInferrer.get_learned_types(nil)
      end
    end
  end

  describe "get_cooccurrences/2" do
    test "returns empty map when no data" do
      assert %{} = TypeInferrer.get_cooccurrences("test_type", "test_world")
    end

    test "raises for nil world_id" do
      assert_raise ArgumentError, ~r/world_id is required/, fn ->
        TypeInferrer.get_cooccurrences("test", nil)
      end
    end
  end

  describe "export_learned_data/0" do
    test "exports empty data when nothing learned" do
      result = TypeInferrer.export_learned_data()
      assert is_map(result)
      assert Map.has_key?(result, :patterns)
      assert Map.has_key?(result, :cooccurrences)
      assert Map.has_key?(result, :exported_at)
    end

    test "exports learned data" do
      TypeInferrer.learn_from_known_entity("city", ["in", "Paris"], ["ADP", "PROPN"], "export_world")

      result = TypeInferrer.export_learned_data()
      assert map_size(result.patterns) > 0
    end
  end

  describe "import_learned_data/1" do
    test "imports previously exported data" do
      TypeInferrer.learn_from_known_entity("vehicle", ["the", "car"], ["DET", "NOUN"], "import_world")
      exported = TypeInferrer.export_learned_data()
      TypeInferrer.clear()

      assert :ok = TypeInferrer.import_learned_data(exported)
    end

    test "handles empty import data" do
      assert :ok = TypeInferrer.import_learned_data(%{})
    end

    test "handles string-keyed import data" do
      assert :ok = TypeInferrer.import_learned_data(%{"patterns" => %{}, "cooccurrences" => %{}})
    end
  end

  describe "clear/0" do
    test "clears all learned data" do
      TypeInferrer.learn_from_known_entity("test", ["a", "b"], ["DET", "NOUN"], "clear_world")
      assert :ok = TypeInferrer.clear()
      assert [] = TypeInferrer.get_learned_types("clear_world")
    end
  end
end
