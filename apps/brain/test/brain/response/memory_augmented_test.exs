defmodule Brain.Response.MemoryAugmentedTest do
  use ExUnit.Case, async: true

  alias Brain.Response.MemoryAugmented

  describe "generate/3" do
    test "raises when opts name no world" do
      assert_raise ArgumentError, ~r/MemoryAugmented.generate requires a :world_id option/, fn ->
        MemoryAugmented.generate("weather.query", [], [])
      end
    end

    test "raises when the world is not a world id" do
      assert_raise ArgumentError, ~r/MemoryAugmented.generate requires a :world_id option/, fn ->
        MemoryAugmented.generate("weather.query", [], world_id: "")
      end
    end
  end

  describe "find_similar_episodes/3" do
    test "raises when opts name no world" do
      assert_raise ArgumentError, ~r/MemoryAugmented.find_similar_episodes requires a :world_id option/, fn ->
        MemoryAugmented.find_similar_episodes("weather.query", [], limit: 3)
      end
    end
  end
end
