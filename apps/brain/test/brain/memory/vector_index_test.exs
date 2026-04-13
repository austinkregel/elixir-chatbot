defmodule Brain.Memory.VectorIndexTest do
  use ExUnit.Case, async: false

  alias Brain.Memory.VectorIndex

  setup do
    table = VectorIndex.new()
    {:ok, table: table}
  end

  describe "basic operations" do
    test "insert and get", %{table: table} do
      embedding = [0.1, 0.2, 0.3]
      :ok = VectorIndex.insert(table, "id1", embedding)

      assert {:ok, ^embedding} = VectorIndex.get(table, "id1")
    end

    test "get returns error for non-existent id", %{table: table} do
      assert {:error, :not_found} = VectorIndex.get(table, "missing")
    end

    test "delete removes an entry", %{table: table} do
      VectorIndex.insert(table, "id1", [0.1, 0.2])
      assert VectorIndex.exists?(table, "id1")

      :ok = VectorIndex.delete(table, "id1")
      refute VectorIndex.exists?(table, "id1")
    end

    test "count returns number of entries", %{table: table} do
      assert VectorIndex.count(table) == 0

      VectorIndex.insert(table, "id1", [0.1])
      VectorIndex.insert(table, "id2", [0.2])

      assert VectorIndex.count(table) == 2
    end

    test "clear removes all entries", %{table: table} do
      VectorIndex.insert(table, "id1", [0.1])
      VectorIndex.insert(table, "id2", [0.2])
      assert VectorIndex.count(table) == 2

      :ok = VectorIndex.clear(table)
      assert VectorIndex.count(table) == 0
    end
  end

  describe "search" do
    test "finds most similar vectors", %{table: table} do
      # Insert vectors
      VectorIndex.insert(table, "a", [1.0, 0.0, 0.0])
      VectorIndex.insert(table, "b", [0.0, 1.0, 0.0])
      VectorIndex.insert(table, "c", [0.9, 0.1, 0.0])

      # Query for vector similar to [1, 0, 0]
      query = [1.0, 0.0, 0.0]
      results = VectorIndex.search(table, query, 2)

      assert length(results) == 2
      [{first_id, first_sim}, {second_id, _second_sim}] = results

      # "a" should be most similar (identical)
      assert first_id == "a"
      assert_in_delta first_sim, 1.0, 0.01

      # "c" should be second (0.9 component in same direction)
      assert second_id == "c"
    end

    test "returns empty list for empty index", %{table: table} do
      results = VectorIndex.search(table, [1.0, 0.0], 5)
      assert results == []
    end

    test "returns fewer results when k exceeds count", %{table: table} do
      VectorIndex.insert(table, "a", [1.0, 0.0])
      VectorIndex.insert(table, "b", [0.0, 1.0])

      results = VectorIndex.search(table, [1.0, 0.0], 10)
      assert length(results) == 2
    end
  end

  describe "cosine_similarity" do
    test "returns 1.0 for identical vectors" do
      vec = [1.0, 2.0, 3.0]
      assert_in_delta VectorIndex.cosine_similarity(vec, vec), 1.0, 0.0001
    end

    test "returns 0.0 for orthogonal vectors" do
      a = [1.0, 0.0]
      b = [0.0, 1.0]
      assert_in_delta VectorIndex.cosine_similarity(a, b), 0.0, 0.0001
    end

    test "returns -1.0 for opposite vectors" do
      a = [1.0, 0.0]
      b = [-1.0, 0.0]
      assert_in_delta VectorIndex.cosine_similarity(a, b), -1.0, 0.0001
    end

    test "returns 0.0 for different length vectors" do
      a = [1.0, 2.0]
      b = [1.0, 2.0, 3.0]
      assert VectorIndex.cosine_similarity(a, b) == +0.0
    end

    test "returns 0.0 for empty vectors" do
      assert VectorIndex.cosine_similarity([], []) == +0.0
    end
  end

  describe "mean_vector" do
    test "computes mean of multiple vectors" do
      vectors = [
        [1.0, 2.0, 3.0],
        [3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0]
      ]

      mean = VectorIndex.mean_vector(vectors)

      assert_in_delta Enum.at(mean, 0), 2.0, 0.0001
      assert_in_delta Enum.at(mean, 1), 3.0, 0.0001
      assert_in_delta Enum.at(mean, 2), 4.0, 0.0001
    end

    test "returns empty list for empty input" do
      assert VectorIndex.mean_vector([]) == []
    end
  end

  describe "export and import" do
    test "exports and imports vectors", %{table: table} do
      VectorIndex.insert(table, "a", [1.0, 0.0])
      VectorIndex.insert(table, "b", [0.0, 1.0])

      exported = VectorIndex.export(table)
      assert length(exported) == 2

      new_table = VectorIndex.new()
      :ok = VectorIndex.import(new_table, exported)

      assert VectorIndex.count(new_table) == 2
      assert {:ok, [1.0, +0.0]} = VectorIndex.get(new_table, "a")
      assert {:ok, [+0.0, 1.0]} = VectorIndex.get(new_table, "b")
    end
  end
end
