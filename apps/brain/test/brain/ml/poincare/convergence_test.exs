defmodule Brain.ML.Poincare.ConvergenceTest do
  use ExUnit.Case, async: false
  @moduletag :convergence

  alias Brain.ML.Poincare.{Distance, Embeddings}

  describe "synthetic tree convergence" do
    @tag timeout: 120_000
    test "learns correct hierarchy from binary tree" do
      # Balanced binary tree, depth 3, 15 nodes
      #        root
      #       /    \
      #     a0      a1
      #    / \     / \
      #  b0   b1  b2  b3
      #  /\ /\  /\ /\
      # c0 c1 c2 c3 c4 c5 c6 c7
      pairs = [
        {"a0", "root"}, {"a1", "root"},
        {"b0", "a0"}, {"b1", "a0"}, {"b2", "a1"}, {"b3", "a1"},
        {"c0", "b0"}, {"c1", "b0"}, {"c2", "b1"}, {"c3", "b1"},
        {"c4", "b2"}, {"c5", "b2"}, {"c6", "b3"}, {"c7", "b3"}
      ]

      {:ok, embeddings, entity_to_idx, _idx_to_entity} = Embeddings.train(pairs,
        dim: 5,
        epochs: 100,
        learning_rate: 0.01,
        num_negatives: 10
      )

      # Parent-child distances should be smaller than non-adjacent distances
      parent_child_dist = average_distance(embeddings, entity_to_idx, pairs)

      non_adjacent = [
        {"c0", "a1"}, {"c1", "b2"}, {"c4", "a0"},
        {"c6", "b1"}, {"c2", "b3"}, {"c5", "b0"}
      ]
      non_adjacent_dist = average_distance(embeddings, entity_to_idx, non_adjacent)

      assert parent_child_dist < non_adjacent_dist,
        "Parent-child dist (#{Float.round(parent_child_dist, 3)}) should be < " <>
        "non-adjacent dist (#{Float.round(non_adjacent_dist, 3)})"

      # Root should be closer to origin than leaves
      root_idx = Map.fetch!(entity_to_idx, "root")
      root_norm = Nx.sqrt(Nx.sum(Nx.pow(embeddings[root_idx], 2))) |> Nx.to_number()

      leaf_norms = for leaf <- ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"] do
        idx = Map.fetch!(entity_to_idx, leaf)
        Nx.sqrt(Nx.sum(Nx.pow(embeddings[idx], 2))) |> Nx.to_number()
      end

      avg_leaf_norm = Enum.sum(leaf_norms) / length(leaf_norms)

      assert root_norm < avg_leaf_norm,
        "Root norm (#{Float.round(root_norm, 3)}) should be < " <>
        "avg leaf norm (#{Float.round(avg_leaf_norm, 3)})"
    end

    @tag timeout: 120_000
    test "deeper nodes have larger norms" do
      pairs = [
        {"level1", "root"},
        {"level2", "level1"},
        {"level3", "level2"},
        {"level4", "level3"}
      ]

      {:ok, embeddings, entity_to_idx, _idx_to_entity} = Embeddings.train(pairs,
        dim: 5,
        epochs: 100,
        learning_rate: 0.01,
        num_negatives: 5
      )

      norms = for name <- ["root", "level1", "level2", "level3", "level4"] do
        idx = Map.fetch!(entity_to_idx, name)
        {name, Nx.sqrt(Nx.sum(Nx.pow(embeddings[idx], 2))) |> Nx.to_number()}
      end

      norm_values = Enum.map(norms, fn {_, n} -> n end)

      # The general trend should be increasing norms for deeper nodes
      # Check that the deepest node has larger norm than the root
      {_, root_norm} = Enum.find(norms, fn {n, _} -> n == "root" end)
      {_, deep_norm} = Enum.find(norms, fn {n, _} -> n == "level4" end)

      assert deep_norm > root_norm,
        "Deepest node norm (#{Float.round(deep_norm, 3)}) should be > " <>
        "root norm (#{Float.round(root_norm, 3)}). All norms: #{inspect(norms)}"
    end
  end

  defp average_distance(embeddings, entity_to_idx, pairs) do
    distances = for {a, b} <- pairs do
      idx_a = Map.fetch!(entity_to_idx, a)
      idx_b = Map.fetch!(entity_to_idx, b)
      Distance.distance(embeddings[idx_a], embeddings[idx_b]) |> Nx.to_number()
    end

    Enum.sum(distances) / max(length(distances), 1)
  end
end
