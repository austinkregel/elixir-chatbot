defmodule Brain.ML.Poincare.EmbeddingsGenServerTest do
  use ExUnit.Case, async: false

  alias Brain.ML.Poincare.Embeddings

  describe "GenServer lifecycle" do
    test "starts and reports not ready without trained model" do
      name = :"poincare_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, pid} = Embeddings.start_link(name: name, world_id: world)
      assert Process.alive?(pid)
      refute Embeddings.ready?(name)
    end

    test "ready? returns false for non-existent server" do
      refute Embeddings.ready?(:"nonexistent_poincare_#{:rand.uniform(100_000)}")
    end

    test "lookup returns error when not ready" do
      name = :"poincare_lookup_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, _pid} = Embeddings.start_link(name: name, world_id: world)

      assert {:error, :not_ready} = Embeddings.lookup("test", name)
    end

    test "entity_distance returns error when not ready" do
      name = :"poincare_dist_test_#{:rand.uniform(100_000)}"
      world = "no_model_#{:rand.uniform(100_000)}"
      {:ok, _pid} = Embeddings.start_link(name: name, world_id: world)

      assert {:error, :not_ready} = Embeddings.entity_distance("a", "b", name)
    end
  end

  describe "train/2 edge cases" do
    test "trains with minimal pairs" do
      pairs = [{"child", "parent"}]
      {:ok, embeddings, entity_to_idx, _idx_to_entity} = Embeddings.train(pairs, dim: 3, epochs: 10)

      assert map_size(entity_to_idx) == 2
      assert Nx.shape(embeddings) == {2, 3}
    end

    test "trains with duplicate pairs" do
      pairs = [
        {"dog", "animal"},
        {"dog", "animal"},
        {"cat", "animal"}
      ]

      {:ok, embeddings, entity_to_idx, _} = Embeddings.train(pairs, dim: 3, epochs: 10)
      assert Map.has_key?(entity_to_idx, "dog")
      assert Map.has_key?(entity_to_idx, "cat")
      assert Map.has_key?(entity_to_idx, "animal")
      assert Nx.axis_size(embeddings, 0) == 3
    end

    test "all embeddings remain inside Poincare ball after training" do
      pairs = [
        {"a", "root"},
        {"b", "root"},
        {"c", "a"},
        {"d", "b"}
      ]

      {:ok, embeddings, entity_to_idx, _} = Embeddings.train(pairs, dim: 5, epochs: 50)

      for {_name, idx} <- entity_to_idx do
        emb = embeddings[idx]
        norm = Nx.sqrt(Nx.sum(Nx.pow(emb, 2))) |> Nx.to_number()
        assert norm < 1.0, "Embedding at idx #{idx} has norm #{norm} >= 1.0"
      end
    end

    test "custom learning rate is respected" do
      pairs = [{"x", "y"}]
      {:ok, _, _, _} = Embeddings.train(pairs, dim: 2, epochs: 5, learning_rate: 0.001)
      {:ok, _, _, _} = Embeddings.train(pairs, dim: 2, epochs: 5, learning_rate: 0.5)
    end
  end

  describe "save/5 and load/1" do
    @tag :tmp_dir
    test "roundtrip preserves data", %{tmp_dir: tmp_dir} do
      pairs = [{"dog", "animal"}, {"cat", "animal"}]
      {:ok, embeddings, entity_to_idx, idx_to_entity} = Embeddings.train(pairs, dim: 3, epochs: 10)

      path = Path.join(tmp_dir, "poincare_test.term")
      Embeddings.save(embeddings, entity_to_idx, idx_to_entity, 3, path)

      assert File.exists?(path)

      {:ok, loaded} = Embeddings.load(path)
      assert Map.has_key?(loaded, :embeddings)
      assert Map.has_key?(loaded, :entity_to_idx)
      assert map_size(loaded.entity_to_idx) == 3
    end

    test "load returns error for missing file" do
      result = Embeddings.load("/tmp/nonexistent_poincare_#{:rand.uniform(100_000)}.term")
      assert {:error, _} = result
    end
  end
end
