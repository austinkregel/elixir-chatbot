defmodule Brain.ML.Poincare.HierarchyDisambiguationIntegrationTest do
  use ExUnit.Case, async: false
  @moduletag :integration

  alias Brain.ML.Poincare.{Distance, Embeddings}

  @hierarchy_pairs [
    {"city", "location"},
    {"country", "location"},
    {"continent", "location"},
    {"person", "entity"},
    {"organization", "entity"},
    {"animal", "entity"},
    {"dog", "animal"},
    {"cat", "animal"},
    {"location", "thing"},
    {"entity", "thing"},
    {"temporal", "thing"},
    {"date", "temporal"},
    {"time", "temporal"}
  ]

  describe "hierarchy disambiguation" do
    @tag timeout: 120_000
    test "trains and correctly ranks entity type compatibility" do
      {:ok, embeddings, entity_to_idx, _idx_to_entity} = Embeddings.train(@hierarchy_pairs,
        dim: 5,
        epochs: 100,
        learning_rate: 0.01,
        num_negatives: 5
      )

      # "city" should be closer to "location" than to "person"
      city_location = entity_dist(embeddings, entity_to_idx, "city", "location")
      city_person = entity_dist(embeddings, entity_to_idx, "city", "person")

      assert city_location < city_person,
        "city-location (#{fmt(city_location)}) should be < city-person (#{fmt(city_person)})"

      # "dog" should be closer to "animal" than to "temporal"
      dog_animal = entity_dist(embeddings, entity_to_idx, "dog", "animal")
      dog_temporal = entity_dist(embeddings, entity_to_idx, "dog", "temporal")

      assert dog_animal < dog_temporal,
        "dog-animal (#{fmt(dog_animal)}) should be < dog-temporal (#{fmt(dog_temporal)})"
    end

    @tag timeout: 60_000
    test "serialization roundtrip preserves embeddings" do
      {:ok, embeddings, entity_to_idx, idx_to_entity} = Embeddings.train(@hierarchy_pairs,
        dim: 5,
        epochs: 30,
        learning_rate: 0.01,
        num_negatives: 5
      )

      tmp_path = Path.join(System.tmp_dir!(), "poincare_test_#{:rand.uniform(100000)}.term")

      try do
        Embeddings.save(embeddings, entity_to_idx, idx_to_entity, 5, tmp_path)
        {:ok, loaded} = Embeddings.load(tmp_path)

        assert loaded.entity_to_idx == entity_to_idx
        assert loaded.dim == 5

        diff = Nx.subtract(loaded.embeddings, embeddings)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

        assert diff < 1.0e-5, "Loaded embeddings should match original, max diff: #{diff}"
      after
        File.rm(tmp_path)
      end
    end

    test "all entities from hierarchy have embeddings after training" do
      {:ok, _embeddings, entity_to_idx, _idx_to_entity} = Embeddings.train(@hierarchy_pairs,
        dim: 5,
        epochs: 10,
        learning_rate: 0.01
      )

      all_entities = @hierarchy_pairs
      |> Enum.flat_map(fn {c, p} -> [c, p] end)
      |> Enum.uniq()

      for entity <- all_entities do
        assert Map.has_key?(entity_to_idx, entity),
          "Entity #{entity} should have an embedding"
      end
    end
  end

  defp entity_dist(embeddings, entity_to_idx, a, b) do
    idx_a = Map.fetch!(entity_to_idx, a)
    idx_b = Map.fetch!(entity_to_idx, b)
    Distance.distance(embeddings[idx_a], embeddings[idx_b]) |> Nx.to_number()
  end

  defp fmt(val), do: Float.round(val, 4)
end
