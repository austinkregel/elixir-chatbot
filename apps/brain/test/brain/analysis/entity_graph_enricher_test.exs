defmodule Brain.Analysis.EntityGraphEnricherTest do
  use ExUnit.Case, async: false

  alias Brain.Analysis.EntityGraphEnricher

  describe "enrich/1" do
    test "returns default enrichment for empty entity list" do
      assert EntityGraphEnricher.enrich([]) == []
    end

    test "adds default graph fields when Atlas is unavailable" do
      entities = [%{value: "TestEntity", entity_type: "location", confidence: 0.9}]

      enriched = EntityGraphEnricher.enrich(entities)
      assert length(enriched) == 1

      [entity] = enriched
      assert Map.has_key?(entity, :graph_known)
      assert Map.has_key?(entity, :graph_type)
      assert Map.has_key?(entity, :graph_neighbor_count)
      assert Map.has_key?(entity, :graph_neighbors)
      assert Map.has_key?(entity, :graph_relationships)
    end

    test "preserves original entity fields" do
      entities = [%{value: "Paris", entity_type: "location", confidence: 0.95}]

      enriched = EntityGraphEnricher.enrich(entities)
      [entity] = enriched

      assert entity.value == "Paris"
      assert entity.entity_type == "location"
      assert entity.confidence == 0.95
    end
  end

  describe "familiarity_score/1" do
    test "returns 0.5 for empty entity list" do
      assert EntityGraphEnricher.familiarity_score([]) == 0.5
    end

    test "returns high score when all entities are graph-known with neighbors" do
      entities = [
        %{graph_known: true, graph_neighbor_count: 10},
        %{graph_known: true, graph_neighbor_count: 8}
      ]

      score = EntityGraphEnricher.familiarity_score(entities)
      assert score > 0.7
    end

    test "returns low score when no entities are graph-known" do
      entities = [
        %{graph_known: false, graph_neighbor_count: 0},
        %{graph_known: false, graph_neighbor_count: 0}
      ]

      score = EntityGraphEnricher.familiarity_score(entities)
      assert score < 0.3
    end

    test "partial knowledge yields intermediate score" do
      entities = [
        %{graph_known: true, graph_neighbor_count: 5},
        %{graph_known: false, graph_neighbor_count: 0}
      ]

      score = EntityGraphEnricher.familiarity_score(entities)
      assert score > 0.2
      assert score < 0.8
    end
  end
end
