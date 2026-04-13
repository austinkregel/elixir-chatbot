defmodule Brain.Analysis.EntityGraphEnricher do
  @moduledoc """
  Batch-enriches extracted entities with knowledge graph context.

  Queries the knowledge graph once per chunk for all entities, annotating
  each entity map with graph-derived fields. This replaces the per-entity
  `entity_context` calls scattered across ContextualEntityInferrer,
  generator, and enricher with a single batch lookup early in the pipeline.

  Enriched fields per entity:
  - `:graph_known` -- boolean, entity exists in knowledge graph
  - `:graph_type` -- graph node label (e.g. "Artist", "City")
  - `:graph_neighbor_count` -- how well the system knows this entity
  - `:graph_neighbors` -- list of neighbor name/type pairs
  - `:graph_relationships` -- paths between co-occurring entities
  """

  alias Brain.Graph.Reader
  require Logger

  @doc """
  Enriches a list of entity maps with knowledge graph context.

  Each entity gains `:graph_known`, `:graph_type`, `:graph_neighbor_count`,
  `:graph_neighbors`, and `:graph_relationships` fields.

  When 2+ entities appear in the same chunk, also discovers
  `relationship_path` between each pair for event understanding.
  """
  def enrich(entities) when is_list(entities) do
    if entities == [] or not atlas_available?() do
      Enum.map(entities, &default_enrichment/1)
    else
      context_results = safe_entity_context(entities)
      enriched = apply_context(entities, context_results)
      apply_relationships(enriched)
    end
  end

  @doc """
  Returns a summary of entity familiarity as a 0.0-1.0 score.

  Useful as an input signal for ContextAccumulator.
  """
  def familiarity_score(entities) when is_list(entities) do
    if entities == [] do
      0.5
    else
      known_count = Enum.count(entities, &Map.get(&1, :graph_known, false))
      total_neighbors = entities |> Enum.map(&Map.get(&1, :graph_neighbor_count, 0)) |> Enum.sum()
      known_ratio = known_count / max(length(entities), 1)
      neighbor_signal = min(total_neighbors / max(length(entities) * 5, 1), 1.0)
      known_ratio * 0.7 + neighbor_signal * 0.3
    end
  end

  defp safe_entity_context(entities) do
    Reader.entity_context(entities, depth: 2)
  rescue
    _ -> Enum.map(entities, fn _ -> %{entity: nil, neighbors: [], node: nil} end)
  catch
    :exit, _ -> Enum.map(entities, fn _ -> %{entity: nil, neighbors: [], node: nil} end)
  end

  defp apply_context(entities, context_results) do
    padded =
      if length(context_results) == length(entities) do
        context_results
      else
        context_results ++
          List.duplicate(
            %{entity: nil, neighbors: [], node: nil},
            max(length(entities) - length(context_results), 0)
          )
      end

    Enum.zip(entities, padded)
    |> Enum.map(fn {entity, ctx} ->
      case ctx do
        %{node: %{properties: props}, neighbors: neighbors} when not is_nil(props) ->
          entity
          |> Map.put(:graph_known, true)
          |> Map.put(:graph_type, Map.get(props, "type", Map.get(props, "label", "")))
          |> Map.put(:graph_neighbor_count, length(neighbors))
          |> Map.put(:graph_neighbors, extract_neighbor_summaries(neighbors))

        _ ->
          default_enrichment(entity)
      end
    end)
  end

  defp apply_relationships(entities) do
    known_entities = Enum.filter(entities, &Map.get(&1, :graph_known, false))

    if length(known_entities) < 2 do
      Enum.map(entities, &Map.put_new(&1, :graph_relationships, []))
    else
      pairs = for a <- known_entities, b <- known_entities, a != b, do: {a, b}

      relationships =
        pairs
        |> Enum.take(10)
        |> Enum.flat_map(fn {a, b} ->
          case safe_relationship_path(a, b) do
            {:ok, path} ->
              a_val = Map.get(a, :value, "")
              b_val = Map.get(b, :value, "")
              [%{from: a_val, to: b_val, path: path}]

            _ ->
              []
          end
        end)

      Enum.map(entities, fn entity ->
        val = Map.get(entity, :value, "")

        relevant =
          Enum.filter(relationships, fn r ->
            r.from == val or r.to == val
          end)

        Map.put(entity, :graph_relationships, relevant)
      end)
    end
  end

  defp safe_relationship_path(entity_a, entity_b) do
    Reader.relationship_path(entity_a, entity_b)
  rescue
    _ -> {:error, :not_found}
  catch
    :exit, _ -> {:error, :not_found}
  end

  defp extract_neighbor_summaries(neighbors) when is_list(neighbors) do
    Enum.map(neighbors, fn neighbor ->
      props = Map.get(neighbor, :properties, %{})

      %{
        name: Map.get(props, "name", ""),
        type: Map.get(props, "type", Map.get(props, "label", ""))
      }
    end)
    |> Enum.reject(fn %{name: n} -> n == "" end)
    |> Enum.take(20)
  end

  defp extract_neighbor_summaries(_), do: []

  defp default_enrichment(entity) do
    entity
    |> Map.put_new(:graph_known, false)
    |> Map.put_new(:graph_type, nil)
    |> Map.put_new(:graph_neighbor_count, 0)
    |> Map.put_new(:graph_neighbors, [])
    |> Map.put_new(:graph_relationships, [])
  end

  defp atlas_available? do
    Brain.AtlasIntegration.available?()
  rescue
    _ -> false
  catch
    :exit, _ -> false
  end
end
