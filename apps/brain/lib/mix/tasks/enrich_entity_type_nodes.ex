defmodule Mix.Tasks.EnrichEntityTypeNodes do
  @shortdoc "Backfill WordNet enrichment on existing EntityType nodes"
  @moduledoc """
  One-shot migration that enriches existing bare EntityType nodes in the
  knowledge_graph with WordNet/ConceptNet semantic properties.

  For each EntityType node, calls `Brain.AtlasIntegration.enrich_existing_node/4`
  which adds whitelisted lexicon properties and creates LexiconFacet nodes
  connected via HAS_LEXICON_FACET edges.

  Idempotent -- safe to re-run. `ensure_node` short-circuits for existing
  LexiconFacet nodes; property SETs overwrite with the same values.

  ## Usage

      mix enrich_entity_type_nodes
  """

  use Mix.Task

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    unless Brain.AtlasIntegration.available?() do
      Mix.raise("Atlas is not reachable. Cannot enrich EntityType nodes. Start Atlas with docker compose up.")
    end

    Mix.shell().info("Querying existing EntityType nodes...")

    case Atlas.Graph.cypher("knowledge_graph",
           "MATCH (n:EntityType) RETURN n.name AS name, id(n) AS node_id"
         ) do
      {:ok, rows} when is_list(rows) ->
        count = length(rows)
        Mix.shell().info("Found #{count} EntityType nodes to enrich.")

        Enum.each(rows, fn row ->
          name = get_field(row, "name")
          node_id = get_field(row, "node_id")

          if name && node_id do
            Mix.shell().info("  Enriching: #{name}")
            Brain.AtlasIntegration.enrich_existing_node("knowledge_graph", node_id, name)
          end
        end)

        if Code.ensure_loaded?(Brain.Analysis.TypeHierarchy) and
             function_exported?(Brain.Analysis.TypeHierarchy, :reload, 0) do
          Brain.Analysis.TypeHierarchy.reload()
          Mix.shell().info("TypeHierarchy ETS reloaded.")
        end

        Mix.shell().info("Enrichment complete for #{count} EntityType nodes.")

      {:ok, []} ->
        Mix.shell().info("No EntityType nodes found in knowledge_graph.")

      {:error, reason} ->
        Mix.raise("Failed to query EntityType nodes: #{inspect(reason)}")
    end
  end

  defp get_field([value], _key), do: value

  defp get_field(row, key) when is_map(row) do
    Map.get(row, key) || Map.get(row, String.to_atom(key))
  end

  defp get_field(row, _key) when is_list(row) do
    List.first(row)
  end

  defp get_field(_, _), do: nil
end
