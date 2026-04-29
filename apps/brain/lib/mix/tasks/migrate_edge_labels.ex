defmodule Mix.Tasks.MigrateEdgeLabels do
  @shortdoc "Migrate legacy edge labels to SCREAMING_SNAKE canonical form"
  @moduledoc """
  One-shot migration that renames legacy edge types in AGE graphs
  to the canonical SCREAMING_SNAKE form defined in Atlas.Graph.EdgeLabels.

  Preserves all edge properties (count, source, etc.) during migration.
  Idempotent -- safe to re-run; only processes edges with legacy labels.

  ## Usage

      mix migrate_edge_labels
  """

  use Mix.Task

  @migrations [
    {"knowledge_graph", "is_a", "IS_A"},
    {"knowledge_graph", "co_occurs_with", "CO_OCCURS_WITH"}
  ]

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    unless Brain.AtlasIntegration.available?() do
      Mix.raise("Atlas is not reachable. Cannot migrate edge labels. Start Atlas with docker compose up.")
    end

    Mix.shell().info("Atlas is reachable. Starting edge label migration...")

    Enum.each(@migrations, fn {graph, old_label, new_label} ->
      migrate_label(graph, old_label, new_label)
    end)

    Mix.shell().info("Edge label migration complete.")
  end

  defp migrate_label(graph, old_label, new_label) do
    count_query = "MATCH ()-[r:#{old_label}]->() RETURN count(r)"

    case Atlas.Graph.cypher(graph, count_query) do
      {:ok, [[count]]} when is_integer(count) and count > 0 ->
        Mix.shell().info("  #{graph}: migrating #{count} #{old_label} -> #{new_label} edges...")

        migrate_query = """
        MATCH (a)-[r:#{old_label}]->(b)
        CREATE (a)-[r2:#{new_label}]->(b)
        SET r2 = properties(r)
        DELETE r
        """

        case Atlas.Graph.cypher(graph, migrate_query) do
          {:ok, _} ->
            Mix.shell().info("  #{graph}: migrated #{old_label} -> #{new_label}")

          {:error, reason} ->
            Mix.shell().error("  #{graph}: migration failed for #{old_label}: #{inspect(reason)}")
        end

      {:ok, [[0]]} ->
        Mix.shell().info("  #{graph}: no #{old_label} edges found (already migrated or none exist)")

      {:ok, []} ->
        Mix.shell().info("  #{graph}: no #{old_label} edges found")

      {:error, reason} ->
        Mix.shell().error("  #{graph}: could not count #{old_label} edges: #{inspect(reason)}")
    end
  end
end
