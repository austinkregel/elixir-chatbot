defmodule Mix.Tasks.IngestEntityTypes do
  @shortdoc "Ingest gazetteer entity types and instances into the Knowledge Graph"
  @moduledoc """
  Creates EntityType nodes (with lexicon enrichment), EntityInstance nodes,
  INSTANCE_OF edges, and ALIAS_OF edges for synonyms in the knowledge_graph.

  ## Node Design

  - **EntityType** nodes represent abstract type concepts (e.g., "account")
  - **EntityInstance** nodes represent specific values (e.g., "checking account")
    with `type`, `world_id`, and `source` properties
  - **INSTANCE_OF** edges connect instances to their type
  - **ALIAS_OF** edges connect synonym surface forms to canonical instances

  ## Usage

      mix ingest_entity_types                  # default world
      mix ingest_entity_types --world my_world # specific world

  Requires Atlas to be running. Fails loud with remediation if not.
  """

  use Mix.Task

  alias Atlas.Graph.EdgeLabels

  @non_entity_files ["anaphora", "slot_mappings", "all"]

  @impl true
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [world: :string])
    world_id = Keyword.get(opts, :world, "default")

    Mix.Task.run("app.start")

    unless Brain.AtlasIntegration.available?() do
      Mix.raise(
        "Atlas.Repo must be running for entity type ingestion. " <>
          "Start with docker compose up or use --skip-entity-ingest."
      )
    end

    Mix.shell().info("Loading entity data for world '#{world_id}'...")

    {:ok, entities} = Brain.ML.DataLoaders.load_all_entities()
    ingestable = ingestable_entity_types(entities)

    Mix.shell().info("Found #{map_size(ingestable)} entity types to ingest.")

    Enum.each(ingestable, fn {entity_type, entries} ->
      ingest_type_with_instances(entity_type, entries, world_id)
    end)

    Mix.shell().info("Entity type ingestion complete.")
  end

  defp ingestable_entity_types(entities) do
    entities
    |> Enum.reject(fn {type, _entries} -> type in @non_entity_files end)
    |> Map.new()
  end

  defp ingest_type_with_instances(entity_type, entries, world_id) do
    {:ok, type_node} =
      Brain.AtlasIntegration.ensure_node("knowledge_graph", "EntityType", %{name: entity_type})

    Brain.AtlasIntegration.enrich_existing_node("knowledge_graph", type_node.id, entity_type)

    instance_count = length(entries)
    Mix.shell().info("  #{entity_type}: #{instance_count} entries")

    Enum.each(entries, fn entry ->
      value = Map.get(entry, :value) || Map.get(entry, "value", "")
      synonyms = Map.get(entry, :synonyms) || Map.get(entry, "synonyms", [])

      if value != "" do
        {:ok, instance_node} =
          Brain.AtlasIntegration.ensure_node("knowledge_graph", "EntityInstance", %{
            name: value,
            type: entity_type,
            world_id: world_id,
            source: "gazetteer"
          })

        Brain.AtlasIntegration.find_or_create_edge(
          "knowledge_graph",
          instance_node.id,
          type_node.id,
          EdgeLabels.instance_of(),
          %{source: "gazetteer"}
        )

        Enum.each(synonyms, fn synonym ->
          if is_binary(synonym) and synonym != "" and synonym != value do
            {:ok, alias_node} =
              Brain.AtlasIntegration.ensure_node("knowledge_graph", "EntityInstance", %{
                name: synonym,
                type: entity_type,
                world_id: world_id,
                source: "gazetteer",
                is_alias: true
              })

            Brain.AtlasIntegration.find_or_create_edge(
              "knowledge_graph",
              alias_node.id,
              instance_node.id,
              EdgeLabels.alias_of(),
              %{source: "gazetteer"}
            )
          end
        end)
      end
    end)
  end
end
