defmodule Mix.Tasks.Migrate.EpisodeEntityNames do
  @moduledoc """
  Backfills the `entity_names` field on all episodes stored in Atlas.

  Episodes created before the `entity_names` column was added will have
  an empty list. This task extracts entity names from the episode state
  text and persists them.

  ## Usage

      mix migrate.episode_entity_names

  ## Options

      --world-id   World to backfill (default: all worlds)
      --dry-run    Show what would be changed without writing
  """
  use Mix.Task

  require Logger

  @shortdoc "Backfill entity_names on episodes"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, strict: [world_id: :string, dry_run: :boolean])
    dry_run? = Keyword.get(opts, :dry_run, false)

    Mix.Task.run("app.start")

    worlds =
      case Keyword.get(opts, :world_id) do
        nil ->
          case Brain.AtlasIntegration.list_memory_worlds() do
            {:ok, ws} -> ws
            _ -> ["default"]
          end

        w ->
          [w]
      end

    total = Enum.reduce(worlds, 0, fn world_id, acc ->
      acc + backfill_world(world_id, dry_run?)
    end)

    Mix.shell().info("Backfill complete: #{total} episode(s) updated across #{length(worlds)} world(s)")
  end

  defp backfill_world(world_id, dry_run?) do
    case Brain.AtlasIntegration.load_episodes(world_id) do
      {:ok, episodes} when episodes != %{} ->
        needs_backfill =
          Enum.filter(episodes, fn {_id, ep} ->
            ep.entity_names == nil or ep.entity_names == []
          end)

        if needs_backfill == [] do
          Mix.shell().info("World #{world_id}: all #{map_size(episodes)} episode(s) already have entity_names")
          0
        else
          Mix.shell().info("World #{world_id}: backfilling #{length(needs_backfill)} of #{map_size(episodes)} episode(s)")

          if dry_run? do
            Enum.each(needs_backfill, fn {id, ep} ->
              text = ep.state || ep.action || ""
              names = do_extract(text)
              Mix.shell().info("  [dry-run] #{id}: #{inspect(names)}")
            end)

            0
          else
            Enum.each(needs_backfill, fn {_id, ep} ->
              text = ep.state || ep.action || ""
              names = do_extract(text)
              updated_ep = %{ep | entity_names: names}
              Brain.AtlasIntegration.persist_episode_sync(updated_ep, world_id)
            end)

            length(needs_backfill)
          end
        end

      _ ->
        Mix.shell().info("World #{world_id}: no episodes found")
        0
    end
  end

  defp do_extract(text) when is_binary(text) and text != "" do
    case Brain.ML.EntityExtractor.extract_entities(text, skip_disambiguation: true) do
      entities when is_list(entities) ->
        Enum.map(entities, fn e ->
          Map.get(e, :value) || Map.get(e, :text, "")
        end)
        |> Enum.reject(&(&1 == ""))

      _ ->
        []
    end
  rescue
    _ -> []
  end

  defp do_extract(_), do: []
end
