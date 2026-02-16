defmodule Mix.Tasks.TrainingWorld do
  @moduledoc "Mix tasks for managing training worlds.\n\n## Commands\n\n    # Create a new world\n    mix training_world.create \"world_name\" [--mode=ephemeral|persistent]\n\n    # Ingest files into a world\n    mix training_world.ingest \"world_id\" \"path/to/files/*.txt\"\n\n    # View world metrics\n    mix training_world.metrics \"world_id\"\n\n    # View discovered entities\n    mix training_world.entities \"world_id\" [--sort=confidence|occurrences]\n\n    # View ambiguous entities\n    mix training_world.ambiguous \"world_id\"\n\n    # View events\n    mix training_world.events \"world_id\" [--type=event_type] [--limit=100]\n\n    # Compare two worlds\n    mix training_world.compare \"world_id_1\" \"world_id_2\"\n\n    # Export world data\n    mix training_world.export \"world_id\" [--output=file.json]\n\n    # Merge worlds\n    mix training_world.merge \"source_id\" \"target_id\" [--require-review]\n\n    # List all worlds\n    mix training_world.list\n\n    # Destroy a world\n    mix training_world.destroy \"world_id\"\n\n    # Checkpoint a persistent world\n    mix training_world.checkpoint \"world_id\"\n\n    # Load a persisted world\n    mix training_world.load \"world_id\"\n"

  use Mix.Task

  @shortdoc "Training world management commands"

  @impl Mix.Task
  def run(_args) do
    Mix.shell().info("Usage: mix training_world.<command> [args]")
    Mix.shell().info("")
    Mix.shell().info("Available commands:")
    Mix.shell().info("  create     - Create a new training world")
    Mix.shell().info("  ingest     - Ingest files into a world")
    Mix.shell().info("  metrics    - View world metrics")
    Mix.shell().info("  entities   - View discovered entities")
    Mix.shell().info("  ambiguous  - View ambiguous entities")
    Mix.shell().info("  events     - View world events")
    Mix.shell().info("  compare    - Compare two worlds")
    Mix.shell().info("  export     - Export world data")
    Mix.shell().info("  merge      - Merge source world into target")
    Mix.shell().info("  list       - List all worlds")
    Mix.shell().info("  destroy    - Destroy a world")
    Mix.shell().info("  checkpoint - Save persistent world to disk")
    Mix.shell().info("  load       - Load persisted world from disk")
    Mix.shell().info("  import     - Import entities from JSON file")
    Mix.shell().info("")
    Mix.shell().info("Run 'mix help training_world.<command>' for more info")
  end
end

defmodule Mix.Tasks.TrainingWorld.Create do
  alias World.Manager
  use Mix.Task

  @shortdoc "Create a new training world"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} = OptionParser.parse(args, strict: [mode: :string])

    case rest do
      [name] ->
        ensure_started()

        mode =
          case Keyword.get(opts, :mode, "ephemeral") do
            "persistent" -> :persistent
            _ -> :ephemeral
          end

        case Manager.create(name, mode: mode) do
          {:ok, world} ->
            Mix.shell().info("Created training world: #{world.id}")
            Mix.shell().info("  Name: #{world.name}")
            Mix.shell().info("  Mode: #{world.mode}")

          {:error, reason} ->
            Mix.shell().error("Failed to create world: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.create <name> [--mode=ephemeral|persistent]")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Ingest do
  alias World.DocumentIngestor
  use Mix.Task

  @shortdoc "Ingest files into a training world"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} =
      OptionParser.parse(args, strict: [chunk_size: :integer, stream: :boolean])

    case rest do
      [world_id, pattern] ->
        ensure_started()

        files = Path.wildcard(pattern)

        if files == [] do
          Mix.shell().error("No files found matching: #{pattern}")
        else
          Mix.shell().info("Found #{length(files)} files to ingest")

          progress_callback = fn progress ->
            case progress.type do
              :file_started ->
                Mix.shell().info(
                  "Processing [#{progress.current}/#{progress.total}]: #{progress.file}"
                )

              :file_completed ->
                Mix.shell().info(
                  "  Completed: #{progress.result.entities_discovered} entities discovered"
                )

              :file_failed ->
                Mix.shell().error("  Failed: #{inspect(progress.error)}")

              _ ->
                :ok
            end
          end

          ingest_opts = [
            chunk_size: Keyword.get(opts, :chunk_size, 5000),
            progress_callback: progress_callback
          ]

          case DocumentIngestor.ingest_files(world_id, files, ingest_opts) do
            {:ok, result} ->
              Mix.shell().info("")
              Mix.shell().info("Ingestion complete:")
              Mix.shell().info("  Documents processed: #{result.documents_processed}")
              Mix.shell().info("  Total chunks: #{result.total_chunks}")
              Mix.shell().info("  Total tokens: #{result.total_tokens}")
              Mix.shell().info("  Entities discovered: #{result.entities_discovered}")
              Mix.shell().info("  Processing time: #{result.processing_time_ms}ms")

              if result.failed_files != [] do
                Mix.shell().error("  Failed files: #{length(result.failed_files)}")
              end

            other ->
              Mix.shell().error("Unexpected result: #{inspect(other)}")
          end
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.ingest <world_id> <file_pattern>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Metrics do
  alias World.Metrics
  alias World.Manager
  use Mix.Task

  @shortdoc "View training world metrics"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id] ->
        ensure_started()

        case Manager.get_metrics(world_id) do
          {:ok, metrics} ->
            summary = Metrics.summary(metrics)

            Mix.shell().info("World Metrics: #{world_id}")
            Mix.shell().info("=" |> String.duplicate(50))
            Mix.shell().info("Documents processed: #{summary.documents}")
            Mix.shell().info("Total tokens: #{summary.tokens}")
            Mix.shell().info("Total sentences: #{summary.sentences}")
            Mix.shell().info("")
            Mix.shell().info("Entities discovered: #{summary.entities_discovered}")
            Mix.shell().info("Entities promoted: #{summary.entities_promoted}")
            Mix.shell().info("Entity types: #{Enum.join(summary.entity_types, ", ")}")
            Mix.shell().info("")
            Mix.shell().info("Ambiguities: #{summary.ambiguity_count}")
            Mix.shell().info("Anomalies: #{summary.anomaly_count}")
            Mix.shell().info("Type conflicts: #{summary.conflict_count}")
            Mix.shell().info("")
            Mix.shell().info("Processing time: #{summary.processing_time_ms}ms")

            if summary.started_at do
              Mix.shell().info("Started: #{DateTime.to_string(summary.started_at)}")
            end

            if summary.last_updated do
              Mix.shell().info("Last updated: #{DateTime.to_string(summary.last_updated)}")
            end

          {:error, reason} ->
            Mix.shell().error("Failed to get metrics: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.metrics <world_id>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Entities do
  alias World.Manager
  use Mix.Task

  @shortdoc "View discovered entities in a training world"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} =
      OptionParser.parse(args, strict: [sort: :string, limit: :integer])

    case rest do
      [world_id] ->
        ensure_started()

        sort =
          case Keyword.get(opts, :sort, "occurrences") do
            "confidence" -> :confidence
            _ -> :occurrences
          end

        limit = Keyword.get(opts, :limit, 50)

        candidates =
          Manager.get_candidates(world_id, sort: sort, limit: limit)

        if candidates == [] do
          Mix.shell().info("No entity candidates found")
        else
          Mix.shell().info("Entity Candidates (sorted by #{sort}):")
          Mix.shell().info("=" |> String.duplicate(70))

          Mix.shell().info(
            String.pad_trailing("Value", 25) <>
              String.pad_trailing("Type", 15) <>
              String.pad_trailing("Occurrences", 12) <>
              "Confidence"
          )

          Mix.shell().info("-" |> String.duplicate(70))

          Enum.each(candidates, fn candidate ->
            value = String.slice(Map.get(candidate, :value, ""), 0, 24)
            type = String.slice(Map.get(candidate, :inferred_type, "unknown"), 0, 14)
            occurrences = Map.get(candidate, :occurrences, 1)
            confidence = Map.get(candidate, :confidence, 0.0)

            Mix.shell().info(
              String.pad_trailing(value, 25) <>
                String.pad_trailing(type, 15) <>
                String.pad_trailing("#{occurrences}", 12) <>
                Float.to_string(Float.round(confidence, 3))
            )
          end)

          Mix.shell().info("")
          Mix.shell().info("Total: #{length(candidates)} candidates")
        end

      _ ->
        Mix.shell().error(
          "Usage: mix training_world.entities <world_id> [--sort=confidence|occurrences] [--limit=N]"
        )
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Ambiguous do
  alias World.Manager
  use Mix.Task

  @shortdoc "View ambiguous entities in a training world"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id] ->
        ensure_started()

        case Manager.get_metrics(world_id) do
          {:ok, metrics} ->
            ambiguities = metrics.ambiguous_entities

            if ambiguities == [] do
              Mix.shell().info("No ambiguous entities found")
            else
              Mix.shell().info("Ambiguous Entities (need human review):")
              Mix.shell().info("=" |> String.duplicate(70))

              Enum.each(ambiguities, fn amb ->
                value = Map.get(amb, :value, "unknown")
                types = Map.get(amb, :types, []) |> Enum.join(", ")
                context = Map.get(amb, :context, "") |> String.slice(0, 50)

                Mix.shell().info("")
                Mix.shell().info("Value: #{value}")
                Mix.shell().info("Possible types: #{types}")
                Mix.shell().info("Context: \"#{context}...\"")
              end)

              Mix.shell().info("")
              Mix.shell().info("Total: #{length(ambiguities)} ambiguous entities")
            end

          {:error, reason} ->
            Mix.shell().error("Failed to get metrics: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.ambiguous <world_id>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Events do
  alias World.Manager
  use Mix.Task

  @shortdoc "View events in a training world"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} =
      OptionParser.parse(args, strict: [type: :string, limit: :integer])

    case rest do
      [world_id] ->
        ensure_started()

        filters = []

        filters =
          if t = Keyword.get(opts, :type) do
            [type: safe_event_type(t)] ++ filters
          else
            filters
          end

        filters = [limit: Keyword.get(opts, :limit, 50)] ++ filters

        events = Manager.get_events(world_id, filters)

        if events == [] do
          Mix.shell().info("No events found")
        else
          Mix.shell().info("Events:")
          Mix.shell().info("=" |> String.duplicate(80))

          Enum.each(events, fn event ->
            timestamp = DateTime.to_string(event.timestamp)
            type = Atom.to_string(event.type)
            data_preview = inspect(event.data) |> String.slice(0, 40)

            Mix.shell().info("[#{timestamp}] #{type}")
            Mix.shell().info("  #{data_preview}...")
          end)

          Mix.shell().info("")
          Mix.shell().info("Total: #{length(events)} events")
        end

      _ ->
        Mix.shell().error(
          "Usage: mix training_world.events <world_id> [--type=event_type] [--limit=N]"
        )
    end
  end

  defp safe_event_type(t) when is_binary(t) do
    String.to_existing_atom(t)
  rescue
    ArgumentError ->
      Mix.raise("Unknown event type: #{t}. Use an existing atom like :entity_discovered")
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Compare do
  alias World.Manager
  use Mix.Task

  @shortdoc "Compare two training worlds"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id_1, world_id_2] ->
        ensure_started()

        case Manager.compare(world_id_1, world_id_2) do
          {:ok, diff} ->
            Mix.shell().info("World Comparison")
            Mix.shell().info("=" |> String.duplicate(50))
            Mix.shell().info("World 1: #{world_id_1}")
            Mix.shell().info("World 2: #{world_id_2}")
            Mix.shell().info("")
            Mix.shell().info("Entity count difference: #{diff.entity_count_diff}")
            Mix.shell().info("Promoted difference: #{diff.promoted_diff}")
            Mix.shell().info("Processing time difference: #{diff.processing_time_diff_ms}ms")
            Mix.shell().info("")

            if map_size(diff.type_distribution_diff) > 0 do
              Mix.shell().info("Type distribution differences:")

              Enum.each(diff.type_distribution_diff, fn {type, count_diff} ->
                sign =
                  if count_diff > 0 do
                    "+"
                  else
                    ""
                  end

                Mix.shell().info("  #{type}: #{sign}#{count_diff}")
              end)
            end

            if diff.unique_to_world1 != [] do
              Mix.shell().info("")

              Mix.shell().info(
                "Types unique to world 1: #{Enum.join(diff.unique_to_world1, ", ")}"
              )
            end

            if diff.unique_to_world2 != [] do
              Mix.shell().info(
                "Types unique to world 2: #{Enum.join(diff.unique_to_world2, ", ")}"
              )
            end

          {:error, reason} ->
            Mix.shell().error("Failed to compare worlds: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.compare <world_id_1> <world_id_2>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Export do
  alias World.Manager
  use Mix.Task

  @shortdoc "Export training world data"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} = OptionParser.parse(args, strict: [output: :string])

    case rest do
      [world_id] ->
        ensure_started()

        case Manager.export(world_id) do
          {:ok, data} ->
            output_file = Keyword.get(opts, :output, "#{world_id}_export.json")
            export_data = prepare_for_json(data)

            case Jason.encode(export_data, pretty: true) do
              {:ok, json} ->
                File.write!(output_file, json)
                Mix.shell().info("Exported world data to: #{output_file}")

              {:error, reason} ->
                Mix.shell().error("Failed to encode data: #{inspect(reason)}")
            end

          {:error, reason} ->
            Mix.shell().error("Failed to export world: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.export <world_id> [--output=file.json]")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end

  defp prepare_for_json(data) do
    data
    |> Map.update(:world, %{}, &struct_to_map/1)
    |> Map.update(:metrics, %{}, &struct_to_map/1)
    |> Map.update(:events, [], fn events ->
      Enum.map(events, &struct_to_map/1)
    end)
  end

  defp struct_to_map(%_{} = struct) do
    struct
    |> Map.from_struct()
    |> Enum.into(%{}, fn {k, v} ->
      {k, prepare_value(v)}
    end)
  end

  defp struct_to_map(other) do
    prepare_value(other)
  end

  defp prepare_value(%DateTime{} = dt) do
    DateTime.to_iso8601(dt)
  end

  defp prepare_value(%_{} = struct) do
    struct_to_map(struct)
  end

  defp prepare_value(map) when is_map(map) do
    Enum.into(map, %{}, fn {k, v} -> {k, prepare_value(v)} end)
  end

  defp prepare_value(list) when is_list(list) do
    Enum.map(list, &prepare_value/1)
  end

  defp prepare_value(atom) when is_atom(atom) do
    Atom.to_string(atom)
  end

  defp prepare_value(other) do
    other
  end
end

defmodule Mix.Tasks.TrainingWorld.Merge do
  alias World.Manager
  use Mix.Task

  @shortdoc "Merge source world into target world"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} =
      OptionParser.parse(args, strict: [require_review: :boolean, min_confidence: :float])

    case rest do
      [source_id, target_id] ->
        ensure_started()

        merge_opts = [
          require_review: Keyword.get(opts, :require_review, true),
          min_confidence: Keyword.get(opts, :min_confidence, 0.7)
        ]

        case Manager.merge(source_id, target_id, merge_opts) do
          {:ok, count} ->
            Mix.shell().info("Merged #{count} entities from #{source_id} to #{target_id}")

          {:needs_review, entities} ->
            Mix.shell().info("The following entities need review before merging:")
            Mix.shell().info("")

            Enum.each(entities, fn {key, info} ->
              Mix.shell().info(
                "  #{key}: #{Map.get(info, :entity_type)} (confidence: #{Map.get(info, :confidence, "N/A")})"
              )
            end)

            Mix.shell().info("")
            Mix.shell().info("Run with --require-review=false to merge without review")

          {:error, reason} ->
            Mix.shell().error("Merge failed: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error(
          "Usage: mix training_world.merge <source_id> <target_id> [--require-review=true] [--min-confidence=0.7]"
        )
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.List do
  alias World.Persistence
  alias World.Manager
  use Mix.Task

  @shortdoc "List all training worlds"

  @impl Mix.Task
  def run(_args) do
    ensure_started()
    active_worlds = Manager.list_worlds()
    persisted_worlds = Persistence.list_persisted_worlds()

    Mix.shell().info("Active Training Worlds:")
    Mix.shell().info("=" |> String.duplicate(60))

    if active_worlds == [] do
      Mix.shell().info("  No active worlds")
    else
      Enum.each(active_worlds, fn world ->
        Mix.shell().info("  #{world.id}")
        Mix.shell().info("    Name: #{world.name}")
        Mix.shell().info("    Mode: #{world.mode}")
        Mix.shell().info("    Created: #{DateTime.to_string(world.created_at)}")
      end)
    end

    Mix.shell().info("")
    Mix.shell().info("Persisted Worlds (on disk):")
    Mix.shell().info("=" |> String.duplicate(60))

    if persisted_worlds == [] do
      Mix.shell().info("  No persisted worlds")
    else
      Enum.each(persisted_worlds, fn world ->
        active =
          if Enum.any?(active_worlds, &(&1.id == world.id)) do
            " [ACTIVE]"
          else
            ""
          end

        Mix.shell().info("  #{world.id}#{active}")
        Mix.shell().info("    Name: #{world.name}")
      end)
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Destroy do
  alias World.Manager
  use Mix.Task

  @shortdoc "Destroy a training world"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id] ->
        ensure_started()

        Mix.shell().info("Destroying world: #{world_id}")

        case Manager.destroy(world_id) do
          :ok ->
            Mix.shell().info("World destroyed successfully")

          {:error, reason} ->
            Mix.shell().error("Failed to destroy world: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.destroy <world_id>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Checkpoint do
  alias World.Manager
  use Mix.Task

  @shortdoc "Save a persistent world to disk"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id] ->
        ensure_started()

        Mix.shell().info("Creating checkpoint for world: #{world_id}")

        case Manager.checkpoint(world_id) do
          :ok ->
            Mix.shell().info("Checkpoint created successfully")

          {:error, :ephemeral_world} ->
            Mix.shell().error("Cannot checkpoint ephemeral world. Create with --mode=persistent")

          {:error, reason} ->
            Mix.shell().error("Checkpoint failed: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.checkpoint <world_id>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Load do
  alias World.Manager
  use Mix.Task

  @shortdoc "Load a persisted world from disk"

  @impl Mix.Task
  def run(args) do
    case args do
      [world_id] ->
        ensure_started()

        Mix.shell().info("Loading world from disk: #{world_id}")

        case Manager.load_world(world_id) do
          {:ok, world} ->
            Mix.shell().info("World loaded successfully")
            Mix.shell().info("  Name: #{world.name}")
            Mix.shell().info("  Mode: #{world.mode}")

          {:error, :not_found} ->
            Mix.shell().error("World not found on disk")

          {:error, reason} ->
            Mix.shell().error("Load failed: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error("Usage: mix training_world.load <world_id>")
    end
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end

defmodule Mix.Tasks.TrainingWorld.Import do
  alias World.Manager
  alias Brain.ML.Gazetteer
  use Mix.Task

  @shortdoc "Import entities from a JSON file into a training world"

  @moduledoc "Imports pre-extracted entities into a training world's gazetteer overlay.\n\nThis is useful when you've already identified entities through preprocessing\nand want to seed a training world with them.\n\n## Usage\n\n    mix training_world.import <world_id> <entities.json> [options]\n\n## Options\n\n  * `--promote` - Automatically promote entities to the gazetteer (default: false)\n  * `--skip-unknown` - Skip entities with type \"unknown\" (default: false)\n\n## JSON Format\n\nThe JSON file should have an \"entities\" array with objects containing:\n  - `value` - The entity text\n  - `entity_type` - The type (person, location, etc.)\n  - `metadata` - Optional metadata\n"

  @impl Mix.Task
  def run(args) do
    {opts, rest, _} =
      OptionParser.parse(args, strict: [promote: :boolean, skip_unknown: :boolean])

    case rest do
      [world_id, json_file] ->
        ensure_started()

        case File.read(json_file) do
          {:ok, content} ->
            case Jason.decode(content) do
              {:ok, data} ->
                import_entities(world_id, data, opts)

              {:error, reason} ->
                Mix.shell().error("Failed to parse JSON: #{inspect(reason)}")
            end

          {:error, reason} ->
            Mix.shell().error("Failed to read file: #{inspect(reason)}")
        end

      _ ->
        Mix.shell().error(
          "Usage: mix training_world.import <world_id> <entities.json> [--promote] [--skip-unknown]"
        )
    end
  end

  defp import_entities(world_id, data, opts) do
    promote = Keyword.get(opts, :promote, false)
    skip_unknown = Keyword.get(opts, :skip_unknown, false)

    entities = Map.get(data, "entities", [])

    Mix.shell().info("Importing #{length(entities)} entities into world: #{world_id}")

    case Manager.get(world_id) do
      {:error, :not_found} ->
        Mix.shell().error("World not found: #{world_id}")
        return()

      {:ok, _world} ->
        :ok
    end

    {imported, skipped, promoted} =
      Enum.reduce(entities, {0, 0, 0}, fn entity, {imp, skip, prom} ->
        entity_type = Map.get(entity, "entity_type", "unknown")
        value = Map.get(entity, "value", "")

        cond do
          skip_unknown and entity_type == "unknown" ->
            {imp, skip + 1, prom}

          String.trim(value) == "" ->
            {imp, skip + 1, prom}

          true ->
            candidate = %{
              value: value,
              inferred_type: entity_type,
              confidence:
                if(entity_type == "unknown") do
                  0.5
                else
                  0.9
                end,
              source: :import,
              metadata: Map.get(entity, "metadata", %{}),
              discovered_at: DateTime.utc_now()
            }

            Manager.add_candidate(world_id, candidate)

            if promote and entity_type != "unknown" do
              Gazetteer.add_to_world(world_id, value, entity_type)
              {imp + 1, skip, prom + 1}
            else
              {imp + 1, skip, prom}
            end
        end
      end)

    Mix.shell().info("")
    Mix.shell().info("Import complete:")
    Mix.shell().info("  Imported: #{imported}")
    Mix.shell().info("  Skipped: #{skipped}")

    if promote do
      Mix.shell().info("  Promoted to gazetteer: #{promoted}")
    else
      Mix.shell().info("  (Use --promote to add to gazetteer)")
    end

    case Manager.get(world_id) do
      {:ok, world} when world.mode == :persistent ->
        Mix.shell().info("")
        Mix.shell().info("Saving checkpoint...")
        Manager.checkpoint(world_id)
        Mix.shell().info("Checkpoint saved.")

      _ ->
        :ok
    end
  end

  defp return do
    :ok
  end

  defp ensure_started do
    Mix.Task.run("app.start")
  end
end
