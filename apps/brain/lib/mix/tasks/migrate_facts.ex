defmodule Mix.Tasks.MigrateFacts do
  @moduledoc """
  Migrates fact JSON files to the new standardized schema.

  This task:
  1. Renames `entity_type` to `entity` (when it contains a subject name)
  2. Adds `entity_type` field with inferred classification
  3. Deduplicates entries in learned.json

  ## Usage

      mix migrate_facts

  ## Options

      --dry-run    Show what would be changed without writing files
      --verbose    Show detailed progress

  """
  use Mix.Task

  alias Brain.FactDatabase.Fact

  @shortdoc "Migrate fact JSON files to standardized schema"

  @facts_dir "data/facts"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} = OptionParser.parse(args, switches: [dry_run: :boolean, verbose: :boolean])
    dry_run? = Keyword.get(opts, :dry_run, false)
    verbose? = Keyword.get(opts, :verbose, false)

    Mix.shell().info("Migrating fact files to standardized schema...")

    if dry_run? do
      Mix.shell().info("(Dry run mode - no files will be modified)")
    end

    facts_dir = Path.join([File.cwd!(), @facts_dir])

    unless File.exists?(facts_dir) do
      Mix.shell().error("Facts directory not found: #{facts_dir}")
      System.halt(1)
    end

    # Get all JSON files except README
    json_files =
      facts_dir
      |> Path.join("*.json")
      |> Path.wildcard()

    {total_migrated, total_deduplicated} =
      Enum.reduce(json_files, {0, 0}, fn file_path, {migrated_acc, dedup_acc} ->
        filename = Path.basename(file_path)

        case migrate_file(file_path, dry_run?, verbose?) do
          {:ok, migrated, deduped} ->
            if verbose? or migrated > 0 or deduped > 0 do
              Mix.shell().info("  #{filename}: #{migrated} facts migrated, #{deduped} duplicates removed")
            end

            {migrated_acc + migrated, dedup_acc + deduped}

          {:error, reason} ->
            Mix.shell().error("  #{filename}: Error - #{inspect(reason)}")
            {migrated_acc, dedup_acc}
        end
      end)

    Mix.shell().info("")
    Mix.shell().info("Migration complete!")
    Mix.shell().info("  Facts migrated: #{total_migrated}")
    Mix.shell().info("  Duplicates removed: #{total_deduplicated}")

    if dry_run? do
      Mix.shell().info("")
      Mix.shell().info("Run without --dry-run to apply changes.")
    end
  end

  defp migrate_file(file_path, dry_run?, verbose?) do
    case File.read(file_path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} ->
            original_facts = Map.get(data, "facts", [])
            is_learned = Path.basename(file_path) == "learned.json"

            # Migrate each fact
            migrated_facts =
              Enum.map(original_facts, fn fact ->
                migrate_fact(fact, verbose?)
              end)

            # Deduplicate if this is learned.json
            {final_facts, dedup_count} =
              if is_learned do
                deduplicate_facts(migrated_facts)
              else
                {migrated_facts, 0}
              end

            # Count how many facts were actually changed
            migrated_count =
              Enum.zip(original_facts, migrated_facts)
              |> Enum.count(fn {old, new} -> old != new end)

            # Write back if not dry run and there are changes
            if not dry_run? and (migrated_count > 0 or dedup_count > 0) do
              updated_data = Map.put(data, "facts", final_facts)
              File.write!(file_path, Jason.encode!(updated_data, pretty: true))
            end

            {:ok, migrated_count, dedup_count}

          {:error, reason} ->
            {:error, {:json_parse, reason}}
        end

      {:error, reason} ->
        {:error, {:file_read, reason}}
    end
  end

  defp migrate_fact(fact, verbose?) do
    # Check if fact needs migration
    has_entity = Map.has_key?(fact, "entity")
    has_entity_type = Map.has_key?(fact, "entity_type")

    cond do
      # Already migrated - has both fields
      has_entity and has_entity_type ->
        fact

      # Needs migration - has entity_type but no entity (legacy format)
      has_entity_type and not has_entity ->
        entity_name = fact["entity_type"]
        category = fact["category"] || "unknown"
        inferred_type = Fact.infer_entity_type(entity_name, category)

        if verbose? do
          Mix.shell().info("    Migrating: #{entity_name} -> entity_type: #{inferred_type}")
        end

        fact
        |> Map.put("entity", entity_name)
        |> Map.put("entity_type", inferred_type)

      # Has entity but no entity_type - add inferred type
      has_entity and not has_entity_type ->
        entity_name = fact["entity"]
        category = fact["category"] || "unknown"
        inferred_type = Fact.infer_entity_type(entity_name, category)

        if verbose? do
          Mix.shell().info("    Adding entity_type: #{inferred_type} for #{entity_name}")
        end

        Map.put(fact, "entity_type", inferred_type)

      # Neither field - use defaults
      true ->
        fact
        |> Map.put_new("entity", "unknown")
        |> Map.put_new("entity_type", "unknown")
    end
  end

  defp deduplicate_facts(facts) do
    # Deduplicate based on entity + fact text
    {unique_facts, _seen} =
      Enum.reduce(facts, {[], MapSet.new()}, fn fact, {acc, seen} ->
        key = {
          String.downcase(fact["entity"] || ""),
          String.downcase(fact["fact"] || "")
        }

        if MapSet.member?(seen, key) do
          {acc, seen}
        else
          {[fact | acc], MapSet.put(seen, key)}
        end
      end)

    # Reverse to maintain original order (newest first)
    dedup_count = length(facts) - length(unique_facts)
    {Enum.reverse(unique_facts), dedup_count}
  end
end
