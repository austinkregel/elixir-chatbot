defmodule Mix.Tasks.CleanupWorlds do
  @moduledoc "Cleans up orphaned training world directories from disk.\n\nAn orphaned directory is one that:\n- Has no valid config.json file, OR\n- Is older than the specified age\n\n## Usage\n\n    # Dry run (default) - shows what would be deleted\n    mix cleanup_worlds\n\n    # Actually delete orphaned directories\n    mix cleanup_worlds --delete\n\n    # Delete directories older than 12 hours\n    mix cleanup_worlds --delete --max-age 12\n\n    # Exclude specific world IDs from cleanup\n    mix cleanup_worlds --delete --exclude my_world --exclude another_world\n\n## Options\n\n  * `--delete` - Actually delete the directories (default is dry run)\n  * `--max-age` - Maximum age in hours before a directory is considered orphaned (default: 24)\n  * `--exclude` - World IDs to exclude from cleanup (can be specified multiple times)\n\n"
  use Mix.Task

  require Logger

  @shortdoc "Cleans up orphaned training world directories"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [delete: :boolean, max_age: :integer, exclude: [:string, :keep]],
        aliases: [d: :delete, m: :max_age, e: :exclude]
      )

    dry_run = not Keyword.get(opts, :delete, false)
    max_age_hours = Keyword.get(opts, :max_age, 24)
    exclude = Keyword.get_values(opts, :exclude) ++ ["default"]

    Mix.shell().info("Training World Cleanup")
    Mix.shell().info("======================")
    Mix.shell().info("")

    if dry_run do
      Mix.shell().info("Mode: DRY RUN (use --delete to actually remove directories)")
    else
      Mix.shell().info("Mode: DELETE")
    end

    Mix.shell().info("Max age: #{max_age_hours} hours")
    Mix.shell().info("Excluded: #{inspect(exclude)}")
    Mix.shell().info("")

    base_path =
      Application.get_env(:world, :training_worlds_path, "apps/world/priv/training_worlds")

    unless File.exists?(base_path) do
      Mix.shell().info("No training worlds directory found at: #{base_path}")
      System.halt(0)
    end

    case File.ls(base_path) do
      {:ok, entries} ->
        all_dirs =
          entries
          |> Enum.filter(&File.dir?(Path.join(base_path, &1)))

        Mix.shell().info("Found #{length(all_dirs)} world directories")
        Mix.shell().info("")
        cutoff = DateTime.add(DateTime.utc_now(), -max_age_hours * 3600, :second)

        {orphaned, valid} =
          Enum.reduce(all_dirs, {[], []}, fn world_id, {orphaned_acc, valid_acc} ->
            if world_id in exclude do
              {orphaned_acc, [{world_id, :excluded} | valid_acc]}
            else
              dir_path = Path.join(base_path, world_id)
              config_path = Path.join(dir_path, "config.json")
              has_config = File.exists?(config_path)

              case File.stat(dir_path) do
                {:ok, %{mtime: mtime}} ->
                  case NaiveDateTime.from_erl(mtime) do
                    {:ok, naive} ->
                      dir_time = DateTime.from_naive!(naive, "Etc/UTC")
                      is_old = DateTime.compare(dir_time, cutoff) == :lt
                      age_str = format_age(dir_time)

                      cond do
                        not has_config ->
                          {[{world_id, :no_config, age_str} | orphaned_acc], valid_acc}

                        is_old ->
                          {[{world_id, :old, age_str} | orphaned_acc], valid_acc}

                        true ->
                          {orphaned_acc, [{world_id, :valid, age_str} | valid_acc]}
                      end

                    _ ->
                      {orphaned_acc, [{world_id, :unknown} | valid_acc]}
                  end

                _ ->
                  {orphaned_acc, [{world_id, :unknown} | valid_acc]}
              end
            end
          end)

        Mix.shell().info("Valid worlds (#{length(valid)}):")

        valid
        |> Enum.sort_by(fn
          {id, _, _} -> id
          {id, _} -> id
        end)
        |> Enum.take(10)
        |> Enum.each(fn
          {id, :excluded} ->
            Mix.shell().info("  #{id} (excluded)")

          {id, :valid, age} ->
            Mix.shell().info("  #{id} (#{age})")

          {id, _, age} ->
            Mix.shell().info("  #{id} (#{age})")

          {id, _} ->
            Mix.shell().info("  #{id}")
        end)

        if length(valid) > 10 do
          Mix.shell().info("  ... and #{length(valid) - 10} more")
        end

        Mix.shell().info("")

        if orphaned == [] do
          Mix.shell().info("No orphaned directories found.")
        else
          Mix.shell().info("Orphaned directories (#{length(orphaned)}):")

          orphaned
          |> Enum.sort_by(fn {id, _, _} -> id end)
          |> Enum.each(fn {id, reason, age} ->
            reason_str =
              case reason do
                :no_config -> "no config.json"
                :old -> "older than #{max_age_hours}h"
                _ -> "unknown"
              end

            Mix.shell().info("  #{id} (#{reason_str}, #{age})")
          end)

          Mix.shell().info("")

          if dry_run do
            Mix.shell().info("Would delete #{length(orphaned)} directories")
            Mix.shell().info("Run with --delete to actually remove them")
          else
            Mix.shell().info("Deleting #{length(orphaned)} directories...")

            deleted =
              Enum.reduce(orphaned, 0, fn {world_id, _, _}, count ->
                dir_path = Path.join(base_path, world_id)

                case File.rm_rf(dir_path) do
                  {:ok, _} ->
                    Mix.shell().info("  Deleted: #{world_id}")
                    count + 1

                  {:error, reason, _} ->
                    Mix.shell().error("  Failed to delete #{world_id}: #{inspect(reason)}")
                    count
                end
              end)

            Mix.shell().info("")
            Mix.shell().info("Deleted #{deleted} directories")
          end
        end

      {:error, reason} ->
        Mix.shell().error("Failed to list directory: #{inspect(reason)}")
        System.halt(1)
    end
  end

  defp format_age(datetime) do
    now = DateTime.utc_now()
    diff_seconds = DateTime.diff(now, datetime, :second)

    cond do
      diff_seconds < 60 -> "#{diff_seconds}s ago"
      diff_seconds < 3600 -> "#{div(diff_seconds, 60)}m ago"
      diff_seconds < 86_400 -> "#{div(diff_seconds, 3600)}h ago"
      true -> "#{div(diff_seconds, 86400)}d ago"
    end
  end
end