defmodule Mix.Tasks.Atlas.Seed do
  @moduledoc """
  Idempotent seed task for baseline data required by the running system.

  Seeds curated facts into Atlas, syncs them to the epistemic system as
  beliefs with JTMS premises, ensures source authority profiles exist,
  and creates default training worlds.

  Safe to run on every deploy — all operations use conflict-ignore or
  existence checks.

  ## Usage

      mix atlas.seed              # Seed all baseline data
      mix atlas.seed --verbose    # Show detailed progress
      mix atlas.seed --dry-run    # Show what would be seeded without writing
  """

  use Mix.Task
  require Logger

  @compile {:no_warn_undefined, [
    World.Manager,
    Brain.FactDatabase,
    Brain.FactDatabase.Integration,
    Brain.Epistemic.SourceAuthority
  ]}

  @shortdoc "Seed baseline facts, beliefs, authority profiles, and worlds"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, strict: [verbose: :boolean, dry_run: :boolean])

    verbose? = Keyword.get(opts, :verbose, false)
    dry_run? = Keyword.get(opts, :dry_run, false)

    Mix.Task.run("app.start")

    if dry_run?, do: Mix.shell().info("(Dry run — no data will be written)\n")

    Mix.shell().info("=== Atlas Seed ===\n")

    results = [
      {"Curated facts", seed_curated_facts(verbose?, dry_run?)},
      {"Facts → Beliefs sync", seed_beliefs(verbose?, dry_run?)},
      {"Source authority", seed_source_authority(verbose?, dry_run?)},
      {"Training worlds", seed_training_worlds(verbose?, dry_run?)}
    ]

    Mix.shell().info("")

    Enum.each(results, fn {label, result} ->
      case result do
        {:ok, msg} -> Mix.shell().info("  #{label}: #{msg}")
        {:skip, msg} -> Mix.shell().info("  #{label}: skipped (#{msg})")
        {:error, msg} -> Mix.shell().error("  #{label}: ERROR — #{msg}")
      end
    end)

    Mix.shell().info("\nSeed complete.")
  end

  # ---------------------------------------------------------------------------
  # Curated facts: read data/facts/*.json and upsert into atlas_learned_facts
  # ---------------------------------------------------------------------------
  defp seed_curated_facts(verbose?, dry_run?) do
    facts_dir =
      Application.get_env(:brain, :facts_dir) ||
        Path.join(File.cwd!(), "data/facts")

    glob = Path.join(facts_dir, "*.json")
    files = Path.wildcard(glob)

    if files == [] do
      {:skip, "no files at #{glob}"}
    else
      all_facts =
        Enum.flat_map(files, fn path ->
          with {:ok, content} <- File.read(path),
               {:ok, data} <- Jason.decode(content) do
            facts = Map.get(data, "facts", [])
            if verbose?, do: Mix.shell().info("  Read #{length(facts)} facts from #{Path.basename(path)}")
            facts
          else
            _ -> []
          end
        end)

      if dry_run? do
        {:ok, "would upsert #{length(all_facts)} facts"}
      else
        inserted =
          Enum.count(all_facts, fn f ->
            attrs = %{
              id: f["id"] || "fact_#{System.unique_integer([:positive])}",
              entity: f["entity"] || "unknown",
              entity_type: f["entity_type"],
              fact: f["fact"] || "",
              category: f["category"] || "learned",
              confidence: f["confidence"] || 1.0,
              verification_source: f["verification_source"]
            }

            changeset =
              Atlas.Schemas.LearnedFact.changeset(
                struct(Atlas.Schemas.LearnedFact),
                attrs
              )

            case Atlas.Repo.insert(changeset, on_conflict: :nothing) do
              {:ok, _} -> true
              _ -> false
            end
          end)

        # Reload FactDatabase so it picks up newly inserted facts
        if inserted > 0 do
          try do
            Brain.FactDatabase.reload()
          catch
            :exit, _ -> :ok
          end
        end

        {:ok, "#{inserted} new facts inserted (#{length(all_facts)} total in files)"}
      end
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ---------------------------------------------------------------------------
  # Sync facts to beliefs + JTMS premises via the existing integration module
  # ---------------------------------------------------------------------------
  defp seed_beliefs(_verbose?, dry_run?) do
    if dry_run? do
      {:ok, "would sync facts to beliefs"}
    else
      case Brain.FactDatabase.Integration.sync_facts_to_beliefs() do
        {:ok, count} -> {:ok, "#{count} facts synced to beliefs"}
        {:error, reason} -> {:error, inspect(reason)}
      end
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ---------------------------------------------------------------------------
  # Ensure bootstrap source authority profiles are in Atlas
  # ---------------------------------------------------------------------------
  defp seed_source_authority(verbose?, dry_run?) do
    if dry_run? do
      {:ok, "would ensure source authority profiles"}
    else
      try do
        if Brain.Epistemic.SourceAuthority.ready?() do
          profiles = Brain.Epistemic.SourceAuthority.list_profiles()
          count = length(profiles)

          if verbose? do
            Mix.shell().info("  SourceAuthority has #{count} profiles loaded")
          end

          {:ok, "#{count} authority profiles active"}
        else
          {:skip, "SourceAuthority not ready"}
        end
      catch
        :exit, _ -> {:skip, "SourceAuthority not available"}
      end
    end
  rescue
    e -> {:error, Exception.message(e)}
  end

  # ---------------------------------------------------------------------------
  # Create default training worlds (idempotent — skips if already exists)
  # ---------------------------------------------------------------------------
  defp seed_training_worlds(verbose?, dry_run?) do
    worlds = ["default", "personal", "work"]

    if dry_run? do
      {:ok, "would ensure worlds: #{Enum.join(worlds, ", ")}"}
    else
      results =
        Enum.map(worlds, fn name ->
          case World.Manager.get(name) do
            {:ok, _} ->
              if verbose?, do: Mix.shell().info("  World '#{name}' already exists")
              {:exists, name}

            {:error, _} ->
              case World.Manager.create(name, mode: :persistent) do
                {:ok, _} ->
                  if verbose?, do: Mix.shell().info("  Created world '#{name}'")
                  {:created, name}

                {:error, reason} ->
                  if verbose?, do: Mix.shell().info("  World '#{name}' create failed: #{inspect(reason)}")
                  {:error, name}
              end
          end
        end)

      created = Enum.count(results, fn {status, _} -> status == :created end)
      existed = Enum.count(results, fn {status, _} -> status == :exists end)

      {:ok, "#{created} created, #{existed} already existed"}
    end
  rescue
    e -> {:error, Exception.message(e)}
  end
end
