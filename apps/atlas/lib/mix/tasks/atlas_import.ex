defmodule Mix.Tasks.Atlas.Import do
  @moduledoc """
  Import file-based data into PostgreSQL via Atlas.

  ## Usage

      mix atlas.import                    # Import all stores
      mix atlas.import --only beliefs,facts   # Import specific stores
      mix atlas.import --dry-run         # Show what would be imported without writing
      mix atlas.import --force           # Re-import even if tables have data

  ## Stores

  - beliefs (belief_store.term)
  - credentials (credentials.enc)
  - episodes (memory_store.term)
  - semantic_facts (memory_store.term)
  - review_candidates (review_queue.term)
  - learned_facts (data/facts/*.json)
  - source_reliability (source_reliability_learned.term)
  - source_authority (source_authority_learned.term)
  - user_models (user_models.term)
  - knowledge (knowledge/*.json)
  - persona_memories (memory/*.json)
  """

  use Mix.Task
  require Logger

  @shortdoc "Import file-based data into PostgreSQL (Atlas)"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        strict: [only: :string, dry_run: :boolean, force: :boolean],
        aliases: [o: :only]
      )

    Mix.Task.run("app.start")

    # Ensure migrations are run
    migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
    case Ecto.Migrator.with_repo(Atlas.Repo, fn _ ->
           Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true)
         end) do
      {:ok, _, _} -> :ok
      _ -> :ok
    end

    import_opts = []
    import_opts = if Keyword.get(opts, :dry_run), do: Keyword.put(import_opts, :dry_run, true), else: import_opts
    import_opts = if Keyword.get(opts, :force), do: Keyword.put(import_opts, :force, true), else: import_opts
    import_opts = if only = Keyword.get(opts, :only), do: Keyword.put(import_opts, :only, only), else: import_opts

    {:ok, summary} = Atlas.Importer.import_all(import_opts)

    Mix.shell().info("")
    Mix.shell().info("Import complete:")

    Enum.each(summary, fn {store, result} ->
      case result do
        {:ok, count} -> Mix.shell().info("  #{store}: #{count} rows")
        {:skip, reason} -> Mix.shell().info("  #{store}: skipped (#{reason})")
        {:error, reason} -> Mix.shell().error("  #{store}: error - #{inspect(reason)}")
      end
    end)

    Mix.shell().info("")
    :ok
  end
end
