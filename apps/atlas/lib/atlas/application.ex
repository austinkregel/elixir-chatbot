defmodule Atlas.Application do
  @moduledoc false

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      Atlas.Repo,
      Atlas.Stats.Collector
    ]

    opts = [strategy: :one_for_one, name: Atlas.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        maybe_run_migrations()
        maybe_auto_import()
        attach_telemetry()
        {:ok, pid}

      error ->
        error
    end
  end

  defp maybe_run_migrations do
    if Application.get_env(:atlas, :auto_migrate, false) do
      Logger.info("Atlas: Running auto-migrations...")

      try do
        migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
        Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true)
        Logger.info("Atlas: Migrations complete")
      rescue
        e ->
          Logger.warning("Atlas: Migration failed - #{Exception.message(e)}")
      catch
        :exit, reason ->
          Logger.warning("Atlas: Migration exited - #{inspect(reason)}")
      end
    end
  end

  @auto_import_stores [
    {"learned_facts", Atlas.Schemas.LearnedFact},
    {"beliefs", Atlas.Schemas.Belief},
    {"episodes", Atlas.Schemas.Episode},
    {"semantic_facts", Atlas.Schemas.SemanticFact},
    {"source_authority", Atlas.Schemas.SourceAuthority}
  ]

  defp maybe_auto_import do
    if Application.get_env(:atlas, :auto_import, false) do
      empty_stores =
        @auto_import_stores
        |> Enum.filter(fn {_name, schema} ->
          Atlas.Repo.aggregate(schema, :count) == 0
        end)
        |> Enum.map(fn {name, _schema} -> name end)

      if empty_stores != [] do
        Logger.info("Atlas: Auto-importing stores with empty tables: #{Enum.join(empty_stores, ", ")}")
        Atlas.Importer.import_all(quiet: true, only: empty_stores)
      end
    end
  rescue
    e ->
      Logger.warning("Atlas: Auto-import failed - #{Exception.message(e)}")
  end

  defp attach_telemetry do
    Atlas.Telemetry.attach_handlers()
  rescue
    _ -> :ok
  end
end
