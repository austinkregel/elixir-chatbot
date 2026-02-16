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
      end
    end
  end

  defp maybe_auto_import do
    if Application.get_env(:atlas, :auto_import, false) do
      if Atlas.Repo.aggregate(Atlas.Schemas.Belief, :count) == 0 do
        Atlas.Importer.import_all(quiet: true)
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
