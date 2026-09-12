defmodule Mix.Tasks.Atlas.MigrateTest do
  use Mix.Task

  @shortdoc "Create the atlas_test schema and run migrations with that prefix"
  @moduledoc """
  Creates the `atlas_test` schema and runs all Atlas migrations with the
  `atlas_test` prefix, without starting the umbrella applications.

  The brain test_helper does the same thing at startup, but `mix test`
  boots the apps first — on a fresh database (CI) every boot-time query
  then fails or retries against missing tables and can starve the
  connection pool. Run this (after `mix atlas.bootstrap_age`) before
  `mix test` so the app boots against a fully-migrated schema.

  ## Usage

      MIX_ENV=test mix atlas.migrate_test
  """

  @impl Mix.Task
  def run(_args) do
    Mix.Task.run("app.config")
    {:ok, _} = Application.ensure_all_started(:ecto_sql)
    {:ok, _} = Application.ensure_all_started(:postgrex)

    # Plain pool: the Sandbox pool from test config would refuse checkouts
    # without an owner.
    {:ok, pid} = Atlas.Repo.start_link(pool: DBConnection.ConnectionPool, pool_size: 2)

    try do
      Atlas.Repo.query!("CREATE SCHEMA IF NOT EXISTS atlas_test", [])
      migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
      Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true, prefix: "atlas_test")
      Mix.shell().info("Atlas: atlas_test schema migrated")
    after
      GenServer.stop(pid)
    end
  end
end
