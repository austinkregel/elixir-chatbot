{:ok, _} = Application.ensure_all_started(:atlas)

# Bootstrap Apache AGE in the test database. The Docker initdb script only
# initializes the default POSTGRES_DB, so the *_test database needs explicit
# bootstrap. Idempotent (CREATE EXTENSION IF NOT EXISTS, conditional graphs).
Mix.Task.run("atlas.bootstrap_age")

# Ensure the dedicated test schema exists before any migration runs.
# All Atlas tables (including schema_migrations) live in atlas_test, never
# colliding with public/ag_catalog and never split across schemas.
Atlas.Repo.query!(~s(CREATE SCHEMA IF NOT EXISTS atlas_test), [])

migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true, prefix: "atlas_test")

Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

ExUnit.start()
