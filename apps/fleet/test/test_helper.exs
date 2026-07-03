# Fleet tests run against the live stack (Postgres + AGE + Brain). We mirror the
# Brain test harness: start Atlas, bootstrap AGE (incl. command_graph), create
# the atlas_test schema, migrate (incl. atlas_command_records), then start Brain
# for real cognition. There is no DB-avoidance and no graceful degradation — if
# Atlas is unreachable the tests fail.
{:ok, _} = Application.ensure_all_started(:atlas)

_ = Mix.Task.run("atlas.bootstrap_age")
Atlas.Repo.query!("CREATE SCHEMA IF NOT EXISTS atlas_test", [])
migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true, prefix: "atlas_test")

# Shared owner covers Brain/Fleet boot until per-test FleetCase owners take over.
bootstrap_owner = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: true)

try do
  {:ok, _} = Application.ensure_all_started(:brain)
  {:ok, _} = Application.ensure_all_started(:fleet)
  Brain.Test.AtlasSandbox.allow_for_test_owner!(bootstrap_owner)
after
  Ecto.Adapters.SQL.Sandbox.stop_owner(bootstrap_owner)
end

if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

ExUnit.start()
