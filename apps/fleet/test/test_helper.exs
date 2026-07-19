# Fleet tests run against the live stack (Postgres + AGE + Brain). We mirror the
# Brain test harness: start Atlas, bootstrap AGE (incl. command_graph), create
# the atlas_test schema, migrate (incl. atlas_command_records), then start Brain
# for real cognition. There is no DB-avoidance and no graceful degradation — if
# Atlas is unreachable the tests fail.
#
# Unlike apps/brain/test/test_helper.exs, this suite does not bootstrap the
# Ouro sidecar (no ensure_ready! call here) — so the umbrella-wide
# `config/test.exs` setting of `backend: :ouro_sidecar` would leave every
# require_llm cognition call in these tests hitting a real, un-launched
# backend and failing loudly (correctly — but these tests need cognition to
# actually succeed to test what they're testing). Point Fleet's own test run
# at openai_compatible/Ollama instead, which is genuinely available. This is
# process-local (each umbrella app's `mix test` is its own BEAM instance), so
# it does not affect apps/brain's own Ouro-specific test suite. Must happen
# before Brain boots, so its supervision tree starts the right backend's
# children (Brain.ML.Generation.children/0), not Ouro's.
Application.put_env(:brain, :generation,
  backend: :openai_compatible,
  openai_compatible: [base_url: "http://localhost:11434/v1", model: "llama3.1:8b"]
)

{:ok, _} = Application.ensure_all_started(:atlas)

# Umbrella `mix test` runs atlas (and other apps) before fleet. Their Sandboxes
# leave `Atlas.Repo` in :manual with no owner, so unqualified `Repo.query!`
# below would raise DBConnection.OwnershipError. Flip to :auto for this
# bootstrap/migration block; start_owner! below moves it to shared mode for
# Brain/Fleet boot. Mirrors apps/brain/test/test_helper.exs.
Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :auto)

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
