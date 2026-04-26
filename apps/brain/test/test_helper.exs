# Gazetteer must exist on disk before Brain starts — EntityExtractor loads it
# at boot and no longer falls back to raw JSON.
Brain.Test.ModelFactory.ensure_gazetteer_on_disk!()

# Start Atlas only first so we can migrate before Brain GenServers (e.g.
# CredentialVault) query `atlas_test.*` tables.
{:ok, _} = Application.ensure_all_started(:atlas)

# Umbrella `mix test` runs atlas (and other apps) before brain. Their Sandboxes
# leave `Atlas.Repo` in :manual with no owner, so unqualified `Repo.query!`
# here raises DBConnection.OwnershipError. A shared owner covers bootstrap,
# migrations, and Brain boot (CredentialVault, etc.) until we hand off to
# per-test `GraphCase` / `BrainCase` owners.
bootstrap_owner = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: true)

try do
  _ = Mix.Task.run("atlas.bootstrap_age")
  Atlas.Repo.query!(~s(CREATE SCHEMA IF NOT EXISTS atlas_test), [])
  migrations_path = Application.app_dir(:atlas, "priv/repo/migrations")
  Ecto.Migrator.run(Atlas.Repo, migrations_path, :up, all: true, prefix: "atlas_test")

  # Start Brain application to get PubSub and core services
  {:ok, _} = Application.ensure_all_started(:brain)

  # Train and persist all test models, then reload MicroClassifiers from disk.
  Brain.Test.ModelFactory.train_and_load_test_models()

  # Eager gazetteer load (not only the async Application init task) so
  # `Gazetteer.loaded?/0` and feature tests see a ready gazetteer.
  case Brain.ML.Gazetteer.load_all() do
    {:ok, _} ->
      :ok

    other ->
      IO.puts(:stderr, "test_helper: Gazetteer.load_all/0 returned #{inspect(other)}")
      raise "test_helper: Gazetteer failed to load"
  end

  # Validate stored models: every required file must exist and deserialize.
  Brain.ML.ModelPreflight.validate_all!()

  # Ouro: `ouro_auto_start` is false in test — we spawn and wait here only.
  if Application.get_env(:brain, :ouro_enabled, true) do
    Brain.ML.Ouro.SidecarLauncher.ensure_ready!(timeout: 120_000)
  end
after
  Ecto.Adapters.SQL.Sandbox.stop_owner(bootstrap_owner)
end

# Per-test isolation: each case template checks out or shares its own owner.
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

# Start HTTP snapshot server for external API mocking
{:ok, _} = Brain.Test.HTTPSnapshot.start_link()

# Configure ExUnit
# Only exclude explicitly incomplete/disabled tests:
# - :wip - Work in progress tests
# - :skip - Temporarily disabled tests
#
# All other tags (:slow, :training, :benchmark, :gpu, :integration) run by default.
# Any skipped behavior is untested behavior.
#
# (Historical note: a `:requires_lstm` tag used to gate tests against the
# in-tree intent-LSTM stack. Both the modules and the tests are gone, so
# the tag no longer exists anywhere in the suite.)
#
# ============================================================================
# HTTP Snapshot Testing (no external API calls)
# ============================================================================
#
# Tests that previously hit external APIs now use HTTP snapshots.
# Snapshots are stored in test/fixtures/http_snapshots/<service>/<name>.json
#
# To update snapshots from real API responses:
#   MIX_ENV=test mix snapshot.record --force
#
# To record a specific snapshot:
#   MIX_ENV=test mix snapshot.record --name semantic_scholar/search_transformer --force
#
# To list available snapshot definitions:
#   MIX_ENV=test mix snapshot.record --list
#
# Run tests serially by default (`max_cases: 1`). The brain suite shares a
# lot of singleton GenServer state — Ouro.Model, Ouro.SidecarLauncher,
# MicroClassifiers, BeliefStore, etc. — and parallel test cases produced
# overlapping `/health` polls, MPS memory pressure, and Sandbox ownership
# races. Set `EXUNIT_MAX_CASES=N` to opt back into parallelism if you
# know the test you're iterating on is safe.
exunit_max_cases =
  case System.get_env("EXUNIT_MAX_CASES") do
    nil ->
      1

    value ->
      case Integer.parse(value) do
        {n, _} when n > 0 -> n
        _ -> 1
      end
  end

ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: :infinity,
  max_cases: exunit_max_cases
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
