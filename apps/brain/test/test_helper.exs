# Start Atlas only first so the Sandbox is in :manual before any Brain
# GenServer (e.g. CredentialVault) queries `atlas_test.*` tables.
{:ok, _} = Application.ensure_all_started(:atlas)

# :manual for the whole run, set before anything else can write. Every test
# then works inside a transaction that is rolled back, and a write from a
# process holding no connection raises DBConnection.OwnershipError instead of
# persisting. Nothing here switches to :auto: the schema, migrations and
# seeded lexicon come from `MIX_ENV=test mix test.prepare`, which the root
# `mix test` alias runs first.
Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

# A shared owner covers application boot until the per-test `GraphCase` /
# `BrainCase` owners take over; whatever boot writes is rolled back with it.
bootstrap_owner = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: true)

try do
  # Fails loudly when the database was not prepared, rather than letting every
  # lexicon-dependent test fail for lack of data.
  Brain.Test.Database.verify_prepared!()

  # Start Brain application to get PubSub and core services
  {:ok, _} = Application.ensure_all_started(:brain)

  Brain.Test.AtlasSandbox.allow_for_test_owner!(bootstrap_owner)

  # The gazetteer loads its sources in init/1, so it must already be loaded
  # the moment the application has started. Checked here, before any test can
  # reload or modify the shared table and mask a regression.
  unless Brain.ML.Gazetteer.loaded?() do
    raise "test_helper: the gazetteer did not load during application startup"
  end

  # Train and persist all test models, then reload MicroClassifiers from disk.
  Brain.Test.ModelFactory.train_and_load_test_models()

  # Validate stored models: every required file must exist and deserialize.
  Brain.ML.ModelPreflight.validate_all!()

  # Ouro: `ouro_auto_start` is false in test — we spawn and wait here only.
  if Application.get_env(:brain, :ouro_enabled, true) do
    Brain.ML.Ouro.SidecarLauncher.ensure_ready!(timeout: 120_000)
  end

  # The lexicon the brain reads all run: seeded by `mix test.prepare`, loaded
  # into the store at boot above (`Lexicon.UserDefined: loaded N facts from
  # Atlas`). Without those facts every word would read as non-negating and
  # anything asserting on morphological negation would fail for lack of data
  # rather than for a real reason.
  #
  # Counted in the database, not through `UserDefined.count/1`: that counts
  # words carrying a *sense*, which the seeded property facts are not, so it
  # reads 0 on a correctly loaded store. Inside the block: the count needs the
  # bootstrap owner's connection.
  IO.puts("test_helper: lexicon ready (#{Brain.Test.Database.count_lexicon_facts()} facts seeded)")
after
  Ecto.Adapters.SQL.Sandbox.stop_owner(bootstrap_owner)
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

# When the Ouro sidecar is disabled (OURO_ENABLED=false), also exclude the
# tests that assert on real Ouro generation.
exunit_exclude =
  if Application.get_env(:brain, :ouro_enabled, true) do
    [:wip, :skip]
  else
    [:wip, :skip, :requires_ouro]
  end

ExUnit.configure(
  exclude: exunit_exclude,
  timeout: :infinity,
  max_cases: exunit_max_cases
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
