# Gazetteer must exist on disk before Brain starts — EntityExtractor loads it
# at boot and no longer falls back to raw JSON.
Brain.Test.ModelFactory.ensure_gazetteer_on_disk!()

# Start Brain application to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)

# Set Atlas.Repo to sandbox mode for test isolation
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

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

# Wait for the Ouro Python sidecar to become healthy.
# The SidecarLauncher auto-starts the server in test config.
if Application.get_env(:brain, :ml, [])[:ouro_auto_start] do
  Brain.ML.Ouro.SidecarLauncher.ensure_ready!(timeout: 120_000)
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
ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: :infinity
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
