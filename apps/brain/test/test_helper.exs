# Start Brain application to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)

# Set Atlas.Repo to sandbox mode for test isolation
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

# Validate stored models match current code expectations.
# Halts the suite immediately with an actionable message if any model
# has a dimension/config mismatch (e.g. needs retraining).
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
# Optional exclusions (add via command line if needed):
# - :requires_lstm - Tests requiring compatible LSTM .term files
#   mix test --exclude requires_lstm
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
