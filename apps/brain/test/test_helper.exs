# Start Brain application to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)

# Start HTTP snapshot server for external API mocking
{:ok, _} = Brain.Test.HTTPSnapshot.start_link()

# Configure ExUnit
# By default, only exclude tests that are explicitly marked as:
# - :slow - Very slow tests (>30s)
# - :training - Tests that train models (expensive)
# - :wip - Work in progress tests
# - :skip - Temporarily disabled tests
# - :benchmark - Performance benchmark tests
# - :gpu - GPU-specific tests
#
# Optional exclusions (add via command line):
# - :requires_lstm - Tests requiring compatible LSTM .term files
#
# Run with: mix test --include slow to run the full suite including slow tests
#
# To skip LSTM-dependent tests when models are incompatible:
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
  exclude: [:slow, :training, :wip, :skip, :benchmark, :gpu],
  timeout: 60_000
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
