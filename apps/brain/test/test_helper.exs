# Start Brain application to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)

# Configure ExUnit
# By default, only exclude tests that are explicitly marked as:
# - :slow - Very slow tests (>30s)
# - :integration - External service integration tests
# - :training - Tests that train models (expensive)
# - :wip - Work in progress tests
# - :skip - Temporarily disabled tests
#
# Run with: mix test --include slow --include integration
# to run the full suite including slow tests
ExUnit.configure(
  exclude: [:slow, :integration, :training, :wip, :skip],
  timeout: 60_000
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
