# Start Brain and World applications to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)

# Configure ExUnit - only exclude explicitly incomplete/disabled tests
# Any skipped behavior is untested behavior.
ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: 60_000
)

ExUnit.start()

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
