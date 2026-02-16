# Start Brain application to get core services
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:tasks)

# Configure ExUnit - only exclude explicitly incomplete/disabled tests
# Any skipped behavior is untested behavior.
ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: 60_000
)

ExUnit.start()
