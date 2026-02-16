# FourthWall test helper
# This app is standalone - no dependencies on Brain or other apps

# Configure ExUnit - only exclude explicitly incomplete/disabled tests
# Any skipped behavior is untested behavior.
ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: 60_000
)

ExUnit.start()
