# FourthWall test helper
# This app is standalone - no dependencies on Brain or other apps

# Configure ExUnit
ExUnit.configure(
  exclude: [:slow, :integration, :wip, :skip],
  timeout: 60_000
)

ExUnit.start()
