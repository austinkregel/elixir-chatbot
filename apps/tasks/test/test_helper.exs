# Start Brain application to get core services
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:tasks)

# Configure ExUnit to exclude slow/integration tests by default
ExUnit.configure(
  exclude: [:slow, :integration, :training, :wip, :skip],
  timeout: 60_000
)

ExUnit.start()
