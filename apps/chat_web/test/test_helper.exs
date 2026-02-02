# Ensure Brain app (which starts PubSub and other core services) is loaded
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)
{:ok, _} = Application.ensure_all_started(:chat_web)

# Configure ExUnit to exclude slow/integration tests by default
# Run with: mix test --include slow --include integration
# to run the full suite
ExUnit.configure(
  exclude: [:slow, :integration, :training, :wip, :skip],
  timeout: 60_000
)

ExUnit.start()

# Load support modules
Code.require_file("support/conn_case.ex", __DIR__)
Code.require_file("support/channel_case.ex", __DIR__)
