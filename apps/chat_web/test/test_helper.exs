# Ensure Brain app (which starts PubSub and other core services) is loaded
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)
{:ok, _} = Application.ensure_all_started(:chat_web)

# Set Atlas.Repo to sandbox mode for test isolation
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

# Configure ExUnit - only exclude explicitly incomplete/disabled tests
# Any skipped behavior is untested behavior.
ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: 60_000
)

ExUnit.start()
