# Ensure Brain app (which starts PubSub and other core services) is loaded
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)
{:ok, _} = Application.ensure_all_started(:chat_web)

# Seed the brain's own lexicon. The lexicon page shows the facts the brain
# owns, so without this it would have nothing to show and its tests would fail
# for lack of data rather than for a real reason. Seeding runs with the pool in
# :auto, before per-test sandboxing, so the facts persist for the whole run.
# It is idempotent.
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :auto)
  {:ok, _} = Brain.Lexicon.Seeder.seed_all()
  {:ok, _} = Brain.Lexicon.UserDefined.reload()
end

# Set Atlas.Repo to sandbox mode for test isolation
if Process.whereis(Atlas.Repo) do
  Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)
end

# Configure ExUnit - only exclude explicitly incomplete/disabled tests.
# Any skipped behavior is untested behavior.
#
# Serial by default (`max_cases: 1`) so chat_web LiveView tests inherit
# the same isolation guarantees the brain suite needs (shared GenServer
# state, Sandbox ownership). Override with `EXUNIT_MAX_CASES=N`.
exunit_max_cases =
  case System.get_env("EXUNIT_MAX_CASES") do
    nil ->
      1

    value ->
      case Integer.parse(value) do
        {n, _} when n > 0 -> n
        _ -> 1
      end
  end

ExUnit.configure(
  exclude: [:wip, :skip],
  timeout: 60_000,
  max_cases: exunit_max_cases
)

ExUnit.start()
