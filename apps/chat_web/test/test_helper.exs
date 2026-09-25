# Start Atlas first so the Sandbox is in :manual before anything queries it.
# :manual for the whole run: every test works inside a transaction that is
# rolled back, and a write from a process holding no connection raises instead
# of persisting. The lexicon the pages read is seeded by
# `MIX_ENV=test mix test.prepare`, which the root `mix test` alias runs first.
{:ok, _} = Application.ensure_all_started(:atlas)
Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

# A shared owner covers application boot until each test's ConnCase owner
# takes over; whatever boot writes is rolled back with it.
bootstrap_owner = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: true)

try do
  {:ok, _} = Application.ensure_all_started(:brain)
  {:ok, _} = Application.ensure_all_started(:world)
  {:ok, _} = Application.ensure_all_started(:chat_web)

  Brain.Test.AtlasSandbox.allow_for_test_owner!(bootstrap_owner)
  Brain.Test.Database.verify_prepared!()

  # Train the test models here, before the clock starts on any test. Several
  # LiveView tests reach them through `Brain.TestHelpers.start_test_services/0`,
  # which trains on first use: run on their own, that lands inside a test's
  # setup and blows the 60s timeout. It only appeared to work when brain's
  # suite ran first in the same VM and left its "already trained" flag behind.
  Brain.Test.ModelFactory.train_and_load_test_models()
after
  Ecto.Adapters.SQL.Sandbox.stop_owner(bootstrap_owner)
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
