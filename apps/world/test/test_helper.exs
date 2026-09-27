# Start Atlas first so the Sandbox is in :manual before anything queries it.
# :manual for the whole run: every test works inside a transaction that is
# rolled back, and a write from a process holding no connection raises instead
# of persisting. World tests previously ran with no owner at all, so their
# episodes and graph writes were committed to the shared test database.
{:ok, _} = Application.ensure_all_started(:atlas)
Ecto.Adapters.SQL.Sandbox.mode(Atlas.Repo, :manual)

# A shared owner covers application boot until each test's own owner takes
# over; whatever boot writes is rolled back with it.
bootstrap_owner = Ecto.Adapters.SQL.Sandbox.start_owner!(Atlas.Repo, shared: true)

try do
  {:ok, _} = Application.ensure_all_started(:brain)
  {:ok, _} = Application.ensure_all_started(:world)

  Brain.Test.AtlasSandbox.allow_for_test_owner!(bootstrap_owner)
  Brain.Test.Database.verify_prepared!()

  # Train the test models here, before the clock starts on any test. World
  # tests reach them through `Brain.TestHelpers.start_test_services/0`, which
  # trains on first use: run on their own, that lands inside a test's setup
  # and blows the 60s timeout. It only appeared to work when brain's suite ran
  # first in the same VM and left its "already trained" flag behind.
  Brain.Test.ModelFactory.train_and_load_test_models()
after
  Ecto.Adapters.SQL.Sandbox.stop_owner(bootstrap_owner)
end

# Serial by default (`max_cases: 1`); override with `EXUNIT_MAX_CASES=N`.
# See `apps/brain/test/test_helper.exs` for the rationale.
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

# Note: Support modules in test/support/ are automatically compiled
# due to elixirc_paths(:test) in mix.exs - no need to require them here
