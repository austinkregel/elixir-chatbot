# Start Brain and World applications to get PubSub and core services
{:ok, _} = Application.ensure_all_started(:brain)
{:ok, _} = Application.ensure_all_started(:world)

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
