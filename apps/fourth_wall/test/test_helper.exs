# FourthWall test helper
# This app is standalone - no dependencies on Brain or other apps

# Serial by default (`max_cases: 1`); override with `EXUNIT_MAX_CASES=N`.
# Kept in sync with the rest of the umbrella so a developer running
# `mix test` from one app sees the same parallelism behavior as another.
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

# Ensure custom mix task modules are available in umbrella runs where
# incremental compilation may skip loading `lib/mix/tasks/**` modules.
mix_task_modules = [
  {Mix.Tasks.CredoFix, "../lib/mix/tasks/credo_fix.ex"},
  {Mix.Tasks.CredoFix.AliasUsage, "../lib/mix/tasks/credo_fix/alias_usage.ex"},
  {Mix.Tasks.CredoFix.LargeNumbers, "../lib/mix/tasks/credo_fix/large_numbers.ex"},
  {Mix.Tasks.CredoFix.LengthCheck, "../lib/mix/tasks/credo_fix/length_check.ex"},
  {Mix.Tasks.CredoFix.MapJoin, "../lib/mix/tasks/credo_fix/map_join.ex"},
  {Mix.Tasks.CredoFix.TrailingWhitespace, "../lib/mix/tasks/credo_fix/trailing_whitespace.ex"},
  {Mix.Tasks.CredoFix.UnusedAlias, "../lib/mix/tasks/credo_fix/unused_alias.ex"}
]

Enum.each(mix_task_modules, fn {module, relative_path} ->
  unless Code.ensure_loaded?(module) do
    path = Path.expand(relative_path, __DIR__)
    Code.require_file(path)
  end
end)

ExUnit.start()
