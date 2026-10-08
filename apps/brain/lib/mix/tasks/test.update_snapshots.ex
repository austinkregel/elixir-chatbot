defmodule Mix.Tasks.Test.UpdateSnapshots do
  @shortdoc "Re-record the analysis snapshots under apps/brain/test/snapshots"

  @moduledoc """
  Runs the `:snapshot` tagged tests with `UPDATE_SNAPSHOTS=true`, so
  `Brain.SnapshotHelper.assert_snapshot/3` writes each snapshot instead of
  comparing it.

      MIX_ENV=test mix test.update_snapshots

  Any further arguments are passed through to `mix test`, so a single
  snapshot can be re-recorded by narrowing the run:

      MIX_ENV=test mix test.update_snapshots apps/brain/test/brain/analysis/edge_cases_comprehensive_test.exs

  ## What a snapshot is, and is not

  A snapshot records that the analysis has not *changed*. It says nothing
  about whether the analysis is *correct*. That distinction is why
  `assert_snapshot/3` is called last in each test, after the assertions that
  check the behaviour: a snapshot is only ever written for a state that
  already satisfied them, so re-recording cannot quietly bless output a test
  says is wrong.

  This task is referenced by `Brain.SnapshotHelper`'s own documentation and by
  the failure message `assert_snapshot/3` raises when a snapshot is missing.
  It did not exist until 2026-09-25 -- the helper had zero callers, the 13
  stored snapshots were read by nothing, and their contents had drifted from
  the live analysis in every field.
  """

  use Mix.Task

  @preferred_cli_env :test

  @impl Mix.Task
  def run(args) do
    unless Mix.env() == :test do
      Mix.raise("test.update_snapshots re-records test fixtures; run it with MIX_ENV=test")
    end

    System.put_env("UPDATE_SNAPSHOTS", "true")

    args =
      if Enum.any?(args, &(&1 == "--only")) do
        args
      else
        args ++ ["--only", "snapshot"]
      end

    Mix.shell().info("Re-recording snapshots (UPDATE_SNAPSHOTS=true, mix test #{Enum.join(args, " ")})")

    Mix.Task.run("test", args)
  end
end
