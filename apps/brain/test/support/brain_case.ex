defmodule Brain.Test.BrainCase do
  @moduledoc """
  ExUnit case for tests that exercise the full Brain stack (pipeline, memory,
  epistemic) without needing graph seed helpers.

  Provides the same `Atlas.Repo` sandbox checkout, AGE `search_path`, and
  `Sandbox.allow` coverage as `Brain.Test.GraphCase`, minus graph imports/seeds.

      use Brain.Test.BrainCase, async: false
  """

  use ExUnit.CaseTemplate

  setup tags do
    pid = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)

    Brain.TestHelpers.start_test_services()

    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(pid) end)

    {:ok, sandbox_pid: pid}
  end
end
