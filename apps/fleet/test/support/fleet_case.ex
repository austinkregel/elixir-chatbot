defmodule Fleet.FleetCase do
  @moduledoc """
  ExUnit case template for Fleet tests that run against the live stack
  (Postgres + Apache AGE + Brain). Atlas is required infrastructure — there is
  no graceful degradation; a missing database is a configuration error and the
  test fails.

  Uses a shared sandbox owner (async: false) so dynamically-spawned ensigns and
  their cognition Tasks all share the test connection.
  """
  use ExUnit.CaseTemplate

  using do
    quote do
      alias Fleet.{Order, Signal}
      alias Atlas.Graph
    end
  end

  setup tags do
    pid = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(pid) end)
    {:ok, sandbox_pid: pid}
  end
end
