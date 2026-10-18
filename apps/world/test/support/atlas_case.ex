defmodule World.AtlasCase do
  @moduledoc """
  The case for a world test that reaches Atlas: creating, loading or purging a
  world creates or drops its graphs, and its memory, beliefs and scheduler
  items are rows in Atlas.

  World tests run the Atlas sandbox in `:manual` mode (`test/test_helper.exs`),
  so a query from a process holding no connection raises. Each test checks
  out its own sandbox connection, in shared mode, and allows the long-lived
  processes that query Atlas on its behalf -- brain's
  (`Brain.Test.AtlasSandbox.atlas_genservers/0`) and `World.Manager` -- on it.
  Everything the test writes is rolled back when it ends.

  The checkout is the first setup the test runs, so it is stopped last: a
  world a test creates (`World.TestWorlds.create_test_world/2`) is removed
  while the connection is still there.

      use World.AtlasCase
  """

  use ExUnit.CaseTemplate

  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags, [World.Manager])
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    :ok
  end
end
