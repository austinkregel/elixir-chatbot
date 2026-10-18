defmodule World.TestWorlds do
  @moduledoc """
  Test worlds created through `World.Manager`, for tests of what manages
  worlds: the world is held by the manager, which creates its tree in brain
  (`Brain.Worlds`), and is removed when the test completes.

  A brain test creates its world in brain alone
  (`Brain.TestHelpers.create_test_world/2`); a test that imports both imports
  `Brain.TestHelpers` without `create_test_world/1,2` and
  `start_world_test_services/0`.

      import Brain.TestHelpers, except: [create_test_world: 1, create_test_world: 2, start_world_test_services: 0]
      import World.TestWorlds
  """

  import ExUnit.Callbacks

  alias Brain.TestWorldSandbox
  alias World.Manager

  @doc "Starts the brain test services and `World.Manager`."
  def start_world_test_services do
    Brain.TestHelpers.start_test_services()
    Brain.TestHelpers.ensure_started(Manager)
    :ok
  end

  @doc """
  Creates a test world through `World.Manager.create/2`, with the options
  `Brain.TestWorldSandbox.test_world_options/2` gives it (a unique `test_` id,
  `:ephemeral` unless `:mode` says otherwise, `:test` metadata); every other
  option goes to the manager. An `on_exit` callback destroys it and deletes its
  persisted data when the test completes.
  """
  def create_test_world(name, opts \\ []) do
    opts = TestWorldSandbox.test_world_options(name, opts)

    case Manager.create(name, opts) do
      {:ok, world} ->
        on_exit({__MODULE__, world.id}, fn -> remove_test_world!(world.id) end)
        {:ok, world}

      error ->
        error
    end
  end

  # Destroys the world's runtime state -- a test may already have done that
  # itself -- and deletes whatever it persisted. Refuses any id that is not a
  # test world's.
  defp remove_test_world!(world_id) do
    :ok = TestWorldSandbox.refuse_unless_test_world!(world_id)

    case Manager.destroy(world_id) do
      :ok -> :ok
      {:error, :not_found} -> :ok
    end

    TestWorldSandbox.delete_test_world_data!(world_id)
  end
end
