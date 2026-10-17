defmodule Mix.Tasks.World.AssignLocalSchedulingTest do
  @moduledoc """
  `mix world.assign_local_scheduling` stores the local scheduler as the
  scheduling driver of each persisted world that has none, and leaves a world
  that has one with the driver it has.

  The driver rows are written inside the test's sandbox transaction; the task
  also reaches every other world persisted in the test world directory, and
  its rows for them are rolled back with the rest.
  """
  use World.AtlasCase, async: false
  import Brain.TestHelpers
  import Ecto.Query, only: [from: 2]

  alias Brain.Services.{Dispatcher, LocalScheduler}
  alias Mix.Tasks.World.AssignLocalScheduling
  alias World.Manager, as: WorldManager

  setup do
    start_world_test_services()
    setup_world_sandbox()

    eventually(
      fn -> WorldManager.ready?() end,
      fn ready -> ready end,
      50,
      20
    )

    previous_shell = Mix.shell()
    Mix.shell(Mix.Shell.Process)
    on_exit(fn -> Mix.shell(previous_shell) end)

    :ok
  end

  test "raises on any argument" do
    assert_raise Mix.Error, ~r/takes no arguments/, fn -> AssignLocalScheduling.run(["--all"]) end
  end

  test "assigns the local scheduler to a persisted world with no driver, and keeps another world's driver" do
    {:ok, without} = create_test_world("assign_scheduling_none", mode: :persistent)
    {:ok, chosen} = create_test_world("assign_scheduling_chosen", mode: :persistent)

    # A world persisted before drivers were stored has no row; a world created
    # since has the driver it was created with.
    without_id = without.id
    Atlas.Repo.delete_all(from(d in Atlas.Schemas.WorldServiceDriver, where: d.world_id == ^without_id))

    assert Dispatcher.scheduling_driver(without.id) == :none
    assert Dispatcher.scheduling_driver(chosen.id) == {:ok, LocalScheduler}

    AssignLocalScheduling.run([])

    assert Dispatcher.scheduling_driver(without.id) == {:ok, LocalScheduler}
    assert Dispatcher.scheduling_driver(chosen.id) == {:ok, LocalScheduler}

    assigned = "  #{without.id}: assigned Brain.Services.LocalScheduler"
    kept = "  #{chosen.id}: keeps Brain.Services.LocalScheduler"
    assert_received {:mix_shell, :info, [^assigned]}
    assert_received {:mix_shell, :info, [^kept]}
  end

  test "raises on a world whose stored driver name is no scheduling driver's" do
    {:ok, stale} = create_test_world("assign_scheduling_stale", mode: :persistent)
    Atlas.WorldServiceDrivers.put!(stale.id, "scheduling", "home_assistant")

    assert_raise ArgumentError, ~r/no scheduling driver is named "home_assistant"/, fn ->
      AssignLocalScheduling.run([])
    end
  end
end
