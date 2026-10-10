defmodule Mix.Tasks.World.AssignLocalScheduling do
  @shortdoc "Store the local scheduler as the scheduling driver of every persisted world that has none"

  @moduledoc """
  Stores `Brain.Services.LocalScheduler` as the scheduling driver of every
  world persisted under `World.Persistence.base_path/0`
  (`World.Persistence.list_persisted_worlds/0`) that has no scheduling driver.

      mix world.assign_local_scheduling

  A world's alarm, timer, reminder and calendar intents go to the scheduling
  driver stored for it, and a world with none raises when one arrives. A world
  created by `World.Manager` gets the local scheduler when it is created; this
  assigns it to worlds that already existed before drivers were stored. A
  world that already has a driver keeps it and is reported with it.
  """

  use Mix.Task

  alias Brain.Services.{Dispatcher, LocalScheduler}
  alias World.Persistence

  @requirements ["app.config"]

  @impl Mix.Task
  def run(args) do
    if args != [], do: Mix.raise("world.assign_local_scheduling takes no arguments, got: #{inspect(args)}")

    {:ok, _} = Application.ensure_all_started(:atlas)

    worlds = Enum.map(Persistence.list_persisted_worlds(), & &1.id)
    if worlds == [], do: Mix.raise("world.assign_local_scheduling: no worlds under #{Persistence.base_path()}")

    Enum.each(worlds, fn world_id ->
      case Dispatcher.scheduling_driver(world_id) do
        {:ok, driver} ->
          Mix.shell().info("  #{world_id}: keeps #{inspect(driver)}")

        :none ->
          :ok = Dispatcher.assign_scheduling_driver(world_id, LocalScheduler)
          Mix.shell().info("  #{world_id}: assigned #{inspect(LocalScheduler)}")
      end
    end)
  end
end
