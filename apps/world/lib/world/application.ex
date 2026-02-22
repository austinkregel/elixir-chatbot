defmodule World.Application do
  @moduledoc false
  alias World.Manager
  use Application

  @impl true
  def start(_type, _args) do
    children = [World.Manager, World.ModelRegistry, World.EntityPromoter]

    opts = [strategy: :one_for_one, name: World.Supervisor]
    result = Supervisor.start_link(children, opts)
    init_default_world()

    result
  end

  defp init_default_world do
    require Logger

    Task.start(fn ->
      Process.sleep(100)

      unless Process.whereis(World.Manager) do
        Logger.debug("Waiting for World.Manager to start...")
        Process.sleep(100)
      end

      unless Process.whereis(World.Manager) do
        Logger.warning("World.Manager not available, skipping default world init")
        :ok
      else
        case Manager.get("default") do
          {:ok, _world} ->
            Logger.debug("Default world already exists")
            :ok

          {:error, :not_found} ->
            Logger.info("Creating default training world...")

            case Manager.create("default",
                   id: "default",
                   mode: :persistent,
                   base_world: nil,
                   metadata: %{description: "Default training world containing base data"}
                 ) do
              {:ok, world} ->
                Logger.info("Default world created", %{id: world.id})
                :ok

              {:error, reason} ->
                Logger.warning("Failed to create default world: #{inspect(reason)}")
                :error
            end
        end
      end
    end)
  end
end
