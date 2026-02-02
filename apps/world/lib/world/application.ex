defmodule World.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      World.Manager,
      World.ModelRegistry
    ]

    opts = [strategy: :one_for_one, name: World.Supervisor]
    result = Supervisor.start_link(children, opts)

    # Initialize default world after startup
    init_default_world()

    result
  end

  defp init_default_world do
    require Logger

    Task.start(fn ->
      # Give GenServers time to start
      Process.sleep(100)

      unless Process.whereis(World.Manager) do
        Logger.debug("Waiting for World.Manager to start...")
        Process.sleep(100)
      end

      unless Process.whereis(World.Manager) do
        Logger.warning("World.Manager not available, skipping default world init")
        :ok
      else
        # Ensure default world exists
        case World.Manager.get("default") do
          {:ok, _world} ->
            Logger.debug("Default world already exists")
            :ok

          {:error, :not_found} ->
            Logger.info("Creating default training world...")

            case World.Manager.create("default",
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
