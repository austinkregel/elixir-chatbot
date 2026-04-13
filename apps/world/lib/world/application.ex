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
      wait_for_dependencies()

      unless Process.whereis(World.Manager) do
        Logger.warning("World.Manager not available, skipping default world init")
      else
        case Manager.get("default") do
          {:ok, _world} ->
            Logger.debug("Default world already exists")

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

              {:error, reason} ->
                Logger.warning("Failed to create default world: #{inspect(reason)}")
            end
        end
      end
    end)
  end

  defp wait_for_dependencies do
    require Logger

    wait_for_process(World.Manager, "World.Manager", 30)
    wait_for_process(Brain.ML.Gazetteer, "Brain.ML.Gazetteer", 30)

    if Process.whereis(Brain.ML.Gazetteer) do
      wait_for_gazetteer_loaded(60)
    end
  end

  defp wait_for_process(_name, label, 0) do
    require Logger
    Logger.warning("#{label} did not start in time")
  end

  defp wait_for_process(name, label, retries) do
    if Process.whereis(name) do
      :ok
    else
      Process.sleep(500)
      wait_for_process(name, label, retries - 1)
    end
  end

  defp wait_for_gazetteer_loaded(0) do
    require Logger
    Logger.warning("Gazetteer still loading after timeout, proceeding anyway")
  end

  defp wait_for_gazetteer_loaded(retries) do
    case Brain.ML.Gazetteer.stats() do
      %{loaded: true} -> :ok
      _ -> Process.sleep(500); wait_for_gazetteer_loaded(retries - 1)
    end
  rescue
    _ -> Process.sleep(500); wait_for_gazetteer_loaded(retries - 1)
  end
end
