defmodule World.Application do
  @moduledoc false
  alias World.Manager
  use Application
  require Logger

  @default_world_id "default"

  @impl true
  def start(_type, _args) do
    # Worlds get graphs only once the unscoped graphs from before graphs were
    # per world have been moved into one. Checked before anything starts, so a
    # database still holding them fails this application's start with the
    # refusal's own message, naming the graphs and the task that moves them,
    # rather than with World.Manager's failed start wrapped in the supervisor's.
    :ok = Atlas.Graph.refuse_unscoped_graphs!()

    children = [World.Manager, World.ModelRegistry, World.EntityPromoter]

    opts = [strategy: :one_for_one, name: World.Supervisor]

    with {:ok, pid} <- Supervisor.start_link(children, opts) do
      ensure_default_world!()
      {:ok, pid}
    end
  end

  # The "default" world exists before this application has finished starting,
  # so nothing that starts after it (the web interface's user plug among them)
  # can ask for it and find it missing. Everything creation reads is already
  # up: World.Manager is started above, and the gazetteer it writes the world's
  # overlay to is in :brain, which :world depends on and which loads its
  # sources in init/1.
  defp ensure_default_world! do
    case Manager.get(@default_world_id) do
      {:ok, _world} ->
        Logger.debug("Default world already exists")

      {:error, :not_found} ->
        Logger.info("Creating default training world...")

        case Manager.create(@default_world_id,
               id: @default_world_id,
               mode: :persistent,
               base_world: nil,
               metadata: %{description: "Default training world containing base data"}
             ) do
          {:ok, world} ->
            Logger.info("Default world created", %{id: world.id})

          {:error, reason} ->
            raise "World.Application: could not create the #{inspect(@default_world_id)} world: #{inspect(reason)}"
        end

      {:error, reason} ->
        raise "World.Application: could not look up the #{inspect(@default_world_id)} world: #{inspect(reason)}"
    end
  end
end
