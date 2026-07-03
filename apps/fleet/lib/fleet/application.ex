defmodule Fleet.Application do
  @moduledoc """
  The Fleet OTP application: the runtime spine of the agent-orchestration layer.

  Starts the addressing + supervision the crew needs to exist as live processes:

    * `Fleet.Registry` — unique-key registry; every ensign is addressed by
      `{:ensign, agent_id}` via-tuple (concepts are data, processes are keyed).
    * `Fleet.TaskSupervisor` — supervises the off-mailbox Tasks an ensign spawns
      for heavy cognition, so an ensign GenServer never blocks.
    * `Fleet.CrewSupervisor` — the `DynamicSupervisor` that spawns/retires ensigns.

  Telemetry handlers attach after the tree is up, mirroring `Brain.Application`.
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      {Registry, keys: :unique, name: Fleet.Registry},
      {Task.Supervisor, name: Fleet.TaskSupervisor},
      Fleet.CrewSupervisor
    ]

    opts = [strategy: :one_for_one, name: Fleet.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Fleet.Telemetry.attach_handlers()
        Logger.info("Fleet application started")
        {:ok, pid}

      other ->
        other
    end
  end
end
