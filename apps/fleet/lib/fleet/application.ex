defmodule Fleet.Application do
  @moduledoc """
  The Fleet OTP application: the runtime spine of the agent-orchestration layer.

  Starts the addressing + supervision the crew needs to exist as live processes:

    * `Fleet.Registry` — unique-key registry; every officer is addressed by
      `{:officer, agent_id}` via-tuple (concepts are data, processes are keyed).
    * `Fleet.TaskSupervisor` — supervises the off-mailbox Tasks an officer spawns
      for heavy cognition, so an officer GenServer never blocks.
    * `Fleet.CrewSupervisor` — the `DynamicSupervisor` that spawns/retires officers.

  Telemetry handlers attach after the tree is up, mirroring `Brain.Application`.
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    children = [
      {Registry, keys: :unique, name: Fleet.Registry},
      {Task.Supervisor, name: Fleet.TaskSupervisor},
      # The holodeck — per-officer workspaces. It holds the Docker socket; the
      # crew only ever proposes workspace tools. Supervised here (not in
      # fourth_wall) so it runs exactly where the crew does and fourth_wall stays
      # a stateless library. Disabled by default (backend :unavailable) until an
      # operator configures the real backend.
      FourthWall.Holodeck,
      Fleet.CrewSupervisor,
      # The ship's black box — periodically samples Fleet.Systems into a live ETS
      # ring + durable rows + a "systems:status" broadcast.
      Fleet.Systems.Sampler,
      # The alarm layer — raises/resolves alerts off the sampler's stream (the
      # relief-of-duty / fleet-brake substrate).
      Fleet.Systems.Monitor
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
