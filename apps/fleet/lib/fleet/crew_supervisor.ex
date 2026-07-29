defmodule Fleet.CrewSupervisor do
  @moduledoc """
  Dynamic supervisor of the crew. Each child is one `Fleet.Officer` GenServer.

  Commissioning a crew member = `start_officer/1`; retiring one = `retire/1`.
  The command hierarchy (ship, rank, chain) is *data* the officer carries, not
  supervision structure — this supervisor only owns process lifecycle.

  Mirrors `Brain.Subprocesses.Supervisor`.
  """

  use DynamicSupervisor
  require Logger

  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Commissions an officer. `opts` are passed to `Fleet.Officer.start_link/1`.

  Identity is **stable and soul-keyed**: `agent_id` defaults to `soul_id`, so
  the agent's durable self (service record, mind-world) can rehydrate across
  restarts. Re-commissioning the same soul is **idempotent** — if it is already
  running, its existing pid is returned (Registry key `{:officer, agent_id}` is
  unique); if it is not, a fresh officer starts and rehydrates from Postgres.
  Anonymous (soul-less) officers fall back to a random id and have no durable self.
  Returns `{:ok, pid, agent_id}`.
  """
  def start_officer(opts \\ []) do
    agent_id = opts[:agent_id] || opts[:soul_id] || generate_id()

    opts =
      opts
      |> Keyword.put(:agent_id, agent_id)
      |> Keyword.put_new(:soul_id, agent_id)

    case DynamicSupervisor.start_child(__MODULE__, {Fleet.Officer, opts}) do
      {:ok, pid} ->
        Logger.info("Officer commissioned", %{agent_id: agent_id, pid: pid})
        {:ok, pid, agent_id}

      {:error, {:already_started, pid}} ->
        Logger.info("Officer already commissioned; resuming", %{agent_id: agent_id, pid: pid})
        {:ok, pid, agent_id}

      {:error, reason} ->
        Logger.error("Failed to commission officer", %{agent_id: agent_id, reason: reason})
        {:error, reason}
    end
  end

  @doc "Retires (terminates) an officer by pid."
  def retire(pid) when is_pid(pid) do
    DynamicSupervisor.terminate_child(__MODULE__, pid)
  end

  @doc "Lists the currently commissioned crew (supervisor children)."
  def list do
    DynamicSupervisor.which_children(__MODULE__)
  end

  @impl true
  def init(_opts) do
    Logger.info("Crew supervisor started")
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
