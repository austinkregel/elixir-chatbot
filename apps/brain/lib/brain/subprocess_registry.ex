defmodule Brain.SubprocessRegistry do
  @moduledoc """
  Registry for managing subprocess GenServers.
  Provides a centralized way to track and communicate with subprocesses.
  """

  use GenServer
  require Logger

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def register_subprocess(type, subprocess_id, pid) do
    GenServer.call(__MODULE__, {:register_subprocess, type, subprocess_id, pid})
  end

  def unregister_subprocess(type, subprocess_id) do
    GenServer.call(__MODULE__, {:unregister_subprocess, type, subprocess_id})
  end

  def list_subprocesses(type \\ nil) do
    GenServer.call(__MODULE__, {:list_subprocesses, type})
  end

  def get_subprocess(type, subprocess_id) do
    GenServer.call(__MODULE__, {:get_subprocess, type, subprocess_id})
  end

  def broadcast_to_subprocesses(type, message) do
    GenServer.cast(__MODULE__, {:broadcast_to_subprocesses, type, message})
  end

  @doc """
  Checks if the subprocess registry is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc """
  Gets statistics about the subprocess registry.
  """
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    state = %{
      subprocesses: %{}
    }

    Logger.info("Subprocess registry started")
    {:ok, state}
  end

  @impl true
  def handle_call({:register_subprocess, type, subprocess_id, pid}, _from, state) do
    key = {type, subprocess_id}

    case Registry.register(Brain.SubprocessRegistry, key, pid) do
      {:ok, _} ->
        updated_subprocesses =
          Map.put(state.subprocesses, key, %{
            type: type,
            subprocess_id: subprocess_id,
            pid: pid,
            registered_at: System.system_time(:millisecond)
          })

        Logger.info("Subprocess registered", %{
          type: type,
          subprocess_id: subprocess_id,
          pid: pid
        })

        {:reply, :ok, %{state | subprocesses: updated_subprocesses}}

      {:error, {:already_registered, _}} ->
        Logger.warning("Subprocess already registered", %{
          type: type,
          subprocess_id: subprocess_id
        })

        {:reply, {:error, :already_registered}, state}
    end
  end

  @impl true
  def handle_call({:unregister_subprocess, type, subprocess_id}, _from, state) do
    key = {type, subprocess_id}

    Registry.unregister(Brain.SubprocessRegistry, key)
    updated_subprocesses = Map.delete(state.subprocesses, key)

    Logger.info("Subprocess unregistered", %{
      type: type,
      subprocess_id: subprocess_id
    })

    {:reply, :ok, %{state | subprocesses: updated_subprocesses}}
  end

  @impl true
  def handle_call({:list_subprocesses, type}, _from, state) do
    subprocesses =
      state.subprocesses
      |> Map.values()
      |> Enum.filter(fn subprocess ->
        if type, do: subprocess.type == type, else: true
      end)
      |> Enum.map(fn subprocess ->
        %{
          type: subprocess.type,
          subprocess_id: subprocess.subprocess_id,
          pid: subprocess.pid,
          registered_at: subprocess.registered_at
        }
      end)

    {:reply, subprocesses, state}
  end

  @impl true
  def handle_call({:get_subprocess, type, subprocess_id}, _from, state) do
    key = {type, subprocess_id}

    case Registry.lookup(Brain.SubprocessRegistry, key) do
      [{_key, pid}] ->
        {:reply, {:ok, pid}, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    types =
      state.subprocesses
      |> Map.values()
      |> Enum.map(& &1.type)
      |> Enum.uniq()

    stats = %{
      registered_count: map_size(state.subprocesses),
      subprocess_types: types
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_cast({:broadcast_to_subprocesses, type, message}, state) do
    subprocesses =
      state.subprocesses
      |> Map.values()
      |> Enum.filter(fn subprocess -> subprocess.type == type end)

    Enum.each(subprocesses, fn subprocess ->
      try do
        GenServer.cast(subprocess.pid, message)
      catch
        :exit, {:noproc, _} ->
          # Process is dead, remove from registry
          key = {subprocess.type, subprocess.subprocess_id}
          Registry.unregister(Brain.SubprocessRegistry, key)
      end
    end)

    Logger.info("Broadcasted message to subprocesses", %{
      type: type,
      count: length(subprocesses)
    })

    {:noreply, state}
  end
end
