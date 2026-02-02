defmodule Brain.Subprocesses.Supervisor do
  @moduledoc """
  Supervisor for managing subprocess GenServers.
  Provides dynamic supervision of HTTP, Conversation, and CLI subprocesses.
  """

  use DynamicSupervisor
  require Logger

  # Client API

  def start_link(opts \\ []) do
    DynamicSupervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def start_http_subprocess(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    port = Keyword.get(opts, :port, 7878)
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    child_spec =
      {Brain.Subprocesses.HttpSubprocess,
       [
         subprocess_id: subprocess_id,
         port: port,
         memory_snapshot: memory_snapshot
       ]}

    case DynamicSupervisor.start_child(__MODULE__, child_spec) do
      {:ok, pid} ->
        Logger.info("HTTP subprocess started", %{
          subprocess_id: subprocess_id,
          port: port,
          pid: pid
        })

        {:ok, pid, subprocess_id}

      {:error, reason} ->
        Logger.error("Failed to start HTTP subprocess", %{
          subprocess_id: subprocess_id,
          reason: reason
        })

        {:error, reason}
    end
  end

  def start_conversation_subprocess(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    conversation_id = Keyword.get(opts, :conversation_id)
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    child_spec =
      {Brain.Subprocesses.ConversationSubprocess,
       [
         subprocess_id: subprocess_id,
         conversation_id: conversation_id,
         memory_snapshot: memory_snapshot
       ]}

    case DynamicSupervisor.start_child(__MODULE__, child_spec) do
      {:ok, pid} ->
        Logger.info("Conversation subprocess started", %{
          subprocess_id: subprocess_id,
          conversation_id: conversation_id,
          pid: pid
        })

        {:ok, pid, subprocess_id}

      {:error, reason} ->
        Logger.error("Failed to start conversation subprocess", %{
          subprocess_id: subprocess_id,
          conversation_id: conversation_id,
          reason: reason
        })

        {:error, reason}
    end
  end

  def start_cli_subprocess(opts \\ []) do
    subprocess_id = Keyword.get(opts, :subprocess_id, generate_id())
    memory_snapshot = Keyword.get(opts, :memory_snapshot, %{})

    child_spec =
      {Brain.Subprocesses.CliSubprocess,
       [
         subprocess_id: subprocess_id,
         memory_snapshot: memory_snapshot
       ]}

    case DynamicSupervisor.start_child(__MODULE__, child_spec) do
      {:ok, pid} ->
        Logger.info("CLI subprocess started", %{
          subprocess_id: subprocess_id,
          pid: pid
        })

        {:ok, pid, subprocess_id}

      {:error, reason} ->
        Logger.error("Failed to start CLI subprocess", %{
          subprocess_id: subprocess_id,
          reason: reason
        })

        {:error, reason}
    end
  end

  def stop_subprocess(pid) do
    case DynamicSupervisor.terminate_child(__MODULE__, pid) do
      :ok ->
        Logger.info("Subprocess stopped", %{pid: pid})
        :ok

      {:error, reason} ->
        Logger.error("Failed to stop subprocess", %{pid: pid, reason: reason})
        {:error, reason}
    end
  end

  def list_subprocesses do
    DynamicSupervisor.which_children(__MODULE__)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    Logger.info("Subprocess supervisor started")
    DynamicSupervisor.init(strategy: :one_for_one)
  end

  # Private Functions

  defp generate_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower)
  end
end
