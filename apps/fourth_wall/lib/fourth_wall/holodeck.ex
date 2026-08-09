defmodule FourthWall.Holodeck do
  @moduledoc """
  Per-officer workspaces: one confined, writable space per soul, created lazily
  and reaped when the officer retires.

  This is the ship's facility, not the officer's. The officer proposes workspace
  tools (`workspace.read/list/write`, `shell.run`); the ship — through this
  module — is the only thing that touches the backend, which holds the Docker
  socket. Propose-not-dispatch, extended from information to the filesystem.

  ## What lives where

    * **This GenServer** owns lifecycle: `ensure/1` creates a workspace at most
      once per soul (idempotent), `reap/1` tears it down. Serialising create and
      destroy through one process keeps two orders for the same soul from racing
      a container into existence twice.
    * **The backend** (`FourthWall.Holodeck.Backend`) is the mechanism —
      `Backend.Docker` is the real boundary, `Backend.Unavailable` the safe
      default, `Backend.Memory` the test double.
    * **`FourthWall.Holodeck.Policy`** validates every path and command first, so
      a refusal names the path or the command.

  Reads, writes, and command execution run in the **calling** process (an
  officer's cognition task), not here — a `mix compile` in one workspace must not
  block every other officer's file read. Only `ensure/1` and `reap/1` are
  serialised.

  Keyed by `soul_id`, the same durable identity as the mind-world
  (`Fleet.MindWorld`), so a soul's workspace persists across orders the way its
  memory does.
  """

  use GenServer
  require Logger

  alias FourthWall.Holodeck.Policy

  @default_backend FourthWall.Holodeck.Backend.Unavailable

  # ── Client ─────────────────────────────────────────────────────────────────

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: Keyword.get(opts, :name, __MODULE__))
  end

  @doc """
  Ensure a workspace exists for `soul_id`, returning `{:ok, ctx}` where `ctx`
  carries the backend module and its handle for the caller to operate on.

  Idempotent: a soul that already has a workspace gets its existing handle back.
  """
  @spec ensure(String.t()) :: {:ok, map()} | {:error, term()}
  def ensure(soul_id) when is_binary(soul_id), do: GenServer.call(server(), {:ensure, soul_id})
  def ensure(_), do: {:error, :no_workspace_identity}

  @doc "Read a file from the soul's workspace, confined to the workspace root."
  @spec read(String.t(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def read(soul_id, path), do: with_workspace(soul_id, path, &apply_backend(&1, :read, [&2]))

  @doc "List a directory in the soul's workspace, confined to the workspace root."
  @spec list(String.t(), String.t()) :: {:ok, String.t()} | {:error, term()}
  def list(soul_id, path), do: with_workspace(soul_id, path, &apply_backend(&1, :list, [&2]))

  @doc "Write a file to the soul's workspace, confined to the workspace root."
  @spec write(String.t(), String.t(), String.t()) :: :ok | {:error, term()}
  def write(soul_id, path, content) when is_binary(content),
    do: with_workspace(soul_id, path, &apply_backend(&1, :write, [&2, content]))

  @doc """
  Run a validated command in the soul's workspace.

  The argv is checked against `Policy.check_command/1` before the workspace is
  ensured, so a forbidden command is refused without side effects.
  """
  @spec run(String.t(), [String.t()]) :: {:ok, map()} | {:error, term()}
  def run(soul_id, argv) when is_binary(soul_id) and is_list(argv) do
    with :ok <- Policy.check_command(argv),
         {:ok, ctx} <- ensure(soul_id) do
      ctx.backend.exec(ctx.handle, argv)
    end
  end

  def run(_soul_id, _argv), do: {:error, :no_workspace_identity}

  @doc "Tear down the soul's workspace. Best-effort; safe to call when none exists."
  @spec reap(String.t()) :: :ok
  def reap(soul_id) when is_binary(soul_id), do: GenServer.call(server(), {:reap, soul_id})
  def reap(_), do: :ok

  # Confine the path, ensure the workspace, then run `fun.(ctx, abs_path)` in the
  # CALLER — heavy backend work stays off the lifecycle GenServer.
  defp with_workspace(soul_id, path, fun) when is_binary(soul_id) do
    with {:ok, abs} <- Policy.resolve_path(path),
         {:ok, ctx} <- ensure(soul_id) do
      fun.(ctx, abs)
    end
  end

  defp with_workspace(_soul_id, _path, _fun), do: {:error, :no_workspace_identity}

  defp apply_backend(ctx, op, args), do: apply(ctx.backend, op, [ctx.handle | args])

  # In an umbrella the Holodeck is a singleton, but tests start their own named
  # instance; `server/0` resolves the process-dictionary override a test sets.
  defp server, do: Process.get(:holodeck_server, __MODULE__)

  # ── Server ─────────────────────────────────────────────────────────────────

  @impl true
  def init(opts) do
    config = Application.get_env(:fourth_wall, :holodeck, [])
    backend = opts[:backend] || config[:backend] || @default_backend

    backend_opts =
      [
        image: opts[:image] || config[:image] || "chatbot-holodeck:latest",
        repo_path: opts[:repo_path] || config[:repo_path] || File.cwd!(),
        network: opts[:network] || config[:network] || "none"
      ]

    {:ok, %{backend: backend, opts: backend_opts, workspaces: %{}}}
  end

  @impl true
  def handle_call({:ensure, soul_id}, _from, state) do
    case Map.fetch(state.workspaces, soul_id) do
      {:ok, handle} ->
        {:reply, {:ok, %{backend: state.backend, handle: handle}}, state}

      :error ->
        case state.backend.create(soul_id, state.opts) do
          {:ok, handle} ->
            state = put_in(state.workspaces[soul_id], handle)
            {:reply, {:ok, %{backend: state.backend, handle: handle}}, state}

          {:error, _} = error ->
            {:reply, error, state}
        end
    end
  end

  @impl true
  def handle_call({:reap, soul_id}, _from, state) do
    case Map.pop(state.workspaces, soul_id) do
      {nil, workspaces} ->
        {:reply, :ok, %{state | workspaces: workspaces}}

      {handle, workspaces} ->
        _ = state.backend.destroy(handle)
        {:reply, :ok, %{state | workspaces: workspaces}}
    end
  end
end
