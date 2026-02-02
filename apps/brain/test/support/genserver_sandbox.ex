defmodule Brain.GenServerSandbox do
  @moduledoc """
  Test sandbox for complete GenServer isolation per test.

  Provides two modes of operation:

  ## Mode 1: Fresh Global State (Simple)
  For most tests that just need clean state but can share global names:

      setup do
        Brain.GenServerSandbox.reset_global_state()
        :ok
      end

  ## Mode 2: Full Isolation (Edge Cases)
  For edge case tests that need completely isolated GenServer instances:

      setup do
        {:ok, ctx} = Brain.GenServerSandbox.checkout()
        {:ok, ctx}
      end

      test "edge case with custom data", %{sandbox: sandbox} do
        # Start isolated Brain with custom name
        {:ok, brain} = Brain.GenServerSandbox.start_isolated(sandbox, Brain, 
          {Brain, "priv/static/demo.echo.json"})
        
        # Use the isolated instance
        {:ok, conv_id} = GenServer.call(brain, {:create_conversation, []})
      end

  ## How It Works

  Each sandbox creates GenServers with unique names based on a sandbox ID:
  - `{:via, Registry, {Brain.GenServerSandbox.Registry, {Module, sandbox_id}}}`

  The sandbox tracks all started processes and cleans them up when the test exits.
  """

  use GenServer
  require Logger

  @registry __MODULE__.Registry
  @supervisor __MODULE__.Supervisor

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(init_arg) do
    {:ok, init_arg}
  end

  # ============================================================================
  # Simple Mode: Reset Global State
  # ============================================================================

  @doc """
  Resets all global GenServers to clean state.
  Use this for tests that don't need full isolation.
  """
  def reset_global_state do
    # Reset Brain if running
    if pid = Process.whereis(Brain) do
      try do
        Brain.reset_state(server: pid)
      catch
        _, _ -> :ok
      end
    end

    # Reload classifier models to ensure clean state
    if Process.whereis(Brain.ML.IntentClassifierSimple) do
      try do
        Brain.ML.IntentClassifierSimple.load_models()
      catch
        _, _ -> :ok
      end
    end

    # Reload gazetteer to ensure clean state
    if Process.whereis(Brain.ML.Gazetteer) do
      try do
        Brain.ML.Gazetteer.load_all()
      catch
        _, _ -> :ok
      end
    end

    :ok
  end

  # ============================================================================
  # Full Isolation Mode
  # ============================================================================

  @doc """
  Checks out a new sandbox for the current test.
  Returns a sandbox context that can be used to start isolated GenServers.

  The sandbox is automatically cleaned up when the test process exits.
  """
  def checkout do
    ensure_infrastructure()

    sandbox_id = generate_sandbox_id()
    owner_pid = self()

    # Track this sandbox
    :ets.insert(sandbox_table(), {sandbox_id, owner_pid, []})

    # Monitor the test process for cleanup
    spawn(fn ->
      ref = Process.monitor(owner_pid)

      receive do
        {:DOWN, ^ref, :process, ^owner_pid, _reason} ->
          cleanup_sandbox(sandbox_id)
      end
    end)

    # Register on_exit callback as well (belt and suspenders)
    ExUnit.Callbacks.on_exit(fn ->
      cleanup_sandbox(sandbox_id)
    end)

    sandbox = %{
      id: sandbox_id,
      owner: owner_pid,
      started: []
    }

    {:ok, %{sandbox: sandbox}}
  end

  @doc """
  Starts an isolated GenServer instance in the sandbox.

  Returns the PID of the started process.

  ## Examples

      # Start with just module (uses default start_link/1)
      {:ok, pid} = start_isolated(sandbox, Brain.MemoryStore)

      # Start with custom child spec
      {:ok, pid} = start_isolated(sandbox, Brain, 
        {Brain, "path/to/artifact.json"})

      # Start Gazetteer with isolated ETS tables
      {:ok, pid} = start_isolated(sandbox, Brain.ML.Gazetteer,
        {Brain.ML.Gazetteer, [table_prefix: :test_123]})
  """
  def start_isolated(%{sandbox: sandbox}, module, child_spec \\ nil) do
    child_spec = child_spec || module
    name = via_name(sandbox.id, module)

    spec = build_child_spec(child_spec, name)

    case DynamicSupervisor.start_child(@supervisor, spec) do
      {:ok, pid} ->
        track_started(sandbox.id, module, pid)
        {:ok, pid}

      {:error, {:already_started, pid}} ->
        {:ok, pid}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Gets the PID of an isolated GenServer in the sandbox.
  """
  def get(%{sandbox: sandbox}, module) do
    case Registry.lookup(@registry, {module, sandbox.id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc """
  Generates a :via tuple for registering a GenServer in the sandbox.
  """
  def via_name(sandbox_id, module) do
    {:via, Registry, {@registry, {module, sandbox_id}}}
  end

  @doc """
  Starts a complete isolated environment with all core services.
  Useful for integration tests that need everything isolated.
  """
  def start_isolated_environment(%{sandbox: sandbox} = ctx) do
    # Core services
    services = [
      {Brain.Metrics.Aggregator, Brain.Metrics.Aggregator},
      {Brain.ML.Gazetteer, {Brain.ML.Gazetteer, [table_prefix: sandbox.id]}},
      {Brain.Analysis.LearningStore, Brain.Analysis.LearningStore},
      {Brain.KnowledgeStore, Brain.KnowledgeStore},
      {Brain.MemoryStore, Brain.MemoryStore},
      {Brain.Memory.Embedder, Brain.Memory.Embedder},
      {Brain.Memory.Store, Brain.Memory.Store},
      {Brain.ML.IntentClassifierSimple, Brain.ML.IntentClassifierSimple}
    ]

    started =
      Enum.reduce(services, %{}, fn {module, spec}, acc ->
        case start_isolated(ctx, module, spec) do
          {:ok, pid} ->
            Map.put(acc, module, pid)

          {:error, reason} ->
            Logger.warning("Failed to start isolated #{inspect(module)}: #{inspect(reason)}")
            acc
        end
      end)

    # Load models into isolated services
    if classifier = Map.get(started, Brain.ML.IntentClassifierSimple) do
      try do
        GenServer.call(classifier, {:load_model, "default"}, 10_000)
      catch
        _, _ -> :ok
      end
    end

    if gazetteer = Map.get(started, Brain.ML.Gazetteer) do
      try do
        GenServer.call(gazetteer, :load_all, 30_000)
      catch
        _, _ -> :ok
      end
    end

    {:ok, Map.put(ctx, :services, started)}
  end

  @doc """
  Starts an isolated Brain with all its dependencies.
  """
  def start_isolated_brain(%{sandbox: sandbox} = ctx, artifact_path \\ "priv/static/demo.echo.json") when is_map(sandbox) do
    # Ensure environment is set up first
    {:ok, ctx} = start_isolated_environment(ctx)

    # Start Brain
    brain_spec = {Brain, artifact_path}
    {:ok, brain_pid} = start_isolated(ctx, Brain, brain_spec)

    {:ok, Map.put(ctx, :brain, brain_pid)}
  end

  # ============================================================================
  # Infrastructure
  # ============================================================================

  defp ensure_infrastructure do
    # Start Registry
    case Registry.start_link(keys: :unique, name: @registry) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    # Start DynamicSupervisor
    case DynamicSupervisor.start_link(strategy: :one_for_one, name: @supervisor) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    # Create ETS table for tracking sandboxes
    if :ets.whereis(sandbox_table()) == :undefined do
      :ets.new(sandbox_table(), [:set, :public, :named_table])
    end

    # Ensure PubSub is running
    case Phoenix.PubSub.Supervisor.start_link(name: Brain.PubSub) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    :ok
  end

  defp sandbox_table, do: :genserver_sandbox_tracking

  defp generate_sandbox_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16(case: :lower) |> String.to_atom()
  end

  defp track_started(sandbox_id, module, pid) do
    case :ets.lookup(sandbox_table(), sandbox_id) do
      [{^sandbox_id, owner, started}] ->
        :ets.insert(sandbox_table(), {sandbox_id, owner, [{module, pid} | started]})

      [] ->
        :ok
    end
  end

  defp cleanup_sandbox(sandbox_id) do
    case :ets.lookup(sandbox_table(), sandbox_id) do
      [{^sandbox_id, _owner, started}] ->
        # Stop all started processes
        Enum.each(started, fn {module, pid} ->
          Logger.debug("Sandbox: Stopping #{inspect(module)} (#{inspect(pid)})")

          try do
            DynamicSupervisor.terminate_child(@supervisor, pid)
          catch
            :exit, _ -> :ok
          end
        end)

        # Remove from tracking table
        :ets.delete(sandbox_table(), sandbox_id)

      [] ->
        :ok
    end
  end

  defp build_child_spec(module, name) when is_atom(module) do
    %{
      id: make_ref(),
      start: {module, :start_link, [[name: name]]},
      restart: :temporary
    }
  end

  defp build_child_spec({module, opts}, name) when is_atom(module) and is_list(opts) do
    # Merge name into opts
    opts = Keyword.put(opts, :name, name)

    %{
      id: make_ref(),
      start: {module, :start_link, [opts]},
      restart: :temporary
    }
  end

  defp build_child_spec({module, arg}, name) when is_atom(module) do
    # Module with a single argument (like Brain with artifact_path)
    %{
      id: make_ref(),
      start: {module, :start_link, [arg, [name: name]]},
      restart: :temporary
    }
  end
end
