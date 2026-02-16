defmodule Brain.GenServerSandbox do
  @moduledoc "Test sandbox for complete GenServer isolation per test.\n\nProvides two modes of operation:\n\n## Mode 1: Fresh Global State (Simple)\nFor most tests that just need clean state but can share global names:\n\n    setup do\n      Brain.GenServerSandbox.reset_global_state()\n      :ok\n    end\n\n## Mode 2: Full Isolation (Edge Cases)\nFor edge case tests that need completely isolated GenServer instances:\n\n    setup do\n      {:ok, ctx} = Brain.GenServerSandbox.checkout()\n      {:ok, ctx}\n    end\n\n    test \"edge case with custom data\", %{sandbox: sandbox} do\n      # Start isolated Brain with custom name\n      {:ok, brain} = Brain.GenServerSandbox.start_isolated(sandbox, Brain,\n        {Brain, \"priv/static/demo.echo.json\"})\n\n      # Use the isolated instance\n      {:ok, conv_id} = GenServer.call(brain, {:create_conversation, []})\n    end\n\n## How It Works\n\nEach sandbox creates GenServers with unique names based on a sandbox ID:\n- `{:via, Registry, {Brain.GenServerSandbox.Registry, {Module, sandbox_id}}}`\n\nThe sandbox tracks all started processes and cleans them up when the test exits.\n"

  alias ExUnit.Callbacks
  alias Brain.ML.Gazetteer
  use GenServer
  require Logger

  @registry __MODULE__.Registry
  @supervisor __MODULE__.Supervisor

  @impl true
  def init(init_arg) do
    {:ok, init_arg}
  end

  @doc "Resets all global GenServers to clean state.\nUse this for tests that don't need full isolation.\n"
  def reset_global_state do
    if pid = Process.whereis(Brain) do
      try do
        Brain.reset_state(server: pid)
      catch
        _, _ -> :ok
      end
    end

    if Process.whereis(Brain.ML.IntentClassifierSimple) do
      try do
        Brain.Test.ModelFactory.train_and_load_test_models()
      catch
        _, _ -> :ok
      end
    end

    if Process.whereis(Brain.ML.Gazetteer) do
      try do
        Gazetteer.load_all()
      catch
        _, _ -> :ok
      end
    end

    :ok
  end

  @doc "Checks out a new sandbox for the current test.\nReturns a sandbox context that can be used to start isolated GenServers.\n\nThe sandbox is automatically cleaned up when the test process exits.\n"
  def checkout do
    ensure_infrastructure()

    sandbox_id = generate_sandbox_id()
    owner_pid = self()
    :ets.insert(sandbox_table(), {sandbox_id, owner_pid, []})

    spawn(fn ->
      ref = Process.monitor(owner_pid)

      receive do
        {:DOWN, ^ref, :process, ^owner_pid, _reason} ->
          cleanup_sandbox(sandbox_id)
      end
    end)

    Callbacks.on_exit(fn ->
      cleanup_sandbox(sandbox_id)
    end)

    sandbox = %{
      id: sandbox_id,
      owner: owner_pid,
      started: []
    }

    {:ok, %{sandbox: sandbox}}
  end

  @doc "Starts an isolated GenServer instance in the sandbox.\n\nReturns the PID of the started process.\n\n## Examples\n\n    # Start with just module (uses default start_link/1)\n    {:ok, pid} = start_isolated(sandbox, Brain.MemoryStore)\n\n    # Start with custom child spec\n    {:ok, pid} = start_isolated(sandbox, Brain,\n      {Brain, \"path/to/artifact.json\"})\n\n    # Start Gazetteer with isolated ETS tables\n    {:ok, pid} = start_isolated(sandbox, Brain.ML.Gazetteer,\n      {Brain.ML.Gazetteer, [table_prefix: :test_123]})\n"
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

  @doc "Gets the PID of an isolated GenServer in the sandbox.\n"
  def get(%{sandbox: sandbox}, module) do
    case Registry.lookup(@registry, {module, sandbox.id}) do
      [{pid, _}] -> pid
      [] -> nil
    end
  end

  @doc "Generates a :via tuple for registering a GenServer in the sandbox.\n"
  def via_name(sandbox_id, module) do
    {:via, Registry, {@registry, {module, sandbox_id}}}
  end

  @doc "Starts a complete isolated environment with all core services.\nUseful for integration tests that need everything isolated.\n"
  def start_isolated_environment(%{sandbox: sandbox} = ctx) do
    services = [
      {Brain.Metrics.Aggregator, Brain.Metrics.Aggregator},
      {Brain.ML.Gazetteer, {Brain.ML.Gazetteer, table_prefix: sandbox.id}},
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

  @doc "Starts an isolated Brain with all its dependencies.\n"
  def start_isolated_brain(
        %{sandbox: sandbox} = ctx,
        artifact_path \\ "priv/static/demo.echo.json"
      )
      when is_map(sandbox) do
    {:ok, ctx} = start_isolated_environment(ctx)
    brain_spec = {Brain, artifact_path}
    {:ok, brain_pid} = start_isolated(ctx, Brain, brain_spec)

    {:ok, Map.put(ctx, :brain, brain_pid)}
  end

  defp ensure_infrastructure do
    case Registry.start_link(keys: :unique, name: @registry) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    case DynamicSupervisor.start_link(strategy: :one_for_one, name: @supervisor) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    if :ets.whereis(sandbox_table()) == :undefined do
      :ets.new(sandbox_table(), [:set, :public, :named_table])
    end

    case Phoenix.PubSub.Supervisor.start_link(name: Brain.PubSub) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end

    :ok
  end

  defp sandbox_table do
    :genserver_sandbox_tracking
  end

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
        Enum.each(started, fn {module, pid} ->
          Logger.debug("Sandbox: Stopping #{inspect(module)} (#{inspect(pid)})")

          try do
            DynamicSupervisor.terminate_child(@supervisor, pid)
          catch
            :exit, _ -> :ok
          end
        end)

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
    opts = Keyword.put(opts, :name, name)

    %{
      id: make_ref(),
      start: {module, :start_link, [opts]},
      restart: :temporary
    }
  end

  defp build_child_spec({module, arg}, name) when is_atom(module) do
    %{
      id: make_ref(),
      start: {module, :start_link, [arg, [name: name]]},
      restart: :temporary
    }
  end
end
