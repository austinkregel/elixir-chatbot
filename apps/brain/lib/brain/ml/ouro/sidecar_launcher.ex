defmodule Brain.ML.Ouro.SidecarLauncher do
  @moduledoc """
  Manages the lifecycle of the Ouro Python sidecar process.

  Spawns `scripts/ouro_server.py` via a Port, monitors it for unexpected
  exits, and polls `/health` until the model is loaded and ready.

  Configuration (under `config :brain, :ml`):

    * `:ouro_auto_start` -- whether to launch the sidecar automatically
      (default `false`; set to `true` in test config)
    * `:ouro_python_cmd` -- Python binary (default `"python3"`)
    * `:ouro_venv_activate` -- path to a venv activate script (optional)
    * `:ouro_server_script` -- path to the server script
      (default `"scripts/ouro_server.py"`)
    * `:ouro_api_url` -- base URL the sidecar will listen on
      (default `"http://localhost:8100"`)
  """

  use GenServer
  require Logger

  alias Brain.ML.Ouro.Client

  defstruct [:port, :os_pid, :status, :exit_reason]

  @health_poll_interval 1_000

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Blocks until the sidecar health endpoint returns ok, or raises after timeout.

  If auto_start is disabled and the sidecar is already reachable, returns
  immediately. If the sidecar cannot be reached within the timeout, raises
  with a diagnostic message.
  """
  def ensure_ready!(opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 120_000)
    name = Keyword.get(opts, :name, __MODULE__)
    deadline = System.monotonic_time(:millisecond) + timeout

    poll_until_healthy(name, deadline)
  end

  @doc "Returns the current status: :starting, :healthy, :stopped, or {:crashed, reason}."
  def status(name \\ __MODULE__) do
    try do
      GenServer.call(name, :status, 5_000)
    catch
      :exit, _ -> :not_running
    end
  end

  @doc "Stops the sidecar process if one is running."
  def stop(name \\ __MODULE__) do
    GenServer.call(name, :stop, 10_000)
  catch
    :exit, _ -> :ok
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(_opts) do
    ml_config = Application.get_env(:brain, :ml, [])
    auto_start = ml_config[:ouro_auto_start] || false

    if auto_start do
      case launch_sidecar() do
        {:ok, port, os_pid} ->
          schedule_health_poll()
          {:ok, %__MODULE__{port: port, os_pid: os_pid, status: :starting}}

        {:error, reason} ->
          Logger.error("SidecarLauncher: failed to start sidecar: #{inspect(reason)}")
          {:ok, %__MODULE__{status: {:crashed, reason}}}
      end
    else
      {:ok, %__MODULE__{status: :disabled}}
    end
  end

  @impl true
  def handle_call(:status, _from, state) do
    {:reply, state.status, state}
  end

  def handle_call(:stop, _from, state) do
    state = kill_sidecar(state)
    {:reply, :ok, %{state | status: :stopped}}
  end

  @impl true
  def handle_info(:health_poll, %{status: :starting} = state) do
    case Client.health_check() do
      :ok ->
        Logger.info("SidecarLauncher: Ouro sidecar is healthy and ready")
        {:noreply, %{state | status: :healthy}}

      {:error, _reason} ->
        schedule_health_poll()
        {:noreply, state}
    end
  end

  def handle_info(:health_poll, state) do
    {:noreply, state}
  end

  def handle_info({port, {:exit_status, code}}, %{port: port} = state) do
    reason = exit_code_reason(code)
    Logger.error("SidecarLauncher: Python sidecar exited (code=#{code}): #{reason}")
    {:noreply, %{state | port: nil, os_pid: nil, status: {:crashed, reason}}}
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) do
    for line <- String.split(to_string(data), "\n", trim: true) do
      Logger.debug("ouro_server.py: #{line}")
    end

    {:noreply, state}
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end

  @impl true
  def terminate(_reason, state) do
    kill_sidecar(state)
    :ok
  end

  # --- Private ---

  defp launch_sidecar do
    ml_config = Application.get_env(:brain, :ml, [])
    python_cmd = ml_config[:ouro_python_cmd] || "python3"
    script = ml_config[:ouro_server_script] || find_server_script()
    api_url = ml_config[:ouro_api_url] || "http://localhost:8100"

    port_number =
      case URI.parse(api_url) do
        %URI{port: p} when is_integer(p) -> p
        _ -> 8100
      end

    unless File.exists?(script) do
      {:error, "Ouro server script not found at #{script}"}
    else
      cmd = build_command(python_cmd, script, port_number, ml_config)
      Logger.info("SidecarLauncher: starting Ouro sidecar: #{cmd}")

      try do
        port =
          Port.open({:spawn, cmd}, [
            :binary,
            :exit_status,
            :use_stdio,
            :stderr_to_stdout,
            {:env, python_env(ml_config)}
          ])

        {:os_pid, os_pid} = Port.info(port, :os_pid)
        {:ok, port, os_pid}
      rescue
        e -> {:error, Exception.message(e)}
      end
    end
  end

  defp build_command(python_cmd, script, port, ml_config) do
    venv_activate = ml_config[:ouro_venv_activate]

    base_cmd = "#{python_cmd} #{script} --port #{port}"

    if venv_activate && File.exists?(venv_activate) do
      "bash -c 'source #{venv_activate} && #{base_cmd}'"
    else
      base_cmd
    end
  end

  defp python_env(ml_config) do
    env = []

    case ml_config[:ouro_venv_path] do
      nil ->
        env

      venv_path ->
        venv_bin = Path.join(venv_path, "bin")
        current_path = System.get_env("PATH") || ""

        [
          {~c"PATH", String.to_charlist("#{venv_bin}:#{current_path}")},
          {~c"VIRTUAL_ENV", String.to_charlist(venv_path)}
          | env
        ]
    end
  end

  defp find_server_script do
    candidates = [
      "scripts/ouro_server.py",
      Path.join(File.cwd!(), "scripts/ouro_server.py"),
      Path.expand("../../../../../../scripts/ouro_server.py", __DIR__)
    ]

    Enum.find(candidates, "scripts/ouro_server.py", &File.exists?/1)
  end

  defp kill_sidecar(%{os_pid: nil} = state), do: state

  defp kill_sidecar(%{os_pid: os_pid, port: port} = state) do
    Logger.info("SidecarLauncher: stopping sidecar (pid=#{os_pid})")

    try do
      System.cmd("kill", [to_string(os_pid)])
    rescue
      _ -> :ok
    end

    if port do
      try do
        Port.close(port)
      rescue
        _ -> :ok
      end
    end

    %{state | port: nil, os_pid: nil}
  end

  defp schedule_health_poll do
    Process.send_after(self(), :health_poll, @health_poll_interval)
  end

  defp poll_until_healthy(name, deadline) do
    cond do
      Client.health_check() == :ok ->
        :ok

      System.monotonic_time(:millisecond) >= deadline ->
        status = status(name)
        raise_startup_failure(status)

      true ->
        Process.sleep(1_000)
        poll_until_healthy(name, deadline)
    end
  end

  defp raise_startup_failure(status) do
    reason =
      case status do
        {:crashed, msg} -> "Sidecar crashed: #{msg}"
        :starting -> "Sidecar is still loading the model (health check not passing)"
        :disabled -> "Sidecar auto-start is disabled"
        other -> "Status: #{inspect(other)}"
      end

    raise """
    Ouro sidecar failed to become ready.
    #{reason}

    Ensure the Python environment is set up:
      python3 -m venv .venv-ouro
      source .venv-ouro/bin/activate
      pip install -r scripts/requirements-ouro.txt

    Or start the sidecar manually before running tests:
      python scripts/ouro_server.py
    """
  end

  defp exit_code_reason(0), do: "clean shutdown"
  defp exit_code_reason(1), do: "general error (check Python traceback above)"
  defp exit_code_reason(2), do: "missing module or import error"
  defp exit_code_reason(127), do: "python command not found"
  defp exit_code_reason(code), do: "exit code #{code}"
end
