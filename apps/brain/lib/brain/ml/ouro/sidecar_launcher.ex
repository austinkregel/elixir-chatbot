defmodule Brain.ML.Ouro.SidecarLauncher do
  @moduledoc """
  Manages the lifecycle of the Ouro Python sidecar process.

  Spawns `scripts/ouro_server.py` via a Port, monitors it for unexpected
  exits, and polls `/health` until the model is loaded and ready.

  Configuration (under `config :brain, :ml`):

    * `:ouro_auto_start` -- whether to launch the sidecar in `init/1` when the
      app starts (default `false`). In test, keep this `false` and call
      `ensure_ready!/1` from `test_helper` so Python is not spawned twice
      relative to test setup and manual sidecars are easier to reason about.
    * `:ouro_python_cmd` -- Python binary (default `"python3"`; in test,
      `config/test.exs` points at the umbrella `.venv` unless `OURO_VENV` is set)
    * `:ouro_venv_path` -- venv root; prepends `bin` to `PATH` for the port (optional)
    * `:ouro_venv_activate` -- path to a venv activate script (optional)
    * `:ouro_server_script` -- path to the server script
      (default `"scripts/ouro_server.py"`)
    * `:ouro_api_url` -- base URL the sidecar will listen on
      (default `"http://localhost:8100"`; in `config/test.exs` a high port is
      chosen so tests do not fight a dev server on 8100; override with env
      `OURO_PORT`)
  """

  use GenServer
  require Logger

  alias Brain.ML.Ouro.Client

  defstruct [:port, :os_pid, :status, :exit_reason, :output_buffer, :launched_at, poll_count: 0]

  @output_buffer_max 48_000

  # Python boot (transformers import + MPS warm-up + model load) is ~5s
  # cold and slower under contention. Poll at 2s rather than 1s so we
  # don't fill the log with `connection refused` warnings before Python
  # has had a chance to bind the socket.
  @health_poll_interval 2_000

  # How often (in poll iterations) to log a "still starting" diagnostic
  # while the sidecar is loading. With @health_poll_interval = 2_000ms,
  # 5 means once every 10 seconds. Keeps the noise down but ensures the
  # operator gets a heartbeat with the last Python output line.
  @starting_diag_every_n_polls 5

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Ensures the sidecar process is running (lazy start when `ouro_auto_start` is
  false), then blocks until the health endpoint returns ok.

  When `:ouro_enabled` is false for the app, returns `:ok` without launching.
  """
  def ensure_ready!(opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 120_000)
    name = Keyword.get(opts, :name, __MODULE__)
    deadline = System.monotonic_time(:millisecond) + timeout
    launch_timeout = min(timeout, 60_000)

    case GenServer.call(name, :ensure_running, launch_timeout) do
      :skipped ->
        :ok

      :ok ->
        poll_until_healthy(name, deadline)

      {:error, reason} ->
        raise """
        Ouro sidecar failed to launch: #{inspect(reason)}

        Ensure the Python environment is set up:
          pip install -r scripts/requirements-ouro.txt

        Or start the sidecar manually before running tests:
          python scripts/ouro_server.py
        """
    end
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
    ouro_enabled = Application.get_env(:brain, :ouro_enabled, true)

    if auto_start and ouro_enabled do
      case launch_sidecar() do
        {:ok, port, os_pid} ->
          schedule_health_poll()
          {:ok,
           %__MODULE__{
             port: port,
             os_pid: os_pid,
             status: :starting,
             output_buffer: "",
             launched_at: System.monotonic_time(:millisecond)
           }}

        {:error, reason} ->
          Logger.error("SidecarLauncher: failed to start sidecar: #{inspect(reason)}")
          {:ok, %__MODULE__{status: {:crashed, reason}, output_buffer: ""}}
      end
    else
      {:ok, %__MODULE__{status: :disabled, output_buffer: ""}}
    end
  end

  @impl true
  def handle_call(:ensure_running, _from, state) do
    if not Application.get_env(:brain, :ouro_enabled, true) do
      {:reply, :skipped, state}
    else
      cond do
        state.status == :healthy ->
          {:reply, :ok, state}

        state.port != nil ->
          {:reply, :ok, state}

        true ->
          case launch_sidecar() do
            {:ok, port, os_pid} ->
              schedule_health_poll()

              {:reply, :ok,
               %{
                 state
                 | port: port,
                   os_pid: os_pid,
                   status: :starting,
                   output_buffer: state.output_buffer || "",
                   launched_at: System.monotonic_time(:millisecond)
               }}

            {:error, reason} ->
              {:reply, {:error, reason},
               %{state | status: {:crashed, reason}, port: nil, os_pid: nil, output_buffer: state.output_buffer || ""}}
          end
      end
    end
  end

  def handle_call(:status, _from, state) do
    {:reply, state.status, state}
  end

  def handle_call(:diagnostics, _from, state) do
    diag = %{
      status: state.status,
      os_pid: state.os_pid,
      output_buffer: state.output_buffer || "",
      uptime_ms: elapsed_ms(state.launched_at)
    }

    {:reply, diag, state}
  end

  def handle_call(:stop, _from, state) do
    state = kill_sidecar(state)
    {:reply, :ok, %{state | status: :stopped}}
  end

  @impl true
  def handle_info(:health_poll, %{status: :starting} = state) do
    case Client.health_check() do
      :ok ->
        elapsed = elapsed_ms(state.launched_at)
        Logger.info("SidecarLauncher: Ouro sidecar is healthy and ready (boot took #{elapsed}ms)")
        {:noreply, %{state | status: :healthy}}

      {:error, reason} ->
        new_count = (state.poll_count || 0) + 1
        log_starting_heartbeat(%{state | poll_count: new_count}, reason)
        schedule_health_poll()
        {:noreply, %{state | poll_count: new_count}}
    end
  end

  def handle_info(:health_poll, state) do
    {:noreply, state}
  end

  def handle_info({port, {:exit_status, code}}, %{port: port} = state) do
    reason = exit_code_reason(code)
    msg = "SidecarLauncher: Python sidecar exited (code=#{code}): #{reason}"
    Logger.error(msg)
    IO.puts(:stderr, "[ouro] " <> msg)

    buf = state.output_buffer || ""

    if buf != "" do
      Logger.error("SidecarLauncher: Python process output (stderr+stdout):\n#{buf}")

      # Also dump the buffer directly so it's visible in test runs where
      # Logger is configured at :warning (Logger.error is shown but multi-line
      # interpolation can be eaten by formatters).
      IO.puts(:stderr, "[ouro] --- last #{byte_size(buf)} bytes of Python output ---")
      IO.puts(:stderr, buf)
      IO.puts(:stderr, "[ouro] --- end Python output ---")
    end

    {:noreply, %{state | port: nil, os_pid: nil, status: {:crashed, reason}, output_buffer: ""}}
  end

  def handle_info({port, {:data, data}}, %{port: port} = state) do
    text = to_string(data)

    # Mirror every line to the host's stderr, prefixed so it's obvious where
    # it came from. This bypasses Logger entirely so test/iex sessions see
    # Python tracebacks, model-load progress, uvicorn errors, etc. in real
    # time — regardless of the configured Logger level.
    #
    # Lines are also forwarded through Logger so file-based handlers and
    # production telemetry capture them; we use `info` (not `debug`) so the
    # default test-env `:warning` filter still suppresses them, but bumping
    # to `:info` (`config :logger, level: :info`) shows the full Python
    # boot trace without code changes. Lines that look like Python
    # tracebacks/errors are escalated to `:warning` so they're visible
    # even at the test-default log level.
    for line <- String.split(text, "\n", trim: true) do
      IO.puts(:stderr, "[ouro] " <> line)

      if python_error_line?(line) do
        Logger.warning("ouro_server.py: #{line}")
      else
        Logger.info("ouro_server.py: #{line}")
      end
    end

    {:noreply, %{state | output_buffer: append_output_buffer(state.output_buffer, text)}}
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
    # PYTHONUNBUFFERED=1 disables stdio buffering inside Python so every
    # `print(...)` and Logger call reaches our port (and therefore the host
    # stderr) immediately. Without this, transformers' download progress and
    # uvicorn startup banner can be hidden behind a 4 KiB pipe buffer until
    # the process exits, which makes debugging launch failures impossible.
    base_env = [
      {~c"PYTHONUNBUFFERED", ~c"1"}
    ]

    case ml_config[:ouro_venv_path] do
      nil ->
        base_env

      venv_path ->
        venv_bin = Path.join(venv_path, "bin")
        current_path = System.get_env("PATH") || ""

        [
          {~c"PATH", String.to_charlist("#{venv_bin}:#{current_path}")},
          {~c"VIRTUAL_ENV", String.to_charlist(venv_path)}
          | base_env
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

  defp append_output_buffer(nil, add), do: append_output_buffer("", add)

  defp append_output_buffer(current, add) when is_binary(current) and is_binary(add) do
    merged = current <> add

    if byte_size(merged) <= @output_buffer_max do
      merged
    else
      binary_part(merged, byte_size(merged) - @output_buffer_max, @output_buffer_max)
    end
  end

  defp schedule_health_poll do
    Process.send_after(self(), :health_poll, @health_poll_interval)
  end

  # `ensure_ready!/1` waits on the launcher's *own* status rather than
  # racing it on `/health`. The launcher already has a `health_poll` loop
  # that flips status to `:healthy` the moment Python answers; reusing
  # that single source of truth keeps us from doubling the request rate
  # (and the warning rate) against a sidecar that's still booting.
  defp poll_until_healthy(name, deadline) do
    case status(name) do
      :healthy ->
        :ok

      {:crashed, _} = crashed ->
        # Fail fast — no point polling the remaining timeout when the
        # launcher has already concluded Python died.
        raise_startup_failure(%{
          status: crashed,
          os_pid: nil,
          output_buffer: tail_buffer(name),
          uptime_ms: nil
        })

      _other ->
        if System.monotonic_time(:millisecond) >= deadline do
          raise_startup_failure(diagnostics(name))
        else
          Process.sleep(500)
          poll_until_healthy(name, deadline)
        end
    end
  end

  defp tail_buffer(name) do
    case diagnostics(name) do
      %{output_buffer: buf} when is_binary(buf) -> buf
      _ -> ""
    end
  end

  # Public-but-internal helper used by `poll_until_healthy/2` and tests
  # that want to introspect what Python has been saying.
  defp diagnostics(name) do
    GenServer.call(name, :diagnostics, 5_000)
  catch
    :exit, _ -> %{status: :not_running, os_pid: nil, output_buffer: "", uptime_ms: nil}
  end

  defp raise_startup_failure(diag) do
    reason =
      case diag.status do
        {:crashed, msg} -> "Sidecar crashed: #{msg}"
        :starting -> "Sidecar is still loading the model (health check not passing)"
        :disabled ->
          "Sidecar was never launched (no health yet); ensure `ensure_ready!/1` ran or set `ouro_auto_start`"
        other -> "Status: #{inspect(other)}"
      end

    tail = format_buffer_tail(diag.output_buffer, 4_000)
    uptime = if is_integer(diag.uptime_ms), do: " (uptime #{diag.uptime_ms}ms)", else: ""
    pid_info = if diag.os_pid, do: " os_pid=#{diag.os_pid}", else: ""

    raise """
    Ouro sidecar failed to become ready#{uptime}.
    #{reason}#{pid_info}

    Last Python stdio (most recent ~4KB):
    #{tail}

    Ensure the Python environment is set up:
      python3 -m venv .venv-ouro
      source .venv-ouro/bin/activate
      pip install -r scripts/requirements-ouro.txt

    Or start the sidecar manually before running tests:
      python scripts/ouro_server.py
    """
  end

  defp format_buffer_tail("", _max), do: "<no output captured from Python yet>"
  defp format_buffer_tail(nil, _max), do: "<no output captured from Python yet>"

  defp format_buffer_tail(buffer, max) when is_binary(buffer) do
    sliced =
      if byte_size(buffer) > max do
        binary_part(buffer, byte_size(buffer) - max, max)
      else
        buffer
      end

    sliced
    |> String.split("\n", trim: false)
    |> Enum.map(&("  | " <> &1))
    |> Enum.join("\n")
  end

  defp exit_code_reason(0), do: "clean shutdown"
  defp exit_code_reason(1), do: "general error (check Python traceback above)"
  defp exit_code_reason(2), do: "missing module or import error"
  defp exit_code_reason(127), do: "python command not found"
  defp exit_code_reason(code), do: "exit code #{code}"

  defp elapsed_ms(nil), do: 0

  defp elapsed_ms(launched_at) when is_integer(launched_at) do
    System.monotonic_time(:millisecond) - launched_at
  end

  # Periodic heartbeat while the sidecar is still loading. We surface this
  # at `:warning` (not `:info`) because the test config filters info, and
  # the operator running tests cares about *why* health is failing for
  # tens of seconds. Throttled to every Nth poll so the log stays sane.
  defp log_starting_heartbeat(%{poll_count: count} = state, reason) do
    if rem(count, @starting_diag_every_n_polls) == 0 do
      elapsed = elapsed_ms(state.launched_at)
      tail = last_python_line(state.output_buffer)

      Logger.warning(
        "SidecarLauncher: still starting (elapsed=#{elapsed}ms, polls=#{count}); " <>
          "health=#{inspect(reason)}; last python line: #{tail}"
      )
    end

    :ok
  end

  defp last_python_line(nil), do: "<no output yet>"
  defp last_python_line(""), do: "<no output yet>"

  defp last_python_line(buffer) when is_binary(buffer) do
    case buffer |> String.split("\n", trim: true) |> List.last() do
      nil -> "<no output yet>"
      line -> String.slice(line, 0, 240)
    end
  end

  # Heuristic: lines that look like Python tracebacks/errors. Promoting
  # these to `:warning` ensures they appear at the test-default log level
  # without flooding the operator with normal model-load logs.
  defp python_error_line?(line) do
    String.contains?(line, [
      "Traceback (most recent",
      "Error:",
      "ERROR",
      "CRITICAL",
      "OSError",
      "RuntimeError",
      "ImportError",
      "ModuleNotFoundError",
      "torch.cuda.OutOfMemoryError",
      "MPSBackendOutOfMemoryError"
    ])
  end
end
