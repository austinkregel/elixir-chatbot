defmodule Brain.MetricsCase do
  @moduledoc """
  Case for tests of `Brain.Metrics.Aggregator` and the telemetry handlers
  that feed it.

  Each test checks out the Atlas sandbox (shared, so the aggregator writes
  inside the test's transaction and everything it writes is rolled back),
  has a running aggregator and the handlers `Brain.Telemetry.attach_handlers/0`
  attaches, as in the running application, and starts in a new node-wide
  telemetry window with no gauges held.

  The missed-event counters live outside the aggregator, for the whole node,
  so a test compares them before and after, never against zero. So does when
  a miss was last logged: a miss from a sender for a reason is logged as it
  happens only when none from that sender for that reason was logged in the
  last ten seconds, so a test that looks for the line logged as it happens
  misses from a sender, for a reason, no other test does.
  """

  use ExUnit.CaseTemplate

  alias Brain.Metrics.Aggregator

  using do
    quote do
      import Brain.MetricsCase
      alias Brain.Metrics.Aggregator
    end
  end

  setup tags do
    {:ok, _apps} = Application.ensure_all_started(:telemetry)
    # Running already when the application is up; started here otherwise.
    {:ok, _apps} = Application.ensure_all_started(:atlas)
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)
    {:ok, _pid} = Brain.TestHelpers.ensure_started(Aggregator)
    :ok = Brain.Telemetry.attach_handlers()
    :ok = Aggregator.reset()
    :ok
  end

  @doc """
  Waits until the aggregator has handled everything sent to it before and
  written it to Atlas: it handles its messages in order, so once it answers
  `Aggregator.flush/0`, it has.
  """
  def counted do
    :ok = Aggregator.flush()
  end

  @doc """
  Has the aggregator log, now, the misses it has not logged yet, as it does
  every ten seconds, and waits until it has.
  """
  def report_missed do
    send(Process.whereis(Aggregator), :report_missed)
    _state = :sys.get_state(Aggregator)
    :ok
  end

  @doc "How many lines of `log` contain `pattern`."
  def lines_with(log, pattern), do: log |> String.split("\n") |> Enum.count(&String.contains?(&1, pattern))

  @doc """
  Stops the aggregator and waits for its supervisor to start it again; its
  memory starts over, so whatever it reads afterwards it reads from Atlas.
  Returns the new process.
  """
  def restart_aggregator! do
    old = Process.whereis(Aggregator)
    ref = Process.monitor(old)
    :ok = GenServer.stop(old)

    receive do
      {:DOWN, ^ref, :process, ^old, _reason} -> :ok
    after
      5_000 -> raise "Brain.Metrics.Aggregator (#{inspect(old)}) did not stop"
    end

    restarted_aggregator!(old, 100)
  end

  defp restarted_aggregator!(old, 0) do
    raise "Brain.Metrics.Aggregator (was #{inspect(old)}) was not started again after it stopped"
  end

  defp restarted_aggregator!(old, attempts) do
    case Process.whereis(Aggregator) do
      pid when is_pid(pid) and pid != old ->
        pid

      _not_yet ->
        Process.sleep(10)
        restarted_aggregator!(old, attempts - 1)
    end
  end

  @doc """
  Stops the aggregator where it is supervised (the application's supervisor,
  or the test's), runs `fun` while it is not running, and starts it again.
  Returns what `fun` returns.
  """
  def while_aggregator_down(fun) when is_function(fun, 0) do
    case Process.whereis(Brain.Supervisor) do
      nil ->
        :ok = ExUnit.Callbacks.stop_supervised(Aggregator)
        refute_running!()
        result = fun.()
        {:ok, _pid} = ExUnit.Callbacks.start_supervised(Aggregator)
        result

      supervisor ->
        :ok = Supervisor.terminate_child(supervisor, Aggregator)
        refute_running!()
        result = fun.()
        {:ok, _pid} = Supervisor.restart_child(supervisor, Aggregator)
        result
    end
  end

  defp refute_running! do
    if Process.whereis(Aggregator), do: raise("Brain.Metrics.Aggregator is still running")
  end

  @doc "How many events from `source` were missed for `reason`, written to Atlas or held in memory."
  def missed(source, reason) do
    Aggregator.get_missed_events().by_source |> Map.get(source, %{}) |> Map.get(reason, 0)
  end

  @doc "Whether the handler `id` is attached to `event`."
  def attached?(event, id) do
    event |> :telemetry.list_handlers() |> Enum.any?(&(&1.id == id))
  end
end
