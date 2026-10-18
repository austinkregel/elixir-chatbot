defmodule Brain.Test.ManualClock do
  @moduledoc """
  A `Brain.Services.SchedulerClock` that stands still: it reads the instant a
  test set (`set!/1`), and before any is set the one `config :brain,
  Brain.Test.ManualClock, :start` names. Its wake-ups never go off by
  themselves; a test moves the clock and wakes the scheduler
  (`Brain.Services.LocalScheduler.wake/0`), so an item fires at exactly the
  instant the test chooses and never while another test runs.

  It is the scheduler's clock in the test environment (config/test.exs).
  Lives under lib/, like `Brain.Test.AtlasSandbox`, so every umbrella app's
  tests can reach it.
  """

  @behaviour Brain.Services.SchedulerClock

  @key {__MODULE__, :now}

  @impl true
  def now do
    case :persistent_term.get(@key, :unset) do
      :unset -> start!()
      %DateTime{} = now -> now
    end
  end

  @doc "Sets the clock to `instant`."
  @spec set!(DateTime.t()) :: :ok
  def set!(%DateTime{} = instant), do: :persistent_term.put(@key, DateTime.shift_zone!(instant, "Etc/UTC"))

  @doc "Sets the clock back to the configured start."
  @spec reset!() :: :ok
  def reset! do
    :persistent_term.erase(@key)
    :ok
  end

  @impl true
  def send_at(pid, _message, %DateTime{}) when is_pid(pid), do: make_ref()

  @impl true
  def cancel(ref) when is_reference(ref), do: :ok

  defp start! do
    case Application.fetch_env(:brain, __MODULE__) do
      {:ok, config} ->
        case Keyword.fetch(config, :start) do
          {:ok, %DateTime{} = start} -> start
          other -> raise ArgumentError, "config :brain, Brain.Test.ManualClock, :start must be a DateTime, got: #{inspect(other)}"
        end

      :error ->
        raise ArgumentError, "config :brain, Brain.Test.ManualClock names no :start, the instant the clock reads before a test sets one"
    end
  end
end
