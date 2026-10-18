defmodule Brain.Services.SchedulerClock do
  @moduledoc """
  The clock `Brain.Services.LocalScheduler` fires items by: what time it is,
  and a wake-up at a given instant. The scheduler reads it from `config
  :brain, :scheduler_clock`, which names a module of this behaviour; it is
  required, and the scheduler does not start without it.

  `Brain.Services.SchedulerClock.SystemClock` is the system's clock. A test names
  a clock it sets itself, so firing happens at the instants the test chooses.
  """

  @doc "The current instant, in UTC."
  @callback now() :: DateTime.t()

  @doc """
  Sends `message` to `pid` at `at`, or as soon as it can when `at` has
  passed. Returns a reference `cancel/1` takes. A clock may wake `pid`
  earlier than `at`, never later; the scheduler reads `now/0` when it wakes.
  """
  @callback send_at(pid(), term(), DateTime.t()) :: reference()

  @doc "Cancels a wake-up `send_at/3` returned. Cancelling one that was sent or cancelled does nothing."
  @callback cancel(reference()) :: :ok

  @doc "The clock the configuration names. Raises when it names none."
  @spec configured!() :: module()
  def configured! do
    case Application.fetch_env(:brain, :scheduler_clock) do
      {:ok, clock} when is_atom(clock) and not is_nil(clock) ->
        Code.ensure_loaded!(clock)
        clock

      other ->
        raise ArgumentError,
              "config :brain, :scheduler_clock must name the Brain.Services.SchedulerClock the local scheduler " <>
                "fires items by, got: #{inspect(other)}"
    end
  end

  defmodule SystemClock do
    @moduledoc """
    The system's clock. A wake-up is a `Process.send_after/3` timer, which
    takes at most 2^32 - 1 milliseconds (about 49 days); one further off
    wakes the scheduler that long from now, and it sets the next wake-up then.
    """

    @behaviour Brain.Services.SchedulerClock

    @max_delay_ms 4_294_967_295

    @impl true
    def now, do: DateTime.utc_now()

    @impl true
    def send_at(pid, message, %DateTime{} = at) do
      delay = at |> DateTime.diff(now(), :millisecond) |> max(0) |> min(@max_delay_ms)
      Process.send_after(pid, message, delay)
    end

    @impl true
    def cancel(ref) when is_reference(ref) do
      Process.cancel_timer(ref)
      :ok
    end
  end
end
