defmodule Brain.Services.SchedulingDriver do
  @moduledoc """
  Behaviour for the driver that serves a world's alarms, timers, reminders and
  calendar items: the `"scheduling"` capability.

  Each world chooses its driver (`Atlas.WorldServiceDrivers`), and
  `Brain.Services.Dispatcher` routes every intent of the alarm, timer,
  reminder and calendar domains to that world's driver. A world with no driver
  chosen raises; none is assumed.

  A driver is also a `Brain.Services.Service`: its `name/0` is the driver name
  stored for a world, its `required_credentials/0` must all be set in a world
  before it can be chosen there, and its `writes?/1` decides whether a call is
  refused with side effects off.

  ## Operations

  The operations are the intents of these domains in
  `priv/analysis/intent_registry.json`. An intent names its operation by its
  domain and its verb, the segment after the domain; `operation/1` reads both
  and looks the pair up, so no other intent maps to an operation.

  | Intent              | Operation            |
  | ------------------- | -------------------- |
  | `alarm.set`         | `set_alarm/3`        |
  | `alarm.snooze`      | `snooze_alarm/3`     |
  | `alarm.cancel`      | `cancel_alarm/3`     |
  | `alarm.check`       | `check_alarms/3`     |
  | `timer.set`         | `set_timer/3`        |
  | `timer.check`       | `check_timer/3`      |
  | `timer.cancel`      | `cancel_timer/3`     |
  | `reminder.create`   | `create_reminder/3`  |
  | `reminder.cancel`   | `cancel_reminder/3`  |
  | `reminder.check`    | `check_reminders/3`  |
  | `calendar.query`    | `query_calendar/3`   |
  | `calendar.schedule` | `schedule_event/3`   |
  | `calendar.cancel`   | `cancel_event/3`     |

  Every operation takes the slots, the driver's credentials read from the
  world, and the world, and returns what `Brain.Services.Service.enrich/4`
  returns.

  `Brain.Services.LocalScheduler` is the only driver. A driver is the point
  where another scheduling backend is added: a module that implements this
  behaviour and is listed in `Brain.Services.Dispatcher`'s drivers.
  """

  alias Brain.Services.Service

  @typedoc "What an operation returns: enrichment data, or why there is none."
  @type result :: {:ok, Service.enrichment_data()} | {:error, Service.error_reason()}

  @typedoc "One of the operations a scheduling driver performs."
  @type operation ::
          :set_alarm
          | :snooze_alarm
          | :cancel_alarm
          | :check_alarms
          | :set_timer
          | :check_timer
          | :cancel_timer
          | :create_reminder
          | :cancel_reminder
          | :check_reminders
          | :query_calendar
          | :schedule_event
          | :cancel_event

  @doc "Sets an alarm."
  @callback set_alarm(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Snoozes the alarm that is ringing."
  @callback snooze_alarm(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Cancels the alarms the slots identify."
  @callback cancel_alarm(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Reports the alarms that are set: those for the day the slots name, or every one when they name none."
  @callback check_alarms(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Starts a timer."
  @callback set_timer(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Reports the timer that is running."
  @callback check_timer(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Cancels the timers the slots identify."
  @callback cancel_timer(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Creates a reminder."
  @callback create_reminder(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Cancels the reminders the slots identify."
  @callback cancel_reminder(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Reports the reminders that are set: those for the day the slots name, or every one when they name none."
  @callback check_reminders(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Reports the calendar's events: those for the day the slots name, or all of them when they name none."
  @callback query_calendar(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Puts an event on the calendar."
  @callback schedule_event(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @doc "Takes the event the slots identify off the calendar."
  @callback cancel_event(Service.slots(), Service.credentials(), world_id :: String.t()) :: result()

  @capability "scheduling"

  @operations %{
    {"alarm", "set"} => :set_alarm,
    {"alarm", "snooze"} => :snooze_alarm,
    {"alarm", "cancel"} => :cancel_alarm,
    {"alarm", "check"} => :check_alarms,
    {"timer", "set"} => :set_timer,
    {"timer", "check"} => :check_timer,
    {"timer", "cancel"} => :cancel_timer,
    {"reminder", "create"} => :create_reminder,
    {"reminder", "cancel"} => :cancel_reminder,
    {"reminder", "check"} => :check_reminders,
    {"calendar", "query"} => :query_calendar,
    {"calendar", "schedule"} => :schedule_event,
    {"calendar", "cancel"} => :cancel_event
  }

  @domains @operations |> Map.keys() |> Enum.map(&elem(&1, 0)) |> Enum.uniq() |> Enum.sort()

  @doc "The capability a scheduling driver serves, as stored for a world."
  @spec capability() :: String.t()
  def capability, do: @capability

  @doc "The intent domains routed to a world's scheduling driver."
  @spec domains() :: [String.t()]
  def domains, do: @domains

  @doc "Every intent a scheduling driver performs, sorted."
  @spec intents() :: [String.t()]
  def intents do
    @operations |> Map.keys() |> Enum.map(fn {domain, verb} -> "#{domain}.#{verb}" end) |> Enum.sort()
  end

  @doc "Whether the intent's domain, its first segment, is routed to a scheduling driver."
  @spec domain?(String.t()) :: boolean()
  def domain?(intent) when is_binary(intent) do
    case String.split(intent, ".", parts: 2) do
      [domain, _rest] -> domain in @domains
      [_undotted] -> false
    end
  end

  @doc """
  The operation an intent names: exactly a domain and a verb, as in
  `alarm.set`. Any other intent, such as `alarm.set.recurring` or `alarm.ring`,
  names no operation and is `:error`.
  """
  @spec operation(String.t()) :: {:ok, operation()} | :error
  def operation(intent) when is_binary(intent) do
    case String.split(intent, ".") do
      [domain, verb] -> Map.fetch(@operations, {domain, verb})
      _other -> :error
    end
  end

  @doc "The intent that names `operation`. Raises for anything that is not an operation."
  @spec intent!(operation()) :: String.t()
  def intent!(operation) when is_atom(operation) do
    case Enum.find(@operations, fn {_intent, op} -> op == operation end) do
      {{domain, verb}, ^operation} ->
        "#{domain}.#{verb}"

      nil ->
        raise ArgumentError,
              "SchedulingDriver: #{inspect(operation)} is not an operation; the operations are " <>
                inspect(@operations |> Map.values() |> Enum.sort())
    end
  end
end
