defmodule Brain.Services.LocalScheduler do
  @moduledoc """
  The scheduling driver (`Brain.Services.SchedulingDriver`) for alarms,
  timers, reminders and calendar items, kept in this installation's own
  store and needing no credentials.

  Alarms, reminders, timers and calendar items belong to a world: each
  stored item is keyed by the world it was set in, and setting, cancelling
  or checking only ever touches that world's items. Every operation, and
  `enrich/4`, raises `ArgumentError` without a world; none is assumed.

  Items are stored, never rung: no alarm, timer, reminder or calendar item
  fires when it is due, so there is never a ringing alarm to snooze, and
  `snooze_alarm/3` says so.

  ## What is stored

  Each item keeps the slot values it was set with, as the text they arrived
  as: an alarm its `:time` and `:date`, a reminder its `:reminder_text`,
  `:time` and `:date`, a timer its `:duration`, and a calendar item its
  `:event`, `:time` and `:date`. A slot that was not given is stored as `""`.
  Nothing is parsed or resolved: a `:date` of `"tomorrow"` is stored as
  `"tomorrow"`, not as the day it named when it was set, and is matched later
  as that text. Items stored before dates were stored have no date (`nil`).

  ## Matching a slot to an item

  A slot that is absent, `nil` or blank names nothing. An item matches when
  every slot that names something equals the item's field, compared with case
  and runs of whitespace ignored (`"7 AM"` matches `"7 am"`); there is no
  partial or fuzzy match. An item whose field is `nil` (a date never stored)
  matches no value. A slot holding anything but a string raises.

  ## Checking

  `check_alarms/3` and `check_reminders/3` report the alarms or reminders the
  world has, in the order they were set, with how many there are.
  `query_calendar/3` reports the calendar item set last, with how many there
  are. Each of the three narrows its items to the day the `:date` slot names,
  matched as above; with no `:date`, it reports all of them. No other slot
  narrows a check. When a check finds nothing, its count is 0 and the fields of
  the item it would report are `""`: no text stands in for "nothing found".
  `check_timer/3` reports the timer set last, with the duration it was set
  with. A duration is stored as the text it arrived as, so how much of it is
  left is not known.

  ## Cancelling: which items a cancel matches

  A cancel identifies items by the fields stored for their kind, each read
  from one slot, matched as above:

  | Kind     | Field      | Slot             |
  | -------- | ---------- | ---------------- |
  | alarm    | time       | `:time`          |
  | alarm    | date       | `:date`          |
  | reminder | text       | `:reminder_text` |
  | reminder | time       | `:time`          |
  | reminder | date       | `:date`          |
  | timer    | duration   | `:duration`      |
  | calendar | event      | `:event`         |
  | calendar | time       | `:time`          |
  | calendar | date       | `:date`          |

  So "delete my reminders for tomorrow" (`:date` `"tomorrow"`) removes the
  reminders stored with the date `"tomorrow"`, and leaves those for another
  day or with no date.

  - `cancel_alarm/3`, `cancel_reminder/3` and `cancel_timer/3` remove every
    matching item of their kind in the world. Slots that identify one item
    remove that one; no identifying slot matches every item of the kind, so
    "cancel my alarms" removes them all.
  - `cancel_event/3` removes exactly one calendar item and never more. With
    no identifying slot it is `{:error, :event_not_identified}`; when several
    items match it is `{:error, {:ambiguous, :calendar, events}}`, naming
    each, and nothing is removed. Either is the response layer's cue to ask
    which event is meant.

  When nothing matches, every cancel is
  `{:error, {:no_matching_items, kind, criteria}}`, with the slot values it
  matched on, and nothing is removed.

  Items are stored in Atlas (`Atlas.Scheduler`), so they survive a restart.
  This module is also the supervised process that owns the ETS table the items
  are read from: it loads every stored item when it starts, and every set or
  cancel goes through it, writing Atlas first and the table after. A cancel
  finds the items it removes inside that same call, so no other write can come
  between matching and removing. Reads go straight to the table.
  """

  @behaviour Brain.Services.Service
  @behaviour Brain.Services.SchedulingDriver

  use GenServer
  require Logger

  alias Brain.Services.SchedulingDriver

  @table :local_scheduler_items

  # The kind stored in Atlas for each item type, and back.
  @kinds %{alarm: "alarm", reminder: "reminder", timer: "timer", calendar: "calendar"}
  @types Map.new(@kinds, fn {type, kind} -> {kind, type} end)

  # The fields a cancel identifies an item of each type by, each with the slot
  # that names it (see "Cancelling" above).
  @identifying_slots %{
    alarm: [time: :time, date: :date],
    reminder: [text: :reminder_text, time: :time, date: :date],
    timer: [duration: :duration],
    calendar: [event: :event, time: :time, date: :date]
  }

  # The field a check narrows the items of each type to, with the slot that
  # names it (see "Checking" above).
  @day_slots %{
    alarm: [date: :date],
    reminder: [date: :date],
    calendar: [date: :date]
  }

  @doc "Starts the scheduler, which loads every stored item into its table."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def name, do: :local_scheduler

  @impl true
  def display_name, do: "Local Scheduler"

  @impl true
  def description, do: "Offline alarm, reminder, and calendar management using local storage"

  @impl true
  def required_credentials, do: []

  # The intents of every scheduling operation (`SchedulingDriver.intents/0`).
  # The dispatcher routes them by the world's chosen driver, not by this list.
  @impl true
  def supported_intents, do: SchedulingDriver.intents()

  @impl true
  def provides_fields do
    [:alarm_status, :alarm_time, :alarm_date, :alarm_times, :alarm_dates, :alarm_count, :reminder_set,
     :reminder_text, :reminder_time, :reminder_date, :reminder_texts, :reminder_times, :reminder_dates,
     :reminder_count, :timer_status, :duration, :timer_durations, :next_event, :event_time,
     :event_date, :event_count, :cancelled_event, :cancelled_count]
  end

  # Setting, creating, scheduling and cancelling store or remove items;
  # checking and querying only read them, and a snooze finds no ringing alarm
  # and changes nothing.
  @writing_operations [
    :set_alarm,
    :cancel_alarm,
    :set_timer,
    :cancel_timer,
    :create_reminder,
    :cancel_reminder,
    :schedule_event,
    :cancel_event
  ]

  @impl true
  def writes?(intent) do
    case SchedulingDriver.operation(to_string(intent)) do
      {:ok, operation} -> operation in @writing_operations
      :error -> false
    end
  end

  @impl true
  def health_check(_credentials) do
    case :ets.whereis(@table) do
      :undefined -> {:error, :not_running}
      _table -> {:ok, %{status: :ok, items: :ets.info(@table, :size)}}
    end
  end

  # The operation is read from the intent's domain and verb
  # (`SchedulingDriver.operation/1`); an intent that names no operation is
  # refused.
  @impl true
  def enrich(intent, slots, credentials, world_id) do
    world_id = world!(world_id, :enrich)

    case SchedulingDriver.operation(to_string(intent)) do
      {:ok, operation} -> apply(__MODULE__, operation, [slots, credentials, world_id])
      :error -> {:error, :unsupported_intent}
    end
  end

  @impl SchedulingDriver
  def set_alarm(slots, _credentials, world_id) do
    world_id = world!(world_id, :set_alarm)
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :alarm, %{time: time, date: date, status: "set"})
    {:ok, %{alarm_status: "set", alarm_time: time, alarm_date: date}}
  end

  # Alarms are stored and never rung, so no alarm is ever ringing to snooze.
  @impl SchedulingDriver
  def snooze_alarm(_slots, _credentials, world_id) do
    _world_id = world!(world_id, :snooze_alarm)
    {:error, :no_ringing_alarm}
  end

  @impl SchedulingDriver
  def cancel_alarm(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_alarm)

    with {:ok, cancelled} <- cancel_matching(world_id, :alarm, slots, :all) do
      {:ok,
       %{
         alarm_status: "cancelled",
         alarm_times: Enum.map(cancelled, & &1.time),
         alarm_dates: Enum.map(cancelled, & &1.date),
         cancelled_count: length(cancelled)
       }}
    end
  end

  @impl SchedulingDriver
  def check_alarms(slots, _credentials, world_id) do
    world_id = world!(world_id, :check_alarms)
    alarms = world_id |> list_items(:alarm) |> Enum.reverse() |> on_day(:alarm, slots)

    {:ok,
     %{
       alarm_status: if(alarms == [], do: "none", else: "set"),
       alarm_times: Enum.map(alarms, & &1.time),
       alarm_dates: Enum.map(alarms, & &1.date),
       alarm_count: length(alarms)
     }}
  end

  @impl SchedulingDriver
  def create_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :create_reminder)
    text = Map.get(slots, :reminder_text) || Map.get(slots, "reminder_text", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :reminder, %{text: text, time: time, date: date, status: "set"})
    {:ok, %{reminder_set: true, reminder_text: text, reminder_time: time, reminder_date: date}}
  end

  @impl SchedulingDriver
  def cancel_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_reminder)

    with {:ok, cancelled} <- cancel_matching(world_id, :reminder, slots, :all) do
      {:ok,
       %{
         reminder_set: false,
         reminder_texts: Enum.map(cancelled, & &1.text),
         reminder_times: Enum.map(cancelled, & &1.time),
         reminder_dates: Enum.map(cancelled, & &1.date),
         cancelled_count: length(cancelled)
       }}
    end
  end

  @impl SchedulingDriver
  def check_reminders(slots, _credentials, world_id) do
    world_id = world!(world_id, :check_reminders)
    reminders = world_id |> list_items(:reminder) |> Enum.reverse() |> on_day(:reminder, slots)

    {:ok,
     %{
       reminder_set: reminders != [],
       reminder_texts: Enum.map(reminders, & &1.text),
       reminder_times: Enum.map(reminders, & &1.time),
       reminder_dates: Enum.map(reminders, & &1.date),
       reminder_count: length(reminders)
     }}
  end

  @impl SchedulingDriver
  def set_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :set_timer)
    duration = Map.get(slots, :duration) || Map.get(slots, "duration", "")
    add_item(world_id, :timer, %{duration: duration, status: "running"})
    {:ok, %{timer_status: "running", duration: duration}}
  end

  @impl SchedulingDriver
  def check_timer(_slots, _credentials, world_id) do
    world_id = world!(world_id, :check_timer)

    case list_items(world_id, :timer) do
      [] -> {:ok, %{timer_status: "none", duration: ""}}
      [latest | _] -> {:ok, %{timer_status: latest.status, duration: latest.duration}}
    end
  end

  @impl SchedulingDriver
  def cancel_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_timer)

    with {:ok, cancelled} <- cancel_matching(world_id, :timer, slots, :all) do
      {:ok,
       %{
         timer_status: "cancelled",
         timer_durations: Enum.map(cancelled, & &1.duration),
         cancelled_count: length(cancelled)
       }}
    end
  end

  @impl SchedulingDriver
  def schedule_event(slots, _credentials, world_id) do
    world_id = world!(world_id, :schedule_event)
    event = Map.get(slots, :event) || Map.get(slots, "event", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :calendar, %{event: event, time: time, date: date})
    {:ok, %{next_event: event, event_time: time, event_date: date}}
  end

  # Takes the one calendar item the slots identify off the world's calendar,
  # and never more than one (see "Cancelling" above).
  @impl SchedulingDriver
  def cancel_event(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_event)

    with {:ok, [cancelled]} <- cancel_matching(world_id, :calendar, slots, :one) do
      {:ok,
       %{
         cancelled_event: cancelled.event,
         event_time: cancelled.time,
         event_date: cancelled.date,
         cancelled_count: 1
       }}
    end
  end

  @impl SchedulingDriver
  def query_calendar(slots, _credentials, world_id) do
    world_id = world!(world_id, :query_calendar)

    case world_id |> list_items(:calendar) |> on_day(:calendar, slots) do
      [] ->
        {:ok, %{next_event: "", event_time: "", event_date: "", event_count: 0}}

      [latest | _] = events ->
        {:ok, %{next_event: latest.event, event_time: latest.time, event_date: latest.date, event_count: length(events)}}
    end
  end

  @impl true
  def enabled? do
    Application.get_env(:brain, :local_scheduler_enabled, true)
  end

  @doc "Deletes every alarm, reminder, timer and calendar item of a world, for a world that is destroyed."
  @spec clear_world(String.t()) :: :ok
  def clear_world(world_id) do
    world_id = world!(world_id, :clear_world)
    Enum.each(Map.keys(@kinds), &cancel_items(world_id, &1))
  end

  # Server callbacks

  @impl true
  def init(_opts) do
    table = :ets.new(@table, [:named_table, :set, :protected, read_concurrency: true])

    items = Atlas.Scheduler.list_items()
    Enum.each(items, &:ets.insert(table, entry(&1)))

    Logger.info("Local scheduler loaded #{length(items)} stored items")
    {:ok, %{table: table}}
  end

  @impl true
  def handle_call({:add, world_id, type, data}, _from, state) do
    item = Atlas.Scheduler.insert_item!(row_attrs(world_id, type, data))
    :ets.insert(state.table, entry(item))
    {:reply, :ok, state}
  end

  @impl true
  def handle_call({:cancel, world_id, type}, _from, state) do
    Atlas.Scheduler.delete_items!(world_id, Map.fetch!(@kinds, type))
    :ets.match_delete(state.table, {{world_id, :_}, type, :_})
    {:reply, :ok, state}
  end

  # Matching and removing happen in this one call, so no other write comes
  # between them. `:all` removes every match; `:one` removes a match only
  # when it is the only one.
  def handle_call({:cancel_matching, world_id, type, criteria, mode}, _from, state) do
    matches =
      world_id
      |> entries(type)
      |> Enum.filter(fn {_id, data} -> matches?(data, criteria) end)
      |> Enum.sort_by(fn {_id, data} -> data.created end, {:asc, DateTime})

    reply =
      case {mode, matches} do
        {_mode, []} ->
          {:error, {:no_matching_items, type, criteria}}

        {:one, [_first, _second | _rest]} ->
          {:error, {:ambiguous, type, Enum.map(matches, fn {_id, data} -> Map.delete(data, :created) end)}}

        {_mode, matches} ->
          ids = Enum.map(matches, &elem(&1, 0))
          Atlas.Scheduler.delete_items_by_id!(world_id, Map.fetch!(@kinds, type), ids)
          Enum.each(ids, &:ets.delete(state.table, {world_id, &1}))
          {:ok, Enum.map(matches, &elem(&1, 1))}
      end

    {:reply, reply, state}
  end

  # Every stored item belongs to the world it was set in; there is no default.
  defp world!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world!(other, function) do
    raise ArgumentError,
          "LocalScheduler.#{function} needs the world whose alarms, reminders, timers and calendar items are used, got: #{inspect(other)}"
  end

  defp add_item(world_id, type, data) do
    :ok = GenServer.call(running!(), {:add, world_id, type, data})
  end

  defp cancel_items(world_id, type) do
    :ok = GenServer.call(running!(), {:cancel, world_id, type})
  end

  # Removes the world's items of `type` that the slots identify, in `mode`
  # (`handle_call/3`). `:one` with no identifying slot is refused before the
  # scheduler is asked, since it could only match every item.
  defp cancel_matching(world_id, type, slots, mode) do
    case {mode, criteria!(Map.fetch!(@identifying_slots, type), type, slots)} do
      {:one, criteria} when criteria == %{} -> {:error, :event_not_identified}
      {mode, criteria} -> GenServer.call(running!(), {:cancel_matching, world_id, type, criteria, mode})
    end
  end

  # The items, of `type`, that are for the day the slots name; all of them
  # when no slot names a day (see "Checking" above).
  defp on_day(items, type, slots) do
    criteria = criteria!(Map.fetch!(@day_slots, type), type, slots)
    Enum.filter(items, &matches?(&1, criteria))
  end

  # The values the slots name for `fields`, each a field of an item of `type`
  # with the slot that names it, compared as `normalize/1` leaves them.
  defp criteria!(fields, type, slots) do
    Enum.reduce(fields, %{}, fn {field, slot}, acc ->
      case Map.get(slots, slot) do
        nil ->
          acc

        value when is_binary(value) ->
          case normalize(value) do
            "" -> acc
            normalized -> Map.put(acc, field, normalized)
          end

        other ->
          raise ArgumentError,
                "LocalScheduler: the #{inspect(slot)} slot identifies #{type} items by their #{field} and " <>
                  "must be a string, got: #{inspect(other)}"
      end
    end)
  end

  # An item stored without the field (`nil`) matches no value given for it.
  defp matches?(data, criteria) do
    Enum.all?(criteria, fn {field, value} ->
      case Map.fetch!(data, field) do
        nil -> false
        stored -> normalize(stored) == value
      end
    end)
  end

  # Case and runs of whitespace are ignored when an item is matched.
  defp normalize(text), do: text |> String.split() |> Enum.join(" ") |> String.downcase()

  # The world's items of one type, most recently created first. Every item is
  # stored with its `:created` time.
  defp list_items(world_id, type) do
    running!()

    world_id
    |> entries(type)
    |> Enum.map(fn {_id, data} -> data end)
    |> Enum.sort_by(& &1.created, {:desc, DateTime})
  end

  # Each of the world's items of one type, with its stored id.
  defp entries(world_id, type) do
    @table
    |> :ets.match_object({{world_id, :_}, type, :_})
    |> Enum.map(fn {{_world_id, id}, _type, data} -> {id, data} end)
  end

  defp running! do
    case Process.whereis(__MODULE__) do
      nil ->
        raise "Brain.Services.LocalScheduler is not running, so alarms, reminders, timers and " <>
                "calendar items can be neither read nor stored"

      pid ->
        pid
    end
  end

  # A calendar item's event is stored as the item's text.
  defp row_attrs(world_id, :calendar, %{event: event, time: time, date: date}),
    do: %{world_id: world_id, kind: @kinds.calendar, text: event, time: time, date: date}

  defp row_attrs(world_id, type, data),
    do: Map.merge(data, %{world_id: world_id, kind: Map.fetch!(@kinds, type)})

  defp entry(%Atlas.Schemas.SchedulerItem{} = item) do
    type = Map.fetch!(@types, item.kind)
    {{item.world_id, item.id}, type, data(type, item)}
  end

  defp data(:alarm, item),
    do: %{time: item.time, date: item.date, status: item.status, created: item.inserted_at}

  defp data(:reminder, item),
    do: %{text: item.text, time: item.time, date: item.date, status: item.status, created: item.inserted_at}

  defp data(:timer, item),
    do: %{duration: item.duration, status: item.status, created: item.inserted_at}

  defp data(:calendar, item),
    do: %{event: item.text, time: item.time, date: item.date, created: item.inserted_at}
end
