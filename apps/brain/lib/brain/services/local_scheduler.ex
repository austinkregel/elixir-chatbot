defmodule Brain.Services.LocalScheduler do
  @moduledoc """
  Offline-friendly scheduling driver (`Brain.Services.SchedulingDriver`) for
  alarms, timers, reminders and calendar items. Provides enrichment data
  without requiring an external service like Home Assistant.

  Alarms, reminders, timers and calendar items belong to a world: each
  stored item is keyed by the world it was set in, and setting, cancelling
  or checking only ever touches that world's items. Every operation, and
  `enrich/4`, raises `ArgumentError` without a world; none is assumed.

  Items are stored, never rung: no alarm, timer, reminder or calendar item
  fires when it is due, so there is never a ringing alarm to snooze, and
  `snooze_alarm/3` says so.

  Items are stored in Atlas (`Atlas.Scheduler`), so they survive a restart.
  This module is also the supervised process that owns the ETS table the items
  are read from: it loads every stored item when it starts, and every set or
  cancel goes through it, writing Atlas first and the table after. Reads go
  straight to the table.
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
    [:alarm_status, :alarm_time, :reminder_set, :reminder_text,
     :reminder_time, :timer_status, :time_remaining, :duration,
     :next_event, :event_time]
  end

  # Setting, creating, scheduling and cancelling store or remove items;
  # checking and querying only read them, and a snooze finds no ringing alarm
  # and changes nothing.
  @writing_operations [:set_alarm, :set_timer, :create_reminder, :schedule_event, :cancel_event]

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
    add_item(world_id, :alarm, %{time: time, status: "set"})
    {:ok, %{alarm_status: "set", alarm_time: time}}
  end

  # Alarms are stored and never rung, so no alarm is ever ringing to snooze.
  @impl SchedulingDriver
  def snooze_alarm(_slots, _credentials, world_id) do
    _world_id = world!(world_id, :snooze_alarm)
    {:error, :no_ringing_alarm}
  end

  @impl SchedulingDriver
  def create_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :create_reminder)
    text = Map.get(slots, :reminder_text) || Map.get(slots, "text", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    add_item(world_id, :reminder, %{text: text, time: time, status: "set"})
    {:ok, %{reminder_set: true, reminder_text: text, reminder_time: time}}
  end

  @impl SchedulingDriver
  def set_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :set_timer)
    duration = Map.get(slots, :duration) || Map.get(slots, "duration", "")
    add_item(world_id, :timer, %{duration: duration, status: "running"})
    {:ok, %{timer_status: "running", duration: duration, time_remaining: duration}}
  end

  @impl SchedulingDriver
  def check_timer(_slots, _credentials, world_id) do
    world_id = world!(world_id, :check_timer)

    case list_items(world_id, :timer) do
      [] -> {:ok, %{timer_status: "none", time_remaining: "", duration: ""}}
      [latest | _] -> {:ok, %{timer_status: latest.status, time_remaining: latest.duration, duration: latest.duration}}
    end
  end

  @impl SchedulingDriver
  def schedule_event(slots, _credentials, world_id) do
    world_id = world!(world_id, :schedule_event)
    event = Map.get(slots, :event) || Map.get(slots, "event", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    add_item(world_id, :calendar, %{event: event, time: time})
    {:ok, %{next_event: event, event_time: time}}
  end

  # Takes every calendar item of the world off its calendar, as cancelling
  # does for every kind of item the scheduler stores.
  @impl SchedulingDriver
  def cancel_event(_slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_event)
    cancel_items(world_id, :calendar)
    {:ok, %{next_event: "", event_time: ""}}
  end

  @impl SchedulingDriver
  def query_calendar(_slots, _credentials, world_id) do
    world_id = world!(world_id, :query_calendar)

    case list_items(world_id, :calendar) do
      [] -> {:ok, %{next_event: "No upcoming events", event_time: ""}}
      [latest | _] -> {:ok, %{next_event: latest.event, event_time: latest.time}}
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

  # The world's items of one type, most recently created first. Every item is
  # stored with its `:created` time.
  defp list_items(world_id, type) do
    running!()

    @table
    |> :ets.match_object({{world_id, :_}, type, :_})
    |> Enum.map(fn {_key, _type, data} -> data end)
    |> Enum.sort_by(& &1.created, {:desc, DateTime})
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
  defp row_attrs(world_id, :calendar, %{event: event, time: time}),
    do: %{world_id: world_id, kind: @kinds.calendar, text: event, time: time}

  defp row_attrs(world_id, type, data),
    do: Map.merge(data, %{world_id: world_id, kind: Map.fetch!(@kinds, type)})

  defp entry(%Atlas.Schemas.SchedulerItem{} = item) do
    type = Map.fetch!(@types, item.kind)
    {{item.world_id, item.id}, type, data(type, item)}
  end

  defp data(:alarm, item), do: %{time: item.time, status: item.status, created: item.inserted_at}

  defp data(:reminder, item),
    do: %{text: item.text, time: item.time, status: item.status, created: item.inserted_at}

  defp data(:timer, item),
    do: %{duration: item.duration, status: item.status, created: item.inserted_at}

  defp data(:calendar, item), do: %{event: item.text, time: item.time, created: item.inserted_at}
end
