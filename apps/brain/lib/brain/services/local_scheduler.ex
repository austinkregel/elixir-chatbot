defmodule Brain.Services.LocalScheduler do
  @moduledoc """
  Offline-friendly service for alarm, reminder, and calendar management
  using ETS storage. Provides enrichment data without requiring an
  external service like Home Assistant.
  """

  @behaviour Brain.Services.Service

  require Logger

  @table :local_scheduler_items

  @impl true
  def name, do: :local_scheduler

  @impl true
  def display_name, do: "Local Scheduler"

  @impl true
  def description, do: "Offline alarm, reminder, and calendar management using local storage"

  @impl true
  def required_credentials, do: []

  @impl true
  def supported_intents do
    [
      "alarm.set",
      "alarm.cancel",
      "alarm.check",
      "reminder.set",
      "reminder.cancel",
      "reminder.check",
      "timer.set",
      "timer.cancel",
      "timer.check",
      "calendar.query",
      "calendar.create"
    ]
  end

  @impl true
  def provides_fields do
    [
      :alarm_status,
      :alarm_time,
      :reminder_set,
      :reminder_text,
      :reminder_time,
      :timer_status,
      :time_remaining,
      :duration,
      :next_event,
      :event_time
    ]
  end

  @impl true
  def health_check(_credentials) do
    ensure_table()
    {:ok, %{status: :ok, items: :ets.info(@table, :size)}}
  end

  @impl true
  def enrich(intent, slots, _credentials) do
    ensure_table()
    intent_str = to_string(intent)
    slots = Brain.Services.Service.normalize_slots(slots)

    cond do
      String.contains?(intent_str, "alarm.set") ->
        time = Map.get(slots, :time) || ""
        id = "alarm_#{System.unique_integer([:positive])}"

        :ets.insert(
          @table,
          {id, :alarm, %{time: time, status: "set", created: DateTime.utc_now()}}
        )

        {:ok, %{alarm_status: "set", alarm_time: time}}

      String.contains?(intent_str, "alarm.cancel") ->
        cancel_items(:alarm)
        {:ok, %{alarm_status: "cancelled"}}

      String.contains?(intent_str, "alarm.check") ->
        alarms = list_items(:alarm)

        case alarms do
          [] -> {:ok, %{alarm_status: "none set", alarm_time: ""}}
          [latest | _] -> {:ok, %{alarm_status: latest.status, alarm_time: latest.time}}
        end

      String.contains?(intent_str, "reminder.set") ->
        text = Map.get(slots, :reminder_text) || Map.get(slots, :text) || ""
        time = Map.get(slots, :time) || ""
        id = "reminder_#{System.unique_integer([:positive])}"

        :ets.insert(
          @table,
          {id, :reminder, %{text: text, time: time, status: "set", created: DateTime.utc_now()}}
        )

        {:ok, %{reminder_set: true, reminder_text: text, reminder_time: time}}

      String.contains?(intent_str, "reminder.cancel") ->
        cancel_items(:reminder)
        {:ok, %{reminder_set: false, reminder_text: "", reminder_time: ""}}

      String.contains?(intent_str, "reminder.check") ->
        reminders = list_items(:reminder)

        case reminders do
          [] ->
            {:ok, %{reminder_set: false, reminder_text: "", reminder_time: ""}}

          [latest | _] ->
            {:ok, %{reminder_set: true, reminder_text: latest.text, reminder_time: latest.time}}
        end

      String.contains?(intent_str, "timer.set") ->
        duration = Map.get(slots, :duration) || ""
        id = "timer_#{System.unique_integer([:positive])}"

        :ets.insert(
          @table,
          {id, :timer,
           %{duration: duration, status: "running", started: System.monotonic_time(:second)}}
        )

        {:ok, %{timer_status: "running", duration: duration, time_remaining: duration}}

      String.contains?(intent_str, "timer.cancel") ->
        cancel_items(:timer)
        {:ok, %{timer_status: "cancelled", time_remaining: ""}}

      String.contains?(intent_str, "timer.check") ->
        timers = list_items(:timer)

        case timers do
          [] ->
            {:ok, %{timer_status: "none", time_remaining: "", duration: ""}}

          [latest | _] ->
            {:ok,
             %{
               timer_status: latest.status,
               time_remaining: latest.duration,
               duration: latest.duration
             }}
        end

      String.contains?(intent_str, "calendar.create") ->
        event = Map.get(slots, :event) || ""
        time = Map.get(slots, :time) || ""
        id = "event_#{System.unique_integer([:positive])}"

        :ets.insert(
          @table,
          {id, :calendar, %{event: event, time: time, created: DateTime.utc_now()}}
        )

        {:ok, %{next_event: event, event_time: time}}

      String.contains?(intent_str, "calendar.query") ->
        events = list_items(:calendar)

        case events do
          [] -> {:ok, %{next_event: "No upcoming events", event_time: ""}}
          [latest | _] -> {:ok, %{next_event: latest.event, event_time: latest.time}}
        end

      true ->
        {:error, :unsupported_intent}
    end
  end

  @impl true
  def enabled? do
    Application.get_env(:brain, :local_scheduler_enabled, true)
  end

  defp ensure_table do
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :set, :public, read_concurrency: true])
    end
  rescue
    ArgumentError -> :ok
  end

  defp list_items(type) do
    ensure_table()

    @table
    |> :ets.tab2list()
    |> Enum.filter(fn {_id, item_type, _data} -> item_type == type end)
    |> Enum.map(fn {_id, _type, data} -> data end)
    |> Enum.sort_by(& &1[:created], {:desc, DateTime})
  rescue
    _ -> []
  end

  defp cancel_items(type) do
    ensure_table()

    @table
    |> :ets.tab2list()
    |> Enum.filter(fn {_id, item_type, _data} -> item_type == type end)
    |> Enum.each(fn {id, _, _} -> :ets.delete(@table, id) end)
  rescue
    _ -> :ok
  end
end
