defmodule Brain.Services.LocalScheduler do
  @moduledoc """
  The scheduling driver (`Brain.Services.SchedulingDriver`) for alarms,
  timers, reminders and calendar items, kept in this installation's own
  store and needing no credentials.

  Alarms, reminders, timers and calendar items belong to a world: each
  stored item is keyed by the world it was set in, and setting, cancelling
  or checking only ever touches that world's items. Every operation, and
  `enrich/4`, raises `ArgumentError` without a world; none is assumed.

  Every operation also needs the moment of the turn it was asked in, as the
  `:turn_time` slot (a `DateTime`; `Brain.Services.Dispatcher` puts it there
  from the turn), and raises without one: what "tomorrow" or "5 pm" names
  depends on when it is said.

  Items are stored, never rung: no alarm, timer, reminder or calendar item
  fires when it is due, so there is never a ringing alarm to snooze, and
  `snooze_alarm/3` says so.

  ## What is stored

  Each item keeps the slot values it was set with, as the text they arrived
  as: an alarm its `:time` and `:date`, a reminder its `:reminder_text`,
  `:time` and `:date`, a timer its `:duration`, and a calendar item its
  `:event`, `:time` and `:date`. A slot that was not given is stored as `""`.
  Items stored before dates were stored have no date (`nil`).

  Beside that text, each item keeps what its date and time, or a timer's
  duration, name, resolved against the turn it was set in
  (`Brain.Temporal.Resolver`): when that time starts and ends, and the time
  of day said. A `:date` of `"tomorrow"` set on a Monday is Tuesday, and stays
  Tuesday. The date and time are resolved together ("tomorrow", "7 am" is 7
  am on Tuesday); a time of day alone is its soonest occurrence. A timer
  starts at the turn it was set in and runs out when its duration has passed.
  An item set with no date, time or duration has nothing resolved
  (`"nothing_said"`), and an item stored before resolutions were has none
  (`"recorded_before_resolution"`: the turn it was set in was not kept).
  What cannot be resolved -- a word no source holds, a day that does not
  exist, a length given as an alarm's time, a timer's duration that names no
  length -- raises, naming the expression, and nothing is stored.

  ## Matching a slot to an item

  A slot that is absent, `nil` or blank names nothing. An item matches when
  every slot that names something matches the item's field:

  - a reminder's text and a calendar item's event: equal, with case and runs
    of whitespace ignored (`"Water Plants"` matches `"water plants"`); there
    is no partial or fuzzy match;
  - a `:date`: resolved against the turn it is said in, it names days; an
    item matches when its resolved time covers one of them, so "tomorrow"
    said on Monday matches what "tomorrow" named when set on Monday, and
    "Tuesday" said on Monday matches it too, but "tomorrow" said on Tuesday
    does not;
  - a `:time`: resolved the same way, an item matches when the time of day
    said is the item's; a day said with it ("tomorrow at 7 am") must be one
    the item covers too. A time that names no time of day ("tomorrow")
    matches as a date does;
  - a `:duration`: an item matches when the timer runs as long as the
    duration names.

  An item with nothing resolved, or stored before resolutions were, matches no
  date, time or duration. A slot holding anything but a string raises, and so
  does one that cannot be resolved.

  ## Checking

  `check_alarms/3` and `check_reminders/3` report the alarms or reminders the
  world has, in the order they were set, with how many there are.
  `query_calendar/3` reports the soonest calendar item still to come
  (`:next_event`): of the items found, those whose time has not passed at the
  turn, ordered by when they start, the first set first among those starting
  together; an item with nothing resolved, or stored before resolutions were,
  has no time to order by and is never the next. It reports how many items it
  found (`:event_count`) and how many of them are still to come
  (`:upcoming_count`). Each of the three narrows its items to the days the
  `:date` slot names, matched as above; with no `:date`, it reports all of
  them. No other slot narrows a check. When a check finds nothing, its count is
  0 and the fields of the item it would report are `""`: no text stands in for
  "nothing found".

  `check_timer/3` reports the timer set last, with the duration it was set
  with and how much of it is left at the turn (`:time_left`, in days, hours,
  minutes and seconds, largest first). A timer whose duration has passed is
  `"finished"`, with nothing left. A timer set with no duration, or stored
  before resolutions were, has no end that is known: it is `"running"`, and
  `:time_left` is empty.

  ## Cancelling: which items a cancel matches

  A cancel identifies items by the fields stored for their kind, each read
  from one slot, matched as above (`identifying_slots/0`):

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
  reminders whose resolved time falls on the day after the turn, and leaves
  those for another day or with none resolved.

  - `cancel_alarm/3`, `cancel_reminder/3` and `cancel_timer/3` remove every
    matching item of their kind in the world. Slots that identify one item
    remove that one; no identifying slot matches every item of the kind, so
    "cancel my alarms" removes them all.
  - `cancel_event/3` removes exactly one calendar item and never more. With
    no identifying slot it is `{:error, {:event_not_identified, events}}`,
    naming every calendar item of the world; when several items match it is
    `{:error, {:ambiguous, :calendar, events}}`, naming each. Nothing is
    removed, and either is the response layer's cue to ask which of the
    events named is meant. With no identifying slot and no calendar items it
    is `{:error, {:no_matching_items, :calendar, %{}}}`.

  When nothing matches, every cancel is
  `{:error, {:no_matching_items, kind, criteria}}`, with the slot values it
  matched on, as they were said with case and runs of whitespace dropped, and
  nothing is removed. The items an error names carry their fields as they
  were said.

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
  alias Brain.Temporal.{Resolution, Resolver, Vocabulary}

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

  # How each identifying field is matched (see "Matching a slot to an item").
  @field_matches %{text: :text, event: :text, date: :date, time: :time, duration: :duration}

  # What an item keeps beside its fields, which no reply or error names.
  @internal_keys [:created, :resolution]

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
     :reminder_count, :timer_status, :duration, :time_left, :timer_durations, :last_set_event, :next_event,
     :event_time, :event_date, :event_count, :upcoming_count, :cancelled_event, :cancelled_count]
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
    turn = turn_time!(slots, :set_alarm)
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :alarm, %{time: time, date: date, status: "set"}, moment!([date, time], turn, :alarm))
    {:ok, %{alarm_status: "set", alarm_time: time, alarm_date: date}}
  end

  # Alarms are stored and never rung, so no alarm is ever ringing to snooze.
  @impl SchedulingDriver
  def snooze_alarm(slots, _credentials, world_id) do
    _world_id = world!(world_id, :snooze_alarm)
    _turn = turn_time!(slots, :snooze_alarm)
    {:error, :no_ringing_alarm}
  end

  @impl SchedulingDriver
  def cancel_alarm(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_alarm)
    turn = turn_time!(slots, :cancel_alarm)

    with {:ok, cancelled} <- cancel_matching(world_id, :alarm, slots, turn, :all) do
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
    turn = turn_time!(slots, :check_alarms)
    alarms = world_id |> list_items(:alarm) |> Enum.reverse() |> on_day(:alarm, slots, turn)

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
    turn = turn_time!(slots, :create_reminder)
    text = Map.get(slots, :reminder_text) || Map.get(slots, "reminder_text", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :reminder, %{text: text, time: time, date: date, status: "set"}, moment!([date, time], turn, :reminder))
    {:ok, %{reminder_set: true, reminder_text: text, reminder_time: time, reminder_date: date}}
  end

  @impl SchedulingDriver
  def cancel_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_reminder)
    turn = turn_time!(slots, :cancel_reminder)

    with {:ok, cancelled} <- cancel_matching(world_id, :reminder, slots, turn, :all) do
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
    turn = turn_time!(slots, :check_reminders)
    reminders = world_id |> list_items(:reminder) |> Enum.reverse() |> on_day(:reminder, slots, turn)

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
    turn = turn_time!(slots, :set_timer)
    duration = Map.get(slots, :duration) || Map.get(slots, "duration", "")
    add_item(world_id, :timer, %{duration: duration, status: "running"}, length!(duration, turn))
    {:ok, %{timer_status: "running", duration: duration}}
  end

  # The timer set last, with how much of it is left at the turn.
  @impl SchedulingDriver
  def check_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :check_timer)
    turn = turn_time!(slots, :check_timer)

    case list_items(world_id, :timer) do
      [] ->
        {:ok, %{timer_status: "none", duration: "", time_left: []}}

      [%{resolution: %Resolution{ends_at: %DateTime{} = ends_at}} = latest | _] ->
        case DateTime.diff(ends_at, turn, :microsecond) do
          left when left > 0 ->
            seconds = div(left + 999_999, 1_000_000)
            {:ok, %{timer_status: latest.status, duration: latest.duration, time_left: Resolver.say_seconds(seconds, Vocabulary.loaded!())}}

          _over ->
            {:ok, %{timer_status: "finished", duration: latest.duration, time_left: []}}
        end

      # Set with no duration, or stored before resolutions were: no end is known.
      [latest | _] ->
        {:ok, %{timer_status: latest.status, duration: latest.duration, time_left: []}}
    end
  end

  @impl SchedulingDriver
  def cancel_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_timer)
    turn = turn_time!(slots, :cancel_timer)

    with {:ok, cancelled} <- cancel_matching(world_id, :timer, slots, turn, :all) do
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
    turn = turn_time!(slots, :schedule_event)
    event = Map.get(slots, :event) || Map.get(slots, "event", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :calendar, %{event: event, time: time, date: date}, moment!([date, time], turn, :calendar))
    {:ok, %{last_set_event: event, event_time: time, event_date: date}}
  end

  # Takes the one calendar item the slots identify off the world's calendar,
  # and never more than one (see "Cancelling" above).
  @impl SchedulingDriver
  def cancel_event(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_event)
    turn = turn_time!(slots, :cancel_event)

    with {:ok, [cancelled]} <- cancel_matching(world_id, :calendar, slots, turn, :one) do
      {:ok,
       %{
         cancelled_event: cancelled.event,
         event_time: cancelled.time,
         event_date: cancelled.date,
         cancelled_count: 1
       }}
    end
  end

  # The soonest item still to come of those found (see "Checking" above).
  @impl SchedulingDriver
  def query_calendar(slots, _credentials, world_id) do
    world_id = world!(world_id, :query_calendar)
    turn = turn_time!(slots, :query_calendar)
    events = world_id |> list_items(:calendar) |> Enum.reverse() |> on_day(:calendar, slots, turn)

    upcoming =
      events
      |> Enum.filter(&still_to_come?(&1, turn))
      |> Enum.sort_by(& &1.resolution.starts_at, DateTime)

    case upcoming do
      [] ->
        {:ok, %{next_event: "", event_time: "", event_date: "", event_count: length(events), upcoming_count: 0}}

      [next | _] ->
        {:ok,
         %{
           next_event: next.event,
           event_time: next.time,
           event_date: next.date,
           event_count: length(events),
           upcoming_count: length(upcoming)
         }}
    end
  end

  # The fields a cancel identifies an item of each kind by, each with the slot
  # that names it (see "Cancelling" above).
  @impl SchedulingDriver
  def identifying_slots, do: @identifying_slots

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
  def handle_call({:add, world_id, type, data, resolution}, _from, state) do
    item = Atlas.Scheduler.insert_item!(row_attrs(world_id, type, data, resolution))
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
  # when it is the only one. `said` is what the slots said, which an error
  # names.
  def handle_call({:cancel_matching, world_id, type, criteria, said, mode}, _from, state) do
    matches =
      world_id
      |> entries(type)
      |> Enum.filter(fn {_id, data} -> matches?(data, criteria) end)
      |> Enum.sort_by(fn {_id, data} -> data.created end, {:asc, DateTime})

    reply =
      case {mode, matches} do
        {_mode, []} ->
          {:error, {:no_matching_items, type, said}}

        {:one, [_first, _second | _rest]} ->
          {:error, {:ambiguous, type, Enum.map(matches, fn {_id, data} -> public(data) end)}}

        {_mode, matches} ->
          ids = Enum.map(matches, &elem(&1, 0))
          Atlas.Scheduler.delete_items_by_id!(world_id, Map.fetch!(@kinds, type), ids)
          Enum.each(ids, &:ets.delete(state.table, {world_id, &1}))
          {:ok, Enum.map(matches, &public(elem(&1, 1)))}
      end

    {:reply, reply, state}
  end

  # Every stored item belongs to the world it was set in; there is no default.
  defp world!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world!(other, function) do
    raise ArgumentError,
          "LocalScheduler.#{function} needs the world whose alarms, reminders, timers and calendar items are used, got: #{inspect(other)}"
  end

  # What a date or time names depends on when it is said; there is no default.
  defp turn_time!(slots, function) do
    case Map.get(slots, :turn_time) do
      %DateTime{} = turn ->
        turn

      other ->
        raise ArgumentError,
              "LocalScheduler.#{function} needs the :turn_time slot, the moment of the turn it is asked in, " <>
                "got: #{inspect(other)}"
    end
  end

  # What an alarm's, reminder's or calendar item's date and time name,
  # together: a resolution, or :nothing_said when neither was given.
  defp moment!(texts, turn, type) do
    case Enum.reject(texts, &blank?/1) do
      [] ->
        :nothing_said

      said ->
        case Resolver.resolve!(said, turn) do
          %Resolution{length: nil} = resolution ->
            resolution

          %Resolution{said: expression} ->
            raise ArgumentError,
                  "LocalScheduler: #{inspect(expression)} names a length, not the date or time of a #{type}"
        end
    end
  end

  # What a timer's duration names: a length, or :nothing_said when none was
  # given.
  defp length!(duration, turn) do
    cond do
      blank?(duration) ->
        :nothing_said

      true ->
        case Resolver.resolve!(duration, turn) do
          %Resolution{length: %Duration{}} = resolution -> resolution
          %Resolution{said: said} -> raise ArgumentError, "LocalScheduler: the timer's duration #{inspect(said)} names no length"
        end
    end
  end

  defp blank?(text) when is_binary(text), do: String.trim(text) == ""

  defp blank?(other),
    do: raise(ArgumentError, "LocalScheduler: a date, time or duration must be a string, got: #{inspect(other)}")

  defp add_item(world_id, type, data, resolution) do
    :ok = GenServer.call(running!(), {:add, world_id, type, data, resolution})
  end

  defp cancel_items(world_id, type) do
    :ok = GenServer.call(running!(), {:cancel, world_id, type})
  end

  # Removes the world's items of `type` that the slots identify, in `mode`
  # (`handle_call/3`). `:one` with no identifying slot is refused before the
  # scheduler is asked, since it could only match every item: the refusal
  # names the world's items of `type`, in the order they were set, for the
  # response layer to ask which is meant.
  defp cancel_matching(world_id, type, slots, turn, mode) do
    fields = Map.fetch!(@identifying_slots, type)
    said = said!(fields, type, slots)

    case {mode, said} do
      {:one, said} when said == %{} ->
        case world_id |> list_items(type) |> Enum.reverse() do
          [] -> {:error, {:no_matching_items, type, said}}
          items -> {:error, {:event_not_identified, Enum.map(items, &public/1)}}
        end

      {mode, said} ->
        criteria = criteria!(said, turn)
        named = Map.new(said, fn {field, value} -> {field, normalize(value)} end)
        GenServer.call(running!(), {:cancel_matching, world_id, type, criteria, named, mode})
    end
  end

  # The items, of `type`, that are for the days the slots name; all of them
  # when no slot names a day (see "Checking" above).
  defp on_day(items, type, slots, turn) do
    criteria = @day_slots |> Map.fetch!(type) |> said!(type, slots) |> criteria!(turn)
    Enum.filter(items, &matches?(&1, criteria))
  end

  # The value each slot says for `fields`, each a field of an item of `type`
  # with the slot that names it, as it was said; slots that name nothing are
  # left out.
  defp said!(fields, type, slots) do
    Enum.reduce(fields, %{}, fn {field, slot}, acc ->
      case Map.get(slots, slot) do
        nil ->
          acc

        value when is_binary(value) ->
          if blank?(value), do: acc, else: Map.put(acc, field, value)

        other ->
          raise ArgumentError,
                "LocalScheduler: the #{inspect(slot)} slot identifies #{type} items by their #{field} and " <>
                  "must be a string, got: #{inspect(other)}"
      end
    end)
  end

  # Each value said, as what an item's field is matched against (see
  # "Matching a slot to an item").
  defp criteria!(said, turn) do
    Map.new(said, fn {field, value} ->
      criterion =
        case Map.fetch!(@field_matches, field) do
          :text ->
            {:text, normalize(value)}

          :duration ->
            case Resolver.resolve!(value, turn) do
              %Resolution{length: %Duration{}} = resolution -> {:length, Resolver.length_seconds(resolution)}
              _other -> raise ArgumentError, "LocalScheduler: the duration #{inspect(value)} names no length"
            end

          kind when kind in [:date, :time] ->
            case Resolver.resolve!(value, turn) do
              %Resolution{length: nil} = resolution -> {kind, resolution}
              _length -> raise ArgumentError, "LocalScheduler: #{inspect(value)} names a length, not a #{kind}"
            end
        end

      {field, criterion}
    end)
  end

  defp matches?(data, criteria) do
    Enum.all?(criteria, fn {field, criterion} -> field_matches?(data, field, criterion) end)
  end

  defp field_matches?(data, field, {:text, value}) do
    case Map.fetch!(data, field) do
      nil -> false
      stored -> normalize(stored) == value
    end
  end

  # An item with nothing resolved matches no date, time or duration.
  defp field_matches?(%{resolution: %Resolution{} = item}, _field, criterion), do: resolved_match?(item, criterion)
  defp field_matches?(_data, _field, _criterion), do: false

  defp resolved_match?(item, {:date, said}), do: Resolver.same_day?(item, said)

  defp resolved_match?(item, {:time, %Resolution{clock: nil} = said}), do: Resolver.same_day?(item, said)

  defp resolved_match?(item, {:time, %Resolution{clock: clock} = said}) do
    item.clock != nil and Time.compare(item.clock, clock) == :eq and
      (not said.day_said or Resolver.same_day?(item, said))
  end

  defp resolved_match?(%Resolution{length: %Duration{}} = item, {:length, seconds}),
    do: Resolver.length_seconds(item) == seconds

  defp resolved_match?(_item, {:length, _seconds}), do: false

  # Whether a calendar item's time has not passed at the turn: a point in
  # time not before it, a day or period not over, a time with no end.
  defp still_to_come?(%{resolution: %Resolution{starts_at: starts_at, ends_at: ends_at}}, turn) do
    cond do
      is_nil(ends_at) -> true
      DateTime.compare(ends_at, starts_at) == :eq -> DateTime.compare(starts_at, turn) != :lt
      true -> DateTime.compare(ends_at, turn) == :gt
    end
  end

  defp still_to_come?(_item, _turn), do: false

  # Case and runs of whitespace are ignored when an item's text is matched.
  defp normalize(text), do: text |> String.split() |> Enum.join(" ") |> String.downcase()

  # An item as replies and errors name it: its fields as they were said.
  defp public(data), do: Map.drop(data, @internal_keys)

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
  defp row_attrs(world_id, :calendar, %{event: event, time: time, date: date}, resolution),
    do: Map.merge(%{world_id: world_id, kind: @kinds.calendar, text: event, time: time, date: date}, resolution_attrs(resolution))

  defp row_attrs(world_id, type, data, resolution),
    do: data |> Map.merge(%{world_id: world_id, kind: Map.fetch!(@kinds, type)}) |> Map.merge(resolution_attrs(resolution))

  defp resolution_attrs(:nothing_said), do: %{resolution: "nothing_said"}

  defp resolution_attrs(%Resolution{} = resolution) do
    %{
      resolution: "resolved",
      turn_time: resolution.turn_time,
      time_zone: resolution.time_zone,
      starts_at: resolution.starts_at,
      ends_at: resolution.ends_at,
      clock: resolution.clock
    }
  end

  defp entry(%Atlas.Schemas.SchedulerItem{} = item) do
    type = Map.fetch!(@types, item.kind)
    {{item.world_id, item.id}, type, type |> data(item) |> Map.put(:resolution, resolution(type, item))}
  end

  defp data(:alarm, item),
    do: %{time: item.time, date: item.date, status: item.status, created: item.inserted_at}

  defp data(:reminder, item),
    do: %{text: item.text, time: item.time, date: item.date, status: item.status, created: item.inserted_at}

  defp data(:timer, item),
    do: %{duration: item.duration, status: item.status, created: item.inserted_at}

  defp data(:calendar, item),
    do: %{event: item.text, time: item.time, date: item.date, created: item.inserted_at}

  # A stored item's resolution, as the resolver gave it when the item was set.
  # A timer's length is the time from its start to its end. Whether a day was
  # said is not kept (nothing matches on it but what a slot says now).
  defp resolution(type, %Atlas.Schemas.SchedulerItem{resolution: "resolved"} = item) do
    said =
      case type do
        :timer -> item.duration
        _dated -> [item.date, item.time] |> Enum.reject(&(&1 in [nil, ""])) |> Enum.join(" ")
      end

    %Resolution{
      said: said,
      turn_time: item.turn_time,
      time_zone: item.time_zone,
      starts_at: item.starts_at,
      ends_at: item.ends_at,
      clock: item.clock,
      day_said: nil,
      length: if(type == :timer, do: Duration.new!(second: DateTime.diff(item.ends_at, item.starts_at, :second)))
    }
  end

  defp resolution(_type, %Atlas.Schemas.SchedulerItem{resolution: "nothing_said"}), do: :nothing_said

  defp resolution(_type, %Atlas.Schemas.SchedulerItem{resolution: "recorded_before_resolution"}),
    do: :recorded_before_resolution
end
