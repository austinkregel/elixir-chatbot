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
  `:turn_time` slot (a `DateTime`), and the zone of the person who asked, as
  the `:time_zone` slot (a `Brain.Temporal.Zone`); `Brain.Services.Dispatcher`
  puts both there from the turn, and an operation raises without either: what
  "tomorrow" or "5 pm" names depends on when and where it is said. Setting an
  alarm, timer or reminder or scheduling a calendar item also needs the web
  user who asks, as the `:web_user_id` slot, and raises without one: the item
  is theirs, and they are told when it fires.

  ## Firing

  An item fires at its due time: a timer when it runs out, an alarm,
  reminder or calendar item at the time of day it was set for, on the day it
  was set for. An item whose time names no time of day ("tomorrow", "later")
  or that has nothing resolved names no instant to fire at, and never fires
  (`"no_due_time"`).

  This module's process wakes at the soonest due time Atlas holds, read from
  Atlas whenever it starts and after every change, fires every item due by
  then, and sets its next wake-up (`Brain.Services.SchedulerClock`, the clock
  `config :brain, :scheduler_clock` names). Firing records in Atlas that the
  item fired and when (`"fired"`); an item that came due while the process was
  not running fires as soon as it starts, and is recorded as fired late
  (`"fired_late"`). A fired alarm is ringing (`"ringing"`) until it is snoozed
  or cancelled, and a fired timer has finished (`"finished"`).

  Each firing is told to the item's owner: `{:scheduler_fired, notification}`
  is broadcast on their topic (`Atlas.WebUsers.topic/1`) on `Brain.PubSub`,
  and on no other (`notification/1`). An item stored before owners were has
  no owner, and never fires (`"owner_not_recorded"`).

  `snooze_alarm/3` snoozes the world's ringing alarms for the length its
  `:duration` slot names, from the turn: each stops ringing and is due again
  then. With no alarm ringing it is `{:error, :no_ringing_alarm}`, and with
  no length said, `{:error, :missing_duration}`, for the response layer to
  ask how long.

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

  Days are read in the zone of the turn the slot is said in, for the item as
  for what is said.

  An item with nothing resolved, or stored before resolutions were, matches no
  date, time or duration. A slot holding anything but a string raises, and so
  does one that cannot be resolved.

  ## Checking

  `check_alarms/3` and `check_reminders/3` report the alarms or reminders the
  world has, in the order they were set, with how many there are. A check of
  alarms also reports the times of those ringing (`:ringing_times`), and its
  `:alarm_status` is `"ringing"` when one is; a check of reminders reports the
  texts of those that have fired (`:fired_reminder_texts`).
  `query_calendar/3` reports the soonest calendar item still to come
  (`:next_event`): of the items found, those whose time has not passed at the
  turn, ordered by when they start, the first set first among those starting
  together; an item with nothing resolved, or stored before resolutions were,
  has no time to order by and is never the next. It reports how many items it
  found (`:event_count`) and how many of them are still to come
  (`:upcoming_count`). Each of the three narrows its items to the days the
  `:date` slot names, a check of reminders to those whose text the
  `:reminder_text` slot names, and a query of the calendar to the items whose
  event the `:event` slot names, each matched as above; with none of them
  said, it reports all of them. No other slot narrows a check. When a check finds nothing, its count is
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
  are read from: it loads every stored item when it starts, and every set,
  cancel, snooze and firing goes through it, writing Atlas first and the
  table after. A cancel
  finds the items it removes inside that same call, so no other write can come
  between matching and removing. Reads go straight to the table.
  """

  @behaviour Brain.Services.Service
  @behaviour Brain.Services.SchedulingDriver

  use GenServer
  require Logger

  alias Brain.Services.{SchedulerClock, SchedulingDriver}
  alias Brain.Temporal.{Resolution, Resolver, Vocabulary, Zone}

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

  # The fields a check narrows the items of each type by, each with the slot
  # that names it (see "Checking" above).
  @check_slots %{
    alarm: [date: :date],
    reminder: [text: :reminder_text, date: :date],
    calendar: [event: :event, date: :date]
  }

  # How each identifying field is matched (see "Matching a slot to an item").
  @field_matches %{text: :text, event: :text, date: :date, time: :time, duration: :duration}

  # What an item keeps beside its fields, which no reply or error names.
  @internal_keys [:id, :created, :resolution, :web_user_id, :due_at, :fire_state, :fired_at]

  # What a fired item's status becomes: an alarm rings, a timer has finished.
  @fired_status %{alarm: "ringing", timer: "finished"}

  @fired_states ["fired", "fired_late"]

  @doc """
  Starts the scheduler, which loads every stored item into its table, fires
  those that came due while it was not running, and wakes when the next is
  due. `opts` may name the `:clock` (a `Brain.Services.SchedulerClock`);
  without it the clock is `config :brain, :scheduler_clock`.
  """
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
    [:alarm_status, :alarm_time, :alarm_date, :alarm_times, :alarm_dates, :alarm_count, :ringing_times,
     :snooze_duration, :reminder_set, :reminder_text, :reminder_time, :reminder_date, :reminder_texts,
     :reminder_times, :reminder_dates, :reminder_count, :fired_reminder_texts, :timer_status, :duration,
     :time_left, :timer_durations, :last_set_event, :next_event, :event_time, :event_date, :event_count,
     :upcoming_count, :cancelled_event, :cancelled_count]
  end

  # Setting, creating, scheduling and cancelling store or remove items, and a
  # snooze reschedules them; checking and querying only read them.
  @writing_operations [
    :set_alarm,
    :snooze_alarm,
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
    turn = turn!(slots, :set_alarm)
    owner = owner!(slots, :set_alarm)
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :alarm, %{time: time, date: date, status: "set"}, moment!([date, time], turn, :alarm), owner)
    {:ok, %{alarm_status: "set", alarm_time: time, alarm_date: date}}
  end

  # The world's ringing alarms, each due again once the length the
  # `:duration` slot names has passed from the turn (see "Firing" above).
  @impl SchedulingDriver
  def snooze_alarm(slots, _credentials, world_id) do
    world_id = world!(world_id, :snooze_alarm)
    turn = turn!(slots, :snooze_alarm)
    duration = Map.get(slots, :duration) || Map.get(slots, "duration", "")

    running!()

    with [_ | _] <- ringing(world_id),
         %Resolution{ends_at: until} <- length!(duration, turn),
         {:ok, alarms} <- GenServer.call(running!(), {:snooze, world_id, until}) do
      {:ok,
       %{
         alarm_status: "snoozed",
         alarm_times: Enum.map(alarms, & &1.time),
         alarm_dates: Enum.map(alarms, & &1.date),
         snooze_duration: duration
       }}
    else
      [] -> {:error, :no_ringing_alarm}
      :nothing_said -> {:error, :missing_duration}
      {:error, :no_ringing_alarm} = error -> error
    end
  end

  # The world's ringing alarms, each with its stored id.
  defp ringing(world_id), do: world_id |> entries(:alarm) |> Enum.filter(fn {_id, data} -> data.status == "ringing" end)

  @impl SchedulingDriver
  def cancel_alarm(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_alarm)
    turn = turn!(slots, :cancel_alarm)

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
    turn = turn!(slots, :check_alarms)
    alarms = world_id |> list_items(:alarm) |> Enum.reverse() |> narrowed(:alarm, slots, turn)
    ringing = Enum.filter(alarms, &(&1.status == "ringing"))

    status =
      cond do
        alarms == [] -> "none"
        ringing != [] -> "ringing"
        true -> "set"
      end

    {:ok,
     %{
       alarm_status: status,
       alarm_times: Enum.map(alarms, & &1.time),
       alarm_dates: Enum.map(alarms, & &1.date),
       alarm_count: length(alarms),
       ringing_times: Enum.map(ringing, & &1.time)
     }}
  end

  @impl SchedulingDriver
  def create_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :create_reminder)
    turn = turn!(slots, :create_reminder)
    owner = owner!(slots, :create_reminder)
    text = Map.get(slots, :reminder_text) || Map.get(slots, "reminder_text", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")

    add_item(
      world_id,
      :reminder,
      %{text: text, time: time, date: date, status: "set"},
      moment!([date, time], turn, :reminder),
      owner
    )

    {:ok, %{reminder_set: true, reminder_text: text, reminder_time: time, reminder_date: date}}
  end

  @impl SchedulingDriver
  def cancel_reminder(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_reminder)
    turn = turn!(slots, :cancel_reminder)

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
    turn = turn!(slots, :check_reminders)
    reminders = world_id |> list_items(:reminder) |> Enum.reverse() |> narrowed(:reminder, slots, turn)

    {:ok,
     %{
       reminder_set: reminders != [],
       reminder_texts: Enum.map(reminders, & &1.text),
       reminder_times: Enum.map(reminders, & &1.time),
       reminder_dates: Enum.map(reminders, & &1.date),
       reminder_count: length(reminders),
       fired_reminder_texts: for(%{fire_state: state, text: text} <- reminders, state in @fired_states, do: text)
     }}
  end

  @impl SchedulingDriver
  def set_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :set_timer)
    turn = turn!(slots, :set_timer)
    owner = owner!(slots, :set_timer)
    duration = Map.get(slots, :duration) || Map.get(slots, "duration", "")
    add_item(world_id, :timer, %{duration: duration, status: "running"}, length!(duration, turn), owner)
    {:ok, %{timer_status: "running", duration: duration}}
  end

  # The timer set last, with how much of it is left at the turn.
  @impl SchedulingDriver
  def check_timer(slots, _credentials, world_id) do
    world_id = world!(world_id, :check_timer)
    turn = turn!(slots, :check_timer)

    case list_items(world_id, :timer) do
      [] ->
        {:ok, %{timer_status: "none", duration: "", time_left: []}}

      [%{resolution: %Resolution{ends_at: %DateTime{} = ends_at}} = latest | _] ->
        case DateTime.diff(ends_at, turn.time, :microsecond) do
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
    turn = turn!(slots, :cancel_timer)

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
    turn = turn!(slots, :schedule_event)
    owner = owner!(slots, :schedule_event)
    event = Map.get(slots, :event) || Map.get(slots, "event", "")
    time = Map.get(slots, :time) || Map.get(slots, "time", "")
    date = Map.get(slots, :date) || Map.get(slots, "date", "")
    add_item(world_id, :calendar, %{event: event, time: time, date: date}, moment!([date, time], turn, :calendar), owner)
    {:ok, %{last_set_event: event, event_time: time, event_date: date}}
  end

  # Takes the one calendar item the slots identify off the world's calendar,
  # and never more than one (see "Cancelling" above).
  @impl SchedulingDriver
  def cancel_event(slots, _credentials, world_id) do
    world_id = world!(world_id, :cancel_event)
    turn = turn!(slots, :cancel_event)

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
    turn = turn!(slots, :query_calendar)
    events = world_id |> list_items(:calendar) |> Enum.reverse() |> narrowed(:calendar, slots, turn)

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

  @doc """
  Fires every item due by the clock's current instant, as a wake-up does, and
  sets the next wake-up. Returns when they have fired.
  """
  @spec wake() :: :ok
  def wake, do: GenServer.call(running!(), :wake)

  @doc """
  What the web user's open pages are told of their alarms ringing now, in
  any world, oldest first (`notification/1`): what a page opened while one
  rings shows.
  """
  @spec ringing_notifications(String.t()) :: [map()]
  def ringing_notifications(web_user_id) when is_binary(web_user_id) do
    running!()

    @table
    |> :ets.match_object({{:_, :_}, :alarm, :_})
    |> Enum.filter(fn {_key, _type, data} -> data.web_user_id == web_user_id and data.status == "ringing" end)
    |> Enum.sort_by(fn {_key, _type, data} -> data.created end, {:asc, DateTime})
    |> Enum.map(&notification/1)
  end

  @doc """
  Every item of the world, oldest first, as `notification/1` gives it with
  its `:fire_state`, its owner (`:web_user_id`) and its `:status` as stored
  (`nil` for a calendar item, which has none): what a page listing them shows.
  Raises without a world.
  """
  @spec items(String.t()) :: [map()]
  def items(world_id) do
    world_id = world!(world_id, :items)
    running!()

    @table
    |> :ets.match_object({{world_id, :_}, :_, :_})
    |> Enum.sort_by(fn {_key, _type, data} -> data.created end, {:asc, DateTime})
    |> Enum.map(fn {_key, _type, data} = entry ->
      entry
      |> notification()
      |> Map.merge(%{fire_state: data.fire_state, web_user_id: data.web_user_id, status: Map.get(data, :status)})
    end)
  end

  @doc """
  What an item's owner is told when it fires, from its table entry:

  - `:id`, `:world_id`, `:kind` (`:alarm`, `:reminder`, `:timer`, `:calendar`);
  - `:said`, the fields it was set with, as they were said (an alarm's
    `:time` and `:date`, a reminder's `:text`, `:time` and `:date`, a timer's
    `:duration`, a calendar item's `:event`, `:time` and `:date`);
  - `:due_at` and `:fired_at`, instants in UTC, and `:late`, whether it fired
    late, having come due while the scheduler was not running;
  - `:status`: an alarm's `"ringing"`, a timer's `"finished"`, or `nil` for a
    reminder or calendar item, which have no status that firing changes.
  """
  @spec notification({{String.t(), String.t()}, atom(), map()}) :: map()
  def notification({{world_id, id}, type, data}) do
    %{
      id: id,
      world_id: world_id,
      kind: type,
      said: Map.drop(data, @internal_keys ++ [:status]),
      due_at: data.due_at,
      fired_at: data.fired_at,
      late: data.fire_state == "fired_late",
      status: Map.get(@fired_status, type)
    }
  end

  # Server callbacks

  @impl true
  def init(opts) do
    clock = Keyword.get_lazy(opts, :clock, &SchedulerClock.configured!/0)
    table = :ets.new(@table, [:named_table, :set, :protected, read_concurrency: true])

    items = Atlas.Scheduler.list_items()
    Enum.each(items, &:ets.insert(table, entry(&1)))

    Logger.info("Local scheduler loaded #{length(items)} stored items")

    # What came due while the scheduler was not running fires now, late.
    state = %{table: table, clock: clock, wake_ref: nil, wake_at: nil}
    fired = fire_due(state, "fired_late")
    if fired != [], do: Logger.info("Local scheduler fired #{length(fired)} items late, due while it was not running")

    {:ok, arm(state)}
  end

  @impl true
  def handle_call({:add, world_id, type, data, resolution, owner}, _from, state) do
    item = Atlas.Scheduler.insert_item!(row_attrs(world_id, type, data, resolution, owner))
    :ets.insert(state.table, entry(item))
    {:reply, :ok, arm(state)}
  end

  @impl true
  def handle_call({:cancel, world_id, type}, _from, state) do
    Atlas.Scheduler.delete_items!(world_id, Map.fetch!(@kinds, type))
    :ets.match_delete(state.table, {{world_id, :_}, type, :_})
    {:reply, :ok, arm(state)}
  end

  # The world's ringing alarms are found and snoozed in this one call, so no
  # cancel or firing comes between. Replies with the alarms snoozed, oldest
  # first.
  def handle_call({:snooze, world_id, until}, _from, state) do
    case ringing(world_id) do
      [] ->
        {:reply, {:error, :no_ringing_alarm}, state}

      ringing ->
        Enum.each(ringing, fn {id, _data} ->
          item = Atlas.Scheduler.update_item!(id, %{status: "set", fire_state: "pending", due_at: until, fired_at: nil})
          :ets.insert(state.table, entry(item))
        end)

        alarms = ringing |> Enum.map(&elem(&1, 1)) |> Enum.sort_by(& &1.created, {:asc, DateTime})
        {:reply, {:ok, alarms}, arm(state)}
    end
  end

  def handle_call(:wake, _from, state) do
    fire_due(state, "fired")
    {:reply, :ok, arm(state)}
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

    {:reply, reply, arm(state)}
  end

  # The clock's wake-up: what is due fires, and the next wake-up is set.
  @impl true
  def handle_info(:wake, state) do
    fire_due(state, "fired")
    {:noreply, arm(state)}
  end

  # Every stored item belongs to the world it was set in; there is no default.
  defp world!(world_id, _function) when is_binary(world_id) and world_id != "", do: world_id

  defp world!(other, function) do
    raise ArgumentError,
          "LocalScheduler.#{function} needs the world whose alarms, reminders, timers and calendar items are used, got: #{inspect(other)}"
  end

  # What a date or time names depends on when and where it is said: the turn
  # is its moment and its zone, and there is no default for either.
  defp turn!(slots, function) do
    time =
      case Map.get(slots, :turn_time) do
        %DateTime{} = turn ->
          turn

        other ->
          raise ArgumentError,
                "LocalScheduler.#{function} needs the :turn_time slot, the moment of the turn it is asked in, " <>
                  "got: #{inspect(other)}"
      end

    zone =
      case Map.get(slots, :time_zone) do
        %Zone{} = zone ->
          zone

        other ->
          raise ArgumentError,
                "LocalScheduler.#{function} needs the :time_zone slot, the Brain.Temporal.Zone of the person who " <>
                  "asks, got: #{inspect(other)}"
      end

    %{time: time, zone: zone}
  end

  # An item belongs to the web user who sets it; there is no default.
  defp owner!(slots, function) do
    case Map.get(slots, :web_user_id) do
      owner when is_binary(owner) and owner != "" ->
        owner

      other ->
        raise ArgumentError,
              "LocalScheduler.#{function} needs the :web_user_id slot, the web user who sets the item and is told " <>
                "when it fires, got: #{inspect(other)}"
    end
  end

  # What an alarm's, reminder's or calendar item's date and time name,
  # together: a resolution, or :nothing_said when neither was given.
  defp moment!(texts, turn, type) do
    case Enum.reject(texts, &blank?/1) do
      [] ->
        :nothing_said

      said ->
        case Resolver.resolve!(said, turn.time, time_zone: turn.zone) do
          %Resolution{length: nil} = resolution ->
            resolution

          %Resolution{said: expression} ->
            raise ArgumentError,
                  "LocalScheduler: #{inspect(expression)} names a length, not the date or time of a #{type}"
        end
    end
  end

  # What a timer's or a snooze's duration names: a length from the turn, or
  # :nothing_said when none was given.
  defp length!(duration, turn) do
    cond do
      blank?(duration) ->
        :nothing_said

      true ->
        case Resolver.resolve!(duration, turn.time, time_zone: turn.zone) do
          %Resolution{length: %Duration{}} = resolution -> resolution
          %Resolution{said: said} -> raise ArgumentError, "LocalScheduler: the duration #{inspect(said)} names no length"
        end
    end
  end

  defp blank?(text) when is_binary(text), do: String.trim(text) == ""

  defp blank?(other),
    do: raise(ArgumentError, "LocalScheduler: a date, time or duration must be a string, got: #{inspect(other)}")

  defp add_item(world_id, type, data, resolution, owner) do
    :ok = GenServer.call(running!(), {:add, world_id, type, data, resolution, owner})
  end

  # Firing (see "Firing" above).

  # Fires every pending item due by the clock's instant, as `fire_state`
  # ("fired", or "fired_late" for what came due while the scheduler was not
  # running): records it in Atlas, then in the table, then tells its owner.
  # Returns the items fired.
  defp fire_due(state, fire_state) do
    now = state.clock.now()

    for item <- Atlas.Scheduler.due_items(now) do
      type = Map.fetch!(@types, item.kind)

      attrs =
        case Map.fetch(@fired_status, type) do
          {:ok, status} -> %{fire_state: fire_state, fired_at: now, status: status}
          :error -> %{fire_state: fire_state, fired_at: now}
        end

      fired = Atlas.Scheduler.update_item!(item.id, attrs)
      entry = entry(fired)
      :ets.insert(state.table, entry)
      Phoenix.PubSub.broadcast(Brain.PubSub, Atlas.WebUsers.topic(fired.web_user_id), {:scheduler_fired, notification(entry)})
      fired
    end
  end

  # Sets the wake-up for the soonest pending item Atlas holds, in place of
  # the one set before; none when nothing is pending.
  defp arm(state) do
    if state.wake_ref, do: state.clock.cancel(state.wake_ref)

    case Atlas.Scheduler.next_due_at() do
      nil -> %{state | wake_ref: nil, wake_at: nil}
      due_at -> %{state | wake_ref: state.clock.send_at(self(), :wake, due_at), wake_at: due_at}
    end
  end

  # When an item fires: a timer when it runs out; an alarm, reminder or
  # calendar item at the time of day it was set for. Anything else names no
  # instant to fire at.
  defp due_at(:timer, %Resolution{length: %Duration{}, ends_at: %DateTime{} = ends_at}), do: ends_at
  defp due_at(type, %Resolution{clock: %Time{}, starts_at: starts_at}) when type != :timer, do: starts_at
  defp due_at(_type, _resolution), do: nil

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

  # The items, of `type`, that match what the check's slots name (the days, and
  # a reminder's text or a calendar item's event); all of them when no such
  # slot names anything (see "Checking" above).
  defp narrowed(items, type, slots, turn) do
    criteria = @check_slots |> Map.fetch!(type) |> said!(type, slots) |> criteria!(turn)
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
            case Resolver.resolve!(value, turn.time, time_zone: turn.zone) do
              %Resolution{length: %Duration{}} = resolution -> {:length, Resolver.length_seconds(resolution)}
              _other -> raise ArgumentError, "LocalScheduler: the duration #{inspect(value)} names no length"
            end

          kind when kind in [:date, :time] ->
            case Resolver.resolve!(value, turn.time, time_zone: turn.zone) do
              %Resolution{length: nil} = resolution -> {kind, resolution, turn.zone}
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

  # Days are read in the zone of the turn the slot was said in.
  defp resolved_match?(item, {:date, said, zone}), do: Resolver.same_day?(item, said, zone)

  defp resolved_match?(item, {:time, %Resolution{clock: nil} = said, zone}), do: Resolver.same_day?(item, said, zone)

  defp resolved_match?(item, {:time, %Resolution{clock: clock} = said, zone}) do
    item.clock != nil and Time.compare(item.clock, clock) == :eq and
      (not said.day_said or Resolver.same_day?(item, said, zone))
  end

  defp resolved_match?(%Resolution{length: %Duration{}} = item, {:length, seconds}),
    do: Resolver.length_seconds(item) == seconds

  defp resolved_match?(_item, {:length, _seconds}), do: false

  # Whether a calendar item's time has not passed at the turn: a point in
  # time not before it, a day or period not over, a time with no end.
  defp still_to_come?(%{resolution: %Resolution{starts_at: starts_at, ends_at: ends_at}}, %{time: turn}) do
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
  defp row_attrs(world_id, :calendar, %{event: event, time: time, date: date}, resolution, owner),
    do: row_attrs(world_id, :calendar, %{text: event, time: time, date: date}, resolution, owner)

  defp row_attrs(world_id, type, data, resolution, owner) do
    data
    |> Map.merge(%{world_id: world_id, kind: Map.fetch!(@kinds, type)})
    |> Map.merge(resolution_attrs(resolution))
    |> Map.merge(fire_attrs(type, resolution, owner))
  end

  defp resolution_attrs(:nothing_said), do: %{resolution: "nothing_said"}

  defp resolution_attrs(%Resolution{} = resolution) do
    %{
      resolution: "resolved",
      turn_time: resolution.turn_time,
      time_zone: resolution.time_zone,
      utc_offset: resolution.utc_offset,
      starts_at: resolution.starts_at,
      ends_at: resolution.ends_at,
      clock: resolution.clock
    }
  end

  # A new item is pending when it names an instant to fire at.
  defp fire_attrs(type, resolution, owner) do
    case due_at(type, resolution) do
      nil -> %{web_user_id: owner, fire_state: "no_due_time"}
      due_at -> %{web_user_id: owner, fire_state: "pending", due_at: due_at}
    end
  end

  defp entry(%Atlas.Schemas.SchedulerItem{} = item) do
    type = Map.fetch!(@types, item.kind)

    data =
      type
      |> data(item)
      |> Map.merge(%{
        id: item.id,
        resolution: resolution(type, item),
        web_user_id: item.web_user_id,
        due_at: item.due_at,
        fire_state: item.fire_state,
        fired_at: item.fired_at
      })

    {{item.world_id, item.id}, type, data}
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
      utc_offset: item.utc_offset,
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
