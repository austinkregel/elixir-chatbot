defmodule Brain.Services.LocalSchedulerTest do
  use ExUnit.Case, async: false

  alias Brain.Services.{Dispatcher, LocalScheduler, SchedulingDriver}

  # Items are stored in Atlas, so the scheduler reads and writes inside this
  # test's sandbox transaction. The scheduler keys every item by world id and
  # reads no world directory, so these tests use fresh `test_` ids and clear
  # both worlds before the transaction is rolled back, which also removes them
  # from the scheduler's table.
  setup tags do
    owner = Brain.Test.AtlasSandbox.checkout_and_configure!(tags)
    on_exit(fn -> Brain.Test.AtlasSandbox.drain_and_stop_owner(owner) end)

    suffix = System.unique_integer([:positive])
    world_a = "test_scheduler_a_#{suffix}"
    world_b = "test_scheduler_b_#{suffix}"

    on_exit(fn ->
      LocalScheduler.clear_world(world_a)
      LocalScheduler.clear_world(world_b)
    end)

    %{world_a: world_a, world_b: world_b}
  end

  # The turns these tests are said in: a Monday morning, unless a test says
  # otherwise.
  @monday ~U[2026-10-12 09:00:00Z]
  @tuesday ~U[2026-10-13 09:00:00Z]

  # The slots of a turn: what was said, and when.
  defp said(slots, turn \\ @monday), do: Map.put(slots, :turn_time, turn)

  # What Atlas holds of the world's items of one kind, oldest first: the
  # alarms and reminders no operation reads back.
  defp stored(world_id, kind) do
    Enum.filter(Atlas.Scheduler.list_items(), &(&1.world_id == world_id and &1.kind == kind))
  end

  defp restart! do
    :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
    {:ok, _restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)
  end

  describe "supported_intents/0" do
    test "is every scheduling operation's intent, and nothing else" do
      assert LocalScheduler.supported_intents() ==
               Enum.sort([
                 "alarm.set",
                 "alarm.snooze",
                 "alarm.cancel",
                 "alarm.check",
                 "timer.set",
                 "timer.check",
                 "timer.cancel",
                 "reminder.create",
                 "reminder.cancel",
                 "reminder.check",
                 "calendar.query",
                 "calendar.schedule",
                 "calendar.cancel"
               ])
    end

    test "each is an intent in the intent registry, and every registry intent of a scheduling domain is one" do
      registry =
        "analysis/intent_registry.json"
        |> Brain.priv_path()
        |> File.read!()
        |> Jason.decode!()

      for intent <- LocalScheduler.supported_intents() do
        assert Map.has_key?(registry, intent), "#{intent} is not in the intent registry"
      end

      for {intent, _schema} <- registry, SchedulingDriver.domain?(intent) do
        assert intent in LocalScheduler.supported_intents(), "registry intent #{intent} names no scheduling operation"
      end
    end

    test "every slot an operation reads is declared by its intent in the intent registry" do
      registry =
        "analysis/intent_registry.json"
        |> Brain.priv_path()
        |> File.read!()
        |> Jason.decode!()

      # The slots each operation stores, identifies items by, or narrows a
      # check by (see the moduledoc).
      reads = %{
        "alarm.set" => ["time", "date"],
        "alarm.cancel" => ["time", "date"],
        "alarm.check" => ["date"],
        "timer.set" => ["duration"],
        "timer.cancel" => ["duration"],
        "reminder.create" => ["reminder_text", "time", "date"],
        "reminder.cancel" => ["reminder_text", "time", "date"],
        "reminder.check" => ["date"],
        "calendar.schedule" => ["event", "time", "date"],
        "calendar.cancel" => ["event", "time", "date"],
        "calendar.query" => ["date"]
      }

      for {intent, slots} <- reads, slot <- slots do
        schema = Map.fetch!(registry, intent)
        declared = Map.fetch!(schema, "required") ++ Map.fetch!(schema, "optional")
        assert slot in declared, "#{intent} reads the #{slot} slot, which its registry entry does not declare"
      end
    end

    test "each has gold-standard examples, some held out and the rest for training" do
      gold = Brain.ML.EvaluationStore.load_gold_standard("intent")
      held_out = Brain.ML.EvaluationStore.load_gold_standard("intent", :held_out)
      training = Brain.ML.EvaluationStore.load_gold_standard("intent", :train)

      for intent <- LocalScheduler.supported_intents() do
        assert Enum.count(gold, &(&1["intent"] == intent)) >= 10, "#{intent} has fewer than 10 examples"
        assert Enum.any?(held_out, &(&1["intent"] == intent)), "#{intent} has no held-out example"
        assert Enum.any?(training, &(&1["intent"] == intent)), "#{intent} has no training example"
      end
    end
  end

  describe "enrich/4 reads the operation from the intent's domain and verb" do
    test "an intent that is not exactly a domain and an operation's verb is refused, and nothing is stored",
         %{world_a: world_a} do
      for intent <- ["alarm.set.recurring", "reminder.set", "calendar.create", "alarm.settings", "my.alarm.set"] do
        assert {:error, :unsupported_intent} = LocalScheduler.enrich(intent, said(%{time: "7am"}), %{}, world_a)
      end

      assert stored(world_a, "alarm") == []
      assert stored(world_a, "reminder") == []
      assert stored(world_a, "calendar") == []
    end

    test "every intent it supports names an operation, and none is refused as unsupported", %{world_a: world_a} do
      for intent <- LocalScheduler.supported_intents() do
        assert {:ok, _operation} = SchedulingDriver.operation(intent)
        refute LocalScheduler.enrich(intent, said(%{}), %{}, world_a) == {:error, :unsupported_intent}, intent
      end
    end

    test "a read operation answers the same through enrich/4 as called directly", %{world_a: world_a} do
      LocalScheduler.set_timer(said(%{duration: "4 minutes"}), %{}, world_a)
      LocalScheduler.schedule_event(said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)

      assert LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a) == LocalScheduler.check_timer(said(%{}), %{}, world_a)

      assert LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a) ==
               LocalScheduler.query_calendar(said(%{}), %{}, world_a)

      assert LocalScheduler.enrich("alarm.snooze", said(%{}), %{}, world_a) ==
               LocalScheduler.snooze_alarm(said(%{}), %{}, world_a)
    end
  end

  describe "operations, each in its world" do
    test "an alarm set in one world is stored in that world only", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{alarm_status: "set", alarm_time: "7am"}} =
               LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{time: "7am", status: "set"}] = stored(world_a, "alarm")
      assert stored(world_b, "alarm") == []
    end

    test "snoozing finds no ringing alarm, since alarms are stored and never rung", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)

      assert {:error, :no_ringing_alarm} =
               LocalScheduler.enrich("alarm.snooze", said(%{duration: "5 minutes"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{time: "7am", status: "set"}] = stored(world_a, "alarm")
    end

    test "each world stores its own reminder", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{reminder_set: true, reminder_text: "water plants", reminder_time: "noon"}} =
               LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", time: "noon"}), %{}, world_a)

      assert {:ok, %{reminder_set: true, reminder_text: "call home", reminder_time: "6pm"}} =
               LocalScheduler.enrich("reminder.create", said(%{reminder_text: "call home", time: "6pm"}), %{}, world_b)

      assert [%Atlas.Schemas.SchedulerItem{text: "water plants", time: "noon", status: "set"}] =
               stored(world_a, "reminder")

      assert [%Atlas.Schemas.SchedulerItem{text: "call home", time: "6pm", status: "set"}] =
               stored(world_b, "reminder")
    end

    test "a timer is read back only in the world it was set in", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", said(%{duration: "10 minutes"}), %{}, world_b)

      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "10 minutes"}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_b)
    end

    test "with two running timers, check reports the one set last", %{world_a: world_a} do
      assert {:ok, %{timer_status: "running"}} =
               LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)

      # Timers are ordered by the microsecond they were set; a millisecond
      # apart orders these two.
      Process.sleep(2)

      assert {:ok, %{timer_status: "running"}} =
               LocalScheduler.enrich("timer.set", said(%{duration: "10 minutes"}), %{}, world_a)

      assert LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a) ==
               {:ok, %{timer_status: "running", duration: "10 minutes", time_left: ["10 minutes"]}}
    end

    test "a scheduled event is read only in the world it was scheduled in", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{last_set_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday", event_count: 1, upcoming_count: 1}} =
               LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)

      assert LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_b) ==
               {:ok, %{next_event: "", event_time: "", event_date: "", event_count: 0, upcoming_count: 0}}
    end

    test "cancelling in one world takes the event off its calendar and leaves another world's",
         %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_b)

      assert {:ok, %{cancelled_event: "dentist", event_time: "Tuesday", cancelled_count: 1}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "dentist"}), %{}, world_a)

      assert {:ok, %{event_count: 0}} = LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, nil)
      end

      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("timer.check", said(%{}), %{}, "")
      end

      assert_raise ArgumentError, ~r/LocalScheduler.set_alarm needs the world/, fn ->
        LocalScheduler.set_alarm(said(%{time: "7am"}), %{}, nil)
      end

      assert_raise ArgumentError, ~r/LocalScheduler.query_calendar needs the world/, fn ->
        LocalScheduler.query_calendar(said(%{}), %{}, "")
      end
    end

    test "raises without the turn's time, and stores nothing", %{world_a: world_a} do
      for intent <- LocalScheduler.supported_intents() do
        assert_raise ArgumentError, ~r/needs the :turn_time slot/, fn ->
          LocalScheduler.enrich(intent, %{time: "7am", date: "tomorrow", duration: "5 minutes"}, %{}, world_a)
        end
      end

      assert_raise ArgumentError, ~r/needs the :turn_time slot/, fn ->
        LocalScheduler.set_alarm(%{time: "7am", turn_time: "2026-10-12"}, %{}, world_a)
      end

      for kind <- Atlas.Schemas.SchedulerItem.kinds(), do: assert(stored(world_a, kind) == [])
    end
  end

  describe "what is stored beside the text" do
    test "an alarm's date and time, resolved together against the turn", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "7 am", date: "tomorrow"}), %{}, world_a)

      assert [
               %Atlas.Schemas.SchedulerItem{
                 time: "7 am",
                 date: "tomorrow",
                 resolution: "resolved",
                 turn_time: ~U[2026-10-12 09:00:00.000000Z],
                 time_zone: "Etc/UTC",
                 starts_at: ~U[2026-10-13 07:00:00.000000Z],
                 ends_at: ~U[2026-10-13 07:00:00.000000Z],
                 clock: ~T[07:00:00]
               }
             ] = stored(world_a, "alarm")
    end

    test "a day alone covers the whole day, and a timer runs from the turn for its duration", %{world_a: world_a} do
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "pay rent", date: "Friday"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)

      assert [
               %Atlas.Schemas.SchedulerItem{
                 starts_at: ~U[2026-10-16 00:00:00.000000Z],
                 ends_at: ~U[2026-10-17 00:00:00.000000Z],
                 clock: nil
               }
             ] = stored(world_a, "reminder")

      assert [
               %Atlas.Schemas.SchedulerItem{
                 duration: "5 minutes",
                 starts_at: ~U[2026-10-12 09:00:00.000000Z],
                 ends_at: ~U[2026-10-12 09:05:00.000000Z]
               }
             ] = stored(world_a, "timer")
    end

    test "an item set with no date, time or duration has nothing resolved", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: ""}), %{}, world_a)

      for kind <- ["alarm", "timer"] do
        assert [%Atlas.Schemas.SchedulerItem{resolution: "nothing_said", turn_time: nil, starts_at: nil}] = stored(world_a, kind)
      end
    end

    test "what cannot be resolved raises, naming it, and nothing is stored", %{world_a: world_a} do
      assert_raise ArgumentError, ~r/cannot resolve "tomorow 7am": "tomorow" is no word/, fn ->
        LocalScheduler.enrich("alarm.set", said(%{time: "7am", date: "tomorow"}), %{}, world_a)
      end

      assert_raise ArgumentError, ~r/cannot resolve "five minutes"/, fn ->
        LocalScheduler.enrich("timer.set", said(%{duration: "five minutes"}), %{}, world_a)
      end

      assert_raise ArgumentError, ~r/"tomorrow" names no length/, fn ->
        LocalScheduler.enrich("timer.set", said(%{duration: "tomorrow"}), %{}, world_a)
      end

      assert_raise ArgumentError, ~r/"5 minutes" names a length, not the date or time of a reminder/, fn ->
        LocalScheduler.enrich("reminder.create", said(%{reminder_text: "stretch", time: "5 minutes"}), %{}, world_a)
      end

      for kind <- Atlas.Schemas.SchedulerItem.kinds(), do: assert(stored(world_a, kind) == [])
    end
  end

  describe "checking alarms and reminders" do
    test "reports every alarm the world has, in the order they were set", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{alarm_status: "none", alarm_times: [], alarm_count: 0}} =
               LocalScheduler.enrich("alarm.check", said(%{}), %{}, world_a)

      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("alarm.set", said(%{time: "9pm"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "noon"}), %{}, world_b)

      assert {:ok, %{alarm_status: "set", alarm_times: ["6am", "9pm"], alarm_count: 2}} =
               LocalScheduler.enrich("alarm.check", said(%{}), %{}, world_a)
    end

    test "reports every reminder the world has, in the order they were set", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{reminder_set: false, reminder_texts: [], reminder_count: 0}} =
               LocalScheduler.enrich("reminder.check", said(%{}), %{}, world_a)

      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", time: "noon"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "call home", time: "6pm"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "pay rent", time: "Friday"}), %{}, world_b)

      assert {:ok,
              %{
                reminder_set: true,
                reminder_texts: ["water plants", "call home"],
                reminder_times: ["noon", "6pm"],
                reminder_count: 2
              }} = LocalScheduler.enrich("reminder.check", said(%{}), %{}, world_a)
    end
  end

  describe "cancelling alarms, reminders and timers" do
    test "slots that identify one alarm cancel that one, by the time of day they name",
         %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6 am"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "7 am"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "6 am"}), %{}, world_b)

      assert {:ok, %{alarm_status: "cancelled", alarm_times: ["6 am"], cancelled_count: 1}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: " 6  AM "}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{time: "7 am"}] = stored(world_a, "alarm")
      assert [%Atlas.Schemas.SchedulerItem{time: "6 am"}] = stored(world_b, "alarm")
    end

    test "a time said another way names the same time of day", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "noon"}), %{}, world_a)

      assert {:ok, %{alarm_times: ["noon"], cancelled_count: 1}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: "12 pm"}), %{}, world_a)
    end

    test "with no identifying slot, every alarm of the world is cancelled", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "8am"}), %{}, world_b)

      assert {:ok, %{alarm_status: "cancelled", cancelled_count: 2}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: ""}), %{}, world_a)

      assert stored(world_a, "alarm") == []
      assert [%Atlas.Schemas.SchedulerItem{time: "8am"}] = stored(world_b, "alarm")
    end

    test "when nothing matches, the error names the kind and what was matched on, and nothing is removed",
         %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)

      assert {:error, {:no_matching_items, :alarm, %{time: "9am"}}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: "9am"}), %{}, world_a)

      assert {:error, {:no_matching_items, :timer, %{}}} = LocalScheduler.enrich("timer.cancel", said(%{}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{time: "6am"}] = stored(world_a, "alarm")
    end

    test "a reminder is identified by its text, its time, or both", %{world_a: world_a} do
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", time: "noon"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "call home", time: "6pm"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "pay rent", time: "6pm"}), %{}, world_a)

      assert {:ok, %{reminder_texts: ["water plants"], cancelled_count: 1}} =
               LocalScheduler.enrich("reminder.cancel", said(%{reminder_text: "Water Plants"}), %{}, world_a)

      assert {:error, {:no_matching_items, :reminder, %{text: "call home", time: "noon"}}} =
               LocalScheduler.enrich("reminder.cancel", said(%{reminder_text: "call home", time: "noon"}), %{}, world_a)

      assert {:ok, %{reminder_texts: ["pay rent"], reminder_times: ["6pm"], cancelled_count: 1}} =
               LocalScheduler.enrich("reminder.cancel", said(%{reminder_text: "pay rent", time: "6pm"}), %{}, world_a)

      assert {:ok, %{reminder_texts: ["call home"], reminder_count: 1}} =
               LocalScheduler.enrich("reminder.check", said(%{}), %{}, world_a)
    end

    test "a timer is identified by how long it runs, and no slot cancels every timer", %{world_a: world_a} do
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "10 minutes"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "20 minutes"}), %{}, world_a)

      assert {:ok, %{timer_status: "cancelled", timer_durations: ["10 minutes"], cancelled_count: 1}} =
               LocalScheduler.enrich("timer.cancel", said(%{duration: "10 minutes"}), %{}, world_a)

      # The same length, said another way.
      assert {:ok, %{timer_durations: ["5 minutes"], cancelled_count: 1}} =
               LocalScheduler.enrich("timer.cancel", said(%{duration: "300 seconds"}), %{}, world_a)

      assert length(stored(world_a, "timer")) == 1

      assert {:ok, %{timer_status: "cancelled", cancelled_count: 1}} =
               LocalScheduler.enrich("timer.cancel", said(%{}), %{}, world_a)

      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)
    end

    test "a slot that is not a string raises, naming the slot", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)

      assert_raise ArgumentError, ~r/the :time slot identifies alarm items by their time and must be a string/, fn ->
        LocalScheduler.cancel_alarm(said(%{time: ~T[06:00:00]}), %{}, world_a)
      end

      assert [%Atlas.Schemas.SchedulerItem{time: "6am"}] = stored(world_a, "alarm")
    end

    test "a slot that cannot be resolved raises, naming it, and nothing is removed", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)

      assert_raise ArgumentError, ~r/cannot resolve "sixish"/, fn ->
        LocalScheduler.enrich("alarm.cancel", said(%{time: "sixish"}), %{}, world_a)
      end

      assert [%Atlas.Schemas.SchedulerItem{time: "6am"}] = stored(world_a, "alarm")
    end
  end

  describe "the timer's time left" do
    test "is how much of its duration is left at the turn it is checked in", %{world_a: world_a} do
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "5 minutes", time_left: ["3 minutes", "30 seconds"]}} =
               LocalScheduler.enrich("timer.check", said(%{}, DateTime.add(@monday, 90)), %{}, world_a)

      assert {:ok, %{timer_status: "running", time_left: ["1 second"]}} =
               LocalScheduler.enrich("timer.check", said(%{}, DateTime.add(@monday, 299)), %{}, world_a)
    end

    test "is nothing once its duration has passed: the timer has finished", %{world_a: world_a} do
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)

      for seconds <- [300, 3600] do
        assert {:ok, %{timer_status: "finished", duration: "5 minutes", time_left: []}} =
                 LocalScheduler.enrich("timer.check", said(%{}, DateTime.add(@monday, seconds)), %{}, world_a)
      end
    end

    test "is not known for a timer set with no duration, or stored before durations were resolved", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", said(%{}), %{}, world_a)

      Atlas.Scheduler.insert_item!(%{
        world_id: world_b,
        kind: "timer",
        duration: "5 minutes",
        status: "running",
        resolution: "recorded_before_resolution"
      })

      restart!()

      assert {:ok, %{timer_status: "running", duration: "", time_left: []}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "5 minutes", time_left: []}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_b)
    end

    test "survives a restart", %{world_a: world_a} do
      LocalScheduler.enrich("timer.set", said(%{duration: "1 hour"}), %{}, world_a)
      restart!()

      assert {:ok, %{time_left: ["59 minutes"]}} =
               LocalScheduler.enrich("timer.check", said(%{}, DateTime.add(@monday, 60)), %{}, world_a)
    end
  end

  describe "identifying_slots/0" do
    test "names, for each kind, the field a cancel identifies an item by and the slot each is read from" do
      assert LocalScheduler.identifying_slots() == %{
               alarm: [time: :time, date: :date],
               reminder: [text: :reminder_text, time: :time, date: :date],
               timer: [duration: :duration],
               calendar: [event: :event, time: :time, date: :date]
             }
    end

    test "an item's slots are its non-blank identifying fields, each under the slot that names it" do
      assert SchedulingDriver.item_slots(LocalScheduler, :reminder, %{text: "call mom", date: "", time: "6 pm", status: "set"}) ==
               %{"reminder_text" => "call mom", "time" => "6 pm"}

      assert SchedulingDriver.item_slots(LocalScheduler, :calendar, %{event: "dentist", time: "3 pm", date: nil}) ==
               %{"event" => "dentist", "time" => "3 pm"}
    end

    test "a kind the driver identifies nothing of, or a module that is no driver, raises" do
      assert_raise ArgumentError, ~r/identifies no :weather items/, fn ->
        SchedulingDriver.item_slots(LocalScheduler, :weather, %{})
      end

      assert_raise ArgumentError, ~r/is not a scheduling driver/, fn ->
        SchedulingDriver.item_slots(Brain.Services.Weather, :calendar, %{})
      end
    end
  end

  describe "cancelling a calendar event" do
    test "with no slot that identifies an event, nothing is removed and the error names every event, to ask which",
         %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "Monday"}), %{}, world_a)

      events = [%{event: "dentist", time: "Tuesday", date: ""}, %{event: "standup", time: "Monday", date: ""}]

      assert {:error, {:event_not_identified, ^events}} = LocalScheduler.enrich("calendar.cancel", said(%{}), %{}, world_a)

      assert {:error, {:event_not_identified, ^events}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "  ", person: "John"}), %{}, world_a)

      assert length(stored(world_a, "calendar")) == 2
    end

    test "with no slot that identifies an event and a single event, the error still names it rather than removing it",
         %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)

      assert {:error, {:event_not_identified, [%{event: "dentist", time: "Tuesday", date: ""}]}} =
               LocalScheduler.enrich("calendar.cancel", said(%{}), %{}, world_a)

      assert length(stored(world_a, "calendar")) == 1
    end

    test "with no slot that identifies an event and no events, nothing matches", %{world_a: world_a} do
      assert {:error, {:no_matching_items, :calendar, %{}}} =
               LocalScheduler.enrich("calendar.cancel", said(%{}), %{}, world_a)
    end

    test "removes only the event the slots identify", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "Monday"}), %{}, world_a)

      assert {:ok, %{cancelled_event: "standup", event_time: "Monday"}} =
               LocalScheduler.enrich("calendar.cancel", said(%{time: "monday"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{text: "dentist", time: "Tuesday"}] = stored(world_a, "calendar")
    end

    test "when several events match, nothing is removed and the error names each", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "Monday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "Thursday"}), %{}, world_a)

      assert {:error, {:ambiguous, :calendar, events}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "standup"}), %{}, world_a)

      assert Enum.sort_by(events, & &1.time) == [
               %{event: "standup", time: "Monday", date: ""},
               %{event: "standup", time: "Thursday", date: ""}
             ]

      assert length(stored(world_a, "calendar")) == 2

      assert {:ok, %{cancelled_event: "standup", event_time: "Thursday"}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "standup", time: "Thursday"}), %{}, world_a)
    end

    test "when no event matches, nothing is removed", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)

      assert {:error, {:no_matching_items, :calendar, %{event: "lunch"}}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "lunch"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{text: "dentist"}] = stored(world_a, "calendar")
    end
  end

  describe "querying the calendar" do
    setup %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "3 pm", date: "Friday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "9 am", date: "tomorrow"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "review", date: "yesterday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "party"}), %{}, world_a)
      :ok
    end

    test "reports the soonest event still to come, whatever order they were set in", %{world_a: world_a} do
      assert LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a) ==
               {:ok, %{next_event: "standup", event_time: "9 am", event_date: "tomorrow", event_count: 4, upcoming_count: 2}}
    end

    test "is relative to the turn: once an event has passed, the next one is the soonest", %{world_a: world_a} do
      wednesday = ~U[2026-10-14 09:00:00Z]

      assert {:ok, %{next_event: "dentist", event_date: "Friday", upcoming_count: 1}} =
               LocalScheduler.enrich("calendar.query", said(%{}, wednesday), %{}, world_a)

      saturday = ~U[2026-10-17 09:00:00Z]

      assert LocalScheduler.enrich("calendar.query", said(%{}, saturday), %{}, world_a) ==
               {:ok, %{next_event: "", event_time: "", event_date: "", event_count: 4, upcoming_count: 0}}
    end

    test "narrowed to a day, reports that day's events", %{world_a: world_a} do
      assert {:ok, %{next_event: "dentist", event_count: 1, upcoming_count: 1}} =
               LocalScheduler.enrich("calendar.query", said(%{date: "friday"}), %{}, world_a)

      assert {:ok, %{next_event: "", event_count: 1, upcoming_count: 0}} =
               LocalScheduler.enrich("calendar.query", said(%{date: "yesterday"}), %{}, world_a)
    end

    test "an event stored before resolutions were has no time to order by, and is never the next",
         %{world_b: world_b} do
      Atlas.Scheduler.insert_item!(%{
        world_id: world_b,
        kind: "calendar",
        text: "dentist",
        date: "tomorrow",
        time: "",
        resolution: "recorded_before_resolution"
      })

      restart!()

      assert LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_b) ==
               {:ok, %{next_event: "", event_time: "", event_date: "", event_count: 1, upcoming_count: 0}}
    end
  end

  describe "dates" do
    test "an alarm, a reminder and a calendar item are stored with the date they were set for",
         %{world_a: world_a} do
      assert {:ok, %{alarm_time: "7am", alarm_date: "tomorrow"}} =
               LocalScheduler.enrich("alarm.set", said(%{time: "7am", date: "tomorrow"}), %{}, world_a)

      assert {:ok, %{reminder_text: "water plants", reminder_date: "Friday"}} =
               LocalScheduler.enrich(
                 "reminder.create",
                 said(%{reminder_text: "water plants", time: "noon", date: "Friday"}),
                 %{},
                 world_a
               )

      assert {:ok, %{last_set_event: "dentist", event_date: "March 5"}} =
               LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "3pm", date: "March 5"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{date: "tomorrow"}] = stored(world_a, "alarm")
      assert [%Atlas.Schemas.SchedulerItem{date: "Friday"}] = stored(world_a, "reminder")
      assert [%Atlas.Schemas.SchedulerItem{date: "March 5", starts_at: ~U[2027-03-05 15:00:00.000000Z]}] = stored(world_a, "calendar")
    end

    test "with no date slot the date is stored empty, and a timer has no date", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{date: ""}] = stored(world_a, "alarm")
      assert [%Atlas.Schemas.SchedulerItem{date: nil}] = stored(world_a, "timer")
    end

    test "a reminder's text is read from the reminder_text slot under either key", %{world_a: world_a} do
      LocalScheduler.enrich("reminder.create", said(%{"reminder_text" => "call home", "time" => "6pm"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{text: "call home", time: "6pm"}] = stored(world_a, "reminder")
    end

    test "a cancel naming a day removes only that day's items, and leaves other days' and undated ones",
         %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", date: "tomorrow"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "pay rent", date: "Tomorrow"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "call home", date: "Friday"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "buy milk"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "feed the cat", date: "tomorrow"}), %{}, world_b)

      assert {:ok, %{reminder_texts: texts, reminder_dates: dates, cancelled_count: 2}} =
               LocalScheduler.enrich("reminder.cancel", said(%{date: "tomorrow"}), %{}, world_a)

      assert Enum.sort(texts) == ["pay rent", "water plants"]
      assert Enum.sort(dates) == ["Tomorrow", "tomorrow"]
      assert world_a |> stored("reminder") |> Enum.map(& &1.text) |> Enum.sort() == ["buy milk", "call home"]
      assert [%Atlas.Schemas.SchedulerItem{text: "feed the cat"}] = stored(world_b, "reminder")
    end

    test "a day is the one it named when it was said: tomorrow said Monday is not tomorrow said Tuesday",
         %{world_a: world_a} do
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", date: "tomorrow"}, @monday), %{}, world_a)

      assert {:ok, %{reminder_count: 0}} =
               LocalScheduler.enrich("reminder.check", said(%{date: "tomorrow"}, @tuesday), %{}, world_a)

      assert {:error, {:no_matching_items, :reminder, %{date: "tomorrow"}}} =
               LocalScheduler.enrich("reminder.cancel", said(%{date: "tomorrow"}, @tuesday), %{}, world_a)

      assert {:ok, %{reminder_texts: ["water plants"], reminder_count: 1}} =
               LocalScheduler.enrich("reminder.check", said(%{date: "today"}, @tuesday), %{}, world_a)

      assert {:ok, %{reminder_texts: ["water plants"], cancelled_count: 1}} =
               LocalScheduler.enrich("reminder.cancel", said(%{date: "Tuesday"}, @monday), %{}, world_a)
    end

    test "a day narrows a cancel together with the other identifying slots", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "7am", date: "Monday"}), %{}, world_a)
      LocalScheduler.enrich("alarm.set", said(%{time: "7am", date: "Tuesday"}), %{}, world_a)

      assert {:ok, %{alarm_times: ["7am"], alarm_dates: ["Tuesday"], cancelled_count: 1}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: "7am", date: "tuesday"}), %{}, world_a)

      assert {:error, {:no_matching_items, :alarm, %{time: "7am", date: "friday"}}} =
               LocalScheduler.enrich("alarm.cancel", said(%{time: "7am", date: "Friday"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{date: "Monday"}] = stored(world_a, "alarm")
    end

    test "a calendar cancel naming a day tells apart events that differ only by day", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "9am", date: "Monday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "9am", date: "Thursday"}), %{}, world_a)

      assert {:ok, %{cancelled_event: "standup", event_date: "Thursday", cancelled_count: 1}} =
               LocalScheduler.enrich("calendar.cancel", said(%{event: "standup", date: "Thursday"}), %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{date: "Monday"}] = stored(world_a, "calendar")
    end

    test "a check naming a day reports only that day's items; with no day, every one", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", said(%{time: "6am", date: "tomorrow"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("alarm.set", said(%{time: "9pm", date: "Friday"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", date: "tomorrow"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "call home", date: "Friday"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", date: "Friday"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", date: "Monday"}), %{}, world_a)

      assert {:ok, %{alarm_status: "set", alarm_times: ["6am"], alarm_dates: ["tomorrow"], alarm_count: 1}} =
               LocalScheduler.enrich("alarm.check", said(%{date: "Tomorrow"}), %{}, world_a)

      assert {:ok, %{alarm_status: "none", alarm_times: [], alarm_count: 0}} =
               LocalScheduler.enrich("alarm.check", said(%{date: "Sunday"}), %{}, world_a)

      assert {:ok, %{alarm_times: ["6am", "9pm"], alarm_count: 2}} =
               LocalScheduler.enrich("alarm.check", said(%{}), %{}, world_a)

      assert {:ok, %{reminder_texts: ["call home"], reminder_dates: ["Friday"], reminder_count: 1}} =
               LocalScheduler.enrich("reminder.check", said(%{date: "friday"}), %{}, world_a)

      assert {:ok, %{next_event: "dentist", event_date: "Friday", event_count: 1}} =
               LocalScheduler.enrich("calendar.query", said(%{date: "Friday"}), %{}, world_a)

      # Monday is today, so standup, set last, is the soonest.
      assert {:ok, %{next_event: "standup", event_date: "Monday", event_count: 2}} =
               LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)

      assert {:ok, %{next_event: "", event_date: "", event_count: 0}} =
               LocalScheduler.enrich("calendar.query", said(%{date: "Sunday"}), %{}, world_a)
    end

    test "an item stored before resolutions were matches no day, and is still cancelled by a cancel naming none",
         %{world_a: world_a} do
      # Written straight to Atlas, as an item stored before then reads back,
      # and loaded by restarting the scheduler.
      Atlas.Scheduler.insert_item!(%{
        world_id: world_a,
        kind: "reminder",
        text: "water plants",
        time: "noon",
        status: "set",
        resolution: "recorded_before_resolution"
      })

      restart!()

      assert [%Atlas.Schemas.SchedulerItem{date: nil, resolution: "recorded_before_resolution"}] = stored(world_a, "reminder")

      assert {:ok, %{reminder_count: 0}} = LocalScheduler.enrich("reminder.check", said(%{date: "tomorrow"}), %{}, world_a)

      assert {:error, {:no_matching_items, :reminder, %{date: "tomorrow"}}} =
               LocalScheduler.enrich("reminder.cancel", said(%{date: "tomorrow"}), %{}, world_a)

      assert {:error, {:no_matching_items, :reminder, %{time: "noon"}}} =
               LocalScheduler.enrich("reminder.cancel", said(%{time: "noon"}), %{}, world_a)

      assert {:ok, %{reminder_texts: ["water plants"], reminder_dates: [nil], cancelled_count: 1}} =
               LocalScheduler.enrich("reminder.cancel", said(%{reminder_text: "water plants"}), %{}, world_a)
    end

    test "a date slot that is not a string raises, naming the slot", %{world_a: world_a} do
      assert_raise ArgumentError, ~r/the :date slot identifies alarm items by their date and must be a string/, fn ->
        LocalScheduler.check_alarms(said(%{date: ~D[2026-10-14]}), %{}, world_a)
      end
    end
  end

  describe "writes?/1" do
    test "is true for setting, creating, scheduling and cancelling, and false for reading and snoozing" do
      for intent <- [
            "alarm.set",
            "alarm.cancel",
            "timer.set",
            "timer.cancel",
            "reminder.create",
            "reminder.cancel",
            "calendar.schedule",
            "calendar.cancel"
          ] do
        assert LocalScheduler.writes?(intent), intent
      end

      for intent <- ["alarm.check", "timer.check", "reminder.check", "calendar.query", "alarm.snooze"] do
        refute LocalScheduler.writes?(intent), intent
      end
    end

    test "is decided by the operation the intent names, not by a part of its name" do
      refute LocalScheduler.writes?("alarm.set.recurring")
      refute LocalScheduler.writes?("my.alarm.set")
    end
  end

  describe "a restart" do
    test "keeps every item, each in the world it was set in, with what it resolved to",
         %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", said(%{duration: "3 minutes"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "standup", time: "Monday"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_b)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_b)

      before = Process.whereis(LocalScheduler)
      :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
      assert :ets.whereis(:local_scheduler_items) == :undefined

      {:ok, restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)
      assert restarted != before

      assert {:ok, %{timer_status: "running", duration: "3 minutes", time_left: ["3 minutes"]}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)

      assert {:ok, %{next_event: "standup", event_time: "Monday"}} =
               LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "5 minutes"}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_b)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_b)

      # What tomorrow named on Monday is still Tuesday after the restart.
      assert {:ok, %{cancelled_event: "dentist"}} =
               LocalScheduler.enrich("calendar.cancel", said(%{date: "today"}, @tuesday), %{}, world_b)
    end

    test "keeps a cancellation, and the items a targeted cancel left", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)
      {:ok, _} = LocalScheduler.enrich("calendar.cancel", said(%{event: "dentist"}), %{}, world_a)

      LocalScheduler.enrich("alarm.set", said(%{time: "6am"}), %{}, world_a)
      Process.sleep(2)
      LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)
      {:ok, _} = LocalScheduler.enrich("alarm.cancel", said(%{time: "6am"}), %{}, world_a)

      restart!()

      assert {:ok, %{event_count: 0}} = LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)
      assert {:ok, %{alarm_times: ["7am"], alarm_count: 1}} = LocalScheduler.enrich("alarm.check", said(%{}), %{}, world_a)
    end
  end

  describe "dispatched by Brain.Services.Dispatcher" do
    setup %{world_a: world_a, world_b: world_b} do
      :ok = Dispatcher.assign_scheduling_driver(world_a, LocalScheduler)
      :ok = Dispatcher.assign_scheduling_driver(world_b, LocalScheduler)
      :ok
    end

    test "a scheduler intent sets and reads back the item in the context's world, at the context's turn",
         %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{timer_status: "running", duration: "7 minutes"}} =
               Dispatcher.dispatch("timer.set", %{duration: "7 minutes"}, %{world_id: world_a, turn_time: @monday})

      assert {:ok, %{timer_status: "running", duration: "7 minutes", time_left: ["6 minutes"]}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: world_a, turn_time: DateTime.add(@monday, 60)})

      assert {:ok, %{timer_status: "none"}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: world_b, turn_time: @monday})

      assert {:ok, %{timer_status: "running", duration: "7 minutes"}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)
    end

    test "a context without the turn's time raises", %{world_a: world_a} do
      assert_raise ArgumentError, ~r/context needs :turn_time/, fn ->
        Dispatcher.dispatch("timer.set", %{duration: "7 minutes"}, %{world_id: world_a})
      end

      assert stored(world_a, "timer") == []
    end

    test "with side effects off, a setting intent is refused and nothing is stored", %{world_a: world_a} do
      assert {:error, :side_effects_off} =
               Dispatcher.dispatch("timer.set", %{duration: "5 minutes"}, %{
                 world_id: world_a,
                 side_effects: false,
                 turn_time: @monday
               })

      assert {:ok, %{timer_status: "none"}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: world_a, side_effects: false, turn_time: @monday})
    end
  end

  describe "clear_world/1" do
    test "removes every item of the world and leaves other worlds'", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", said(%{time: "7am"}), %{}, world_a)
      LocalScheduler.enrich("reminder.create", said(%{reminder_text: "water plants", time: "noon"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "5 minutes"}), %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", said(%{event: "dentist", time: "Tuesday"}), %{}, world_a)
      LocalScheduler.enrich("timer.set", said(%{duration: "8 minutes"}), %{}, world_b)

      assert :ok = LocalScheduler.clear_world(world_a)

      for kind <- Atlas.Schemas.SchedulerItem.kinds() do
        assert stored(world_a, kind) == [], "#{kind} items of #{world_a} are still stored"
      end

      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", said(%{}), %{}, world_a)

      assert {:ok, %{event_count: 0}} = LocalScheduler.enrich("calendar.query", said(%{}), %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "8 minutes"}} =
               LocalScheduler.enrich("timer.check", said(%{}), %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.clear_world needs the world/, fn ->
        LocalScheduler.clear_world(nil)
      end
    end
  end
end
