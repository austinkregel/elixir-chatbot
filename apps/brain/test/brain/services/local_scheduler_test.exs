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

  # What Atlas holds of the world's items of one kind, oldest first: the
  # alarms and reminders no operation reads back.
  defp stored(world_id, kind) do
    Enum.filter(Atlas.Scheduler.list_items(), &(&1.world_id == world_id and &1.kind == kind))
  end

  describe "supported_intents/0" do
    test "is every intent the classifier emits in the scheduling domains, and nothing else" do
      assert LocalScheduler.supported_intents() ==
               Enum.sort([
                 "alarm.set",
                 "alarm.snooze",
                 "timer.set",
                 "timer.check",
                 "reminder.create",
                 "calendar.query",
                 "calendar.schedule",
                 "calendar.cancel"
               ])
    end

    test "each is an intent in the intent registry" do
      registry =
        "analysis/intent_registry.json"
        |> Brain.priv_path()
        |> File.read!()
        |> Jason.decode!()

      for intent <- LocalScheduler.supported_intents() do
        assert Map.has_key?(registry, intent), "#{intent} is not in the intent registry"
      end
    end
  end

  describe "enrich/4 reads the operation from the intent's domain and verb" do
    test "an intent that is not exactly a domain and an operation's verb is refused, and nothing is stored",
         %{world_a: world_a} do
      for intent <- ["alarm.set.recurring", "reminder.set", "calendar.create", "alarm.settings", "my.alarm.set"] do
        assert {:error, :unsupported_intent} = LocalScheduler.enrich(intent, %{time: "7am"}, %{}, world_a)
      end

      assert stored(world_a, "alarm") == []
      assert stored(world_a, "reminder") == []
      assert stored(world_a, "calendar") == []
    end

    test "every intent it supports names an operation, and none is refused as unsupported", %{world_a: world_a} do
      for intent <- LocalScheduler.supported_intents() do
        assert {:ok, _operation} = SchedulingDriver.operation(intent)
        refute LocalScheduler.enrich(intent, %{}, %{}, world_a) == {:error, :unsupported_intent}, intent
      end
    end

    test "a read operation answers the same through enrich/4 as called directly", %{world_a: world_a} do
      LocalScheduler.set_timer(%{duration: "4 minutes"}, %{}, world_a)
      LocalScheduler.schedule_event(%{event: "dentist", time: "Tuesday"}, %{}, world_a)

      assert LocalScheduler.enrich("timer.check", %{}, %{}, world_a) == LocalScheduler.check_timer(%{}, %{}, world_a)

      assert LocalScheduler.enrich("calendar.query", %{}, %{}, world_a) ==
               LocalScheduler.query_calendar(%{}, %{}, world_a)

      assert LocalScheduler.enrich("alarm.snooze", %{}, %{}, world_a) == LocalScheduler.snooze_alarm(%{}, %{}, world_a)
    end
  end

  describe "operations, each in its world" do
    test "an alarm set in one world is stored in that world only", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{alarm_status: "set", alarm_time: "7am"}} =
               LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)

      assert [%Atlas.Schemas.SchedulerItem{time: "7am", status: "set"}] = stored(world_a, "alarm")
      assert stored(world_b, "alarm") == []
    end

    test "snoozing finds no ringing alarm, since alarms are stored and never rung", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)

      assert {:error, :no_ringing_alarm} = LocalScheduler.enrich("alarm.snooze", %{duration: "5 minutes"}, %{}, world_a)
      assert [%Atlas.Schemas.SchedulerItem{time: "7am", status: "set"}] = stored(world_a, "alarm")
    end

    test "each world stores its own reminder", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{reminder_set: true, reminder_text: "water plants", reminder_time: "noon"}} =
               LocalScheduler.enrich("reminder.create", %{reminder_text: "water plants", time: "noon"}, %{}, world_a)

      assert {:ok, %{reminder_set: true, reminder_text: "call home", reminder_time: "6pm"}} =
               LocalScheduler.enrich("reminder.create", %{reminder_text: "call home", time: "6pm"}, %{}, world_b)

      assert [%Atlas.Schemas.SchedulerItem{text: "water plants", time: "noon", status: "set"}] =
               stored(world_a, "reminder")

      assert [%Atlas.Schemas.SchedulerItem{text: "call home", time: "6pm", status: "set"}] =
               stored(world_b, "reminder")
    end

    test "a timer is read back only in the world it was set in", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", %{duration: "10 minutes"}, %{}, world_b)

      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", %{}, %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "10 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_b)
    end

    test "with two running timers, check reports the one set last", %{world_a: world_a} do
      assert {:ok, %{timer_status: "running"}} =
               LocalScheduler.enrich("timer.set", %{duration: "5 minutes"}, %{}, world_a)

      # Timers are ordered by the microsecond they were set; a millisecond
      # apart orders these two.
      Process.sleep(2)

      assert {:ok, %{timer_status: "running"}} =
               LocalScheduler.enrich("timer.set", %{duration: "10 minutes"}, %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "10 minutes", time_remaining: "10 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_a)
    end

    test "a scheduled event is read only in the world it was scheduled in", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.schedule", %{event: "dentist", time: "Tuesday"}, %{}, world_a)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{next_event: "No upcoming events", event_time: ""}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_b)
    end

    test "cancelling in one world takes its events off its calendar and leaves another world's",
         %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("calendar.schedule", %{event: "dentist", time: "Tuesday"}, %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", %{event: "standup", time: "Monday"}, %{}, world_b)

      assert {:ok, %{next_event: "", event_time: ""}} = LocalScheduler.enrich("calendar.cancel", %{}, %{}, world_a)

      assert {:ok, %{next_event: "No upcoming events"}} = LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{next_event: "standup", event_time: "Monday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, nil)
      end

      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("timer.check", %{}, %{}, "")
      end

      assert_raise ArgumentError, ~r/LocalScheduler.set_alarm needs the world/, fn ->
        LocalScheduler.set_alarm(%{time: "7am"}, %{}, nil)
      end

      assert_raise ArgumentError, ~r/LocalScheduler.query_calendar needs the world/, fn ->
        LocalScheduler.query_calendar(%{}, %{}, "")
      end
    end
  end

  describe "writes?/1" do
    test "is true for setting, creating, scheduling and cancelling, and false for reading and snoozing" do
      assert LocalScheduler.writes?("alarm.set")
      assert LocalScheduler.writes?("timer.set")
      assert LocalScheduler.writes?("reminder.create")
      assert LocalScheduler.writes?("calendar.schedule")
      assert LocalScheduler.writes?("calendar.cancel")

      refute LocalScheduler.writes?("timer.check")
      refute LocalScheduler.writes?("calendar.query")
      refute LocalScheduler.writes?("alarm.snooze")
    end

    test "is decided by the operation the intent names, not by a part of its name" do
      refute LocalScheduler.writes?("alarm.set.recurring")
      refute LocalScheduler.writes?("my.alarm.set")
    end
  end

  describe "a restart" do
    test "keeps every item, each in the world it was set in", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", %{duration: "3 minutes"}, %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", %{event: "standup", time: "Monday"}, %{}, world_a)
      LocalScheduler.enrich("timer.set", %{duration: "5 minutes"}, %{}, world_b)
      LocalScheduler.enrich("calendar.schedule", %{event: "dentist", time: "Tuesday"}, %{}, world_b)

      before = Process.whereis(LocalScheduler)
      :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
      assert :ets.whereis(:local_scheduler_items) == :undefined

      {:ok, restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)
      assert restarted != before

      assert {:ok, %{timer_status: "running", duration: "3 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_a)

      assert {:ok, %{next_event: "standup", event_time: "Monday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "5 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_b)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_b)
    end

    test "keeps a cancellation", %{world_a: world_a} do
      LocalScheduler.enrich("calendar.schedule", %{event: "dentist", time: "Tuesday"}, %{}, world_a)
      LocalScheduler.enrich("calendar.cancel", %{}, %{}, world_a)

      :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
      {:ok, _restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)

      assert {:ok, %{next_event: "No upcoming events"}} = LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)
    end
  end

  describe "dispatched by Brain.Services.Dispatcher" do
    setup %{world_a: world_a, world_b: world_b} do
      :ok = Dispatcher.assign_scheduling_driver(world_a, LocalScheduler)
      :ok = Dispatcher.assign_scheduling_driver(world_b, LocalScheduler)
      :ok
    end

    test "a scheduler intent sets and reads back the item in the context's world", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{timer_status: "running", duration: "7 minutes"}} =
               Dispatcher.dispatch("timer.set", %{duration: "7 minutes"}, %{world_id: world_a})

      assert {:ok, %{timer_status: "running", duration: "7 minutes"}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: world_a})

      assert {:ok, %{timer_status: "none"}} = Dispatcher.dispatch("timer.check", %{}, %{world_id: world_b})

      assert {:ok, %{timer_status: "running", duration: "7 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_a)
    end

    test "with side effects off, a setting intent is refused and nothing is stored", %{world_a: world_a} do
      assert {:error, :side_effects_off} =
               Dispatcher.dispatch("timer.set", %{duration: "5 minutes"}, %{
                 world_id: world_a,
                 side_effects: false
               })

      assert {:ok, %{timer_status: "none"}} =
               Dispatcher.dispatch("timer.check", %{}, %{world_id: world_a, side_effects: false})
    end
  end

  describe "clear_world/1" do
    test "removes every item of the world and leaves other worlds'", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)
      LocalScheduler.enrich("reminder.create", %{reminder_text: "water plants", time: "noon"}, %{}, world_a)
      LocalScheduler.enrich("timer.set", %{duration: "5 minutes"}, %{}, world_a)
      LocalScheduler.enrich("calendar.schedule", %{event: "dentist", time: "Tuesday"}, %{}, world_a)
      LocalScheduler.enrich("timer.set", %{duration: "8 minutes"}, %{}, world_b)

      assert :ok = LocalScheduler.clear_world(world_a)

      for kind <- Atlas.Schemas.SchedulerItem.kinds() do
        assert stored(world_a, kind) == [], "#{kind} items of #{world_a} are still stored"
      end

      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", %{}, %{}, world_a)

      assert {:ok, %{next_event: "No upcoming events"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "8 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.clear_world needs the world/, fn ->
        LocalScheduler.clear_world(nil)
      end
    end
  end
end
