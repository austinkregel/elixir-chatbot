defmodule Brain.Services.LocalSchedulerTest do
  use ExUnit.Case, async: false

  alias Brain.Services.LocalScheduler

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

  describe "enrich/4 world scoping" do
    test "an alarm set in one world is not seen in another", %{world_a: world_a, world_b: world_b} do
      assert {:ok, %{alarm_status: "set", alarm_time: "7am"}} =
               LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)

      assert {:ok, %{alarm_status: "set", alarm_time: "7am"}} =
               LocalScheduler.enrich("alarm.check", %{}, %{}, world_a)

      assert {:ok, %{alarm_status: "none set", alarm_time: ""}} =
               LocalScheduler.enrich("alarm.check", %{}, %{}, world_b)
    end

    test "each world reads back its own reminder", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("reminder.set", %{reminder_text: "water plants", time: "noon"}, %{}, world_a)
      LocalScheduler.enrich("reminder.set", %{reminder_text: "call home", time: "6pm"}, %{}, world_b)

      assert {:ok, %{reminder_set: true, reminder_text: "water plants", reminder_time: "noon"}} =
               LocalScheduler.enrich("reminder.check", %{}, %{}, world_a)

      assert {:ok, %{reminder_set: true, reminder_text: "call home", reminder_time: "6pm"}} =
               LocalScheduler.enrich("reminder.check", %{}, %{}, world_b)
    end

    test "cancelling in one world leaves another world's items", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("timer.set", %{duration: "5 minutes"}, %{}, world_a)
      LocalScheduler.enrich("timer.set", %{duration: "10 minutes"}, %{}, world_b)

      assert {:ok, %{timer_status: "cancelled"}} =
               LocalScheduler.enrich("timer.cancel", %{}, %{}, world_a)

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

    test "a calendar item is read only in the world it was created in", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("calendar.create", %{event: "dentist", time: "Tuesday"}, %{}, world_a)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{next_event: "No upcoming events", event_time: ""}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, nil)
      end

      assert_raise ArgumentError, ~r/LocalScheduler.enrich needs the world/, fn ->
        LocalScheduler.enrich("alarm.check", %{}, %{}, "")
      end
    end
  end

  describe "a restart" do
    test "keeps every item, each in the world it was set in", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)
      LocalScheduler.enrich("reminder.set", %{reminder_text: "water plants", time: "noon"}, %{}, world_a)
      LocalScheduler.enrich("timer.set", %{duration: "5 minutes"}, %{}, world_b)
      LocalScheduler.enrich("calendar.create", %{event: "dentist", time: "Tuesday"}, %{}, world_b)

      before = Process.whereis(LocalScheduler)
      :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
      assert :ets.whereis(:local_scheduler_items) == :undefined

      {:ok, restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)
      assert restarted != before

      assert {:ok, %{alarm_status: "set", alarm_time: "7am"}} =
               LocalScheduler.enrich("alarm.check", %{}, %{}, world_a)

      assert {:ok, %{reminder_set: true, reminder_text: "water plants", reminder_time: "noon"}} =
               LocalScheduler.enrich("reminder.check", %{}, %{}, world_a)

      assert {:ok, %{timer_status: "running", duration: "5 minutes"}} =
               LocalScheduler.enrich("timer.check", %{}, %{}, world_b)

      assert {:ok, %{next_event: "dentist", event_time: "Tuesday"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_b)

      assert {:ok, %{alarm_status: "none set"}} = LocalScheduler.enrich("alarm.check", %{}, %{}, world_b)
      assert {:ok, %{timer_status: "none"}} = LocalScheduler.enrich("timer.check", %{}, %{}, world_a)
    end

    test "keeps a cancellation", %{world_a: world_a} do
      LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)
      LocalScheduler.enrich("alarm.cancel", %{}, %{}, world_a)

      :ok = Supervisor.terminate_child(Brain.Supervisor, LocalScheduler)
      {:ok, _restarted} = Supervisor.restart_child(Brain.Supervisor, LocalScheduler)

      assert {:ok, %{alarm_status: "none set"}} = LocalScheduler.enrich("alarm.check", %{}, %{}, world_a)
    end
  end

  describe "clear_world/1" do
    test "removes every item of the world and leaves other worlds'", %{world_a: world_a, world_b: world_b} do
      LocalScheduler.enrich("alarm.set", %{time: "7am"}, %{}, world_a)
      LocalScheduler.enrich("calendar.create", %{event: "dentist", time: "Tuesday"}, %{}, world_a)
      LocalScheduler.enrich("alarm.set", %{time: "8am"}, %{}, world_b)

      assert :ok = LocalScheduler.clear_world(world_a)

      assert {:ok, %{alarm_status: "none set"}} = LocalScheduler.enrich("alarm.check", %{}, %{}, world_a)

      assert {:ok, %{next_event: "No upcoming events"}} =
               LocalScheduler.enrich("calendar.query", %{}, %{}, world_a)

      assert {:ok, %{alarm_status: "set", alarm_time: "8am"}} =
               LocalScheduler.enrich("alarm.check", %{}, %{}, world_b)
    end

    test "raises without a world" do
      assert_raise ArgumentError, ~r/LocalScheduler.clear_world needs the world/, fn ->
        LocalScheduler.clear_world(nil)
      end
    end
  end
end
