defmodule Brain.Services.LocalSchedulerTest do
  use ExUnit.Case, async: false

  alias Brain.Services.LocalScheduler

  # The scheduler keys every item by world id; it reads no world directory,
  # so these tests use fresh `test_` ids and cancel what they set.
  setup do
    suffix = System.unique_integer([:positive])
    world_a = "test_scheduler_a_#{suffix}"
    world_b = "test_scheduler_b_#{suffix}"

    on_exit(fn ->
      for world <- [world_a, world_b],
          intent <- ["alarm.cancel", "reminder.cancel", "timer.cancel"] do
        LocalScheduler.enrich(intent, %{}, %{}, world)
      end
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
end
