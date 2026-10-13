defmodule Atlas.SchedulerTest do
  @moduledoc """
  The store behind the local scheduler: an item is read back with the world and
  kind it was stored with, deleting removes only one world's items of one kind,
  and an item without a world or with an unknown kind is refused.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Scheduler
  alias Atlas.Schemas.SchedulerItem

  @world_a "test_scheduler_store_a"
  @world_b "test_scheduler_store_b"

  defp items_of(world_id), do: Enum.filter(Scheduler.list_items(), &(&1.world_id == world_id))

  describe "insert_item!/1 and list_items/0" do
    test "reads back what was stored, oldest first" do
      Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})

      Scheduler.insert_item!(%{
        world_id: @world_a,
        kind: "reminder",
        text: "water plants",
        time: "noon",
        status: "set"
      })

      assert [
               %SchedulerItem{kind: "alarm", time: "7am", status: "set", text: nil},
               %SchedulerItem{kind: "reminder", text: "water plants", time: "noon", status: "set"}
             ] = items_of(@world_a)
    end

    test "reads back the date an alarm, reminder or calendar item was stored for, and nil for one stored without" do
      Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am", date: "tomorrow", status: "set"})
      Process.sleep(1)

      Scheduler.insert_item!(%{
        world_id: @world_a,
        kind: "reminder",
        text: "water plants",
        time: "noon",
        date: "Friday",
        status: "set"
      })

      Process.sleep(1)
      Scheduler.insert_item!(%{world_id: @world_a, kind: "calendar", text: "dentist", time: "3pm", date: "March 5"})
      Process.sleep(1)
      Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})

      assert [
               %SchedulerItem{kind: "alarm", date: "tomorrow"},
               %SchedulerItem{kind: "reminder", date: "Friday"},
               %SchedulerItem{kind: "calendar", date: "March 5"},
               %SchedulerItem{kind: "timer", date: nil}
             ] = items_of(@world_a)
    end

    test "the date column is nullable and has no default, so no date is ever assumed" do
      %{rows: [[nullable, default]]} =
        Atlas.Repo.query!(
          """
          SELECT is_nullable, column_default FROM information_schema.columns
          WHERE table_schema = current_schema() AND table_name = 'atlas_scheduler_items' AND column_name = 'date'
          """,
          []
        )

      assert nullable == "YES"
      assert default == nil
    end

    test "keeps an empty field as the empty string it was given" do
      Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "", status: "set"})

      assert [%SchedulerItem{kind: "alarm", time: "", status: "set"}] = items_of(@world_a)
    end

    test "refuses an item without a world, with a blank world, or of an unknown kind" do
      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{world_id: "", kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{world_id: "   ", kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{world_id: @world_a, kind: "appointment", time: "7am"})
      end

      assert items_of(@world_a) == []
    end
  end

  describe "delete_items!/2" do
    test "deletes one world's items of one kind and keeps the rest" do
      Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})
      Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "9 minutes", status: "running"})
      Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})
      Scheduler.insert_item!(%{world_id: @world_b, kind: "timer", duration: "10 minutes", status: "running"})

      assert Scheduler.delete_items!(@world_a, "timer") == 2

      assert [%SchedulerItem{kind: "alarm"}] = items_of(@world_a)
      assert [%SchedulerItem{kind: "timer", duration: "10 minutes"}] = items_of(@world_b)
    end
  end

  describe "delete_items_by_id!/3" do
    test "deletes only the items named, and keeps the world's others" do
      five = Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})
      Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "9 minutes", status: "running"})

      assert Scheduler.delete_items_by_id!(@world_a, "timer", [five.id]) == 1

      assert [%SchedulerItem{kind: "timer", duration: "9 minutes"}] = items_of(@world_a)
    end

    test "deletes nothing and raises when an id names another world's item or another kind" do
      own = Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})
      other_world = Scheduler.insert_item!(%{world_id: @world_b, kind: "alarm", time: "8am", status: "set"})
      other_kind = Scheduler.insert_item!(%{world_id: @world_a, kind: "reminder", text: "call home", status: "set"})

      assert_raise ArgumentError, ~r/only 1 of the ids/, fn ->
        Scheduler.delete_items_by_id!(@world_a, "alarm", [own.id, other_world.id])
      end

      assert_raise ArgumentError, ~r/only 1 of the ids/, fn ->
        Scheduler.delete_items_by_id!(@world_a, "alarm", [own.id, other_kind.id])
      end

      assert length(items_of(@world_a)) == 2
      assert [%SchedulerItem{time: "8am"}] = items_of(@world_b)
    end
  end
end
