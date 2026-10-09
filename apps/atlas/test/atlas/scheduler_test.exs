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
end
