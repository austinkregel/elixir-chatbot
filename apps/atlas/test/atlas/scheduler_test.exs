defmodule Atlas.SchedulerTest do
  @moduledoc """
  The store behind the local scheduler: an item is read back with the world and
  kind it was stored with, and with its resolved time when it has one;
  deleting removes only one world's items of one kind; and an item without a
  world, with an unknown kind, or without its resolution is refused.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Scheduler
  alias Atlas.Schemas.SchedulerItem

  @world_a "test_scheduler_store_a"
  @world_b "test_scheduler_store_b"

  @migration_version 20261014120000

  defp items_of(world_id), do: Enum.filter(Scheduler.list_items(), &(&1.world_id == world_id))

  # An item stored as text only, as every item was before resolutions were.
  defp text_item!(attrs), do: Scheduler.insert_item!(Map.put(attrs, :resolution, "recorded_before_resolution"))

  @resolved %{
    resolution: "resolved",
    turn_time: ~U[2026-10-12 09:00:00.000000Z],
    time_zone: "Etc/UTC",
    starts_at: ~U[2026-10-13 07:00:00.000000Z],
    ends_at: ~U[2026-10-13 07:00:00.000000Z],
    clock: ~T[07:00:00]
  }

  defp column(name) do
    %{rows: [[nullable, default]]} =
      Atlas.Repo.query!(
        """
        SELECT is_nullable, column_default FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'atlas_scheduler_items' AND column_name = $1
        """,
        [name]
      )

    {nullable, default}
  end

  describe "insert_item!/1 and list_items/0" do
    test "reads back what was stored, oldest first" do
      text_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})

      text_item!(%{
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
      text_item!(%{world_id: @world_a, kind: "alarm", time: "7am", date: "tomorrow", status: "set"})
      Process.sleep(1)

      text_item!(%{
        world_id: @world_a,
        kind: "reminder",
        text: "water plants",
        time: "noon",
        date: "Friday",
        status: "set"
      })

      Process.sleep(1)
      text_item!(%{world_id: @world_a, kind: "calendar", text: "dentist", time: "3pm", date: "March 5"})
      Process.sleep(1)
      text_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})

      assert [
               %SchedulerItem{kind: "alarm", date: "tomorrow"},
               %SchedulerItem{kind: "reminder", date: "Friday"},
               %SchedulerItem{kind: "calendar", date: "March 5"},
               %SchedulerItem{kind: "timer", date: nil}
             ] = items_of(@world_a)
    end

    test "reads back a resolved item's turn, zone, start, end and clock beside the text it was said as" do
      Scheduler.insert_item!(Map.merge(@resolved, %{world_id: @world_a, kind: "alarm", time: "7 am", date: "tomorrow", status: "set"}))

      assert [
               %SchedulerItem{
                 time: "7 am",
                 date: "tomorrow",
                 resolution: "resolved",
                 turn_time: ~U[2026-10-12 09:00:00.000000Z],
                 time_zone: "Etc/UTC",
                 starts_at: ~U[2026-10-13 07:00:00.000000Z],
                 ends_at: ~U[2026-10-13 07:00:00.000000Z],
                 clock: ~T[07:00:00]
               }
             ] = items_of(@world_a)
    end

    test "reads back an item with nothing said to resolve, with no resolved fields" do
      Scheduler.insert_item!(%{world_id: @world_a, kind: "timer", duration: "", status: "running", resolution: "nothing_said"})

      assert [%SchedulerItem{resolution: "nothing_said", turn_time: nil, starts_at: nil, ends_at: nil, clock: nil}] =
               items_of(@world_a)
    end

    test "the date column is nullable and has no default, so no date is ever assumed" do
      assert column("date") == {"YES", nil}
    end

    test "every item names its resolution: the column is not nullable and has no default" do
      assert column("resolution") == {"NO", nil}
    end

    test "the resolved columns are nullable and have no default, so no time is ever assumed" do
      for name <- ~w(turn_time time_zone starts_at ends_at clock) do
        assert column(name) == {"YES", nil}, name
      end
    end

    test "keeps an empty field as the empty string it was given" do
      text_item!(%{world_id: @world_a, kind: "alarm", time: "", status: "set"})

      assert [%SchedulerItem{kind: "alarm", time: "", status: "set"}] = items_of(@world_a)
    end

    test "refuses an item without a world, with a blank world, or of an unknown kind" do
      assert_raise Ecto.InvalidChangesetError, fn ->
        text_item!(%{kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        text_item!(%{world_id: "", kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        text_item!(%{world_id: "   ", kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        text_item!(%{world_id: @world_a, kind: "appointment", time: "7am"})
      end

      assert items_of(@world_a) == []
    end

    test "refuses an item without a resolution, or with one it does not know" do
      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am"})
      end

      assert_raise Ecto.InvalidChangesetError, fn ->
        Scheduler.insert_item!(%{world_id: @world_a, kind: "alarm", time: "7am", resolution: "guessed"})
      end

      assert items_of(@world_a) == []
    end

    test "refuses a resolved item without its turn, zone or start, and resolved fields on an item that is not resolved" do
      for field <- [:turn_time, :time_zone, :starts_at] do
        assert_raise Ecto.InvalidChangesetError, fn ->
          Scheduler.insert_item!(Map.merge(Map.delete(@resolved, field), %{world_id: @world_a, kind: "alarm", time: "7am"}))
        end
      end

      for resolution <- ["nothing_said", "recorded_before_resolution"], field <- [:starts_at, :ends_at, :clock] do
        assert_raise Ecto.InvalidChangesetError, fn ->
          Scheduler.insert_item!(%{
            :world_id => @world_a,
            :kind => "alarm",
            :resolution => resolution,
            field => Map.fetch!(@resolved, field)
          })
        end
      end

      assert items_of(@world_a) == []
    end
  end

  describe "the migration that adds resolutions" do
    # Rolled back and run again inside this test's sandbox transaction, so the
    # schema is as it was when the test ends. The migrator runs a migration in
    # a task of its own, which reaches the test's connection through the
    # shared sandbox; the migration lock would hold that connection first.
    @migrate_in_sandbox [log: false, prefix: "atlas_test", migration_lock: false]

    test "marks every item stored before it as recorded before resolution, with no resolved fields" do
      [migration] =
        "priv/repo/migrations/#{@migration_version}_*.exs"
        |> Path.wildcard()
        |> Enum.flat_map(&Code.require_file/1)
        |> Enum.map(&elem(&1, 0))
        |> Enum.filter(&function_exported?(&1, :up, 0))

      assert :ok = Ecto.Migrator.down(Atlas.Repo, @migration_version, migration, @migrate_in_sandbox)

      Atlas.Repo.query!(
        """
        INSERT INTO atlas_scheduler_items (id, world_id, kind, time, date, status, inserted_at, updated_at)
        VALUES (gen_random_uuid(), $1, 'alarm', '7am', 'tomorrow', 'set', now(), now())
        """,
        [@world_a]
      )

      assert :ok = Ecto.Migrator.up(Atlas.Repo, @migration_version, migration, @migrate_in_sandbox)

      assert [
               %SchedulerItem{
                 time: "7am",
                 date: "tomorrow",
                 resolution: "recorded_before_resolution",
                 turn_time: nil,
                 time_zone: nil,
                 starts_at: nil,
                 ends_at: nil,
                 clock: nil
               }
             ] = items_of(@world_a)
    end
  end

  describe "delete_items!/2" do
    test "deletes one world's items of one kind and keeps the rest" do
      text_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})
      text_item!(%{world_id: @world_a, kind: "timer", duration: "9 minutes", status: "running"})
      text_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})
      text_item!(%{world_id: @world_b, kind: "timer", duration: "10 minutes", status: "running"})

      assert Scheduler.delete_items!(@world_a, "timer") == 2

      assert [%SchedulerItem{kind: "alarm"}] = items_of(@world_a)
      assert [%SchedulerItem{kind: "timer", duration: "10 minutes"}] = items_of(@world_b)
    end
  end

  describe "delete_items_by_id!/3" do
    test "deletes only the items named, and keeps the world's others" do
      five = text_item!(%{world_id: @world_a, kind: "timer", duration: "5 minutes", status: "running"})
      text_item!(%{world_id: @world_a, kind: "timer", duration: "9 minutes", status: "running"})

      assert Scheduler.delete_items_by_id!(@world_a, "timer", [five.id]) == 1

      assert [%SchedulerItem{kind: "timer", duration: "9 minutes"}] = items_of(@world_a)
    end

    test "deletes nothing and raises when an id names another world's item or another kind" do
      own = text_item!(%{world_id: @world_a, kind: "alarm", time: "7am", status: "set"})
      other_world = text_item!(%{world_id: @world_b, kind: "alarm", time: "8am", status: "set"})
      other_kind = text_item!(%{world_id: @world_a, kind: "reminder", text: "call home", status: "set"})

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
