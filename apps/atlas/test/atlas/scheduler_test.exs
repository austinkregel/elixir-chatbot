defmodule Atlas.SchedulerTest do
  @moduledoc """
  The store behind the local scheduler: an item is read back with the world and
  kind it was stored with, with its resolved time when it has one, and with its
  owner and firing; deleting removes only one world's items of one kind; and an
  item without a world, with an unknown kind, without its resolution, or
  without the firing fields its fire state carries is refused.
  """
  use Atlas.DataCase, async: false

  alias Atlas.Scheduler
  alias Atlas.Schemas.SchedulerItem

  @world_a "test_scheduler_store_a"
  @world_b "test_scheduler_store_b"

  @resolution_migration 20261014120000
  @firing_migration 20261014150000

  defp items_of(world_id), do: Enum.filter(Scheduler.list_items(), &(&1.world_id == world_id))

  # An item stored as text only, as every item was before resolutions and
  # owners were.
  defp text_item!(attrs),
    do: Scheduler.insert_item!(Map.merge(attrs, %{resolution: "recorded_before_resolution", fire_state: "owner_not_recorded"}))

  @resolved %{
    resolution: "resolved",
    turn_time: ~U[2026-10-12 09:00:00.000000Z],
    time_zone: "Etc/UTC",
    utc_offset: 0,
    starts_at: ~U[2026-10-13 07:00:00.000000Z],
    ends_at: ~U[2026-10-13 07:00:00.000000Z],
    clock: ~T[07:00:00]
  }

  defp web_user!, do: Atlas.WebUsers.create!(@world_a)

  # A resolved alarm of a web user, due at 7 am.
  defp pending_alarm!(owner, changes \\ %{}) do
    @resolved
    |> Map.merge(%{world_id: @world_a, kind: "alarm", time: "7 am", date: "", status: "set"})
    |> Map.merge(%{web_user_id: owner.id, fire_state: "pending", due_at: @resolved.starts_at})
    |> Map.merge(changes)
    |> Scheduler.insert_item!()
  end

  # Migrations are rolled back and run again inside the test's sandbox
  # transaction, so the schema is as it was when the test ends. The migrator
  # runs a migration in a task of its own, which reaches the test's connection
  # through the shared sandbox; the migration lock would hold that connection
  # first.
  @migrate_in_sandbox [log: false, prefix: "atlas_test", migration_lock: false]

  # The migration's module, named for its file as Ecto names it, loaded from
  # the file when it is not loaded yet.
  defp migration!(version) do
    [file] = Path.wildcard("priv/repo/migrations/#{version}_*.exs")
    name = file |> Path.basename(".exs") |> String.replace_prefix("#{version}_", "") |> Macro.camelize()
    module = Module.concat(Atlas.Repo.Migrations, name)

    unless Code.ensure_loaded?(module), do: Code.require_file(file)
    module
  end

  defp down!(version), do: assert(:ok = Ecto.Migrator.down(Atlas.Repo, version, migration!(version), @migrate_in_sandbox))
  defp up!(version), do: assert(:ok = Ecto.Migrator.up(Atlas.Repo, version, migration!(version), @migrate_in_sandbox))

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
      Scheduler.insert_item!(
        Map.merge(@resolved, %{world_id: @world_a, kind: "alarm", time: "7 am", date: "tomorrow", status: "set", fire_state: "owner_not_recorded"})
      )

      assert [
               %SchedulerItem{
                 time: "7 am",
                 date: "tomorrow",
                 resolution: "resolved",
                 turn_time: ~U[2026-10-12 09:00:00.000000Z],
                 time_zone: "Etc/UTC",
                 utc_offset: 0,
                 starts_at: ~U[2026-10-13 07:00:00.000000Z],
                 ends_at: ~U[2026-10-13 07:00:00.000000Z],
                 clock: ~T[07:00:00]
               }
             ] = items_of(@world_a)
    end

    test "reads back an item with nothing said to resolve, with no resolved fields" do
      owner = web_user!()

      Scheduler.insert_item!(%{
        world_id: @world_a,
        kind: "timer",
        duration: "",
        status: "running",
        resolution: "nothing_said",
        web_user_id: owner.id,
        fire_state: "no_due_time"
      })

      assert [%SchedulerItem{resolution: "nothing_said", turn_time: nil, utc_offset: nil, starts_at: nil, ends_at: nil, clock: nil}] =
               items_of(@world_a)
    end

    test "reads back an item's owner, its due time and whether it has fired" do
      owner = web_user!()
      %SchedulerItem{id: id} = pending_alarm!(owner)
      owner_id = owner.id

      assert [%SchedulerItem{web_user_id: ^owner_id, fire_state: "pending", due_at: ~U[2026-10-13 07:00:00.000000Z], fired_at: nil}] =
               items_of(@world_a)

      Scheduler.update_item!(id, %{fire_state: "fired", fired_at: ~U[2026-10-13 07:00:00.000000Z], status: "ringing"})

      assert [%SchedulerItem{fire_state: "fired", fired_at: ~U[2026-10-13 07:00:00.000000Z], status: "ringing"}] = items_of(@world_a)
    end

    test "refuses a fire state without the firing fields it carries, or with others, and an owner who is no web user" do
      owner = web_user!()

      for changes <- [
            %{fire_state: nil},
            %{fire_state: "rung"},
            %{web_user_id: nil},
            %{due_at: nil},
            %{fired_at: ~U[2026-10-13 07:00:00Z]},
            %{fire_state: "fired"},
            %{fire_state: "no_due_time"},
            %{fire_state: "owner_not_recorded"},
            %{web_user_id: Ecto.UUID.generate()}
          ] do
        assert_raise Ecto.InvalidChangesetError, fn -> pending_alarm!(owner, changes) end
      end

      assert items_of(@world_a) == []
    end

    test "refuses a change that leaves an item's firing fields inconsistent, and keeps it as it was" do
      %SchedulerItem{id: id} = pending_alarm!(web_user!())

      assert_raise Ecto.InvalidChangesetError, fn -> Scheduler.update_item!(id, %{fire_state: "fired"}) end
      assert_raise Ecto.NoResultsError, fn -> Scheduler.update_item!(Ecto.UUID.generate(), %{fire_state: "fired"}) end

      assert [%SchedulerItem{fire_state: "pending"}] = items_of(@world_a)
    end

    test "the soonest due time of the pending items, and the pending items due by an instant, soonest first" do
      owner = web_user!()
      assert Scheduler.next_due_at() == nil

      eight = pending_alarm!(owner, %{time: "8 am", due_at: ~U[2026-10-13 08:00:00Z]})
      seven = pending_alarm!(owner)

      pending_alarm!(owner, %{
        time: "6 am",
        due_at: ~U[2026-10-13 06:00:00Z],
        fire_state: "fired",
        fired_at: ~U[2026-10-13 06:00:00Z]
      })

      assert Scheduler.next_due_at() == ~U[2026-10-13 07:00:00.000000Z]
      assert Scheduler.due_items(~U[2026-10-13 06:59:59Z]) == []
      assert Enum.map(Scheduler.due_items(~U[2026-10-13 07:00:00Z]), & &1.id) == [seven.id]
      assert Enum.map(Scheduler.due_items(~U[2026-10-14 00:00:00Z]), & &1.id) == [seven.id, eight.id]
    end

    test "the date column is nullable and has no default, so no date is ever assumed" do
      assert column("date") == {"YES", nil}
    end

    test "every item names its resolution: the column is not nullable and has no default" do
      assert column("resolution") == {"NO", nil}
    end

    test "the resolved columns are nullable and have no default, so no time is ever assumed" do
      for name <- ~w(turn_time time_zone utc_offset starts_at ends_at clock) do
        assert column(name) == {"YES", nil}, name
      end
    end

    test "every item names its fire state, and no owner, due time or firing is ever assumed" do
      assert column("fire_state") == {"NO", nil}

      for name <- ~w(web_user_id due_at fired_at) do
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

    test "refuses a resolved item without its turn, zone, offset or start, and resolved fields on an item that is not resolved" do
      for field <- [:turn_time, :time_zone, :utc_offset, :starts_at] do
        assert_raise Ecto.InvalidChangesetError, fn ->
          Scheduler.insert_item!(
            Map.merge(Map.delete(@resolved, field), %{world_id: @world_a, kind: "alarm", time: "7am", fire_state: "owner_not_recorded"})
          )
        end
      end

      for resolution <- ["nothing_said", "recorded_before_resolution"], field <- [:utc_offset, :starts_at, :ends_at, :clock] do
        assert_raise Ecto.InvalidChangesetError, fn ->
          Scheduler.insert_item!(%{
            :world_id => @world_a,
            :kind => "alarm",
            :resolution => resolution,
            :fire_state => "owner_not_recorded",
            field => Map.fetch!(@resolved, field)
          })
        end
      end

      assert items_of(@world_a) == []
    end
  end

  describe "the migration that adds resolutions" do
    test "marks every item stored before it as recorded before resolution, with no resolved fields" do
      # The migration after it, which adds owners and firing, is rolled back
      # first and run again after.
      down!(@firing_migration)
      down!(@resolution_migration)

      Atlas.Repo.query!(
        """
        INSERT INTO atlas_scheduler_items (id, world_id, kind, time, date, status, inserted_at, updated_at)
        VALUES (gen_random_uuid(), $1, 'alarm', '7am', 'tomorrow', 'set', now(), now())
        """,
        [@world_a]
      )

      up!(@resolution_migration)

      assert [
               %{
                 "time" => "7am",
                 "date" => "tomorrow",
                 "resolution" => "recorded_before_resolution",
                 "turn_time" => nil,
                 "time_zone" => nil,
                 "starts_at" => nil,
                 "ends_at" => nil,
                 "clock" => nil
               }
             ] = rows_of(@world_a)

      up!(@firing_migration)
    end
  end

  describe "the migration that adds owners and firing" do
    test "marks every item stored before it as having no owner recorded, never to fire, and gives each resolved one the offset 0" do
      down!(@firing_migration)

      Atlas.Repo.query!(
        """
        INSERT INTO atlas_scheduler_items
          (id, world_id, kind, time, date, status, resolution, turn_time, time_zone, starts_at, ends_at, clock, inserted_at, updated_at)
        VALUES
          (gen_random_uuid(), $1, 'alarm', '7 am', 'tomorrow', 'set', 'resolved',
           '2026-10-12 09:00:00', 'Etc/UTC', '2026-10-13 07:00:00', '2026-10-13 07:00:00', '07:00:00', now(), now() - interval '1 second'),
          (gen_random_uuid(), $1, 'timer', NULL, NULL, 'running', 'nothing_said',
           NULL, NULL, NULL, NULL, NULL, now() + interval '1 second', now())
        """,
        [@world_a]
      )

      up!(@firing_migration)

      assert [
               %SchedulerItem{
                 kind: "alarm",
                 resolution: "resolved",
                 utc_offset: 0,
                 fire_state: "owner_not_recorded",
                 web_user_id: nil,
                 due_at: nil,
                 fired_at: nil
               },
               %SchedulerItem{
                 kind: "timer",
                 resolution: "nothing_said",
                 utc_offset: nil,
                 fire_state: "owner_not_recorded",
                 web_user_id: nil,
                 due_at: nil,
                 fired_at: nil
               }
             ] = items_of(@world_a)
    end
  end

  # The world's rows as the table holds them, by column name, oldest first.
  defp rows_of(world_id) do
    %{columns: columns, rows: rows} =
      Atlas.Repo.query!("SELECT * FROM atlas_scheduler_items WHERE world_id = $1 ORDER BY inserted_at", [world_id])

    Enum.map(rows, &(columns |> Enum.zip(&1) |> Map.new()))
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
