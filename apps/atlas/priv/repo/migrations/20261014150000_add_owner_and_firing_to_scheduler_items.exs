defmodule Atlas.Repo.Migrations.AddOwnerAndFiringToSchedulerItems do
  use Ecto.Migration

  # Who an alarm, reminder, timer or calendar item belongs to, the offset its
  # time was read with, and whether it has fired:
  #
  #   web_user_id  the web user who set it, whose open pages are told when it
  #                fires; a uuid referencing web_users. A web user with items
  #                cannot be deleted (ON DELETE RESTRICT).
  #   utc_offset   for a resolved item, the offset from UTC, in seconds east,
  #                its start was read with in its time_zone
  #   due_at       when it fires: a timer when it runs out, any other item at
  #                the time of day it was set for; NULL when nothing names
  #                such an instant
  #   fire_state   "pending" (it fires at due_at), "fired", "fired_late" (it
  #                was due while the scheduler was not running and fired when
  #                it started), "no_due_time" (nothing names when it fires),
  #                or "owner_not_recorded" (see below)
  #   fired_at     when it fired; NULL until it has
  #
  # No column default: every item written from now on names its owner, its
  # fire state and, when resolved, its offset.
  #
  # Every item stored before these columns existed was stored without its
  # owner: no operation recorded who set it, so there is no one to tell when it
  # fires. Those rows are marked "owner_not_recorded", with no owner, no due
  # time and no firing; they never fire, and are still read, checked and
  # cancelled as before. Before this, the only zone a time could be read in
  # was "Etc/UTC", so every resolved row was read with the offset 0, and is
  # given it; a resolved row in any other zone fails the migration rather than
  # being given an offset it was not read with. No row is deleted.
  def up do
    alter table(:atlas_scheduler_items) do
      add :web_user_id, references(:web_users, type: :binary_id, on_delete: :restrict)
      add :utc_offset, :integer
      add :due_at, :utc_datetime_usec
      add :fire_state, :string
      add :fired_at, :utc_datetime_usec
    end

    execute("""
    DO $$
    BEGIN
      IF EXISTS (
        SELECT 1 FROM atlas_scheduler_items
        WHERE resolution = 'resolved' AND time_zone IS DISTINCT FROM 'Etc/UTC'
      ) THEN
        RAISE EXCEPTION 'atlas_scheduler_items holds a resolved item read in a zone other than Etc/UTC; its offset is not known';
      END IF;
    END
    $$
    """)

    execute("UPDATE atlas_scheduler_items SET utc_offset = 0 WHERE resolution = 'resolved'")

    execute("""
    UPDATE atlas_scheduler_items
    SET fire_state = 'owner_not_recorded', web_user_id = NULL, due_at = NULL, fired_at = NULL
    """)

    alter table(:atlas_scheduler_items) do
      modify :fire_state, :string, null: false
    end

    create index(:atlas_scheduler_items, [:fire_state, :due_at])
    create index(:atlas_scheduler_items, [:web_user_id])
  end

  def down do
    drop index(:atlas_scheduler_items, [:web_user_id])
    drop index(:atlas_scheduler_items, [:fire_state, :due_at])

    alter table(:atlas_scheduler_items) do
      remove :web_user_id
      remove :utc_offset
      remove :due_at
      remove :fire_state
      remove :fired_at
    end
  end
end
