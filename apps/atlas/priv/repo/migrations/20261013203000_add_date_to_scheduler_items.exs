defmodule Atlas.Repo.Migrations.AddDateToSchedulerItems do
  use Ecto.Migration

  # The day an alarm, reminder or calendar item is for, as the date slot gave
  # it, so a cancel or a check can name a day. Free text, like `time` and
  # `duration`. No column default: every item written from now on names its
  # date when its kind has one.
  #
  # Every item stored before this column existed was stored without its date:
  # no operation read the date slot, so none is recorded anywhere to migrate.
  # Those rows are left NULL, "no date recorded", which matches no day a
  # cancel or a check names. Timers have no date, and are NULL too. No row is
  # deleted.
  def up do
    alter table(:atlas_scheduler_items) do
      add :date, :string
    end

    execute("UPDATE atlas_scheduler_items SET date = NULL")
  end

  def down do
    alter table(:atlas_scheduler_items) do
      remove :date
    end
  end
end
