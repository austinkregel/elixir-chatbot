defmodule Atlas.Repo.Migrations.AddResolutionToSchedulerItems do
  use Ecto.Migration

  # What an alarm's, reminder's or calendar item's date and time, or a
  # timer's duration, name, resolved against the turn they were said in, kept
  # beside the text they were said as:
  #
  #   resolution  how the item's time was resolved: "resolved", "nothing_said"
  #               (no date, time or duration was given), or
  #               "recorded_before_resolution" (see below)
  #   turn_time   the moment of the turn the item was set in
  #   time_zone   the zone its days and clock times were read in
  #   starts_at   when the time it names begins (a timer: when it was set)
  #   ends_at     when that time ends, exclusive: the same moment for a point
  #               in time, the start of the day after a day or period, when a
  #               timer runs out; NULL when it has no end ("later")
  #   clock       the time of day said, NULL when none was
  #
  # No column default: every item written from now on names its resolution.
  #
  # Every item stored before these columns existed was stored as text only:
  # the turn it was set in was never recorded, so "tomorrow" stored then names
  # no day that can be recovered. Those rows are marked
  # "recorded_before_resolution", their other resolution columns NULL; they
  # match no day, time or duration a cancel or a check names. No row is
  # deleted.
  def up do
    alter table(:atlas_scheduler_items) do
      add :resolution, :string
      add :turn_time, :utc_datetime_usec
      add :time_zone, :string
      add :starts_at, :utc_datetime_usec
      add :ends_at, :utc_datetime_usec
      add :clock, :time
    end

    execute("UPDATE atlas_scheduler_items SET resolution = 'recorded_before_resolution'")

    alter table(:atlas_scheduler_items) do
      modify :resolution, :string, null: false
    end
  end

  def down do
    alter table(:atlas_scheduler_items) do
      remove :resolution
      remove :turn_time
      remove :time_zone
      remove :starts_at
      remove :ends_at
      remove :clock
    end
  end
end
