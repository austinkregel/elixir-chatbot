defmodule Atlas.Repo.Migrations.CreateSchedulerItems do
  use Ecto.Migration

  def change do
    # Alarms, reminders, timers and calendar items set through the local
    # scheduler, so they survive a restart. Each belongs to the world it was set
    # in. No column defaults: the world and the kind are named on every write.
    create table(:atlas_scheduler_items, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false
      add :kind, :string, null: false
      add :time, :string
      add :text, :string
      add :duration, :string
      add :status, :string

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_scheduler_items, [:world_id, :kind])
  end
end
