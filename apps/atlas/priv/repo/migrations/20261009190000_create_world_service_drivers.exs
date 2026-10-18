defmodule Atlas.Repo.Migrations.CreateWorldServiceDrivers do
  use Ecto.Migration

  def change do
    # Which driver serves a capability (such as "scheduling": alarms, timers,
    # reminders and calendar items) in a world. One row per world and
    # capability. No column defaults: a row exists only once a driver has been
    # chosen for the world, and the world, capability and driver are named on
    # every write.
    create table(:atlas_world_service_drivers, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false
      add :capability, :string, null: false
      add :driver, :string, null: false

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_world_service_drivers, [:world_id, :capability])
  end
end
