defmodule Atlas.Repo.Migrations.CreateSettings do
  use Ecto.Migration

  def change do
    # Node-wide settings that are changed at runtime and must survive a restart,
    # one row per setting, keyed by its name. No column defaults: a row exists
    # only once its value has been written.
    create table(:atlas_settings, primary_key: false) do
      add :name, :string, primary_key: true
      add :value, :string, null: false

      timestamps(type: :utc_datetime_usec)
    end
  end
end
