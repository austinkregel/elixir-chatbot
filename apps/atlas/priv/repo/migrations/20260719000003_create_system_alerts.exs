defmodule Atlas.Repo.Migrations.CreateSystemAlerts do
  use Ecto.Migration

  def change do
    create table(:atlas_system_alerts, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :ship_id, :string, null: false
      add :system_id, :string, null: false
      add :severity, :string, null: false
      add :kind, :string, null: false
      add :message, :string
      add :occurred_at, :utc_datetime_usec, null: false

      timestamps(type: :utc_datetime_usec, updated_at: false)
    end

    create index(:atlas_system_alerts, [:ship_id, :occurred_at])
  end
end
