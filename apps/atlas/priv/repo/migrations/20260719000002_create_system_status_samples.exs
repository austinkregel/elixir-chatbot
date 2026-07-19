defmodule Atlas.Repo.Migrations.CreateSystemStatusSamples do
  use Ecto.Migration

  def change do
    create table(:atlas_system_status_samples, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :ship_id, :string, null: false
      add :system_id, :string, null: false
      add :status, :string, null: false
      add :metric, :float
      add :sampled_at, :utc_datetime_usec, null: false

      timestamps(type: :utc_datetime_usec, updated_at: false)
    end

    # The two hot query paths: a system's history on a ship, and the ship trend.
    create index(:atlas_system_status_samples, [:ship_id, :system_id, :sampled_at])
  end
end
