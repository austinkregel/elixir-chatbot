defmodule Atlas.Repo.Migrations.CreateCommandRecords do
  use Ecto.Migration

  def change do
    create table(:atlas_command_records, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :kind, :string, null: false
      add :order_id, :string
      add :from_agent, :string, null: false
      add :to_agent, :string
      add :world_id, :string
      add :authority, :string
      add :verdict, :string
      add :reason, :string
      add :payload, :map, default: %{}
      add :issued_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_command_records, [:order_id])
    create index(:atlas_command_records, [:from_agent])
    create index(:atlas_command_records, [:kind])
    create index(:atlas_command_records, [:issued_at])
  end
end
