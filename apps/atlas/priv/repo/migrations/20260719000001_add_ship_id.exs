defmodule Atlas.Repo.Migrations.AddShipId do
  use Ecto.Migration

  # A "ship" is one instance of the umbrella. Stamp ship identity onto the durable
  # accountability records and the service summary so state is filterable per ship
  # for the fleet-of-ships future (Fleet.Ship / Fleet.Clearance).
  def change do
    alter table(:atlas_command_records) do
      add :ship_id, :string
    end

    alter table(:atlas_service_summaries) do
      add :ship_id, :string
    end

    create index(:atlas_command_records, [:ship_id])
  end
end
