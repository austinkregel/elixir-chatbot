defmodule Atlas.Repo.Migrations.AddWorldIdToBeliefs do
  use Ecto.Migration

  # A belief is held in one world. Beliefs recorded before the column existed
  # were held in the "default" world, so they are assigned to it here,
  # explicitly, before the column becomes NOT NULL. No row is deleted. The
  # column has no default: every belief written from now on names its world.
  def up do
    alter table(:atlas_beliefs) do
      add :world_id, :string
    end

    execute("UPDATE atlas_beliefs SET world_id = 'default' WHERE world_id IS NULL")

    alter table(:atlas_beliefs) do
      modify :world_id, :string, null: false
    end

    create index(:atlas_beliefs, [:world_id])
  end

  def down do
    drop index(:atlas_beliefs, [:world_id])

    alter table(:atlas_beliefs) do
      remove :world_id
    end
  end
end
