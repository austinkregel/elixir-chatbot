defmodule Atlas.Repo.Migrations.AddWorldIdToBeliefs do
  use Ecto.Migration

  def change do
    alter table(:atlas_beliefs) do
      add :world_id, :string
    end

    # Legacy rows belong to the shared default world.
    execute(
      "UPDATE atlas_beliefs SET world_id = 'default' WHERE world_id IS NULL",
      "SELECT 1"
    )

    create index(:atlas_beliefs, [:world_id])
    create index(:atlas_beliefs, [:world_id, :user_id])
  end
end
