defmodule Atlas.Repo.Migrations.AddWorldIdToLearningSessions do
  use Ecto.Migration

  # A learning session researches for one world. Sessions recorded before the
  # column existed ran in the "default" world, so they are assigned to it here,
  # explicitly, before the column becomes NOT NULL. The column has no default:
  # every session written from now on names its world.
  def up do
    alter table(:atlas_learning_sessions) do
      add :world_id, :string
    end

    execute("UPDATE atlas_learning_sessions SET world_id = 'default' WHERE world_id IS NULL")

    alter table(:atlas_learning_sessions) do
      modify :world_id, :string, null: false
    end

    create index(:atlas_learning_sessions, [:world_id])
  end

  def down do
    drop index(:atlas_learning_sessions, [:world_id])

    alter table(:atlas_learning_sessions) do
      remove :world_id
    end
  end
end
