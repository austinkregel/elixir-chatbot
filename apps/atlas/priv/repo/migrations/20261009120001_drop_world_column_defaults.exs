defmodule Atlas.Repo.Migrations.DropWorldColumnDefaults do
  use Ecto.Migration

  # No row is written into a world it did not name. These world columns were
  # created with a database default of "default", which filled in the world for
  # any insert that left it out; the defaults are dropped so such an insert
  # fails instead.
  #
  # The credential, episode, semantic fact and knowledge entry columns were
  # created NOT NULL, so none of their rows is without a world. The intent
  # review candidate column was created nullable: rows recorded without a world
  # are assigned to the "default" world, explicitly, before it becomes NOT NULL.
  def up do
    alter table(:atlas_credentials) do
      modify :world, :string, null: false, default: nil
    end

    alter table(:atlas_episodes) do
      modify :world_id, :string, null: false, default: nil
    end

    alter table(:atlas_semantic_facts) do
      modify :world_id, :string, null: false, default: nil
    end

    alter table(:atlas_knowledge_entries) do
      modify :world_id, :string, null: false, default: nil
    end

    execute("UPDATE atlas_intent_review_candidates SET world_id = 'default' WHERE world_id IS NULL")

    alter table(:atlas_intent_review_candidates) do
      modify :world_id, :string, null: false
    end
  end

  def down do
    alter table(:atlas_intent_review_candidates) do
      modify :world_id, :string, null: true
    end

    alter table(:atlas_knowledge_entries) do
      modify :world_id, :string, null: false, default: "default"
    end

    alter table(:atlas_semantic_facts) do
      modify :world_id, :string, null: false, default: "default"
    end

    alter table(:atlas_episodes) do
      modify :world_id, :string, null: false, default: "default"
    end

    alter table(:atlas_credentials) do
      modify :world, :string, null: false, default: "default"
    end
  end
end
