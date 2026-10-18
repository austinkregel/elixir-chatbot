defmodule Atlas.Repo.Migrations.AddWorldIdToReviewCandidates do
  use Ecto.Migration

  # A review candidate belongs to the world its finding was found for. Until
  # now the world was recorded only inside the finding (`finding->>'world_id'`),
  # and a finding without one was about no world. A row whose finding names a
  # world is assigned that world; a row whose finding names none is assigned to
  # the "default" world, explicitly, before the column becomes NOT NULL. No row
  # is deleted. The column has no default: every candidate written from now on
  # names its world.
  def up do
    alter table(:atlas_review_candidates) do
      add :world_id, :string
    end

    execute("""
    UPDATE atlas_review_candidates
    SET world_id = finding->>'world_id'
    WHERE world_id IS NULL AND COALESCE(finding->>'world_id', '') <> ''
    """)

    execute("UPDATE atlas_review_candidates SET world_id = 'default' WHERE world_id IS NULL")

    alter table(:atlas_review_candidates) do
      modify :world_id, :string, null: false
    end

    create index(:atlas_review_candidates, [:world_id])
  end

  # The world goes back inside the finding, where it was recorded before.
  def down do
    execute("UPDATE atlas_review_candidates SET finding = jsonb_set(finding, '{world_id}', to_jsonb(world_id))")

    drop index(:atlas_review_candidates, [:world_id])

    alter table(:atlas_review_candidates) do
      remove :world_id
    end
  end
end
