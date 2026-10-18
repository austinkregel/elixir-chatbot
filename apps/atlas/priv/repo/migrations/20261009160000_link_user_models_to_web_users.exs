defmodule Atlas.Repo.Migrations.LinkUserModelsToWebUsers do
  use Ecto.Migration

  # A user model is what the system holds about one web user, in every world.
  # Its user_id becomes that user's id: a uuid referencing web_users, so a model
  # cannot be written for a user who does not exist.
  #
  # Existing rows are kept. Each must already hold a web user's id; a row that
  # does not is not converted silently, and the cast or the constraint fails the
  # migration, naming the value. The development and test databases held 0
  # rows in atlas_user_models when this was written.
  #
  # A web user with a user model cannot be deleted (ON DELETE RESTRICT): the
  # facts held about them go only when they are removed on purpose.
  def up do
    execute("ALTER TABLE atlas_user_models ALTER COLUMN user_id TYPE uuid USING user_id::uuid")

    execute("""
    ALTER TABLE atlas_user_models
      ADD CONSTRAINT atlas_user_models_user_id_fkey
      FOREIGN KEY (user_id) REFERENCES web_users(id) ON DELETE RESTRICT
    """)
  end

  def down do
    execute("ALTER TABLE atlas_user_models DROP CONSTRAINT atlas_user_models_user_id_fkey")
    execute("ALTER TABLE atlas_user_models ALTER COLUMN user_id TYPE varchar(255) USING user_id::text")
  end
end
