defmodule Atlas.Repo.Migrations.CreateWebUsers do
  use Ecto.Migration

  def change do
    # One row per browser that has visited the web interface. The id is kept in
    # the browser's signed session cookie, so the same person returns as the same
    # user across visits.
    create table(:web_users, primary_key: false) do
      add :id, :binary_id, primary_key: true

      # The world this user's conversations run in. No column default: a user is
      # created with the world it starts in, named by the caller.
      add :selected_world_id, :string, null: false

      timestamps(type: :utc_datetime_usec)
    end
  end
end
