defmodule Atlas.Repo.Migrations.AddEntityNamesToEpisodes do
  use Ecto.Migration

  def change do
    alter table(:atlas_episodes) do
      add :entity_names, {:array, :string}, default: [], null: false
    end
  end
end
