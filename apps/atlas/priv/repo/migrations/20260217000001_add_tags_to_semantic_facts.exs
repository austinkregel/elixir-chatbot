defmodule Atlas.Repo.Migrations.AddTagsToSemanticFacts do
  use Ecto.Migration

  def change do
    alter table(:atlas_semantic_facts) do
      add :tags, {:array, :string}, default: []
    end
  end
end
