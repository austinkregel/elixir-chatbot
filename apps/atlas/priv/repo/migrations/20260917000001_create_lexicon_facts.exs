defmodule Atlas.Repo.Migrations.CreateLexiconFacts do
  use Ecto.Migration

  def change do
    create table(:atlas_lexicon_facts, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :word, :string, null: false
      add :kind, :string, null: false
      add :key, :string, null: false
      add :ref, :string, null: false, default: ""
      add :value, :map, null: false, default: %{}
      add :source, :string, null: false
      add :confidence, :float, null: false, default: 1.0
      add :frequency, :integer, null: false, default: 1
      add :archived, :boolean, null: false, default: false
      add :last_observed_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    # One fact per (word, kind, key, ref, source). Source is part of the
    # identity so a learned fact can sit beside the seeded one it contradicts
    # rather than overwrite it; which one applies is decided from context at
    # read time, not by the storage layer.
    create unique_index(:atlas_lexicon_facts, [:word, :kind, :key, :ref, :source])
    create index(:atlas_lexicon_facts, [:word])
    create index(:atlas_lexicon_facts, [:kind, :key])
  end
end
