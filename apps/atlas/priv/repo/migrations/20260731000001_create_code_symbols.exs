defmodule Atlas.Repo.Migrations.CreateCodeSymbols do
  use Ecto.Migration

  def change do
    create table(:atlas_code_symbols, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false
      add :name, :string, null: false
      add :qualified_name, :string, null: false
      add :entity_type, :string, null: false
      add :language, :string, null: false
      # Part of the row's identity, so they carry defaults rather than NULL:
      # Postgres treats NULLs as distinct in a unique index, which would let the
      # same symbol be inserted endlessly.
      add :file_path, :string, null: false, default: ""
      add :line, :integer, null: false, default: 0
      add :column, :integer, null: false, default: 0
      add :metadata, :map, null: false, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    # A symbol's identity is *where it is*, not just what it is called. Two
    # functions can share a qualified name across files, and an extractor run
    # twice over the same file must not double the corpus — this is the upsert
    # conflict target that makes re-indexing idempotent.
    create unique_index(:atlas_code_symbols, [:world_id, :qualified_name, :entity_type, :file_path, :line],
             name: :atlas_code_symbols_identity
           )

    # The four read paths: by bare name, by qualified name, by kind, by file.
    # Every one is world-scoped first, so a query never scans another corpus.
    #
    # The two name lookups are FUNCTIONAL indexes on lower(...) because symbol
    # lookup has always been case-insensitive here (the ETS keys were
    # downcased). A plain index on the raw column would simply not be used by
    # the query that actually runs.
    create index(:atlas_code_symbols, ["world_id", "lower(name)"],
             name: :atlas_code_symbols_world_lower_name
           )

    create index(:atlas_code_symbols, ["world_id", "lower(qualified_name)"],
             name: :atlas_code_symbols_world_lower_qualified
           )

    create index(:atlas_code_symbols, [:world_id, :entity_type])
    create index(:atlas_code_symbols, [:world_id, :file_path])

    create table(:atlas_code_relations, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false
      add :from_qualified, :string, null: false
      add :relation_type, :string, null: false
      add :to_qualified, :string, null: false

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(
             :atlas_code_relations,
             [:world_id, :from_qualified, :relation_type, :to_qualified],
             name: :atlas_code_relations_identity
           )

    create index(:atlas_code_relations, [:world_id, :from_qualified, :relation_type])
    create index(:atlas_code_relations, [:world_id, :to_qualified, :relation_type])
  end
end
