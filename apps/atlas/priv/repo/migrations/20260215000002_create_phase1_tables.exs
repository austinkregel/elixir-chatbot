defmodule Atlas.Repo.Migrations.CreatePhase1Tables do
  use Ecto.Migration

  def change do
    # ========================================================================
    # Credentials table (replaces CredentialVault .enc file)
    # ========================================================================
    create table(:atlas_credentials) do
      add :world, :string, null: false, default: "default"
      add :service, :string, null: false
      add :key, :string, null: false
      add :encrypted_value, :binary, null: false

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_credentials, [:world, :service, :key])

    # ========================================================================
    # Beliefs table (replaces BeliefStore .term file)
    # ========================================================================
    create table(:atlas_beliefs, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :subject, :string, null: false
      add :predicate, :string, null: false
      add :object, :text, null: false
      add :confidence, :float, null: false
      add :source, :string, null: false
      add :source_authority, :string
      add :user_id, :string
      add :node_id, :string
      add :retracted, :boolean, null: false, default: false
      add :last_confirmed, :utc_datetime_usec
      add :provenance, :map, default: %{}
      add :metadata, :map, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_beliefs, [:user_id])
    create index(:atlas_beliefs, [:subject])
    create index(:atlas_beliefs, [:predicate])
    create index(:atlas_beliefs, [:retracted, :confidence])

    # ========================================================================
    # Episodes table (replaces Memory.Store episodes)
    # ========================================================================
    create table(:atlas_episodes, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false, default: "default"
      add :state, :text, null: false
      add :action, :text, null: false
      add :outcome, :text, null: false
      add :tags, {:array, :string}, null: false, default: []
      add :embedding, {:array, :float}, null: false, default: []
      add :semantic_id, :binary_id

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_episodes, [:world_id])
    create index(:atlas_episodes, [:tags], using: :gin)

    # ========================================================================
    # Semantic Facts table (replaces Memory.Store semantics)
    # ========================================================================
    create table(:atlas_semantic_facts, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :world_id, :string, null: false, default: "default"
      add :content, :text, null: false
      add :category, :string, null: false
      add :confidence, :float, null: false
      add :embedding, {:array, :float}, null: false, default: []
      add :source_episodes, {:array, :string}, null: false, default: []

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_semantic_facts, [:world_id])

    # ========================================================================
    # Review Candidates table (replaces ReviewQueue .term file)
    # ========================================================================
    create table(:atlas_review_candidates, primary_key: false) do
      add :id, :string, primary_key: true
      add :status, :string, null: false, default: "pending"
      add :finding, :map, null: false, default: %{}
      add :aggregate_confidence, :float, null: false
      add :corroborating_sources, {:array, :map}, null: false, default: []
      add :conflicting_findings, {:array, :map}, null: false, default: []
      add :existing_contradictions, {:array, :map}, null: false, default: []
      add :reviewer_notes, :text
      add :reviewed_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_review_candidates, [:status])
    create index(:atlas_review_candidates, [:status, :aggregate_confidence])

    # ========================================================================
    # Learned Facts table (replaces FactDatabase.Integration .json file)
    # ========================================================================
    create table(:atlas_learned_facts, primary_key: false) do
      add :id, :string, primary_key: true
      add :entity, :string, null: false
      add :entity_type, :string
      add :fact, :text, null: false
      add :category, :string, null: false, default: "learned"
      add :confidence, :float, null: false
      add :verification_source, :string

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_learned_facts, [:entity])
    create index(:atlas_learned_facts, [:category])
  end
end
