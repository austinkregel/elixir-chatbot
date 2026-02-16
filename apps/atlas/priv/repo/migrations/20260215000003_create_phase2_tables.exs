defmodule Atlas.Repo.Migrations.CreatePhase2Tables do
  use Ecto.Migration

  def change do
    # ========================================================================
    # Source Reliability table (replaces source_reliability_learned.term)
    # ========================================================================
    create table(:atlas_source_reliability) do
      add :domain, :string, null: false
      add :reliability_score, :float
      add :bias_rating, :string
      add :trust_tier, :string
      add :confirmed_count, :integer, default: 0
      add :rejected_count, :integer, default: 0
      add :admin_decisions, {:array, :map}, default: []
      add :metadata, :map, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_source_reliability, [:domain])

    # ========================================================================
    # Source Authority table (replaces source_authority_learned.term)
    # ========================================================================
    create table(:atlas_source_authority) do
      add :authority_key, :string, null: false
      add :confirmed_count, :integer, default: 0
      add :contradicted_count, :integer, default: 0
      add :total_added, :integer, default: 0
      add :credibility, :float
      add :last_updated, :utc_datetime_usec
      add :metadata, :map, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_source_authority, [:authority_key])

    # ========================================================================
    # User Models table (replaces user_models.term)
    # ========================================================================
    create table(:atlas_user_models) do
      add :user_id, :string, null: false
      add :facts, :map, default: %{}
      add :interaction_patterns, :map, default: %{}
      add :epistemic_bounds, :map, default: %{}
      add :provenance_map, :map, default: %{}
      add :disclosure_history, {:array, :map}, default: []

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_user_models, [:user_id])

    # ========================================================================
    # Knowledge Entries table (replaces KnowledgeStore per-persona .json)
    # ========================================================================
    create table(:atlas_knowledge_entries) do
      add :world_id, :string, null: false, default: "default"
      add :persona_name, :string, null: false
      add :category, :string, null: false
      add :key, :string, null: false
      add :data, :map, null: false, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create unique_index(:atlas_knowledge_entries, [:world_id, :persona_name, :category, :key],
             name: :atlas_knowledge_entries_unique_entry
           )

    create index(:atlas_knowledge_entries, [:persona_name])
    create index(:atlas_knowledge_entries, [:world_id, :persona_name])

    # ========================================================================
    # Persona Memories table (replaces legacy MemoryStore per-persona .json)
    # ========================================================================
    create table(:atlas_persona_memories) do
      add :persona_name, :string, null: false
      add :role, :string, null: false
      add :content, :text, null: false
      add :context, :map, default: %{}
      add :message_timestamp, :bigint

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_persona_memories, [:persona_name])
  end
end
