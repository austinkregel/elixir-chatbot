defmodule Atlas.Repo.Migrations.CreateLearningTables do
  use Ecto.Migration

  def change do
    # ========================================================================
    # Learning Sessions table (ephemeral tier - session lifecycle)
    # ========================================================================
    create table(:atlas_learning_sessions, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :topic, :string
      add :status, :string, null: false, default: "active"
      add :started_at, :utc_datetime_usec
      add :completed_at, :utc_datetime_usec
      add :findings_count, :integer, default: 0
      add :approved_count, :integer, default: 0
      add :rejected_count, :integer, default: 0
      add :hypotheses_tested, :integer, default: 0
      add :hypotheses_supported, :integer, default: 0
      add :hypotheses_falsified, :integer, default: 0
      add :source_type, :string, default: "web"
      add :metadata, :map, default: %{}

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_learning_sessions, [:status])
    create index(:atlas_learning_sessions, [:started_at])

    # ========================================================================
    # Research Goals table (per-session goals)
    # ========================================================================
    create table(:atlas_research_goals, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :session_id, references(:atlas_learning_sessions, type: :binary_id, on_delete: :delete_all),
        null: false

      add :topic, :string, null: false
      add :questions, {:array, :string}, default: []
      add :constraints, :map, default: %{}
      add :priority, :string, default: "normal"
      add :status, :string, null: false, default: "pending"

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_research_goals, [:session_id])
    create index(:atlas_research_goals, [:status])

    # ========================================================================
    # Investigations table (scientific investigations)
    # ========================================================================
    create table(:atlas_investigations, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :session_id, references(:atlas_learning_sessions, type: :binary_id, on_delete: :delete_all),
        null: false

      add :goal_id, references(:atlas_research_goals, type: :binary_id, on_delete: :nilify_all)

      add :topic, :string, null: false
      add :status, :string, null: false, default: "planning"
      add :conclusion, :string
      add :independent_variable, :string, default: "source"
      add :dependent_variable, :string, default: "claim"
      add :constants, {:array, :string}, default: []
      add :methodology_notes, :text
      add :started_at, :utc_datetime_usec
      add :concluded_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_investigations, [:session_id])
    create index(:atlas_investigations, [:goal_id])
    create index(:atlas_investigations, [:status])

    # ========================================================================
    # Hypotheses table (testable claims)
    # ========================================================================
    create table(:atlas_hypotheses, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :investigation_id,
          references(:atlas_investigations, type: :binary_id, on_delete: :delete_all),
          null: false

      add :claim, :text, null: false
      add :entity, :string
      add :derived_from, :text
      add :prediction, :text
      add :status, :string, null: false, default: "untested"
      add :confidence, :float, default: 0.0
      add :confidence_level, :string, default: "none"
      add :source_count, :integer, default: 0
      add :replication_count, :integer, default: 0
      add :tested_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_hypotheses, [:investigation_id])
    create index(:atlas_hypotheses, [:status])

    # ========================================================================
    # Evidence table (gathered findings linked to investigations)
    # ========================================================================
    create table(:atlas_evidence, primary_key: false) do
      add :id, :binary_id, primary_key: true

      add :investigation_id,
          references(:atlas_investigations, type: :binary_id, on_delete: :delete_all),
          null: false

      add :hypothesis_id, references(:atlas_hypotheses, type: :binary_id, on_delete: :nilify_all)

      add :claim, :text
      add :entity, :string
      add :entity_type, :string
      add :source_url, :string
      add :source_domain, :string
      add :source_title, :string
      add :source_reliability, :float
      add :source_bias, :string
      add :source_trust_tier, :string
      add :raw_context, :text
      add :confidence, :float, default: 0.5
      add :corroboration_group, :string
      add :evidence_type, :string, default: "unassociated"
      add :extracted_at, :utc_datetime_usec

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_evidence, [:investigation_id])
    create index(:atlas_evidence, [:hypothesis_id])
    create index(:atlas_evidence, [:entity])
    create index(:atlas_evidence, [:evidence_type])
  end
end
