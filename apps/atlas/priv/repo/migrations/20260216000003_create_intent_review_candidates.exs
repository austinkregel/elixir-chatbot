defmodule Atlas.Repo.Migrations.CreateIntentReviewCandidates do
  use Ecto.Migration

  def change do
    create table(:atlas_intent_review_candidates, primary_key: false) do
      add :id, :string, primary_key: true
      add :text, :text, null: false
      add :status, :string, null: false, default: "pending"
      add :predicted_intent, :string, null: false
      add :best_score, :float, null: false
      add :second_score, :float
      add :margin, :float
      add :top_k, {:array, :map}, default: []
      add :extracted_entities, {:array, :map}, default: []
      add :slot_fill_summary, :map, default: %{}
      add :annotation, :map, default: %{}
      add :conversation_id, :string
      add :world_id, :string
      add :reviewer_notes, :text
      add :reviewed_at, :utc_datetime_usec
      add :promotion_action, :string
      add :promoted_to_intent, :string

      timestamps(type: :utc_datetime_usec)
    end

    create index(:atlas_intent_review_candidates, [:status])
    create index(:atlas_intent_review_candidates, [:predicted_intent])
    create index(:atlas_intent_review_candidates, [:world_id])
  end
end
