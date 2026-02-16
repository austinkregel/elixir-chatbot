defmodule Atlas.Schemas.IntentReviewCandidate do
  @moduledoc """
  Ecto schema for intent review queue candidates.

  Replaces the file-based IntentReviewQueue `.term` persistence.
  Intent review candidates represent novel utterances that may
  represent new intents, ready for admin review.
  """

  use Ecto.Schema
  import Ecto.Changeset

  @primary_key {:id, :string, autogenerate: false}

  schema "atlas_intent_review_candidates" do
    field :text, :string
    field :status, :string, default: "pending"
    field :predicted_intent, :string
    field :best_score, :float
    field :second_score, :float
    field :margin, :float
    field :top_k, {:array, :map}, default: []
    field :extracted_entities, {:array, :map}, default: []
    field :slot_fill_summary, :map, default: %{}
    field :annotation, :map, default: %{}
    field :conversation_id, :string
    field :world_id, :string
    field :reviewer_notes, :string
    field :reviewed_at, :utc_datetime_usec
    field :promotion_action, :string
    field :promoted_to_intent, :string

    timestamps(type: :utc_datetime_usec)
  end

  @required_fields ~w(id text status predicted_intent best_score)a
  @optional_fields ~w(second_score margin top_k extracted_entities slot_fill_summary annotation conversation_id world_id reviewer_notes reviewed_at promotion_action promoted_to_intent)a

  def changeset(candidate, attrs) do
    candidate
    |> cast(attrs, @required_fields ++ @optional_fields)
    |> validate_required(@required_fields)
    |> validate_inclusion(:status, ~w(pending approved rejected deferred))
  end
end
