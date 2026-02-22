defmodule Brain.Analysis.Types.IntentReviewCandidate do
  @moduledoc """
  A candidate utterance that may represent a novel intent, ready for admin review.
  """

  @type status :: :pending | :approved | :rejected | :deferred

  @type span_annotation :: %{
    start_pos: non_neg_integer(),
    end_pos: non_neg_integer(),
    text: String.t(),
    type: :entity_mention | :slot | :intent_cue,
    entity_type: String.t() | nil,
    slot_name: String.t() | nil,
    canonical_value: String.t() | nil
  }

  @type annotation :: %{
    tags: [atom()],
    notes: String.t() | nil,
    domain_guess: String.t() | nil,
    spans: [span_annotation()]
  }

  @type t :: %__MODULE__{
    id: String.t(),
    text: String.t(),
    timestamp: DateTime.t(),
    conversation_id: String.t() | nil,
    world_id: String.t() | nil,
    predicted_intent: String.t(),
    best_score: float(),
    second_score: float(),
    margin: float(),
    top_k: [{String.t(), float()}],
    extracted_entities: [map()],
    slot_fill_summary: map(),
    annotation: annotation(),
    status: status(),
    reviewed_at: DateTime.t() | nil,
    reviewer_notes: String.t() | nil,
    promotion_action: :variation | :new_intent | nil,
    promoted_to_intent: String.t() | nil
  }

  @enforce_keys [:id, :text, :timestamp, :predicted_intent, :best_score]
  defstruct [
    :id,
    :text,
    :timestamp,
    :conversation_id,
    :world_id,
    :predicted_intent,
    :best_score,
    :second_score,
    :margin,
    :top_k,
    :extracted_entities,
    :slot_fill_summary,
    :reviewed_at,
    :reviewer_notes,
    :promotion_action,
    :promoted_to_intent,
    annotation: %{tags: [], notes: nil, domain_guess: nil, spans: []},
    status: :pending
  ]

  @doc """
  Creates a new IntentReviewCandidate from analysis results.
  """
  def new(text, predicted_intent, best_score, opts \\ []) do
    %__MODULE__{
      id: generate_id(),
      text: text,
      timestamp: Keyword.get(opts, :timestamp, DateTime.utc_now()),
      conversation_id: Keyword.get(opts, :conversation_id),
      world_id: Keyword.get(opts, :world_id),
      predicted_intent: predicted_intent,
      best_score: best_score,
      second_score: Keyword.get(opts, :second_score, 0.0),
      margin: Keyword.get(opts, :margin, 0.0),
      top_k: Keyword.get(opts, :top_k, []),
      extracted_entities: Keyword.get(opts, :extracted_entities, []),
      slot_fill_summary: Keyword.get(opts, :slot_fill_summary, %{}),
      annotation: Keyword.get(opts, :annotation, %{tags: [], notes: nil, domain_guess: nil, spans: []}),
      promotion_action: Keyword.get(opts, :promotion_action),
      promoted_to_intent: Keyword.get(opts, :promoted_to_intent)
    }
  end

  @doc """
  Updates the annotation for a candidate.
  """
  def update_annotation(%__MODULE__{} = candidate, annotation_updates) when is_map(annotation_updates) do
    updated_annotation =
      candidate.annotation
      |> Map.merge(annotation_updates, fn
        _key, old_val, new_val when is_list(old_val) and is_list(new_val) ->
          # Merge lists (e.g., tags, spans)
          Enum.uniq(old_val ++ new_val)

        _key, _old_val, new_val ->
          new_val
      end)

    %{candidate | annotation: updated_annotation}
  end

  @doc """
  Marks a candidate as approved.
  """
  def approve(%__MODULE__{} = candidate, notes \\ nil, promotion_action \\ nil, promoted_to_intent \\ nil) do
    %{candidate |
      status: :approved,
      reviewed_at: DateTime.utc_now(),
      reviewer_notes: notes,
      promotion_action: promotion_action,
      promoted_to_intent: promoted_to_intent
    }
  end

  @doc """
  Marks a candidate as rejected.
  """
  def reject(%__MODULE__{} = candidate, notes \\ nil) do
    %{candidate |
      status: :rejected,
      reviewed_at: DateTime.utc_now(),
      reviewer_notes: notes
    }
  end

  @doc """
  Marks a candidate as deferred for later review.
  """
  def defer(%__MODULE__{} = candidate, notes \\ nil) do
    %{candidate |
      status: :deferred,
      reviewed_at: DateTime.utc_now(),
      reviewer_notes: notes
    }
  end

  defp generate_id do
    :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
  end
end
