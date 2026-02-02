defmodule World.Events do
  @moduledoc """
  All observable events in a training world.

  Every learning action emits an event for full observability.
  Events are stored in the world's event log and can trigger telemetry.
  """

  # Entity Discovery
  @type event_type ::
          :entity_candidate_detected
          | :entity_candidate_updated
          | :entity_promoted_to_gazetteer
          | :entity_type_inferred
          | :entity_ambiguity_detected
          | :entity_ambiguity_resolved
          | :entity_occurrence
          # Type Inference
          | :context_pattern_learned
          | :cooccurrence_detected
          | :type_confidence_changed
          # Memory/Learning
          | :fact_learned
          | :memory_created
          | :memory_consolidated
          # Anomalies/Unexpected
          | :unexpected_pattern
          | :confidence_anomaly
          | :type_conflict
          # Processing
          | :document_processed
          | :batch_complete
          | :world_created
          | :world_destroyed
          | :checkpoint_created

  @type t :: %__MODULE__{
          id: String.t(),
          world_id: String.t(),
          type: event_type(),
          timestamp: DateTime.t(),
          data: map(),
          context: map(),
          confidence: float() | nil,
          previous_state: map() | nil,
          new_state: map() | nil
        }

  @enforce_keys [:id, :world_id, :type, :timestamp]
  defstruct [
    :id,
    :world_id,
    :type,
    :timestamp,
    data: %{},
    context: %{},
    confidence: nil,
    previous_state: nil,
    new_state: nil
  ]

  @doc """
  Creates a new event.
  """
  def new(world_id, type, data \\ %{}, opts \\ []) do
    %__MODULE__{
      id: generate_id(),
      world_id: world_id,
      type: type,
      timestamp: DateTime.utc_now(),
      data: data,
      context: Keyword.get(opts, :context, %{}),
      confidence: Keyword.get(opts, :confidence),
      previous_state: Keyword.get(opts, :previous_state),
      new_state: Keyword.get(opts, :new_state)
    }
  end

  @doc """
  Emits a telemetry event for this world event.
  """
  def emit_telemetry(%__MODULE__{} = event) do
    :telemetry.execute(
      [:chat_bot, :learning, event.type],
      %{confidence: event.confidence || 0.0, count: 1},
      %{
        world_id: event.world_id,
        event_id: event.id,
        data: event.data,
        context: event.context,
        timestamp: event.timestamp
      }
    )
  end

  defp generate_id do
    :crypto.strong_rand_bytes(8)
    |> Base.url_encode64(padding: false)
  end
end
