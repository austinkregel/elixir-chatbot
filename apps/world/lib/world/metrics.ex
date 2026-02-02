defmodule World.Metrics do
  @moduledoc """
  Aggregates metrics for a training world.

  Tracks both expected metrics and anomalies, providing full observability
  into what the learning system is discovering and how it's behaving.
  """

  @type t :: %__MODULE__{
          # Size metrics
          documents_processed: non_neg_integer(),
          total_tokens: non_neg_integer(),
          total_sentences: non_neg_integer(),
          # Discovery metrics
          entities_discovered: non_neg_integer(),
          entities_promoted: non_neg_integer(),
          entities_by_type: %{String.t() => non_neg_integer()},
          ambiguous_entities: list(map()),
          # Confidence distribution
          confidence_histogram: %{String.t() => non_neg_integer()},
          low_confidence_entities: list(map()),
          high_confidence_entities: list(map()),
          # Co-occurrence tracking
          cooccurrence_counts: %{{String.t(), String.t()} => non_neg_integer()},
          # Anomalies
          anomalies: list(map()),
          type_conflicts: list(map()),
          # Timing
          started_at: DateTime.t() | nil,
          last_updated: DateTime.t() | nil,
          processing_time_ms: non_neg_integer()
        }

  defstruct documents_processed: 0,
            total_tokens: 0,
            total_sentences: 0,
            entities_discovered: 0,
            entities_promoted: 0,
            entities_by_type: %{},
            ambiguous_entities: [],
            confidence_histogram: %{},
            low_confidence_entities: [],
            high_confidence_entities: [],
            cooccurrence_counts: %{},
            anomalies: [],
            type_conflicts: [],
            started_at: nil,
            last_updated: nil,
            processing_time_ms: 0

  @doc """
  Creates a new metrics struct with initial timestamp.
  """
  def new do
    %__MODULE__{
      started_at: DateTime.utc_now(),
      last_updated: DateTime.utc_now()
    }
  end

  @doc """
  Updates the last_updated timestamp.
  """
  def touch(%__MODULE__{} = metrics) do
    %{metrics | last_updated: DateTime.utc_now()}
  end

  @doc """
  Increments document count and updates timing.
  """
  def record_document(%__MODULE__{} = metrics, token_count, sentence_count, processing_ms) do
    %{
      metrics
      | documents_processed: metrics.documents_processed + 1,
        total_tokens: metrics.total_tokens + token_count,
        total_sentences: metrics.total_sentences + sentence_count,
        processing_time_ms: metrics.processing_time_ms + processing_ms,
        last_updated: DateTime.utc_now()
    }
  end

  @doc """
  Records a newly discovered entity candidate.
  """
  def record_entity_discovered(%__MODULE__{} = metrics, entity_type, confidence) do
    # Update count by type
    type_counts = Map.update(metrics.entities_by_type, entity_type, 1, &(&1 + 1))

    # Update confidence histogram (bucket by 0.1 increments)
    bucket = bucket_confidence(confidence)
    histogram = Map.update(metrics.confidence_histogram, bucket, 1, &(&1 + 1))

    %{
      metrics
      | entities_discovered: metrics.entities_discovered + 1,
        entities_by_type: type_counts,
        confidence_histogram: histogram,
        last_updated: DateTime.utc_now()
    }
  end

  @doc """
  Records an entity promoted to gazetteer.
  """
  def record_entity_promoted(%__MODULE__{} = metrics) do
    %{
      metrics
      | entities_promoted: metrics.entities_promoted + 1,
        last_updated: DateTime.utc_now()
    }
  end

  @doc """
  Records an ambiguous entity (multiple possible types).
  """
  def record_ambiguity(%__MODULE__{} = metrics, entity_info) do
    # Limit stored ambiguities to prevent memory growth
    ambiguities =
      [entity_info | metrics.ambiguous_entities]
      |> Enum.take(1000)

    %{metrics | ambiguous_entities: ambiguities, last_updated: DateTime.utc_now()}
  end

  @doc """
  Records a co-occurrence between two entities.
  """
  def record_cooccurrence(%__MODULE__{} = metrics, entity1, entity2) do
    # Normalize key order for consistent counting
    key =
      if entity1 <= entity2 do
        {entity1, entity2}
      else
        {entity2, entity1}
      end

    cooccurrences = Map.update(metrics.cooccurrence_counts, key, 1, &(&1 + 1))
    %{metrics | cooccurrence_counts: cooccurrences, last_updated: DateTime.utc_now()}
  end

  @doc """
  Records an anomaly event.
  """
  def record_anomaly(%__MODULE__{} = metrics, anomaly_info) do
    anomalies =
      [anomaly_info | metrics.anomalies]
      |> Enum.take(500)

    %{metrics | anomalies: anomalies, last_updated: DateTime.utc_now()}
  end

  @doc """
  Records a type conflict (entity typed differently in different contexts).
  """
  def record_type_conflict(%__MODULE__{} = metrics, conflict_info) do
    conflicts =
      [conflict_info | metrics.type_conflicts]
      |> Enum.take(500)

    %{metrics | type_conflicts: conflicts, last_updated: DateTime.utc_now()}
  end

  @doc """
  Compares two worlds' metrics, returning a diff report.
  Useful for A/B testing.
  """
  def diff(%__MODULE__{} = world1, %__MODULE__{} = world2) do
    %{
      entity_count_diff: world1.entities_discovered - world2.entities_discovered,
      promoted_diff: world1.entities_promoted - world2.entities_promoted,
      type_distribution_diff: diff_maps(world1.entities_by_type, world2.entities_by_type),
      confidence_diff: diff_maps(world1.confidence_histogram, world2.confidence_histogram),
      unique_to_world1: unique_types(world1.entities_by_type, world2.entities_by_type),
      unique_to_world2: unique_types(world2.entities_by_type, world1.entities_by_type),
      anomaly_count_diff: length(world1.anomalies) - length(world2.anomalies),
      conflict_count_diff: length(world1.type_conflicts) - length(world2.type_conflicts),
      processing_time_diff_ms: world1.processing_time_ms - world2.processing_time_ms
    }
  end

  @doc """
  Returns a summary of the metrics for display.
  """
  def summary(%__MODULE__{} = metrics) do
    %{
      documents: metrics.documents_processed,
      tokens: metrics.total_tokens,
      sentences: metrics.total_sentences,
      entities_discovered: metrics.entities_discovered,
      entities_promoted: metrics.entities_promoted,
      entity_types: Map.keys(metrics.entities_by_type),
      ambiguity_count: length(metrics.ambiguous_entities),
      anomaly_count: length(metrics.anomalies),
      conflict_count: length(metrics.type_conflicts),
      processing_time_ms: metrics.processing_time_ms,
      started_at: metrics.started_at,
      last_updated: metrics.last_updated
    }
  end

  # Private helpers

  defp bucket_confidence(nil), do: "unknown"

  defp bucket_confidence(confidence) when is_number(confidence) do
    bucket = trunc(confidence * 10) / 10
    "#{Float.round(bucket, 1)}"
  end

  defp diff_maps(map1, map2) do
    all_keys = MapSet.union(MapSet.new(Map.keys(map1)), MapSet.new(Map.keys(map2)))

    Enum.reduce(all_keys, %{}, fn key, acc ->
      val1 = Map.get(map1, key, 0)
      val2 = Map.get(map2, key, 0)
      diff = val1 - val2

      if diff != 0 do
        Map.put(acc, key, diff)
      else
        acc
      end
    end)
  end

  defp unique_types(map1, map2) do
    map1_keys = MapSet.new(Map.keys(map1))
    map2_keys = MapSet.new(Map.keys(map2))

    MapSet.difference(map1_keys, map2_keys)
    |> MapSet.to_list()
  end
end
