defmodule ChatBot.Learning.WorldPersistence do
  @moduledoc """
  Persistence layer for training worlds.

  Handles saving and loading world data:
  - Persistent worlds: Saved to JSON files in priv/training_worlds/
  - Ephemeral worlds: In-memory only (ETS), no disk persistence

  Directory structure:
    priv/training_worlds/{world_id}/
      config.json          - World configuration
      gazetteer_overlay.json - Entities added in this world
      discovered_entities.json - Candidates pending promotion
      events.jsonl         - Append-only event log
      metrics.json         - Aggregated metrics
      type_inferrer.json   - Learned type inference data
  """

  require Logger

  alias ChatBot.Learning.{TrainingWorld, WorldMetrics, WorldEvents, TypeInferrer}

  @base_path "priv/training_worlds"

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Returns the base path for training world storage.
  """
  def base_path do
    Application.get_env(:chat_bot, :training_worlds_path, @base_path)
  end

  @doc """
  Returns the path for a specific world's data.
  """
  def world_path(world_id) do
    Path.join(base_path(), world_id)
  end

  @doc """
  Saves all world data to disk.

  Only works for persistent worlds.
  """
  def save(world_id, data) when is_binary(world_id) and is_map(data) do
    world = Map.get(data, :world)

    if world && world.mode == :persistent do
      path = world_path(world_id)

      with :ok <- ensure_directory(path),
           :ok <- save_config(path, world),
           :ok <- save_metrics(path, Map.get(data, :metrics)),
           :ok <- save_candidates(path, Map.get(data, :candidates, [])),
           :ok <- save_overlay(path, Map.get(data, :overlay, [])),
           :ok <- save_events(path, Map.get(data, :events, [])),
           :ok <- save_type_inferrer(path) do
        Logger.info("Saved training world", %{world_id: world_id, path: path})
        :ok
      else
        {:error, reason} = error ->
          Logger.error("Failed to save training world", %{world_id: world_id, reason: reason})
          error
      end
    else
      {:error, :ephemeral_world}
    end
  end

  @doc """
  Loads a world from disk.
  """
  def load(world_id) when is_binary(world_id) do
    path = world_path(world_id)

    if File.exists?(path) do
      with {:ok, world} <- load_config(path),
           {:ok, metrics} <- load_metrics(path),
           {:ok, candidates} <- load_candidates(path),
           {:ok, overlay} <- load_overlay(path),
           {:ok, events} <- load_events(path) do
        # Load type inferrer data
        load_type_inferrer(path)

        {:ok,
         %{
           world: world,
           metrics: metrics,
           candidates: candidates,
           overlay: overlay,
           events: events
         }}
      else
        {:error, reason} = error ->
          Logger.error("Failed to load training world", %{world_id: world_id, reason: reason})
          error
      end
    else
      {:error, :not_found}
    end
  end

  @doc """
  Deletes all persisted data for a world.
  """
  def delete(world_id) when is_binary(world_id) do
    path = world_path(world_id)

    if File.exists?(path) do
      case File.rm_rf(path) do
        {:ok, _} ->
          Logger.info("Deleted training world data", %{world_id: world_id})
          :ok

        {:error, reason, _} ->
          {:error, reason}
      end
    else
      :ok
    end
  end

  @doc """
  Lists all persisted worlds.
  """
  def list_persisted_worlds do
    path = base_path()

    if File.exists?(path) do
      case File.ls(path) do
        {:ok, entries} ->
          entries
          |> Enum.filter(&File.dir?(Path.join(path, &1)))
          |> Enum.map(fn world_id ->
            config_path = Path.join([path, world_id, "config.json"])

            if File.exists?(config_path) do
              case load_config(Path.join(path, world_id)) do
                {:ok, world} -> world
                _ -> nil
              end
            else
              nil
            end
          end)
          |> Enum.filter(&(&1 != nil))

        {:error, _} ->
          []
      end
    else
      []
    end
  end

  @doc """
  Appends an event to the event log file.

  More efficient than rewriting the entire events file.
  """
  def append_event(world_id, %WorldEvents{} = event) do
    path = world_path(world_id)
    events_path = Path.join(path, "events.jsonl")

    if File.exists?(path) do
      event_json = encode_event(event)

      case File.open(events_path, [:append, :utf8]) do
        {:ok, file} ->
          IO.write(file, event_json <> "\n")
          File.close(file)
          :ok

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:error, :world_not_found}
    end
  end

  # ============================================================================
  # Private Functions - Saving
  # ============================================================================

  defp ensure_directory(path) do
    case File.mkdir_p(path) do
      :ok -> :ok
      {:error, reason} -> {:error, {:mkdir_failed, reason}}
    end
  end

  defp save_config(path, %TrainingWorld{} = world) do
    config_path = Path.join(path, "config.json")

    data = %{
      id: world.id,
      name: world.name,
      mode: Atom.to_string(world.mode),
      base_world: world.base_world,
      created_at: DateTime.to_iso8601(world.created_at),
      config: world.config,
      metadata: world.metadata
    }

    write_json(config_path, data)
  end

  defp save_metrics(path, %WorldMetrics{} = metrics) do
    metrics_path = Path.join(path, "metrics.json")

    data = %{
      documents_processed: metrics.documents_processed,
      total_tokens: metrics.total_tokens,
      total_sentences: metrics.total_sentences,
      entities_discovered: metrics.entities_discovered,
      entities_promoted: metrics.entities_promoted,
      entities_by_type: metrics.entities_by_type,
      ambiguous_entities: metrics.ambiguous_entities,
      confidence_histogram: metrics.confidence_histogram,
      low_confidence_entities: metrics.low_confidence_entities,
      high_confidence_entities: metrics.high_confidence_entities,
      cooccurrence_counts: encode_cooccurrence_counts(metrics.cooccurrence_counts),
      anomalies: metrics.anomalies,
      type_conflicts: metrics.type_conflicts,
      started_at: encode_datetime(metrics.started_at),
      last_updated: encode_datetime(metrics.last_updated),
      processing_time_ms: metrics.processing_time_ms
    }

    write_json(metrics_path, data)
  end

  defp save_metrics(_path, nil), do: :ok

  defp save_candidates(path, candidates) when is_list(candidates) do
    candidates_path = Path.join(path, "discovered_entities.json")

    data =
      Enum.map(candidates, fn candidate ->
        candidate
        |> Map.update(:discovered_at, nil, &encode_datetime/1)
      end)

    write_json(candidates_path, data)
  end

  defp save_overlay(path, overlay) when is_list(overlay) do
    overlay_path = Path.join(path, "gazetteer_overlay.json")

    data =
      Enum.map(overlay, fn {key, info} ->
        %{key: key, info: info}
      end)

    write_json(overlay_path, data)
  end

  defp save_events(path, events) when is_list(events) do
    events_path = Path.join(path, "events.jsonl")

    content =
      events
      |> Enum.reverse()
      |> Enum.map(&encode_event/1)
      |> Enum.join("\n")

    case File.write(events_path, content <> "\n") do
      :ok -> :ok
      {:error, reason} -> {:error, {:write_events_failed, reason}}
    end
  end

  defp save_type_inferrer(path) do
    inferrer_path = Path.join(path, "type_inferrer.json")
    data = TypeInferrer.export_learned_data()
    write_json(inferrer_path, data)
  end

  # ============================================================================
  # Private Functions - Loading
  # ============================================================================

  defp load_config(path) do
    config_path = Path.join(path, "config.json")

    case read_json(config_path) do
      {:ok, data} ->
        world = %TrainingWorld{
          id: Map.get(data, "id"),
          name: Map.get(data, "name"),
          mode: String.to_existing_atom(Map.get(data, "mode", "ephemeral")),
          base_world: Map.get(data, "base_world"),
          created_at: parse_datetime(Map.get(data, "created_at")),
          config: atomize_keys(Map.get(data, "config", %{})),
          metadata: Map.get(data, "metadata", %{})
        }

        {:ok, world}

      error ->
        error
    end
  end

  defp load_metrics(path) do
    metrics_path = Path.join(path, "metrics.json")

    case read_json(metrics_path) do
      {:ok, data} ->
        metrics = %WorldMetrics{
          documents_processed: Map.get(data, "documents_processed", 0),
          total_tokens: Map.get(data, "total_tokens", 0),
          total_sentences: Map.get(data, "total_sentences", 0),
          entities_discovered: Map.get(data, "entities_discovered", 0),
          entities_promoted: Map.get(data, "entities_promoted", 0),
          entities_by_type: Map.get(data, "entities_by_type", %{}),
          ambiguous_entities: Map.get(data, "ambiguous_entities", []),
          confidence_histogram: Map.get(data, "confidence_histogram", %{}),
          low_confidence_entities: Map.get(data, "low_confidence_entities", []),
          high_confidence_entities: Map.get(data, "high_confidence_entities", []),
          cooccurrence_counts: decode_cooccurrence_counts(Map.get(data, "cooccurrence_counts", %{})),
          anomalies: Map.get(data, "anomalies", []),
          type_conflicts: Map.get(data, "type_conflicts", []),
          started_at: parse_datetime(Map.get(data, "started_at")),
          last_updated: parse_datetime(Map.get(data, "last_updated")),
          processing_time_ms: Map.get(data, "processing_time_ms", 0)
        }

        {:ok, metrics}

      {:error, :enoent} ->
        {:ok, WorldMetrics.new()}

      error ->
        error
    end
  end

  defp load_candidates(path) do
    candidates_path = Path.join(path, "discovered_entities.json")

    case read_json(candidates_path) do
      {:ok, data} when is_list(data) ->
        candidates =
          Enum.map(data, fn candidate ->
            candidate
            |> atomize_keys()
            |> Map.update(:discovered_at, nil, &parse_datetime/1)
          end)

        {:ok, candidates}

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  defp load_overlay(path) do
    overlay_path = Path.join(path, "gazetteer_overlay.json")

    case read_json(overlay_path) do
      {:ok, data} when is_list(data) ->
        overlay =
          Enum.map(data, fn item ->
            key = Map.get(item, "key")
            info = atomize_keys(Map.get(item, "info", %{}))
            {key, info}
          end)

        {:ok, overlay}

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  defp load_events(path) do
    events_path = Path.join(path, "events.jsonl")

    if File.exists?(events_path) do
      case File.read(events_path) do
        {:ok, content} ->
          events =
            content
            |> String.split("\n", trim: true)
            |> Enum.map(&decode_event/1)
            |> Enum.filter(&(&1 != nil))
            |> Enum.reverse()

          {:ok, events}

        {:error, reason} ->
          {:error, {:read_events_failed, reason}}
      end
    else
      {:ok, []}
    end
  end

  defp load_type_inferrer(path) do
    inferrer_path = Path.join(path, "type_inferrer.json")

    case read_json(inferrer_path) do
      {:ok, data} ->
        # Convert string keys back to atoms where needed
        patterns =
          Map.get(data, "patterns", %{})
          |> Enum.into(%{}, fn {k, v} -> {k, v} end)

        cooccurrences =
          Map.get(data, "cooccurrences", %{})
          |> Enum.into(%{}, fn {k, v} -> {k, v} end)

        TypeInferrer.import_learned_data(%{patterns: patterns, cooccurrences: cooccurrences})

      {:error, :enoent} ->
        :ok

      _ ->
        :ok
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp write_json(path, data) do
    case Jason.encode(data, pretty: true) do
      {:ok, json} ->
        case File.write(path, json) do
          :ok -> :ok
          {:error, reason} -> {:error, {:write_failed, path, reason}}
        end

      {:error, reason} ->
        {:error, {:encode_failed, reason}}
    end
  end

  defp read_json(path) do
    case File.read(path) do
      {:ok, content} ->
        case Jason.decode(content) do
          {:ok, data} -> {:ok, data}
          {:error, reason} -> {:error, {:decode_failed, reason}}
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp encode_event(%WorldEvents{} = event) do
    data = %{
      id: event.id,
      world_id: event.world_id,
      type: Atom.to_string(event.type),
      timestamp: DateTime.to_iso8601(event.timestamp),
      data: event.data,
      context: event.context,
      confidence: event.confidence,
      previous_state: event.previous_state,
      new_state: event.new_state
    }

    Jason.encode!(data)
  end

  defp decode_event(json) do
    case Jason.decode(json) do
      {:ok, data} ->
        %WorldEvents{
          id: Map.get(data, "id"),
          world_id: Map.get(data, "world_id"),
          type: String.to_existing_atom(Map.get(data, "type", "unknown")),
          timestamp: parse_datetime(Map.get(data, "timestamp")),
          data: Map.get(data, "data", %{}),
          context: Map.get(data, "context", %{}),
          confidence: Map.get(data, "confidence"),
          previous_state: Map.get(data, "previous_state"),
          new_state: Map.get(data, "new_state")
        }

      _ ->
        nil
    end
  rescue
    _ -> nil
  end

  defp encode_datetime(nil), do: nil
  defp encode_datetime(%DateTime{} = dt), do: DateTime.to_iso8601(dt)
  defp encode_datetime(other), do: other

  defp parse_datetime(nil), do: nil
  defp parse_datetime(""), do: nil

  defp parse_datetime(str) when is_binary(str) do
    case DateTime.from_iso8601(str) do
      {:ok, dt, _offset} -> dt
      _ -> nil
    end
  end

  defp parse_datetime(other), do: other

  defp encode_cooccurrence_counts(counts) when is_map(counts) do
    Enum.into(counts, %{}, fn {{a, b}, count} ->
      {"#{a}|#{b}", count}
    end)
  end

  defp decode_cooccurrence_counts(encoded) when is_map(encoded) do
    Enum.into(encoded, %{}, fn {key, count} ->
      case String.split(key, "|", parts: 2) do
        [a, b] -> {{a, b}, count}
        _ -> {{key, ""}, count}
      end
    end)
  end

  defp atomize_keys(map) when is_map(map) do
    Enum.into(map, %{}, fn {k, v} ->
      key =
        if is_binary(k) do
          try do
            String.to_existing_atom(k)
          rescue
            _ -> String.to_atom(k)
          end
        else
          k
        end

      value = if is_map(v), do: atomize_keys(v), else: v
      {key, value}
    end)
  end

  defp atomize_keys(other), do: other
end
