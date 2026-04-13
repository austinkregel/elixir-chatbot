defmodule World.Persistence do
  @moduledoc "Persistence layer for training worlds.\n\nHandles saving and loading world data:\n- Persistent worlds: Saved to JSON files in priv/training_worlds/\n- Ephemeral worlds: In-memory only (ETS), no disk persistence\n\nDirectory structure:\n  priv/training_worlds/{world_id}/\n    config.json          - World configuration\n    gazetteer_overlay.json - Entities added in this world\n    discovered_entities.json - Candidates pending promotion\n    events.jsonl         - Append-only event log\n    metrics.json         - Aggregated metrics\n    type_inferrer.json   - Learned type inference data\n    episodes.json        - Episodic memories (NEW)\n    semantics.json       - Semantic facts (NEW)\n    knowledge.json       - Learned knowledge (NEW)\n    intents/             - World-specific intent data (NEW)\n    models/              - World-specific trained models (NEW)\n"

  alias Brain.Memory.Store
  alias Brain.Memory.Types
  require Logger

  alias World.{TrainingWorld, TypeInferrer}
  alias World.Metrics, as: WorldMetrics
  alias World.Events, as: WorldEvents
  alias Types.{Episode, SemanticFact}

  @doc "Returns the base path for training world storage.\n\nIn test mode (when `:test_world_sandbox` is enabled), returns a temp directory\nto ensure test worlds are completely isolated from production worlds.\n"
  def base_path do
    if Application.get_env(:world, :test_world_sandbox) do
      Path.join(System.tmp_dir!(), "chat_bot_test_worlds")
    else
      Application.get_env(:world, :training_worlds_path) || World.priv_path("training_worlds")
    end
  end

  @doc "Returns the path for a specific world's data.\n"
  def world_path(world_id) do
    Path.join(base_path(), world_id)
  end

  @doc "Saves all world data to disk.\n\nOnly works for persistent worlds.\n"
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
           :ok <- save_type_inferrer(path),
           :ok <- save_episodes(path, Map.get(data, :episodes, [])),
           :ok <- save_semantics(path, Map.get(data, :semantics, [])),
           :ok <- save_knowledge(path, Map.get(data, :knowledge, %{})) do
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

  @doc "Loads a world from disk.\n"
  def load(world_id) when is_binary(world_id) do
    path = world_path(world_id)

    if File.exists?(path) do
      with {:ok, world} <- load_config(path),
           {:ok, metrics} <- load_metrics(path),
           {:ok, candidates} <- load_candidates(path),
           {:ok, overlay} <- load_overlay(path),
           {:ok, events} <- load_events(path),
           {:ok, episodes} <- load_episodes(path),
           {:ok, semantics} <- load_semantics(path),
           {:ok, knowledge} <- load_knowledge(path) do
        load_type_inferrer(path)

        {:ok,
         %{
           world: world,
           metrics: metrics,
           candidates: candidates,
           overlay: overlay,
           events: events,
           episodes: episodes,
           semantics: semantics,
           knowledge: knowledge
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

  @doc "Deletes all persisted data for a world.\n"
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

  @doc "Lists all persisted worlds.\n"
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

  @doc "Saves world-specific memory (episodes and semantics) to disk.\n"
  def save_memory(world_id) when is_binary(world_id) do
    path = world_path(world_id)

    if File.exists?(path) do
      case Store.all_episodes(world_id: world_id) do
        {:ok, episodes} ->
          case Store.all_semantics(world_id: world_id) do
            {:ok, semantics} ->
              with :ok <- save_episodes(path, episodes),
                   :ok <- save_semantics(path, semantics) do
                Logger.info("Saved world memory", %{
                  world_id: world_id,
                  episodes: length(episodes),
                  semantics: length(semantics)
                })

                :ok
              end

            error ->
              error
          end

        error ->
          error
      end
    else
      {:error, :world_not_found}
    end
  end

  @doc "Loads world-specific memory (episodes and semantics) into the Memory.Store.\n"
  def load_memory(world_id) when is_binary(world_id) do
    path = world_path(world_id)

    if File.exists?(path) do
      with {:ok, episodes} <- load_episodes(path),
           {:ok, semantics} <- load_semantics(path) do
        Enum.each(episodes, fn episode ->
          Store.add_episode_direct(episode, world_id: world_id)
        end)

        Enum.each(semantics, fn semantic ->
          Store.add_semantic(semantic, world_id: world_id)
        end)

        Logger.info("Loaded world memory", %{
          world_id: world_id,
          episodes: length(episodes),
          semantics: length(semantics)
        })

        {:ok, %{episodes: length(episodes), semantics: length(semantics)}}
      end
    else
      {:error, :world_not_found}
    end
  end

  @doc "Appends an event to the event log file.\n\nMore efficient than rewriting the entire events file.\n"
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

  defp save_metrics(_path, nil) do
    :ok
  end

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
      |> Enum.map_join(
        "\n",
        &encode_event/1
      )

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

  defp save_episodes(path, episodes) when is_list(episodes) do
    episodes_path = Path.join(path, "episodes.json")

    data =
      Enum.map(episodes, fn episode ->
        %{
          id: episode.id,
          state: episode.state,
          action: episode.action,
          outcome: episode.outcome,
          tags: episode.tags,
          timestamp: episode.timestamp,
          semantic_id: episode.semantic_id,
          embedding: episode.embedding
        }
      end)

    write_json(episodes_path, data)
  end

  defp save_episodes(_path, _) do
    :ok
  end

  defp save_semantics(path, semantics) when is_list(semantics) do
    semantics_path = Path.join(path, "semantics.json")

    data =
      Enum.map(semantics, fn semantic ->
        %{
          id: semantic.id,
          representation: semantic.representation,
          evidence_ids: semantic.evidence_ids,
          tags: semantic.tags,
          timestamp: semantic.timestamp,
          embedding: semantic.embedding
        }
      end)

    write_json(semantics_path, data)
  end

  defp save_semantics(_path, _) do
    :ok
  end

  defp save_knowledge(path, knowledge) when is_map(knowledge) do
    knowledge_path = Path.join(path, "knowledge.json")
    write_json(knowledge_path, knowledge)
  end

  defp save_knowledge(_path, _) do
    :ok
  end

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
          cooccurrence_counts:
            decode_cooccurrence_counts(Map.get(data, "cooccurrence_counts", %{})),
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

  defp load_episodes(path) do
    episodes_path = Path.join(path, "episodes.json")

    case read_json(episodes_path) do
      {:ok, data} when is_list(data) ->
        episodes =
          Enum.map(data, fn ep ->
            %Episode{
              id: Map.get(ep, "id"),
              state: Map.get(ep, "state", ""),
              action: Map.get(ep, "action", ""),
              outcome: Map.get(ep, "outcome", ""),
              tags: Map.get(ep, "tags", []),
              timestamp: Map.get(ep, "timestamp", 0),
              semantic_id: Map.get(ep, "semantic_id"),
              embedding: Map.get(ep, "embedding", [])
            }
          end)

        {:ok, episodes}

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  defp load_semantics(path) do
    semantics_path = Path.join(path, "semantics.json")

    case read_json(semantics_path) do
      {:ok, data} when is_list(data) ->
        semantics =
          Enum.map(data, fn sem ->
            %SemanticFact{
              id: Map.get(sem, "id"),
              representation: Map.get(sem, "representation", ""),
              evidence_ids: Map.get(sem, "evidence_ids", []),
              tags: Map.get(sem, "tags", []),
              timestamp: Map.get(sem, "timestamp", 0),
              embedding: Map.get(sem, "embedding", [])
            }
          end)

        {:ok, semantics}

      {:error, :enoent} ->
        {:ok, []}

      error ->
        error
    end
  end

  defp load_knowledge(path) do
    knowledge_path = Path.join(path, "knowledge.json")

    case read_json(knowledge_path) do
      {:ok, data} when is_map(data) ->
        {:ok, data}

      {:error, :enoent} ->
        {:ok, %{}}

      error ->
        error
    end
  end

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

  defp encode_datetime(nil) do
    nil
  end

  defp encode_datetime(%DateTime{} = dt) do
    DateTime.to_iso8601(dt)
  end

  defp encode_datetime(other) do
    other
  end

  defp parse_datetime(nil) do
    nil
  end

  defp parse_datetime("") do
    nil
  end

  defp parse_datetime(str) when is_binary(str) do
    case DateTime.from_iso8601(str) do
      {:ok, dt, _offset} -> dt
      _ -> nil
    end
  end

  defp parse_datetime(other) do
    other
  end

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
            ArgumentError -> k
          end
        else
          k
        end

      value =
        if is_map(v) do
          atomize_keys(v)
        else
          v
        end

      {key, value}
    end)
  end

  defp atomize_keys(other) do
    other
  end
end
