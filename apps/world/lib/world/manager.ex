defmodule World.Manager do
  @moduledoc "Manages training world lifecycle - creation, destruction, checkpointing.\n\nThis GenServer maintains the registry of active training worlds and\ncoordinates their data isolation while sharing computational processes.\n"

  alias Brain.ML.Gazetteer
  alias Brain.Memory.Store, as: MemoryStore
  use GenServer
  require Logger

  alias World.TrainingWorld
  alias World.Events, as: WorldEvents
  alias World.Metrics, as: WorldMetrics
  alias World.Persistence, as: WorldPersistence

  @ets_worlds :learning_worlds
  @ets_candidates :learning_candidates
  @ets_events :learning_events

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Creates a new training world.\n\n## Options\n  - `:mode` - :ephemeral (default) or :persistent\n  - `:base` - ID of parent world to inherit from (nil = empty)\n  - `:config` - Custom configuration map\n  - `:metadata` - Additional metadata\n"
  def create(name, opts \\ []) when is_binary(name) do
    GenServer.call(__MODULE__, {:create_world, name, opts}, 30_000)
  end

  @doc "Destroys a training world from memory while preserving persisted data.\n\nFor persistent worlds, this removes runtime ETS/Gazetteer state but leaves\ndisk checkpoints intact so the world can be loaded again via `load_world/1`.\n"
  def destroy(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:destroy_world, world_id}, 30_000)
  end

  @doc "Permanently deletes a training world from memory and disk.\n\nFor persistent worlds, this removes both runtime state and persisted files.\n"
  def purge(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:purge_world, world_id}, 30_000)
  end

  @doc "Gets a training world by ID.\n"
  def get(world_id) when is_binary(world_id) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] -> {:ok, world}
      [] -> {:error, :not_found}
    end
  rescue
    ArgumentError -> {:error, :table_not_ready}
  end

  @doc "Lists all active training worlds.\n\n## Options\n  - `:include_test` - If false (default), excludes test worlds from the list.\n                      Test worlds are identified by having `test: true` in metadata.\n"
  def list_worlds(opts \\ []) do
    include_test = Keyword.get(opts, :include_test, false)

    try do
      :ets.tab2list(@ets_worlds)
      |> Enum.filter(fn
        {id, %TrainingWorld{} = world} when is_binary(id) ->
          if include_test do
            true
          else
            not is_test_world?(world)
          end

        _ ->
          false
      end)
      |> Enum.map(fn {_id, world} -> world end)
    rescue
      ArgumentError -> []
    end
  end

  defp is_test_world?(%TrainingWorld{metadata: metadata}) when is_map(metadata) do
    Map.get(metadata, :test, false) == true
  end

  defp is_test_world?(_) do
    false
  end

  @doc "Gets the metrics for a training world.\n"
  def get_metrics(world_id) when is_binary(world_id) do
    case :ets.lookup(@ets_worlds, {:metrics, world_id}) do
      [{{:metrics, ^world_id}, metrics}] -> {:ok, metrics}
      [] -> {:error, :not_found}
    end
  rescue
    ArgumentError -> {:error, :table_not_ready}
  end

  @doc "Updates the metrics for a training world.\n\nThis is a non-blocking operation that updates ETS directly for performance.\n"
  def update_metrics(world_id, update_fn)
      when is_binary(world_id) and is_function(update_fn, 1) do
    try do
      case :ets.lookup(@ets_worlds, {:metrics, world_id}) do
        [{{:metrics, ^world_id}, metrics}] ->
          new_metrics = update_fn.(metrics)
          :ets.insert(@ets_worlds, {{:metrics, world_id}, new_metrics})
          {:ok, new_metrics}

        [] ->
          {:error, :not_found}
      end
    rescue
      ArgumentError -> {:error, :table_not_ready}
    end
  end

  @doc "Records an event in a training world's event log.\n"
  def record_event(world_id, event_type, data \\ %{}, opts \\ []) do
    GenServer.cast(__MODULE__, {:record_event, world_id, event_type, data, opts})
  end

  @doc "Gets all events for a training world, optionally filtered by type.\n"
  def get_events(world_id, filters \\ []) do
    type_filter = Keyword.get(filters, :type)
    limit = Keyword.get(filters, :limit, 1000)

    try do
      :ets.lookup(@ets_events, world_id)
      |> Enum.flat_map(fn {_world_id, events} -> events end)
      |> maybe_filter_by_type(type_filter)
      |> Enum.take(limit)
    rescue
      ArgumentError -> []
    end
  end

  @doc "Adds an entity candidate to a world's candidate pool.\n"
  def add_candidate(world_id, candidate) when is_binary(world_id) and is_map(candidate) do
    GenServer.cast(__MODULE__, {:add_candidate, world_id, candidate})
  end

  @doc "Gets all entity candidates for a world.\n"
  def get_candidates(world_id, opts \\ []) do
    try do
      case :ets.lookup(@ets_candidates, world_id) do
        [{^world_id, candidates}] ->
          candidates
          |> maybe_sort_candidates(Keyword.get(opts, :sort))
          |> Enum.take(Keyword.get(opts, :limit, 1000))

        [] ->
          []
      end
    rescue
      ArgumentError -> []
    end
  end

  @doc "Promotes a candidate to the world's gazetteer overlay.\n"
  def promote_candidate(world_id, candidate_value, entity_type) do
    GenServer.call(__MODULE__, {:promote_candidate, world_id, candidate_value, entity_type}, 30_000)
  end

  @doc "Creates a checkpoint for a persistent world.\n"
  def checkpoint(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:checkpoint, world_id}, 60_000)
  end

  @doc "Loads a persistent world from disk.\n"
  def load_world(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:load_world, world_id}, 60_000)
  end

  @doc "Exports a world's data for review.\n"
  def export(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:export, world_id}, 60_000)
  end

  @doc "Compares two worlds.\n"
  def compare(world_id_1, world_id_2) do
    with {:ok, metrics1} <- get_metrics(world_id_1),
         {:ok, metrics2} <- get_metrics(world_id_2) do
      {:ok, WorldMetrics.diff(metrics1, metrics2)}
    end
  end

  @doc "Merges approved learnings from source world to target world.\n"
  def merge(source_world_id, target_world_id, opts \\ []) do
    GenServer.call(__MODULE__, {:merge, source_world_id, target_world_id, opts}, 60_000)
  end

  @doc "Checks if the world manager is ready.\n"
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Reloads persisted worlds from disk.\n\nUseful when worlds have been saved by another process (e.g., mix task)\nand you want the running server to pick them up.\n"
  def reload_persisted_worlds do
    GenServer.call(__MODULE__, :reload_persisted_worlds, 60_000)
  end

  @doc "Cleans up orphaned world directories from disk.\n\nAn orphaned directory is one that exists on disk but:\n- Is not currently loaded in memory (ETS)\n- Has no valid config.json file\n\n## Options\n  - `:dry_run` - If true (default), only reports what would be deleted\n  - `:max_age_hours` - Only delete directories older than this (default: 24)\n  - `:exclude` - List of world IDs to never delete (default: [\"default\"])\n\nReturns `{:ok, deleted_count}` or `{:ok, {would_delete, directories}}` for dry run.\n"
  def cleanup_orphaned_worlds(opts \\ []) do
    GenServer.call(__MODULE__, {:cleanup_orphaned_worlds, opts}, 120_000)
  end

  @impl true
  def init(_opts) do
    create_tables()
    do_load_persisted_worlds()
    Logger.info("WorldManager started")
    {:ok, %{initialized: true}}
  end

  defp do_load_persisted_worlds do
    persisted = WorldPersistence.list_persisted_worlds()

    loaded =
      Enum.reduce(persisted, 0, fn world, count ->
        world_id = world.id

        case :ets.lookup(@ets_worlds, world_id) do
          [{^world_id, _}] ->
            count

          [] ->
            case WorldPersistence.load(world_id) do
              {:ok, data} ->
                :ets.insert(@ets_worlds, {world_id, data.world})
                :ets.insert(@ets_worlds, {{:metrics, world_id}, data.metrics})
                :ets.insert(@ets_candidates, {world_id, data.candidates})
                :ets.insert(@ets_events, {world_id, data.events})
                Gazetteer.restore_world_overlay(world_id, data.overlay)

                Logger.info("Loaded persisted world", %{id: world_id, name: world.name})
                count + 1

              {:error, reason} ->
                Logger.warning("Failed to load persisted world", %{id: world_id, reason: reason})
                count
            end
        end
      end)

    loaded
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:reload_persisted_worlds, _from, state) do
    loaded = do_load_persisted_worlds()
    {:reply, {:ok, loaded}, state}
  end

  @impl true
  def handle_call({:cleanup_orphaned_worlds, opts}, _from, state) do
    result = do_cleanup_orphaned_worlds(opts)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:create_world, name, opts}, _from, state) do
    world = TrainingWorld.new(name, opts)
    metrics = WorldMetrics.new()
    :ets.insert(@ets_worlds, {world.id, world})
    :ets.insert(@ets_worlds, {{:metrics, world.id}, metrics})
    :ets.insert(@ets_candidates, {world.id, []})
    :ets.insert(@ets_events, {world.id, []})

    try do
      Gazetteer.create_world_overlay(world.id)
    catch
      :exit, {:timeout, _} ->
        Logger.warning("Gazetteer overlay creation timed out for world #{world.id}, will retry on next access")
    end

    event = WorldEvents.new(world.id, :world_created, %{name: name, mode: world.mode})
    append_event(world.id, event)

    if world.config[:emit_telemetry] do
      WorldEvents.emit_telemetry(event)
    end

    if world.mode == :persistent do
      events = [event]
      overlay = Gazetteer.get_world_overlay(world.id)

      case WorldPersistence.save(world.id, %{
             world: world,
             metrics: metrics,
             candidates: [],
             events: events,
             overlay: overlay
           }) do
        :ok ->
          Logger.info("Auto-saved persistent world to disk", %{id: world.id})

        {:error, reason} ->
          Logger.warning("Failed to auto-save persistent world", %{id: world.id, reason: reason})
      end
    end

    Logger.info("Created training world", %{id: world.id, name: name, mode: world.mode})
    {:reply, {:ok, world}, state}
  end

  @impl true
  def handle_call({:destroy_world, world_id}, _from, state) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] ->
        drop_world_runtime_state(world_id)
        Logger.info("Destroyed training world runtime state", %{id: world_id, name: world.name})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:purge_world, world_id}, _from, state) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] ->
        drop_world_runtime_state(world_id)

        if world.mode == :persistent do
          case WorldPersistence.delete(world_id) do
            :ok ->
              Logger.info("Purged persisted world data from disk", %{id: world_id})

            {:error, reason} ->
              Logger.warning("Failed to purge persisted world data", %{
                id: world_id,
                reason: reason
              })
          end
        end

        Logger.info("Purged training world", %{id: world_id, name: world.name})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:promote_candidate, world_id, candidate_value, entity_type}, _from, state) do
    result = Gazetteer.add_to_world(world_id, candidate_value, entity_type)

    case result do
      {:ok, _} ->
        update_metrics_internal(world_id, &WorldMetrics.record_entity_promoted/1)

        event =
          WorldEvents.new(world_id, :entity_promoted_to_gazetteer, %{
            value: candidate_value,
            entity_type: entity_type
          })

        append_event(world_id, event)
        maybe_emit_telemetry(world_id, event)

        {:reply, :ok, state}

      error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:checkpoint, world_id}, _from, state) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] ->
        if world.mode == :persistent do
          metrics = get_metrics_internal(world_id)
          candidates = get_candidates(world_id)
          events = get_events(world_id)
          overlay = Gazetteer.get_world_overlay(world_id)

          episodes =
            case MemoryStore.all_episodes(world_id: world_id) do
              {:ok, eps} -> eps
              _ -> []
            end

          semantics =
            case MemoryStore.all_semantics(world_id: world_id) do
              {:ok, sems} -> sems
              _ -> []
            end

          result =
            WorldPersistence.save(world_id, %{
              world: world,
              metrics: metrics,
              candidates: candidates,
              events: events,
              overlay: overlay,
              episodes: episodes,
              semantics: semantics
            })

          case result do
            :ok ->
              event = WorldEvents.new(world_id, :checkpoint_created, %{})
              append_event(world_id, event)
              {:reply, :ok, state}

            error ->
              {:reply, error, state}
          end
        else
          {:reply, {:error, :ephemeral_world}, state}
        end

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:load_world, world_id}, _from, state) do
    case WorldPersistence.load(world_id) do
      {:ok, data} ->
        :ets.insert(@ets_worlds, {world_id, data.world})
        :ets.insert(@ets_worlds, {{:metrics, world_id}, data.metrics})
        :ets.insert(@ets_candidates, {world_id, data.candidates})
        :ets.insert(@ets_events, {world_id, data.events})
        Gazetteer.restore_world_overlay(world_id, data.overlay)

        {:reply, {:ok, data.world}, state}

      error ->
        {:reply, error, state}
    end
  end

  @impl true
  def handle_call({:export, world_id}, _from, state) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] ->
        export_data = %{
          world: world,
          metrics: get_metrics_internal(world_id),
          candidates: get_candidates(world_id),
          events: get_events(world_id),
          overlay: Gazetteer.get_world_overlay(world_id)
        }

        {:reply, {:ok, export_data}, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:merge, source_id, target_id, opts}, _from, state) do
    require_review = Keyword.get(opts, :require_review, true)
    min_confidence = Keyword.get(opts, :min_confidence, 0.7)

    with {:ok, _source} <- get(source_id),
         {:ok, _target} <- get(target_id) do
      source_overlay = Gazetteer.get_world_overlay(source_id)

      entities_to_merge =
        source_overlay
        |> Enum.filter(fn {_key, info} ->
          confidence = Map.get(info, :confidence, 1.0)
          confidence >= min_confidence
        end)

      if require_review and entities_to_merge != [] do
        {:reply, {:needs_review, entities_to_merge}, state}
      else
        Enum.each(entities_to_merge, fn {_key, info} ->
          Gazetteer.add_to_world(
            target_id,
            info.value,
            info.entity_type,
            Map.drop(info, [:value, :entity_type])
          )
        end)

        {:reply, {:ok, length(entities_to_merge)}, state}
      end
    end
  end

  @impl true
  def handle_cast({:record_event, world_id, event_type, data, opts}, state) do
    event = WorldEvents.new(world_id, event_type, data, opts)
    append_event(world_id, event)
    maybe_emit_telemetry(world_id, event)
    {:noreply, state}
  end

  @impl true
  def handle_cast({:add_candidate, world_id, candidate}, state) do
    case :ets.lookup(@ets_candidates, world_id) do
      [{^world_id, candidates}] ->
        candidate_value = Map.get(candidate, :value)

        {updated_candidates, is_new} =
          case Enum.find_index(candidates, &(Map.get(&1, :value) == candidate_value)) do
            nil ->
              {[candidate | candidates], true}

            idx ->
              existing = Enum.at(candidates, idx)
              updated = Map.update(existing, :occurrences, 1, &(&1 + 1))
              {List.replace_at(candidates, idx, updated), false}
          end

        limited = Enum.take(updated_candidates, 10_000)
        :ets.insert(@ets_candidates, {world_id, limited})

        if is_new do
          entity_type = Map.get(candidate, :inferred_type, "unknown")
          confidence = Map.get(candidate, :confidence, 0.5)

          update_metrics_internal(
            world_id,
            &WorldMetrics.record_entity_discovered(&1, entity_type, confidence)
          )
        end

      [] ->
        :ok
    end

    {:noreply, state}
  end

  defp create_tables do
    if :ets.whereis(@ets_worlds) != :undefined do
      :ets.delete(@ets_worlds)
    end

    :ets.new(@ets_worlds, [:set, :public, :named_table, read_concurrency: true])

    if :ets.whereis(@ets_candidates) != :undefined do
      :ets.delete(@ets_candidates)
    end

    :ets.new(@ets_candidates, [:set, :public, :named_table, read_concurrency: true])

    if :ets.whereis(@ets_events) != :undefined do
      :ets.delete(@ets_events)
    end

    :ets.new(@ets_events, [:set, :public, :named_table, read_concurrency: true])
  end

  defp append_event(world_id, event) do
    case :ets.lookup(@ets_events, world_id) do
      [{^world_id, events}] ->
        updated = [event | events] |> Enum.take(10_000)
        :ets.insert(@ets_events, {world_id, updated})

      [] ->
        :ets.insert(@ets_events, {world_id, [event]})
    end
  end

  defp maybe_filter_by_type(events, nil) do
    events
  end

  defp maybe_filter_by_type(events, type) do
    Enum.filter(events, &(&1.type == type))
  end

  defp maybe_sort_candidates(candidates, nil) do
    candidates
  end

  defp maybe_sort_candidates(candidates, :confidence) do
    Enum.sort_by(candidates, &Map.get(&1, :confidence, 0), :desc)
  end

  defp maybe_sort_candidates(candidates, :occurrences) do
    Enum.sort_by(candidates, &Map.get(&1, :occurrences, 1), :desc)
  end

  defp maybe_sort_candidates(candidates, _) do
    candidates
  end

  defp get_metrics_internal(world_id) do
    case :ets.lookup(@ets_worlds, {:metrics, world_id}) do
      [{{:metrics, ^world_id}, metrics}] -> metrics
      [] -> WorldMetrics.new()
    end
  end

  defp update_metrics_internal(world_id, update_fn) do
    case :ets.lookup(@ets_worlds, {:metrics, world_id}) do
      [{{:metrics, ^world_id}, metrics}] ->
        new_metrics = update_fn.(metrics)
        :ets.insert(@ets_worlds, {{:metrics, world_id}, new_metrics})

      [] ->
        :ok
    end
  end

  defp maybe_emit_telemetry(world_id, event) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] ->
        if world.config[:emit_telemetry] do
          WorldEvents.emit_telemetry(event)
        end

      [] ->
        :ok
    end
  end

  defp do_cleanup_orphaned_worlds(opts) do
    dry_run = Keyword.get(opts, :dry_run, true)
    max_age_hours = Keyword.get(opts, :max_age_hours, 24)
    exclude = Keyword.get(opts, :exclude, ["default"])

    base_path = WorldPersistence.base_path()

    if File.exists?(base_path) do
      case File.ls(base_path) do
        {:ok, entries} ->
          loaded_world_ids =
            try do
              :ets.tab2list(@ets_worlds)
              |> Enum.filter(fn
                {id, %TrainingWorld{}} when is_binary(id) -> true
                _ -> false
              end)
              |> Enum.map(fn {id, _} -> id end)
              |> MapSet.new()
            rescue
              ArgumentError -> MapSet.new()
            end

          cutoff = DateTime.add(DateTime.utc_now(), -max_age_hours * 3600, :second)

          orphaned =
            entries
            |> Enum.filter(&File.dir?(Path.join(base_path, &1)))
            |> Enum.reject(&(&1 in exclude))
            |> Enum.reject(&MapSet.member?(loaded_world_ids, &1))
            |> Enum.filter(fn world_id ->
              dir_path = Path.join(base_path, world_id)
              config_path = Path.join(dir_path, "config.json")
              has_valid_config = File.exists?(config_path)

              case File.stat(dir_path) do
                {:ok, %{mtime: mtime}} ->
                  case NaiveDateTime.from_erl(mtime) do
                    {:ok, naive} ->
                      dir_time = DateTime.from_naive!(naive, "Etc/UTC")
                      is_old = DateTime.compare(dir_time, cutoff) == :lt
                      not has_valid_config or is_old

                    _ ->
                      false
                  end

                _ ->
                  false
              end
            end)

          if dry_run do
            {:ok, {:would_delete, orphaned}}
          else
            deleted =
              Enum.reduce(orphaned, 0, fn world_id, count ->
                case WorldPersistence.delete(world_id) do
                  :ok ->
                    Logger.info("Cleaned up orphaned world directory", %{world_id: world_id})
                    count + 1

                  {:error, reason} ->
                    Logger.warning("Failed to cleanup orphaned world", %{
                      world_id: world_id,
                      reason: reason
                    })

                    count
                end
              end)

            {:ok, deleted}
          end

        {:error, reason} ->
          {:error, reason}
      end
    else
      {:ok, 0}
    end
  end

  defp drop_world_runtime_state(world_id) do
    :ets.delete(@ets_worlds, world_id)
    :ets.delete(@ets_worlds, {:metrics, world_id})
    :ets.delete(@ets_candidates, world_id)
    :ets.delete(@ets_events, world_id)
    Gazetteer.destroy_world_overlay(world_id)
  end
end
