defmodule World.Manager do
  @moduledoc """
  Manages training world lifecycle - creation, destruction, checkpointing.

  This GenServer maintains the registry of active training worlds and
  coordinates their data isolation while sharing computational processes.
  """

  use GenServer
  require Logger

  alias World.TrainingWorld
  alias World.Events, as: WorldEvents
  alias World.Metrics, as: WorldMetrics
  alias World.Persistence, as: WorldPersistence

  @ets_worlds :learning_worlds
  @ets_candidates :learning_candidates
  @ets_events :learning_events

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Creates a new training world.

  ## Options
    - `:mode` - :ephemeral (default) or :persistent
    - `:base` - ID of parent world to inherit from (nil = empty)
    - `:config` - Custom configuration map
    - `:metadata` - Additional metadata
  """
  def create(name, opts \\ []) when is_binary(name) do
    GenServer.call(__MODULE__, {:create_world, name, opts})
  end

  @doc """
  Destroys a training world and cleans up its data.
  """
  def destroy(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:destroy_world, world_id})
  end

  @doc """
  Gets a training world by ID.
  """
  def get(world_id) when is_binary(world_id) do
    case :ets.lookup(@ets_worlds, world_id) do
      [{^world_id, world}] -> {:ok, world}
      [] -> {:error, :not_found}
    end
  rescue
    ArgumentError -> {:error, :table_not_ready}
  end

  @doc """
  Lists all active training worlds.

  ## Options
    - `:include_test` - If false (default), excludes test worlds from the list.
                        Test worlds are identified by having `test: true` in metadata.
  """
  def list_worlds(opts \\ []) do
    include_test = Keyword.get(opts, :include_test, false)

    try do
      :ets.tab2list(@ets_worlds)
      |> Enum.filter(fn
        {id, %TrainingWorld{} = world} when is_binary(id) ->
          # Filter out test worlds unless explicitly requested
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

  defp is_test_world?(_), do: false

  @doc """
  Gets the metrics for a training world.
  """
  def get_metrics(world_id) when is_binary(world_id) do
    case :ets.lookup(@ets_worlds, {:metrics, world_id}) do
      [{{:metrics, ^world_id}, metrics}] -> {:ok, metrics}
      [] -> {:error, :not_found}
    end
  rescue
    ArgumentError -> {:error, :table_not_ready}
  end

  @doc """
  Updates the metrics for a training world.

  This is a non-blocking operation that updates ETS directly for performance.
  """
  def update_metrics(world_id, update_fn)
      when is_binary(world_id) and is_function(update_fn, 1) do
    # Direct ETS update for performance (table is public)
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

  @doc """
  Records an event in a training world's event log.
  """
  def record_event(world_id, event_type, data \\ %{}, opts \\ []) do
    GenServer.cast(__MODULE__, {:record_event, world_id, event_type, data, opts})
  end

  @doc """
  Gets all events for a training world, optionally filtered by type.
  """
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

  @doc """
  Adds an entity candidate to a world's candidate pool.
  """
  def add_candidate(world_id, candidate) when is_binary(world_id) and is_map(candidate) do
    GenServer.cast(__MODULE__, {:add_candidate, world_id, candidate})
  end

  @doc """
  Gets all entity candidates for a world.
  """
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

  @doc """
  Promotes a candidate to the world's gazetteer overlay.
  """
  def promote_candidate(world_id, candidate_value, entity_type) do
    GenServer.call(__MODULE__, {:promote_candidate, world_id, candidate_value, entity_type})
  end

  @doc """
  Creates a checkpoint for a persistent world.
  """
  def checkpoint(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:checkpoint, world_id}, 60_000)
  end

  @doc """
  Loads a persistent world from disk.
  """
  def load_world(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:load_world, world_id}, 60_000)
  end

  @doc """
  Exports a world's data for review.
  """
  def export(world_id) when is_binary(world_id) do
    GenServer.call(__MODULE__, {:export, world_id})
  end

  @doc """
  Compares two worlds.
  """
  def compare(world_id_1, world_id_2) do
    with {:ok, metrics1} <- get_metrics(world_id_1),
         {:ok, metrics2} <- get_metrics(world_id_2) do
      {:ok, WorldMetrics.diff(metrics1, metrics2)}
    end
  end

  @doc """
  Merges approved learnings from source world to target world.
  """
  def merge(source_world_id, target_world_id, opts \\ []) do
    GenServer.call(__MODULE__, {:merge, source_world_id, target_world_id, opts}, 60_000)
  end

  @doc """
  Checks if the world manager is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc """
  Reloads persisted worlds from disk.

  Useful when worlds have been saved by another process (e.g., mix task)
  and you want the running server to pick them up.
  """
  def reload_persisted_worlds do
    GenServer.call(__MODULE__, :reload_persisted_worlds, 60_000)
  end

  @doc """
  Cleans up orphaned world directories from disk.

  An orphaned directory is one that exists on disk but:
  - Is not currently loaded in memory (ETS)
  - Has no valid config.json file

  ## Options
    - `:dry_run` - If true (default), only reports what would be deleted
    - `:max_age_hours` - Only delete directories older than this (default: 24)
    - `:exclude` - List of world IDs to never delete (default: ["default"])

  Returns `{:ok, deleted_count}` or `{:ok, {would_delete, directories}}` for dry run.
  """
  def cleanup_orphaned_worlds(opts \\ []) do
    GenServer.call(__MODULE__, {:cleanup_orphaned_worlds, opts}, 120_000)
  end

  # ============================================================================
  # GenServer Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    create_tables()
    do_load_persisted_worlds()
    Logger.info("WorldManager started")
    {:ok, %{initialized: true}}
  end

  defp do_load_persisted_worlds do
    # Load all persisted worlds from disk
    persisted = WorldPersistence.list_persisted_worlds()

    loaded =
      Enum.reduce(persisted, 0, fn world, count ->
        world_id = world.id

        # Check if already loaded
        case :ets.lookup(@ets_worlds, world_id) do
          [{^world_id, _}] ->
            # Already loaded, skip
            count

          [] ->
            # Not loaded, load from disk
            case WorldPersistence.load(world_id) do
              {:ok, data} ->
                # Restore to ETS
                :ets.insert(@ets_worlds, {world_id, data.world})
                :ets.insert(@ets_worlds, {{:metrics, world_id}, data.metrics})
                :ets.insert(@ets_candidates, {world_id, data.candidates})
                :ets.insert(@ets_events, {world_id, data.events})

                # Restore gazetteer overlay
                Brain.ML.Gazetteer.restore_world_overlay(world_id, data.overlay)

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

    # Store world and metrics
    :ets.insert(@ets_worlds, {world.id, world})
    :ets.insert(@ets_worlds, {{:metrics, world.id}, metrics})
    :ets.insert(@ets_candidates, {world.id, []})
    :ets.insert(@ets_events, {world.id, []})

    # Initialize gazetteer overlay for this world
    Brain.ML.Gazetteer.create_world_overlay(world.id)

    # Record creation event
    event = WorldEvents.new(world.id, :world_created, %{name: name, mode: world.mode})
    append_event(world.id, event)

    if world.config[:emit_telemetry] do
      WorldEvents.emit_telemetry(event)
    end

    # Auto-save persistent worlds to disk immediately
    if world.mode == :persistent do
      events = [event]
      overlay = Brain.ML.Gazetteer.get_world_overlay(world.id)

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
        # Clean up ETS entries
        :ets.delete(@ets_worlds, world_id)
        :ets.delete(@ets_worlds, {:metrics, world_id})
        :ets.delete(@ets_candidates, world_id)
        :ets.delete(@ets_events, world_id)

        # Clean up gazetteer overlay
        Brain.ML.Gazetteer.destroy_world_overlay(world_id)

        # Clean up persisted data from disk (for persistent worlds)
        if world.mode == :persistent do
          case WorldPersistence.delete(world_id) do
            :ok ->
              Logger.info("Deleted persisted world data from disk", %{id: world_id})

            {:error, reason} ->
              Logger.warning("Failed to delete persisted world data", %{
                id: world_id,
                reason: reason
              })
          end
        end

        Logger.info("Destroyed training world", %{id: world_id, name: world.name})
        {:reply, :ok, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:promote_candidate, world_id, candidate_value, entity_type}, _from, state) do
    # Add to gazetteer overlay
    result = Brain.ML.Gazetteer.add_to_world(world_id, candidate_value, entity_type)

    case result do
      {:ok, _} ->
        # Update metrics
        update_metrics_internal(world_id, &WorldMetrics.record_entity_promoted/1)

        # Record event
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
          overlay = Brain.ML.Gazetteer.get_world_overlay(world_id)

          result =
            WorldPersistence.save(world_id, %{
              world: world,
              metrics: metrics,
              candidates: candidates,
              events: events,
              overlay: overlay
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
        # Restore to ETS
        :ets.insert(@ets_worlds, {world_id, data.world})
        :ets.insert(@ets_worlds, {{:metrics, world_id}, data.metrics})
        :ets.insert(@ets_candidates, {world_id, data.candidates})
        :ets.insert(@ets_events, {world_id, data.events})

        # Restore gazetteer overlay
        Brain.ML.Gazetteer.restore_world_overlay(world_id, data.overlay)

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
          overlay: Brain.ML.Gazetteer.get_world_overlay(world_id)
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
      # Get promoted entities from source overlay
      source_overlay = Brain.ML.Gazetteer.get_world_overlay(source_id)

      # Filter by confidence if required
      entities_to_merge =
        source_overlay
        |> Enum.filter(fn {_key, info} ->
          confidence = Map.get(info, :confidence, 1.0)
          confidence >= min_confidence
        end)

      if require_review and length(entities_to_merge) > 0 do
        # Return entities for review instead of merging
        {:reply, {:needs_review, entities_to_merge}, state}
      else
        # Merge entities to target
        Enum.each(entities_to_merge, fn {_key, info} ->
          Brain.ML.Gazetteer.add_to_world(
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
        # Check if candidate already exists (by value)
        candidate_value = Map.get(candidate, :value)

        {updated_candidates, is_new} =
          case Enum.find_index(candidates, &(Map.get(&1, :value) == candidate_value)) do
            nil ->
              # New candidate
              {[candidate | candidates], true}

            idx ->
              # Existing candidate - update occurrence count
              existing = Enum.at(candidates, idx)
              updated = Map.update(existing, :occurrences, 1, &(&1 + 1))
              {List.replace_at(candidates, idx, updated), false}
          end

        # Limit candidates to prevent unbounded growth
        limited = Enum.take(updated_candidates, 10_000)
        :ets.insert(@ets_candidates, {world_id, limited})

        # Update metrics
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

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp create_tables do
    # Main worlds table
    if :ets.whereis(@ets_worlds) != :undefined do
      :ets.delete(@ets_worlds)
    end

    :ets.new(@ets_worlds, [:set, :public, :named_table, read_concurrency: true])

    # Candidates table
    if :ets.whereis(@ets_candidates) != :undefined do
      :ets.delete(@ets_candidates)
    end

    :ets.new(@ets_candidates, [:set, :public, :named_table, read_concurrency: true])

    # Events table
    if :ets.whereis(@ets_events) != :undefined do
      :ets.delete(@ets_events)
    end

    :ets.new(@ets_events, [:set, :public, :named_table, read_concurrency: true])
  end

  defp append_event(world_id, event) do
    case :ets.lookup(@ets_events, world_id) do
      [{^world_id, events}] ->
        # Prepend for efficiency, limit to prevent unbounded growth
        updated = [event | events] |> Enum.take(10_000)
        :ets.insert(@ets_events, {world_id, updated})

      [] ->
        :ets.insert(@ets_events, {world_id, [event]})
    end
  end

  defp maybe_filter_by_type(events, nil), do: events

  defp maybe_filter_by_type(events, type) do
    Enum.filter(events, &(&1.type == type))
  end

  defp maybe_sort_candidates(candidates, nil), do: candidates

  defp maybe_sort_candidates(candidates, :confidence) do
    Enum.sort_by(candidates, &Map.get(&1, :confidence, 0), :desc)
  end

  defp maybe_sort_candidates(candidates, :occurrences) do
    Enum.sort_by(candidates, &Map.get(&1, :occurrences, 1), :desc)
  end

  defp maybe_sort_candidates(candidates, _), do: candidates

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
          # Get all currently loaded world IDs
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

          # Calculate cutoff time
          cutoff = DateTime.add(DateTime.utc_now(), -max_age_hours * 3600, :second)

          # Find orphaned directories
          orphaned =
            entries
            |> Enum.filter(&File.dir?(Path.join(base_path, &1)))
            |> Enum.reject(&(&1 in exclude))
            |> Enum.reject(&MapSet.member?(loaded_world_ids, &1))
            |> Enum.filter(fn world_id ->
              dir_path = Path.join(base_path, world_id)
              config_path = Path.join(dir_path, "config.json")

              # Check if it's a valid world (has config.json)
              has_valid_config = File.exists?(config_path)

              # Check directory age
              case File.stat(dir_path) do
                {:ok, %{mtime: mtime}} ->
                  # Convert mtime (erlang datetime) to DateTime
                  case NaiveDateTime.from_erl(mtime) do
                    {:ok, naive} ->
                      dir_time = DateTime.from_naive!(naive, "Etc/UTC")
                      is_old = DateTime.compare(dir_time, cutoff) == :lt

                      # Delete if: no valid config OR old enough
                      not has_valid_config or is_old

                    _ ->
                      # Can't parse time, skip
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
end
