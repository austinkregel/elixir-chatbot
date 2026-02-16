defmodule World.EntityPromoter do
  @moduledoc """
  Periodically scans entity candidates across worlds and auto-promotes
  frequently-occurring, high-confidence entities to the Gazetteer.

  Promotion criteria:
  - Entity has >= 3 occurrences
  - Aggregated confidence >= 0.6
  - Entity type is determined (not "unknown")

  Uses `Gazetteer.add_to_world/4` for world-scoped isolation.
  Scans every 10 minutes.
  """

  use GenServer
  require Logger

  @scan_interval_ms 10 * 60 * 1000
  @min_occurrences 3
  @min_confidence 0.6

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns current promoter stats."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats, 5_000)
  end

  @doc "Checks if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @doc "Trigger a manual scan."
  def scan_now(name \\ __MODULE__) do
    GenServer.cast(name, :scan)
  end

  @impl true
  def init(_opts) do
    Process.send_after(self(), :scan, @scan_interval_ms)

    {:ok,
     %{
       total_promoted: 0,
       last_scan: nil,
       promoted_entities: []
     }}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_cast(:scan, state) do
    state = do_scan(state)
    {:noreply, state}
  end

  @impl true
  def handle_info(:scan, state) do
    state = do_scan(state)
    Process.send_after(self(), :scan, @scan_interval_ms)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # --- Private ---

  defp do_scan(state) do
    if gazetteer_available?() and manager_available?() do
      worlds = list_worlds()

      promoted =
        Enum.flat_map(worlds, fn world_id ->
          scan_world(world_id)
        end)

      if promoted != [] do
        Logger.info("EntityPromoter: promoted #{length(promoted)} entities",
          entities: Enum.map(promoted, & &1.value)
        )
      end

      %{
        state
        | total_promoted: state.total_promoted + length(promoted),
          last_scan: DateTime.utc_now(),
          promoted_entities: (promoted ++ state.promoted_entities) |> Enum.take(100)
      }
    else
      state
    end
  rescue
    e ->
      Logger.debug("EntityPromoter scan error: #{Exception.message(e)}")
      state
  end

  defp scan_world(world_id) do
    candidates = World.Manager.get_candidates(world_id)

    # Aggregate by entity value
    aggregated = World.EntityDiscoverer.aggregate_discoveries(candidates)

    # Filter for promotion-eligible entities
    promotable =
      Enum.filter(aggregated, fn entity ->
        entity.occurrences >= @min_occurrences and
          entity.confidence >= @min_confidence and
          entity.inferred_type != "unknown" and
          entity.inferred_type != nil
      end)

    # Promote each eligible entity
    Enum.flat_map(promotable, fn entity ->
      case promote_entity(entity, world_id) do
        :ok -> [entity]
        _ -> []
      end
    end)
  rescue
    _ -> []
  end

  defp promote_entity(entity, world_id) do
    Brain.ML.Gazetteer.add_to_world(
      world_id,
      entity.value,
      entity.inferred_type,
      %{
        source: :auto_promoted,
        confidence: entity.confidence,
        occurrences: entity.occurrences,
        promoted_at: DateTime.utc_now()
      }
    )

    :ok
  rescue
    _ -> :error
  end

  defp list_worlds do
    World.Manager.list_worlds()
    |> Enum.map(fn
      {id, _config} -> id
      %{id: id} -> id
      id when is_binary(id) -> id
      _ -> nil
    end)
    |> Enum.reject(&is_nil/1)
  rescue
    _ -> []
  end

  defp gazetteer_available? do
    try do
      Brain.ML.Gazetteer.loaded?()
    catch
      :exit, _ -> false
    end
  rescue
    _ -> false
  end

  defp manager_available? do
    Process.whereis(World.Manager) != nil
  rescue
    _ -> false
  end
end
