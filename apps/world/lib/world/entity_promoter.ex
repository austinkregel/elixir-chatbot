defmodule World.EntityPromoter do
  @moduledoc """
  Periodically scans entity candidates across worlds and suggests the ones
  it has seen often enough, and confidently enough, for human review.

  It never adds to the gazetteer itself: the gazetteer learns only what a
  reviewer approves (`Brain.Knowledge.ReviewQueue`). Each suggestion is a
  pending review candidate scoped to its world; once approved, the entity
  joins that world's gazetteer overlay.

  An entity is suggested only when:
  - it has been observed at least `min_occurrences` times in the world -- the
    minimum sample before the promoter trusts what it has seen,
  - its confidence, averaged over those observations, is at least
    `min_confidence`,
  - its type is determined (not "unknown"),
  - it passes the knowledge-graph gate, and
  - the queue does not already hold it for this world and type, in any
    status -- so a reviewer is not asked twice, even about something they
    rejected.

  The thresholds come from `config :world, World.EntityPromoter`.
  Scans every 10 minutes.
  """

  use GenServer
  require Logger

  alias Brain.Knowledge.ReviewQueue
  alias Brain.Knowledge.Types.{Finding, ReviewCandidate, SourceInfo}

  @scan_interval_ms 10 * 60 * 1000

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

  @doc """
  Suggests for review every entity among a world's `candidates` that meets
  the criteria in the moduledoc, and returns the aggregated entities it
  suggested.

  ## Options

  - `:config` -- keyword list replacing `config :world, World.EntityPromoter`.
  """
  @spec suggest(String.t(), [map()], keyword()) :: [map()]
  def suggest(world_id, candidates, opts \\ []) when is_binary(world_id) and is_list(candidates) do
    config = Keyword.get_lazy(opts, :config, fn -> Application.fetch_env!(:world, __MODULE__) end)
    min_occurrences = Keyword.fetch!(config, :min_occurrences)
    min_confidence = Keyword.fetch!(config, :min_confidence)

    candidates
    |> World.EntityDiscoverer.aggregate_discoveries()
    |> Enum.filter(fn entity ->
      entity.occurrences >= min_occurrences and
        entity.confidence >= min_confidence and
        entity.inferred_type not in [nil, "unknown"] and
        not ReviewQueue.has_entity_candidate?(entity.value, entity.inferred_type, world_id) and
        passes_kg_gate?(entity, world_id)
    end)
    |> Enum.map(fn entity ->
      case ReviewQueue.add(review_candidate(entity, world_id)) do
        {:ok, _id} ->
          entity

        {:error, reason} ->
          raise "EntityPromoter: review queue refused #{inspect(entity.value)} " <>
                  "(#{entity.inferred_type}) from world #{world_id}: #{inspect(reason)}"
      end
    end)
  end

  @impl true
  def init(_opts) do
    Process.send_after(self(), :scan, @scan_interval_ms)

    {:ok,
     %{
       total_suggested: 0,
       last_scan: nil,
       suggested_entities: []
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
    suggested =
      Enum.flat_map(list_worlds(), fn world_id ->
        suggest(world_id, World.Manager.get_candidates(world_id))
      end)

    if suggested != [] do
      Logger.info("EntityPromoter: suggested #{length(suggested)} entities for review",
        entities: Enum.map(suggested, & &1.value)
      )
    end

    %{
      state
      | total_suggested: state.total_suggested + length(suggested),
        last_scan: DateTime.utc_now(),
        suggested_entities: (suggested ++ state.suggested_entities) |> Enum.take(100)
    }
  end

  defp review_candidate(entity, world_id) do
    source =
      SourceInfo.new("world://#{world_id}/entity_promoter",
        title: "Entities observed in world #{world_id}"
      )

    finding =
      Finding.new("#{entity.value} is a #{entity.inferred_type}", entity.value, source,
        entity_type: entity.inferred_type,
        confidence: entity.confidence,
        raw_context: entity.contexts |> Enum.filter(&is_binary/1) |> Enum.join("\n"),
        world_id: world_id
      )

    ReviewCandidate.new(finding, aggregate_confidence: entity.confidence)
  end

  defp list_worlds do
    Enum.map(World.Manager.list_worlds(), & &1.id)
  end

  defp passes_kg_gate?(entity, _world_id) do
    config = Application.get_env(:brain, :kg_signals, [])

    if Keyword.get(config, :enabled, true) and Keyword.get(config, :entity_promoter_kg_gate, true) do
      entity_has_quality_triple?(entity.value)
    else
      true
    end
  end

  defp entity_has_quality_triple?(entity_name) do
    unless Brain.ML.KnowledgeGraph.TripleScorer.ready?() do
      true
    else
      canonical_relations =
        Brain.ML.KnowledgeGraph.PredicateNormalizer.canonical_relations()
        |> Enum.take(10)

      Enum.any?(canonical_relations, fn relation ->
        case Brain.ML.KnowledgeGraph.TripleScorer.score(entity_name, relation, "entity") do
          {:ok, score} when score >= 0.4 -> true
          _ -> false
        end
      end)
    end
  rescue
    _ -> true
  end
end
