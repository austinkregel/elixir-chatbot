defmodule Brain.Analysis.IntentReviewQueue do
  @moduledoc """
  ETS-backed queue for pending intent review candidates.

  Stores candidate utterances with full analysis metadata, tracks review status,
  and supports bulk operations. Data is persisted to disk for durability.

  ## Features

  - Fast concurrent reads via ETS
  - Persistence to disk
  - Status tracking (pending, approved, rejected, deferred)
  - Bulk approve/reject operations
  - Annotation support (tags, notes, span annotations)

  ## Example

      candidate = IntentReviewCandidate.new("What's the weather?", "weather.query", 0.45)
      {:ok, _} = IntentReviewQueue.add(candidate)
      
      pending = IntentReviewQueue.get_pending()
      {:ok, approved} = IntentReviewQueue.approve(candidate.id, "Variation of weather.query")
  """

  use GenServer
  require Logger

  alias Brain.Analysis.Types.IntentReviewCandidate

  @ets_table :intent_review_queue

  defp persistence_path do
    configured = Application.get_env(:brain, :intent_review_queue_path)
    if configured do
      configured
    else
      Brain.priv_path("data/intent_review_queue.term")
    end
  end

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adds a candidate to the review queue.
  """
  @spec add(IntentReviewCandidate.t()) :: {:ok, String.t()} | {:error, term()}
  def add(%IntentReviewCandidate{} = candidate) do
    GenServer.call(__MODULE__, {:add, candidate})
  end

  @doc """
  Gets all pending candidates.

  ## Options
    - :limit - Maximum number to return (default: 100)
    - :sort_by - Sort field (:novelty_score, :created_at, :margin)
    - :filter - Filter function
  """
  @spec get_pending(keyword()) :: [IntentReviewCandidate.t()]
  def get_pending(opts \\ []) do
    GenServer.call(__MODULE__, {:get_pending, opts})
  end

  @doc """
  Gets candidates by status.

  ## Options
    - :limit - Maximum number to return (default: 100)
    - :sort_by - Sort field (:novelty_score, :created_at, :reviewed_at)
  """
  @spec get_by_status(atom(), keyword()) :: [IntentReviewCandidate.t()]
  def get_by_status(status, opts \\ []) when status in [:pending, :approved, :rejected, :deferred] do
    GenServer.call(__MODULE__, {:get_by_status, status, opts})
  end

  @doc """
  Gets a specific candidate by ID.
  """
  @spec get(String.t()) :: {:ok, IntentReviewCandidate.t()} | {:error, :not_found}
  def get(id) when is_binary(id) do
    GenServer.call(__MODULE__, {:get, id})
  end

  @doc """
  Updates the annotation for a candidate.
  """
  @spec update_annotation(String.t(), map()) :: {:ok, IntentReviewCandidate.t()} | {:error, term()}
  def update_annotation(id, annotation_updates) when is_binary(id) and is_map(annotation_updates) do
    GenServer.call(__MODULE__, {:update_annotation, id, annotation_updates})
  end

  @doc """
  Approves a candidate.
  """
  @spec approve(String.t(), String.t() | nil, atom() | nil, String.t() | nil) :: {:ok, IntentReviewCandidate.t()} | {:error, term()}
  def approve(id, notes \\ nil, promotion_action \\ nil, promoted_to_intent \\ nil) when is_binary(id) do
    GenServer.call(__MODULE__, {:approve, id, notes, promotion_action, promoted_to_intent})
  end

  @doc """
  Rejects a candidate.
  """
  @spec reject(String.t(), String.t() | nil) :: {:ok, IntentReviewCandidate.t()} | {:error, term()}
  def reject(id, notes \\ nil) when is_binary(id) do
    GenServer.call(__MODULE__, {:reject, id, notes})
  end

  @doc """
  Defers a candidate for later review.
  """
  @spec defer(String.t(), String.t() | nil) :: {:ok, IntentReviewCandidate.t()} | {:error, term()}
  def defer(id, notes \\ nil) when is_binary(id) do
    GenServer.call(__MODULE__, {:defer, id, notes})
  end

  @doc """
  Bulk approves multiple candidates.
  """
  @spec bulk_approve([String.t()]) :: {:ok, non_neg_integer()}
  def bulk_approve(ids) when is_list(ids) do
    GenServer.call(__MODULE__, {:bulk_approve, ids}, 60_000)
  end

  @doc """
  Bulk rejects multiple candidates.
  """
  @spec bulk_reject([String.t()]) :: {:ok, non_neg_integer()}
  def bulk_reject(ids) when is_list(ids) do
    GenServer.call(__MODULE__, {:bulk_reject, ids}, 60_000)
  end

  @doc """
  Gets queue statistics.
  """
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Clears all candidates (useful for testing).
  """
  @spec clear() :: :ok
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc """
  Persists the queue to disk.
  """
  @spec persist() :: :ok | {:error, term()}
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc """
  Checks if the queue is ready.
  """
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    # Create ETS table
    table = :ets.new(@ets_table, [:set, :public, :named_table, read_concurrency: true])

    state = %{
      table: table,
      stats: %{
        pending: 0,
        approved: 0,
        rejected: 0,
        deferred: 0,
        approved_today: 0,
        rejected_today: 0,
        last_reset: Date.utc_today()
      }
    }

    # Load from disk
    state = load_from_disk(state)

    Logger.info("IntentReviewQueue initialized", pending: state.stats.pending)

    {:ok, state}
  end

  @impl true
  def handle_call({:add, candidate}, _from, state) do
    :ets.insert(@ets_table, {candidate.id, candidate})

    new_stats = %{state.stats | pending: state.stats.pending + 1}
    new_state = %{state | stats: new_stats}

    # Auto-persist
    persist_to_disk(new_state)

    # Broadcast update
    broadcast_update(:candidate_added, candidate)

    {:reply, {:ok, candidate.id}, new_state}
  end

  @impl true
  def handle_call({:get_pending, opts}, _from, state) do
    state = maybe_reset_daily_stats(state)

    limit = Keyword.get(opts, :limit, 100)
    sort_by = Keyword.get(opts, :sort_by, :novelty_score)
    filter_fn = Keyword.get(opts, :filter)

    candidates =
      :ets.tab2list(@ets_table)
      |> Enum.map(fn {_id, candidate} -> candidate end)
      |> Enum.filter(&(&1.status == :pending))
      |> maybe_apply_filter(filter_fn)
      |> sort_candidates(sort_by)
      |> Enum.take(limit)

    {:reply, candidates, state}
  end

  @impl true
  def handle_call({:get_by_status, status, opts}, _from, state) do
    state = maybe_reset_daily_stats(state)

    limit = Keyword.get(opts, :limit, 100)
    sort_by = Keyword.get(opts, :sort_by, :reviewed_at)

    candidates =
      :ets.tab2list(@ets_table)
      |> Enum.map(fn {_id, candidate} -> candidate end)
      |> Enum.filter(&(&1.status == status))
      |> sort_candidates(sort_by)
      |> Enum.take(limit)

    {:reply, candidates, state}
  end

  @impl true
  def handle_call({:get, id}, _from, state) do
    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] -> {:reply, {:ok, candidate}, state}
      [] -> {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:update_annotation, id, annotation_updates}, _from, state) do
    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = IntentReviewCandidate.update_annotation(candidate, annotation_updates)
        :ets.insert(@ets_table, {id, updated})
        persist_to_disk(state)
        broadcast_update(:annotation_updated, updated)
        {:reply, {:ok, updated}, state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:approve, id, notes, promotion_action, promoted_to_intent}, _from, state) do
    state = maybe_reset_daily_stats(state)

    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = IntentReviewCandidate.approve(candidate, notes, promotion_action, promoted_to_intent)
        :ets.insert(@ets_table, {id, updated})

        # Update stats
        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            approved: state.stats.approved + 1,
            approved_today: state.stats.approved_today + 1
        }

        new_state = %{state | stats: new_stats}

        # Persist and broadcast
        persist_to_disk(new_state)
        broadcast_update(:candidate_approved, updated)

        Logger.info("Intent candidate approved",
          id: id,
          intent: updated.predicted_intent,
          promotion_action: promotion_action
        )

        {:reply, {:ok, updated}, new_state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:reject, id, notes}, _from, state) do
    state = maybe_reset_daily_stats(state)

    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = IntentReviewCandidate.reject(candidate, notes)
        :ets.insert(@ets_table, {id, updated})

        # Update stats
        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            rejected: state.stats.rejected + 1,
            rejected_today: state.stats.rejected_today + 1
        }

        new_state = %{state | stats: new_stats}

        # Persist and broadcast
        persist_to_disk(new_state)
        broadcast_update(:candidate_rejected, updated)

        {:reply, {:ok, updated}, new_state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:defer, id, notes}, _from, state) do
    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = IntentReviewCandidate.defer(candidate, notes)
        :ets.insert(@ets_table, {id, updated})

        # Update stats
        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            deferred: state.stats.deferred + 1
        }

        new_state = %{state | stats: new_stats}

        persist_to_disk(new_state)
        broadcast_update(:candidate_deferred, updated)

        {:reply, {:ok, updated}, new_state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:bulk_approve, ids}, _from, state) do
    state = maybe_reset_daily_stats(state)

    approved_count =
      ids
      |> Enum.reduce(0, fn id, count ->
        case :ets.lookup(@ets_table, id) do
          [{^id, candidate}] when candidate.status == :pending ->
            updated = IntentReviewCandidate.approve(candidate, "Bulk approved")
            :ets.insert(@ets_table, {id, updated})
            count + 1

          _ ->
            count
        end
      end)

    new_stats = %{
      state.stats
      | pending: max(0, state.stats.pending - approved_count),
        approved: state.stats.approved + approved_count,
        approved_today: state.stats.approved_today + approved_count
    }

    new_state = %{state | stats: new_stats}
    persist_to_disk(new_state)
    broadcast_update(:bulk_approved, %{count: approved_count})

    {:reply, {:ok, approved_count}, new_state}
  end

  @impl true
  def handle_call({:bulk_reject, ids}, _from, state) do
    state = maybe_reset_daily_stats(state)

    rejected_count =
      ids
      |> Enum.reduce(0, fn id, count ->
        case :ets.lookup(@ets_table, id) do
          [{^id, candidate}] when candidate.status == :pending ->
            updated = IntentReviewCandidate.reject(candidate, "Bulk rejected")
            :ets.insert(@ets_table, {id, updated})
            count + 1

          _ ->
            count
        end
      end)

    new_stats = %{
      state.stats
      | pending: max(0, state.stats.pending - rejected_count),
        rejected: state.stats.rejected + rejected_count,
        rejected_today: state.stats.rejected_today + rejected_count
    }

    new_state = %{state | stats: new_stats}
    persist_to_disk(new_state)
    broadcast_update(:bulk_rejected, %{count: rejected_count})

    {:reply, {:ok, rejected_count}, new_state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    state = maybe_reset_daily_stats(state)
    {:reply, state.stats, state}
  end

  @impl true
  def handle_call(:clear, _from, state) do
    :ets.delete_all_objects(@ets_table)

    new_state = %{
      state
      | stats: %{
          pending: 0,
          approved: 0,
          rejected: 0,
          deferred: 0,
          approved_today: 0,
          rejected_today: 0,
          last_reset: Date.utc_today()
        }
    }

    persist_to_disk(new_state)
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    result = persist_to_disk(state)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # ============================================================================
  # Private Functions
  # ============================================================================

  defp maybe_reset_daily_stats(state) do
    today = Date.utc_today()

    if state.stats.last_reset != today do
      new_stats = %{
        state.stats
        | approved_today: 0,
          rejected_today: 0,
          last_reset: today
      }

      %{state | stats: new_stats}
    else
      state
    end
  end

  defp maybe_apply_filter(candidates, nil), do: candidates

  defp maybe_apply_filter(candidates, filter_fn) when is_function(filter_fn, 1) do
    Enum.filter(candidates, filter_fn)
  end

  defp sort_candidates(candidates, :novelty_score) do
    # Novelty score = inverse of best_score (lower confidence = more novel)
    # Also consider margin (small margin = ambiguous = more novel)
    Enum.sort_by(candidates, fn c ->
      novelty = (1.0 - c.best_score) * 0.7 + (1.0 - c.margin) * 0.3
      -novelty
    end)
  end

  defp sort_candidates(candidates, :created_at) do
    Enum.sort_by(candidates, & &1.timestamp, {:desc, DateTime})
  end

  defp sort_candidates(candidates, :reviewed_at) do
    Enum.sort_by(
      candidates,
      fn c -> c.reviewed_at || ~U[1970-01-01 00:00:00Z] end,
      {:desc, DateTime}
    )
  end

  defp sort_candidates(candidates, :margin) do
    Enum.sort_by(candidates, & &1.margin)
  end

  defp sort_candidates(candidates, _), do: candidates

  defp load_from_disk(state) do
    path = persistence_path()

    if File.exists?(path) do
      case File.read(path) do
        {:ok, binary} ->
          try do
            data = :erlang.binary_to_term(binary)
            candidates = Map.get(data, :candidates, [])
            stats = Map.get(data, :stats, state.stats)

            # Restore to ETS
            Enum.each(candidates, fn {id, candidate} ->
              :ets.insert(@ets_table, {id, candidate})
            end)

            Logger.info("Loaded intent review queue from disk", candidates: length(candidates))

            %{state | stats: stats}
          rescue
            e ->
              Logger.warning("Failed to parse intent review queue", error: inspect(e))
              state
          end

        {:error, reason} ->
          Logger.warning("Failed to read intent review queue", reason: inspect(reason))
          state
      end
    else
      state
    end
  end

  defp persist_to_disk(state) do
    path = persistence_path()

    candidates = :ets.tab2list(@ets_table)

    data = %{
      candidates: candidates,
      stats: state.stats,
      version: 1
    }

    # Ensure directory exists
    path |> Path.dirname() |> File.mkdir_p!()

    case File.write(path, :erlang.term_to_binary(data)) do
      :ok -> :ok
      {:error, reason} ->
        Logger.error("Failed to persist intent review queue", reason: inspect(reason))
        {:error, reason}
    end
  end

  defp broadcast_update(event, data) do
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.broadcast(Brain.PubSub, "intent:review", {event, data})
    end
  rescue
    _ -> :ok
  end
end
