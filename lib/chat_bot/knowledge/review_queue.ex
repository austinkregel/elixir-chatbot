defmodule ChatBot.Knowledge.ReviewQueue do
  @moduledoc """
  ETS-backed queue for pending knowledge reviews.

  Stores candidate facts with full provenance, tracks review status,
  and supports bulk operations. Data is persisted to disk for durability.

  ## Features

  - Fast concurrent reads via ETS
  - Persistence to disk
  - Status tracking (pending, approved, rejected, deferred)
  - Bulk approve/reject operations
  - Integration with FactDatabase, Gazetteer, and BeliefStore on approval

  ## Example

      candidate = ReviewCandidate.new(finding)
      {:ok, _} = ReviewQueue.add(candidate)
      
      pending = ReviewQueue.get_pending()
      {:ok, approved} = ReviewQueue.approve(candidate.id, "Verified correct")
  """

  use GenServer
  require Logger

  alias ChatBot.Knowledge.{HtmlProcessor, SourceReliability, Types}
  alias ChatBot.Knowledge.Types.ReviewCandidate
  alias ChatBot.FactDatabase.Integration, as: FactIntegration
  alias ChatBot.Epistemic.BeliefStore
  alias ChatBot.ML.Gazetteer
  alias ChatBot.Telemetry

  @ets_table :knowledge_review_queue
  @persistence_path "priv/data/review_queue.term"

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Adds a candidate to the review queue.
  """
  @spec add(ReviewCandidate.t()) :: {:ok, String.t()} | {:error, term()}
  def add(%ReviewCandidate{} = candidate) do
    GenServer.call(__MODULE__, {:add, candidate})
  end

  @doc """
  Gets all pending candidates.

  ## Options
    - :limit - Maximum number to return (default: 100)
    - :sort_by - Sort field (:confidence, :created_at)
    - :filter - Filter function
  """
  @spec get_pending(keyword()) :: [ReviewCandidate.t()]
  def get_pending(opts \\ []) do
    GenServer.call(__MODULE__, {:get_pending, opts})
  end

  @doc """
  Gets candidates by status.

  ## Options
    - :limit - Maximum number to return (default: 100)
    - :sort_by - Sort field (:confidence, :created_at, :reviewed_at)
  """
  @spec get_by_status(atom(), keyword()) :: [ReviewCandidate.t()]
  def get_by_status(status, opts \\ []) when status in [:pending, :approved, :rejected, :deferred] do
    GenServer.call(__MODULE__, {:get_by_status, status, opts})
  end

  @doc """
  Gets a specific candidate by ID.
  """
  @spec get(String.t()) :: {:ok, ReviewCandidate.t()} | {:error, :not_found}
  def get(id) when is_binary(id) do
    GenServer.call(__MODULE__, {:get, id})
  end

  @doc """
  Approves a candidate and integrates it into the knowledge systems.
  """
  @spec approve(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def approve(id, reviewer_notes \\ nil) when is_binary(id) do
    Telemetry.span(:knowledge_review, %{action: :approve, id: id}, fn ->
      GenServer.call(__MODULE__, {:approve, id, reviewer_notes})
    end)
  end

  @doc """
  Rejects a candidate.
  """
  @spec reject(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def reject(id, reviewer_notes \\ nil) when is_binary(id) do
    Telemetry.span(:knowledge_review, %{action: :reject, id: id}, fn ->
      GenServer.call(__MODULE__, {:reject, id, reviewer_notes})
    end)
  end

  @doc """
  Defers a candidate for later review.
  """
  @spec defer(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def defer(id, reviewer_notes \\ nil) when is_binary(id) do
    GenServer.call(__MODULE__, {:defer, id, reviewer_notes})
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
  Adds a contradiction for review.

  Used when JTMS detects a conflict between new knowledge and existing beliefs.
  """
  @spec add_contradiction(map(), map()) :: {:ok, String.t()}
  def add_contradiction(new_fact, existing_belief) do
    GenServer.call(__MODULE__, {:add_contradiction, new_fact, existing_belief})
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

  @doc """
  Cleans up the queue by rejecting items that contain HTML/JavaScript fragments.

  Returns the number of items cleaned.
  """
  @spec cleanup_html_fragments() :: {:ok, non_neg_integer()}
  def cleanup_html_fragments do
    GenServer.call(__MODULE__, :cleanup_html_fragments, 120_000)
  end

  @doc """
  Checks if a candidate's claim appears to be an HTML/JavaScript fragment.
  """
  @spec is_html_fragment?(ReviewCandidate.t()) :: boolean()
  def is_html_fragment?(%ReviewCandidate{} = candidate) do
    claim = get_claim_text(candidate)
    is_invalid_content?(claim)
  end

  defp get_claim_text(%ReviewCandidate{finding: %{claim: claim}}) when is_binary(claim), do: claim
  defp get_claim_text(_), do: ""

  defp is_invalid_content?(text) when is_binary(text) do
    # Check for HTML/JavaScript patterns that indicate garbage content
    html_patterns = [
      # HTML tags
      ~r/<[a-z][^>]*>/i,
      # JavaScript code patterns
      ~r/\bvar\s+\w+\s*=/,
      ~r/\bfunction\s*\(/,
      ~r/\bnew\s+XMLHttpRequest/i,
      ~r/\bdocument\./,
      ~r/\bwindow\./,
      ~r/\.addEventListener\(/,
      ~r/\.getElementById\(/,
      ~r/\.querySelector\(/,
      # CSS patterns
      ~r/\{[^}]*:[^}]*;[^}]*\}/,
      # Common HTML attributes in text
      ~r/\bclass="[^"]+"/,
      ~r/\bdata-[a-z-]+="/i,
      ~r/\bhref="[^"]+"/,
      ~r/\bsrc="[^"]+"/,
      # Encoded content
      ~r/%[0-9A-Fa-f]{2}/,
      # Very short meaningless fragments
      ~r/^[a-z]\.[a-z]+\([^)]*\)$/i
    ]

    # Check if the text matches any problematic pattern
    Enum.any?(html_patterns, fn pattern ->
      Regex.match?(pattern, text)
    end) or
      # Also check using the HtmlProcessor
      HtmlProcessor.is_html?(text) or
      # Very short content that's likely garbage
      (String.length(text) < 20 and String.contains?(text, ["(", ")", "{", "}", "="]))
  end

  defp is_invalid_content?(_), do: true

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

    Logger.info("ReviewQueue initialized", pending: state.stats.pending)

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
    sort_by = Keyword.get(opts, :sort_by, :confidence)
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
  def handle_call({:approve, id, notes}, _from, state) do
    state = maybe_reset_daily_stats(state)

    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = ReviewCandidate.approve(candidate, notes)
        :ets.insert(@ets_table, {id, updated})

        # Integrate into knowledge systems
        integrate_approved_candidate(updated)

        # Update source reliability (positive feedback)
        record_source_feedback(updated, :approved)

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

        Logger.info("Candidate approved",
          id: id,
          entity: updated.finding.entity,
          claim: String.slice(updated.finding.claim, 0, 50)
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
        updated = ReviewCandidate.reject(candidate, notes)
        :ets.insert(@ets_table, {id, updated})

        # Update source reliability (negative feedback)
        record_source_feedback(updated, :rejected)

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

        Logger.info("Candidate rejected",
          id: id,
          entity: updated.finding.entity
        )

        {:reply, {:ok, updated}, new_state}

      [] ->
        {:reply, {:error, :not_found}, state}
    end
  end

  @impl true
  def handle_call({:defer, id, notes}, _from, state) do
    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = ReviewCandidate.defer(candidate, notes)
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
            updated = ReviewCandidate.approve(candidate, "Bulk approved")
            :ets.insert(@ets_table, {id, updated})
            integrate_approved_candidate(updated)
            record_source_feedback(updated, :approved)
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
            updated = ReviewCandidate.reject(candidate, "Bulk rejected")
            :ets.insert(@ets_table, {id, updated})
            record_source_feedback(updated, :rejected)
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
  def handle_call({:add_contradiction, new_fact, existing_belief}, _from, state) do
    # Create a special candidate for contradiction review
    finding = %Types.Finding{
      id: generate_id(),
      claim: "CONTRADICTION: #{inspect(new_fact)} vs #{inspect(existing_belief)}",
      entity: Map.get(new_fact, :entity, "unknown"),
      entity_type: Map.get(new_fact, :entity_type),
      source: Types.SourceInfo.new("internal://contradiction"),
      raw_context: "New fact contradicts existing belief",
      extracted_at: DateTime.utc_now(),
      confidence: 0.5
    }

    candidate =
      ReviewCandidate.new(finding,
        existing_contradictions: [existing_belief],
        aggregate_confidence: 0.3
      )

    :ets.insert(@ets_table, {candidate.id, candidate})

    new_stats = %{state.stats | pending: state.stats.pending + 1}
    new_state = %{state | stats: new_stats}

    persist_to_disk(new_state)
    broadcast_update(:contradiction_added, candidate)

    {:reply, {:ok, candidate.id}, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:cleanup_html_fragments, _from, state) do
    # Find all pending items that are HTML fragments
    html_ids =
      :ets.tab2list(@ets_table)
      |> Enum.filter(fn {_id, candidate} ->
        candidate.status == :pending and is_html_fragment?(candidate)
      end)
      |> Enum.map(fn {id, _} -> id end)

    # Reject them in batch
    rejected_count =
      Enum.reduce(html_ids, 0, fn id, count ->
        case :ets.lookup(@ets_table, id) do
          [{^id, candidate}] ->
            updated = %{candidate | status: :rejected, reviewed_at: DateTime.utc_now()}
            :ets.insert(@ets_table, {id, updated})
            count + 1

          _ ->
            count
        end
      end)

    # Update stats
    new_stats = %{
      state.stats
      | pending: max(0, state.stats.pending - rejected_count),
        rejected: state.stats.rejected + rejected_count,
        rejected_today: state.stats.rejected_today + rejected_count
    }

    Logger.info("Cleaned up HTML fragments from review queue", rejected: rejected_count)

    # Persist changes
    new_state = %{state | stats: new_stats}
    persist_to_disk(new_state)

    {:reply, {:ok, rejected_count}, new_state}
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

  defp sort_candidates(candidates, :confidence) do
    Enum.sort_by(candidates, & &1.aggregate_confidence, :desc)
  end

  defp sort_candidates(candidates, :created_at) do
    Enum.sort_by(candidates, & &1.finding.extracted_at, {:desc, DateTime})
  end

  defp sort_candidates(candidates, :reviewed_at) do
    Enum.sort_by(
      candidates,
      fn c -> c.reviewed_at || ~U[1970-01-01 00:00:00Z] end,
      {:desc, DateTime}
    )
  end

  defp sort_candidates(candidates, _), do: candidates

  defp integrate_approved_candidate(%ReviewCandidate{finding: finding} = candidate) do
    # Add to FactDatabase
    try do
      FactIntegration.add_fact(
        finding.entity,
        finding.claim,
        category: "learned",
        entity_type: finding.entity_type,
        confidence: candidate.aggregate_confidence,
        verification_source: finding.source.url
      )
    rescue
      e -> Logger.warning("Failed to add to FactDatabase", error: Exception.message(e))
    end

    # Add to Gazetteer if entity type is appropriate
    if finding.entity_type in ["person", "location", "organization", "city", "country"] do
      try do
        Gazetteer.add_entry(
          finding.entity,
          finding.entity_type,
          %{source: :knowledge_expansion, confidence: candidate.aggregate_confidence}
        )
      rescue
        e -> Logger.warning("Failed to add to Gazetteer", error: Exception.message(e))
      catch
        :exit, _ -> Logger.warning("Gazetteer not available")
      end
    end

    # Add to BeliefStore
    try do
      if BeliefStore.ready?() do
        BeliefStore.add_belief(
          :world,
          normalize_predicate(finding.entity),
          finding.claim,
          source: :learned,
          confidence: candidate.aggregate_confidence,
          provenance: ["knowledge_expansion", finding.source.url]
        )
      end
    rescue
      e -> Logger.warning("Failed to add to BeliefStore", error: Exception.message(e))
    end

    :ok
  end

  defp normalize_predicate(entity) when is_binary(entity) do
    entity
    |> String.downcase()
    |> String.replace(~r/[^a-z0-9]+/, "_")
    |> String.to_atom()
  end

  defp normalize_predicate(_), do: :unknown

  defp record_source_feedback(%ReviewCandidate{finding: finding}, decision) do
    if SourceReliability.ready?() do
      SourceReliability.record_feedback(
        finding.source.domain,
        decision,
        candidate_id: finding.id
      )
    end
  end

  defp load_from_disk(state) do
    path = Path.join(File.cwd!(), @persistence_path)

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

            Logger.info("Loaded review queue from disk", candidates: length(candidates))

            %{state | stats: stats}
          rescue
            e ->
              Logger.warning("Failed to parse review queue", error: inspect(e))
              state
          end

        {:error, reason} ->
          Logger.warning("Failed to read review queue", reason: inspect(reason))
          state
      end
    else
      state
    end
  end

  defp persist_to_disk(state) do
    path = Path.join(File.cwd!(), @persistence_path)

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
        Logger.error("Failed to persist review queue", reason: inspect(reason))
        {:error, reason}
    end
  end

  defp broadcast_update(event, data) do
    if Process.whereis(ChatBot.PubSub) do
      Phoenix.PubSub.broadcast(ChatBot.PubSub, "knowledge:review", {event, data})
    end
  rescue
    _ -> :ok
  end

  defp generate_id do
    :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
  end
end
