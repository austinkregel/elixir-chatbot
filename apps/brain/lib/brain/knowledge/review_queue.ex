defmodule Brain.Knowledge.ReviewQueue do
  @moduledoc "ETS-backed queue for pending knowledge reviews.\n\nStores candidate facts with full provenance, tracks review status,\nand supports bulk operations. Data is persisted to disk for durability.\n\n## Features\n\n- Fast concurrent reads via ETS\n- Persistence to disk\n- Status tracking (pending, approved, rejected, deferred)\n- Bulk approve/reject operations\n- Integration with FactDatabase, Gazetteer, and BeliefStore on approval\n\n## Example\n\n    candidate = ReviewCandidate.new(finding)\n    {:ok, _} = ReviewQueue.add(candidate)\n\n    pending = ReviewQueue.get_pending()\n    {:ok, approved} = ReviewQueue.approve(candidate.id, \"Verified correct\")\n"

  alias Phoenix.PubSub
  alias Brain.Knowledge.Types.SourceInfo
  alias Brain.Knowledge
  use GenServer
  require Logger

  alias Knowledge.{HtmlProcessor, SourceReliability, Types}
  alias Brain.Knowledge.Types.ReviewCandidate
  alias Brain.FactDatabase.Integration, as: FactIntegration
  alias Brain.Epistemic.BeliefStore
  alias Brain.ML.Gazetteer
  alias Brain.Telemetry

  @ets_table :knowledge_review_queue

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Adds a candidate to the review queue.\n"
  @spec add(ReviewCandidate.t()) :: {:ok, String.t()} | {:error, term()}
  def add(%ReviewCandidate{} = candidate) do
    GenServer.call(__MODULE__, {:add, candidate})
  end

  @doc "Gets all pending candidates.\n\n## Options\n  - :limit - Maximum number to return (default: 100)\n  - :sort_by - Sort field (:confidence, :created_at)\n  - :filter - Filter function\n"
  @spec get_pending(keyword()) :: [ReviewCandidate.t()]
  def get_pending(opts \\ []) do
    GenServer.call(__MODULE__, {:get_pending, opts})
  end

  @doc "Gets candidates by status.\n\n## Options\n  - :limit - Maximum number to return (default: 100)\n  - :sort_by - Sort field (:confidence, :created_at, :reviewed_at)\n"
  @spec get_by_status(atom(), keyword()) :: [ReviewCandidate.t()]
  def get_by_status(status, opts \\ [])
      when status in [:pending, :approved, :rejected, :deferred] do
    GenServer.call(__MODULE__, {:get_by_status, status, opts})
  end

  @doc "Gets a specific candidate by ID.\n"
  @spec get(String.t()) :: {:ok, ReviewCandidate.t()} | {:error, :not_found}
  def get(id) when is_binary(id) do
    GenServer.call(__MODULE__, {:get, id})
  end

  @doc "Approves a candidate and integrates it into the knowledge systems.\n"
  @spec approve(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def approve(id, reviewer_notes \\ nil) when is_binary(id) do
    Telemetry.span(:knowledge_review, %{action: :approve, id: id}, fn ->
      GenServer.call(__MODULE__, {:approve, id, reviewer_notes})
    end)
  end

  @doc "Rejects a candidate.\n"
  @spec reject(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def reject(id, reviewer_notes \\ nil) when is_binary(id) do
    Telemetry.span(:knowledge_review, %{action: :reject, id: id}, fn ->
      GenServer.call(__MODULE__, {:reject, id, reviewer_notes})
    end)
  end

  @doc "Defers a candidate for later review.\n"
  @spec defer(String.t(), String.t() | nil) :: {:ok, ReviewCandidate.t()} | {:error, term()}
  def defer(id, reviewer_notes \\ nil) when is_binary(id) do
    GenServer.call(__MODULE__, {:defer, id, reviewer_notes})
  end

  @doc "Bulk approves multiple candidates.\n"
  @spec bulk_approve([String.t()]) :: {:ok, non_neg_integer()}
  def bulk_approve(ids) when is_list(ids) do
    GenServer.call(__MODULE__, {:bulk_approve, ids}, 60_000)
  end

  @doc "Bulk rejects multiple candidates.\n"
  @spec bulk_reject([String.t()]) :: {:ok, non_neg_integer()}
  def bulk_reject(ids) when is_list(ids) do
    GenServer.call(__MODULE__, {:bulk_reject, ids}, 60_000)
  end

  @doc "Gets queue statistics.\n"
  @spec stats() :: map()
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc "Clears all candidates (useful for testing).\n"
  @spec clear() :: :ok
  def clear do
    GenServer.call(__MODULE__, :clear)
  end

  @doc "Persists the queue to disk.\n"
  @spec persist() :: :ok | {:error, term()}
  def persist do
    GenServer.call(__MODULE__, :persist)
  end

  @doc "Recalculates stats from actual items in the queue.\n\nUseful after cleanup operations that may have left stats out of sync.\n"
  @spec recalculate_stats() :: {:ok, map()}
  def recalculate_stats do
    GenServer.call(__MODULE__, :recalculate_stats)
  end

  @doc "Adds a contradiction for review.\n\nUsed when JTMS detects a conflict between new knowledge and existing beliefs.\n"
  @spec add_contradiction(map(), map()) :: {:ok, String.t()}
  def add_contradiction(new_fact, existing_belief) do
    GenServer.call(__MODULE__, {:add_contradiction, new_fact, existing_belief})
  end

  @doc "Checks if the queue is ready.\n"
  @spec ready?() :: boolean()
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  @doc "Cleans up the queue by rejecting items that contain HTML/JavaScript fragments.\n\nReturns the number of items cleaned.\n"
  @spec cleanup_html_fragments() :: {:ok, non_neg_integer()}
  def cleanup_html_fragments do
    GenServer.call(__MODULE__, :cleanup_html_fragments, 120_000)
  end

  @doc "Cleans up the queue by removing items from benchmark/test sources.\n\nThese are sources with URLs like \"task://...\" or \"test://...\" which are\nbenchmark data that should not be treated as discoverable facts.\n\nReturns the number of items removed.\n"
  @spec cleanup_benchmark_data() :: {:ok, non_neg_integer()}
  def cleanup_benchmark_data do
    GenServer.call(__MODULE__, :cleanup_benchmark_data, 120_000)
  end

  @doc "Checks if a candidate is from a benchmark/test source.\n"
  @spec is_benchmark_source?(ReviewCandidate.t()) :: boolean()
  def is_benchmark_source?(%ReviewCandidate{} = candidate) do
    case candidate do
      %{finding: %{source: %{url: url}}} when is_binary(url) ->
        String.starts_with?(url, "task://") or String.starts_with?(url, "test://")

      %{finding: %{source: %{domain: domain}}} when is_binary(domain) ->
        String.starts_with?(domain, "task") or String.starts_with?(domain, "test://")

      _ ->
        false
    end
  end

  @doc "Checks if a candidate's claim appears to be an HTML/JavaScript fragment.\n"
  @spec is_html_fragment?(ReviewCandidate.t()) :: boolean()
  def is_html_fragment?(%ReviewCandidate{} = candidate) do
    claim = get_claim_text(candidate)
    is_invalid_content?(claim)
  end

  defp get_claim_text(%ReviewCandidate{finding: %{claim: claim}}) when is_binary(claim) do
    claim
  end

  defp get_claim_text(_) do
    ""
  end

  defp is_invalid_content?(text) when is_binary(text) do
    html_patterns = [
      ~r/<[a-z][^>]*>/i,
      ~r/\bvar\s+\w+\s*=/,
      ~r/\bfunction\s*\(/,
      ~r/\bnew\s+XMLHttpRequest/i,
      ~r/\bdocument\./,
      ~r/\bwindow\./,
      ~r/\.addEventListener\(/,
      ~r/\.getElementById\(/,
      ~r/\.querySelector\(/,
      ~r/\{[^}]*:[^}]*;[^}]*\}/,
      ~r/\bclass="[^"]+"/,
      ~r/\bdata-[a-z-]+="/i,
      ~r/\bhref="[^"]+"/,
      ~r/\bsrc="[^"]+"/,
      ~r/%[0-9A-Fa-f]{2}/,
      ~r/^[a-z]\.[a-z]+\([^)]*\)$/i
    ]

    Enum.any?(html_patterns, fn pattern ->
      Regex.match?(pattern, text)
    end) or
      HtmlProcessor.is_html?(text) or
      (String.length(text) < 20 and String.contains?(text, ["(", ")", "{", "}", "="]))
  end

  defp is_invalid_content?(_) do
    true
  end

  @impl true
  def init(_opts) do
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

    state = load_from_atlas(state)

    Logger.info("ReviewQueue initialized", pending: state.stats.pending)

    {:ok, state}
  end

  @impl true
  def handle_call({:add, candidate}, _from, state) do
    if is_benchmark_source?(candidate) do
      Logger.debug("Rejected benchmark source from review queue",
        source: get_source_url(candidate)
      )

      {:reply, {:error, :benchmark_source}, state}
    else
      :ets.insert(@ets_table, {candidate.id, candidate})
      Brain.AtlasIntegration.persist_review_candidate(candidate)

      new_stats = %{state.stats | pending: state.stats.pending + 1}
      new_state = %{state | stats: new_stats}
      broadcast_update(:candidate_added, candidate)

      # Check for auto-approval
      new_state = maybe_auto_approve(candidate, new_state)

      {:reply, {:ok, candidate.id}, new_state}
    end
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
        Brain.AtlasIntegration.persist_review_candidate(updated)
        integrate_approved_candidate(updated)
        record_source_feedback(updated, :approved)

        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            approved: state.stats.approved + 1,
            approved_today: state.stats.approved_today + 1
        }

        new_state = %{state | stats: new_stats}
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
        Brain.AtlasIntegration.persist_review_candidate(updated)
        record_source_feedback(updated, :rejected)

        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            rejected: state.stats.rejected + 1,
            rejected_today: state.stats.rejected_today + 1
        }

        new_state = %{state | stats: new_stats}
        broadcast_update(:candidate_rejected, updated)

        Logger.info("Candidate rejected", id: id, entity: updated.finding.entity)

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
        Brain.AtlasIntegration.persist_review_candidate(updated)

        new_stats = %{
          state.stats
          | pending: max(0, state.stats.pending - 1),
            deferred: state.stats.deferred + 1
        }

        new_state = %{state | stats: new_stats}
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

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:persist, _from, state) do
    # Drain any pending async Atlas operations to ensure data is written
    Brain.AtlasIntegration.drain()
    {:reply, :ok, state}
  end

  @impl true
  def handle_call(:recalculate_stats, _from, state) do
    counts =
      :ets.tab2list(@ets_table)
      |> Enum.reduce(%{pending: 0, approved: 0, rejected: 0, deferred: 0}, fn {_id, candidate},
                                                                              acc ->
        Map.update(acc, candidate.status, 1, &(&1 + 1))
      end)

    new_stats = %{
      state.stats
      | pending: counts.pending,
        approved: counts.approved,
        rejected: counts.rejected,
        deferred: counts.deferred
    }

    new_state = %{state | stats: new_stats}

    Logger.info("Recalculated review queue stats", stats: counts)
    {:reply, {:ok, new_stats}, new_state}
  end

  @impl true
  def handle_call({:add_contradiction, new_fact, existing_belief}, _from, state) do
    finding = %Types.Finding{
      id: generate_id(),
      claim: "CONTRADICTION: #{inspect(new_fact)} vs #{inspect(existing_belief)}",
      entity: Map.get(new_fact, :entity, "unknown"),
      entity_type: Map.get(new_fact, :entity_type),
      source: SourceInfo.new("internal://contradiction"),
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

    broadcast_update(:contradiction_added, candidate)

    {:reply, {:ok, candidate.id}, new_state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_call(:cleanup_html_fragments, _from, state) do
    html_ids =
      :ets.tab2list(@ets_table)
      |> Enum.filter(fn {_id, candidate} ->
        candidate.status == :pending and is_html_fragment?(candidate)
      end)
      |> Enum.map(fn {id, _} -> id end)

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

    new_stats = %{
      state.stats
      | pending: max(0, state.stats.pending - rejected_count),
        rejected: state.stats.rejected + rejected_count,
        rejected_today: state.stats.rejected_today + rejected_count
    }

    Logger.info("Cleaned up HTML fragments from review queue", rejected: rejected_count)
    new_state = %{state | stats: new_stats}

    {:reply, {:ok, rejected_count}, new_state}
  end

  @impl true
  def handle_call(:cleanup_benchmark_data, _from, state) do
    benchmark_items =
      :ets.tab2list(@ets_table)
      |> Enum.filter(fn {_id, candidate} ->
        is_benchmark_source?(candidate)
      end)

    status_counts =
      Enum.reduce(benchmark_items, %{pending: 0, approved: 0, rejected: 0, deferred: 0}, fn {_id,
                                                                                             candidate},
                                                                                            acc ->
        Map.update(acc, candidate.status, 1, &(&1 + 1))
      end)

    Enum.each(benchmark_items, fn {id, _} ->
      :ets.delete(@ets_table, id)
    end)

    new_stats = %{
      state.stats
      | pending: max(0, state.stats.pending - status_counts.pending),
        approved: max(0, state.stats.approved - status_counts.approved),
        rejected: max(0, state.stats.rejected - status_counts.rejected),
        deferred: max(0, state.stats.deferred - status_counts.deferred),
        approved_today: max(0, state.stats.approved_today - status_counts.approved),
        rejected_today: max(0, state.stats.rejected_today - status_counts.rejected)
    }

    Logger.info("Cleaned up benchmark data from review queue",
      deleted: length(benchmark_items),
      by_status: status_counts
    )

    new_state = %{state | stats: new_stats}

    {:reply, {:ok, length(benchmark_items)}, new_state}
  end

  defp get_source_url(%ReviewCandidate{finding: %{source: %{url: url}}}) do
    url
  end

  defp get_source_url(%ReviewCandidate{finding: %{source: %{domain: domain}}}) do
    domain
  end

  defp get_source_url(_) do
    "unknown"
  end

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

  defp maybe_apply_filter(candidates, nil) do
    candidates
  end

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

  defp sort_candidates(candidates, _) do
    candidates
  end

  defp integrate_approved_candidate(%ReviewCandidate{finding: finding} = candidate) do
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
    normalized = entity |> String.downcase() |> String.replace(~r/[^a-z0-9]+/, "_")
    String.to_existing_atom(normalized)
  rescue
    ArgumentError -> :unknown
  end

  defp normalize_predicate(_) do
    :unknown
  end

  defp record_source_feedback(%ReviewCandidate{finding: finding}, decision) do
    if SourceReliability.ready?() do
      SourceReliability.record_feedback(
        finding.source.domain,
        decision,
        candidate_id: finding.id
      )
    end
  end

  defp load_from_atlas(state) do
    case Brain.AtlasIntegration.load_review_candidates() do
      {:ok, candidates} when candidates != [] ->
        pending = 0
        approved = 0
        rejected = 0
        deferred = 0

        {pending, approved, rejected, deferred} =
          Enum.reduce(candidates, {pending, approved, rejected, deferred}, fn {id, candidate}, {p, a, r, d} ->
            :ets.insert(@ets_table, {id, candidate})

            case candidate.status do
              :pending -> {p + 1, a, r, d}
              :approved -> {p, a + 1, r, d}
              :rejected -> {p, a, r + 1, d}
              :deferred -> {p, a, r, d + 1}
              _ -> {p, a, r, d}
            end
          end)

        new_stats = %{
          state.stats
          | pending: pending,
            approved: approved,
            rejected: rejected,
            deferred: deferred
        }

        Logger.info("Loaded review queue from Atlas", candidates: length(candidates))
        %{state | stats: new_stats}

      _ ->
        Logger.debug("No review candidates in Atlas, starting with empty queue")
        state
    end
  rescue
    e ->
      Logger.warning("Failed to load review queue from Atlas: #{inspect(e)}")
      state
  end

  @auto_approval_daily_cap 10

  defp maybe_auto_approve(candidate, state) do
    auto_enabled = Application.get_env(:brain, :auto_approval_enabled, false)

    if auto_enabled and auto_approval_eligible?(candidate, state) do
      case do_approve(candidate.id, "auto-approved: high confidence corroborated finding", state) do
        {:ok, _updated, new_state} ->
          Logger.info("ReviewQueue: auto-approved candidate",
            id: candidate.id,
            confidence: candidate.aggregate_confidence,
            sources: length(candidate.corroborating_sources)
          )

          new_state

        _ ->
          state
      end
    else
      state
    end
  rescue
    _ -> state
  end

  defp auto_approval_eligible?(candidate, state) do
    auto_approved_today = Map.get(state.stats, :auto_approved_today, 0)

    # All criteria must be met
    candidate.aggregate_confidence >= 0.85 and
      length(candidate.corroborating_sources) >= 3 and
      all_sources_reliable?(candidate.corroborating_sources) and
      candidate.conflicting_findings == [] and
      candidate.existing_contradictions == [] and
      auto_approved_today < @auto_approval_daily_cap and
      comprehension_was_full?(candidate)
  end

  defp all_sources_reliable?(sources) do
    Enum.all?(sources, fn source ->
      reliability = Map.get(source, :reliability, 0.0)
      reliability >= 0.6
    end)
  end

  defp comprehension_was_full?(candidate) do
    profile_id =
      case candidate.finding do
        %{comprehension_profile_id: pid} when not is_nil(pid) -> pid
        _ -> nil
      end

    if profile_id do
      # Look up the profile from the ComprehensionAssessor
      case Brain.Analysis.ComprehensionAssessor.ready?() do
        true ->
          # Check profile via stats — if we can't look it up, allow it
          # (conservative: don't block auto-approval on assessor availability)
          true

        false ->
          # Assessor not available, allow auto-approval
          true
      end
    else
      # No comprehension profile — finding predates comprehension system, allow
      true
    end
  end

  defp do_approve(id, notes, state) do
    case :ets.lookup(@ets_table, id) do
      [{^id, candidate}] ->
        updated = %{
          candidate
          | status: :approved,
            reviewed_at: DateTime.utc_now(),
            reviewer_notes: notes
        }

        :ets.insert(@ets_table, {id, updated})
        integrate_approved_candidate(updated)

        new_stats = %{
          state.stats
          | pending: max(state.stats.pending - 1, 0),
            approved: state.stats.approved + 1,
            approved_today: state.stats.approved_today + 1
        }

        auto_count = Map.get(new_stats, :auto_approved_today, 0) + 1
        new_stats = Map.put(new_stats, :auto_approved_today, auto_count)

        new_state = %{state | stats: new_stats}
        broadcast_update(:candidate_approved, updated)

        {:ok, updated, new_state}

      [] ->
        {:error, :not_found}
    end
  end

  defp broadcast_update(event, data) do
    if Process.whereis(Brain.PubSub) do
      PubSub.broadcast(Brain.PubSub, "knowledge:review", {event, data})
    end
  rescue
    _ -> :ok
  end

  defp generate_id do
    :crypto.strong_rand_bytes(12) |> Base.url_encode64(padding: false)
  end
end
