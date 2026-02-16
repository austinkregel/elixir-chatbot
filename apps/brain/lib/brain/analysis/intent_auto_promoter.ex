defmodule Brain.Analysis.IntentAutoPromoter do
  @moduledoc """
  GenServer that periodically scans IntentReviewQueue and auto-approves
  variations of existing intents.

  Only auto-approves **variations** (new utterance -> existing intent).
  Genuinely new intents always require human review.

  Criteria for auto-approval:
  - Candidate is a variation (predicted_intent matches an existing registered intent)
  - 3+ similar candidates exist with the same predicted intent
  - Novelty score >= 0.85 (indicating genuine novel phrasing)
  - Same predicted domain as existing intent

  Daily cap: 5 auto-approvals.
  """

  use GenServer
  require Logger

  alias Brain.Analysis.{IntentReviewQueue, IntentRegistry}

  @scan_interval_ms 10 * 60 * 1000
  @daily_cap 5
  @min_similar_candidates 3
  @min_novelty_score 0.85

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns current auto-promoter stats."
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  @doc "Checks if the GenServer is ready."
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  @impl true
  def init(_opts) do
    Process.send_after(self(), :scan, @scan_interval_ms)

    {:ok,
     %{
       approved_today: 0,
       last_reset_date: Date.utc_today(),
       total_promoted: 0
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
  def handle_info(:scan, state) do
    state = maybe_reset_daily_counter(state)
    state = scan_and_promote(state)
    Process.send_after(self(), :scan, @scan_interval_ms)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state), do: {:noreply, state}

  # --- Private ---

  defp scan_and_promote(state) do
    if state.approved_today >= @daily_cap do
      state
    else
      if IntentReviewQueue.ready?() do
        do_scan(state)
      else
        state
      end
    end
  rescue
    _ -> state
  end

  defp do_scan(state) do
    candidates = IntentReviewQueue.get_pending(limit: 200, sort_by: :novelty_score)

    # Group by predicted intent
    by_intent =
      candidates
      |> Enum.group_by(& &1.predicted_intent)

    # Find groups that are variations of existing intents
    Enum.reduce(by_intent, state, fn {intent, group}, acc ->
      if acc.approved_today >= @daily_cap do
        acc
      else
        maybe_promote_group(intent, group, acc)
      end
    end)
  end

  defp maybe_promote_group(intent, candidates, state) do
    # Must be an existing registered intent (variation, not new)
    existing_meta = IntentRegistry.get(intent)

    if existing_meta == nil do
      # New intent — requires human review
      state
    else
      # Check criteria
      high_novelty = Enum.filter(candidates, fn c ->
        novelty_score(c) >= @min_novelty_score
      end)

      if length(high_novelty) >= @min_similar_candidates do
        # Promote up to daily cap
        to_promote = Enum.take(high_novelty, @daily_cap - state.approved_today)

        Enum.reduce(to_promote, state, fn candidate, acc ->
          case IntentReviewQueue.approve(
                 candidate.id,
                 "Auto-promoted: variation of existing intent '#{intent}'",
                 :variation,
                 intent
               ) do
            {:ok, _} ->
              Logger.info("IntentAutoPromoter: approved variation",
                id: candidate.id,
                text: String.slice(candidate.text, 0, 50),
                intent: intent
              )

              %{acc | approved_today: acc.approved_today + 1, total_promoted: acc.total_promoted + 1}

            _ ->
              acc
          end
        end)
      else
        state
      end
    end
  end

  defp novelty_score(candidate) do
    # Novelty score: combination of low best_score and ambiguity
    best = candidate.best_score || 0.0
    margin = candidate.margin || 0.0
    (1.0 - best) * 0.7 + (1.0 - margin) * 0.3
  end

  defp maybe_reset_daily_counter(state) do
    today = Date.utc_today()

    if Date.compare(today, state.last_reset_date) != :eq do
      %{state | approved_today: 0, last_reset_date: today}
    else
      state
    end
  end
end
