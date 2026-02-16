defmodule Brain.Analysis.ComprehensionAssessor do
  @moduledoc """
  Self-interrogating comprehension gate GenServer.

  Assesses whether the system genuinely understands text by scoring 8 comprehension
  dimensions derived from existing pipeline outputs. Weights evolve via EMA based
  on approval/rejection outcomes from ReviewQueue.

  Subscribes to PubSub "knowledge:review" for outcome feedback.
  Stores profiles in ETS for later outcome matching (TTL: 7 days).
  Persists evolved weights to JSON for restart durability.
  """

  use GenServer
  require Logger

  alias Brain.Analysis.ComprehensionAssessor.{ComprehensionProfile, DimensionEvaluators}
  alias Brain.Analysis.ChunkAnalysis

  @weights_table :comprehension_weights
  @profiles_table :comprehension_profiles
  @stats_table :comprehension_stats

  # 7-day TTL for profile history
  @profile_ttl_seconds 7 * 24 * 60 * 60

  # Cleanup every hour
  @cleanup_interval_ms 60 * 60 * 1000

  # Minimum outcomes before weight evolution begins
  @cold_start_threshold 10

  # EMA alpha for weight updates
  @ema_alpha 0.05

  # Max weight snapshots to keep for rollback
  @max_weight_history 5

  @weights_path "priv/analysis/comprehension_weights.json"

  # --- Public API ---

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Assesses comprehension of a list of ChunkAnalysis results.

  Returns a ComprehensionProfile with per-dimension scores, composite score,
  verdict, gaps, and learnable flag.
  """
  def assess(analyses, name \\ __MODULE__) when is_list(analyses) do
    GenServer.call(name, {:assess, analyses})
  end

  @doc """
  Records an outcome (approval/rejection) for weight evolution.
  """
  def record_outcome(profile_id, outcome, name \\ __MODULE__)
      when outcome in [:approved, :rejected] do
    GenServer.cast(name, {:record_outcome, profile_id, outcome})
  end

  @doc """
  Returns current stats: total assessments, outcome counts, current weights.
  """
  def stats(name \\ __MODULE__) do
    GenServer.call(name, :stats)
  end

  @doc """
  Resets weights to equal initial values. Safety valve for bad drift.
  """
  def reset_weights(name \\ __MODULE__) do
    GenServer.call(name, :reset_weights)
  end

  @doc """
  Checks if the GenServer is ready.
  """
  def ready?(name \\ __MODULE__) do
    try do
      GenServer.call(name, :ready?, 100)
    catch
      :exit, _ -> false
    end
  end

  # --- GenServer Callbacks ---

  @impl true
  def init(opts) do
    # Create ETS tables
    :ets.new(@weights_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@profiles_table, [:named_table, :set, :public, read_concurrency: true])
    :ets.new(@stats_table, [:named_table, :set, :public, read_concurrency: true])

    # Load persisted weights or use defaults
    weights = load_weights(opts)
    store_weights(weights)

    # Initialize stats
    :ets.insert(@stats_table, {:total_assessments, 0})
    :ets.insert(@stats_table, {:approved_count, 0})
    :ets.insert(@stats_table, {:rejected_count, 0})

    # Subscribe to knowledge review PubSub for outcome feedback
    if Process.whereis(Brain.PubSub) do
      Phoenix.PubSub.subscribe(Brain.PubSub, "knowledge:review")
    end

    # Schedule periodic profile cleanup
    Process.send_after(self(), :cleanup_profiles, @cleanup_interval_ms)

    Logger.info("ComprehensionAssessor started with #{map_size(weights)} dimension weights")

    {:ok,
     %{
       weights: weights,
       weight_history: [],
       outcome_count: 0
     }}
  end

  @impl true
  def handle_call({:assess, analyses}, _from, state) do
    profile = do_assess(analyses, state.weights)

    # Store profile for later outcome matching
    :ets.insert(@profiles_table, {profile.id, profile, System.system_time(:second)})

    # Update stats
    increment_stat(:total_assessments)

    {:reply, profile, state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      total_assessments: get_stat(:total_assessments),
      approved_count: get_stat(:approved_count),
      rejected_count: get_stat(:rejected_count),
      outcome_count: state.outcome_count,
      current_weights: state.weights,
      weight_history_count: length(state.weight_history),
      cold_start: state.outcome_count < @cold_start_threshold
    }

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:reset_weights, _from, state) do
    weights = default_weights()
    store_weights(weights)
    persist_weights(weights)

    Logger.info("ComprehensionAssessor weights reset to defaults")

    {:reply, :ok, %{state | weights: weights, weight_history: []}}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  @impl true
  def handle_cast({:record_outcome, profile_id, outcome}, state) do
    case :ets.lookup(@profiles_table, profile_id) do
      [{^profile_id, profile, _ts}] ->
        new_state = evolve_weights(state, profile, outcome)
        increment_stat(:"#{outcome}_count")
        {:noreply, new_state}

      [] ->
        Logger.debug("ComprehensionAssessor: no profile found for #{profile_id}")
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:candidate_approved, candidate}, state) do
    profile_id = extract_profile_id(candidate)

    if profile_id do
      new_state = handle_outcome(state, profile_id, :approved)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info({:candidate_rejected, candidate}, state) do
    profile_id = extract_profile_id(candidate)

    if profile_id do
      new_state = handle_outcome(state, profile_id, :rejected)
      {:noreply, new_state}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info(:cleanup_profiles, state) do
    cleanup_expired_profiles()
    Process.send_after(self(), :cleanup_profiles, @cleanup_interval_ms)
    {:noreply, state}
  end

  @impl true
  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # --- Private Functions ---

  defp do_assess(analyses, weights) when is_list(analyses) do
    # Evaluate all dimensions across all chunks, taking the aggregate
    all_dimensions =
      analyses
      |> Enum.map(fn
        %ChunkAnalysis{} = analysis ->
          DimensionEvaluators.evaluate_all(analysis)

        _ ->
          %{}
      end)
      |> Enum.reject(&(&1 == %{}))

    if all_dimensions == [] do
      # No valid analyses — return garbled
      ComprehensionProfile.build(
        Map.new(ComprehensionProfile.dimension_names(), fn d -> {d, {0.0, %{}}} end),
        weights
      )
    else
      # Average dimension scores across chunks
      averaged =
        ComprehensionProfile.dimension_names()
        |> Map.new(fn dim ->
          scores_and_evidence =
            all_dimensions
            |> Enum.map(&Map.get(&1, dim, {0.0, %{}}))

          avg_score =
            scores_and_evidence
            |> Enum.map(fn {s, _} -> s end)
            |> then(fn scores ->
              if scores == [], do: 0.0, else: Enum.sum(scores) / length(scores)
            end)

          # Take evidence from the highest-scoring chunk
          best_evidence =
            scores_and_evidence
            |> Enum.max_by(fn {s, _} -> s end, fn -> {0.0, %{}} end)
            |> elem(1)

          {dim, {avg_score, best_evidence}}
        end)

      ComprehensionProfile.build(averaged, weights)
    end
  end

  defp handle_outcome(state, profile_id, outcome) do
    case :ets.lookup(@profiles_table, profile_id) do
      [{^profile_id, profile, _ts}] ->
        increment_stat(:"#{outcome}_count")
        evolve_weights(state, profile, outcome)

      [] ->
        state
    end
  end

  defp evolve_weights(state, _profile, _outcome)
       when state.outcome_count < @cold_start_threshold - 1 do
    %{state | outcome_count: state.outcome_count + 1}
  end

  defp evolve_weights(state, profile, outcome) do
    new_count = state.outcome_count + 1
    old_weights = state.weights

    new_weights =
      case outcome do
        :approved ->
          # Boost weights for dimensions that scored highly
          boost_high_scorers(old_weights, profile.dimensions)

        :rejected ->
          # Reduce weights for dimensions that scored high but led to rejection
          penalize_false_positives(old_weights, profile.dimensions)
      end

    # Normalize weights to sum to 1.0
    new_weights = normalize_weights(new_weights)

    # Store and persist
    store_weights(new_weights)
    persist_weights(new_weights)

    # Save to history
    history = [old_weights | state.weight_history] |> Enum.take(@max_weight_history)

    %{state | weights: new_weights, outcome_count: new_count, weight_history: history}
  end

  defp boost_high_scorers(weights, dimensions) do
    Map.new(weights, fn {dim, weight} ->
      case Map.get(dimensions, dim) do
        %{score: score} when score >= 0.6 ->
          # EMA boost: move weight toward (weight + bonus)
          bonus = score * @ema_alpha
          {dim, weight + bonus}

        _ ->
          {dim, weight}
      end
    end)
  end

  defp penalize_false_positives(weights, dimensions) do
    Map.new(weights, fn {dim, weight} ->
      case Map.get(dimensions, dim) do
        %{score: score} when score >= 0.6 ->
          # High score but outcome was rejection: reduce trust in this dimension
          penalty = score * @ema_alpha
          {dim, max(weight - penalty, 0.01)}

        %{score: score} when score < 0.4 ->
          # Low score that still passed: this dimension should have been more important
          boost = (1.0 - score) * @ema_alpha * 0.5
          {dim, weight + boost}

        _ ->
          {dim, weight}
      end
    end)
  end

  defp normalize_weights(weights) do
    total = weights |> Map.values() |> Enum.sum()

    if total > 0.0 do
      Map.new(weights, fn {k, v} -> {k, v / total} end)
    else
      default_weights()
    end
  end

  defp default_weights do
    dims = ComprehensionProfile.dimension_names()
    weight = 1.0 / length(dims)
    Map.new(dims, fn d -> {d, weight} end)
  end

  defp store_weights(weights) do
    Enum.each(weights, fn {dim, weight} ->
      :ets.insert(@weights_table, {dim, weight})
    end)
  end

  defp load_weights(opts) do
    path = Keyword.get(opts, :weights_path, weights_file_path())

    case File.read(path) do
      {:ok, json} ->
        case Jason.decode(json) do
          {:ok, data} when is_map(data) ->
            Map.new(data, fn {k, v} -> {String.to_existing_atom(k), v / 1.0} end)

          _ ->
            default_weights()
        end

      {:error, _} ->
        default_weights()
    end
  rescue
    _ -> default_weights()
  end

  defp persist_weights(weights) do
    path = weights_file_path()
    dir = Path.dirname(path)

    File.mkdir_p!(dir)

    json_data =
      weights
      |> Map.new(fn {k, v} -> {Atom.to_string(k), Float.round(v, 6)} end)
      |> Jason.encode!(pretty: true)

    File.write(path, json_data)
  rescue
    e ->
      Logger.warning("Failed to persist comprehension weights: #{Exception.message(e)}")
  end

  defp weights_file_path do
    case Application.get_env(:brain, :priv_path) do
      nil -> Path.join(:code.priv_dir(:brain), "analysis/comprehension_weights.json")
      path -> Path.join(path, "analysis/comprehension_weights.json")
    end
  rescue
    _ -> @weights_path
  end

  defp extract_profile_id(candidate) do
    # Look for comprehension_profile_id in the finding or candidate
    cond do
      is_map(candidate) and is_map(Map.get(candidate, :finding)) ->
        Map.get(candidate.finding, :comprehension_profile_id)

      is_map(candidate) ->
        Map.get(candidate, :comprehension_profile_id)

      true ->
        nil
    end
  end

  defp increment_stat(key) do
    :ets.update_counter(@stats_table, key, {2, 1}, {key, 0})
  rescue
    _ -> :ok
  end

  defp get_stat(key) do
    case :ets.lookup(@stats_table, key) do
      [{^key, val}] -> val
      _ -> 0
    end
  rescue
    _ -> 0
  end

  defp cleanup_expired_profiles do
    now = System.system_time(:second)
    cutoff = now - @profile_ttl_seconds

    # Match profiles older than cutoff
    :ets.select_delete(@profiles_table, [
      {{:_, :_, :"$1"}, [{:<, :"$1", cutoff}], [true]}
    ])
  rescue
    _ -> :ok
  end
end
