defmodule Brain.Analysis.AnalyzerCalibration do
  @moduledoc """
  Tracks historical accuracy of analyzers and calibrates their scores.

  Different analyzers (memory similarity, pattern recognition, keyword matching)
  produce confidence scores on different scales. This module:

  1. Tracks: "when analyzer X says Y% confident, how often is it actually right?"
  2. Calibrates raw scores based on historical accuracy
  3. Periodically recalibrates when error exceeds threshold

  This addresses the "racing analyzers that aren't commensurate" risk.
  """

  use GenServer
  require Logger

  @recalibration_interval 100
  @max_calibration_error 0.15
  @bucket_count 10

  # ETS table for fast reads
  @table :analyzer_calibration

  # Analyzers we track
  @tracked_analyzers [
    :memory_similarity,
    :pattern_recognition,
    :structural,
    :keyword,
    :model,
    :early_confidence
  ]

  # Default calibration (assume analyzers are initially well-calibrated)
  @default_bucket_accuracy 0.8

  # State structure
  defstruct outcomes_since_recalibration: %{},
            total_outcomes: %{},
            last_recalibration: %{}

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Calibrates a raw confidence score based on historical accuracy.

  Returns {calibrated_score, calibration_error}
  """
  def calibrate(analyzer, raw_confidence) when is_atom(analyzer) and is_float(raw_confidence) do
    bucket = confidence_bucket(raw_confidence)
    historical_accuracy = get_bucket_accuracy(analyzer, bucket)
    calibration_error = get_calibration_error(analyzer)

    # Calibrated score = raw * historical_accuracy
    calibrated = raw_confidence * historical_accuracy

    {calibrated, calibration_error}
  end

  @doc """
  Records an outcome for calibration tracking.

  Call this after determining whether an interpretation was correct.
  """
  def track_outcome(analyzer, predicted_confidence, was_correct)
      when is_atom(analyzer) and is_float(predicted_confidence) and is_boolean(was_correct) do
    GenServer.cast(__MODULE__, {:track_outcome, analyzer, predicted_confidence, was_correct})
  end

  @doc """
  Gets the calibration error for an analyzer.

  Lower is better - represents average deviation between predicted and actual accuracy.
  """
  def get_calibration_error(analyzer) when is_atom(analyzer) do
    case :ets.lookup(@table, {:error, analyzer}) do
      [{_, error}] -> error
      [] -> 0.0
    end
  end

  @doc """
  Gets the historical accuracy for a specific confidence bucket.
  """
  def get_bucket_accuracy(analyzer, bucket) when is_atom(analyzer) and is_integer(bucket) do
    case :ets.whereis(@table) do
      :undefined ->
        @default_bucket_accuracy

      _tid ->
        case :ets.lookup(@table, {:bucket, analyzer, bucket}) do
          [{_, accuracy, _count}] -> accuracy
          [] -> @default_bucket_accuracy
        end
    end
  end

  @doc """
  Forces recalibration of a specific analyzer.
  """
  def recalibrate(analyzer) when is_atom(analyzer) do
    GenServer.call(__MODULE__, {:recalibrate, analyzer})
  end

  @doc """
  Returns calibration stats for all analyzers.
  """
  def stats do
    GenServer.call(__MODULE__, :stats)
  end

  @doc """
  Checks if an analyzer is dominating (winning too often).

  Useful for stability self-reflection (ST2).
  """
  def is_dominating?(analyzer, threshold \\ 0.7) do
    case :ets.lookup(@table, {:win_rate, analyzer}) do
      [{_, win_rate}] -> win_rate > threshold
      [] -> false
    end
  end

  @doc """
  Returns the list of tracked analyzers.
  """
  def tracked_analyzers, do: @tracked_analyzers

  @doc """
  Checks if the analyzer calibration is ready.
  """
  def ready? do
    try do
      GenServer.call(__MODULE__, :ready?, 100)
    catch
      :exit, {:timeout, _} -> false
      :exit, {:noproc, _} -> false
    end
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Create ETS table for fast reads
    :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])

    # Initialize buckets for each analyzer
    initialize_calibration_data()

    state = %__MODULE__{
      outcomes_since_recalibration: Map.new(@tracked_analyzers, fn a -> {a, 0} end),
      total_outcomes: Map.new(@tracked_analyzers, fn a -> {a, 0} end),
      last_recalibration:
        Map.new(@tracked_analyzers, fn a -> {a, System.monotonic_time(:second)} end)
    }

    Logger.info("AnalyzerCalibration started", %{tracked: @tracked_analyzers})

    {:ok, state}
  end

  @impl true
  def handle_cast({:track_outcome, analyzer, predicted_confidence, was_correct}, state) do
    bucket = confidence_bucket(predicted_confidence)

    # Update bucket stats
    update_bucket_stats(analyzer, bucket, was_correct)

    # Track outcomes since last recalibration
    new_outcomes = Map.update(state.outcomes_since_recalibration, analyzer, 1, &(&1 + 1))
    new_total = Map.update(state.total_outcomes, analyzer, 1, &(&1 + 1))

    # Check if recalibration needed
    outcomes_count = Map.get(new_outcomes, analyzer, 0)
    error = calculate_calibration_error(analyzer)

    new_state =
      if outcomes_count >= @recalibration_interval or error > @max_calibration_error do
        perform_recalibration(analyzer)

        %{
          state
          | outcomes_since_recalibration: Map.put(new_outcomes, analyzer, 0),
            total_outcomes: new_total,
            last_recalibration:
              Map.put(state.last_recalibration, analyzer, System.monotonic_time(:second))
        }
      else
        %{state | outcomes_since_recalibration: new_outcomes, total_outcomes: new_total}
      end

    {:noreply, new_state}
  end

  @impl true
  def handle_call({:recalibrate, analyzer}, _from, state) do
    perform_recalibration(analyzer)

    new_state = %{
      state
      | outcomes_since_recalibration: Map.put(state.outcomes_since_recalibration, analyzer, 0),
        last_recalibration:
          Map.put(state.last_recalibration, analyzer, System.monotonic_time(:second))
    }

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats =
      Enum.map(@tracked_analyzers, fn analyzer ->
        error = get_calibration_error(analyzer)
        total = Map.get(state.total_outcomes, analyzer, 0)
        since_recal = Map.get(state.outcomes_since_recalibration, analyzer, 0)

        {analyzer,
         %{
           calibration_error: error,
           total_outcomes: total,
           outcomes_since_recalibration: since_recal,
           bucket_accuracies: get_all_bucket_accuracies(analyzer)
         }}
      end)
      |> Map.new()

    {:reply, stats, state}
  end

  @impl true
  def handle_call(:ready?, _from, state) do
    {:reply, true, state}
  end

  # Private functions

  defp initialize_calibration_data do
    # Initialize each analyzer with default bucket accuracies
    for analyzer <- @tracked_analyzers, bucket <- 0..(@bucket_count - 1) do
      :ets.insert(@table, {{:bucket, analyzer, bucket}, @default_bucket_accuracy, 0})
    end

    # Initialize error tracking
    for analyzer <- @tracked_analyzers do
      :ets.insert(@table, {{:error, analyzer}, 0.0})
      :ets.insert(@table, {{:win_rate, analyzer}, 0.0})
    end
  end

  defp confidence_bucket(confidence) when is_float(confidence) do
    # Map 0.0-1.0 to buckets 0-9
    bucket = trunc(confidence * @bucket_count)
    min(bucket, @bucket_count - 1)
  end

  defp update_bucket_stats(analyzer, bucket, was_correct) do
    case :ets.lookup(@table, {:bucket, analyzer, bucket}) do
      [{key, current_accuracy, count}] ->
        # Exponential moving average for accuracy
        alpha = min(1.0 / (count + 1), 0.1)
        outcome_value = if was_correct, do: 1.0, else: 0.0
        new_accuracy = current_accuracy * (1 - alpha) + outcome_value * alpha
        new_count = count + 1

        :ets.insert(@table, {key, new_accuracy, new_count})

      [] ->
        # First entry for this bucket
        accuracy = if was_correct, do: 1.0, else: 0.0
        :ets.insert(@table, {{:bucket, analyzer, bucket}, accuracy, 1})
    end
  end

  defp calculate_calibration_error(analyzer) do
    # Calculate average absolute difference between bucket midpoint and actual accuracy
    bucket_data =
      Enum.map(0..(@bucket_count - 1), fn bucket ->
        case :ets.lookup(@table, {:bucket, analyzer, bucket}) do
          [{_, accuracy, count}] when count > 0 ->
            midpoint = (bucket + 0.5) / @bucket_count
            {abs(midpoint - accuracy) * count, count}

          _ ->
            {0.0, 0}
        end
      end)

    total_weighted_error = Enum.sum(Enum.map(bucket_data, &elem(&1, 0)))
    total_count = Enum.sum(Enum.map(bucket_data, &elem(&1, 1)))

    error =
      if total_count > 0 do
        total_weighted_error / total_count
      else
        0.0
      end

    # Update stored error
    :ets.insert(@table, {{:error, analyzer}, error})

    error
  end

  defp perform_recalibration(analyzer) do
    # Recalculate error
    error = calculate_calibration_error(analyzer)

    Logger.debug("Recalibrated analyzer", %{
      analyzer: analyzer,
      calibration_error: error
    })

    :ok
  end

  defp get_all_bucket_accuracies(analyzer) do
    Enum.map(0..(@bucket_count - 1), fn bucket ->
      case :ets.lookup(@table, {:bucket, analyzer, bucket}) do
        [{_, accuracy, count}] -> {bucket, accuracy, count}
        [] -> {bucket, @default_bucket_accuracy, 0}
      end
    end)
  end
end
