defmodule Brain.ML.TrainingServer do
  @moduledoc """
  GenServer for managing async ML training jobs with scheduling.

  Provides a centralized interface for starting, monitoring, and scheduling
  training runs for the various ML models (TF-IDF classifier, unified LSTM,
  response scorer).

  ## Model Types

  - `:tfidf` - Classical TF-IDF + centroid intent classifier
  - `:unified` - Unified LSTM model (intent, NER, sentiment, speech act)
  - `:response` - LSTM response scorer

  ## Usage

      # Start training
      TrainingServer.start_training(:unified, epochs: 20, name: "experiment_1")

      # Check status
      TrainingServer.get_status()
      # => :idle | {:training, :unified, ~U[2024-01-15 12:00:00Z]}

      # Schedule recurring training
      TrainingServer.schedule(:tfidf, [], 24)  # every 24 hours

      # List and cancel schedules
      TrainingServer.list_schedules()
      TrainingServer.cancel_schedule("schedule_abc123")
  """

  use GenServer
  require Logger

  alias Brain.ML.LSTM.ExperimentTracker

  @type model_type :: :tfidf | :unified | :response
  @type status :: :idle | {:training, model_type(), DateTime.t()}
  @type schedule :: %{
          id: String.t(),
          model_type: model_type(),
          config: keyword(),
          interval_hours: pos_integer(),
          timer_ref: reference()
        }

  defstruct status: :idle,
            task_ref: nil,
            schedules: []

  # ============================================================================
  # Client API
  # ============================================================================

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Start an async training job for the given model type.

  ## Options
  - `:epochs` - Number of training epochs (default varies by model)
  - `:batch_size` - Training batch size (default varies by model)
  - `:name` - Experiment name for tracking
  - `:learning_rate` - Learning rate override

  Returns `{:ok, model_type}` or `{:error, reason}`.
  """
  @spec start_training(model_type(), keyword()) :: {:ok, model_type()} | {:error, term()}
  def start_training(model_type, config \\ [], name \\ __MODULE__) do
    GenServer.call(name, {:start_training, model_type, config}, 5_000)
  end

  @doc """
  Get the current training status.

  Returns `:idle` or `{:training, model_type, started_at}`.
  """
  @spec get_status(GenServer.name()) :: status()
  def get_status(name \\ __MODULE__) do
    try do
      GenServer.call(name, :get_status, 2_000)
    catch
      :exit, _ -> :idle
    end
  end

  @doc """
  Cancel an in-progress training job.

  Returns `:ok` or `{:error, :not_training}`.
  """
  @spec cancel(GenServer.name()) :: :ok | {:error, :not_training}
  def cancel(name \\ __MODULE__) do
    GenServer.call(name, :cancel, 5_000)
  end

  @doc """
  Schedule recurring training for a model.

  ## Parameters
  - `model_type` - The model to train
  - `config` - Training configuration (keyword list)
  - `interval_hours` - Hours between training runs

  Returns `{:ok, schedule_id}`.
  """
  @spec schedule(model_type(), keyword(), pos_integer(), GenServer.name()) ::
          {:ok, String.t()}
  def schedule(model_type, config \\ [], interval_hours, name \\ __MODULE__) do
    GenServer.call(name, {:schedule, model_type, config, interval_hours}, 5_000)
  end

  @doc """
  List all active schedules.
  """
  @spec list_schedules(GenServer.name()) :: [schedule()]
  def list_schedules(name \\ __MODULE__) do
    try do
      GenServer.call(name, :list_schedules, 2_000)
    catch
      :exit, _ -> []
    end
  end

  @doc """
  Cancel an active schedule by ID.

  Returns `:ok` or `{:error, :not_found}`.
  """
  @spec cancel_schedule(String.t(), GenServer.name()) :: :ok | {:error, :not_found}
  def cancel_schedule(schedule_id, name \\ __MODULE__) do
    GenServer.call(name, {:cancel_schedule, schedule_id}, 5_000)
  end

  # ============================================================================
  # Server Callbacks
  # ============================================================================

  @impl true
  def init(_opts) do
    {:ok, %__MODULE__{}}
  end

  @impl true
  def handle_call({:start_training, model_type, config}, _from, %{status: :idle} = state) do
    unless model_type in [:tfidf, :unified, :response] do
      {:reply, {:error, :invalid_model_type}, state}
    else
      started_at = DateTime.utc_now()

      broadcast_progress({:training_started, model_type, started_at})

      task =
        Task.async(fn ->
          run_training(model_type, config)
        end)

      new_state = %{
        state
        | status: {:training, model_type, started_at},
          task_ref: task.ref
      }

      {:reply, {:ok, model_type}, new_state}
    end
  end

  def handle_call({:start_training, _model_type, _config}, _from, state) do
    {:training, current_model, _started_at} = state.status
    {:reply, {:error, {:already_training, current_model}}, state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    {:reply, state.status, state}
  end

  @impl true
  def handle_call(:cancel, _from, %{status: :idle} = state) do
    {:reply, {:error, :not_training}, state}
  end

  def handle_call(:cancel, _from, %{task_ref: ref} = state) when is_reference(ref) do
    Process.demonitor(ref, [:flush])

    {:training, model_type, _started_at} = state.status

    broadcast_progress({:training_cancelled, model_type})

    Logger.info("TrainingServer: Cancelled training for #{model_type}")

    new_state = %{state | status: :idle, task_ref: nil}
    {:reply, :ok, new_state}
  end

  def handle_call(:cancel, _from, state) do
    new_state = %{state | status: :idle, task_ref: nil}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:schedule, model_type, config, interval_hours}, _from, state) do
    schedule_id = generate_schedule_id()
    interval_ms = interval_hours * 60 * 60 * 1_000
    timer_ref = Process.send_after(self(), {:scheduled_training, schedule_id}, interval_ms)

    schedule = %{
      id: schedule_id,
      model_type: model_type,
      config: config,
      interval_hours: interval_hours,
      timer_ref: timer_ref
    }

    new_state = %{state | schedules: [schedule | state.schedules]}

    Logger.info(
      "TrainingServer: Scheduled #{model_type} training every #{interval_hours}h (id: #{schedule_id})"
    )

    broadcast_progress({:schedule_added, schedule_id, model_type, interval_hours})

    {:reply, {:ok, schedule_id}, new_state}
  end

  @impl true
  def handle_call(:list_schedules, _from, state) do
    schedules =
      Enum.map(state.schedules, fn s ->
        %{
          id: s.id,
          model_type: s.model_type,
          config: s.config,
          interval_hours: s.interval_hours
        }
      end)

    {:reply, schedules, state}
  end

  @impl true
  def handle_call({:cancel_schedule, schedule_id}, _from, state) do
    case Enum.find(state.schedules, fn s -> s.id == schedule_id end) do
      nil ->
        {:reply, {:error, :not_found}, state}

      schedule ->
        Process.cancel_timer(schedule.timer_ref)

        new_schedules = Enum.reject(state.schedules, fn s -> s.id == schedule_id end)
        new_state = %{state | schedules: new_schedules}

        Logger.info("TrainingServer: Cancelled schedule #{schedule_id}")
        broadcast_progress({:schedule_cancelled, schedule_id})

        {:reply, :ok, new_state}
    end
  end

  # ============================================================================
  # Info Handlers
  # ============================================================================

  @impl true
  def handle_info({ref, result}, %{task_ref: ref} = state) when is_reference(ref) do
    # Task completed - flush the DOWN message
    Process.demonitor(ref, [:flush])

    {:training, model_type, started_at} = state.status
    elapsed = DateTime.diff(DateTime.utc_now(), started_at, :second)

    case result do
      {:ok, training_result} ->
        Logger.info(
          "TrainingServer: Training #{model_type} completed in #{elapsed}s"
        )

        record_experiment(model_type, training_result, elapsed)
        broadcast_progress({:training_complete, model_type, {:ok, training_result}})
        maybe_reload_model(model_type)

      {:error, reason} ->
        Logger.warning(
          "TrainingServer: Training #{model_type} failed after #{elapsed}s: #{inspect(reason)}"
        )

        broadcast_progress({:training_complete, model_type, {:error, reason}})
    end

    {:noreply, %{state | status: :idle, task_ref: nil}}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task_ref: ref} = state) do
    {:training, model_type, _started_at} = state.status

    Logger.warning("TrainingServer: Training task for #{model_type} crashed: #{inspect(reason)}")
    broadcast_progress({:training_complete, model_type, {:error, {:crashed, reason}}})

    {:noreply, %{state | status: :idle, task_ref: nil}}
  end

  def handle_info({:scheduled_training, schedule_id}, state) do
    case Enum.find(state.schedules, fn s -> s.id == schedule_id end) do
      nil ->
        {:noreply, state}

      schedule ->
        # Re-schedule the next run
        interval_ms = schedule.interval_hours * 60 * 60 * 1_000
        new_timer_ref = Process.send_after(self(), {:scheduled_training, schedule_id}, interval_ms)

        updated_schedule = %{schedule | timer_ref: new_timer_ref}

        new_schedules =
          Enum.map(state.schedules, fn s ->
            if s.id == schedule_id, do: updated_schedule, else: s
          end)

        new_state = %{state | schedules: new_schedules}

        # Start training if idle
        case state.status do
          :idle ->
            Logger.info(
              "TrainingServer: Scheduled training triggered for #{schedule.model_type} (schedule: #{schedule_id})"
            )

            started_at = DateTime.utc_now()
            broadcast_progress({:training_started, schedule.model_type, started_at})

            task =
              Task.async(fn ->
                run_training(schedule.model_type, schedule.config)
              end)

            {:noreply,
             %{
               new_state
               | status: {:training, schedule.model_type, started_at},
                 task_ref: task.ref
             }}

          {:training, current_model, _} ->
            Logger.info(
              "TrainingServer: Skipping scheduled #{schedule.model_type} training, already training #{current_model}"
            )

            {:noreply, new_state}
        end
    end
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # ============================================================================
  # Private: Training Dispatch
  # ============================================================================

  defp run_training(:tfidf, _config) do
    Brain.ML.Trainer.train_and_save()
  end

  defp run_training(:unified, config) do
    Brain.ML.LSTM.UnifiedModel.train(config)
  end

  defp run_training(:response, config) do
    Brain.Response.LSTMResponse.train(config)
  end

  # ============================================================================
  # Private: Post-Training Actions
  # ============================================================================

  defp record_experiment(model_type, training_result, elapsed_seconds) do
    try do
      ExperimentTracker.record(%{
        name: "auto_#{model_type}_#{System.os_time(:second)}",
        config: extract_config(training_result),
        training_time_seconds: elapsed_seconds,
        notes: "Trained via TrainingServer"
      })
    rescue
      e ->
        Logger.warning(
          "TrainingServer: Failed to record experiment: #{Exception.message(e)}"
        )
    end
  end

  defp extract_config(training_result) when is_map(training_result) do
    Map.get(training_result, :config) || Map.get(training_result, :vocabularies, %{})
  end

  defp extract_config(_), do: %{}

  defp maybe_reload_model(:unified) do
    if model_ready?(Brain.ML.LSTM.UnifiedModel) do
      try do
        Brain.ML.LSTM.UnifiedModel.reload()
      rescue
        _ -> :ok
      catch
        :exit, _ -> :ok
      end
    end
  end

  defp maybe_reload_model(:response) do
    if model_ready?(Brain.Response.LSTMResponse) do
      try do
        Brain.Response.LSTMResponse.reload()
      rescue
        _ -> :ok
      catch
        :exit, _ -> :ok
      end
    end
  end

  defp maybe_reload_model(_), do: :ok

  defp model_ready?(module) do
    try do
      module.ready?()
      true
    rescue
      _ -> false
    catch
      :exit, _ -> false
    end
  end

  # ============================================================================
  # Private: Broadcasting
  # ============================================================================

  defp broadcast_progress(message) do
    Phoenix.PubSub.broadcast(Brain.PubSub, "training:progress", message)
  end

  # ============================================================================
  # Private: Utilities
  # ============================================================================

  defp generate_schedule_id do
    :crypto.strong_rand_bytes(8) |> Base.url_encode64(padding: false)
  end
end
