defmodule Brain.ML.TrainingServer do
  @moduledoc """
  GenServer for managing async ML training jobs with scheduling.

  Provides a centralized interface for starting, monitoring, and scheduling
  training runs for ML models. One job runs at a time, in a process under
  `Brain.ML.TrainingServer.TaskSupervisor`, so a job that crashes is
  reported rather than taking the server down with it.

  ## Model Types

  - `:tfidf` - Classical TF-IDF + centroid intent classifier
  - `:pos` - a recorded POS tagger run (`Brain.Training.POSRuns`); config is
    the run params (`sentences`, `max_epochs`, `patience`, `seed`)
  - `:gen_micro_data`, `:train_micro`, `:gen_framing_data`,
    `:train_framing`, `:evaluate`, `:reload_models`

  ## Progress

  Start, completion and cancellation are broadcast on `"training:progress"`.
  A POS run also broadcasts on `"training:pos"`: `{:pos_run_started, id}`,
  `{:pos_epoch, id, progress}` after every epoch, and
  `{:pos_run_finished, id, status}`.

  ## Usage

      TrainingServer.start_training(:pos, sentences: 2_000, max_epochs: 100, patience: nil, seed: 42)
      TrainingServer.get_status()
      # => :idle | {:training, :pos, ~U[2026-09-19 12:00:00Z]}
      TrainingServer.cancel()

      TrainingServer.schedule(:tfidf, [], 24)  # every 24 hours
      TrainingServer.list_schedules()
      TrainingServer.cancel_schedule("schedule_abc123")

  A POS run left `running` by a previous server is marked `interrupted`
  when the server starts.
  """

  alias Phoenix.PubSub
  alias Brain.ML.Trainer
  alias Brain.Training.POSRuns
  use GenServer
  require Logger

  @type model_type ::
          :tfidf | :pos | :gen_micro_data | :train_micro | :gen_framing_data | :train_framing | :evaluate
  @type status :: :idle | {:training, model_type(), DateTime.t()}
  @type schedule :: %{
          id: String.t(),
          model_type: model_type(),
          config: keyword(),
          interval_hours: pos_integer(),
          timer_ref: reference()
        }

  @task_supervisor Brain.ML.TrainingServer.TaskSupervisor
  @pos_topic "training:pos"

  defstruct status: :idle, task: nil, run_id: nil, schedules: [], task_supervisor: nil, runs_root: nil

  @doc """
  Starts the server. Options: `:name`; `:task_supervisor` to run jobs under
  (default `Brain.ML.TrainingServer.TaskSupervisor`); `:runs_root` for POS
  runs (default `Brain.Training.POSRuns.root/0`).
  """
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "The PubSub topic POS runs broadcast their progress on."
  def pos_topic, do: @pos_topic

  @doc "Start an async training job for the given model type.\n\n## Options\n- `:epochs` - Number of training epochs (default varies by model)\n- `:batch_size` - Training batch size (default varies by model)\n- `:name` - Experiment name for tracking\n- `:learning_rate` - Learning rate override\n\nFor `:pos` the options are the run params; invalid params are refused\nwith `{:error, message}` before anything starts.\n\nReturns `{:ok, model_type}` (`{:ok, :pos, run_id}` for a POS run) or `{:error, reason}`.\n"
  @spec start_training(model_type(), keyword(), GenServer.name()) ::
          {:ok, model_type()} | {:ok, :pos, String.t()} | {:error, term()}
  def start_training(model_type, config \\ [], name \\ __MODULE__) do
    GenServer.call(name, {:start_training, model_type, config}, 5000)
  end

  @doc "Get the current training status: `:idle` or `{:training, model_type, started_at}`."
  @spec get_status(GenServer.name()) :: status()
  def get_status(name \\ __MODULE__) do
    GenServer.call(name, :get_status, 2000)
  end

  @doc "The id of the POS run in progress, or nil."
  @spec current_run(GenServer.name()) :: String.t() | nil
  def current_run(name \\ __MODULE__) do
    GenServer.call(name, :current_run, 2000)
  end

  @doc """
  Stops the job in progress: its process is killed, and a POS run is
  recorded as `cancelled`. Returns `:ok` or `{:error, :not_training}`.
  """
  @spec cancel(GenServer.name()) :: :ok | {:error, :not_training}
  def cancel(name \\ __MODULE__) do
    GenServer.call(name, :cancel, 15_000)
  end

  @doc "Schedule recurring training for a model.\n\n## Parameters\n- `model_type` - The model to train\n- `config` - Training configuration (keyword list)\n- `interval_hours` - Hours between training runs\n\nReturns `{:ok, schedule_id}`.\n"
  @spec schedule(model_type(), keyword(), pos_integer(), GenServer.name()) ::
          {:ok, String.t()}
  def schedule(model_type, config \\ [], interval_hours, name \\ __MODULE__) do
    GenServer.call(name, {:schedule, model_type, config, interval_hours}, 5000)
  end

  @doc "List all active schedules.\n"
  @spec list_schedules(GenServer.name()) :: [schedule()]
  def list_schedules(name \\ __MODULE__) do
    GenServer.call(name, :list_schedules, 2000)
  end

  @doc "Cancel an active schedule by ID.\n\nReturns `:ok` or `{:error, :not_found}`.\n"
  @spec cancel_schedule(String.t(), GenServer.name()) :: :ok | {:error, :not_found}
  def cancel_schedule(schedule_id, name \\ __MODULE__) do
    GenServer.call(name, {:cancel_schedule, schedule_id}, 5000)
  end

  @impl true
  def init(opts) do
    runs_root = Keyword.get_lazy(opts, :runs_root, &POSRuns.root/0)

    case POSRuns.mark_interrupted!(runs_root) do
      [] -> :ok
      ids -> Logger.warning("TrainingServer: POS runs #{inspect(ids)} were running when the app stopped; marked interrupted")
    end

    {:ok,
     %__MODULE__{
       task_supervisor: Keyword.get(opts, :task_supervisor, @task_supervisor),
       runs_root: runs_root
     }}
  end

  @valid_types [:tfidf, :pos, :gen_micro_data, :train_micro, :gen_framing_data, :train_framing, :evaluate, :reload_models]

  @impl true
  def handle_call({:start_training, model_type, config}, _from, %{status: :idle} = state) do
    cond do
      model_type not in @valid_types ->
        {:reply, {:error, :invalid_model_type}, state}

      model_type == :pos ->
        try do
          params = POSRuns.params!(config)
          state = launch(:pos, params, state)
          {:reply, {:ok, :pos, state.run_id}, state}
        rescue
          e in ArgumentError -> {:reply, {:error, Exception.message(e)}, state}
        end

      true ->
        {:reply, {:ok, model_type}, launch(model_type, config, state)}
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

  def handle_call(:current_run, _from, state) do
    {:reply, state.run_id, state}
  end

  @impl true
  def handle_call(:cancel, _from, %{status: :idle} = state) do
    {:reply, {:error, :not_training}, state}
  end

  def handle_call(:cancel, _from, %{task: %Task{} = task} = state) do
    :ok = Task.Supervisor.terminate_child(state.task_supervisor, task.pid)
    Process.demonitor(task.ref, [:flush])

    {:training, model_type, _started_at} = state.status

    if state.run_id do
      POSRuns.mark!(state.run_id, "cancelled", state.runs_root)
      PubSub.broadcast(Brain.PubSub, @pos_topic, {:pos_run_finished, state.run_id, "cancelled"})
    end

    broadcast_progress({:training_cancelled, model_type})
    Logger.info("TrainingServer: Cancelled training for #{model_type}")

    {:reply, :ok, idle(state)}
  end

  @impl true
  def handle_call({:schedule, model_type, config, interval_hours}, _from, state) do
    schedule_id = generate_schedule_id()
    interval_ms = interval_hours * 60 * 60 * 1000
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

  @impl true
  def handle_info({ref, result}, %{task: %Task{ref: ref}} = state) do
    Process.demonitor(ref, [:flush])

    {:training, model_type, started_at} = state.status
    elapsed = DateTime.diff(DateTime.utc_now(), started_at, :second)

    case result do
      {:ok, training_result} ->
        Logger.info("TrainingServer: Training #{model_type} completed in #{elapsed}s")

        broadcast_progress({:training_complete, model_type, {:ok, training_result}})
        maybe_reload_model(model_type)

      {:error, reason} ->
        Logger.warning(
          "TrainingServer: Training #{model_type} failed after #{elapsed}s: #{inspect(reason)}"
        )

        broadcast_progress({:training_complete, model_type, {:error, reason}})
    end

    if state.run_id, do: PubSub.broadcast(Brain.PubSub, @pos_topic, {:pos_run_finished, state.run_id, "completed"})

    {:noreply, idle(state)}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, %{task: %Task{ref: ref}} = state) do
    {:training, model_type, _started_at} = state.status

    Logger.warning("TrainingServer: Training task for #{model_type} crashed: #{inspect(reason)}")
    broadcast_progress({:training_complete, model_type, {:error, {:crashed, reason}}})

    if state.run_id do
      # POSRuns.run!/2 records an exception itself; an exit leaves the run
      # recorded as running.
      if POSRuns.get!(state.run_id, state.runs_root)["status"] == "running",
        do: POSRuns.mark!(state.run_id, "failed", state.runs_root, "process exited: #{inspect(reason)}")

      PubSub.broadcast(Brain.PubSub, @pos_topic, {:pos_run_finished, state.run_id, "failed"})
    end

    {:noreply, idle(state)}
  end

  def handle_info({:scheduled_training, schedule_id}, state) do
    case Enum.find(state.schedules, fn s -> s.id == schedule_id end) do
      nil ->
        {:noreply, state}

      schedule ->
        interval_ms = schedule.interval_hours * 60 * 60 * 1000

        new_timer_ref =
          Process.send_after(self(), {:scheduled_training, schedule_id}, interval_ms)

        updated_schedule = %{schedule | timer_ref: new_timer_ref}

        new_schedules =
          Enum.map(state.schedules, fn s ->
            if s.id == schedule_id do
              updated_schedule
            else
              s
            end
          end)

        new_state = %{state | schedules: new_schedules}

        case state.status do
          :idle ->
            Logger.info(
              "TrainingServer: Scheduled training triggered for #{schedule.model_type} (schedule: #{schedule_id})"
            )

            {:noreply, launch(schedule.model_type, schedule.config, new_state)}

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

  # Runs the job under the task supervisor, not linked: a job that crashes
  # arrives as :DOWN instead of killing the server.
  defp launch(model_type, config, state) do
    started_at = DateTime.utc_now()
    {run_id, job} = job(model_type, config, state)

    broadcast_progress({:training_started, model_type, started_at})
    if run_id, do: PubSub.broadcast(Brain.PubSub, @pos_topic, {:pos_run_started, run_id})

    task = Task.Supervisor.async_nolink(state.task_supervisor, job)
    %{state | status: {:training, model_type, started_at}, task: task, run_id: run_id}
  end

  defp job(:pos, params, state) do
    run_id = POSRuns.new_id()
    root = state.runs_root

    run = fn ->
      {:ok,
       POSRuns.run!(params,
         id: run_id,
         root: root,
         on_epoch: fn id, progress -> PubSub.broadcast(Brain.PubSub, @pos_topic, {:pos_epoch, id, progress}) end
       )}
    end

    {run_id, run}
  end

  defp job(model_type, config, _state), do: {nil, fn -> run_training(model_type, config) end}

  defp idle(state), do: %{state | status: :idle, task: nil, run_id: nil}

  defp run_training(:tfidf, _config) do
    Trainer.train_and_save()
  end

  defp run_training(:gen_micro_data, config) do
    args = case Keyword.get(config, :only) do
      nil -> []
      name -> ["--only", to_string(name)]
    end

    Mix.Task.rerun("gen_micro_data", args)
    {:ok, :gen_micro_data_complete}
  end

  defp run_training(:train_micro, config) do
    args = case Keyword.get(config, :only) do
      nil -> []
      name -> ["--only", to_string(name)]
    end

    Mix.Task.rerun("train_micro", args)
    {:ok, :train_micro_complete}
  end

  defp run_training(:gen_framing_data, _config) do
    Mix.Task.rerun("gen_framing_data", ["--corpus", "gvfc"])
    {:ok, :gen_framing_data_complete}
  end

  defp run_training(:train_framing, _config) do
    Mix.Task.rerun("train_framing", [])
    {:ok, :train_framing_complete}
  end

  defp run_training(:evaluate, config) do
    task = Keyword.get(config, :task, "intent")
    Mix.Task.rerun("evaluate.#{task}", ["--save"])
    {:ok, {:evaluate_complete, task}}
  end

  defp run_training(:reload_models, _config) do
    Brain.ML.MicroClassifiers.reload()
    {:ok, :reload_complete}
  end

  defp maybe_reload_model(:tfidf = model_type) do
    result =
      try do
        Brain.ML.MicroClassifiers.reload()
      rescue
        e ->
          Logger.warning(
            "TrainingServer: MicroClassifiers.reload/0 raised after #{model_type} training: #{Exception.message(e)}"
          )

          {:error, {:exception, Exception.message(e)}}
      catch
        :exit, reason ->
          Logger.warning(
            "TrainingServer: MicroClassifiers.reload/0 exited after #{model_type} training: #{inspect(reason)}"
          )

          {:error, {:exit, reason}}
      end

    broadcast_progress({:model_reloaded, model_type, result})
    result
  end

  defp maybe_reload_model(:train_micro), do: maybe_reload_model(:tfidf)
  defp maybe_reload_model(:train_framing), do: maybe_reload_model(:tfidf)
  defp maybe_reload_model(:reload_models), do: :ok

  defp maybe_reload_model(other) do
    Logger.debug("TrainingServer: no reload handler for model_type=#{inspect(other)}")
    :ok
  end

  defp broadcast_progress(message) do
    PubSub.broadcast(Brain.PubSub, "training:progress", message)
  end

  defp generate_schedule_id do
    :crypto.strong_rand_bytes(8) |> Base.url_encode64(padding: false)
  end
end
