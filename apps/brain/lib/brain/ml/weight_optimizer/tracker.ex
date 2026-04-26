defmodule Brain.ML.WeightOptimizer.Tracker do
  @moduledoc """
  In-memory state + persistence for `Brain.ML.WeightOptimizer` runs.

  Attaches to the `[:brain, :weight_optimizer, _]` telemetry stream
  emitted by `Brain.ML.WeightOptimizer.optimize/2`, maintains a list
  of currently-running and recently-completed runs, persists completed
  runs to disk, and broadcasts lifecycle events on the
  `weight_optimizer:progress` PubSub topic so LiveViews can render
  live progress.

  Whether a run is launched from the UI via `start_run/2`, from a Mix
  task (`mix train_micro`), or from an iex prompt, every run flows
  through the same telemetry pipeline and lands here.

  ## PubSub messages broadcast on `weight_optimizer:progress`

  - `{:run_started, run}` — a run has started; `run` is a summary map.
  - `{:generation, run_id, snapshot}` — per-generation update.
  - `{:run_complete, run}` — a run has finished (success or early stop).
  - `{:run_failed, run}` — a run raised or the spawned task crashed.
  - `{:run_cancelled, run_id}` — a run was cancelled via `cancel_run/1`.

  ## Persisted runs

  Completed runs are written to
  `priv/weight_optimizer/runs/<run_id>.json`. On startup, the most
  recent `recent_limit` runs are loaded back into memory so the UI is
  populated immediately after a restart.
  """

  use GenServer
  require Logger

  alias Phoenix.PubSub

  @pubsub Brain.PubSub
  @topic "weight_optimizer:progress"

  @recent_limit 50
  @history_cap_per_run 500

  @feature_vector_classifiers ~w(
    intent_full
    intent_domain
    tense_class
    aspect_class
    urgency
    certainty_level
  )

  defstruct active: %{},
            recent: [],
            recent_limit: @recent_limit,
            monitors: %{}

  ## Public API ─────────────────────────────────────────────────────

  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name, __MODULE__)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Returns the PubSub topic for live progress events.
  """
  def topic, do: @topic

  @doc """
  Returns the list of feature-vector classifiers eligible for GA
  weight optimization. Mirrors `Mix.Tasks.TrainMicro.@feature_vector_classifiers`.
  """
  def feature_vector_classifiers, do: @feature_vector_classifiers

  @doc """
  Returns active runs (newest first).
  """
  def list_active(name \\ __MODULE__) do
    GenServer.call(name, :list_active)
  end

  @doc """
  Returns recently completed/failed runs (newest first), capped at
  `recent_limit`.
  """
  def list_recent(name \\ __MODULE__) do
    GenServer.call(name, :list_recent)
  end

  @doc """
  Look up a single run (active or recent) by id.
  """
  def get_run(run_id, name \\ __MODULE__) do
    GenServer.call(name, {:get_run, run_id})
  end

  @doc """
  Kick off a new GA run for `classifier` with the supplied opts.

  Loads the training data from `data/classifiers/<classifier>.json`,
  spawns a supervised Task that runs `Brain.ML.WeightOptimizer.optimize/2`,
  and returns `{:ok, run_id}`. The run becomes visible via the
  `weight_optimizer:progress` topic as the GA emits telemetry.

  Returns `{:error, reason}` if the classifier name is unknown, the
  data file is missing/malformed, or no feature-vector training pairs
  could be parsed.
  """
  @spec start_run(String.t(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def start_run(classifier, opts \\ [], name \\ __MODULE__) do
    GenServer.call(name, {:start_run, classifier, opts}, 15_000)
  end

  @doc """
  Cancels an active run by killing the spawned task. The run is moved
  into recent with `status: :cancelled`.
  """
  @spec cancel_run(String.t()) :: :ok | {:error, :not_found}
  def cancel_run(run_id, name \\ __MODULE__) do
    GenServer.call(name, {:cancel_run, run_id})
  end

  ## GenServer ──────────────────────────────────────────────────────

  @impl true
  def init(opts) do
    recent_limit = Keyword.get(opts, :recent_limit, @recent_limit)
    attach_telemetry()

    state = %__MODULE__{
      recent_limit: recent_limit,
      recent: load_persisted_runs(recent_limit)
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:list_active, _from, state) do
    runs =
      state.active
      |> Map.values()
      |> Enum.sort_by(& &1.started_at, {:desc, DateTime})

    {:reply, runs, state}
  end

  def handle_call(:list_recent, _from, state) do
    {:reply, state.recent, state}
  end

  def handle_call({:get_run, run_id}, _from, state) do
    run =
      Map.get(state.active, run_id) ||
        Enum.find(state.recent, fn r -> r.run_id == run_id end)

    {:reply, run, state}
  end

  def handle_call({:start_run, classifier, opts}, _from, state) do
    case launch_task(classifier, opts) do
      {:ok, run_id, pid, ref} ->
        new_state = %{state | monitors: Map.put(state.monitors, ref, {run_id, pid})}
        {:reply, {:ok, run_id}, new_state}

      {:error, reason} = err ->
        Logger.warning(
          "WeightOptimizer.Tracker: refusing to start run for #{inspect(classifier)}: #{inspect(reason)}"
        )

        {:reply, err, state}
    end
  end

  def handle_call({:cancel_run, run_id}, _from, state) do
    case Map.get(state.active, run_id) do
      nil ->
        {:reply, {:error, :not_found}, state}

      run ->
        case run.task_pid do
          pid when is_pid(pid) ->
            Process.exit(pid, :kill)

          _ ->
            :ok
        end

        cancelled = run |> Map.put(:status, :cancelled) |> Map.put(:completed_at, now())
        new_state =
          state
          |> remove_active(run_id)
          |> push_recent(cancelled)

        broadcast({:run_cancelled, run_id})
        {:reply, :ok, new_state}
    end
  end

  @impl true
  def handle_info({:telemetry, :start, _measurements, metadata}, state) do
    run = build_active_run(metadata)
    new_state = put_in(state.active[run.run_id], run)

    broadcast({:run_started, summary(run)})
    {:noreply, new_state}
  end

  def handle_info({:telemetry, :generation, measurements, metadata}, state) do
    run_id = metadata.run_id

    case Map.get(state.active, run_id) do
      nil ->
        # GA running outside this Tracker's lifetime (e.g. Tracker restarted
        # mid-run). Synthesize a partial entry so the UI still tracks it.
        run =
          metadata
          |> build_active_run()
          |> apply_generation(measurements, metadata)

        broadcast({:run_started, summary(run)})
        broadcast({:generation, run_id, generation_snapshot(run, measurements, metadata)})
        {:noreply, put_in(state.active[run_id], run)}

      run ->
        updated = apply_generation(run, measurements, metadata)
        broadcast({:generation, run_id, generation_snapshot(updated, measurements, metadata)})
        {:noreply, put_in(state.active[run_id], updated)}
    end
  end

  def handle_info({:telemetry, :stop, measurements, metadata}, state) do
    run_id = metadata.run_id
    base = Map.get(state.active, run_id) || build_active_run(metadata)

    completed =
      base
      |> Map.merge(%{
        status: metadata.status,
        completed_at: now(),
        duration_ms: measurements.duration_ms,
        best_fitness: measurements.best_fitness,
        best_generation: measurements.best_generation,
        generations_run: measurements.generations_run,
        alive_dims: measurements.alive_dims,
        total_dims: measurements.total_dims,
        history: clamp_history(metadata[:history] || base.history),
        weights_summary: weights_summary(metadata[:weights])
      })

    persist(completed)

    new_state =
      state
      |> remove_active(run_id)
      |> push_recent(completed)

    broadcast({:run_complete, completed})
    {:noreply, new_state}
  end

  def handle_info({:telemetry, :exception, measurements, metadata}, state) do
    run_id = metadata.run_id
    base = Map.get(state.active, run_id) || build_active_run(metadata)

    failed =
      base
      |> Map.merge(%{
        status: :error,
        completed_at: now(),
        duration_ms: measurements.duration_ms,
        error: %{
          kind: metadata.kind,
          reason: inspect(metadata.reason)
        }
      })

    persist(failed)

    new_state =
      state
      |> remove_active(run_id)
      |> push_recent(failed)

    broadcast({:run_failed, failed})
    {:noreply, new_state}
  end

  def handle_info({:DOWN, ref, :process, _pid, reason}, state) do
    case Map.pop(state.monitors, ref) do
      {nil, _} ->
        {:noreply, state}

      {{run_id, _pid}, monitors} ->
        new_state = %{state | monitors: monitors}

        case reason do
          :normal -> {:noreply, new_state}
          :killed -> {:noreply, new_state}
          _ -> {:noreply, finalize_crashed_run(new_state, run_id, reason)}
        end
    end
  end

  def handle_info(_other, state), do: {:noreply, state}

  ## Telemetry ──────────────────────────────────────────────────────

  # Telemetry warns when a handler is an anonymous function or a local
  # function capture because such handlers can't be hot-code-upgraded
  # and the BEAM must walk the closure's captured env on every event.
  # We use a module-function capture (`&__MODULE__.__handle_telemetry__/4`)
  # and route the destination pid via the per-handler `config` map so
  # the function itself stays stateless.

  defp attach_telemetry do
    pid = self()

    events = [
      [:brain, :weight_optimizer, :start],
      [:brain, :weight_optimizer, :generation],
      [:brain, :weight_optimizer, :stop],
      [:brain, :weight_optimizer, :exception]
    ]

    Enum.each(events, fn event ->
      handler_id = "weight-optimizer-tracker-" <> Enum.join(event, "-")

      :telemetry.detach(handler_id)

      :telemetry.attach(
        handler_id,
        event,
        &__MODULE__.__handle_telemetry__/4,
        %{target: pid}
      )
    end)

    :ok
  end

  @doc false
  def __handle_telemetry__(event, measurements, metadata, %{target: target})
      when is_pid(target) do
    tag = List.last(event)
    send(target, {:telemetry, tag, measurements, metadata})
    :ok
  end

  ## Internals ──────────────────────────────────────────────────────

  defp launch_task(classifier, opts) do
    cond do
      classifier not in @feature_vector_classifiers ->
        {:error, {:unknown_classifier, classifier}}

      true ->
        case load_training_data(classifier) do
          {:ok, training_data} ->
            run_opts =
              opts
              |> Keyword.put(:classifier, classifier)
              |> Keyword.put_new_lazy(:run_id, fn -> generate_run_id(classifier) end)
              |> Keyword.put_new(:verbose, false)

            run_id = Keyword.fetch!(run_opts, :run_id)

            # Errors inside `optimize/2` already emit a `:exception`
            # telemetry event before re-raising, so the Tracker's
            # bookkeeping is driven by telemetry rather than the
            # spawn_monitor rescue. The rescue exists only to keep the
            # error message on the logger when the GA crashes.
            {pid, ref} =
              spawn_monitor(fn ->
                try do
                  Brain.ML.WeightOptimizer.optimize(training_data, run_opts)
                rescue
                  e ->
                    Logger.warning(
                      "WeightOptimizer.Tracker: run #{run_id} crashed: #{Exception.message(e)}"
                    )

                    :ok
                end
              end)

            {:ok, run_id, pid, ref}

          {:error, reason} ->
            {:error, reason}
        end
    end
  end

  defp build_active_run(meta) do
    %{
      run_id: meta.run_id,
      classifier: classifier_to_string(meta.classifier),
      status: :running,
      started_at: now(),
      completed_at: nil,
      task_pid: nil,
      dim: meta[:dim],
      n_train: meta[:n_train],
      n_val: meta[:n_val],
      n_classes: meta[:n_classes],
      opts: scrub_opts(meta[:opts] || %{}),
      generation: 0,
      best_fitness: 0.0,
      gen_best_fitness: 0.0,
      raw_acc: 0.0,
      balanced_acc: 0.0,
      avg_fitness: 0.0,
      stale_count: 0,
      mutation_rate: 0.0,
      mutation_sigma: 0.0,
      history: []
    }
  end

  defp apply_generation(run, m, _metadata) do
    appended =
      run.history
      |> Kernel.++([{m.generation, m.gen_best_fitness}])
      |> clamp_history()

    %{
      run
      | generation: m.generation,
        best_fitness: m.best_fitness,
        gen_best_fitness: m.gen_best_fitness,
        raw_acc: m.raw_acc,
        balanced_acc: m.balanced_acc,
        avg_fitness: m.avg_fitness,
        stale_count: m.stale_count,
        mutation_rate: m.mutation_rate,
        mutation_sigma: m.mutation_sigma,
        history: appended
    }
  end

  defp summary(run) do
    Map.drop(run, [:task_pid])
  end

  defp generation_snapshot(run, m, metadata) do
    %{
      run_id: run.run_id,
      classifier: run.classifier,
      generation: m.generation,
      best_fitness: m.best_fitness,
      gen_best_fitness: m.gen_best_fitness,
      raw_acc: m.raw_acc,
      balanced_acc: m.balanced_acc,
      avg_fitness: m.avg_fitness,
      stale_count: m.stale_count,
      mutation_rate: m.mutation_rate,
      mutation_sigma: m.mutation_sigma,
      improved?: Map.get(metadata, :improved?, false),
      history: run.history
    }
  end

  defp clamp_history(history) when is_list(history) do
    case length(history) - @history_cap_per_run do
      drop when drop > 0 -> Enum.drop(history, drop)
      _ -> history
    end
  end

  defp clamp_history(_), do: []

  defp finalize_crashed_run(state, run_id, reason) do
    case Map.get(state.active, run_id) do
      nil ->
        state

      run ->
        crashed =
          run
          |> Map.put(:status, :error)
          |> Map.put(:completed_at, now())
          |> Map.put(:error, %{kind: :exit, reason: inspect(reason)})

        persist(crashed)
        broadcast({:run_failed, crashed})

        state
        |> remove_active(run_id)
        |> push_recent(crashed)
    end
  end

  defp remove_active(state, run_id) do
    %{state | active: Map.delete(state.active, run_id)}
  end

  defp push_recent(state, run) do
    new_recent =
      [run | state.recent]
      |> Enum.take(state.recent_limit)

    %{state | recent: new_recent}
  end

  defp broadcast(msg) do
    PubSub.broadcast(@pubsub, @topic, msg)
  end

  defp now, do: DateTime.utc_now()

  defp classifier_to_string(c) when is_binary(c), do: c
  defp classifier_to_string(c) when is_atom(c), do: Atom.to_string(c)
  defp classifier_to_string(other), do: inspect(other)

  defp scrub_opts(opts) when is_map(opts), do: opts

  defp scrub_opts(opts) when is_list(opts) do
    opts
    |> Keyword.drop([:seed, :run_id, :classifier])
    |> Map.new()
  end

  defp scrub_opts(_), do: %{}

  defp weights_summary(weights) when is_list(weights) do
    alive = Enum.count(weights, &(&1 > 0.01))
    total = length(weights)

    %{
      alive: alive,
      total: total,
      max: weights |> Enum.max(fn -> 0.0 end) |> Float.round(3),
      mean: weights |> mean() |> Float.round(3)
    }
  end

  defp weights_summary(_), do: nil

  defp mean([]), do: 0.0
  defp mean(values), do: Enum.sum(values) / length(values)

  defp generate_run_id(classifier) do
    ts = System.os_time(:millisecond)
    rand = :crypto.strong_rand_bytes(3) |> Base.url_encode64(padding: false)
    "ga-#{classifier}-#{ts}-#{rand}"
  end

  ## Training data loading ─────────────────────────────────────────

  defp load_training_data(classifier) do
    path = data_file_path(classifier)

    with {:ok, json} <- File.read(path),
         {:ok, entries} <- Jason.decode(json),
         {:ok, pairs} <- extract_feature_vector_pairs(entries) do
      {:ok, pairs}
    else
      {:error, %Jason.DecodeError{} = e} ->
        {:error, {:invalid_json, Exception.message(e)}}

      {:error, reason} when is_atom(reason) ->
        {:error, {:io_error, reason, path}}

      {:error, _} = err ->
        err
    end
  end

  defp extract_feature_vector_pairs(entries) when is_list(entries) do
    pairs =
      Enum.flat_map(entries, fn
        %{"feature_vector" => vec, "label" => label}
        when is_list(vec) and is_binary(label) and length(vec) > 0 ->
          [{vec, label}]

        _ ->
          []
      end)

    case pairs do
      [] -> {:error, :no_feature_vector_records}
      _ -> {:ok, pairs}
    end
  end

  defp extract_feature_vector_pairs(_), do: {:error, :invalid_training_payload}

  # Mirrors Mix.Tasks.TrainMicro.data_file_path/1 — the brain `priv/`
  # directory is a build-time symlink, so we have to walk back to the
  # umbrella root to reach `data/classifiers/`.
  defp data_file_path(name) do
    priv_dir = :code.priv_dir(:brain) |> to_string()

    umbrella_root =
      case File.read_link(priv_dir) do
        {:ok, link_target} ->
          parent = Path.dirname(priv_dir)
          real_priv = Path.join(parent, link_target) |> Path.expand()
          Path.join(real_priv, "../../..") |> Path.expand()

        {:error, _} ->
          Path.join(priv_dir, "../../../../..") |> Path.expand()
      end

    Path.join(umbrella_root, "data/classifiers/#{name}.json")
  end

  ## Persistence ───────────────────────────────────────────────────

  defp persist(run) do
    dir = runs_dir()
    File.mkdir_p!(dir)

    path = Path.join(dir, "#{run.run_id}.json")

    payload =
      run
      |> Map.drop([:task_pid])
      |> stringify_for_json()

    case Jason.encode(payload, pretty: true) do
      {:ok, json} ->
        File.write!(path, json)
        :ok

      {:error, reason} ->
        Logger.warning(
          "WeightOptimizer.Tracker: failed to encode run #{run.run_id} — #{inspect(reason)}"
        )

        :ok
    end
  rescue
    e ->
      Logger.warning(
        "WeightOptimizer.Tracker: persist failed for #{run.run_id}: #{Exception.message(e)}"
      )

      :ok
  end

  defp load_persisted_runs(limit) do
    dir = runs_dir()

    if File.dir?(dir) do
      dir
      |> File.ls!()
      |> Enum.filter(&String.ends_with?(&1, ".json"))
      |> Enum.sort(:desc)
      |> Enum.take(limit)
      |> Enum.flat_map(fn filename ->
        path = Path.join(dir, filename)

        with {:ok, content} <- File.read(path),
             {:ok, decoded} <- Jason.decode(content) do
          [restore_run(decoded)]
        else
          _ -> []
        end
      end)
    else
      []
    end
  end

  defp restore_run(decoded) do
    decoded
    |> Map.new(fn {k, v} -> {String.to_atom(k), v} end)
    |> coerce_status()
    |> coerce_history()
    |> coerce_datetimes()
  end

  defp coerce_status(%{status: status} = run) when is_binary(status) do
    %{run | status: String.to_atom(status)}
  end

  defp coerce_status(run), do: run

  defp coerce_history(%{history: history} = run) when is_list(history) do
    coerced =
      Enum.map(history, fn
        [g, f] -> {g, f}
        {g, f} -> {g, f}
        other -> other
      end)

    %{run | history: coerced}
  end

  defp coerce_history(run), do: run

  defp coerce_datetimes(run) do
    run
    |> Map.update(:started_at, nil, &parse_datetime/1)
    |> Map.update(:completed_at, nil, &parse_datetime/1)
  end

  defp parse_datetime(nil), do: nil

  defp parse_datetime(%DateTime{} = dt), do: dt

  defp parse_datetime(value) when is_binary(value) do
    case DateTime.from_iso8601(value) do
      {:ok, dt, _} -> dt
      _ -> nil
    end
  end

  defp parse_datetime(_), do: nil

  defp stringify_for_json(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {to_string(k), encode_value(v)} end)
  end

  defp encode_value({a, b}), do: [a, b]
  defp encode_value(%DateTime{} = dt), do: DateTime.to_iso8601(dt)
  defp encode_value(value) when is_atom(value) and not is_boolean(value) and not is_nil(value),
    do: Atom.to_string(value)
  defp encode_value(map) when is_map(map), do: stringify_for_json(map)
  defp encode_value(list) when is_list(list), do: Enum.map(list, &encode_value/1)
  defp encode_value(value), do: value

  defp runs_dir do
    Path.join(Brain.priv_path("weight_optimizer"), "runs")
  end
end
