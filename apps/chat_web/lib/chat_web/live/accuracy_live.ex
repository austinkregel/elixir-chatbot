defmodule ChatWeb.AccuracyLive do
  @moduledoc "ML model accuracy dashboard for viewing evaluation results.\n\nProvides visibility into:\n- Per-task evaluation metrics (intent, NER, sentiment, speech act)\n- Per-class precision, recall, F1, and support\n- Accuracy trends over evaluation runs (inline SVG charts)\n"

  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.ML.EvaluationStore

  alias Brain.ML.WeightOptimizer

  @tasks ~w(intent ner sentiment speech_act)
  @task_labels %{
    "intent" => "Intent",
    "ner" => "NER",
    "sentiment" => "Sentiment",
    "speech_act" => "Speech Act"
  }

  @optimizer_default_form %{
    "classifier" => "intent_full",
    "population_size" => "100",
    "max_generations" => "200",
    "early_stop_generations" => "15",
    "mutation_rate" => "0.12",
    "mutation_sigma" => "0.25"
  }

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Phoenix.PubSub.subscribe(Brain.PubSub, "evaluation:complete")
      Phoenix.PubSub.subscribe(Brain.PubSub, WeightOptimizer.Tracker.topic())
    end

    {:ok, socket}
  end

  @impl true
  def handle_params(_params, _uri, socket) do
    socket =
      socket
      |> assign(:active_tab, "intent")
      |> assign(:sort_field, "label")
      |> assign(:sort_dir, :asc)
      |> assign(:import_intents, [])
      |> assign(:import_grouped, %{})
      |> assign(:import_selected, MapSet.new())
      |> assign(:import_limit, nil)
      |> assign(:import_loading, false)
      |> assign(:import_filter, "")
      |> assign(:gold_stats, %{})
      |> assign(:optimizer_form, @optimizer_default_form)
      |> assign(:optimizer_error, nil)
      |> load_all_data()
      |> load_optimizer_runs()

    {:noreply, socket}
  end

  @impl true
  def handle_event("switch_tab", %{"tab" => "import"}, socket) do
    {:noreply, push_navigate(socket, to: ~p"/training-studio?tab=browse&source=intent_gold")}
  end

  def handle_event("switch_tab", %{"tab" => tab}, socket) do
    socket = assign(socket, :active_tab, tab)

    {:noreply, socket}
  end

  def handle_event("sort_table", %{"field" => field}, socket) do
    {new_field, new_dir} =
      if socket.assigns.sort_field == field do
        {field, toggle_sort_dir(socket.assigns.sort_dir)}
      else
        {field, :asc}
      end

    {:noreply, socket |> assign(:sort_field, new_field) |> assign(:sort_dir, new_dir)}
  end

  def handle_event("run_evaluation", _params, socket) do
    task = socket.assigns.active_tab

    case Brain.ML.TrainingServer.start_training(:evaluate, task: task) do
      {:ok, _} ->
        {:noreply, put_flash(socket, :info, "Evaluation started for #{task}. Results will appear when complete.")}

      {:error, {:already_training, current}} ->
        {:noreply, put_flash(socket, :error, "Already running: #{current}. Please wait.")}

      {:error, reason} ->
        {:noreply, put_flash(socket, :error, "Failed to start evaluation: #{inspect(reason)}")}
    end
  end

  def handle_event("refresh_data", _params, socket) do
    {:noreply, socket |> load_all_data() |> load_optimizer_runs()}
  end

  def handle_event("update_optimizer_form", params, socket) do
    form = Map.merge(socket.assigns.optimizer_form, Map.take(params, Map.keys(@optimizer_default_form)))
    {:noreply, assign(socket, :optimizer_form, form)}
  end

  def handle_event("start_optimizer_run", params, socket) do
    form = Map.merge(socket.assigns.optimizer_form, Map.take(params, Map.keys(@optimizer_default_form)))

    with {:ok, classifier} <- pick_classifier(form),
         {:ok, opts} <- parse_optimizer_opts(form) do
      case WeightOptimizer.Tracker.start_run(classifier, opts) do
        {:ok, run_id} ->
          socket =
            socket
            |> assign(:optimizer_form, form)
            |> assign(:optimizer_error, nil)
            |> put_flash(:info, "Started GA run #{run_id} for #{classifier}")

          {:noreply, socket}

        {:error, reason} ->
          {:noreply,
           socket
           |> assign(:optimizer_form, form)
           |> assign(:optimizer_error, format_optimizer_error(reason))}
      end
    else
      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:optimizer_form, form)
         |> assign(:optimizer_error, reason)}
    end
  end

  def handle_event("cancel_optimizer_run", %{"run_id" => run_id}, socket) do
    case WeightOptimizer.Tracker.cancel_run(run_id) do
      :ok ->
        {:noreply, put_flash(socket, :info, "Cancelled run #{run_id}")}

      {:error, :not_found} ->
        {:noreply, put_flash(socket, :error, "Run #{run_id} is no longer active")}
    end
  end

  def handle_event("switch_world", %{"world_id" => _world_id}, socket) do
    {:noreply, socket}
  end

  def handle_event("refresh_worlds", _params, socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_info({:evaluation_complete, %{task: task}}, socket) do
    Logger.debug("AccuracyLive: Received evaluation complete for #{task}")
    socket = load_all_data(socket)

    socket =
      if socket.assigns.active_tab == task do
        put_flash(socket, :info, "#{@task_labels[task] || task} evaluation updated")
      else
        socket
      end

    {:noreply, socket}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_info({:run_started, run}, socket) do
    {:noreply, upsert_active_run(socket, run)}
  end

  def handle_info({:generation, run_id, snapshot}, socket) do
    {:noreply, apply_snapshot(socket, run_id, snapshot)}
  end

  def handle_info({:run_complete, run}, socket) do
    {:noreply, finalize_run(socket, run)}
  end

  def handle_info({:run_failed, run}, socket) do
    {:noreply, finalize_run(socket, run)}
  end

  def handle_info({:run_cancelled, run_id}, socket) do
    socket =
      case Map.get(socket.assigns.optimizer_active, run_id) do
        nil ->
          socket

        run ->
          load_optimizer_runs_after_cancel(socket, run_id, run)
      end

    {:noreply, socket}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 class="text-xl font-bold">Accuracy Dashboard</h1>
            <p class="text-sm text-base-content/60">
              ML model evaluation results and weight-optimizer runs
            </p>
          </div>
          <div class="flex items-center gap-2">
            <.btn variant={:outline} size={:sm} phx-click="refresh_data">
              <.icon name="hero-arrow-path" class="size-4" /> Refresh
            </.btn>
            <.btn variant={:primary} size={:sm} phx-click="run_evaluation">
              <.icon name="hero-play" class="size-4" /> Run Evaluation
            </.btn>
          </div>
        </div>
      </:page_header>

      <div class="p-4 sm:p-6 lg:p-8 space-y-6">
        <!-- Tab Navigation -->
        <div class="border-b border-base-300">
          <nav class="flex gap-1 -mb-px" role="tablist">
            <.tab_button
              :for={task <- @tasks}
              tab={task}
              label={@task_labels[task]}
              active={@active_tab == task}
            />
            <.tab_button
              tab="optimizer"
              label="Optimizer"
              active={@active_tab == "optimizer"}
            />
            <.link
              navigate={~p"/training-studio?tab=browse&source=intent_gold"}
              class="px-3 py-1.5 text-sm font-medium rounded-lg text-base-content/70 hover:bg-base-200 transition-colors"
            >
              Training Studio &rarr;
            </.link>
          </nav>
        </div>

        <!-- Tab Content -->
        <%= cond do %>
          <% @active_tab == "optimizer" -> %>
            <.optimizer_panel
              active_runs={@optimizer_active_list}
              recent_runs={@optimizer_recent}
              form={@optimizer_form}
              error={@optimizer_error}
              available_classifiers={@optimizer_classifiers}
            />
          <% true -> %>
            <.task_panel
              task={@active_tab}
              evaluation={@evaluations[@active_tab]}
              trend={@trends[@active_tab]}
              sort_field={@sort_field}
              sort_dir={@sort_dir}
            />
        <% end %>
      </div>
    </.app_shell>
    """
  end

  attr(:tab, :string, required: true)
  attr(:label, :string, required: true)
  attr(:active, :boolean, default: false)

  defp tab_button(assigns) do
    ~H"""
    <button
      phx-click="switch_tab"
      phx-value-tab={@tab}
      role="tab"
      aria-selected={to_string(@active)}
      class={[
        "px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap",
        if(@active,
          do: "border-primary text-primary",
          else: "border-transparent text-base-content/60 hover:text-base-content hover:border-base-300"
        )
      ]}
    >
      {@label}
    </button>
    """
  end

  attr(:task, :string, required: true)
  attr(:evaluation, :map, default: nil)
  attr(:trend, :list, default: [])
  attr(:sort_field, :string, default: "label")
  attr(:sort_dir, :atom, default: :asc)

  defp task_panel(assigns) do
    assigns = assign(assigns, :task_label, @task_labels[assigns.task] || assigns.task)

    ~H"""
    <%= if @evaluation do %>
      <!-- Summary Cards -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <.metric_card
          label="Accuracy"
          value={format_percent(@evaluation["accuracy"])}
          icon="hero-check-circle"
        />
        <.metric_card
          label="Macro F1"
          value={format_percent(@evaluation["macro_f1"])}
          icon="hero-chart-bar"
        />
        <.metric_card
          label="Weighted F1"
          value={format_percent(@evaluation["weighted_f1"])}
          icon="hero-chart-bar-square"
        />
        <.metric_card
          label="Total Examples"
          value={to_string(@evaluation["total_examples"] || 0)}
          icon="hero-document-text"
        />
      </div>

      <!-- Accuracy Trend Chart -->
      <%= if length(@trend) > 1 do %>
        <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
          <h3 class="text-sm font-semibold text-base-content/70 mb-3">Accuracy Trend</h3>
          <.trend_chart points={@trend} />
        </div>
      <% end %>

      <!-- Per-Class Metrics Table -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
        <div class="p-4 border-b border-base-300/50">
          <h3 class="text-sm font-semibold text-base-content/70">
            Per-Class Metrics ({@task_label})
          </h3>
        </div>
        <div class="overflow-x-auto">
          <table class="table table-sm w-full">
            <thead>
              <tr class="border-b border-base-300/50">
                <.sortable_th field="label" label="Label" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="precision" label="Precision" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="recall" label="Recall" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="f1" label="F1" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="support" label="Support" sort_field={@sort_field} sort_dir={@sort_dir} />
              </tr>
            </thead>
            <tbody>
              <tr
                :for={{label, metrics} <- sorted_per_class(@evaluation["per_class"], @sort_field, @sort_dir)}
                class="border-b border-base-300/30 hover:bg-base-200/30"
              >
                <td class="font-mono text-xs">{label}</td>
                <td class={metric_color_class(metrics["precision"])}>{format_percent(metrics["precision"])}</td>
                <td class={metric_color_class(metrics["recall"])}>{format_percent(metrics["recall"])}</td>
                <td class={metric_color_class(metrics["f1"])}>{format_percent(metrics["f1"])}</td>
                <td class="text-base-content/70">{metrics["support"]}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    <% else %>
      <.empty_state task={@task} task_label={@task_label} />
    <% end %>
    """
  end

  attr(:label, :string, required: true)
  attr(:value, :string, required: true)
  attr(:icon, :string, required: true)

  defp metric_card(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
      <div class="flex items-center gap-2 mb-1">
        <.icon name={@icon} class="size-4 text-primary/70" />
        <span class="text-xs text-base-content/60 font-medium">{@label}</span>
      </div>
      <div class="text-2xl font-bold">{@value}</div>
    </div>
    """
  end

  attr(:field, :string, required: true)
  attr(:label, :string, required: true)
  attr(:sort_field, :string, required: true)
  attr(:sort_dir, :atom, required: true)

  defp sortable_th(assigns) do
    ~H"""
    <th
      class="cursor-pointer select-none hover:bg-base-200/50 transition-colors text-xs font-semibold text-base-content/70 uppercase tracking-wider"
      phx-click="sort_table"
      phx-value-field={@field}
    >
      <div class="flex items-center gap-1">
        {@label}
        <%= if @sort_field == @field do %>
          <span class="text-primary">
            <%= if @sort_dir == :asc do %>
              <.icon name="hero-chevron-up-mini" class="size-3" />
            <% else %>
              <.icon name="hero-chevron-down-mini" class="size-3" />
            <% end %>
          </span>
        <% end %>
      </div>
    </th>
    """
  end

  attr(:task, :string, required: true)
  attr(:task_label, :string, required: true)

  defp empty_state(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-12 text-center">
      <div class="mx-auto w-12 h-12 rounded-full bg-base-200 flex items-center justify-center mb-4">
        <.icon name="hero-chart-bar" class="size-6 text-base-content/40" />
      </div>
      <h3 class="text-lg font-semibold text-base-content/70 mb-2">No evaluations yet</h3>
      <p class="text-sm text-base-content/50 mb-4">
        Run an evaluation to see accuracy metrics for {@task_label}.
      </p>
      <code class="text-xs bg-base-200 px-3 py-1.5 rounded-lg text-base-content/70">
        mix evaluate.{@task} --save
      </code>
    </div>
    """
  end

  attr(:active_runs, :list, default: [])
  attr(:recent_runs, :list, default: [])
  attr(:form, :map, required: true)
  attr(:error, :string, default: nil)
  attr(:available_classifiers, :list, default: [])

  defp optimizer_panel(assigns) do
    ~H"""
    <div class="space-y-6">
      <!-- Summary Cards -->
      <.optimizer_summary active={@active_runs} recent={@recent_runs} />

      <!-- Run Launcher -->
      <.optimizer_launcher form={@form} error={@error} available_classifiers={@available_classifiers} />

      <!-- Active Runs -->
      <%= if @active_runs != [] do %>
        <div class="space-y-3">
          <h3 class="text-sm font-semibold text-base-content/70 flex items-center gap-2">
            <span class="flex h-2 w-2 rounded-full bg-success animate-pulse"></span>
            Active runs ({length(@active_runs)})
          </h3>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <.active_run_card :for={run <- @active_runs} run={run} />
          </div>
        </div>
      <% end %>

      <!-- Recent Runs -->
      <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
        <div class="p-4 border-b border-base-300/50 flex items-center justify-between">
          <h3 class="text-sm font-semibold text-base-content/70">
            Recent runs
            <span class="text-base-content/40 font-normal ml-1">
              ({length(@recent_runs)} stored)
            </span>
          </h3>
        </div>
        <%= if @recent_runs == [] do %>
          <div class="p-12 text-center">
            <div class="mx-auto w-12 h-12 rounded-full bg-base-200 flex items-center justify-center mb-4">
              <.icon name="hero-cpu-chip" class="size-6 text-base-content/40" />
            </div>
            <h3 class="text-lg font-semibold text-base-content/70 mb-2">No optimizer runs yet</h3>
            <p class="text-sm text-base-content/50 mb-4">
              Kick off a run from the launcher above, or train a feature-vector micro-classifier from the CLI.
            </p>
            <code class="text-xs bg-base-200 px-3 py-1.5 rounded-lg text-base-content/70">
              mix train_micro --only intent_full
            </code>
          </div>
        <% else %>
          <div class="overflow-x-auto">
            <table class="table table-sm w-full">
              <thead>
                <tr class="border-b border-base-300/50">
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Status</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Classifier</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Best Fitness</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Best Gen</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Generations</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Alive Dims</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Population</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Mut Rate</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Mut Sigma</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Early Stop</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Started</th>
                  <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Duration</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  :for={{run, idx} <- Enum.with_index(@recent_runs)}
                  class={[
                    "border-b border-base-300/30 hover:bg-base-200/30",
                    if(idx == 0 and run[:status] == :complete, do: "bg-success/5", else: "")
                  ]}
                >
                  <td>
                    <.run_status_badge status={Map.get(run, :status, :complete)} />
                  </td>
                  <td class="font-mono text-xs">{Map.get(run, :classifier, "-")}</td>
                  <td class={metric_color_class(Map.get(run, :best_fitness))}>
                    {format_percent(Map.get(run, :best_fitness))}
                  </td>
                  <td class="text-base-content/70">{Map.get(run, :best_generation, "-")}</td>
                  <td class="text-base-content/70">{Map.get(run, :generations_run, "-")}</td>
                  <td class="text-base-content/70">
                    {format_alive_dims(Map.get(run, :alive_dims), Map.get(run, :total_dims))}
                  </td>
                  <td class="text-base-content/70 text-xs">
                    {run_opt(run, :population_size, "population_size")}
                  </td>
                  <td class="text-base-content/70 text-xs">
                    {run_opt(run, :mutation_rate, "mutation_rate")}
                  </td>
                  <td class="text-base-content/70 text-xs">
                    {run_opt(run, :mutation_sigma, "mutation_sigma")}
                  </td>
                  <td class="text-base-content/70 text-xs">
                    {run_opt(run, :early_stop_generations, "early_stop_generations")}
                  </td>
                  <td class="text-xs text-base-content/60">
                    {format_time_ago(Map.get(run, :started_at))}
                  </td>
                  <td class="text-xs text-base-content/60">
                    {format_duration(Map.get(run, :duration_ms))}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        <% end %>
      </div>
    </div>
    """
  end

  attr(:active, :list, required: true)
  attr(:recent, :list, required: true)

  defp optimizer_summary(assigns) do
    completed = Enum.filter(assigns.recent, &(&1[:status] == :complete or &1[:status] == :early_stop))

    best_run =
      case completed do
        [] -> nil
        runs -> Enum.max_by(runs, &(Map.get(&1, :best_fitness) || 0.0))
      end

    last_completed =
      case completed do
        [] -> nil
        [first | _] -> first
      end

    assigns =
      assigns
      |> assign(:best_run, best_run)
      |> assign(:last_completed, last_completed)

    ~H"""
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <.metric_card
        label="Active runs"
        value={to_string(length(@active))}
        icon="hero-bolt"
      />
      <.metric_card
        label="Stored runs"
        value={to_string(length(@recent))}
        icon="hero-archive-box"
      />
      <.metric_card
        label="Best fitness"
        value={format_percent(@best_run && Map.get(@best_run, :best_fitness))}
        icon="hero-trophy"
      />
      <.metric_card
        label="Last completed"
        value={format_time_ago(@last_completed && Map.get(@last_completed, :completed_at))}
        icon="hero-clock"
      />
    </div>
    """
  end

  attr(:form, :map, required: true)
  attr(:error, :string, default: nil)
  attr(:available_classifiers, :list, default: [])

  defp optimizer_launcher(assigns) do
    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-4">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-semibold text-base-content/70">
          Launch a new GA run
        </h3>
        <span class="text-xs text-base-content/40">
          Genetic algorithm — per-dimension feature weights
        </span>
      </div>
      <form phx-change="update_optimizer_form" phx-submit="start_optimizer_run" class="space-y-3">
        <div class="grid grid-cols-2 lg:grid-cols-3 gap-3">
          <div class="form-control">
            <label class="label py-1"><span class="label-text text-xs">Classifier</span></label>
            <select name="classifier" class="select select-sm select-bordered">
              <option :for={c <- @available_classifiers} value={c} selected={@form["classifier"] == c}>
                {c}
              </option>
            </select>
          </div>
          <.optimizer_input form={@form} field="population_size" label="Population size" min="10" step="10" />
          <.optimizer_input form={@form} field="max_generations" label="Max generations" min="10" step="10" />
          <.optimizer_input form={@form} field="early_stop_generations" label="Early stop (gens)" min="1" step="1" />
          <.optimizer_input form={@form} field="mutation_rate" label="Mutation rate" min="0" max="1" step="0.01" />
          <.optimizer_input form={@form} field="mutation_sigma" label="Mutation sigma" min="0" max="3" step="0.05" />
        </div>
        <%= if @error do %>
          <div class="text-error text-xs mt-1">{@error}</div>
        <% end %>
        <div class="flex justify-end">
          <button type="submit" class="btn btn-primary btn-sm">
            <.icon name="hero-rocket-launch" class="size-4" /> Start run
          </button>
        </div>
      </form>
    </div>
    """
  end

  attr(:form, :map, required: true)
  attr(:field, :string, required: true)
  attr(:label, :string, required: true)
  attr(:min, :string, default: nil)
  attr(:max, :string, default: nil)
  attr(:step, :string, default: nil)

  defp optimizer_input(assigns) do
    ~H"""
    <div class="form-control">
      <label class="label py-1"><span class="label-text text-xs">{@label}</span></label>
      <input
        type="number"
        name={@field}
        value={@form[@field]}
        min={@min}
        max={@max}
        step={@step}
        class="input input-sm input-bordered"
      />
    </div>
    """
  end

  attr(:run, :map, required: true)

  defp active_run_card(assigns) do
    history = Map.get(assigns.run, :history) || []
    max_gen = get_in(assigns.run, [:opts, :max_generations]) || get_in(assigns.run, [:opts, "max_generations"]) || 200
    progress_pct = min(round((Map.get(assigns.run, :generation, 0) + 1) / max(max_gen, 1) * 100), 100)

    assigns =
      assigns
      |> assign(:history, history)
      |> assign(:max_gen, max_gen)
      |> assign(:progress_pct, progress_pct)

    ~H"""
    <div class="bg-base-100 rounded-xl border border-base-300/50 p-4 space-y-3">
      <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
          <div class="flex items-center gap-2">
            <span class="font-mono text-sm font-semibold truncate">{Map.get(@run, :classifier, "-")}</span>
            <.run_status_badge status={Map.get(@run, :status, :running)} />
          </div>
          <div class="text-xs text-base-content/50 font-mono truncate">{Map.get(@run, :run_id)}</div>
        </div>
        <button
          phx-click="cancel_optimizer_run"
          phx-value-run_id={Map.get(@run, :run_id)}
          class="btn btn-ghost btn-xs"
          title="Cancel run"
        >
          <.icon name="hero-x-mark" class="size-4" /> Cancel
        </button>
      </div>

      <!-- Progress bar -->
      <div class="space-y-1">
        <div class="flex justify-between text-xs text-base-content/60">
          <span>Generation {Map.get(@run, :generation, 0)} / {@max_gen}</span>
          <span>{@progress_pct}%</span>
        </div>
        <progress class="progress progress-primary w-full" value={@progress_pct} max="100"></progress>
      </div>

      <!-- Metrics grid -->
      <div class="grid grid-cols-3 gap-2 text-xs">
        <.kv_block label="Best" value={format_percent(Map.get(@run, :best_fitness))} highlight={true} />
        <.kv_block label="Raw" value={format_percent(Map.get(@run, :raw_acc))} />
        <.kv_block label="Balanced" value={format_percent(Map.get(@run, :balanced_acc))} />
        <.kv_block label="Avg" value={format_percent(Map.get(@run, :avg_fitness))} />
        <.kv_block label="Stale" value={to_string(Map.get(@run, :stale_count, 0))} />
        <.kv_block label="Mut" value={format_mutation(Map.get(@run, :mutation_rate), Map.get(@run, :mutation_sigma))} />
      </div>
      <!-- Config grid -->
      <div class="grid grid-cols-4 gap-2 text-xs">
        <.kv_block label="Population" value={run_opt(@run, :population_size, "population_size")} />
        <.kv_block label="Mut Rate" value={run_opt(@run, :mutation_rate, "mutation_rate")} />
        <.kv_block label="Mut Sigma" value={run_opt(@run, :mutation_sigma, "mutation_sigma")} />
        <.kv_block label="Early Stop" value={run_opt(@run, :early_stop_generations, "early_stop_generations")} />
      </div>

      <!-- Sparkline -->
      <%= if length(@history) > 1 do %>
        <.sparkline history={@history} />
      <% end %>
    </div>
    """
  end

  attr(:label, :string, required: true)
  attr(:value, :string, required: true)
  attr(:highlight, :boolean, default: false)

  defp kv_block(assigns) do
    ~H"""
    <div class={[
      "rounded-lg px-2 py-1.5",
      if(@highlight, do: "bg-primary/10", else: "bg-base-200/40")
    ]}>
      <div class="text-[10px] uppercase tracking-wider text-base-content/50">{@label}</div>
      <div class={[
        "font-mono text-sm",
        if(@highlight, do: "font-semibold text-primary", else: "")
      ]}>{@value}</div>
    </div>
    """
  end

  attr(:status, :atom, required: true)

  defp run_status_badge(assigns) do
    {variant, label} =
      case assigns.status do
        :running -> {:info, "running"}
        :complete -> {:success, "complete"}
        :early_stop -> {:success, "early stop"}
        :cancelled -> {:warning, "cancelled"}
        :error -> {:error, "error"}
        other -> {:ghost, to_string(other)}
      end

    assigns = assigns |> assign(:variant, variant) |> assign(:label, label)

    ~H"""
    <.badge variant={@variant} size={:xs}>{@label}</.badge>
    """
  end

  attr(:history, :list, required: true)

  defp sparkline(assigns) do
    points = Enum.map(assigns.history, fn {g, f} -> {g, f} end)

    {gens, fits} = Enum.unzip(points)
    min_g = Enum.min(gens)
    max_g = Enum.max(gens)
    g_range = max(max_g - min_g, 1)

    min_f = Enum.min(fits)
    max_f = Enum.max(fits)
    f_range = max(max_f - min_f, 0.001)

    width = 280
    height = 48

    polyline =
      points
      |> Enum.map_join(" ", fn {g, f} ->
        x = (g - min_g) / g_range * width
        y = height - (f - min_f) / f_range * height
        "#{Float.round(x * 1.0, 1)},#{Float.round(y * 1.0, 1)}"
      end)

    assigns =
      assigns
      |> assign(:polyline, polyline)
      |> assign(:width, width)
      |> assign(:height, height)
      |> assign(:max_f, max_f)
      |> assign(:min_f, min_f)

    ~H"""
    <div class="flex items-center gap-2">
      <svg viewBox={"0 0 #{@width} #{@height}"} class="w-full h-12" preserveAspectRatio="none" role="img" aria-label="Fitness sparkline">
        <polyline
          points={@polyline}
          fill="none"
          stroke="oklch(var(--p))"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
      </svg>
      <div class="text-[10px] text-base-content/50 font-mono whitespace-nowrap">
        {format_percent(@min_f)} → {format_percent(@max_f)}
      </div>
    </div>
    """
  end

  attr(:points, :list, required: true)

  defp trend_chart(assigns) do
    points = assigns.points
    count = length(points)

    if count < 2 do
      assigns = assign(assigns, :message, "Not enough data points for a chart.")

      ~H"""
      <div class="text-sm text-base-content/50 text-center py-4">{@message}</div>
      """
    else
      width = 600
      height = 200
      padding_x = 50
      padding_y = 20
      chart_width = width - padding_x * 2
      chart_height = height - padding_y * 2

      values = Enum.map(points, fn p -> (p.value || 0) * 100 end)
      min_val = max(Enum.min(values) - 5, 0)
      max_val = min(Enum.max(values) + 5, 100)
      val_range = max(max_val - min_val, 1)

      polyline_points =
        values
        |> Enum.with_index()
        |> Enum.map_join(
          " ",
          fn {val, i} ->
            x = padding_x + i / max(count - 1, 1) * chart_width
            y = padding_y + (1 - (val - min_val) / val_range) * chart_height
            "#{Float.round(x * 1.0, 1)},#{Float.round(y * 1.0, 1)}"
          end
        )

      dots =
        values
        |> Enum.with_index()
        |> Enum.map(fn {val, i} ->
          x = padding_x + i / max(count - 1, 1) * chart_width
          y = padding_y + (1 - (val - min_val) / val_range) * chart_height
          %{x: Float.round(x * 1.0, 1), y: Float.round(y * 1.0, 1), val: Float.round(val, 1)}
        end)

      y_ticks =
        for i <- 0..4 do
          val = min_val + i / 4 * val_range
          y = padding_y + (1 - i / 4) * chart_height
          %{val: Float.round(val, 0), y: Float.round(y * 1.0, 1)}
        end

      assigns =
        assigns
        |> assign(:width, width)
        |> assign(:height, height)
        |> assign(:padding_x, padding_x)
        |> assign(:padding_y, padding_y)
        |> assign(:chart_width, chart_width)
        |> assign(:chart_height, chart_height)
        |> assign(:polyline_points, polyline_points)
        |> assign(:dots, dots)
        |> assign(:y_ticks, y_ticks)

      ~H"""
      <svg viewBox={"0 0 #{@width} #{@height}"} class="w-full h-auto max-h-48" role="img" aria-label="Accuracy trend chart">
        <!-- Grid lines -->
        <line
          :for={tick <- @y_ticks}
          x1={@padding_x}
          y1={tick.y}
          x2={@padding_x + @chart_width}
          y2={tick.y}
          stroke="currentColor"
          stroke-opacity="0.1"
          stroke-dasharray="4,4"
        />
        <!-- Y-axis labels -->
        <text
          :for={tick <- @y_ticks}
          x={@padding_x - 8}
          y={tick.y + 4}
          text-anchor="end"
          class="fill-base-content/40"
          font-size="10"
        >
          {trunc(tick.val)}%
        </text>
        <!-- Trend line -->
        <polyline
          points={@polyline_points}
          fill="none"
          stroke="oklch(var(--p))"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
        <!-- Data points -->
        <circle
          :for={dot <- @dots}
          cx={dot.x}
          cy={dot.y}
          r="4"
          fill="oklch(var(--p))"
          stroke="oklch(var(--b1))"
          stroke-width="2"
        />
        <!-- Value labels on dots -->
        <text
          :for={dot <- @dots}
          x={dot.x}
          y={dot.y - 10}
          text-anchor="middle"
          class="fill-base-content/60"
          font-size="9"
        >
          {dot.val}%
        </text>
      </svg>
      """
    end
  end

  defp load_all_data(socket) do
    evaluations = load_evaluations()
    trends = load_trends()

    socket
    |> assign(:tasks, @tasks)
    |> assign(:task_labels, @task_labels)
    |> assign(:evaluations, evaluations)
    |> assign(:trends, trends)
  end

  # Optimizer state ----------------------------------------------------

  defp load_optimizer_runs(socket) do
    {active, recent, classifiers} = fetch_optimizer_state()

    socket
    |> assign(:optimizer_active, Map.new(active, &{&1.run_id, &1}))
    |> assign(:optimizer_active_list, active)
    |> assign(:optimizer_recent, recent)
    |> assign(:optimizer_classifiers, classifiers)
  end

  defp fetch_optimizer_state do
    active =
      try do
        WeightOptimizer.Tracker.list_active()
      catch
        :exit, _ -> []
      end

    recent =
      try do
        WeightOptimizer.Tracker.list_recent()
      catch
        :exit, _ -> []
      end

    classifiers =
      try do
        WeightOptimizer.Tracker.feature_vector_classifiers()
      catch
        :exit, _ -> []
      end

    {active, recent, classifiers}
  end

  defp upsert_active_run(socket, run) do
    active = Map.put(socket.assigns.optimizer_active, run.run_id, run)
    list = active |> Map.values() |> Enum.sort_by(& &1.started_at, {:desc, DateTime})

    socket
    |> assign(:optimizer_active, active)
    |> assign(:optimizer_active_list, list)
  end

  defp apply_snapshot(socket, run_id, snapshot) do
    case Map.get(socket.assigns.optimizer_active, run_id) do
      nil ->
        # Snapshot for a run we haven't seen yet — pull tracker state to catch up.
        load_optimizer_runs(socket)

      run ->
        merged = Map.merge(run, Map.delete(snapshot, :run_id))
        upsert_active_run(socket, merged)
    end
  end

  defp finalize_run(socket, run) do
    active = Map.delete(socket.assigns.optimizer_active, run.run_id)
    list = active |> Map.values() |> Enum.sort_by(& &1.started_at, {:desc, DateTime})

    recent =
      [run | Enum.reject(socket.assigns.optimizer_recent, &(&1.run_id == run.run_id))]
      |> Enum.take(50)

    socket
    |> assign(:optimizer_active, active)
    |> assign(:optimizer_active_list, list)
    |> assign(:optimizer_recent, recent)
  end

  defp load_optimizer_runs_after_cancel(socket, run_id, run) do
    cancelled = Map.put(run, :status, :cancelled)
    finalize_run(socket, cancelled |> Map.put(:run_id, run_id))
  end

  # Optimizer form parsing --------------------------------------------

  defp pick_classifier(form) do
    classifier = String.trim(form["classifier"] || "")

    cond do
      classifier == "" ->
        {:error, "Pick a classifier."}

      classifier in WeightOptimizer.Tracker.feature_vector_classifiers() ->
        {:ok, classifier}

      true ->
        {:error, "Unknown classifier: #{classifier}"}
    end
  end

  defp parse_optimizer_opts(form) do
    with {:ok, pop} <- parse_pos_int(form["population_size"], "Population size"),
         {:ok, max_gen} <- parse_pos_int(form["max_generations"], "Max generations"),
         {:ok, early_stop} <- parse_pos_int(form["early_stop_generations"], "Early stop generations"),
         {:ok, mut_rate} <- parse_unit_float(form["mutation_rate"], "Mutation rate"),
         {:ok, mut_sigma} <- parse_pos_float(form["mutation_sigma"], "Mutation sigma") do
      {:ok,
       [
         population_size: pop,
         max_generations: max_gen,
         early_stop_generations: early_stop,
         mutation_rate: mut_rate,
         mutation_sigma: mut_sigma
       ]}
    end
  end

  defp parse_pos_int(nil, label), do: {:error, "#{label} is required."}

  defp parse_pos_int(str, label) do
    case Integer.parse(String.trim(str)) do
      {n, ""} when n > 0 -> {:ok, n}
      _ -> {:error, "#{label} must be a positive integer."}
    end
  end

  defp parse_unit_float(nil, label), do: {:error, "#{label} is required."}

  defp parse_unit_float(str, label) do
    case Float.parse(String.trim(str)) do
      {f, ""} when f >= 0.0 and f <= 1.0 -> {:ok, f}
      _ -> {:error, "#{label} must be a number between 0 and 1."}
    end
  end

  defp parse_pos_float(nil, label), do: {:error, "#{label} is required."}

  defp parse_pos_float(str, label) do
    case Float.parse(String.trim(str)) do
      {f, ""} when f > 0.0 -> {:ok, f}
      _ -> {:error, "#{label} must be a positive number."}
    end
  end

  defp format_optimizer_error({:unknown_classifier, name}),
    do: "Unknown classifier: #{name}"

  defp format_optimizer_error({:io_error, reason, path}),
    do: "Could not read training data (#{reason}) at #{path}"

  defp format_optimizer_error({:invalid_json, msg}),
    do: "Training data isn't valid JSON: #{msg}"

  defp format_optimizer_error(:no_feature_vector_records),
    do: "Training file has no feature_vector records. Run `mix gen_micro_data`."

  defp format_optimizer_error(other), do: inspect(other)

  defp load_evaluations do
    Map.new(@tasks, fn task ->
      result =
        try do
          EvaluationStore.latest(task)
        rescue
          _ -> nil
        end

      {task, result}
    end)
  end

  defp load_trends do
    Map.new(@tasks, fn task ->
      trend =
        try do
          EvaluationStore.trend(task, :accuracy)
        rescue
          _ -> []
        end

      {task, trend}
    end)
  end

  defp format_percent(nil) do
    "-"
  end

  defp format_percent(val) when is_float(val) and val <= 1.0 do
    "#{Float.round(val * 100, 1)}%"
  end

  defp format_percent(val) when is_float(val) do
    "#{Float.round(val, 1)}%"
  end

  defp format_percent(val) when is_integer(val) do
    "#{val}%"
  end

  defp format_percent(_) do
    "-"
  end

  defp run_opt(run, atom_key, string_key) do
    val =
      get_in(run, [:opts, atom_key]) ||
        get_in(run, [:opts, string_key]) ||
        Map.get(run, atom_key) ||
        Map.get(run, string_key)

    case val do
      nil -> "-"
      f when is_float(f) -> Float.round(f, 3) |> to_string()
      other -> to_string(other)
    end
  end

  defp format_alive_dims(nil, _), do: "-"
  defp format_alive_dims(alive, nil), do: to_string(alive)
  defp format_alive_dims(alive, total), do: "#{alive}/#{total}"

  defp format_duration(nil), do: "-"

  defp format_duration(ms) when is_integer(ms) do
    cond do
      ms < 1_000 -> "#{ms} ms"
      ms < 60_000 -> "#{Float.round(ms / 1000, 1)} s"
      ms < 3_600_000 -> "#{Float.round(ms / 60_000, 1)} min"
      true -> "#{Float.round(ms / 3_600_000, 2)} h"
    end
  end

  defp format_duration(_), do: "-"

  defp format_time_ago(nil), do: "-"

  defp format_time_ago(%DateTime{} = dt) do
    seconds = DateTime.diff(DateTime.utc_now(), dt, :second)

    cond do
      seconds < 5 -> "just now"
      seconds < 60 -> "#{seconds}s ago"
      seconds < 3600 -> "#{div(seconds, 60)}m ago"
      seconds < 86_400 -> "#{div(seconds, 3600)}h ago"
      true -> "#{div(seconds, 86_400)}d ago"
    end
  end

  defp format_time_ago(_), do: "-"

  defp format_mutation(nil, _), do: "-"
  defp format_mutation(_, nil), do: "-"

  defp format_mutation(rate, sigma) when is_number(rate) and is_number(sigma) do
    "#{Float.round(rate * 1.0, 3)}/#{Float.round(sigma * 1.0, 3)}"
  end

  defp format_mutation(_, _), do: "-"

  defp metric_color_class(nil) do
    "text-base-content/50"
  end

  defp metric_color_class(val) when is_number(val) do
    cond do
      val >= 0.9 -> "text-success font-medium"
      val >= 0.7 -> "text-warning font-medium"
      true -> "text-error font-medium"
    end
  end

  defp metric_color_class(_) do
    "text-base-content/50"
  end

  defp sorted_per_class(nil, _field, _dir) do
    []
  end

  defp sorted_per_class(per_class, sort_field, sort_dir) when is_map(per_class) do
    per_class
    |> Enum.to_list()
    |> Enum.sort_by(
      fn {label, metrics} ->
        case sort_field do
          "label" -> label
          "precision" -> metrics["precision"] || 0
          "recall" -> metrics["recall"] || 0
          "f1" -> metrics["f1"] || 0
          "support" -> metrics["support"] || 0
          _ -> label
        end
      end,
      sort_dir
    )
  end

  defp toggle_sort_dir(:asc) do
    :desc
  end

  defp toggle_sort_dir(:desc) do
    :asc
  end

end
