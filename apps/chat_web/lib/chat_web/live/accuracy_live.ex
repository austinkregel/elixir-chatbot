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

  @run_evaluation_confirm_id "confirm-run-evaluation"
  @optimizer_confirm_id "confirm-start-optimizer-run"

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
      |> assign(:open_confirm, nil)
      |> assign(:confirm_error, nil)
      |> assign(:run_evaluation_confirm_id, @run_evaluation_confirm_id)
      |> assign(:optimizer_confirm_id, @optimizer_confirm_id)
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

  def handle_event("open_confirm", %{"id" => id}, socket) do
    {:noreply, socket |> assign(:open_confirm, id) |> assign(:confirm_error, nil)}
  end

  def handle_event("close_confirm", _params, socket) do
    {:noreply, close_confirm(socket)}
  end

  def handle_event("run_evaluation", _params, socket) do
    task = socket.assigns.active_tab

    case Brain.ML.TrainingServer.start_training(:evaluate, task: task) do
      {:ok, _} ->
        {:noreply,
         socket
         |> close_confirm()
         |> put_flash(:info, "Evaluation started for #{task}. Results will appear when complete.")}

      {:error, {:already_training, current}} ->
        {:noreply, assign(socket, :confirm_error, "Already running: #{current}. Wait for it to finish.")}

      {:error, reason} ->
        {:noreply, assign(socket, :confirm_error, "Failed to start evaluation: #{inspect(reason)}")}
    end
  end

  def handle_event("refresh_data", _params, socket) do
    {:noreply, socket |> load_all_data() |> load_optimizer_runs()}
  end

  def handle_event("update_optimizer_form", params, socket) do
    form = Map.merge(socket.assigns.optimizer_form, Map.take(params, Map.keys(@optimizer_default_form)))
    {:noreply, assign(socket, :optimizer_form, form)}
  end

  def handle_event("request_optimizer_run", params, socket) do
    form = Map.merge(socket.assigns.optimizer_form, Map.take(params, Map.keys(@optimizer_default_form)))

    with {:ok, _classifier} <- pick_classifier(form),
         {:ok, _opts} <- parse_optimizer_opts(form) do
      {:noreply,
       socket
       |> assign(:optimizer_form, form)
       |> assign(:optimizer_error, nil)
       |> assign(:open_confirm, @optimizer_confirm_id)
       |> assign(:confirm_error, nil)}
    else
      {:error, reason} ->
        {:noreply,
         socket
         |> assign(:optimizer_form, form)
         |> assign(:optimizer_error, reason)}
    end
  end

  def handle_event("start_optimizer_run", _params, socket) do
    form = socket.assigns.optimizer_form

    with {:ok, classifier} <- pick_classifier(form),
         {:ok, opts} <- parse_optimizer_opts(form) do
      case WeightOptimizer.Tracker.start_run(classifier, opts) do
        {:ok, run_id} ->
          {:noreply,
           socket
           |> close_confirm()
           |> put_flash(:info, "Started GA run #{run_id} for #{classifier}")}

        {:error, reason} ->
          {:noreply, assign(socket, :confirm_error, format_optimizer_error(reason))}
      end
    else
      {:error, reason} ->
        {:noreply, assign(socket, :confirm_error, reason)}
    end
  end

  def handle_event("cancel_optimizer_run", %{"run_id" => run_id}, socket) do
    case WeightOptimizer.Tracker.cancel_run(run_id) do
      :ok ->
        {:noreply, socket |> close_confirm() |> put_flash(:info, "Cancelled run #{run_id}")}

      {:error, :not_found} ->
        {:noreply, assign(socket, :confirm_error, "Run #{run_id} is no longer active.")}
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
        <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-space-lg">
          <div>
            <h1 class="text-title text-ink">Accuracy Dashboard</h1>
            <p class="text-body text-ink-muted">
              ML model evaluation results and weight-optimizer runs
            </p>
          </div>
          <div class="flex flex-col items-end gap-space-sm">
            <div class="flex flex-wrap items-center gap-space-sm">
              <.btn variant={:outline} size={:sm} phx-click="refresh_data">
                <.icon name="hero-arrow-path" class="size-4" /> Refresh
              </.btn>
              <.btn
                id="run-evaluation"
                variant={:primary}
                size={:sm}
                reach={:shared}
                target={"#{task_label(@active_tab)} evaluation"}
                icon="hero-play"
                phx-click="open_confirm"
                phx-value-id={@run_evaluation_confirm_id}
              >
                Run Evaluation
              </.btn>
            </div>
            <.execute_confirm
              id={@run_evaluation_confirm_id}
              open={@open_confirm == @run_evaluation_confirm_id}
              reach={:shared}
              verb="Run evaluation"
              target={"#{task_label(@active_tab)} evaluation"}
              consequence={"Runs the #{task_label(@active_tab)} evaluation on this node and saves its result to Atlas and to this node's evaluation results files. The dashboard and this page show the latest saved result."}
              on_confirm="run_evaluation"
              on_cancel="close_confirm"
              trigger_id="run-evaluation"
              error={@confirm_error}
            />
          </div>
        </div>
      </:page_header>

      <div class="p-space-lg space-y-space-xl">
        <!-- Tab Navigation -->
        <nav class="flex flex-wrap items-center gap-space-md">
          <.tabs>
            <.tab
              :for={task <- @tasks}
              phx-click="switch_tab"
              phx-value-tab={task}
              active={@active_tab == task}
            >
              {@task_labels[task]}
            </.tab>
            <.tab
              phx-click="switch_tab"
              phx-value-tab="optimizer"
              active={@active_tab == "optimizer"}
            >
              Optimizer
            </.tab>
          </.tabs>
          <.link
            navigate={~p"/training-studio?tab=browse&source=intent_gold"}
            class="text-body-dense text-accent hover:underline"
          >
            Training Studio &rarr;
          </.link>
        </nav>

        <!-- Tab Content -->
        <%= cond do %>
          <% @active_tab == "optimizer" -> %>
            <.optimizer_panel
              active_runs={@optimizer_active_list}
              recent_runs={@optimizer_recent}
              form={@optimizer_form}
              error={@optimizer_error}
              available_classifiers={@optimizer_classifiers}
              confirm_id={@optimizer_confirm_id}
              open_confirm={@open_confirm}
              confirm_error={@confirm_error}
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

  attr(:task, :string, required: true)
  attr(:evaluation, :any,
    default: nil,
    doc: "the latest result, nil when none is saved, or {:could_not_ask, detail}"
  )
  attr(:trend, :list, default: [])
  attr(:sort_field, :string, default: "label")
  attr(:sort_dir, :atom, default: :asc)

  defp task_panel(assigns) do
    assigns = assign(assigns, :task_label, task_label(assigns.task))

    ~H"""
    <%= case @evaluation do %>
      <% {:could_not_ask, detail} -> %>
        <.empty_panel kind={:could_not_ask}>{detail}</.empty_panel>
      <% nil -> %>
        <.card>
          <.empty_message icon="hero-chart-bar" title="No evaluations yet" command={"mix evaluate.#{@task} --save"}>
            Run an evaluation to see accuracy metrics for {@task_label}.
          </.empty_message>
        </.card>
      <% %{} -> %>
      <!-- Summary Cards -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-space-lg">
        <.stat_kpi
          label="Accuracy"
          value={metric_value(@evaluation["accuracy"])}
          sublabel={metric_sublabel("accuracy", @evaluation["accuracy"], @evaluation["total_examples"])}
          icon="hero-check-circle"
        />
        <.stat_kpi
          label="Macro F1"
          value={metric_value(@evaluation["macro_f1"])}
          sublabel={metric_sublabel("macro-F1", @evaluation["macro_f1"], @evaluation["total_examples"])}
          icon="hero-chart-bar"
        >
          <:verdict><.macro_f1_gate task={@task} /></:verdict>
        </.stat_kpi>
        <.stat_kpi
          label="Weighted F1"
          value={metric_value(@evaluation["weighted_f1"])}
          sublabel={metric_sublabel("weighted F1", @evaluation["weighted_f1"], @evaluation["total_examples"])}
          icon="hero-chart-bar-square"
        />
        <.stat_kpi
          label="Total Examples"
          value={count_value(@evaluation["total_examples"])}
          sublabel={if is_nil(@evaluation["total_examples"]), do: "no example count in this result"}
          icon="hero-document-text"
        />
      </div>

      <!-- Accuracy Trend Chart -->
      <%= if length(@trend) > 1 do %>
        <.card>
          <.card_body>
            <h3 class="text-heading text-ink mb-space-md">Accuracy Trend</h3>
            <.trend_chart points={@trend} />
          </.card_body>
        </.card>
      <% end %>

      <!-- Per-Class Metrics Table -->
      <.card class="overflow-hidden">
        <div class="p-space-lg border-b border-border">
          <h3 class="text-heading text-ink">
            Per-Class Metrics ({@task_label})
          </h3>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-body-dense text-ink tabular-nums">
            <thead class="bg-surface-sunk">
              <tr>
                <.sortable_th field="label" label="Label" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="precision" label="Precision" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="recall" label="Recall" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="f1" label="F1" sort_field={@sort_field} sort_dir={@sort_dir} />
                <.sortable_th field="support" label="Support" sort_field={@sort_field} sort_dir={@sort_dir} />
              </tr>
            </thead>
            <tbody class="divide-y divide-border">
              <tr
                :for={{label, metrics} <- sorted_per_class(@evaluation["per_class"], @sort_field, @sort_dir)}
                class="even:bg-surface-sunk"
              >
                <td class="h-row-compact px-space-sm text-value text-ink">{label}</td>
                <td class={["h-row-compact px-space-sm", metric_class(:rate, metrics["precision"])]}>
                  {metric_value(metrics["precision"])}
                </td>
                <td class={["h-row-compact px-space-sm", metric_class(:rate, metrics["recall"])]}>
                  {metric_value(metrics["recall"])}
                </td>
                <td class={["h-row-compact px-space-sm", metric_class(:rate, metrics["f1"])]}>
                  {metric_value(metrics["f1"])}
                </td>
                <td class={["h-row-compact px-space-sm", metric_class(:count, metrics["support"])]}>
                  {count_value(metrics["support"])}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </.card>
    <% end %>
    """
  end

  attr(:field, :string, required: true)
  attr(:label, :string, required: true)
  attr(:sort_field, :string, required: true)
  attr(:sort_dir, :atom, required: true)

  defp sortable_th(assigns) do
    ~H"""
    <th
      class="h-row-compact px-space-sm cursor-pointer select-none transition-colors text-label text-ink-muted hover:text-ink"
      phx-click="sort_table"
      phx-value-field={@field}
    >
      <div class="flex items-center gap-space-xs">
        {@label}
        <%= if @sort_field == @field do %>
          <span class="text-accent">
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

  attr(:icon, :string, required: true)
  attr(:title, :string, required: true)
  attr(:command, :string, required: true)
  slot(:inner_block, required: true)

  defp empty_message(assigns) do
    ~H"""
    <div class="p-space-3xl text-center">
      <div class="mx-auto mb-space-lg flex size-12 items-center justify-center rounded-md bg-surface-sunk">
        <.icon name={@icon} class="size-6 text-ink-muted" />
      </div>
      <h3 class="mb-space-sm text-heading text-ink">{@title}</h3>
      <p class="mb-space-lg text-body text-ink-muted">
        {render_slot(@inner_block)}
      </p>
      <code class="rounded-sm bg-surface-sunk px-space-md py-space-xs text-ref text-ink">
        {@command}
      </code>
    </div>
    """
  end

  attr(:active_runs, :list, default: [])
  attr(:recent_runs, :list, default: [])
  attr(:form, :map, required: true)
  attr(:error, :string, default: nil)
  attr(:available_classifiers, :list, default: [])
  attr(:confirm_id, :string, required: true)
  attr(:open_confirm, :string, default: nil)
  attr(:confirm_error, :string, default: nil)

  defp optimizer_panel(assigns) do
    ~H"""
    <div class="space-y-space-xl">
      <!-- Summary Cards -->
      <.optimizer_summary active={@active_runs} recent={@recent_runs} />

      <!-- Run Launcher -->
      <.optimizer_launcher
        form={@form}
        error={@error}
        available_classifiers={@available_classifiers}
        confirm_id={@confirm_id}
        open_confirm={@open_confirm}
        confirm_error={@confirm_error}
      />

      <!-- Active Runs -->
      <%= if @active_runs != [] do %>
        <div class="space-y-space-md">
          <h3 class="flex items-center gap-space-sm text-subheading text-ink">
            <.status_dot status={:running} pulse={true} />
            Active runs ({length(@active_runs)})
          </h3>
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
            <.active_run_card
              :for={run <- @active_runs}
              run={run}
              open_confirm={@open_confirm}
              confirm_error={@confirm_error}
            />
          </div>
        </div>
      <% end %>

      <!-- Recent Runs -->
      <.card class="overflow-hidden">
        <div class="p-space-lg border-b border-border flex items-center justify-between">
          <h3 class="text-heading text-ink">
            Recent runs
            <span class="ml-space-xs text-body text-ink-muted">
              ({length(@recent_runs)} stored)
            </span>
          </h3>
        </div>
        <%= if @recent_runs == [] do %>
          <.empty_message
            icon="hero-cpu-chip"
            title="No optimizer runs yet"
            command="mix train_micro --only intent_full"
          >
            Kick off a run from the launcher above, or train a feature-vector micro-classifier from the CLI.
          </.empty_message>
        <% else %>
          <div class="overflow-x-auto">
            <table class="w-full text-left text-body-dense text-ink tabular-nums">
              <thead class="bg-surface-sunk">
                <tr>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Status</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Classifier</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Best Fitness</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Best Gen</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Generations</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Alive Dims</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Population</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Mut Rate</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Mut Sigma</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Early Stop</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Started</th>
                  <th class="h-row-compact px-space-sm text-label text-ink-muted">Duration</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-border">
                <tr :for={run <- @recent_runs} class="even:bg-surface-sunk">
                  <td class="h-row-compact px-space-sm">
                    <.run_status_badge status={Map.get(run, :status, :complete)} />
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-ink">{Map.get(run, :classifier, "-")}</td>
                  <td class={["h-row-compact px-space-sm", metric_class(:heuristic, Map.get(run, :best_fitness))]}>
                    {format_percent(Map.get(run, :best_fitness))}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-score-count">
                    {Map.get(run, :best_generation, "-")}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-score-count">
                    {Map.get(run, :generations_run, "-")}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-score-count">
                    {format_alive_dims(Map.get(run, :alive_dims), Map.get(run, :total_dims))}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-ink">
                    {run_opt(run, :population_size, "population_size")}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-ink">
                    {run_opt(run, :mutation_rate, "mutation_rate")}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-ink">
                    {run_opt(run, :mutation_sigma, "mutation_sigma")}
                  </td>
                  <td class="h-row-compact px-space-sm text-value text-ink">
                    {run_opt(run, :early_stop_generations, "early_stop_generations")}
                  </td>
                  <td class="h-row-compact px-space-sm text-caption text-ink-muted">
                    {format_time_ago(Map.get(run, :started_at))}
                  </td>
                  <td class="h-row-compact px-space-sm text-caption text-ink-muted">
                    {format_duration(Map.get(run, :duration_ms))}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        <% end %>
      </.card>
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
    <div class="grid grid-cols-2 lg:grid-cols-4 gap-space-lg">
      <.stat_kpi
        label="Active runs"
        value={to_string(length(@active))}
        icon="hero-bolt"
      />
      <.stat_kpi
        label="Stored runs"
        value={to_string(length(@recent))}
        icon="hero-archive-box"
      />
      <.stat_kpi
        label="Best fitness"
        value={format_percent(@best_run && Map.get(@best_run, :best_fitness))}
        icon="hero-trophy"
      />
      <.stat_kpi
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
  attr(:confirm_id, :string, required: true)
  attr(:open_confirm, :string, default: nil)
  attr(:confirm_error, :string, default: nil)

  defp optimizer_launcher(assigns) do
    ~H"""
    <.card>
      <.card_body>
        <div class="flex items-center justify-between mb-space-lg">
          <h3 class="text-heading text-ink">
            Launch a new GA run
          </h3>
          <span class="text-caption text-ink-muted">
            Genetic algorithm — per-dimension feature weights
          </span>
        </div>
        <form phx-change="update_optimizer_form" phx-submit="request_optimizer_run" class="space-y-space-md">
          <div class="grid grid-cols-2 lg:grid-cols-3 gap-space-md">
            <.input
              type="select"
              name="classifier"
              label="Classifier"
              options={@available_classifiers}
              value={@form["classifier"]}
            />
            <.optimizer_input form={@form} field="population_size" label="Population size" min="10" step="10" />
            <.optimizer_input form={@form} field="max_generations" label="Max generations" min="10" step="10" />
            <.optimizer_input form={@form} field="early_stop_generations" label="Early stop (gens)" min="1" step="1" />
            <.optimizer_input form={@form} field="mutation_rate" label="Mutation rate" min="0" max="1" step="0.01" />
            <.optimizer_input form={@form} field="mutation_sigma" label="Mutation sigma" min="0" max="3" step="0.05" />
          </div>
          <%= if @error do %>
            <div class="mt-space-xs text-caption text-red">{@error}</div>
          <% end %>
          <div class="flex justify-end">
            <.btn
              id="start-optimizer-run"
              type="submit"
              variant={:primary}
              size={:sm}
              reach={:shared}
              target={"#{@form["classifier"]} run"}
              icon="hero-rocket-launch"
            >
              Start run
            </.btn>
          </div>
        </form>
        <.execute_confirm
          id={@confirm_id}
          open={@open_confirm == @confirm_id}
          reach={:shared}
          verb="Start run"
          target={"#{@form["classifier"]} run"}
          consequence={"Starts a #{@form["classifier"]} weight-optimizer run on this node. Everyone viewing this page sees it as it runs, and its record is written to this node's optimizer runs folder when it finishes."}
          on_confirm="start_optimizer_run"
          on_cancel="close_confirm"
          trigger_id="start-optimizer-run"
          error={@confirm_error}
          class="mt-space-md ml-auto"
        />
      </.card_body>
    </.card>
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
    <.input
      type="number"
      name={@field}
      label={@label}
      value={@form[@field]}
      min={@min}
      max={@max}
      step={@step}
    />
    """
  end

  attr(:run, :map, required: true)
  attr(:open_confirm, :string, default: nil)
  attr(:confirm_error, :string, default: nil)

  defp active_run_card(assigns) do
    history = Map.get(assigns.run, :history) || []
    max_gen = get_in(assigns.run, [:opts, :max_generations]) || get_in(assigns.run, [:opts, "max_generations"]) || 200
    progress_pct = min(round((Map.get(assigns.run, :generation, 0) + 1) / max(max_gen, 1) * 100), 100)
    run_id = Map.fetch!(assigns.run, :run_id)

    assigns =
      assigns
      |> assign(:history, history)
      |> assign(:max_gen, max_gen)
      |> assign(:progress_pct, progress_pct)
      |> assign(:run_id, run_id)
      |> assign(:cancel_confirm_id, "confirm-cancel-run-#{run_id}")
      |> assign(:cancel_trigger_id, "cancel-run-#{run_id}")

    ~H"""
    <.card>
      <.card_body class="space-y-space-md">
      <div class="flex items-start justify-between gap-space-sm">
        <div class="min-w-0">
          <div class="flex items-center gap-space-sm">
            <span class="text-value-strong text-ink truncate">{Map.get(@run, :classifier, "-")}</span>
            <.run_status_badge status={Map.get(@run, :status, :running)} />
          </div>
          <div class="text-ref text-ink-muted truncate">{@run_id}</div>
        </div>
        <.btn
          id={@cancel_trigger_id}
          variant={:outline}
          size={:xs}
          reach={:shared}
          target={@run_id}
          icon="hero-x-mark"
          phx-click="open_confirm"
          phx-value-id={@cancel_confirm_id}
        >
          Cancel run
        </.btn>
      </div>
      <.execute_confirm
        id={@cancel_confirm_id}
        open={@open_confirm == @cancel_confirm_id}
        reach={:shared}
        verb="Cancel run"
        target={@run_id}
        consequence="Stops this run and moves it to recent runs as cancelled, for everyone viewing this page. A cancelled run is not written to disk, so it is gone after a restart."
        on_confirm={JS.push("cancel_optimizer_run", value: %{run_id: @run_id})}
        on_cancel="close_confirm"
        trigger_id={@cancel_trigger_id}
        error={@confirm_error}
        class="ml-auto"
      />

      <!-- Progress bar -->
      <div class="space-y-space-xs">
        <div class="flex justify-between text-caption text-ink-muted">
          <span>Generation {Map.get(@run, :generation, 0)} / {@max_gen}</span>
          <span>{@progress_pct}%</span>
        </div>
        <div
          class="h-space-sm w-full rounded-sm bg-progress-track"
          role="progressbar"
          aria-valuemin="0"
          aria-valuemax="100"
          aria-valuenow={@progress_pct}
        >
          <div class="h-full rounded-sm bg-progress-fill" style={"width: #{@progress_pct}%"} />
        </div>
      </div>

      <!-- Metrics grid -->
      <div class="grid grid-cols-3 gap-space-sm">
        <.kv_block label="Best" value={format_percent(Map.get(@run, :best_fitness))} highlight={true} />
        <.kv_block label="Raw" value={format_percent(Map.get(@run, :raw_acc))} />
        <.kv_block label="Balanced" value={format_percent(Map.get(@run, :balanced_acc))} />
        <.kv_block label="Avg" value={format_percent(Map.get(@run, :avg_fitness))} />
        <.kv_block label="Stale" value={to_string(Map.get(@run, :stale_count, 0))} />
        <.kv_block label="Mut" value={format_mutation(Map.get(@run, :mutation_rate), Map.get(@run, :mutation_sigma))} />
      </div>
      <!-- Config grid -->
      <div class="grid grid-cols-4 gap-space-sm">
        <.kv_block label="Population" value={run_opt(@run, :population_size, "population_size")} />
        <.kv_block label="Mut Rate" value={run_opt(@run, :mutation_rate, "mutation_rate")} />
        <.kv_block label="Mut Sigma" value={run_opt(@run, :mutation_sigma, "mutation_sigma")} />
        <.kv_block label="Early Stop" value={run_opt(@run, :early_stop_generations, "early_stop_generations")} />
      </div>

      <!-- Sparkline -->
      <%= if length(@history) > 1 do %>
        <.sparkline history={@history} />
      <% end %>
      </.card_body>
    </.card>
    """
  end

  attr(:label, :string, required: true)
  attr(:value, :string, required: true)
  attr(:highlight, :boolean, default: false)

  defp kv_block(assigns) do
    ~H"""
    <div class={[
      "rounded-sm px-space-sm py-space-xs",
      if(@highlight, do: "bg-accent-wash", else: "bg-surface-sunk")
    ]}>
      <div class="text-label text-ink-muted">{@label}</div>
      <div class={
        if(@highlight, do: "text-value-strong text-accent", else: "text-value text-ink")
      }>{@value}</div>
    </div>
    """
  end

  @run_status_badges %{
    running: {:info, "running"},
    complete: {:success, "complete"},
    early_stop: {:success, "early stop"},
    cancelled: {:default, "cancelled"},
    error: {:error, "error"}
  }

  attr(:status, :atom, required: true)

  defp run_status_badge(assigns) do
    {variant, label} =
      case Map.fetch(@run_status_badges, assigns.status) do
        {:ok, badge} ->
          badge

        :error ->
          raise ArgumentError,
                "ChatWeb.AccuracyLive.run_status_badge/1: no treatment for optimizer run status " <>
                  "#{inspect(assigns.status)}. The statuses are #{inspect(Map.keys(@run_status_badges))}."
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
    <div class="flex items-center gap-space-sm">
      <svg viewBox={"0 0 #{@width} #{@height}"} class="w-full h-12" preserveAspectRatio="none" role="img" aria-label="Fitness sparkline">
        <polyline
          points={@polyline}
          fill="none"
          stroke="var(--blue)"
          stroke-width="2"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
      </svg>
      <div class="text-offset text-ink-muted whitespace-nowrap">
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
      <div class="py-space-lg text-center text-body text-ink-muted">{@message}</div>
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
          stroke="var(--border)"
          stroke-dasharray="4,4"
        />
        <!-- Y-axis labels -->
        <text
          :for={tick <- @y_ticks}
          x={@padding_x - 8}
          y={tick.y + 4}
          text-anchor="end"
          class="fill-ink-muted"
          font-size="10"
        >
          {trunc(tick.val)}%
        </text>
        <!-- Trend line -->
        <polyline
          points={@polyline_points}
          fill="none"
          stroke="var(--blue)"
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
          fill="var(--blue)"
          stroke="var(--surface)"
          stroke-width="2"
        />
        <!-- Value labels on dots -->
        <text
          :for={dot <- @dots}
          x={dot.x}
          y={dot.y - 10}
          text-anchor="middle"
          class="fill-ink-muted"
          font-size="9"
        >
          {dot.val}%
        </text>
      </svg>
      """
    end
  end

  defp task_label(task), do: @task_labels[task] || task

  defp close_confirm(socket) do
    socket |> assign(:open_confirm, nil) |> assign(:confirm_error, nil)
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

  # A store that raised could not be asked, which is not the same finding as a
  # store that holds no evaluation: the first shows as the "could not be
  # asked" empty state with the exception, the second as no evaluations yet.
  defp load_evaluations do
    Map.new(@tasks, fn task ->
      result =
        try do
          EvaluationStore.latest(task)
        rescue
          e ->
            {:could_not_ask,
             "Brain.ML.EvaluationStore.latest(#{inspect(task)}) raised #{inspect(e.__struct__)}: " <>
               Exception.message(e)}
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

  # An evaluation metric is a 0..1 rate shown in ink with its kind beside it.
  # One missing from the result reads as absent, never as a dash or a zero.
  defp metric_value(nil), do: "absent"
  defp metric_value(value) when is_number(value), do: format_percent(value)

  defp metric_value(other) do
    raise ArgumentError, "ChatWeb.AccuracyLive.metric_value/1: a metric must be a number, got #{inspect(other)}"
  end

  defp count_value(nil), do: "absent"
  defp count_value(count) when is_integer(count), do: Integer.to_string(count)

  defp count_value(other) do
    raise ArgumentError, "ChatWeb.AccuracyLive.count_value/1: a count must be an integer, got #{inspect(other)}"
  end

  defp metric_sublabel(kind, nil, _examples), do: "no #{kind} in this result"
  defp metric_sublabel(kind, _value, nil), do: "#{kind} · no example count in this result"
  defp metric_sublabel(kind, _value, examples) when is_integer(examples), do: "#{kind} · #{examples} examples"

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

  # A metric cell is styled by what kind of number it holds, never by how good
  # the number is: a threshold coloring would be a verdict nobody declared.
  #
  #   * `:rate` — precision, recall and F1, measured against labeled examples:
  #     bounded 0 to 1, shown as a percentage.
  #   * `:heuristic` — the optimizer's composite fitness: bounded, but a blend
  #     of raw and balanced accuracy rather than a probability.
  #   * `:count` — whole numbers such as support, in the count style.
  #
  # Text takes ink in every kind; the score hues are for non-text marks only.
  # A missing value is muted so it never reads as a measured zero.
  @metric_kind_classes %{
    rate: "text-value text-ink",
    heuristic: "text-value text-ink",
    count: "text-value text-score-count"
  }

  defp metric_class(kind, value) do
    case Map.fetch(@metric_kind_classes, kind) do
      {:ok, class} ->
        if is_number(value), do: class, else: "text-value text-ink-muted"

      :error ->
        raise ArgumentError,
              "ChatWeb.AccuracyLive.metric_class/2: no treatment for metric kind #{inspect(kind)}. " <>
                "The kinds are #{inspect(Map.keys(@metric_kind_classes))}."
    end
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
