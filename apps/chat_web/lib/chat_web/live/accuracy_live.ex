defmodule ChatWeb.AccuracyLive do
  @moduledoc """
  ML model accuracy dashboard for viewing evaluation results and experiment comparisons.

  Provides visibility into:
  - Per-task evaluation metrics (intent, NER, sentiment, speech act)
  - Per-class precision, recall, F1, and support
  - Accuracy trends over evaluation runs (inline SVG charts)
  - Experiment comparisons from ExperimentTracker
  """

  use ChatWeb, :live_view
  require Logger

  import ChatWeb.AppShell

  alias Brain.ML.EvaluationStore
  alias Brain.ML.LSTM.ExperimentTracker

  @tasks ~w(intent ner sentiment speech_act)
  @task_labels %{
    "intent" => "Intent",
    "ner" => "NER",
    "sentiment" => "Sentiment",
    "speech_act" => "Speech Act"
  }

  @impl true
  def mount(_params, _session, socket) do
    {:ok, socket}
  end

  @impl true
  def handle_params(_params, _uri, socket) do
    socket =
      socket
      |> assign(:active_tab, "intent")
      |> assign(:sort_field, "label")
      |> assign(:sort_dir, :asc)
      |> load_all_data()

    {:noreply, socket}
  end

  # ============================================================================
  # Event Handlers
  # ============================================================================

  @impl true
  def handle_event("switch_tab", %{"tab" => tab}, socket) do
    {:noreply, assign(socket, :active_tab, tab)}
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
    socket = put_flash(socket, :info, "Evaluation triggered. Run `mix evaluate.intent --save` from the terminal for full results.")
    {:noreply, socket}
  end

  def handle_event("refresh_data", _params, socket) do
    {:noreply, load_all_data(socket)}
  end

  def handle_event("switch_world", %{"world_id" => _world_id}, socket) do
    {:noreply, socket}
  end

  def handle_event("refresh_worlds", _params, socket) do
    {:noreply, socket}
  end

  @impl true
  def handle_info({:world_context_changed, _world_id}, socket) do
    {:noreply, socket}
  end

  # ============================================================================
  # Render
  # ============================================================================

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
              ML model evaluation results and experiment comparisons
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
              tab="experiments"
              label="Experiments"
              active={@active_tab == "experiments"}
            />
          </nav>
        </div>

        <!-- Tab Content -->
        <%= if @active_tab == "experiments" do %>
          <.experiments_panel experiments={@experiments} />
        <% else %>
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

  # ============================================================================
  # Function Components
  # ============================================================================

  attr :tab, :string, required: true
  attr :label, :string, required: true
  attr :active, :boolean, default: false

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

  attr :task, :string, required: true
  attr :evaluation, :map, default: nil
  attr :trend, :list, default: []
  attr :sort_field, :string, default: "label"
  attr :sort_dir, :atom, default: :asc

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

  attr :label, :string, required: true
  attr :value, :string, required: true
  attr :icon, :string, required: true

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

  attr :field, :string, required: true
  attr :label, :string, required: true
  attr :sort_field, :string, required: true
  attr :sort_dir, :atom, required: true

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

  attr :task, :string, required: true
  attr :task_label, :string, required: true

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

  attr :experiments, :list, default: []

  defp experiments_panel(assigns) do
    ~H"""
    <%= if @experiments != [] do %>
      <div class="bg-base-100 rounded-xl border border-base-300/50 overflow-hidden">
        <div class="p-4 border-b border-base-300/50">
          <h3 class="text-sm font-semibold text-base-content/70">
            Experiment Comparison
            <span class="text-base-content/40 font-normal ml-1">(sorted by validation accuracy)</span>
          </h3>
        </div>
        <div class="overflow-x-auto">
          <table class="table table-sm w-full">
            <thead>
              <tr class="border-b border-base-300/50">
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Name</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Val Acc</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Val Loss</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Train Acc</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Epochs</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Time</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Hidden</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">Batch</th>
                <th class="text-xs font-semibold text-base-content/70 uppercase tracking-wider">LR</th>
              </tr>
            </thead>
            <tbody>
              <tr
                :for={{exp, idx} <- Enum.with_index(@experiments)}
                class={[
                  "border-b border-base-300/30 hover:bg-base-200/30",
                  if(idx == 0, do: "bg-success/5", else: "")
                ]}
              >
                <td class="font-medium">
                  {exp.name || "unnamed"}
                  <%= if idx == 0 do %>
                    <.badge variant={:success} size={:xs} class="ml-1">best</.badge>
                  <% end %>
                </td>
                <td class="font-mono text-xs">{exp.val_acc}</td>
                <td class="font-mono text-xs">{exp.val_loss}</td>
                <td class="font-mono text-xs">{exp.train_acc}</td>
                <td class="text-base-content/70">{exp.epochs}</td>
                <td class="text-base-content/70">{exp.time}</td>
                <td class="text-base-content/70">{exp.hidden || "-"}</td>
                <td class="text-base-content/70">{exp.batch || "-"}</td>
                <td class="font-mono text-xs">{exp.lr || "-"}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    <% else %>
      <div class="bg-base-100 rounded-xl border border-base-300/50 p-12 text-center">
        <div class="mx-auto w-12 h-12 rounded-full bg-base-200 flex items-center justify-center mb-4">
          <.icon name="hero-beaker" class="size-6 text-base-content/40" />
        </div>
        <h3 class="text-lg font-semibold text-base-content/70 mb-2">No experiments yet</h3>
        <p class="text-sm text-base-content/50 mb-4">
          Train models with different configurations to compare results.
        </p>
        <code class="text-xs bg-base-200 px-3 py-1.5 rounded-lg text-base-content/70">
          mix train_lstm --epochs 10 --name "my_experiment"
        </code>
      </div>
    <% end %>
    """
  end

  # ============================================================================
  # Inline SVG Trend Chart
  # ============================================================================

  attr :points, :list, required: true

  defp trend_chart(assigns) do
    points = assigns.points
    count = length(points)

    if count < 2 do
      assigns = assign(assigns, :message, "Not enough data points for a chart.")

      ~H"""
      <div class="text-sm text-base-content/50 text-center py-4">{@message}</div>
      """
    else
      # Chart dimensions
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

      # Build polyline points string
      polyline_points =
        values
        |> Enum.with_index()
        |> Enum.map(fn {val, i} ->
          x = padding_x + i / max(count - 1, 1) * chart_width
          y = padding_y + (1 - (val - min_val) / val_range) * chart_height
          "#{Float.round(x * 1.0, 1)},#{Float.round(y * 1.0, 1)}"
        end)
        |> Enum.join(" ")

      # Build dot positions
      dots =
        values
        |> Enum.with_index()
        |> Enum.map(fn {val, i} ->
          x = padding_x + i / max(count - 1, 1) * chart_width
          y = padding_y + (1 - (val - min_val) / val_range) * chart_height
          %{x: Float.round(x * 1.0, 1), y: Float.round(y * 1.0, 1), val: Float.round(val, 1)}
        end)

      # Y-axis labels (5 ticks)
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

  # ============================================================================
  # Data Loading
  # ============================================================================

  defp load_all_data(socket) do
    evaluations = load_evaluations()
    trends = load_trends()
    experiments = load_experiments()

    socket
    |> assign(:tasks, @tasks)
    |> assign(:task_labels, @task_labels)
    |> assign(:evaluations, evaluations)
    |> assign(:trends, trends)
    |> assign(:experiments, experiments)
  end

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

  defp load_experiments do
    try do
      ExperimentTracker.compare_all()
    rescue
      _ -> []
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp format_percent(nil), do: "-"
  defp format_percent(val) when is_float(val) and val <= 1.0, do: "#{Float.round(val * 100, 1)}%"
  defp format_percent(val) when is_float(val), do: "#{Float.round(val, 1)}%"
  defp format_percent(val) when is_integer(val), do: "#{val}%"
  defp format_percent(_), do: "-"

  defp metric_color_class(nil), do: "text-base-content/50"

  defp metric_color_class(val) when is_number(val) do
    cond do
      val >= 0.9 -> "text-success font-medium"
      val >= 0.7 -> "text-warning font-medium"
      true -> "text-error font-medium"
    end
  end

  defp metric_color_class(_), do: "text-base-content/50"

  defp sorted_per_class(nil, _field, _dir), do: []

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

  defp toggle_sort_dir(:asc), do: :desc
  defp toggle_sort_dir(:desc), do: :asc
end
