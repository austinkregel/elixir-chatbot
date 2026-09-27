defmodule ChatWeb.POSTrainingLive do
  @moduledoc """
  Isolation page for training the POS tagger.

  Start a recorded run (`Brain.Training.POSRuns`) through
  `Brain.ML.TrainingServer`, watch its learning curve epoch by epoch, compare
  runs to see where more epochs stop paying, and promote a snapshot to the
  test suite's model or to the model the app serves.

  Runs are read from disk, so they outlive the page and the app; a run can
  be left going and looked at later.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Brain.ML.TrainingServer
  alias Brain.Training.POSRuns

  @palette ~w(#6366f1 #f59e0b #10b981 #ef4444 #8b5cf6 #0ea5e9 #ec4899 #84cc16)
  @compared_by_default 4

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: Phoenix.PubSub.subscribe(Brain.PubSub, TrainingServer.pos_topic())

    runs = POSRuns.list()

    {:ok,
     socket
     |> assign(:runs, runs)
     |> assign(:current_run, TrainingServer.current_run())
     |> assign(:compared, runs |> Enum.take(@compared_by_default) |> MapSet.new(& &1["id"]))
     |> assign(:log_x, true)
     |> assign(:promoting, nil)
     |> assign(:form, to_form(default_params(), as: :run))}
  end

  defp default_params do
    %{"sentences" => "2000", "max_epochs" => "1000", "patience" => "", "seed" => to_string(Brain.ML.TrainingSeed.get!())}
  end

  # ---------------------------------------------------------------------------
  # Events
  # ---------------------------------------------------------------------------

  @impl true
  def handle_event("start", %{"run" => form}, socket) do
    with {:ok, params} <- parse(form),
         {:ok, :pos, id} <- TrainingServer.start_training(:pos, params) do
      {:noreply,
       socket
       |> assign(:current_run, id)
       |> assign(:compared, MapSet.put(socket.assigns.compared, id))
       |> assign(:form, to_form(form, as: :run))
       |> put_flash(:info, "Run #{id} started")}
    else
      {:error, {:already_training, model}} ->
        {:noreply, put_flash(socket, :error, "The training server is already training #{model}")}

      {:error, message} ->
        {:noreply, socket |> assign(:form, to_form(form, as: :run)) |> put_flash(:error, to_string(message))}
    end
  end

  def handle_event("cancel", _params, socket) do
    case TrainingServer.cancel() do
      :ok -> {:noreply, socket |> reload_runs() |> assign(:current_run, nil) |> put_flash(:info, "Run cancelled")}
      {:error, :not_training} -> {:noreply, put_flash(socket, :error, "Nothing is training")}
    end
  end

  def handle_event("toggle_compare", %{"id" => id}, socket) do
    compared = socket.assigns.compared
    compared = if MapSet.member?(compared, id), do: MapSet.delete(compared, id), else: MapSet.put(compared, id)
    {:noreply, assign(socket, :compared, compared)}
  end

  def handle_event("toggle_log_x", _params, socket) do
    {:noreply, assign(socket, :log_x, not socket.assigns.log_x)}
  end

  def handle_event("promote", %{"run_id" => id, "snapshot" => snapshot, "target" => target}, socket)
      when target in ["test", "production"] do
    target = String.to_existing_atom(target)

    {:noreply,
     socket
     |> assign(:promoting, {id, snapshot, target})
     |> start_async(:promote, fn -> POSRuns.promote!(id, snapshot, target) end)}
  end

  @impl true
  def handle_async(:promote, {:ok, {path, model}}, socket) do
    {_id, snapshot, target} = socket.assigns.promoting
    e = model.evaluation

    {:noreply,
     socket
     |> assign(:promoting, nil)
     |> reload_runs()
     |> put_flash(
       :info,
       "#{snapshot} promoted to the #{target} model (#{path}): test accuracy #{pct(e.accuracy)}, lookup baseline #{pct(e.lookup_baseline)}"
     )}
  end

  def handle_async(:promote, {:exit, reason}, socket) do
    message =
      case reason do
        {exception, _stack} when is_exception(exception) -> Exception.message(exception)
        other -> inspect(other)
      end

    {:noreply, socket |> assign(:promoting, nil) |> put_flash(:error, "Promotion refused: #{message}")}
  end

  @impl true
  def handle_info({:pos_run_started, id}, socket) do
    {:noreply, socket |> assign(:current_run, id) |> reload_runs()}
  end

  def handle_info({:pos_epoch, id, _progress}, socket) do
    {:noreply, replace_run(socket, POSRuns.get!(id))}
  end

  def handle_info({:pos_run_finished, _id, _status}, socket) do
    {:noreply, socket |> assign(:current_run, nil) |> reload_runs()}
  end

  defp reload_runs(socket), do: assign(socket, :runs, POSRuns.list())

  defp replace_run(socket, run) do
    runs = socket.assigns.runs

    runs =
      if Enum.any?(runs, &(&1["id"] == run["id"])),
        do: Enum.map(runs, &if(&1["id"] == run["id"], do: run, else: &1)),
        else: [run | runs]

    assign(socket, :runs, runs)
  end

  # Form strings to run params; blank sentences means all of them, blank
  # patience means never stop early.
  defp parse(form) do
    with {:ok, sentences} <- integer_or_nil(form["sentences"], "sentences"),
         {:ok, max_epochs} <- integer_or_nil(form["max_epochs"], "max epochs"),
         {:ok, patience} <- integer_or_nil(form["patience"], "patience"),
         {:ok, seed} <- integer_or_nil(form["seed"], "seed") do
      {:ok, [sentences: sentences, max_epochs: max_epochs, patience: patience, seed: seed]}
    end
  end

  defp integer_or_nil(value, label) do
    case String.trim(value || "") do
      "" ->
        {:ok, nil}

      text ->
        case Integer.parse(text) do
          {n, ""} -> {:ok, n}
          _ -> {:error, "#{label} must be a whole number, got #{inspect(text)}"}
        end
    end
  end

  # ---------------------------------------------------------------------------
  # Render
  # ---------------------------------------------------------------------------

  @impl true
  def render(assigns) do
    current = Enum.find(assigns.runs, &(&1["id"] == assigns.current_run))

    series =
      assigns.runs
      |> Enum.filter(&MapSet.member?(assigns.compared, &1["id"]))
      |> Enum.reverse()
      |> Enum.with_index()
      |> Enum.map(fn {run, i} -> %{run: run, color: Enum.at(@palette, rem(i, length(@palette)))} end)

    assigns = assign(assigns, current: current, series: series)

    ~H"""
    <.app_shell
      current_world_id={@current_world_id}
      available_worlds={@available_worlds}
      current_path={@current_path}
      system_ready={@system_ready}
      flash={@flash}
    >
      <:page_header>
        <div>
          <h1 class="text-xl font-bold">POS Training</h1>
          <p class="text-sm text-base-content/60">
            Train the part-of-speech tagger on EWT, watch where more epochs stop paying, promote a snapshot
          </p>
        </div>
      </:page_header>

      <div class="p-4 space-y-4">
        <div class="card bg-base-200">
          <div class="card-body p-4">
            <h2 class="font-semibold">New run</h2>
            <.form for={@form} id="pos-run-form" phx-submit="start" class="grid grid-cols-2 md:grid-cols-5 gap-3 items-end">
              <label class="form-control">
                <span class="label-text text-xs">Sentences (blank: all 12,544)</span>
                <input type="text" name="run[sentences]" value={@form[:sentences].value} class="input input-bordered input-sm" />
              </label>
              <label class="form-control">
                <span class="label-text text-xs">Max epochs</span>
                <input type="text" name="run[max_epochs]" value={@form[:max_epochs].value} class="input input-bordered input-sm" />
              </label>
              <label class="form-control">
                <span class="label-text text-xs">Early stop after (blank: never)</span>
                <input type="text" name="run[patience]" value={@form[:patience].value} class="input input-bordered input-sm" />
              </label>
              <label class="form-control">
                <span class="label-text text-xs">Seed</span>
                <input type="text" name="run[seed]" value={@form[:seed].value} class="input input-bordered input-sm" />
              </label>
              <button type="submit" class="btn btn-primary btn-sm" disabled={@current_run != nil}>Start run</button>
            </.form>
            <p class="text-xs text-base-content/50">
              Snapshots are kept at epochs {Enum.join(POSRuns.snapshot_epochs(), ", ")}, the last epoch, and the best on dev.
              Measured on this machine: about 12 s per epoch on 2,000 sentences, 50 s on all of them.
            </p>
          </div>
        </div>

        <div :if={@current} id="current-run" class="card bg-base-200">
          <div class="card-body p-4">
            <div class="flex items-center justify-between">
              <h2 class="font-semibold">Running: <span class="font-mono text-sm">{@current["id"]}</span></h2>
              <button phx-click="cancel" data-confirm="Stop this run? Its snapshots so far are kept." class="btn btn-error btn-sm">
                Cancel
              </button>
            </div>
            <.progress run={@current} />
          </div>
        </div>

        <div class="card bg-base-200">
          <div class="card-body p-4 space-y-2">
            <div class="flex items-center justify-between">
              <h2 class="font-semibold">Dev accuracy by epoch</h2>
              <button phx-click="toggle_log_x" class="btn btn-ghost btn-xs">
                Epoch axis: {if @log_x, do: "log", else: "linear"}
              </button>
            </div>
            <.curve_chart series={@series} field="dev_accuracy" log_x={@log_x} percent={true} />
            <h2 class="font-semibold pt-2">Training loss by epoch</h2>
            <.curve_chart series={@series} field="loss" log_x={@log_x} percent={false} />
            <div class="flex flex-wrap gap-3 text-xs">
              <span :for={s <- @series} class="flex items-center gap-1">
                <span class="inline-block w-3 h-3 rounded-sm" style={"background: #{s.color}"}></span>
                <span class="font-mono">{s.run["id"]}</span>
                <span class="text-base-content/50">{describe(s.run["params"])}</span>
              </span>
            </div>
          </div>
        </div>

        <div class="card bg-base-200">
          <div class="card-body p-4">
            <h2 class="font-semibold">Runs</h2>
            <p :if={@runs == []} class="text-sm text-base-content/50">No runs yet.</p>
            <div :if={@runs != []} class="overflow-x-auto">
              <table id="pos-runs" class="table table-sm">
                <thead>
                  <tr>
                    <th>Compare</th>
                    <th>Run</th>
                    <th>Status</th>
                    <th>Settings</th>
                    <th>Epochs</th>
                    <th>Best dev</th>
                    <th title="First epoch within this much of the run's best dev accuracy">Within 0.5 / 0.1 pt of best</th>
                    <th>Promote</th>
                  </tr>
                </thead>
                <tbody>
                  <tr :for={run <- @runs} id={"run-" <> run["id"]}>
                    <td>
                      <input
                        type="checkbox"
                        class="checkbox checkbox-xs"
                        checked={MapSet.member?(@compared, run["id"])}
                        phx-click="toggle_compare"
                        phx-value-id={run["id"]}
                      />
                    </td>
                    <td class="font-mono text-xs">{run["id"]}</td>
                    <td><span class={["badge badge-sm", status_class(run["status"])]}>{run["status"]}</span></td>
                    <td class="text-xs">{describe(run["params"])}</td>
                    <td>{length(run["curve"] || [])}</td>
                    <td>{best_text(run)}</td>
                    <td class="text-xs">{plateau_text(run)}</td>
                    <td>
                      <form :if={(run["snapshots"] || []) != []} phx-submit="promote" class="flex gap-1 items-center">
                        <input type="hidden" name="run_id" value={run["id"]} />
                        <select name="snapshot" class="select select-bordered select-xs">
                          <option :for={s <- run["snapshots"]} value={s}>{s}</option>
                        </select>
                        <button name="target" value="test" class="btn btn-xs" disabled={@promoting != nil}>Test</button>
                        <button
                          name="target"
                          value="production"
                          class="btn btn-xs btn-warning"
                          disabled={@promoting != nil}
                          data-confirm="Replace the POS model the app serves with this snapshot?"
                        >
                          Production
                        </button>
                      </form>
                      <div :for={p <- run["promotions"] || []} class="text-[10px] text-base-content/60">
                        {p["snapshot"]} → {p["target"]}: test {pct(p["test_accuracy"])} vs lookup {pct(p["lookup_baseline"])}
                      </div>
                      <div :if={run["error"]} class="text-[10px] text-error">{run["error"]}</div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p :if={@promoting} class="text-sm text-base-content/60">
              Measuring {elem(@promoting, 1)} of {elem(@promoting, 0)} on the test split...
            </p>
          </div>
        </div>
      </div>
    </.app_shell>
    """
  end

  attr :run, :map, required: true

  defp progress(assigns) do
    run = assigns.run
    curve = run["curve"] || []
    last = List.last(curve)
    max_epochs = run["params"]["max_epochs"]

    eta =
      case last do
        %{"epoch" => epoch, "elapsed_ms" => ms} when epoch > 0 -> div(ms, epoch) * (max_epochs - epoch)
        _ -> nil
      end

    assigns = assign(assigns, last: last, max_epochs: max_epochs, eta: eta)

    ~H"""
    <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
      <div>Epoch <strong>{if @last, do: @last["epoch"], else: 0}</strong> / {@max_epochs}</div>
      <div>Dev <strong>{if @last, do: pct(@last["dev_accuracy"]), else: "—"}</strong></div>
      <div>Best <strong>{best_text(@run)}</strong></div>
      <div>Loss <strong>{if @last, do: Float.round(@last["loss"], 4), else: "—"}</strong></div>
      <div>Left <strong>{if @eta, do: duration(@eta), else: "—"}</strong></div>
    </div>
    """
  end

  attr :series, :list, required: true
  attr :field, :string, required: true
  attr :log_x, :boolean, required: true
  attr :percent, :boolean, required: true

  defp curve_chart(assigns) do
    points =
      Enum.map(assigns.series, fn s ->
        {s.color, for(p <- s.run["curve"] || [], is_number(p[assigns.field]), do: {p["epoch"], p[assigns.field]})}
      end)

    all = Enum.flat_map(points, &elem(&1, 1))

    if length(all) < 2 do
      assigns = assign(assigns, :message, "Pick a run with at least two epochs to plot.")

      ~H"""
      <div class="text-sm text-base-content/50 text-center py-4">{@message}</div>
      """
    else
      {width, height, pad_l, pad_r, pad_t, pad_b} = {640, 220, 52, 12, 10, 26}
      plot_w = width - pad_l - pad_r
      plot_h = height - pad_t - pad_b

      max_x = all |> Enum.map(&elem(&1, 0)) |> Enum.max() |> max(2)
      ys = Enum.map(all, &elem(&1, 1))
      {min_y, max_y} = {Enum.min(ys), Enum.max(ys)}
      range_y = max(max_y - min_y, 1.0e-6)

      scale_x = fn x ->
        if assigns.log_x,
          do: pad_l + :math.log(x) / :math.log(max_x) * plot_w,
          else: pad_l + (x - 1) / (max_x - 1) * plot_w
      end

      scale_y = fn y -> pad_t + (1 - (y - min_y) / range_y) * plot_h end

      lines =
        for {color, pts} <- points, pts != [] do
          %{color: color, points: Enum.map_join(pts, " ", fn {x, y} -> "#{r(scale_x.(x))},#{r(scale_y.(y))}" end)}
        end

      x_ticks =
        (if assigns.log_x, do: [1, 5, 10, 25, 50, 100, 250, 500, 1000], else: Enum.map(0..4, &max(1, round(&1 * max_x / 4))))
        |> Enum.filter(&(&1 <= max_x))
        |> Enum.uniq()
        |> Enum.map(&%{label: &1, x: r(scale_x.(&1))})

      y_ticks =
        for i <- 0..4 do
          v = min_y + i / 4 * range_y
          %{label: if(assigns.percent, do: pct(v), else: Float.round(v, 3)), y: r(scale_y.(v))}
        end

      assigns =
        assign(assigns,
          width: width,
          height: height,
          pad_l: pad_l,
          plot_w: plot_w,
          pad_t: pad_t,
          plot_h: plot_h,
          lines: lines,
          x_ticks: x_ticks,
          y_ticks: y_ticks
        )

      ~H"""
      <svg viewBox={"0 0 #{@width} #{@height}"} class="w-full h-auto" role="img" aria-label={"#{@field} by epoch"}>
        <line :for={t <- @y_ticks} x1={@pad_l} y1={t.y} x2={@pad_l + @plot_w} y2={t.y} stroke="currentColor" stroke-opacity="0.1" />
        <text :for={t <- @y_ticks} x={@pad_l - 6} y={t.y + 3} text-anchor="end" font-size="10" class="fill-base-content/50">{t.label}</text>
        <text :for={t <- @x_ticks} x={t.x} y={@pad_t + @plot_h + 16} text-anchor="middle" font-size="10" class="fill-base-content/50">{t.label}</text>
        <polyline :for={l <- @lines} points={l.points} fill="none" stroke={l.color} stroke-width="1.5" stroke-linejoin="round" />
      </svg>
      """
    end
  end

  # ---------------------------------------------------------------------------
  # Formatting
  # ---------------------------------------------------------------------------

  defp describe(nil), do: ""

  defp describe(params) do
    sentences = if params["sentences"], do: "#{params["sentences"]} sentences", else: "all sentences"
    stop = if params["patience"], do: "stop after #{params["patience"]}", else: "no early stop"
    "#{sentences}, #{params["max_epochs"]} epochs, #{stop}, seed #{params["seed"]}"
  end

  defp best_text(%{"best" => %{"dev_accuracy" => acc, "epoch" => epoch}}), do: "#{pct(acc)} @ #{epoch}"
  defp best_text(_run), do: "—"

  # The first epoch whose dev accuracy came within 0.5 and 0.1 points of the
  # run's best: where the run's gains flattened out.
  defp plateau_text(%{"best" => %{"dev_accuracy" => best}, "curve" => curve}) do
    first_within = fn margin ->
      Enum.find_value(curve, "—", fn p -> if is_number(p["dev_accuracy"]) and p["dev_accuracy"] >= best - margin, do: p["epoch"] end)
    end

    "#{first_within.(0.005)} / #{first_within.(0.001)}"
  end

  defp plateau_text(_run), do: "—"

  defp status_class("running"), do: "badge-info"
  defp status_class("completed"), do: "badge-success"
  defp status_class("failed"), do: "badge-error"
  defp status_class(_), do: "badge-ghost"

  defp pct(nil), do: "—"
  defp pct(x), do: "#{Float.round(x * 100, 2)}%"

  defp duration(ms) do
    s = div(ms, 1000)
    h = div(s, 3600)
    m = div(rem(s, 3600), 60)
    if h > 0, do: "#{h}h #{m}m", else: "#{m}m #{rem(s, 60)}s"
  end

  defp r(x), do: Float.round(x * 1.0, 1)
end
