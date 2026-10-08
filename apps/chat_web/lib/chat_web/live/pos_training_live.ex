defmodule ChatWeb.POSTrainingLive do
  @moduledoc """
  Isolation page for training the POS tagger.

  Start a recorded run (`Brain.Training.POSRuns`) through
  `Brain.ML.TrainingServer`, watch its learning curve epoch by epoch, compare
  runs to see where more epochs stop paying, and promote a snapshot to the
  test suite's model or to the model the app serves. A promotion that
  overwrites a model file confirms first, because no copy of the replaced
  model is kept.

  Runs are read from disk, so they outlive the page and the app; a run can
  be left going and looked at later.
  """

  use ChatWeb, :live_view

  import ChatWeb.AppShell

  alias Brain.ML.POSTagger
  alias Brain.ML.POSTagger.LexicalFeatures
  alias Brain.ML.TrainingServer
  alias Brain.Training.POSRuns

  # Imperatives the treebank barely teaches. Whether the tagger heads these with
  # a VERB is the thing this page cannot show from a learning curve.
  @default_probe "switch off the heating and dim the lamp"

  # Compared runs are categories with no meaning of their own, and the design
  # language reserves every hue for a meaning. So every series is drawn in the
  # data-mark blue and told apart by its stroke pattern and its marker shape,
  # each repeated in the legend beside the run's id. Patterns vary fastest, so
  # the first four series differ in line before they differ in marker.
  @series_dashes [nil, "8 4", "2 3", "10 3 2 3"]
  @series_markers [:filled_circle, :hollow_square, :filled_triangle, :hollow_diamond]
  @compared_by_default 4

  # How many runs the curve chart draws distinctly: one per pattern and marker
  # pair. The compare control stops at this many; `series_style/1` raises past
  # it, as the guard behind the control.
  @max_compared length(@series_dashes) * length(@series_markers)

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
     |> assign(:pending_promotion, nil)
     |> assign(:open_confirm, nil)
     |> assign(:confirm_error, nil)
     |> assign(:left_out_run, nil)
     |> assign(:probe, to_form(%{"text" => @default_probe}, as: :probe))
     |> assign(:tagged, nil)
     |> assign(:tag_error, nil)
     |> assign(:form, to_form(default_params(), as: :run))}
  end

  defp shares(by_pos) do
    ~w(noun verb adj adv)
    |> Enum.map_join("/", fn p -> by_pos |> Map.get(p, 0.0) |> pct0() end)
  end

  defp pct0(x), do: "#{round((x || 0.0) * 100)}"
  defp pct1(x), do: "#{Float.round((x || 0.0) * 1.0, 2)}"

  defp tag_tokens(text) do
    tokens = text |> Brain.ML.Tokenizer.tokenize() |> Enum.map(& &1.text)

    cond do
      tokens == [] ->
        {:error, "Nothing to tag."}

      true ->
        case POSTagger.load_model() do
          {:ok, model} ->
            tags = POSTagger.predict_tags(tokens, model)

            rows =
              Enum.zip(tokens, tags)
              |> Enum.map(fn {token, tag} ->
                lex = Map.new(LexicalFeatures.explain(token))

                %{
                  token: token,
                  tag: tag,
                  known: lex[:known_to_lexicon] == 1.0,
                  present: Enum.filter(~w(noun verb adj adv), &(lex[:"has_#{&1}"] == 1.0)),
                  senses: Map.new(~w(noun verb adj adv), &{&1, lex[:"sense_share_#{&1}"]}),
                  freqs: Map.new(~w(noun verb adj adv), &{&1, lex[:"freq_share_#{&1}"]}),
                  has_frequency: lex[:has_frequency] == 1.0,
                  polysemy: lex[:polysemy],
                  closed: Enum.filter(Brain.Lexicon.ClosedClass.all_classes(), &(lex[:"closed_#{String.downcase(&1)}"] == 1.0))
                }
              end)

            {:ok, rows}

          {:error, reason} ->
            {:error, "No usable POS model: #{reason}"}
        end
    end
  rescue
    e -> {:error, Exception.message(e)}
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
       |> compare_new_run(id)
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
    left_out = if socket.assigns.left_out_run == id, do: nil, else: socket.assigns.left_out_run
    {:noreply, socket |> assign(:compared, compared) |> assign(:left_out_run, left_out)}
  end

  def handle_event("toggle_log_x", _params, socket) do
    {:noreply, assign(socket, :log_x, not socket.assigns.log_x)}
  end

  # A learning curve cannot answer "is `switch` a verb here". Test accuracy over
  # 25,094 tokens moves by 0.02% when a word like `dim` goes from wrong to right
  # on all of its twelve occurrences, so the per-token view is the only place the
  # lexical channel's effect is visible.
  def handle_event("tag", %{"probe" => %{"text" => text}}, socket) do
    socket = assign(socket, :probe, to_form(%{"text" => text}, as: :probe))

    case tag_tokens(text) do
      {:ok, rows} -> {:noreply, socket |> assign(:tagged, rows) |> assign(:tag_error, nil)}
      {:error, message} -> {:noreply, socket |> assign(:tagged, nil) |> assign(:tag_error, message)}
    end
  end

  # A promotion writes the target model file with no copy of what was there,
  # so one that overwrites a file is a removal and confirms first. Either
  # target confirms when its model file exists, and acts at once when there
  # is nothing there to overwrite.
  def handle_event("promote", %{"run_id" => id, "snapshot" => snapshot, "target" => target}, socket)
      when target in ["test", "production"] do
    promotion = {id, snapshot, String.to_existing_atom(target)}

    if confirms_promotion?(elem(promotion, 2)) do
      {:noreply,
       socket
       |> assign(:pending_promotion, promotion)
       |> assign(:open_confirm, promote_confirm_id(id))
       |> assign(:confirm_error, nil)}
    else
      {:noreply, start_promotion(socket, promotion)}
    end
  end

  def handle_event("confirm_promote", _params, socket) do
    case {socket.assigns.pending_promotion, socket.assigns.promoting} do
      {nil, _} ->
        raise ArgumentError,
              "ChatWeb.POSTrainingLive: confirm_promote arrived with no promotion awaiting confirmation."

      {_pending, {id, snapshot, _target}} ->
        {:noreply,
         assign(socket, :confirm_error, "#{snapshot} of #{id} is still being measured. Confirm again when it finishes.")}

      {pending, nil} ->
        {:noreply, socket |> assign(:confirm_error, nil) |> start_promotion(pending)}
    end
  end

  def handle_event("close_confirm", _params, socket) do
    {:noreply, close_confirm(socket)}
  end

  @impl true
  def handle_async(:promote, {:ok, {path, model}}, socket) do
    {_id, snapshot, target} = promoted = socket.assigns.promoting
    e = model.evaluation

    {:noreply,
     socket
     |> assign(:promoting, nil)
     |> close_confirm_for(promoted)
     |> reload_runs()
     |> put_flash(
       :info,
       "#{snapshot} promoted to the #{target} model (#{path}): test accuracy #{pct(e.accuracy)}, lookup baseline #{pct(e.lookup_baseline)}"
     )}
  end

  # A confirmed promotion that fails shows the failure in its confirmation
  # panel, which stays open; one that needed no confirmation reports in a flash.
  def handle_async(:promote, {:exit, reason}, socket) do
    message =
      case reason do
        {exception, _stack} when is_exception(exception) -> Exception.message(exception)
        other -> inspect(other)
      end

    promoted = socket.assigns.promoting
    socket = assign(socket, :promoting, nil)

    if socket.assigns.pending_promotion == promoted do
      {:noreply, assign(socket, :confirm_error, "Promotion refused: #{message}")}
    else
      {:noreply, put_flash(socket, :error, "Promotion refused: #{message}")}
    end
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

  # A started run joins the comparison unless the chart already draws as many
  # runs as it can tell apart; then it is left out, and the cap note says so.
  defp compare_new_run(socket, id) do
    if MapSet.size(socket.assigns.compared) >= @max_compared do
      assign(socket, :left_out_run, id)
    else
      assign(socket, :compared, MapSet.put(socket.assigns.compared, id))
    end
  end

  defp start_promotion(socket, {id, snapshot, target} = promotion) do
    socket
    |> assign(:promoting, promotion)
    |> start_async(:promote, fn -> POSRuns.promote!(id, snapshot, target) end)
  end

  defp confirms_promotion?(target), do: File.exists?(POSRuns.target_path(target))

  defp promote_confirm_id(run_id), do: "confirm-promote-" <> run_id

  defp close_confirm(socket) do
    socket
    |> assign(:open_confirm, nil)
    |> assign(:confirm_error, nil)
    |> assign(:pending_promotion, nil)
  end

  defp close_confirm_for(socket, promotion) do
    if socket.assigns.pending_promotion == promotion, do: close_confirm(socket), else: socket
  end

  # The confirmation for the promotion awaiting it: what it overwrites and
  # what is lost.
  defp promotion_confirm(nil), do: nil

  defp promotion_confirm({id, snapshot, target}) do
    path = model_file(target)
    role = if target == :production, do: "the model the app serves", else: "the model the test suite loads"

    consequence =
      if File.exists?(POSRuns.target_path(target)),
        do:
          "Measures snapshot #{snapshot} of run #{id} on the test split, then writes it over #{path}, #{role}. " <>
            "The model there now is overwritten and no copy of it is kept.",
        else:
          "Measures snapshot #{snapshot} of run #{id} on the test split, then writes it to #{path}, #{role}. " <>
            "No model file is there now."

    %{
      run_id: id,
      id: promote_confirm_id(id),
      verb: if(target == :production, do: "Promote to production", else: "Promote to test"),
      target: path,
      consequence: consequence,
      trigger_id: "promote-#{target}-#{id}"
    }
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
      |> Enum.map(fn {run, i} -> Map.put(series_style(i), :run, run) end)

    assigns =
      assign(assigns,
        current: current,
        series: series,
        compare_full: MapSet.size(assigns.compared) >= @max_compared,
        max_compared: @max_compared,
        promotion_confirm: promotion_confirm(assigns.pending_promotion)
      )

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
          <h1 class="text-title text-ink">POS Training</h1>
          <p class="text-body text-ink-muted">
            Train the part-of-speech tagger on EWT, watch where more epochs stop paying, promote a snapshot
          </p>
        </div>
      </:page_header>

      <div class="p-space-lg space-y-space-lg">
        <.card>
          <.card_body class="space-y-space-sm">
            <h2 class="text-heading text-ink">Tag a sentence</h2>
            <p class="text-caption text-ink-muted">
              The tag the model serves, beside what the lexicon knows about each token.
              A learning curve cannot show whether <code class="text-ref">switch</code> heads a command as a verb.
            </p>

            <.form for={@probe} id="pos-probe-form" phx-submit="tag" class="flex gap-space-sm items-end">
              <div class="grow">
                <.text_input
                  name="probe[text]"
                  value={@probe[:text].value}
                  placeholder="switch off the heating"
                />
              </div>
              <.btn type="submit" variant={:primary}>Tag</.btn>
            </.form>

            <.alert :if={@tag_error} variant={:warning}>{@tag_error}</.alert>

            <div :if={@tagged} class="overflow-x-auto">
              <table class="w-full text-left text-body-dense text-ink tabular-nums">
                <thead class="bg-surface-sunk">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">token</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">tag</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">lexicon</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">senses n/v/a/r</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">frequency n/v/a/r</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">poly</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <tr :for={row <- @tagged} class="even:bg-surface-sunk">
                    <td class="h-row-compact px-space-sm text-value text-ink">{row.token}</td>
                    <td class="h-row-compact px-space-sm">
                      <.badge variant={if row.tag == "VERB", do: :success, else: :default}>
                        {row.tag}
                      </.badge>
                    </td>
                    <td class="h-row-compact px-space-sm">
                      <span :if={not row.known} class="text-ink-muted">unknown</span>
                      <span :if={row.known and row.closed != []} class="text-value">
                        closed: {Enum.join(row.closed, " ")}
                      </span>
                      <span :if={row.known and row.closed == [] and row.present != []} class="text-value">
                        {Enum.join(row.present, " ")}
                      </span>
                      <span :if={row.known and row.closed == [] and row.present == []} class="text-ink-muted">
                        no senses
                      </span>
                    </td>
                    <td class="h-row-compact px-space-sm text-value text-ink">{shares(row.senses)}</td>
                    <td class="h-row-compact px-space-sm text-value text-ink">
                      <span :if={row.has_frequency}>{shares(row.freqs)}</span>
                      <span :if={not row.has_frequency} class="text-ink-muted">none</span>
                    </td>
                    <td class="h-row-compact px-space-sm text-value text-ink">{pct1(row.polysemy)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-space-sm">
            <h2 class="text-heading text-ink">New run</h2>
            <.form for={@form} id="pos-run-form" phx-submit="start" class="grid grid-cols-2 md:grid-cols-5 gap-space-md items-end">
              <.input name="run[sentences]" label="Sentences (blank: all 12,544)" value={@form[:sentences].value} />
              <.input name="run[max_epochs]" label="Max epochs" value={@form[:max_epochs].value} />
              <.input name="run[patience]" label="Early stop after (blank: never)" value={@form[:patience].value} />
              <.input name="run[seed]" label="Seed" value={@form[:seed].value} />
              <.btn type="submit" variant={:primary} class="mb-space-sm" disabled={@current_run != nil}>Start run</.btn>
            </.form>
            <p class="text-caption text-ink-muted">
              Snapshots are kept at epochs {Enum.join(POSRuns.snapshot_epochs(), ", ")}, the last epoch, and the best on dev.
              Measured on this machine: about 12 s per epoch on 2,000 sentences, 50 s on all of them.
            </p>
          </.card_body>
        </.card>

        <.card :if={@current} id="current-run">
          <.card_body class="space-y-space-sm">
            <div class="flex items-center justify-between">
              <h2 class="text-heading text-ink">Running: <span class="text-value">{@current["id"]}</span></h2>
              <.btn
                id="cancel-run"
                variant={:outline}
                size={:sm}
                reach={:local}
                target={"#{@current["id"]}/run.json"}
                phx-click="cancel"
              >
                Cancel
              </.btn>
            </div>
            <.progress run={@current} />
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-space-sm">
            <div class="flex items-center justify-between">
              <h2 class="text-heading text-ink">Dev accuracy by epoch</h2>
              <.btn variant={:ghost} size={:xs} phx-click="toggle_log_x">
                Epoch axis: {if @log_x, do: "log", else: "linear"}
              </.btn>
            </div>
            <.curve_chart series={@series} field="dev_accuracy" log_x={@log_x} percent={true} />
            <h2 class="pt-space-sm text-heading text-ink">Training loss by epoch</h2>
            <.curve_chart series={@series} field="loss" log_x={@log_x} percent={false} />
            <div class="flex flex-wrap gap-space-md text-caption">
              <span :for={s <- @series} class="flex items-center gap-space-xs">
                <.legend_swatch style={s} />
                <span class="text-ref text-ink">{s.run["id"]}</span>
                <span class="text-ink-muted">{describe(s.run["params"])}</span>
              </span>
            </div>
          </.card_body>
        </.card>

        <.card>
          <.card_body class="space-y-space-sm">
            <h2 class="text-heading text-ink">Runs</h2>
            <p :if={@runs == []} class="text-body text-ink-muted">No runs yet.</p>
            <p :if={@compare_full} id="compare-limit" class="text-caption text-ink-muted">
              {@max_compared} runs compared, the most the chart draws distinctly. Untick one to compare another.
              <span :if={@left_out_run && not MapSet.member?(@compared, @left_out_run)}>
                Run <span class="text-ref">{@left_out_run}</span> started without joining the comparison.
              </span>
            </p>
            <div :if={@runs != []} class="overflow-x-auto">
              <table id="pos-runs" class="w-full text-left text-body-dense text-ink tabular-nums">
                <thead class="bg-surface-sunk">
                  <tr>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Compare</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Run</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Status</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Settings</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Epochs</th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Best dev</th>
                    <th
                      class="h-row-compact px-space-sm text-label text-ink-muted"
                      title="First epoch within this much of the run's best dev accuracy"
                    >
                      Within 0.5 / 0.1 pt of best
                    </th>
                    <th class="h-row-compact px-space-sm text-label text-ink-muted">Promote</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-border">
                  <tr :for={run <- @runs} id={"run-" <> run["id"]} class="even:bg-surface-sunk">
                    <td class="px-space-sm py-space-xs">
                      <input
                        type="checkbox"
                        class="size-4 accent-primary cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
                        checked={MapSet.member?(@compared, run["id"])}
                        disabled={@compare_full and not MapSet.member?(@compared, run["id"])}
                        aria-describedby={@compare_full && "compare-limit"}
                        phx-click="toggle_compare"
                        phx-value-id={run["id"]}
                      />
                    </td>
                    <td class="px-space-sm py-space-xs text-ref text-ink">{run["id"]}</td>
                    <td class="px-space-sm py-space-xs">
                      <.badge variant={status_variant(run["status"])}>{run["status"]}</.badge>
                    </td>
                    <td class="px-space-sm py-space-xs text-caption text-ink">{describe(run["params"])}</td>
                    <td class="px-space-sm py-space-xs text-value text-score-count">{length(run["curve"] || [])}</td>
                    <td class="px-space-sm py-space-xs text-value text-ink">{best_text(run)}</td>
                    <td class="px-space-sm py-space-xs text-value text-ink">{plateau_text(run)}</td>
                    <td class="px-space-sm py-space-xs">
                      <form :if={(run["snapshots"] || []) != []} phx-submit="promote" class="flex gap-space-xs items-start">
                        <input type="hidden" name="run_id" value={run["id"]} />
                        <.input type="select" name="snapshot" options={run["snapshots"]} value={nil} />
                        <.btn
                          id={"promote-test-" <> run["id"]}
                          name="target"
                          value="test"
                          reach={:local}
                          target={model_file(:test)}
                          disabled={@promoting != nil}
                        >
                          Test
                        </.btn>
                        <.btn
                          id={"promote-production-" <> run["id"]}
                          name="target"
                          value="production"
                          reach={:local}
                          target={model_file(:production)}
                          disabled={@promoting != nil}
                        >
                          Production
                        </.btn>
                      </form>
                      <.execute_confirm
                        :if={@promotion_confirm && @promotion_confirm.run_id == run["id"]}
                        id={@promotion_confirm.id}
                        open={@open_confirm == @promotion_confirm.id}
                        reach={:local}
                        removes
                        verb={@promotion_confirm.verb}
                        target={@promotion_confirm.target}
                        consequence={@promotion_confirm.consequence}
                        on_confirm="confirm_promote"
                        on_cancel="close_confirm"
                        trigger_id={@promotion_confirm.trigger_id}
                        error={@confirm_error}
                        class="mt-space-sm"
                      />
                      <div :for={p <- run["promotions"] || []} class="text-caption text-ink-muted">
                        {p["snapshot"]} → {p["target"]}: test {pct(p["test_accuracy"])} vs lookup {pct(p["lookup_baseline"])}
                      </div>
                      <div :if={run["error"]} class="text-caption text-red">{run["error"]}</div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p :if={@promoting} class="text-body text-ink-muted">
              Measuring {elem(@promoting, 1)} of {elem(@promoting, 0)} on the test split...
            </p>
          </.card_body>
        </.card>
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
    <div class="grid grid-cols-2 md:grid-cols-5 gap-space-sm text-body text-ink">
      <div>Epoch <strong class="text-value-strong">{if @last, do: @last["epoch"], else: 0}</strong> / {@max_epochs}</div>
      <div>Dev <strong class="text-value-strong">{if @last, do: pct(@last["dev_accuracy"]), else: "—"}</strong></div>
      <div>Best <strong class="text-value-strong">{best_text(@run)}</strong></div>
      <div>Loss <strong class="text-value-strong">{if @last, do: Float.round(@last["loss"], 4), else: "—"}</strong></div>
      <div>Left <strong class="text-value-strong">{if @eta, do: duration(@eta), else: "—"}</strong></div>
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
        {s, for(p <- s.run["curve"] || [], is_number(p[assigns.field]), do: {p["epoch"], p[assigns.field]})}
      end)

    all = Enum.flat_map(points, &elem(&1, 1))

    if length(all) < 2 do
      assigns = assign(assigns, :message, "Pick a run with at least two epochs to plot.")

      ~H"""
      <div class="py-space-lg text-center text-body text-ink-muted">{@message}</div>
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

      tick_epochs =
        (if assigns.log_x, do: [1, 5, 10, 25, 50, 100, 250, 500, 1000], else: Enum.map(0..4, &max(1, round(&1 * max_x / 4))))
        |> Enum.filter(&(&1 <= max_x))
        |> Enum.uniq()

      x_ticks = Enum.map(tick_epochs, &%{label: &1, x: r(scale_x.(&1))})

      # Each series carries its marker at the tick epochs it reached and at its
      # last epoch, so the marker identifies the line without crowding it.
      lines =
        for {s, pts} <- points, pts != [] do
          {last_epoch, _} = List.last(pts)

          %{
            dash: s.dash,
            marker: s.marker,
            points: Enum.map_join(pts, " ", fn {x, y} -> "#{r(scale_x.(x))},#{r(scale_y.(y))}" end),
            markers:
              for({x, y} <- pts, x in tick_epochs or x == last_epoch, do: %{x: r(scale_x.(x)), y: r(scale_y.(y))})
          }
        end

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
        <line :for={t <- @y_ticks} x1={@pad_l} y1={t.y} x2={@pad_l + @plot_w} y2={t.y} stroke="var(--border)" />
        <text :for={t <- @y_ticks} x={@pad_l - 6} y={t.y + 3} text-anchor="end" font-size="10" class="fill-ink-muted">{t.label}</text>
        <text :for={t <- @x_ticks} x={t.x} y={@pad_t + @plot_h + 16} text-anchor="middle" font-size="10" class="fill-ink-muted">{t.label}</text>
        <g :for={l <- @lines}>
          <polyline
            points={l.points}
            fill="none"
            stroke="var(--blue)"
            stroke-width="1.5"
            stroke-linejoin="round"
            stroke-dasharray={l.dash}
          />
          <.series_marker :for={m <- l.markers} shape={l.marker} x={m.x} y={m.y} />
        </g>
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

  @status_variants %{
    "running" => :info,
    "completed" => :success,
    "failed" => :error,
    "cancelled" => :default,
    "interrupted" => :default
  }

  defp status_variant(status) do
    case Map.fetch(@status_variants, status) do
      {:ok, variant} ->
        variant

      :error ->
        raise ArgumentError,
              "ChatWeb.POSTrainingLive: no badge for run status #{inspect(status)}. " <>
                "The statuses are #{inspect(Map.keys(@status_variants))}."
    end
  end

  # The model file a promotion overwrites, as its reach badge names it.
  defp model_file(target), do: target |> POSRuns.target_path() |> Path.relative_to_cwd()

  defp pct(nil), do: "—"
  defp pct(x), do: "#{Float.round(x * 100, 2)}%"

  defp duration(ms) do
    s = div(ms, 1000)
    h = div(s, 3600)
    m = div(rem(s, 3600), 60)
    if h > 0, do: "#{h}h #{m}m", else: "#{m}m #{rem(s, 60)}s"
  end

  defp r(x), do: Float.round(x * 1.0, 1)

  # The i-th compared run's line pattern and marker. There are as many distinct
  # styles as patterns times markers; past that two runs would be drawn alike
  # and could not be told apart, so it raises instead.
  defp series_style(i) do
    dashes = length(@series_dashes)

    case Enum.fetch(@series_markers, div(i, dashes)) do
      {:ok, marker} ->
        %{dash: Enum.at(@series_dashes, rem(i, dashes)), marker: marker}

      :error ->
        raise ArgumentError,
              "ChatWeb.POSTrainingLive: #{i + 1} runs are compared, and the curve chart can draw " <>
                "#{dashes * length(@series_markers)} runs distinctly. Untick a run to compare fewer."
    end
  end

  attr :style, :map, required: true

  defp legend_swatch(assigns) do
    ~H"""
    <svg viewBox="0 0 32 12" class="h-3 w-8 shrink-0 overflow-visible" aria-hidden="true">
      <line x1="0" y1="6" x2="32" y2="6" stroke="var(--blue)" stroke-width="1.5" stroke-dasharray={@style.dash} />
      <.series_marker shape={@style.marker} x={16} y={6} />
    </svg>
    """
  end

  attr :shape, :atom, required: true
  attr :x, :any, required: true
  attr :y, :any, required: true

  defp series_marker(%{shape: :filled_circle} = assigns) do
    ~H"""
    <circle cx={@x} cy={@y} r="3" fill="var(--blue)" />
    """
  end

  defp series_marker(%{shape: :hollow_square} = assigns) do
    ~H"""
    <rect x={@x - 2.75} y={@y - 2.75} width="5.5" height="5.5" fill="var(--surface)" stroke="var(--blue)" stroke-width="1.5" />
    """
  end

  defp series_marker(%{shape: :filled_triangle} = assigns) do
    ~H"""
    <path d={"M#{@x} #{@y - 3.5} L#{@x + 3.5} #{@y + 3} L#{@x - 3.5} #{@y + 3} Z"} fill="var(--blue)" />
    """
  end

  defp series_marker(%{shape: :hollow_diamond} = assigns) do
    ~H"""
    <path
      d={"M#{@x} #{@y - 3.5} L#{@x + 3.5} #{@y} L#{@x} #{@y + 3.5} L#{@x - 3.5} #{@y} Z"}
      fill="var(--surface)"
      stroke="var(--blue)"
      stroke-width="1.5"
    />
    """
  end

  defp series_marker(%{shape: shape}) do
    raise ArgumentError,
          "ChatWeb.POSTrainingLive.series_marker/1: no marker for #{inspect(shape)}. " <>
            "The markers are #{inspect(@series_markers)}."
  end
end
