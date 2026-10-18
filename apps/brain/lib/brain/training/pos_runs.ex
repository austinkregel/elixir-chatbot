defmodule Brain.Training.POSRuns do
  @moduledoc """
  Recorded POS training runs: each run's settings, its learning curve and
  model snapshots, kept on disk so a run of hundreds of epochs can be
  watched, compared with others and used after the app restarts.

  A run lives in `<training_runs_path>/pos/<id>/`:

  - `run.json` -- settings, fixture provenance, status and the per-epoch
    curve (training loss, dev accuracy). Rewritten after every epoch.
  - `best.term` -- the model at the best dev epoch so far.
  - `epoch-<n>.term` -- the model after epoch `n`, for each of
    `snapshot_epochs/0` the run reaches, and after its final epoch.

  A run is measured on the EWT test split only when one of its snapshots is
  promoted (`promote!/3`), so the test split is read once per model chosen
  rather than once per epoch.

  Statuses: `running`, `completed`, `failed`, `cancelled`, and
  `interrupted` for a run that was running when the app stopped.
  """

  alias Brain.ML.POSTagger
  alias Brain.Training.POS

  @snapshot_epochs [5, 10, 25, 50, 100, 250, 500, 1000]

  @doc "The epochs at which every run saves a model snapshot."
  def snapshot_epochs, do: @snapshot_epochs

  @doc "Where POS runs are kept: `<training_runs_path>/pos`."
  @spec root() :: Path.t()
  def root do
    :brain |> Application.fetch_env!(:ml) |> Keyword.fetch!(:training_runs_path) |> Path.join("pos")
  end

  @doc "A new run id: the UTC start time and a random suffix."
  @spec new_id() :: String.t()
  def new_id do
    stamp = DateTime.utc_now() |> Calendar.strftime("%Y%m%dT%H%M%SZ")
    stamp <> "-" <> Base.encode16(:crypto.strong_rand_bytes(3), case: :lower)
  end

  @doc """
  Trains a tagger and records the run.

  ## Params

  - `:sentences` -- train on the first `n` sentences of the train and dev
    splits; `nil` for all of them.
  - `:max_epochs` -- epochs to run.
  - `:patience` -- stop after this many epochs without a dev improvement;
    `nil` never stops early.
  - `:seed` -- training seed.

  ## Options

  - `:id` -- the run id (default `new_id/0`).
  - `:root` -- where runs are kept (default `root/0`).
  - `:on_epoch` -- `fn run_id, progress -> any end`, called after each epoch
    once `run.json` records it.

  Returns the finished run. A run that raises is recorded as `failed` with
  the error before the error propagates.
  """
  @spec run!(map(), keyword()) :: map()
  def run!(params, opts \\ []) do
    params = params!(params)
    id = Keyword.get_lazy(opts, :id, &new_id/0)
    dir = Path.join(Keyword.get_lazy(opts, :root, &root/0), id)
    on_epoch = Keyword.get(opts, :on_epoch, fn _id, _progress -> :ok end)

    if File.exists?(dir), do: raise(ArgumentError, "POSRuns: run #{id} already exists at #{dir}")
    File.mkdir_p!(dir)

    run = %{
      "id" => id,
      "model" => "pos_tagger",
      "status" => "running",
      "started_at" => now(),
      "finished_at" => nil,
      "params" => Map.new(params, fn {k, v} -> {to_string(k), v} end),
      "inputs" => Enum.map(POS.inputs(), &Map.new(&1, fn {k, v} -> {to_string(k), v} end)),
      "curve" => [],
      "snapshots" => [],
      "best" => nil,
      "promotions" => [],
      "error" => nil
    }

    write!(dir, run)

    try do
      train = POS.load_split!("train", params.sentences)
      dev = POS.load_split!("dev", params.sentences)

      {:ok, model} =
        POSTagger.train(train,
          dev: dev,
          seed: params.seed,
          config: [max_epochs: params.max_epochs, patience: params.patience],
          inputs: POS.inputs(),
          on_epoch: fn progress, model_at ->
            record_epoch!(dir, progress, model_at, params.max_epochs)
            on_epoch.(id, progress)
          end
        )

      update!(dir, fn run ->
        final = "epoch-#{model.training.epochs_run}"

        run
        |> Map.put("status", "completed")
        |> Map.put("finished_at", now())
        |> Map.put("snapshots", Enum.uniq(run["snapshots"] ++ [final]))
      end)
    rescue
      e ->
        update!(dir, fn run ->
          run |> Map.put("status", "failed") |> Map.put("finished_at", now()) |> Map.put("error", Exception.message(e))
        end)

        reraise e, __STACKTRACE__
    end
  end

  @doc "Every recorded run, newest first."
  @spec list(Path.t()) :: [map()]
  def list(root \\ root()) do
    case File.ls(root) do
      {:ok, ids} -> ids |> Enum.sort(:desc) |> Enum.map(&get!(&1, root))
      {:error, :enoent} -> []
      {:error, reason} -> raise "POSRuns: cannot list #{root}: #{:file.format_error(reason)}"
    end
  end

  @doc "The recorded run `id`. Raises when there is none."
  @spec get!(String.t(), Path.t()) :: map()
  def get!(id, root \\ root()) do
    path = Path.join([root, id, "run.json"])

    case File.read(path) do
      {:ok, body} -> Jason.decode!(body)
      {:error, reason} -> raise "POSRuns: cannot read run #{id} at #{path}: #{:file.format_error(reason)}"
    end
  end

  @doc """
  Records that run `id` ended as `status` (`cancelled`, `interrupted`, or
  `failed` with `error`) outside `run!/2`: its process was stopped or died
  without raising.
  """
  @spec mark!(String.t(), String.t(), Path.t(), String.t() | nil) :: map()
  def mark!(id, status, root \\ root(), error \\ nil) when status in ["cancelled", "interrupted", "failed"] do
    update!(Path.join(root, id), fn run ->
      run |> Map.put("status", status) |> Map.put("finished_at", now()) |> Map.put("error", error)
    end)
  end

  @doc """
  Marks every run still recorded as `running` as `interrupted`. Called when
  the training server starts: no run survives the process that ran it.
  """
  @spec mark_interrupted!(Path.t()) :: [String.t()]
  def mark_interrupted!(root \\ root()) do
    for %{"status" => "running", "id" => id} <- list(root) do
      mark!(id, "interrupted", root)
      id
    end
  end

  @doc """
  Measures snapshot `snapshot` (`"best"` or `"epoch-<n>"`) of run `id` on the
  EWT test split and saves it as the `:test` model (the one the test suite
  loads) or the `:production` model. The evaluation is attached and the
  promotion recorded in the run.

  Raises, saving nothing, when the snapshot does not beat the lookup
  baseline learned from the sentences it trained on.

  ## Options

  - `:root` -- where runs are kept (default `root/0`).
  - `:to` -- save here instead of the target's configured path.
  """
  @spec promote!(String.t(), String.t(), :test | :production, keyword()) :: {Path.t(), map()}
  def promote!(id, snapshot, target, opts \\ []) when target in [:test, :production] do
    root = Keyword.get_lazy(opts, :root, &root/0)
    run = get!(id, root)

    unless snapshot in run["snapshots"],
      do: raise(ArgumentError, "POSRuns: run #{id} has no snapshot #{inspect(snapshot)}; it has #{inspect(run["snapshots"])}")

    {:ok, model} = POSTagger.load_model(Path.join([root, id, snapshot <> ".term"]))
    train = POS.load_split!("train", run["params"]["sentences"])
    model = POS.measure!(model, train, POS.load_split!("test"))

    out = Keyword.get_lazy(opts, :to, fn -> target_path(target) end)
    {:ok, path} = POSTagger.save_model(model, out)

    update!(Path.join(root, id), fn run ->
      promotion = %{
        "target" => to_string(target),
        "snapshot" => snapshot,
        "path" => path,
        "test_accuracy" => model.evaluation.accuracy,
        "lookup_baseline" => model.evaluation.lookup_baseline,
        "at" => now()
      }

      Map.update!(run, "promotions", &(&1 ++ [promotion]))
    end)

    {path, model}
  end

  @doc "Where a promoted model is saved: the test suite's model, or the one the app serves."
  @spec target_path(:test | :production) :: Path.t()
  def target_path(:test), do: :brain |> Application.fetch_env!(:ml) |> Keyword.fetch!(:pos_test_model_path)
  def target_path(:production), do: POSTagger.model_path()

  # ---------------------------------------------------------------------------

  @doc """
  The run params `run!/2` accepts, as an atom-keyed map, from atom- or
  string-keyed input. Raises ArgumentError for an unknown, missing or
  invalid param.
  """
  @spec params!(map() | keyword()) :: map()
  def params!(params) do
    params = Map.new(params, fn {k, v} -> {atom_key!(k), v} end)

    checks = [
      sentences: &(is_nil(&1) or (is_integer(&1) and &1 > 0)),
      max_epochs: &(is_integer(&1) and &1 > 0),
      patience: &(is_nil(&1) or (is_integer(&1) and &1 > 0)),
      seed: &is_integer/1
    ]

    for {key, valid?} <- checks do
      unless Map.has_key?(params, key) and valid?.(Map.fetch!(params, key)) do
        raise ArgumentError, "POSRuns: #{key} is missing or invalid: #{inspect(Map.get(params, key))}"
      end
    end

    Map.take(params, Keyword.keys(checks))
  end

  defp atom_key!(key) when key in [:sentences, :max_epochs, :patience, :seed], do: key
  defp atom_key!(key) when key in ["sentences", "max_epochs", "patience", "seed"], do: String.to_existing_atom(key)
  defp atom_key!(key), do: raise(ArgumentError, "POSRuns: unknown param #{inspect(key)}")

  defp record_epoch!(dir, progress, model_at, max_epochs) do
    snapshot? = progress.epoch in @snapshot_epochs or progress.epoch == max_epochs
    model = if progress.improved? or snapshot?, do: model_at.()

    if progress.improved?, do: {:ok, _} = POSTagger.save_model(model, Path.join(dir, "best.term"))
    snapshot = "epoch-#{progress.epoch}"
    if snapshot?, do: {:ok, _} = POSTagger.save_model(model, Path.join(dir, snapshot <> ".term"))

    update!(dir, fn run ->
      point = Map.new(progress, fn {k, v} -> {k |> to_string() |> String.trim_trailing("?"), v} end)

      snapshots =
        run["snapshots"]
        |> then(&if(progress.improved? and "best" not in &1, do: ["best" | &1], else: &1))
        |> then(&if(snapshot?, do: &1 ++ [snapshot], else: &1))

      best =
        if progress.improved?,
          do: %{"epoch" => progress.epoch, "dev_accuracy" => progress.dev_accuracy},
          else: run["best"]

      %{run | "curve" => run["curve"] ++ [point], "snapshots" => snapshots, "best" => best}
    end)
  end

  defp update!(dir, fun) do
    run = dir |> Path.join("run.json") |> File.read!() |> Jason.decode!() |> fun.()
    write!(dir, run)
    run
  end

  # Written to a temporary file and renamed, so a reader never sees half a file.
  defp write!(dir, run) do
    path = Path.join(dir, "run.json")
    tmp = path <> ".tmp"
    File.write!(tmp, Jason.encode!(run, pretty: true))
    File.rename!(tmp, path)
  end

  defp now, do: DateTime.utc_now() |> DateTime.truncate(:second) |> DateTime.to_iso8601()
end
