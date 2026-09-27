defmodule Brain.Training.POS do
  @moduledoc """
  Trains, measures and saves the POS tagger from the committed EWT fixtures.
  `mix pos.train`, `mix train` and `mix train_models` call
  `train_and_save!/1`. Recorded runs from the POS training page
  (`/training/pos`, `Brain.Training.POSRuns`) load the same fixtures and pass
  the same gate through `load_split!/2` and `measure!/3`.

  1. Trains `Brain.ML.POSTagger` on the EWT train fixture, stopping on dev.
  2. Measures it on the held-out test fixture.
  3. Refuses to save a model that does not beat the most-frequent-tag
     baseline on the same test split (each word's most common training tag,
     ignoring context). A tagger that cannot beat a lookup table is broken;
     the tagger this replaced scored 68% against the lookup's 86%.
  4. Saves the model with the evaluation attached -- accuracy, accuracy on
     words outside the training vocabulary, per-tag recall and precision, and
     the confusion matrix -- and the fixtures' paths and SHA-256, so a model
     can be traced to the exact data it learned from.
  """

  alias Brain.ML.POSTagger
  alias Brain.Training.Fixture

  @fixture_dir "training/pos"
  @splits ~w(train dev test)

  @doc """
  Trains, measures and saves the tagger. Returns `{path, model}`; the
  evaluation is `model.evaluation`. Raises, saving nothing, when a fixture is
  invalid or the model does not beat the lookup baseline.

  ## Options

  - `:out` -- where to save (default: the configured POS model path).
  - `:config` -- overrides for `config :brain, :pos_tagger`.
  - `:limit` -- use only the first `n` sentences of each split, for a quick
    model (tests); the gate still applies.
  """
  @spec train_and_save!(keyword()) :: {Path.t(), map()}
  def train_and_save!(opts \\ []) do
    limit = Keyword.get(opts, :limit)
    [train, dev, test] = Enum.map(@splits, &load_split!(&1, limit))

    {:ok, model} =
      POSTagger.train(train,
        dev: dev,
        config: Keyword.get(opts, :config, []),
        inputs: inputs()
      )

    model = measure!(model, train, test)
    {:ok, path} = POSTagger.save_model(model, Keyword.get(opts, :out))
    {path, model}
  end

  @doc """
  The sequences of one EWT split (`"train"`, `"dev"` or `"test"`), validated
  by `Brain.Training.Fixture.load!/1`; only the first `limit` when given.
  """
  @spec load_split!(String.t(), pos_integer() | nil) :: [map()]
  def load_split!(split, limit \\ nil) do
    sequences = fixture_paths() |> Map.fetch!(split) |> Fixture.load!() |> Fixture.pos_sequences()
    if limit, do: Enum.take(sequences, limit), else: sequences
  end

  @doc "Provenance of the fixtures, as recorded in a model's `training.inputs`."
  @spec inputs() :: [map()]
  def inputs, do: Enum.map(@splits, &input(Map.fetch!(fixture_paths(), &1)))

  @doc """
  Measures `model` on the `test` sequences against the lookup baseline
  learned from `train`, the sentences it trained on, and attaches the
  evaluation. Raises when the model does not beat the baseline.
  """
  @spec measure!(map(), [map()], [map()]) :: map()
  def measure!(model, train, test) do
    report = POSTagger.evaluate(test, model)
    baseline = lookup_baseline(train, test)

    if report.accuracy <= baseline do
      raise "POS training: test accuracy #{pct(report.accuracy)} does not beat the " <>
              "most-frequent-tag baseline #{pct(baseline)}; the model was not saved"
    end

    %{model | evaluation: Map.merge(report, %{split: "ud_ewt.test", lookup_baseline: baseline})}
  end

  @doc """
  Raises unless `model` was trained on the fixtures as they are now and
  carries an evaluation that beats its lookup baseline: a model trained on
  older fixtures, or never measured, is not the model the tests describe.
  """
  @spec check_current!(map(), Path.t()) :: :ok
  def check_current!(model, path) do
    recorded = model |> get_in([:training, :inputs]) |> List.wrap() |> Map.new(&{&1.name, &1.sha256})
    current = Map.new(inputs(), &{&1.name, &1.sha256})

    unless recorded == current do
      raise "POS model #{path} was trained on fixtures #{inspect(recorded)}, but they are now " <>
              "#{inspect(current)}. Train and promote a new model on the POS training page."
    end

    case model.evaluation do
      %{accuracy: accuracy, lookup_baseline: baseline} when accuracy > baseline ->
        :ok

      other ->
        raise "POS model #{path} has no evaluation beating its lookup baseline (#{inspect(other && Map.take(other, [:accuracy, :lookup_baseline]))})"
    end
  end

  @doc "The EWT fixture path for each split."
  @spec fixture_paths() :: %{String.t() => Path.t()}
  def fixture_paths do
    dir = Brain.priv_path(@fixture_dir)
    Map.new(@splits, &{&1, Path.join(dir, "ud_ewt.#{&1}.json")})
  end

  @doc """
  Accuracy of tagging each test word with its most frequent training tag
  (lowercased), and an unseen word with the most frequent tag overall.
  """
  @spec lookup_baseline([map()], [map()]) :: float()
  def lookup_baseline(train, test) do
    pairs = Enum.flat_map(train, &Enum.zip(Enum.map(&1.tokens, fn t -> String.downcase(t) end), &1.tags))

    best =
      pairs
      |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
      |> Map.new(fn {w, tags} -> {w, tags |> Enum.frequencies() |> Enum.max_by(&elem(&1, 1)) |> elem(0)} end)

    fallback = pairs |> Enum.frequencies_by(&elem(&1, 1)) |> Enum.max_by(&elem(&1, 1)) |> elem(0)
    test_pairs = Enum.flat_map(test, &Enum.zip(&1.tokens, &1.tags))
    correct = Enum.count(test_pairs, fn {w, g} -> Map.get(best, String.downcase(w), fallback) == g end)
    correct / length(test_pairs)
  end

  defp input(path) do
    sha = :crypto.hash(:sha256, File.read!(path)) |> Base.encode16(case: :lower)
    %{name: Path.basename(path), version: "repo:apps/brain/priv/#{@fixture_dir}/#{Path.basename(path)}", sha256: sha}
  end

  defp pct(x), do: "#{Float.round(x * 100, 2)}%"
end
