defmodule Brain.ML.FeatureVectorClassifier do
  @moduledoc """
  Dense centroid-based classifier that consumes numeric feature vectors
  (`list(float())`) and returns `{:ok, label, confidence, details}`.

  This classifier exists so the `Brain.Analysis.ChunkProfile` axis
  classifiers (`:intent_domain`, `:tense_class`, `:aspect_class`,
  `:urgency`, `:certainty_level`) cannot be biased by token identity in
  their training data: they never see text, only numeric features
  engineered by `Brain.Analysis.FeatureExtractor`.

  Compared to `Brain.ML.SimpleClassifier` (TF-IDF over raw tokens), this
  classifier skips vocabulary/IDF entirely and operates on pre-computed
  dense features. It is architecturally incapable of reacting to a
  specific proper noun, because proper nouns are represented in the
  feature vector only through aggregate dimensions (PROPN count,
  entity-type one-hot, lexical-domain fingerprint, etc.).

  ## Model shape

      %{
        kind: :feature_vector,
        input_dim: non_neg_integer(),
        label_centroids: %{String.t() => list(float())},
        weights: list(float()) | nil
      }

  Models are plain Elixir terms and round-trip through
  `:erlang.term_to_binary/1` cleanly, so they slot into the same
  on-disk format (`priv/ml_models/micro/<name>.term`) that
  `Brain.ML.MicroClassifiers` already loads.

  ## Per-Dimension Weighting

  When `weights` is present in the model, both training centroids and
  classification inputs are multiplied element-wise by the weight
  vector before cosine similarity. This allows the classifier to
  emphasize discriminative dimensions and suppress noisy ones.

  Models without weights (or with `weights: nil`) behave exactly as
  before — uniform weighting across all dimensions.

  ## Classification

  `classify/2` ranks every label centroid by cosine similarity against
  the (optionally weighted) input vector and returns the winner.
  Confidence is a monotone function of the margin between the winner
  and the runner-up, mapped into `[0.0, 1.0]`.

  An empty model, or an input vector whose dimensionality does not
  match the model's declared `input_dim`, returns `:error`.
  """

  @type training_example :: {list(float()), String.t()}

  @type model :: %{
          kind: :feature_vector,
          input_dim: non_neg_integer(),
          label_centroids: %{String.t() => list(float())},
          weights: list(float()) | nil
        }

  @type classification ::
          {:ok, label :: String.t(), confidence :: float(), details :: map()}
          | :error

  @doc """
  Train a centroid model from a list of `{feature_vector, label}` pairs.

  Accepts an optional keyword list with:
  - `:weights` — a list of per-dimension floats. When provided, training
    vectors are multiplied element-wise by the weights before computing
    centroids, and the weights are stored in the model for use at
    classification time.
  - `:balance` — when `true`, subsample majority classes to the median
    class size before computing centroids. This prevents large classes
    from dominating the centroid space while preserving minority classes.

  Raises `ArgumentError` if training examples disagree on vector
  dimensionality. Empty input produces an empty model (which always
  returns `:error` from `classify/2`).
  """
  @spec train([training_example()], keyword()) :: model()
  def train(training, opts \\ [])
  def train([], _opts), do: empty_model()

  def train(training, opts) when is_list(training) do
    weights = Keyword.get(opts, :weights)
    balance? = Keyword.get(opts, :balance, false)

    training
    |> validate_dimensions!()
    |> build_model(weights, balance?)
  end

  @doc """
  Classify a feature vector against a trained model.

  Returns `{:ok, label, confidence, details}` on success; `:error` if
  the model is empty or the input vector's length does not match
  `model.input_dim`.
  """
  @spec classify(list(float()), model()) :: classification()
  def classify(_vec, %{label_centroids: centroids}) when map_size(centroids) == 0,
    do: :error

  def classify(vec, %{label_centroids: centroids, input_dim: dim} = model) when is_list(vec) do
    if length(vec) != dim do
      :error
    else
      weighted_vec = apply_model_weights(vec, model)
      {label, top_score, second_score} = rank(weighted_vec, centroids)

      details = %{
        top_score: top_score,
        second_score: second_score,
        margin: top_score - second_score
      }

      {:ok, label, confidence(top_score, second_score), details}
    end
  end

  @doc "Declared input dimensionality of the trained model."
  @spec input_dim(model() | map()) :: non_neg_integer()
  def input_dim(%{input_dim: dim}) when is_integer(dim), do: dim
  def input_dim(_), do: 0

  @doc "Static `:feature_vector` tag so callers can branch on model kind."
  @spec kind(model() | map()) :: :feature_vector | :unknown
  def kind(%{kind: :feature_vector}), do: :feature_vector
  def kind(_), do: :unknown

  # ---- internals ----

  defp empty_model do
    %{kind: :feature_vector, input_dim: 0, label_centroids: %{}, weights: nil}
  end

  defp validate_dimensions!([{first_vec, _label} | _] = training)
       when is_list(first_vec) do
    expected = length(first_vec)

    Enum.each(training, fn
      {vec, _label} when is_list(vec) and length(vec) == expected ->
        :ok

      {vec, _label} when is_list(vec) ->
        raise ArgumentError,
              "feature-vector dimension mismatch: expected #{expected}, got #{length(vec)}"

      other ->
        raise ArgumentError,
              "invalid training example (expected {list(float()), String.t()}): #{inspect(other)}"
    end)

    training
  end

  defp build_model([{first_vec, _} | _] = training, weights, balance?) do
    input_dim = length(first_vec)

    if weights != nil and length(weights) != input_dim do
      raise ArgumentError,
            "weights length #{length(weights)} does not match input_dim #{input_dim}"
    end

    centroid_training = if balance?, do: balance_classes(training), else: training

    weighted_training =
      if weights do
        Enum.map(centroid_training, fn {vec, label} ->
          {elementwise_multiply(vec, weights), label}
        end)
      else
        centroid_training
      end

    label_centroids =
      weighted_training
      |> Enum.group_by(fn {_vec, label} -> label end, fn {vec, _label} -> vec end)
      |> Map.new(fn {label, vecs} -> {label, centroid(vecs, input_dim)} end)

    %{
      kind: :feature_vector,
      input_dim: input_dim,
      label_centroids: label_centroids,
      weights: weights
    }
  end

  defp balance_classes(training) do
    by_label = Enum.group_by(training, fn {_vec, label} -> label end)
    class_sizes = Enum.map(by_label, fn {_label, examples} -> length(examples) end) |> Enum.sort()

    n = length(class_sizes)
    cap = Enum.at(class_sizes, min(3 * div(n, 4), n - 1))

    Enum.flat_map(by_label, fn {_label, examples} ->
      if length(examples) > cap do
        Enum.take_random(examples, cap)
      else
        examples
      end
    end)
  end

  defp centroid([vec], _dim), do: vec

  defp centroid(vecs, dim) do
    n = length(vecs)
    zero = List.duplicate(0.0, dim)

    vecs
    |> Enum.reduce(zero, fn vec, acc -> elementwise_add(vec, acc) end)
    |> Enum.map(&(&1 / n))
  end

  defp elementwise_add(a, b) when is_list(a) and is_list(b) do
    Enum.zip_with(a, b, fn x, y -> x + y end)
  end

  defp elementwise_multiply(a, b) when is_list(a) and is_list(b) do
    Enum.zip_with(a, b, fn x, y -> x * y end)
  end

  defp apply_model_weights(vec, %{weights: weights}) when is_list(weights) do
    elementwise_multiply(vec, weights)
  end

  defp apply_model_weights(vec, _model), do: vec

  defp rank(vec, centroids) do
    ranked =
      centroids
      |> Enum.map(fn {label, centroid} -> {label, cosine_similarity(vec, centroid)} end)
      |> Enum.sort_by(fn {_l, score} -> -score end)

    [{top_label, top_score} | rest] = ranked

    second_score =
      case rest do
        [{_l, s} | _] -> s
        [] -> 0.0
      end

    {top_label, top_score, second_score}
  end

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) do
    {dot, mag_a_sq, mag_b_sq} =
      Enum.zip_reduce(a, b, {0.0, 0.0, 0.0}, fn x, y, {d, ma, mb} ->
        {d + x * y, ma + x * x, mb + y * y}
      end)

    mag_a = :math.sqrt(mag_a_sq)
    mag_b = :math.sqrt(mag_b_sq)

    cond do
      mag_a == 0.0 or mag_b == 0.0 -> 0.0
      true -> dot / (mag_a * mag_b)
    end
  end

  # Maps the margin between the winner and the runner-up into [0.0, 1.0].
  # margin == 0     -> 0.5 (no differentiation)
  # margin >= 0.25  -> 1.0
  # margin <= -0.5  -> 0.0 (should not happen after sort)
  defp confidence(top, second) do
    margin = top - second
    max(0.0, min(1.0, margin * 2.0 + 0.5))
  end
end
