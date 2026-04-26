defmodule Brain.ML.FeatureVectorClassifier do
  @moduledoc """
  Dense centroid-based classifier with multi-prototype support that consumes
  numeric feature vectors (`list(float())`) and returns
  `{:ok, label, confidence, details}`.

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
        label_centroids: %{String.t() => list(list(float()))},
        weights: list(float()) | nil,
        platt_params: {float(), float()} | nil
      }

  Models are plain Elixir terms and round-trip through
  `:erlang.term_to_binary/1` cleanly, so they slot into the same
  on-disk format (`priv/ml_models/micro/<name>.term`) that
  `Brain.ML.MicroClassifiers` already loads.

  ## Multi-Prototype Support

  Each class can have multiple prototype centroids (sub-centroids),
  computed via K-means++ clustering on the class's training vectors. The
  number of sub-centroids per class scales with the number of training
  examples: `k = min(k_max, max(1, div(n, min_per_proto)))`.
  Classification scores each label by its best-matching sub-centroid.

  Old models with a single centroid per label (stored as a flat list of
  floats rather than a list-of-lists) are automatically detected and
  normalized at classification time.

  ## Per-Dimension Weighting

  When `weights` is present in the model, both training centroids and
  classification inputs are multiplied element-wise by the weight
  vector before cosine similarity. This allows the classifier to
  emphasize discriminative dimensions and suppress noisy ones.

  Models without weights (or with `weights: nil`) behave exactly as
  before — uniform weighting across all dimensions.

  ## Confidence Calibration (Platt Scaling)

  When sufficient validation data is available (>= 50 held-out examples),
  confidence scores are calibrated via Platt scaling — a sigmoid mapping
  from raw margin to probability of correctness. Falls back to the
  original linear margin remap when `platt_params` is nil.

  ## Classification

  `classify/2` ranks every label by its best sub-centroid cosine
  similarity against the (optionally weighted) input vector and returns
  the winner. Confidence is derived from the margin between the winner
  and the runner-up, mapped through either Platt scaling or a linear
  remap into `[0.0, 1.0]`.

  An empty model, or an input vector whose dimensionality does not
  match the model's declared `input_dim`, returns `:error`.
  """

  @k_max 4
  @min_per_proto 10
  @kmeans_seed 42
  @kmeans_max_iters 100
  @convergence_threshold 1.0e-6

  @type training_example :: {list(float()), String.t()}

  @type model :: %{
          kind: :feature_vector,
          input_dim: non_neg_integer(),
          label_centroids: %{String.t() => list(list(float()))},
          weights: list(float()) | nil,
          platt_params: {float(), float()} | nil
        }

  @type classification ::
          {:ok, label :: String.t(), confidence :: float(), details :: map()}
          | :error

  @doc """
  Train a multi-prototype centroid model from a list of
  `{feature_vector, label}` pairs.

  Accepts an optional keyword list with:
  - `:weights` — a list of per-dimension floats. When provided, training
    vectors are multiplied element-wise by the weights before computing
    centroids, and the weights are stored in the model for use at
    classification time.
  - `:balance` — when `true`, subsample majority classes to the median
    class size before computing centroids. This prevents large classes
    from dominating the centroid space while preserving minority classes.

  Each class receives `k = min(#{@k_max}, max(1, n ÷ #{@min_per_proto}))`
  sub-centroids, computed via K-means++ with a fixed seed for
  determinism. When sufficient validation data is available, Platt
  scaling parameters are fitted for confidence calibration.

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
  @spec classify(list(float()), model(), keyword()) :: classification()
  def classify(_vec, %{label_centroids: centroids}) when map_size(centroids) == 0,
    do: :error

  def classify(vec, %{label_centroids: centroids, input_dim: dim} = model, opts \\ [])
      when is_list(vec) do
    if length(vec) != dim do
      :error
    else
      weighted_vec = apply_model_weights(vec, model)
      normalized = normalize_centroids(centroids)
      top_k_limit = Keyword.get(opts, :top_k_limit, 5)
      {label, top_score, second_score, top_k} = rank_with_top_k(weighted_vec, normalized, top_k_limit)

      details = %{
        top_score: top_score,
        second_score: second_score,
        margin: top_score - second_score,
        top_k: top_k
      }

      {:ok, label, confidence(top_score, second_score, model), details}
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
    %{kind: :feature_vector, input_dim: 0, label_centroids: %{}, weights: nil, platt_params: nil}
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

    sorted_training =
      Enum.sort_by(centroid_training, fn {vec, label} ->
        {label, :erlang.phash2(vec)}
      end)

    weighted_training =
      if weights do
        Enum.map(sorted_training, fn {vec, label} ->
          {elementwise_multiply(vec, weights), label}
        end)
      else
        sorted_training
      end

    by_label =
      Enum.group_by(weighted_training, fn {_vec, label} -> label end, fn {vec, _label} -> vec end)

    label_centroids =
      by_label
      |> Enum.map(fn {label, vecs} ->
        k = k_for_class(length(vecs))

        protos =
          if k == 1 do
            [centroid(vecs, input_dim)]
          else
            kmeans_plus_plus(vecs, k, @kmeans_seed)
          end

        {label, protos}
      end)
      |> Enum.sort_by(fn {label, _} -> label end)
      |> Map.new()

    base_model = %{
      kind: :feature_vector,
      input_dim: input_dim,
      label_centroids: label_centroids,
      weights: weights,
      platt_params: nil
    }

    maybe_fit_platt(base_model, sorted_training)
  end

  defp k_for_class(n), do: min(@k_max, max(1, div(n, @min_per_proto)))

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

  # ---- K-means++ ----

  defp kmeans_plus_plus(vectors, k, seed) do
    :rand.seed(:exsss, {seed, seed, seed})
    n = length(vectors)
    vec_array = :array.from_list(vectors)
    dim = length(hd(vectors))

    first_idx = :rand.uniform(n) - 1
    centers = [:array.get(first_idx, vec_array)]

    centers = init_remaining_centers(centers, vec_array, n, k)
    lloyds(vectors, centers, dim, @kmeans_max_iters)
  end

  defp init_remaining_centers(centers, _vec_array, _n, k) when length(centers) >= k, do: centers

  defp init_remaining_centers(centers, vec_array, n, k) do
    distances =
      for i <- 0..(n - 1) do
        vec = :array.get(i, vec_array)
        Enum.map(centers, &squared_distance(vec, &1)) |> Enum.min()
      end

    total = Enum.sum(distances)

    new_idx =
      if total == 0.0 do
        :rand.uniform(n) - 1
      else
        threshold = :rand.uniform() * total
        pick_index(distances, threshold, 0)
      end

    new_center = :array.get(new_idx, vec_array)
    init_remaining_centers(centers ++ [new_center], vec_array, n, k)
  end

  defp pick_index([d | _rest], threshold, idx) when threshold <= d, do: idx
  defp pick_index([d | rest], threshold, idx), do: pick_index(rest, threshold - d, idx + 1)
  defp pick_index([], _threshold, idx), do: max(0, idx - 1)

  # ---- Lloyd's algorithm ----

  defp lloyds(_vectors, centers, _dim, 0), do: centers

  defp lloyds(vectors, centers, dim, remaining) do
    assignments =
      Enum.map(vectors, fn vec ->
        centers
        |> Enum.with_index()
        |> Enum.min_by(fn {c, _i} -> squared_distance(vec, c) end)
        |> elem(1)
      end)

    groups =
      Enum.zip(vectors, assignments)
      |> Enum.group_by(fn {_vec, idx} -> idx end, fn {vec, _idx} -> vec end)

    new_centers =
      Enum.map(0..(length(centers) - 1), fn i ->
        case Map.get(groups, i) do
          nil -> Enum.at(centers, i)
          vecs -> centroid(vecs, dim)
        end
      end)

    max_shift =
      Enum.zip(centers, new_centers)
      |> Enum.map(fn {old, new_c} -> squared_distance(old, new_c) end)
      |> Enum.max()
      |> :math.sqrt()

    if max_shift < @convergence_threshold do
      new_centers
    else
      lloyds(vectors, new_centers, dim, remaining - 1)
    end
  end

  # ---- Platt scaling ----

  defp maybe_fit_platt(model, sorted_training) do
    n = length(sorted_training)
    val_size = div(n, 5)

    if val_size < 50 do
      model
    else
      {_train_split, val_split} = Enum.split(sorted_training, n - val_size)

      margins_and_correct =
        val_split
        |> Enum.map(fn {vec, true_label} ->
          case classify(vec, model) do
            {:ok, pred_label, _conf, %{margin: margin}} ->
              {margin, pred_label == true_label}

            :error ->
              nil
          end
        end)
        |> Enum.reject(&is_nil/1)

      if length(margins_and_correct) >= 50 do
        platt_params = fit_platt_params(margins_and_correct)
        %{model | platt_params: platt_params}
      else
        model
      end
    end
  end

  defp fit_platt_params(margins_and_correct) do
    a_values = [-20.0, -15.0, -10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0]
    b_values = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    {_best_nll, best_params} =
      for a <- a_values, b <- b_values, reduce: {:infinity, {-4.0, 0.0}} do
        {best_nll, best_params} ->
          nll = compute_nll(margins_and_correct, a, b)
          if nll < best_nll, do: {nll, {a, b}}, else: {best_nll, best_params}
      end

    best_params
  end

  defp compute_nll(margins_and_correct, a, b) do
    eps = 1.0e-15

    Enum.reduce(margins_and_correct, 0.0, fn {margin, correct?}, acc ->
      raw = 1.0 / (1.0 + :math.exp(a * margin + b))
      p = max(eps, min(1.0 - eps, raw))

      if correct? do
        acc - :math.log(p)
      else
        acc - :math.log(1.0 - p)
      end
    end)
  end

  # ---- backward compatibility ----

  defp normalize_centroids(label_centroids) do
    Map.new(label_centroids, fn
      {label, [first | _] = protos} when is_list(first) -> {label, protos}
      {label, centroid_vec} when is_list(centroid_vec) -> {label, [centroid_vec]}
    end)
  end

  # ---- vector math ----

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

  defp squared_distance(a, b) when is_list(a) and is_list(b) do
    Enum.zip_with(a, b, fn x, y -> (x - y) * (x - y) end) |> Enum.sum()
  end

  defp apply_model_weights(vec, %{weights: weights}) when is_list(weights) do
    elementwise_multiply(vec, weights)
  end

  defp apply_model_weights(vec, _model), do: vec

  defp rank_with_top_k(vec, label_centroids, limit) when is_integer(limit) and limit >= 1 do
    ranked =
      label_centroids
      |> Enum.map(fn {label, protos} ->
        best_score = protos |> Enum.map(&cosine_similarity(vec, &1)) |> Enum.max()
        {label, best_score}
      end)
      |> Enum.sort_by(fn {_l, score} -> -score end)

    top_k = Enum.take(ranked, limit)
    [{top_label, top_score} | rest] = ranked

    second_score =
      case rest do
        [{_l, s} | _] -> s
        [] -> 0.0
      end

    {top_label, top_score, second_score, top_k}
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

  defp confidence(top, second, model) do
    margin = top - second

    case Map.get(model, :platt_params) do
      {a, b} ->
        1.0 / (1.0 + :math.exp(a * margin + b))

      nil ->
        max(0.0, min(1.0, margin * 2.0 + 0.5))
    end
  end
end
