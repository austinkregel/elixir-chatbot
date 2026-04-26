defmodule Mix.Tasks.TrainFraming do
  @moduledoc """
  Train the `:framing_class` document-level feature-vector classifier.

  Reads `data/classifiers/framing_class.json` (produced by `mix gen_framing_data`),
  runs the GA weight optimizer with the same composite fitness function used for
  chunk-level micro-classifiers, and writes two artifacts:

  - `priv/ml_models/micro/framing_class.term` — the trained centroid model
  - `priv/ml_models/micro/framing_neutral_centroid.term` — the mean vector of
    the most neutrally-framed class, used by `FramingDetector.deviation_from_neutral/1`

  ## Usage

      mix train_framing [options]

  ## Options

      --skip-weight-optimization  Train without GA (uniform weights)
      --no-balance                Disable balanced centroid computation
      --neutral-class NAME        Label to use as the neutral-framing reference
                                  (default: auto-detect smallest deviation class)
      --verbose                   Print per-class metrics
      --publish                   Publish model via ModelStore
      --seed-weights PATH         Initialize GA with weights from a prior model
  """

  use Mix.Task
  require Logger

  alias Brain.ML.FeatureVectorClassifier
  alias Brain.ML.ModelStore
  alias Brain.ML.WeightOptimizer

  @shortdoc "Train the :framing_class document-level classifier"

  @data_path "data/classifiers/framing_class.json"

  @impl Mix.Task
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          skip_weight_optimization: :boolean,
          no_balance: :boolean,
          neutral_class: :string,
          verbose: :boolean,
          publish: :boolean,
          seed_weights: :string
        ]
      )

    data_path = resolve_data_path()

    with {:ok, json} <- File.read(data_path),
         {:ok, entries} <- Jason.decode(json),
         {:ok, training_data} <- extract_training_data(entries) do
      Mix.shell().info("Loaded #{length(training_data)} training records from #{data_path}")

      labels = training_data |> Enum.map(&elem(&1, 1)) |> Enum.frequencies()

      Mix.shell().info("Classes: #{map_size(labels)}")

      if opts[:verbose] do
        labels
        |> Enum.sort_by(fn {_l, c} -> -c end)
        |> Enum.each(fn {label, count} ->
          Mix.shell().info("  #{String.pad_trailing(label, 20)} #{count}")
        end)
      end

      skip_optimization? = opts[:skip_weight_optimization] || false
      balance? = !(opts[:no_balance] || false)

      model = train_model(training_data, skip_optimization?, balance?, opts)

      output_dir = Path.join(get_models_path(), "micro")
      File.mkdir_p!(output_dir)

      model_path = Path.join(output_dir, "framing_class.term")
      File.write!(model_path, :erlang.term_to_binary(model))
      Mix.shell().info("Saved model to #{model_path}")

      neutral_centroid = compute_neutral_centroid(model, training_data, opts[:neutral_class])
      centroid_path = Path.join(output_dir, "framing_neutral_centroid.term")
      File.write!(centroid_path, :erlang.term_to_binary(neutral_centroid))
      Mix.shell().info("Saved neutral centroid to #{centroid_path}")

      if opts[:publish] do
        ModelStore.publish(model_path, ModelStore.version_prefix() <> "micro/framing_class.term")

        ModelStore.publish(
          centroid_path,
          ModelStore.version_prefix() <> "micro/framing_neutral_centroid.term"
        )
      end

      evaluate_model(model, training_data, opts[:verbose])
    else
      {:error, :enoent} ->
        Mix.raise("""
        Training data not found at #{data_path}.
        Run `mix gen_framing_data --corpus gvfc` first.
        """)

      {:error, stage, reason} ->
        Mix.raise("Training failed at #{stage}: #{inspect(reason)}")

      {:error, reason} ->
        Mix.raise("Training failed: #{inspect(reason)}")
    end
  end

  defp extract_training_data(entries) do
    pairs =
      Enum.flat_map(entries, fn
        %{"feature_vector" => vec, "label" => label}
        when is_list(vec) and is_binary(label) and length(vec) > 0 ->
          [{vec, label}]

        _ ->
          []
      end)

    case pairs do
      [] -> {:error, :shape_mismatch, "no valid {feature_vector, label} records"}
      _ -> {:ok, pairs}
    end
  end

  defp train_model(training_data, skip_optimization?, balance?, opts) do
    n_classes = training_data |> Enum.map(&elem(&1, 1)) |> Enum.uniq() |> length()

    cond do
      n_classes < 2 ->
        Mix.shell().info("Only #{n_classes} class — skipping GA")
        FeatureVectorClassifier.train(training_data, balance: balance?)

      skip_optimization? ->
        Mix.shell().info("Skipping weight optimization (--skip-weight-optimization)")
        FeatureVectorClassifier.train(training_data, balance: balance?)

      true ->
        Mix.shell().info(
          "Running GA weight optimization (#{n_classes} classes, " <>
            "#{length(training_data)} examples)..."
        )

        ga_opts =
          case opts[:seed_weights] do
            nil ->
              [verbose: true]

            path ->
              case File.read(path) do
                {:ok, bin} ->
                  prior_model = :erlang.binary_to_term(bin)

                  case Map.get(prior_model, :weights) do
                    w when is_list(w) ->
                      Mix.shell().info("Seeding GA with #{length(w)} weights from #{path}")
                      [verbose: true, seed_weights: w]

                    _ ->
                      Mix.shell().info("No weights in seed model, using default init")
                      [verbose: true]
                  end

                _ ->
                  Mix.shell().info("Could not read seed weights from #{path}")
                  [verbose: true]
              end
          end

        result = WeightOptimizer.optimize(training_data, Keyword.put(ga_opts, :classifier, "framing_class"))

        Mix.shell().info(
          "GA complete: #{Float.round(result.fitness * 100, 1)}% composite fitness " <>
            "at generation #{result.generation}"
        )

        alive = Enum.count(result.weights, &(&1 > 0.01))
        Mix.shell().info("#{alive}/#{length(result.weights)} dimensions active")

        FeatureVectorClassifier.train(training_data, weights: result.weights, balance: balance?)
    end
  end

  defp compute_neutral_centroid(model, training_data, explicit_class) do
    centroids = model.label_centroids

    neutral_label =
      case explicit_class do
        nil -> find_most_neutral_class(centroids)
        label -> label
      end

    Mix.shell().info("Neutral-framing reference class: #{neutral_label}")

    case Map.get(centroids, neutral_label) do
      nil ->
        neutral_vecs =
          training_data
          |> Enum.filter(fn {_vec, label} -> label == neutral_label end)
          |> Enum.map(fn {vec, _label} -> vec end)

        if neutral_vecs == [] do
          Mix.shell().info("Warning: no examples for neutral class, using grand mean")
          mean_all_centroids(Map.values(centroids))
        else
          compute_mean(neutral_vecs)
        end

      protos when is_list(protos) ->
        # `label_centroids` values are k-means prototype lists, so collapse
        # the chosen class's protos to a single flat reference vector before
        # we persist it. Anything else would fail ModelPreflight.
        compute_mean(protos)
    end
  end

  defp find_most_neutral_class(centroids) when map_size(centroids) == 0, do: "other"

  defp find_most_neutral_class(centroids) do
    grand_mean = mean_all_centroids(Map.values(centroids))

    {label, _} =
      centroids
      |> Enum.map(fn {label, protos} ->
        {label, cosine_similarity(compute_mean(protos), grand_mean)}
      end)
      |> Enum.max_by(fn {_label, sim} -> sim end)

    label
  end

  # Collapse `%{label => list(list(float))}.values()` (i.e. a list of
  # per-label proto-lists) down to a single flat float vector by first
  # taking the mean within each class, then the mean across classes. This
  # matches the operator's intuition of "the grand mean of class centroids"
  # and ensures every class contributes equally regardless of its k value.
  defp mean_all_centroids([]), do: []

  defp mean_all_centroids(per_class_protos) do
    per_class_protos
    |> Enum.map(&compute_mean/1)
    |> compute_mean()
  end

  defp compute_mean([]), do: []
  defp compute_mean([single]) when is_list(single), do: single

  defp compute_mean([first | _] = vecs) when is_list(first) do
    dim = length(first)
    n = length(vecs)
    zero = List.duplicate(0.0, dim)

    vecs
    |> Enum.reduce(zero, fn vec, acc ->
      Enum.zip_with(acc, vec, &(&1 + &2))
    end)
    |> Enum.map(&(&1 / n))
  end

  defp evaluate_model(model, training_data, verbose?) do
    predictions =
      Enum.map(training_data, fn {vec, true_label} ->
        case FeatureVectorClassifier.classify(vec, model) do
          {:ok, pred_label, _conf, _details} -> {true_label, pred_label}
          _ -> {true_label, "??"}
        end
      end)

    total = length(predictions)
    correct = Enum.count(predictions, fn {t, p} -> t == p end)
    accuracy = if total > 0, do: correct / total, else: 0.0

    Mix.shell().info("\n--- Training Set Evaluation ---")
    Mix.shell().info("  Accuracy: #{Float.round(accuracy * 100, 1)}% (#{correct}/#{total})")

    if verbose? do
      by_class =
        predictions
        |> Enum.group_by(fn {true_label, _} -> true_label end)
        |> Enum.sort_by(fn {_label, items} -> -length(items) end)

      Mix.shell().info("\n  Per-class breakdown:")

      Enum.each(by_class, fn {label, items} ->
        tp = Enum.count(items, fn {t, p} -> t == p end)
        n = length(items)
        recall = if n > 0, do: Float.round(tp / n * 100, 1), else: 0.0
        Mix.shell().info("    #{String.pad_trailing(label, 20)} #{recall}% recall (#{tp}/#{n})")
      end)
    end
  end

  defp cosine_similarity(a, b) when is_list(a) and is_list(b) do
    {dot, mag_a_sq, mag_b_sq} =
      Enum.zip_reduce(a, b, {0.0, 0.0, 0.0}, fn x, y, {d, ma, mb} ->
        {d + x * y, ma + x * x, mb + y * y}
      end)

    mag_a = :math.sqrt(mag_a_sq)
    mag_b = :math.sqrt(mag_b_sq)

    if mag_a == 0.0 or mag_b == 0.0, do: 0.0, else: dot / (mag_a * mag_b)
  end

  defp get_models_path do
    Application.get_env(:brain, :ml, [])[:models_path] || Brain.priv_path("ml_models")
  end

  defp resolve_data_path do
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

    Path.join(umbrella_root, @data_path)
  end
end
