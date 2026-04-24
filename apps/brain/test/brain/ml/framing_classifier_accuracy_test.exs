defmodule Brain.ML.FramingClassifierAccuracyTest do
  @moduledoc """
  Accuracy regression test for the `:framing_class` document-level classifier.

  Loads the trained model that `mix train_framing` writes to disk and
  evaluates it on a deterministic held-out 20% slice of the training data.

  The test guards against two regressions:

    1. The trained model still loads with the expected shape (`:feature_vector`,
       non-empty centroids, GA-optimized weights).
    2. Macro F1 on the held-out split stays comfortably above the random
       baseline for a 9-class problem (~11%).

  ## Threshold rationale

  This corpus (GVFC-derived, ~1300 examples across 9 framing classes,
  heavily imbalanced — `gun_rights` 373 vs `economic` 38) does not admit
  high-accuracy classification with a plain centroid + cosine model. As of
  2026-04-22 the production model trained via `mix train_framing` produces:

      held-out macro F1: ~22.5%
      held-out accuracy:  ~25.0%
      random baseline:    ~11.1%

  Asserting `@min_macro_f1 = 0.18` keeps the test honest: it fails if the
  model collapses back toward random (as a vanilla unweighted centroid does
  at ~13%) but does not require us to invent gains the model can't deliver.
  Raise the threshold when the model improves; do not lower it without
  documenting why.

  Prerequisites:

    - `mix gen_framing_data --corpus gvfc` has been run (writes
      `data/classifiers/framing_class.json`).
    - `mix train_framing` has been run (writes
      `priv/ml_models/micro/framing_class.term`).

  Both prerequisites are produced by the standard model-training pipeline;
  the test fails loudly if either is missing rather than silently skipping.
  """

  use ExUnit.Case, async: false

  alias Brain.ML.FeatureVectorClassifier

  @data_path "data/classifiers/framing_class.json"
  @min_macro_f1 0.18
  @random_baseline_floor 0.13
  @split_seed {42, 137, 256}

  describe "framing classifier accuracy" do
    test "trained model beats random baseline on held-out split" do
      data_path = resolve_data_path()
      model_path = resolve_model_path()

      assert File.exists?(data_path),
             """
             Framing training data not found at #{data_path}.
             Run `mix gen_framing_data --corpus gvfc` to (re)generate it.
             """

      assert File.exists?(model_path),
             """
             Trained framing model not found at #{model_path}.
             Run `mix train_framing` to produce it.
             """

      model =
        model_path
        |> File.read!()
        |> :erlang.binary_to_term()

      assert FeatureVectorClassifier.kind(model) == :feature_vector,
             "Loaded model has unexpected kind: #{inspect(Map.get(model, :kind))}"

      assert is_list(Map.get(model, :weights)),
             "Trained framing model is missing GA-optimized weights; " <>
               "did `mix train_framing --skip-weight-optimization` get run?"

      assert map_size(Map.get(model, :label_centroids, %{})) >= 2,
             "Trained framing model has fewer than 2 label centroids"

      data = load_training_data!(data_path)

      assert length(data) >= 100,
             "Refusing to evaluate on tiny corpus (#{length(data)} examples). " <>
               "Re-run `mix gen_framing_data --corpus gvfc`."

      :rand.seed(:exsplus, @split_seed)
      {_train, test_set} = data |> Enum.shuffle() |> split_at(0.8)

      {macro_f1, accuracy, per_class_f1} = evaluate(model, test_set)

      report(macro_f1, accuracy, per_class_f1, length(test_set))

      assert macro_f1 >= @min_macro_f1,
             """
             Framing macro F1 #{pct(macro_f1)} fell below the regression floor
             of #{pct(@min_macro_f1)} on a #{length(test_set)}-example held-out
             split. The trained model is regressing toward random baseline
             (~#{pct(@random_baseline_floor)}). Investigate `mix train_framing`
             output, the GA fitness curve, and the upstream features in
             Brain.Analysis.FeatureExtractor before lowering this threshold.
             """

      assert accuracy >= @random_baseline_floor,
             "Framing accuracy #{pct(accuracy)} is at random-guess baseline; the " <>
               "trained model is not learning anything from the features."
    end
  end

  defp load_training_data!(path) do
    {:ok, json} = File.read(path)
    {:ok, entries} = Jason.decode(json)

    Enum.flat_map(entries, fn
      %{"feature_vector" => vec, "label" => label}
      when is_list(vec) and is_binary(label) and length(vec) > 0 ->
        [{vec, label}]

      _ ->
        []
    end)
  end

  defp split_at(list, fraction) do
    n = round(length(list) * fraction)
    Enum.split(list, n)
  end

  defp evaluate(model, test_set) do
    predictions =
      Enum.map(test_set, fn {vec, true_label} ->
        case FeatureVectorClassifier.classify(vec, model) do
          {:ok, pred_label, _conf, _details} -> {true_label, pred_label}
          :error -> {true_label, "__error__"}
        end
      end)

    labels =
      predictions
      |> Enum.flat_map(fn {t, p} -> [t, p] end)
      |> Enum.uniq()

    per_class_f1 =
      Enum.map(labels, fn label ->
        tp = Enum.count(predictions, fn {t, p} -> t == label and p == label end)
        fp = Enum.count(predictions, fn {t, p} -> t != label and p == label end)
        fn_ = Enum.count(predictions, fn {t, p} -> t == label and p != label end)

        precision = if tp + fp > 0, do: tp / (tp + fp), else: 0.0
        recall = if tp + fn_ > 0, do: tp / (tp + fn_), else: 0.0

        f1 =
          if precision + recall > 0,
            do: 2 * precision * recall / (precision + recall),
            else: 0.0

        {label, f1}
      end)

    macro_f1 =
      case per_class_f1 do
        [] -> 0.0
        scores -> Enum.sum(Enum.map(scores, fn {_l, f} -> f end)) / length(scores)
      end

    accuracy = Enum.count(predictions, fn {t, p} -> t == p end) / max(length(predictions), 1)

    {macro_f1, accuracy, per_class_f1}
  end

  defp report(macro_f1, accuracy, per_class_f1, n) do
    IO.puts("\n--- Framing Classifier Accuracy ---")
    IO.puts("  Held-out set:        #{n} examples")
    IO.puts("  Accuracy:            #{pct(accuracy)}")
    IO.puts("  Macro F1:            #{pct(macro_f1)}")
    IO.puts("  Min Macro F1:        #{pct(@min_macro_f1)}")
    IO.puts("  Random baseline:     ~11% (9 classes)")

    per_class_f1
    |> Enum.sort_by(fn {_l, f} -> -f end)
    |> Enum.each(fn {label, f1} ->
      IO.puts("    #{String.pad_trailing(label, 22)} F1=#{pct(f1)}")
    end)
  end

  defp pct(score) when is_number(score), do: "#{Float.round(score * 100, 1)}%"

  defp resolve_data_path do
    Path.join(umbrella_root(), @data_path)
  end

  defp resolve_model_path do
    # We always evaluate the model that `mix train_framing` produces, regardless
    # of the test env's `models_path` override. The test-only models_path
    # (apps/brain/test/ml_models) holds throw-away artifacts re-trained on
    # fixture data each run; the real GA-optimized framing_class lives in
    # apps/brain/priv/ml_models/micro/.
    candidates =
      [
        Application.get_env(:brain, :ml, [])[:models_path],
        Brain.priv_path("ml_models"),
        Path.join(umbrella_root(), "apps/brain/priv/ml_models")
      ]
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&Path.join([&1, "micro", "framing_class.term"]))

    Enum.find(candidates, hd(candidates), &File.exists?/1)
  end

  defp umbrella_root do
    priv_dir = :code.priv_dir(:brain) |> to_string()

    case File.read_link(priv_dir) do
      {:ok, link_target} ->
        parent = Path.dirname(priv_dir)
        real_priv = Path.join(parent, link_target) |> Path.expand()
        Path.join(real_priv, "../../..") |> Path.expand()

      {:error, _} ->
        Path.join(priv_dir, "../../../../..") |> Path.expand()
    end
  end
end
